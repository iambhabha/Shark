//! Self-play match harness — pit two UCI engine binaries against each other over
//! many games and report the score and an estimated Elo difference.
//!
//! This is the tool we use to *prove* a change made the engine stronger: build
//! the old and the new binary, play a few hundred games, and read off the Elo
//! delta with an error bar. It is deliberately self-contained — the only referee
//! is the `mythos` crate itself, so the harness never trusts an engine's opinion
//! about legality or game termination.
//!
//! The design has three parts:
//!
//! * an [`Engine`] driver that wraps a spawned UCI process (piped stdin/stdout)
//!   and speaks the minimal slice of UCI we need: `uci`/`isready`, `ucinewgame`,
//!   `position startpos moves ...`, `go movetime|depth`, and `bestmove`;
//! * a [`play_game`] referee that maintains a [`Position`], asks each side's
//!   engine for a move, *validates it against the crate's own legal move list*,
//!   and adjudicates every game-end rule (mate, stalemate, 50-move, threefold
//!   repetition, insufficient material, ply cap);
//! * a match loop that alternates colors, seeds each opening twice with reversed
//!   colors so no opening biases one side, and tallies the score.
//!
//! Robustness: an engine that hangs (never prints `bestmove`) would otherwise
//! freeze the whole match, so every read runs on a reader thread feeding a
//! channel, and each move has a generous deadline. A move that misses the
//! deadline — like an illegal or missing move — simply loses that engine the
//! game rather than wedging the harness.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use mythos::movegen::generate_legal;
use mythos::position::Position;
use mythos::types::{Color, PieceType};

// ---------------------------------------------------------------------------
// Configuration.
// ---------------------------------------------------------------------------

/// The parsed command-line configuration for a match.
struct Config {
    /// Path to engine A's binary.
    engine_a: String,
    /// Path to engine B's binary.
    engine_b: String,
    /// Number of games to play.
    games: usize,
    /// Milliseconds per move for `go movetime` (ignored when `depth` is set).
    movetime: u64,
    /// Fixed search depth for `go depth D`, or `None` to use `movetime`.
    depth: Option<u32>,
    /// Hard cap on plies before a game is adjudicated a draw.
    maxplies: usize,
}

/// The built-in opening lines (UCI moves, space-separated). Each is played twice
/// with colors reversed so a lopsided opening can't bias one engine.
const OPENINGS: &[&str] = &[
    "e2e4 e7e5",
    "d2d4 d7d5",
    "c2c4",
    "g1f3 d7d5",
    "e2e4 c7c5",
    "d2d4 g8f6",
    "e2e4 e7e6",
    "c2c4 e7e5",
    "g1f3 g8f6",
    "b1c3",
];

/// How long a single engine move may take before we treat the engine as hung and
/// forfeit the game. Generous relative to `movetime`/typical depths so a slow but
/// working engine is never falsely killed.
const MOVE_DEADLINE: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// CLI parsing.
// ---------------------------------------------------------------------------

/// Parse `argv` into a [`Config`], or return an error string describing misuse.
fn parse_args(args: &[String]) -> Result<Config, String> {
    // First two positional args are the engine paths.
    let mut positional = Vec::new();
    let mut games = 40usize;
    let mut movetime = 100u64;
    let mut depth: Option<u32> = None;
    let mut maxplies = 300usize;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--games" => {
                i += 1;
                let v = args.get(i).ok_or("--games needs a value")?;
                games = v.parse().map_err(|_| format!("bad --games value: {v}"))?;
            }
            "--movetime" => {
                i += 1;
                let v = args.get(i).ok_or("--movetime needs a value")?;
                movetime = v.parse().map_err(|_| format!("bad --movetime value: {v}"))?;
            }
            "--depth" => {
                i += 1;
                let v = args.get(i).ok_or("--depth needs a value")?;
                depth = Some(v.parse().map_err(|_| format!("bad --depth value: {v}"))?);
            }
            "--maxplies" => {
                i += 1;
                let v = args.get(i).ok_or("--maxplies needs a value")?;
                maxplies = v.parse().map_err(|_| format!("bad --maxplies value: {v}"))?;
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }

    if positional.len() < 2 {
        return Err("usage: selfplay <engine_a.exe> <engine_b.exe> \
                    [--games N] [--movetime MS] [--depth D] [--maxplies P]"
            .to_string());
    }

    Ok(Config {
        engine_a: positional[0].clone(),
        engine_b: positional[1].clone(),
        games,
        movetime,
        depth,
        maxplies,
    })
}

// ---------------------------------------------------------------------------
// Engine driver — one spawned UCI process.
// ---------------------------------------------------------------------------

/// A live UCI engine subprocess we can ask for moves.
///
/// Reads happen on a dedicated thread that funnels every stdout line into a
/// channel; the main thread pulls lines with a timeout so a wedged engine turns
/// into a lost game instead of a hung harness.
struct Engine {
    /// The child process handle (kept so we can kill it on `quit`).
    child: Child,
    /// The child's stdin — where we write UCI commands.
    stdin: ChildStdin,
    /// Lines the reader thread has pulled off the child's stdout.
    lines: Receiver<String>,
    /// A short display name (the binary's file name) for the summary line.
    name: String,
}

impl Engine {
    /// Spawn `path`, complete the `uci`/`uciok` and `isready`/`readyok` handshake,
    /// and return a ready-to-use driver.
    fn spawn(path: &str) -> Result<Engine, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("failed to spawn {path}: {e}"))?;

        let stdin = child.stdin.take().ok_or("child has no stdin")?;
        let stdout = child.stdout.take().ok_or("child has no stdout")?;

        // Pump stdout on its own thread so a slow/stuck engine can never block the
        // referee: the channel recv on the main side always has a timeout.
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        // A send error means the receiver was dropped (engine
                        // quit); stop reading.
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break, // EOF or read error: reader thread ends.
                }
            }
        });

        // The display name is just the file stem, so the summary reads nicely.
        let name = std::path::Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string());

        let mut engine = Engine {
            child,
            stdin,
            lines: rx,
            name,
        };

        // Handshake: `uci` -> ... -> `uciok`, then `isready` -> `readyok`.
        engine.send("uci")?;
        engine.read_until(|l| l.trim() == "uciok")?;
        engine.send("isready")?;
        engine.read_until(|l| l.trim() == "readyok")?;

        Ok(engine)
    }

    /// Write one command line (a newline is appended) to the engine's stdin.
    fn send(&mut self, cmd: &str) -> Result<(), String> {
        writeln!(self.stdin, "{cmd}").map_err(|e| format!("write to engine failed: {e}"))?;
        self.stdin
            .flush()
            .map_err(|e| format!("flush to engine failed: {e}"))
    }

    /// Pull lines (each with the [`MOVE_DEADLINE`] timeout) until `pred` matches
    /// one, discarding the rest. Errors on timeout or EOF.
    fn read_until<F: Fn(&str) -> bool>(&self, pred: F) -> Result<String, String> {
        loop {
            match self.lines.recv_timeout(MOVE_DEADLINE) {
                Ok(line) => {
                    if pred(&line) {
                        return Ok(line);
                    }
                    // Not the line we want (e.g. an `info` line): keep reading.
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err("engine timed out".to_string());
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("engine closed its output (crash/EOF)".to_string());
                }
            }
        }
    }

    /// Start a fresh game: `ucinewgame` then a fresh `isready`/`readyok` sync.
    fn new_game(&mut self) -> Result<(), String> {
        self.send("ucinewgame")?;
        self.send("isready")?;
        self.read_until(|l| l.trim() == "readyok")?;
        Ok(())
    }

    /// Ask the engine for its best move from the given move sequence.
    ///
    /// Sends `position startpos [moves ...]` then either `go depth D` or
    /// `go movetime MS`, and reads until a `bestmove` line. Returns the UCI move
    /// token, or `None` for `bestmove 0000` / `bestmove (none)` / timeout / EOF —
    /// each of which the referee treats as a loss for this engine.
    fn bestmove(&mut self, moves: &[String], movetime: u64, depth: Option<u32>) -> Option<String> {
        // Build and send the position command.
        let pos_cmd = if moves.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", moves.join(" "))
        };
        if self.send(&pos_cmd).is_err() {
            return None;
        }

        // Build and send the go command.
        let go_cmd = match depth {
            Some(d) => format!("go depth {d}"),
            None => format!("go movetime {movetime}"),
        };
        if self.send(&go_cmd).is_err() {
            return None;
        }

        // Read until a bestmove line (info lines are skipped by the predicate).
        let line = self.read_until(|l| l.trim_start().starts_with("bestmove")).ok()?;

        // `bestmove <move> [ponder <move>]` — take the token after "bestmove".
        let mv = line.split_whitespace().nth(1)?;
        if mv == "0000" || mv == "(none)" {
            return None;
        }
        Some(mv.to_string())
    }

    /// Ask the engine to quit; give it a moment, then kill if it lingers.
    fn quit(&mut self) {
        // Best-effort: if the pipe is already broken this just fails silently.
        let _ = self.send("quit");
        thread::sleep(Duration::from_millis(100));
        // If it has not exited on its own, make sure it does.
        match self.child.try_wait() {
            Ok(Some(_)) => {} // already gone.
            _ => {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Game result.
// ---------------------------------------------------------------------------

/// Who won a single game, from White's point of view, plus the reason.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Winner {
    White,
    Black,
    Draw,
}

/// The outcome of one game: the winner and a short human reason string.
struct GameResult {
    winner: Winner,
    reason: &'static str,
}

// ---------------------------------------------------------------------------
// Referee — game-end detection with the crate as the source of truth.
// ---------------------------------------------------------------------------

/// Draw by insufficient material: K vs K, K+minor vs K, K vs K+minor (a lone
/// knight or bishop against a bare king; anything more can mate, so is not a
/// draw here).
fn insufficient_material(pos: &Position) -> bool {
    // Any pawn, rook, or queen on the board means mate is still possible.
    for pt in [PieceType::Pawn, PieceType::Rook, PieceType::Queen] {
        if pos.pieces_cp(Color::White, pt).any() || pos.pieces_cp(Color::Black, pt).any() {
            return false;
        }
    }

    // Count the remaining minor pieces per side.
    let minors = |c: Color| {
        pos.pieces_cp(c, PieceType::Knight).count() + pos.pieces_cp(c, PieceType::Bishop).count()
    };
    let (w, b) = (minors(Color::White), minors(Color::Black));

    // Total minors 0 (K vs K) or exactly one minor on one side (K+minor vs K).
    w + b <= 1
}

/// Check every terminal condition *before* the side to move plays, in the order
/// the spec prescribes. Returns `Some(result)` if the game is over.
///
/// `reps` maps a Zobrist key to how many times that exact position has occurred
/// so far in this game (used for the threefold-repetition rule).
fn check_game_over(
    pos: &mut Position,
    reps: &HashMap<u64, u8>,
    ply: usize,
    maxplies: usize,
) -> Option<GameResult> {
    // 1. No legal moves: checkmate or stalemate.
    let legal = generate_legal(pos);
    if legal.is_empty() {
        return Some(if pos.in_check() {
            // The side to move is checkmated, so the *other* side won.
            GameResult {
                winner: match pos.side_to_move() {
                    Color::White => Winner::Black,
                    Color::Black => Winner::White,
                },
                reason: "checkmate",
            }
        } else {
            GameResult {
                winner: Winner::Draw,
                reason: "stalemate",
            }
        });
    }

    // 2. Fifty-move rule (100 half-moves without a pawn move or capture).
    if pos.halfmove_clock() >= 100 {
        return Some(GameResult {
            winner: Winner::Draw,
            reason: "50-move rule",
        });
    }

    // 3. Threefold repetition: the current position has now been seen 3 times.
    if reps.get(&pos.key()).copied().unwrap_or(0) >= 3 {
        return Some(GameResult {
            winner: Winner::Draw,
            reason: "threefold repetition",
        });
    }

    // 4. Insufficient material.
    if insufficient_material(pos) {
        return Some(GameResult {
            winner: Winner::Draw,
            reason: "insufficient material",
        });
    }

    // 5. Ply cap: adjudicate a draw so a game can never run forever.
    if ply >= maxplies {
        return Some(GameResult {
            winner: Winner::Draw,
            reason: "max plies",
        });
    }

    None
}

// ---------------------------------------------------------------------------
// One game.
// ---------------------------------------------------------------------------

/// Play a single game between `white` and `black`, seeded with `opening`
/// (space-separated UCI moves applied as forced, validated by the referee).
///
/// Returns the [`GameResult`] from White's perspective. An engine that returns
/// an illegal or missing move immediately loses the game (and the reason names
/// it), which is exactly how this harness catches engine bugs.
fn play_game(
    white: &mut Engine,
    black: &mut Engine,
    opening: &str,
    cfg: &Config,
) -> GameResult {
    let mut pos = Position::startpos();
    let mut moves: Vec<String> = Vec::new();
    let mut reps: HashMap<u64, u8> = HashMap::new();

    // Count the start position as one occurrence for repetition detection.
    *reps.entry(pos.key()).or_insert(0) += 1;

    // Apply the opening line as forced moves. Each is validated against the
    // crate's legal list — a bad built-in opening is a harness bug, so we panic.
    for uci in opening.split_whitespace() {
        let legal = generate_legal(&mut pos);
        let chosen = (&legal).into_iter().find(|m| m.to_string() == uci);
        let m = chosen.unwrap_or_else(|| panic!("built-in opening move {uci} is illegal in {}", pos.to_fen()));
        pos.make_move(m);
        moves.push(uci.to_string());
        *reps.entry(pos.key()).or_insert(0) += 1;
    }

    // Fresh game state for both engines.
    if white.new_game().is_err() {
        return GameResult { winner: Winner::Black, reason: "white failed ucinewgame" };
    }
    if black.new_game().is_err() {
        return GameResult { winner: Winner::White, reason: "black failed ucinewgame" };
    }

    // The main play loop. `ply` counts plies from the start of the game
    // (including the opening moves already applied).
    let mut ply = moves.len();
    loop {
        // Terminal checks come first, before we ask anyone to move.
        if let Some(result) = check_game_over(&mut pos, &reps, ply, cfg.maxplies) {
            return result;
        }

        // Whose engine is on move?
        let stm = pos.side_to_move();
        let engine = match stm {
            Color::White => &mut *white,
            Color::Black => &mut *black,
        };

        // The side that loses if this engine misbehaves is the *other* side.
        let opponent = match stm {
            Color::White => Winner::Black,
            Color::Black => Winner::White,
        };

        // Ask for a move. `None` = timeout / EOF / bestmove 0000 -> forfeit.
        let reply = match engine.bestmove(&moves, cfg.movetime, cfg.depth) {
            Some(mv) => mv,
            None => {
                return GameResult {
                    winner: opponent,
                    reason: "no move (timeout/crash)",
                };
            }
        };

        // Validate the reply against the crate's own legal move list.
        let legal = generate_legal(&mut pos);
        let chosen = (&legal).into_iter().find(|m| m.to_string() == reply);
        let m = match chosen {
            Some(m) => m,
            None => {
                // An illegal move loses the game — this is the bug catcher.
                return GameResult {
                    winner: opponent,
                    reason: "illegal move",
                };
            }
        };

        // Apply it, record it, bump the repetition count, and continue.
        pos.make_move(m);
        moves.push(reply);
        *reps.entry(pos.key()).or_insert(0) += 1;
        ply += 1;
    }
}

// ---------------------------------------------------------------------------
// Scoring & Elo.
// ---------------------------------------------------------------------------

/// The running tally of a match, always from engine A's perspective.
struct Score {
    wins: u32,
    losses: u32,
    draws: u32,
}

impl Score {
    fn new() -> Score {
        Score { wins: 0, losses: 0, draws: 0 }
    }

    /// Total games recorded so far.
    fn games(&self) -> u32 {
        self.wins + self.losses + self.draws
    }

    /// The match score in [0, 1]: (W + 0.5 D) / games.
    fn fraction(&self) -> f64 {
        let g = self.games();
        if g == 0 {
            return 0.5;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / g as f64
    }
}

/// The Elo difference implied by a score fraction, and a 95%-ish error margin.
///
/// Elo from score: `-400 * log10(1/score - 1)`. For a clean sweep (score 0 or 1)
/// there is no finite estimate, so we report the `>+400` / `<-400` bound.
///
/// The error bar uses the per-game result variance: each game scores 1 / 0.5 / 0
/// from A's view, so `sigma = sqrt(var/games)` is the standard error of the mean
/// score. We convert that score error into an Elo error with the local slope of
/// the Elo curve, `dElo/dscore = (400/ln 10) / (score (1-score))`.
fn elo_report(score: &Score) -> String {
    let n = score.games();
    if n == 0 {
        return "Elo n/a (no games)".to_string();
    }
    let frac = score.fraction();

    // A clean sweep has no finite Elo; report the bound and stop.
    if frac <= 0.0 {
        return "Elo <-400 (clean sweep)".to_string();
    }
    if frac >= 1.0 {
        return "Elo >+400 (clean sweep)".to_string();
    }

    let diff = -400.0 * (1.0 / frac - 1.0).log10();

    // Sample variance of the per-game score (population form over the n games).
    let nf = n as f64;
    let mean = frac;
    // Each result r contributes (r - mean)^2; sum via the three outcome buckets.
    let var = (score.wins as f64 * (1.0 - mean).powi(2)
        + score.draws as f64 * (0.5 - mean).powi(2)
        + score.losses as f64 * (0.0 - mean).powi(2))
        / nf;
    let sigma = (var / nf).sqrt(); // standard error of the mean score.

    // Local derivative of Elo w.r.t. score at this operating point.
    let slope = (400.0 / std::f64::consts::LN_10) / (frac * (1.0 - frac));
    let err = slope * sigma;

    format!("Elo {diff:+.1} +/- {err:.1}")
}

// ---------------------------------------------------------------------------
// Match loop & main.
// ---------------------------------------------------------------------------

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = parse_args(&args)?;

    // Spawn both engines (handshake happens inside spawn()).
    let mut engine_a = Engine::spawn(&cfg.engine_a)?;
    let mut engine_b = Engine::spawn(&cfg.engine_b)?;
    let name_a = engine_a.name.clone();
    let name_b = engine_b.name.clone();

    let mut score = Score::new();

    for k in 0..cfg.games {
        // Opening variety: cycle through the list, but play each opening TWICE
        // with colors reversed. Two consecutive games (a "pair") share the same
        // opening; the pair index selects the opening.
        let opening = OPENINGS[(k / 2) % OPENINGS.len()];

        // Alternate colors each game: even game -> A is White, odd -> A is Black.
        let a_is_white = k % 2 == 0;

        // Play with A and B mapped onto White/Black for this game.
        let result = if a_is_white {
            play_game(&mut engine_a, &mut engine_b, opening, &cfg)
        } else {
            play_game(&mut engine_b, &mut engine_a, opening, &cfg)
        };

        // Translate the White-perspective result into A's perspective and tally.
        let (a_scored, tag) = match result.winner {
            Winner::Draw => ("1/2-1/2", 't'),
            Winner::White => (if a_is_white { "1-0" } else { "0-1" }, if a_is_white { 'w' } else { 'l' }),
            Winner::Black => (if a_is_white { "0-1" } else { "1-0" }, if a_is_white { 'l' } else { 'w' }),
        };
        match tag {
            'w' => score.wins += 1,
            'l' => score.losses += 1,
            _ => score.draws += 1,
        }

        // Compact per-game line.
        let side = if a_is_white { "A-as-white" } else { "A-as-black" };
        println!(
            "game {}/{}: {side} [{opening}] -> {a_scored} ({})",
            k + 1,
            cfg.games,
            result.reason
        );
    }

    // Final summary from A's perspective.
    let pct = 100.0 * score.fraction();
    println!(
        "\nEngine A ({name_a}) vs Engine B ({name_b}): +{} -{} ={}  score {:.1}%  {}",
        score.wins,
        score.losses,
        score.draws,
        pct,
        elo_report(&score)
    );

    // Shut both engines down cleanly.
    engine_a.quit();
    engine_b.quit();

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("selfplay: {e}");
        std::process::exit(1);
    }
}
