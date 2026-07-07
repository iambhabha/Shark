//! Training-data generator — self-play many games and record labeled quiet
//! positions to disk for later NNUE training.
//!
//! The idea is the standard "self-play + label" recipe used to bootstrap an
//! NNUE net:
//!
//!  * Play a large number of games where **both sides are the engine itself**,
//!    each move chosen by a fast, shallow [`Searcher::search`].
//!  * Diversify the games with a short **random opening** (a handful of random
//!    legal plies) so the set is not a thousand copies of the same main line.
//!  * For every *quiet, non-check, out-of-opening* position we record a training
//!    sample: the FEN, the engine's search **score** (from the side-to-move's
//!    view), and — once the game finishes — the eventual **result** (again from
//!    that position's side-to-move view). The score teaches the net the engine's
//!    static judgement; the game result anchors it to reality. A later trainer
//!    blends the two.
//!
//! The referee is the `mythos` crate itself, exactly as in the self-play harness:
//! we never trust anything but [`generate_legal`] for legality and game end.
//!
//! ### Why the filters
//!
//! We deliberately *skip* noisy positions:
//!
//!  * **in check** — the static eval of a position mid-check is jumpy and not
//!    what the net will be asked to judge in a quiet leaf;
//!  * **best move is a capture/promotion** — a "tactical" position whose score
//!    hinges on a pending material swing adds label noise (the true value lives
//!    a few plies deeper, after the dust settles);
//!  * **inside the random opening** — those plies are random, not engine play,
//!    so their scores are meaningless.
//!
//! ### Output
//!
//! One sample per line, pipe-separated and trivial to parse later:
//!
//! ```text
//! <FEN> | <stm_score_cp> | <stm_result>
//! ```
//!
//! where `stm_result` is `1.0` (the side to move eventually won), `0.5` (draw),
//! or `0.0` (lost). A `#` header line documents the format. Search prints its
//! UCI `info` lines to **stdout**; all data goes to the **file**, and progress to
//! **stderr**, so the three streams never collide.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use mythos::movegen::generate_legal;
use mythos::position::Position;
use mythos::search::{SearchLimits, SearchResult, Searcher};
use mythos::types::{Color, Move, MoveType};

// ---------------------------------------------------------------------------
// Tuning constants.
// ---------------------------------------------------------------------------

/// Transposition-table size (MB) for each per-thread [`Searcher`]. Small on
/// purpose: datagen runs many shallow searches, so a big TT would just waste RAM.
const TT_SIZE_MB: usize = 16;

/// Minimum / maximum number of random opening plies played at the start of a
/// game to diversify the set (chosen uniformly in this inclusive range).
const OPENING_MIN_PLIES: usize = 4;
const OPENING_MAX_PLIES: usize = 10;

/// Hard cap on total plies per game before we adjudicate, so a game can never run
/// forever.
const MAX_GAME_PLIES: usize = 200;

/// Score clamp for recorded samples (centipawns). Scores are clipped to
/// `[-SCORE_CLAMP, SCORE_CLAMP]`, and a forced-mate score is stored as the
/// signed clamp value rather than the raw ±30000 mate number.
const SCORE_CLAMP: i32 = 3000;

/// Any |score| at or above this is treated as a forced mate (mirrors the search's
/// `MATE`/`MATE_IN_MAX` band: `MATE = 30000`, `MAX_PLY = 128`).
const MATE_THRESHOLD: i32 = 30000 - 128;

/// Adjudication threshold: if the side-to-move's score stays beyond this (in
/// centipawns) at the ply cap, we award the game to the better side instead of
/// calling it a draw.
const ADJUDICATE_CP: i32 = 800;

/// Flush the shared writer roughly every this many samples so a crash loses at
/// most a small tail of data.
const FLUSH_EVERY: usize = 4096;

// ---------------------------------------------------------------------------
// Deterministic PRNG — splitmix64 (no external crates).
// ---------------------------------------------------------------------------

/// A tiny, fast, fully deterministic PRNG (splitmix64). Seeded per (base, game)
/// so a run is exactly reproducible from `--seed`, and each game draws an
/// independent stream.
struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator. Any seed (including 0) is fine for splitmix64.
    fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    /// Next raw 64-bit value (the standard splitmix64 step).
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// A uniform value in `0..bound` (bound must be non-zero). Modulo bias is
    /// negligible for the tiny bounds we use (move counts, small ply ranges).
    #[inline]
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Mix two seeds into one so `(base_seed, game_index)` gives an independent,
/// reproducible stream per game. This is the splitmix64 finalizer applied to the
/// sum, which decorrelates nearby indices well.
fn mix_seed(base: u64, game_index: u64) -> u64 {
    let mut z = base
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(game_index.wrapping_add(1).wrapping_mul(0xD1B54A32D192ED03));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// Configuration.
// ---------------------------------------------------------------------------

/// The parsed command-line configuration for a datagen run.
struct Config {
    /// Where to write the samples.
    output: String,
    /// Total number of games to generate across all threads.
    games: usize,
    /// Fixed search depth per move (used when `nodes` is `None`).
    depth: u32,
    /// Node budget per move; when `Some`, limits searches by nodes instead of depth.
    nodes: Option<u64>,
    /// Number of worker threads.
    threads: usize,
    /// Base PRNG seed for reproducibility.
    seed: u64,
}

/// Print a usage line to stderr.
fn usage() -> String {
    "usage: datagen <output_file> [--games N] [--depth D] [--nodes X] \
     [--threads T] [--seed S]"
        .to_string()
}

/// Parse `argv` (already stripped of the program name) into a [`Config`].
fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut positional: Vec<String> = Vec::new();
    let mut games = 1000usize;
    let mut depth = 8u32;
    let mut nodes: Option<u64> = None;
    // Default thread count: available parallelism, capped at 8.
    let mut threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8);
    let mut seed = 1u64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                let v = args.get(i).ok_or("--games needs a value")?;
                games = v.parse().map_err(|_| format!("bad --games value: {v}"))?;
            }
            "--depth" => {
                i += 1;
                let v = args.get(i).ok_or("--depth needs a value")?;
                depth = v.parse().map_err(|_| format!("bad --depth value: {v}"))?;
            }
            "--nodes" => {
                i += 1;
                let v = args.get(i).ok_or("--nodes needs a value")?;
                nodes = Some(v.parse().map_err(|_| format!("bad --nodes value: {v}"))?);
            }
            "--threads" => {
                i += 1;
                let v = args.get(i).ok_or("--threads needs a value")?;
                threads = v.parse().map_err(|_| format!("bad --threads value: {v}"))?;
            }
            "--seed" => {
                i += 1;
                let v = args.get(i).ok_or("--seed needs a value")?;
                seed = v.parse().map_err(|_| format!("bad --seed value: {v}"))?;
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }

    if positional.is_empty() {
        return Err(usage());
    }
    if threads == 0 {
        threads = 1;
    }

    Ok(Config {
        output: positional[0].clone(),
        games,
        depth,
        nodes,
        threads,
        seed,
    })
}

// ---------------------------------------------------------------------------
// Sample & game bookkeeping.
// ---------------------------------------------------------------------------

/// One position recorded during a game, before the final result is known.
///
/// We keep the FEN and the side-to-move score at record time; the game result
/// (from this position's side-to-move view) is filled in once the game ends.
struct Sample {
    /// The position, as a FEN string.
    fen: String,
    /// Search score in centipawns, from this position's side-to-move view,
    /// already clamped to `[-SCORE_CLAMP, SCORE_CLAMP]`.
    stm_score: i32,
    /// The side to move in this position — used to project the final game result
    /// onto this sample's perspective.
    stm: Color,
}

/// The terminal outcome of a game, from White's point of view.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    WhiteWin,
    BlackWin,
    Draw,
}

impl Outcome {
    /// The result value in `{1.0, 0.5, 0.0}` from `stm`'s perspective.
    fn result_for(self, stm: Color) -> f64 {
        match self {
            Outcome::Draw => 0.5,
            Outcome::WhiteWin => {
                if stm == Color::White {
                    1.0
                } else {
                    0.0
                }
            }
            Outcome::BlackWin => {
                if stm == Color::Black {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Move classification helpers (mirrors the search's notion of "quiet").
// ---------------------------------------------------------------------------

/// Does `m` capture an enemy piece (including en passant)?
fn is_capture(pos: &Position, m: Move) -> bool {
    if m.move_type() == MoveType::EnPassant {
        return true;
    }
    pos.piece_at(m.to_sq()).is_some()
}

/// Is `m` a "noisy" move — a capture or a promotion? Positions whose best move is
/// noisy are skipped (their score hinges on a pending material swing).
fn is_noisy(pos: &Position, m: Move) -> bool {
    is_capture(pos, m) || m.move_type() == MoveType::Promotion
}

// ---------------------------------------------------------------------------
// Score / outcome helpers.
// ---------------------------------------------------------------------------

/// Clamp a raw search score into the recorded range, mapping forced mates to the
/// signed clamp value.
fn clamp_score(score: i32) -> i32 {
    if score >= MATE_THRESHOLD {
        SCORE_CLAMP
    } else if score <= -MATE_THRESHOLD {
        -SCORE_CLAMP
    } else {
        score.clamp(-SCORE_CLAMP, SCORE_CLAMP)
    }
}

/// Adjudicate the ply-capped game from White's last score: a big enough edge is a
/// win for the better side, otherwise a draw.
fn adjudicate(white_score: i32) -> Outcome {
    if white_score > ADJUDICATE_CP {
        Outcome::WhiteWin
    } else if white_score < -ADJUDICATE_CP {
        Outcome::BlackWin
    } else {
        Outcome::Draw
    }
}

// ---------------------------------------------------------------------------
// Worker thread.
// ---------------------------------------------------------------------------

/// Everything a worker needs to run its slice of the games.
struct Worker {
    /// Global config (depth/nodes/seed).
    depth: u32,
    nodes: Option<u64>,
    seed: u64,
    /// The half-open range of global game indices this worker owns.
    game_lo: u64,
    game_hi: u64,
    /// Shared, buffered output guarded by a mutex (locked per *batch*, not line).
    writer: Arc<Mutex<BufWriter<File>>>,
    /// Shared progress counters (for the stderr progress line).
    games_done: Arc<AtomicU64>,
    samples_done: Arc<AtomicU64>,
    total_games: u64,
    /// The stop flag handed to every search (never set true here, but the API
    /// requires it; a real datagen could wire in a Ctrl-C handler).
    stop: Arc<AtomicBool>,
}

/// Format one sample into an output line: `<FEN> | <cp> | <result>`.
fn format_line(fen: &str, cp: i32, result: f64) -> String {
    format!("{fen} | {cp} | {result:.1}\n")
}

impl Worker {
    /// Generate this worker's games, appending samples to the shared writer.
    fn run(&self) {
        // A per-thread searcher (owns its own TT + heuristic tables).
        let mut searcher = Searcher::new(TT_SIZE_MB);

        // Build the search limits once: node-limited if `--nodes` was given, else
        // fixed-depth.
        let limits = SearchLimits {
            depth: if self.nodes.is_some() {
                None
            } else {
                Some(self.depth)
            },
            nodes: self.nodes,
            ..SearchLimits::default()
        };

        // Buffer samples locally and flush a batch under the lock periodically, so
        // we contend on the mutex rarely rather than once per line.
        let mut batch: Vec<String> = Vec::new();
        let mut since_flush = 0usize;

        for g in self.game_lo..self.game_hi {
            let game_seed = mix_seed(self.seed, g);

            // Fresh searcher state per game so one game's TT/killers don't bleed
            // into the next (keeps games independent and reproducible).
            searcher.clear();

            let mut game_samples: Vec<Labeled> = Vec::new();
            let n = play_game_labeled(
                &mut searcher,
                &limits,
                &self.stop,
                game_seed,
                &mut game_samples,
            );

            // Turn labeled samples into output lines.
            for s in &game_samples {
                batch.push(format_line(&s.fen, s.stm_score, s.result));
            }
            since_flush += n;

            // Progress + periodic flush.
            let done = self.games_done.fetch_add(1, Ordering::Relaxed) + 1;
            self.samples_done.fetch_add(n as u64, Ordering::Relaxed);

            if since_flush >= FLUSH_EVERY {
                self.flush_batch(&mut batch);
                since_flush = 0;
            }

            // Emit a progress line to stderr every so often (not to stdout / file).
            if done % 25 == 0 || done == self.total_games {
                let total_samples = self.samples_done.load(Ordering::Relaxed);
                eprintln!(
                    "datagen: {done}/{} games, {total_samples} samples",
                    self.total_games
                );
            }
        }

        // Final flush of whatever is left in this worker's batch.
        self.flush_batch(&mut batch);
    }

    /// Write and clear the local batch under a single lock acquisition.
    fn flush_batch(&self, batch: &mut Vec<String>) {
        if batch.is_empty() {
            return;
        }
        let mut w = self.writer.lock().expect("output mutex poisoned");
        for line in batch.drain(..) {
            // Best-effort: a write error here means the disk is gone; report and
            // bail loudly rather than silently dropping data.
            if let Err(e) = w.write_all(line.as_bytes()) {
                eprintln!("datagen: write error: {e}");
                break;
            }
        }
        let _ = w.flush();
    }
}

// ---------------------------------------------------------------------------
// Labeled sample (result attached) & the game driver that produces them.
// ---------------------------------------------------------------------------

/// A finished, labeled sample ready to be written: FEN, side-to-move score, and
/// the game result from the side-to-move's perspective.
struct Labeled {
    fen: String,
    stm_score: i32,
    result: f64,
}

/// Play one game and return its samples already labeled with the game result.
///
/// This is the clean entry the worker uses. It plays the game (random opening +
/// engine self-play), then projects the final [`Outcome`] onto every recorded
/// position via [`Outcome::result_for`]. Returns the number of samples.
fn play_game_labeled(
    searcher: &mut Searcher,
    limits: &SearchLimits,
    stop: &Arc<AtomicBool>,
    game_seed: u64,
    out: &mut Vec<Labeled>,
) -> usize {
    // Retry with perturbed seeds if the random opening dead-ends.
    for attempt in 0..8u64 {
        let mut rng = Rng::new(game_seed ^ (attempt.wrapping_mul(0xA0761D6478BD642F)));
        let mut raw: Vec<Sample> = Vec::new();
        if let Some(outcome) = run_game(searcher, limits, stop, &mut rng, &mut raw) {
            for s in raw {
                out.push(Labeled {
                    result: outcome.result_for(s.stm),
                    fen: s.fen,
                    stm_score: s.stm_score,
                });
            }
            return out.len();
        }
    }
    0
}

/// Run a single game attempt: random opening, then engine self-play to the end,
/// collecting raw (unlabeled) samples. Returns `Some(outcome)` on a real game, or
/// `None` if the random opening ended the game before engine play (retry signal).
fn run_game(
    searcher: &mut Searcher,
    limits: &SearchLimits,
    stop: &Arc<AtomicBool>,
    rng: &mut Rng,
    out: &mut Vec<Sample>,
) -> Option<Outcome> {
    let mut pos = Position::startpos();

    let mut reps: HashMap<u64, u8> = HashMap::new();
    *reps.entry(pos.key()).or_insert(0) += 1;

    // --- Random opening. ---------------------------------------------------
    let opening_plies =
        OPENING_MIN_PLIES + rng.below(OPENING_MAX_PLIES - OPENING_MIN_PLIES + 1);
    for _ in 0..opening_plies {
        let legal = generate_legal(&mut pos);
        if legal.is_empty() {
            return None; // random line dead-ended; caller retries.
        }
        let m = legal[rng.below(legal.len())];
        pos.make_move(m);
        *reps.entry(pos.key()).or_insert(0) += 1;
    }
    if generate_legal(&mut pos).is_empty() {
        return None;
    }

    // --- Engine self-play. -------------------------------------------------
    let mut ply = opening_plies;
    let mut last_white_score: i32 = 0;

    let outcome = loop {
        // Terminal checks first.
        let legal = generate_legal(&mut pos);
        if legal.is_empty() {
            break if pos.in_check() {
                mated_outcome(pos.side_to_move())
            } else {
                Outcome::Draw
            };
        }
        if pos.halfmove_clock() >= 100 {
            break Outcome::Draw;
        }
        if reps.get(&pos.key()).copied().unwrap_or(0) >= 3 {
            break Outcome::Draw;
        }
        if ply >= MAX_GAME_PLIES {
            break adjudicate(last_white_score);
        }

        // Engine move.
        let result: SearchResult = searcher.search(&pos, limits, stop);
        let best = result.best_move;
        if best.is_none() {
            // Terminal position the search reported — resolve as above.
            break if pos.in_check() {
                mated_outcome(pos.side_to_move())
            } else {
                Outcome::Draw
            };
        }

        let stm = pos.side_to_move();
        let stm_score = result.score;
        last_white_score = match stm {
            Color::White => stm_score,
            Color::Black => -stm_score,
        };

        // Record a quiet, non-check sample.
        if !pos.in_check() && !is_noisy(&pos, best) {
            out.push(Sample {
                fen: pos.to_fen(),
                stm_score: clamp_score(stm_score),
                stm,
            });
        }

        pos.make_move(best);
        *reps.entry(pos.key()).or_insert(0) += 1;
        ply += 1;
    };

    Some(outcome)
}

/// The outcome when the side `stm` is checkmated (the *other* side won).
fn mated_outcome(stm: Color) -> Outcome {
    match stm {
        Color::White => Outcome::BlackWin,
        Color::Black => Outcome::WhiteWin,
    }
}

// ---------------------------------------------------------------------------
// main / run.
// ---------------------------------------------------------------------------

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = parse_args(&args)?;

    // Open the output file and wrap it in a buffered writer behind a mutex.
    let file = File::create(&cfg.output)
        .map_err(|e| format!("cannot create output file {}: {e}", cfg.output))?;
    let mut writer = BufWriter::new(file);

    // Header line documenting the format (starts with `#` so parsers can skip it).
    let mode = match cfg.nodes {
        Some(n) => format!("nodes={n}"),
        None => format!("depth={}", cfg.depth),
    };
    writeln!(
        writer,
        "# Mythos datagen: <FEN> | <stm_score_cp> | <stm_result>  \
         (result 1.0/0.5/0.0 from side-to-move; score clamped to +/-{SCORE_CLAMP}cp; \
         {mode}, seed={})",
        cfg.seed
    )
    .map_err(|e| format!("cannot write header: {e}"))?;
    writer
        .flush()
        .map_err(|e| format!("cannot flush header: {e}"))?;

    let writer = Arc::new(Mutex::new(writer));
    let games_done = Arc::new(AtomicU64::new(0));
    let samples_done = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));

    let total_games = cfg.games as u64;

    // Split the games as evenly as possible across the workers.
    let threads = cfg.threads.max(1);
    let base = cfg.games / threads;
    let extra = cfg.games % threads;

    eprintln!(
        "datagen: {} games on {} thread(s), {}, seed {} -> {}",
        cfg.games, threads, mode, cfg.seed, cfg.output
    );

    let mut handles = Vec::new();
    let mut lo: u64 = 0;
    for t in 0..threads {
        // The first `extra` workers get one more game so the total is exact.
        let count = base + if t < extra { 1 } else { 0 };
        let hi = lo + count as u64;

        let worker = Worker {
            depth: cfg.depth,
            nodes: cfg.nodes,
            seed: cfg.seed,
            game_lo: lo,
            game_hi: hi,
            writer: Arc::clone(&writer),
            games_done: Arc::clone(&games_done),
            samples_done: Arc::clone(&samples_done),
            total_games,
            stop: Arc::clone(&stop),
        };

        handles.push(thread::spawn(move || worker.run()));
        lo = hi;
    }

    // Wait for every worker to finish; propagate a panic as an error string.
    for h in handles {
        h.join().map_err(|_| "a worker thread panicked".to_string())?;
    }

    // Final flush (workers already flush, but be certain nothing is buffered).
    {
        let mut w = writer.lock().expect("output mutex poisoned");
        w.flush().map_err(|e| format!("final flush failed: {e}"))?;
    }

    let total_samples = samples_done.load(Ordering::Relaxed);
    let total_done = games_done.load(Ordering::Relaxed);
    eprintln!("datagen: done — {total_done} games, {total_samples} samples written to {}", cfg.output);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("datagen: {e}");
        std::process::exit(1);
    }
}
