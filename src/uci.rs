//! The UCI (Universal Chess Interface) protocol loop — how a GUI talks to Mythos.
//!
//! UCI is a line-based text protocol: the GUI writes commands to our stdin
//! (`position`, `go`, `stop`, ...) and reads our replies from stdout (`info`,
//! `bestmove`, ...). This module is the outermost layer of the engine — it owns
//! the current [`Position`], the persistent [`Searcher`] (and therefore the
//! transposition table), and drives everything below it.
//!
//! The one genuinely tricky part is **staying responsive while thinking**. A UCI
//! engine must react to `stop` (and answer `isready`) even in the middle of a
//! long search. We get that by running the search on its own thread, sharing a
//! [`AtomicBool`] "stop" flag the search polls: the main thread keeps reading
//! stdin and can flip that flag at any moment. The [`Searcher`] lives behind a
//! `Mutex` so its transposition table survives across `go` commands, but note the
//! search thread *holds that lock for the whole search* — so the main thread must
//! never block on it to answer `isready` (see below).
//!
//! The command parsing is factored into small pure helpers — [`parse_position`],
//! [`parse_go`], [`apply_uci_move`] — that touch neither stdin nor threads, so
//! they can be unit-tested directly.

use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::eval::evaluate;
use crate::movegen::generate_legal;
use crate::perft::{format_divide, perft_divide};
use crate::position::Position;
use crate::search::{SearchLimits, Searcher};
use crate::types::Move;

/// The transposition-table size (in MB) a fresh engine starts with, and the
/// `Hash` UCI option's advertised default.
const DEFAULT_HASH_MB: usize = 16;

// ---------------------------------------------------------------------------
// The main loop.
// ---------------------------------------------------------------------------

/// Run the UCI read-eval-print loop over stdin until `quit` (or EOF).
///
/// Blocks the calling thread for the lifetime of the engine; spawns a worker
/// thread per `go` command for the actual search so this loop stays free to read
/// further commands (notably `stop`).
pub fn uci_loop() {
    // The searcher owns the TT, which must persist across searches — so it lives
    // behind a shared `Mutex`, never recreated except for a `Hash` resize.
    let mut searcher = Arc::new(Mutex::new(Searcher::new(DEFAULT_HASH_MB)));

    // If a default net (`mythos.nnue`) is found next to the exe or in the working
    // directory, load it so the engine plays with NNUE out of the box. Absent one,
    // we say nothing and quietly use the hand-crafted evaluation.
    if let Some(net) = crate::nnue::load_default() {
        searcher.lock().unwrap().set_net(Some(net));
        println!("info string NNUE evaluation loaded");
    }
    // The flag the running search polls; the main thread flips it for `stop`.
    let stop = Arc::new(AtomicBool::new(false));
    // The current board the next `go` will search from.
    let mut position = Position::startpos();
    // The in-flight search thread, if one is running.
    let mut search_handle: Option<JoinHandle<()>> = None;

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // stdin closed / read error: shut down cleanly.
        };

        let mut tokens = line.split_whitespace();
        let Some(command) = tokens.next() else {
            continue; // empty line: ignore (UCI convention).
        };
        // Everything after the first word, verbatim, for the parse helpers.
        let args = line[command.len()..].trim_start();

        match command {
            "uci" => print_id(),

            // `isready` must be answered even mid-search. The search thread holds
            // the searcher lock the whole time it runs, so we deliberately do NOT
            // touch the searcher here — we just reply immediately.
            "isready" => println!("readyok"),

            "ucinewgame" => {
                // A new game: no search may be running, then wipe learned state.
                stop_search(&stop, &mut search_handle);
                searcher.lock().unwrap().clear();
                position = Position::startpos();
            }

            "setoption" => {
                // Reconfiguring touches the searcher, so make sure it is idle.
                stop_search(&stop, &mut search_handle);
                handle_setoption(args, &mut searcher);
            }

            "position" => {
                // A new search must not be racing the old position; be safe.
                stop_search(&stop, &mut search_handle);
                position = parse_position(args);
            }

            "go" => {
                let limits = parse_go(args);
                start_search(&limits, &position, &searcher, &stop, &mut search_handle);
            }

            "stop" => stop.store(true, Ordering::Relaxed),

            "quit" => {
                stop_search(&stop, &mut search_handle);
                break;
            }

            // --- Non-UCI debugging conveniences. -----------------------------
            "d" => {
                println!("{position}");
            }
            "eval" => {
                stop_search(&stop, &mut search_handle);
                println!("eval: {} cp", evaluate(&position));
            }
            "perft" => {
                stop_search(&stop, &mut search_handle);
                handle_perft(args, &mut position);
            }

            // Unknown command: ignore silently, as UCI requires.
            _ => {}
        }
    }

    // On the way out (quit or EOF) make sure no search thread is left running.
    stop_search(&stop, &mut search_handle);
}

// ---------------------------------------------------------------------------
// `go` — launch (and tear down) the search thread.
// ---------------------------------------------------------------------------

/// Spawn the search on a worker thread so the main loop stays responsive.
///
/// Any previous search is stopped and joined first — we never run two at once.
/// The worker locks the shared [`Searcher`] (keeping the TT warm), runs the
/// search, and prints the `bestmove` line itself.
fn start_search(
    limits: &SearchLimits,
    position: &Position,
    searcher: &Arc<Mutex<Searcher>>,
    stop: &Arc<AtomicBool>,
    handle: &mut Option<JoinHandle<()>>,
) {
    // Never run two searches concurrently: stop and join any predecessor.
    stop_search(stop, handle);

    // Fresh start for this search.
    stop.store(false, Ordering::Relaxed);

    let searcher = Arc::clone(searcher);
    let stop = Arc::clone(stop);
    let root = position.clone();
    let limits = limits.clone();

    *handle = Some(std::thread::spawn(move || {
        // Hold the searcher lock for the whole search so the TT (and the
        // heuristic tables) persist and stay consistent.
        let mut searcher = searcher.lock().unwrap();
        let result = searcher.search(&root, &limits, &stop);
        // Report the chosen move. A position with no legal move yields
        // `Move::NONE`, which UCI represents as the null move "0000".
        if result.best_move.is_none() {
            println!("bestmove 0000");
        } else {
            println!("bestmove {}", result.best_move);
        }
    }));
}

/// Stop any running search and wait for its thread to finish.
///
/// Sets the stop flag (which the search polls and honours by returning promptly),
/// then joins the worker so its `bestmove` has definitely been printed before we
/// move on. A no-op if nothing is running.
fn stop_search(stop: &Arc<AtomicBool>, handle: &mut Option<JoinHandle<()>>) {
    if let Some(h) = handle.take() {
        stop.store(true, Ordering::Relaxed);
        // If the worker panicked there is nothing sensible to do but carry on.
        let _ = h.join();
    }
}

// ---------------------------------------------------------------------------
// `uci` — identify ourselves and advertise our options.
// ---------------------------------------------------------------------------

/// Print the `id` lines, the supported options, and the closing `uciok`.
fn print_id() {
    println!("id name Mythos 0.1.0");
    println!("id author Mythos contributors");
    println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 4096");
    println!("option name Clear Hash type button");
    println!("option name UseNNUE type check default true");
    println!("option name EvalFile type string default <empty>");
    println!("uciok");
}

// ---------------------------------------------------------------------------
// `setoption` — the options we understand.
// ---------------------------------------------------------------------------

/// Handle `setoption name <Name> [value <V>]`.
///
/// We support `Hash` (recreates the searcher with the requested TT size — the old
/// table's contents are intentionally dropped), `Clear Hash` (a button that just
/// wipes the current table), `EvalFile` (load an NNUE net from a path), and
/// `UseNNUE` (a toggle that loads the default net or clears it to force HCE). Any
/// unknown option is ignored gracefully.
fn handle_setoption(args: &str, searcher: &mut Arc<Mutex<Searcher>>) {
    // Expected shape: `name <Name...> [value <V...>]`. The option name can be
    // several words ("Clear Hash"), so split on the `value` keyword.
    let rest = match args.strip_prefix("name ").or_else(|| {
        // Tolerate exactly "name" with nothing after, and leading whitespace.
        args.trim().strip_prefix("name").map(|s| s.trim_start())
    }) {
        Some(r) => r,
        None => return, // malformed: no `name` — ignore.
    };

    let (name, value) = match rest.split_once(" value ") {
        Some((n, v)) => (n.trim(), Some(v.trim())),
        None => (rest.trim(), None),
    };

    // Option names are matched case-insensitively (GUIs are inconsistent).
    if name.eq_ignore_ascii_case("Hash") {
        if let Some(mb) = value.and_then(|v| v.parse::<usize>().ok()) {
            let mb = mb.clamp(1, 4096);
            // Recreate the searcher with the new size, preserving nothing.
            *searcher = Arc::new(Mutex::new(Searcher::new(mb)));
        }
    } else if name.eq_ignore_ascii_case("Clear Hash") {
        searcher.lock().unwrap().clear();
    } else if name.eq_ignore_ascii_case("EvalFile") {
        // Load an NNUE net from an explicit path, replacing any current net.
        if let Some(path) = value {
            match crate::nnue::Net::load(path) {
                Ok(net) => {
                    searcher.lock().unwrap().set_net(Some(net));
                    println!("info string loaded net {path}");
                }
                Err(_) => println!("info string failed to load net"),
            }
        }
    } else if name.eq_ignore_ascii_case("UseNNUE") {
        // `true` loads the default net (if one is available); `false` clears any
        // net so the engine falls back to the hand-crafted evaluation.
        let on = value.map(|v| v.eq_ignore_ascii_case("true")).unwrap_or(false);
        if on {
            match crate::nnue::load_default() {
                Some(net) => {
                    searcher.lock().unwrap().set_net(Some(net));
                    println!("info string NNUE evaluation loaded");
                }
                None => println!("info string failed to load net"),
            }
        } else {
            searcher.lock().unwrap().set_net(None);
        }
    }
    // Unknown option: ignored.
}

// ---------------------------------------------------------------------------
// `perft` — the debugging move-count command.
// ---------------------------------------------------------------------------

/// Handle `perft N`: run a per-move divide from the current position and print it.
fn handle_perft(args: &str, position: &mut Position) {
    let depth: u32 = match args.split_whitespace().next().and_then(|s| s.parse().ok()) {
        Some(d) => d,
        None => {
            println!("perft: expected a depth, e.g. `perft 5`");
            return;
        }
    };
    let divide = perft_divide(position, depth);
    print!("{}", format_divide(&divide));
}

// ---------------------------------------------------------------------------
// Pure, testable parsers.
// ---------------------------------------------------------------------------

/// Parse everything after the word `position` into the resulting [`Position`].
///
/// Accepts `startpos [moves ...]` and `fen <6 fields> [moves ...]`. The base
/// position is set up first (start position or the FEN), then each move token is
/// applied in order by matching it against the legal moves. A malformed FEN falls
/// back to the start position; an unrecognised move token stops move application
/// (the position so far is returned).
pub fn parse_position(args: &str) -> Position {
    let mut tokens = args.split_whitespace().peekable();

    let mut pos = match tokens.next() {
        Some("startpos") => Position::startpos(),
        Some("fen") => {
            // The FEN is the up-to-six fields before an optional `moves` keyword.
            let mut fen = String::new();
            while let Some(&tok) = tokens.peek() {
                if tok == "moves" {
                    break;
                }
                if !fen.is_empty() {
                    fen.push(' ');
                }
                fen.push_str(tok);
                tokens.next();
            }
            Position::from_fen(&fen).unwrap_or_else(|_| Position::startpos())
        }
        // Nothing or something unexpected: default to the start position.
        _ => Position::startpos(),
    };

    // Skip the `moves` keyword if present, then apply each move in turn.
    if let Some(&tok) = tokens.peek()
        && tok == "moves"
    {
        tokens.next();
    }
    for mv in tokens {
        if !apply_uci_move(&mut pos, mv) {
            break; // stop at the first move we can't make (illegal / garbage).
        }
    }

    pos
}

/// Parse everything after the word `go` into a [`SearchLimits`].
///
/// Recognises `depth N`, `movetime MS`, `wtime/btime/winc/binc MS`, `movestogo N`,
/// `nodes N`, and `infinite`. Unknown tokens are skipped. Anything that fails to
/// parse as a number leaves that field unset.
pub fn parse_go(args: &str) -> SearchLimits {
    let mut limits = SearchLimits::default();
    let mut tokens = args.split_whitespace();

    // Each keyword consumes the following token and parses it into the field's
    // own numeric type; a missing or unparseable value simply leaves it unset.
    while let Some(tok) = tokens.next() {
        match tok {
            "depth" => limits.depth = tokens.next().and_then(|s| s.parse().ok()),
            "movetime" => limits.movetime_ms = tokens.next().and_then(|s| s.parse().ok()),
            "wtime" => limits.wtime = tokens.next().and_then(|s| s.parse().ok()),
            "btime" => limits.btime = tokens.next().and_then(|s| s.parse().ok()),
            "winc" => limits.winc = tokens.next().and_then(|s| s.parse().ok()),
            "binc" => limits.binc = tokens.next().and_then(|s| s.parse().ok()),
            "movestogo" => limits.movestogo = tokens.next().and_then(|s| s.parse().ok()),
            "nodes" => limits.nodes = tokens.next().and_then(|s| s.parse().ok()),
            "infinite" => limits.infinite = true,
            _ => {} // unknown token: ignore.
        }
    }

    limits
}

/// Find the legal move whose UCI string equals `mv_str` and make it on `pos`.
///
/// Returns `true` if such a move existed and was played, `false` if the token did
/// not match any legal move (illegal, malformed, or a null move). Relies on the
/// fact that `Move`'s `Display` *is* UCI long-algebraic notation, so we simply
/// compare each legal move's rendering against the token.
pub fn apply_uci_move(pos: &mut Position, mv_str: &str) -> bool {
    let mut found = Move::NONE;
    for m in &generate_legal(pos) {
        if m.to_string() == mv_str {
            found = m;
            break;
        }
    }
    if found.is_none() {
        return false;
    }
    pos.make_move(found);
    true
}

// ---------------------------------------------------------------------------
// Tests — the pure helpers, no stdin / threads involved.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // -- parse_position ---------------------------------------------------

    #[test]
    fn parse_position_startpos() {
        let pos = parse_position("startpos");
        assert_eq!(pos.to_fen(), STARTPOS_FEN);
    }

    #[test]
    fn parse_position_startpos_with_moves() {
        // 1. e4 e5 2. Nf3 — after three plies it is Black to move.
        let pos = parse_position("startpos moves e2e4 e7e5 g1f3");
        let fen = pos.to_fen();
        // Black to move after an odd number of plies.
        assert!(fen.contains(" b "), "expected Black to move, got FEN: {fen}");
        // Piece-placement spot checks via the FEN's first field.
        let placement = fen.split(' ').next().unwrap();
        // White pawn advanced to e4 (rank 4), knight developed to f3 (rank 3),
        // black pawn on e5 (rank 5). Rank 4 (from rank 8 down: index 4) is "4P3".
        let ranks: Vec<&str> = placement.split('/').collect();
        assert_eq!(ranks[3], "4p3", "black pawn should sit on e5"); // rank 5
        assert_eq!(ranks[4], "4P3", "white pawn should sit on e4"); // rank 4
        assert_eq!(ranks[5], "5N2", "white knight should sit on f3"); // rank 3
        // And the full FEN is exactly the well-known position after these moves.
        assert_eq!(
            fen,
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        );
    }

    #[test]
    fn parse_position_fen_with_moves() {
        let pos = parse_position(
            "fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves d2d4",
        );
        // After 1. d4 it is Black to move, with the double push's ep target on d3.
        assert_eq!(
            pos.to_fen(),
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"
        );
    }

    #[test]
    fn parse_position_bad_fen_falls_back() {
        // A garbage FEN should not panic; it falls back to the start position.
        let pos = parse_position("fen not a real fen at all xx yy");
        assert_eq!(pos.to_fen(), STARTPOS_FEN);
    }

    // -- parse_go ---------------------------------------------------------

    #[test]
    fn parse_go_depth() {
        let limits = parse_go("depth 8");
        assert_eq!(limits.depth, Some(8));
    }

    #[test]
    fn parse_go_time_controls() {
        let limits = parse_go("wtime 300000 btime 300000 winc 1000 binc 1000");
        assert_eq!(limits.wtime, Some(300_000));
        assert_eq!(limits.btime, Some(300_000));
        assert_eq!(limits.winc, Some(1000));
        assert_eq!(limits.binc, Some(1000));
    }

    #[test]
    fn parse_go_movetime() {
        let limits = parse_go("movetime 1000");
        assert_eq!(limits.movetime_ms, Some(1000));
    }

    #[test]
    fn parse_go_infinite() {
        let limits = parse_go("infinite");
        assert!(limits.infinite);
    }

    #[test]
    fn parse_go_nodes() {
        let limits = parse_go("nodes 100000");
        assert_eq!(limits.nodes, Some(100_000));
    }

    #[test]
    fn parse_go_movestogo_and_mixed() {
        // A realistic mixed command; every recognised field should land.
        let limits = parse_go("wtime 60000 btime 60000 movestogo 20 depth 12");
        assert_eq!(limits.wtime, Some(60_000));
        assert_eq!(limits.btime, Some(60_000));
        assert_eq!(limits.movestogo, Some(20));
        assert_eq!(limits.depth, Some(12));
        assert!(!limits.infinite);
    }

    // -- apply_uci_move ---------------------------------------------------

    #[test]
    fn apply_uci_move_legal_and_illegal() {
        let mut pos = Position::startpos();
        // A legal opening move plays and returns true.
        assert!(apply_uci_move(&mut pos, "e2e4"));
        assert_eq!(pos.side_to_move(), crate::types::Color::Black);

        // An illegal move from the start position returns false and is a no-op.
        let mut pos = Position::startpos();
        let before = pos.to_fen();
        assert!(!apply_uci_move(&mut pos, "e2e5")); // pawn can't jump 3 squares
        assert_eq!(pos.to_fen(), before, "an illegal move must not change the board");
    }

    #[test]
    fn apply_uci_move_promotion() {
        // A crafted position with a white pawn on e7 that can promote on e8. The
        // black king is on a8 so e8 is empty and the straight push is legal.
        let mut pos = Position::from_fen("k7/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(apply_uci_move(&mut pos, "e7e8q"), "e7e8q should be legal");
        // e8 now holds a white queen (rank 8 = "k3Q3": king a8, queen e8) and
        // e7 is empty, so it is now Black to move.
        assert_eq!(
            pos.to_fen(),
            "k3Q3/8/8/8/8/8/8/4K3 b - - 0 1",
            "e8 should now hold a promoted queen"
        );
    }

    #[test]
    fn apply_uci_move_unknown_token() {
        // Total garbage is rejected without a panic.
        let mut pos = Position::startpos();
        assert!(!apply_uci_move(&mut pos, "zzzz"));
    }
}
