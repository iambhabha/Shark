//! Mythos engine binary.
//!
//! By default this starts the UCI protocol loop, so any chess GUI (Arena,
//! Cute Chess, BanksiaGUI, a lichess-bot, ...) can talk to the engine. A few
//! command-line helpers are available for quick manual checks:
//!
//! - `mythos`            → run the UCI loop (what a GUI expects)
//! - `mythos bench`      → run a fixed perft benchmark and exit
//! - `mythos perft <N>`  → perft the start position to depth N and exit

use std::time::Instant;

use mythos::perft::perft;
use mythos::position::Position;
use mythos::uci::uci_loop;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("bench") => bench(),
        Some("perft") => {
            let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            let mut pos = Position::startpos();
            let start = Instant::now();
            let nodes = perft(&mut pos, depth);
            let secs = start.elapsed().as_secs_f64();
            println!("perft({depth}) = {nodes} in {secs:.3}s ({:.0} nps)", nodes as f64 / secs);
        }
        _ => {
            // No args: behave like a normal UCI engine.
            uci_loop();
        }
    }
}

/// A fixed perft benchmark — a quick way to sanity-check speed after a change.
fn bench() {
    let mut pos = Position::startpos();
    let start = Instant::now();
    let nodes = perft(&mut pos, 5);
    let secs = start.elapsed().as_secs_f64();
    println!("bench: perft(5) = {nodes} nodes in {secs:.3}s ({:.0} nps)", nodes as f64 / secs);
}
