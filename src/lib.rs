//! # Mythos 🦈
//!
//! A UCI chess engine written in Rust, built by studying the Stockfish C++
//! engine and reimplementing its ideas idiomatically.
//!
//! The crate is split into small modules that mirror the classic chess-engine
//! subsystems. Right now (Phase 0) only the foundation is in place:
//!
//! - [`types`]    — Color, PieceType, Piece, Square, Direction, Move
//! - [`bitboard`] — the 64-bit board-set primitive everything is built on
//! - [`attacks`]  — knight/king/pawn tables + magic-bitboard sliders
//! - [`zobrist`]  — hashing keys for positions (TT + repetition)
//! - [`position`] — board state, FEN parsing, make/undo move
//! - [`movegen`]  — legal move generation ([`perft`]-validated)
//! - [`perft`]    — move-generation correctness counter
//!
//! Phase 0 (foundation) is complete and `perft`-verified. Coming next
//! (Phase 1): search, evaluation, and the UCI protocol loop.

pub mod attacks;
pub mod bitboard;
pub mod eval;
pub mod movegen;
pub mod nnue;
pub mod perft;
pub mod position;
pub mod search;
pub mod see;
pub mod tt;
pub mod types;
pub mod uci;
pub mod zobrist;

// Re-export the most-used types at the crate root for convenience.
pub use bitboard::Bitboard;
pub use eval::evaluate;
pub use movegen::{MoveList, generate_legal};
pub use position::Position;
pub use search::{SearchLimits, SearchResult, Searcher};
pub use see::{see, see_ge, see_value};
pub use tt::{Bound, TranspositionTable, TtEntry};
pub use types::{Color, Direction, Move, MoveType, Piece, PieceType, Square};
pub use zobrist::{ZOBRIST, Zobrist};
