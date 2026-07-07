//! Zobrist hashing keys.
//!
//! Zobrist hashing gives every board position a near-unique 64-bit fingerprint.
//! The trick: assign a random 64-bit key to every independent feature of a
//! position — "there is a white knight on f3", "Black may castle queenside",
//! "the en-passant file is d", "it is Black to move" — and XOR the keys of all
//! the features that are present into a single accumulator.
//!
//! Because XOR is its own inverse (`k ^ x ^ x == k`), the hash can be updated
//! *incrementally*: to move a knight off f3 you just XOR out its old key and
//! XOR in the new one, instead of rehashing the whole board. That property is
//! what makes the transposition table and repetition detection cheap.
//!
//! The keys must be identical on every run (so a saved TT or an opening book
//! stays valid), so they are generated once from a fixed-seed PRNG rather than
//! from any source of real randomness.

use std::sync::LazyLock;

use crate::types::{Piece, Square};

// ---------------------------------------------------------------------------
// A tiny deterministic PRNG.
//
// We use splitmix64: a very small, high-quality generator with no bad seeds.
// Seeded with a fixed constant it produces the same stream on every machine,
// which is exactly what we want for reproducible Zobrist keys.
// ---------------------------------------------------------------------------

/// splitmix64 state — just a 64-bit counter that we scramble on each draw.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    const fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Produce the next pseudo-random 64-bit value and advance the state.
    #[inline]
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The fixed seed. Change this only if you deliberately want to invalidate
/// every previously-saved hash (e.g. a persisted transposition table).
const ZOBRIST_SEED: u64 = 0x4D59_5448_4F53_5A42; // "MYTHOSZB" in ASCII.

// ---------------------------------------------------------------------------
// The key table.
// ---------------------------------------------------------------------------

/// All the random keys that make up a position hash.
pub struct Zobrist {
    /// One key per (piece-index 0..12, square 0..64). The piece index is the
    /// dense `Piece::index()` (white pieces 0..5, black pieces 6..11).
    pub psq: [[u64; 64]; 12],
    /// One key per 4-bit castling-rights mask (0..16).
    pub castling: [u64; 16],
    /// One key per en-passant file (0..8).
    pub en_passant: [u64; 8],
    /// XOR'd into the hash when it is Black to move.
    pub side: u64,
}

impl Zobrist {
    /// Generate the full key table from the fixed seed.
    ///
    /// This is deterministic: every call produces byte-identical tables, so the
    /// values are stable across runs and machines.
    pub fn new() -> Self {
        let mut rng = SplitMix64::new(ZOBRIST_SEED);

        let mut psq = [[0u64; 64]; 12];
        for piece_keys in psq.iter_mut() {
            for key in piece_keys.iter_mut() {
                *key = rng.next();
            }
        }

        let mut castling = [0u64; 16];
        for key in castling.iter_mut() {
            *key = rng.next();
        }

        let mut en_passant = [0u64; 8];
        for key in en_passant.iter_mut() {
            *key = rng.next();
        }

        let side = rng.next();

        Zobrist {
            psq,
            castling,
            en_passant,
            side,
        }
    }

    /// The key for a given piece standing on a given square.
    #[inline]
    pub fn piece(&self, piece: Piece, sq: Square) -> u64 {
        self.psq[piece.index()][sq.index()]
    }

    /// The key for a 4-bit castling-rights mask (0..16).
    #[inline]
    pub fn castle(&self, rights: u8) -> u64 {
        self.castling[(rights & 0xF) as usize]
    }

    /// The key for an en-passant file (0..8).
    #[inline]
    pub fn ep(&self, file: u8) -> u64 {
        self.en_passant[(file & 7) as usize]
    }

    /// The side-to-move key, XOR'd in when it is Black to move.
    #[inline]
    pub fn side(&self) -> u64 {
        self.side
    }
}

impl Default for Zobrist {
    fn default() -> Self {
        Self::new()
    }
}

/// The single shared key table, built lazily on first use.
///
/// Access is cheap: after the one-time initialization this is just a pointer
/// dereference, so hot paths can read `ZOBRIST.piece(..)` freely.
pub static ZOBRIST: LazyLock<Zobrist> = LazyLock::new(Zobrist::new);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Color, PieceType};
    use std::collections::HashSet;

    /// The keys must be reproducible: two independently-built tables must be
    /// byte-for-byte identical, and so must two reads of the shared static.
    #[test]
    fn determinism() {
        let a = Zobrist::new();
        let b = Zobrist::new();
        assert_eq!(a.psq, b.psq);
        assert_eq!(a.castling, b.castling);
        assert_eq!(a.en_passant, b.en_passant);
        assert_eq!(a.side, b.side);

        // The shared static must match a freshly-built table too.
        assert_eq!(ZOBRIST.psq, a.psq);
        assert_eq!(ZOBRIST.side, a.side);
    }

    /// Every piece-square key must be distinct and non-zero. Collisions here
    /// would defeat the whole point of hashing, and a zero key would silently
    /// vanish under XOR.
    #[test]
    fn psq_keys_distinct_and_nonzero() {
        let z = Zobrist::new();
        let mut seen = HashSet::new();
        for piece_keys in z.psq.iter() {
            for &key in piece_keys.iter() {
                assert_ne!(key, 0, "a psq key was zero");
                seen.insert(key);
            }
        }
        // 12 pieces * 64 squares = 768 keys, all unique.
        assert_eq!(seen.len(), 12 * 64);
    }

    /// The whole key set (psq + castling + en-passant + side) should also be
    /// free of collisions.
    #[test]
    fn all_keys_distinct() {
        let z = Zobrist::new();
        let mut seen = HashSet::new();
        for piece_keys in z.psq.iter() {
            for &key in piece_keys.iter() {
                seen.insert(key);
            }
        }
        for &key in z.castling.iter() {
            seen.insert(key);
        }
        for &key in z.en_passant.iter() {
            seen.insert(key);
        }
        seen.insert(z.side);

        let total = 12 * 64 + 16 + 8 + 1;
        assert_eq!(seen.len(), total, "found a collision across all key tables");
    }

    /// The core invariant that makes incremental hashing work: XORing the same
    /// key in twice cancels out, returning the hash to its original value.
    #[test]
    fn xor_cancellation() {
        let z = Zobrist::new();
        let piece = Piece::new(Color::White, PieceType::Knight);
        let key = z.piece(piece, Square::F3);

        // Starting from some arbitrary running hash...
        let original: u64 = 0x0123_4567_89AB_CDEF;
        let with_piece = original ^ key;
        assert_ne!(with_piece, original, "applying a key must change the hash");

        // ...applying the same key again undoes it exactly.
        let back = with_piece ^ key;
        assert_eq!(back, original);

        // Same story for the other key kinds.
        let h = original ^ z.side() ^ z.side();
        assert_eq!(h, original);
        let h = original ^ z.castle(0b1011) ^ z.castle(0b1011);
        assert_eq!(h, original);
        let h = original ^ z.ep(3) ^ z.ep(3);
        assert_eq!(h, original);
    }
}
