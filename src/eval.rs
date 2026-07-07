//! Evaluation: score a position in centipawns, from the side-to-move's view.
//!
//! This is a **hand-crafted evaluation** (HCE) — no neural net. It implements
//! the well-known, public-domain **PeSTO** scheme (Pawn, King, and piece-Square
//! Tables, Only — the "Rofchade" tables popularised by Ronald Friederich). The
//! idea is simple and fast, which is exactly what we want on low-end hardware:
//!
//!  * Every piece is worth a base **material** value.
//!  * Every piece also gets a **piece-square** bonus/penalty for *where* it
//!    stands (a knight in the centre is worth more than one in the corner).
//!  * Both of those come in two flavours — a **midgame** value and an
//!    **endgame** value — because good squares change as pieces come off. A king
//!    hides in the corner in the middlegame but marches to the centre in the
//!    endgame, so its two tables are almost mirror images.
//!
//! We blend the two with a **tapered eval**: a `phase` number counts how much
//! material is still on the board (24 = full, 0 = bare kings) and we linearly
//! interpolate between the midgame and endgame scores. This gives a single,
//! smoothly-changing number with no ugly discontinuity when a queen trades off.
//!
//! ### Positional terms on top of PeSTO
//!
//! Material + piece-square tables alone play positionally naive chess, so on top
//! of that base we add the classic hand-crafted terms every strong HCE engine
//! carries — **mobility**, **king safety** (pawn shield + attackers on the king
//! ring), **passed pawns**, **pawn-structure** penalties (doubled / isolated /
//! backward), the **bishop pair**, **rooks on open files**, and a small **tempo**
//! bonus for the side to move. Every one of them is *tapered* (a midgame value
//! and an endgame value blended by the same phase) and every one is accumulated
//! **White-positive** — computed identically for both colors with only the board
//! geometry mirrored — so the whole eval stays perfectly color-symmetric.
//!
//! ### Table orientation (the classic footgun)
//!
//! The published PeSTO tables are written the way you *read* a board — rank 8 at
//! the top, a-file on the left. Mythos's [`Square`] index is the opposite:
//! `0 = a1`, `63 = h8`, going *up* the board. To avoid confusion we store every
//! table already flipped into Mythos's a1-first order, so a White piece indexes
//! its table directly by `square.index()`. A Black piece uses the *same* White
//! tables but mirrored vertically via [`Square::flip_rank`] — a Black knight on
//! c6 should score like a White knight on c3. The [`tests`] module asserts full
//! color symmetry to catch any orientation slip.

use crate::attacks::{
    bishop_attacks, king_attacks, knight_attacks, queen_attacks, rook_attacks,
};
use crate::bitboard::{Bitboard, FILE_A_BB, FILE_H_BB};
use crate::position::Position;
use crate::types::{Color, Direction, PieceType, Square};

// ---------------------------------------------------------------------------
// Material values (centipawns), one pair per piece type: [midgame, endgame].
//
// These are the standard PeSTO material values. The king's "value" is 0 here:
// both kings are always on the board, so it cancels out and never affects the
// score. (Search handles the king via checkmate, not material.)
// ---------------------------------------------------------------------------

/// Midgame material value of each piece type, indexed by `PieceType::index()`.
const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
/// Endgame material value of each piece type, indexed by `PieceType::index()`.
const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];

// ---------------------------------------------------------------------------
// Game-phase weights.
//
// Phase counts remaining non-pawn material: knight/bishop = 1, rook = 2,
// queen = 4. A full starting army (minus pawns and kings) is
// 4*1 + 4*1 + 4*2 + 2*4 = 24, so `MAX_PHASE = 24`. As pieces come off the phase
// falls toward 0 (a bare-king endgame).
// ---------------------------------------------------------------------------

/// Phase contribution of each piece type, indexed by `PieceType::index()`.
const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];
/// The phase value of the full starting position (excluding pawns/kings).
const MAX_PHASE: i32 = 24;

// ---------------------------------------------------------------------------
// Piece-square tables — midgame and endgame, one 64-entry table per piece type.
//
// IMPORTANT: these are stored in Mythos's a1-first order. Index 0 is a1, index 7
// is h1, index 56 is a8. So each table below reads rank 1 first (bottom row of
// the array) up to rank 8 (top row) — which is why, laid out as text, they look
// vertically flipped compared to the tables you'll find published online.
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const MG_PAWN: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    -35,  -1, -20, -23, -15,  24,  38, -22,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -14,  13,   6,  21,  23,  12,  17, -23,
     -6,   7,  26,  31,  65,  56,  25, -20,
     98, 134,  61,  95,  68, 126,  34, -11,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const EG_PAWN: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     13,   8,   8,  10,  13,   0,   2,  -7,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     32,  24,  13,   5,  -2,   4,  17,  17,
     94, 100,  85,  67,  56,  53,  82,  84,
    178, 173, 158, 134, 147, 132, 165, 187,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const MG_KNIGHT: [i32; 64] = [
   -105, -21, -58, -33, -17, -28, -19, -23,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -13,   4,  16,  13,  28,  19,  21,  -8,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -47,  60,  37,  65,  84, 129,  73,  44,
    -73, -41,  72,  36,  23,  62,   7, -17,
   -167, -89, -34, -49,  61, -97, -15, -107,
];

#[rustfmt::skip]
const EG_KNIGHT: [i32; 64] = [
    -29, -51, -23, -15, -22, -18, -50, -64,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -58, -38, -13, -28, -31, -27, -63, -99,
];

#[rustfmt::skip]
const MG_BISHOP: [i32; 64] = [
    -33,  -3, -14, -21, -13, -12, -39, -21,
      4,  15,  16,   0,   7,  21,  33,   1,
      0,  15,  15,  15,  14,  27,  18,  10,
     -6,  13,  13,  26,  34,  12,  10,   4,
     -4,   5,  19,  50,  37,  37,   7,  -2,
    -16,  37,  43,  40,  35,  50,  37,  -2,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -29,   4, -82, -37, -25, -42,   7,  -8,
];

#[rustfmt::skip]
const EG_BISHOP: [i32; 64] = [
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
     -3,   9,  12,   9,  14,  10,   3,   2,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
];

#[rustfmt::skip]
const MG_ROOK: [i32; 64] = [
    -19, -13,   1,  17,  16,   7, -37, -26,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -24, -11,   7,  26,  24,  35,  -8, -20,
     -5,  19,  26,  36,  17,  45,  61,  16,
     27,  32,  58,  62,  80,  67,  26,  44,
     32,  42,  32,  51,  63,   9,  31,  43,
];

#[rustfmt::skip]
const EG_ROOK: [i32; 64] = [
     -9,   2,   3,  -1,  -5, -13,   4, -20,
    -12,  -8,  -7,  -6,  -6,  -7,   0,   3,
      6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
     11,  13,  13,  11,  -3,   3,   8,   3,
     13,  10,  18,  15,  12,  12,   8,   5,
];

#[rustfmt::skip]
const MG_QUEEN: [i32; 64] = [
     -1, -18,  -9,  10, -15, -25, -31, -50,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -28,   0,  29,  12,  59,  44,  43,  45,
];

#[rustfmt::skip]
const EG_QUEEN: [i32; 64] = [
    -33, -28, -22, -43,  -5, -32, -20, -41,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -18,  28,  19,  47,  31,  34,  39,  23,
      3,  22,  24,  45,  57,  40,  57,  36,
    -20,   6,   9,  49,  47,  35,  19,   9,
    -17,  20,  32,  41,  58,  25,  30,   0,
     -9,  22,  22,  27,  27,  19,  10,  20,
];

#[rustfmt::skip]
const MG_KING: [i32; 64] = [
    -15,  36,  12, -54,   8, -28,  24,  14,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -14, -14, -22, -46, -44, -30, -15, -27,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -17, -20, -12, -27, -30, -25, -14, -36,
     -9,  24,   2, -16, -20,   6,  22, -22,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -65,  23,  16, -15, -56, -34,   2,  13,
];

#[rustfmt::skip]
const EG_KING: [i32; 64] = [
    -53, -34, -21, -11, -28, -14, -24, -43,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -18,  -4,  21,  24,  27,  23,   9, -11,
     -8,  22,  24,  27,  26,  33,  26,   3,
     10,  17,  23,  15,  20,  45,  44,  13,
    -12,  17,  14,  17,  17,  38,  23,  11,
    -74, -35, -18, -18, -11,  15,   4, -17,
];

/// Midgame PST for every piece type, indexed by `PieceType::index()`.
const MG_PST: [&[i32; 64]; 6] =
    [&MG_PAWN, &MG_KNIGHT, &MG_BISHOP, &MG_ROOK, &MG_QUEEN, &MG_KING];
/// Endgame PST for every piece type, indexed by `PieceType::index()`.
const EG_PST: [&[i32; 64]; 6] =
    [&EG_PAWN, &EG_KNIGHT, &EG_BISHOP, &EG_ROOK, &EG_QUEEN, &EG_KING];

// ---------------------------------------------------------------------------
// Positional-term weights (all centipawns, all tapered [midgame, endgame]).
//
// These are conservative values in the range used by strong open-source HCE
// engines. Each table is indexed by "how many safe squares does this piece
// attack", so a piece with more room to move scores more.
// ---------------------------------------------------------------------------

/// Midgame mobility bonus for a knight, indexed by its number of safe moves (0..8).
#[rustfmt::skip]
const MG_KNIGHT_MOB: [i32; 9] = [-30, -20, -10, 0, 8, 14, 18, 22, 26];
/// Endgame mobility bonus for a knight, indexed by its number of safe moves (0..8).
#[rustfmt::skip]
const EG_KNIGHT_MOB: [i32; 9] = [-30, -20, -10, 0, 6, 11, 15, 18, 21];

/// Midgame mobility bonus for a bishop, indexed by its number of safe moves (0..13).
#[rustfmt::skip]
const MG_BISHOP_MOB: [i32; 14] =
    [-25, -12, 0, 8, 14, 19, 23, 26, 29, 31, 33, 35, 37, 39];
/// Endgame mobility bonus for a bishop, indexed by its number of safe moves (0..13).
#[rustfmt::skip]
const EG_BISHOP_MOB: [i32; 14] =
    [-25, -12, 0, 7, 13, 18, 22, 25, 28, 30, 32, 34, 36, 38];

/// Midgame mobility bonus for a rook, indexed by its number of safe moves (0..14).
#[rustfmt::skip]
const MG_ROOK_MOB: [i32; 15] =
    [-20, -10, 0, 4, 7, 10, 13, 16, 18, 20, 22, 23, 24, 25, 26];
/// Endgame mobility bonus for a rook, indexed by its number of safe moves (0..14).
#[rustfmt::skip]
const EG_ROOK_MOB: [i32; 15] =
    [-30, -15, 0, 8, 15, 21, 26, 30, 34, 37, 40, 42, 44, 45, 46];

/// Midgame mobility bonus for a queen, indexed by its number of safe moves (0..27).
#[rustfmt::skip]
const MG_QUEEN_MOB: [i32; 28] = [
    -15, -10, -5, 0, 3, 6, 8, 10, 12, 13, 14, 15, 16, 17,
     18,  19, 20, 21, 22, 23, 24, 25, 25, 26, 26, 27, 27, 28,
];
/// Endgame mobility bonus for a queen, indexed by its number of safe moves (0..27).
#[rustfmt::skip]
const EG_QUEEN_MOB: [i32; 28] = [
    -20, -13, -7, 0, 5, 9, 13, 16, 19, 22, 24, 26, 28, 30,
     32,  34, 35, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48,
];

/// Endgame passed-pawn bonus by the pawn's relative rank (0..7, from its own
/// side). Bigger the further advanced; rank 0/7 are unreachable for a pawn.
#[rustfmt::skip]
const EG_PASSED_PAWN: [i32; 8] = [0, 10, 17, 15, 35, 75, 175, 0];
/// Midgame passed-pawn bonus by relative rank — smaller than the endgame value,
/// because a passer matters far more once the pieces are off.
#[rustfmt::skip]
const MG_PASSED_PAWN: [i32; 8] = [0, 5, 8, 10, 20, 40, 65, 0];

/// Doubled-pawn penalty [midgame, endgame] — two or more of your pawns on one file.
const DOUBLED_PAWN: [i32; 2] = [-10, -20];
/// Isolated-pawn penalty [midgame, endgame] — no friendly pawn on either adjacent file.
const ISOLATED_PAWN: [i32; 2] = [-5, -15];
/// Backward-pawn penalty [midgame, endgame] — a pawn that cannot advance and has
/// no friendly pawn beside or behind it to support the push.
const BACKWARD_PAWN: [i32; 2] = [-8, -10];

/// Bishop-pair bonus [midgame, endgame] — holding both bishops.
const BISHOP_PAIR: [i32; 2] = [25, 40];

/// Rook on a fully open file (no pawns of either color) [midgame, endgame].
const ROOK_OPEN_FILE: [i32; 2] = [25, 10];
/// Rook on a semi-open file (no *friendly* pawns) [midgame, endgame].
const ROOK_SEMI_OPEN_FILE: [i32; 2] = [12, 6];

/// Bonus per friendly pawn shielding the king (midgame only — an endgame king
/// wants to march out, not hide).
const KING_SHIELD: i32 = 9;

/// Attacker weight per enemy piece type touching the king ring, indexed by
/// `PieceType::index()`. Pawns and kings do not contribute to the attack count.
#[rustfmt::skip]
const KING_ATTACK_WEIGHT: [i32; 6] = [0, 2, 2, 3, 5, 0];

/// Nonlinear king-danger table: index by the summed attacker weight (clamped to
/// its length) to get the midgame penalty. Rises steeply so that several pieces
/// swarming the king is punished far more than the sum of individual attackers,
/// then flattens so a single blunder never dwarfs the rest of the eval.
#[rustfmt::skip]
const KING_SAFETY_TABLE: [i32; 30] = [
      0,   0,   4,   8,  14,  22,  32,  44,  58,  74,
     92, 112, 134, 158, 184, 212, 242, 274, 308, 344,
    382, 400, 416, 430, 442, 452, 460, 466, 470, 472,
];

/// Tempo bonus (midgame) for having the move.
const TEMPO_MG: i32 = 10;
/// Tempo bonus (endgame) for having the move.
const TEMPO_EG: i32 = 5;

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

/// The midgame material value of a piece type, in centipawns.
///
/// Handy for move ordering and static exchange evaluation (SEE) later on, which
/// only need a single "how much is this piece worth" number.
#[inline]
pub fn piece_value(pt: PieceType) -> i32 {
    MG_VALUE[pt.index()]
}

/// Everything about one color's pawns that the positional terms reuse, computed
/// once per evaluation so no term walks the pawns twice.
struct PawnInfo {
    /// This color's pawns.
    pawns: Bitboard,
    /// The squares this color's pawns attack (both diagonals, unioned).
    attacks: Bitboard,
    /// Every square *in front of and diagonally in front of* this color's pawns,
    /// all the way to the far edge — the "front span". A pawn is passed iff no
    /// enemy pawn lies in the enemy front span reflected onto it; we build it here
    /// so the passed-pawn test is a single intersection.
    front_span: Bitboard,
}

/// The union of a file mask's two neighbours — every square on the files
/// immediately left and right of the given one. Used for the isolated / backward
/// "adjacent file" pawn tests.
#[inline]
fn adjacent_files(files: Bitboard) -> Bitboard {
    let west = Bitboard((files.0 & !FILE_A_BB) >> 1);
    let east = Bitboard((files.0 & !FILE_H_BB) << 1);
    west | east
}

impl PawnInfo {
    /// Gather one color's pawn geometry. `forward` is the direction that color's
    /// pawns advance (North for White, South for Black), so the whole thing is
    /// written once and mirrored purely by that direction.
    fn new(pawns: Bitboard, color: Color) -> PawnInfo {
        let (fwd, west_cap, east_cap) = match color {
            Color::White => (Direction::North, Direction::NorthWest, Direction::NorthEast),
            Color::Black => (Direction::South, Direction::SouthWest, Direction::SouthEast),
        };
        let attacks = pawns.shift(west_cap) | pawns.shift(east_cap);

        // Repeatedly push the pawns forward and OR them in to sweep out every
        // square ahead of them, then splay one file left/right for the diagonal
        // reach: that is the full region an enemy pawn must avoid to be passed.
        let mut span = Bitboard::EMPTY;
        let mut walk = pawns.shift(fwd);
        while walk.any() {
            span |= walk;
            walk = walk.shift(fwd);
        }
        let front_span = span
            | Bitboard((span.0 & !FILE_A_BB) >> 1)
            | Bitboard((span.0 & !FILE_H_BB) << 1);

        PawnInfo {
            pawns,
            attacks,
            front_span,
        }
    }
}

/// Evaluate `pos` in centipawns, **from the side-to-move's perspective**.
///
/// A positive score means the side to move is better; a negative score means it
/// is worse. This is the negamax convention the search expects — so if it is
/// Black to move we negate the White-oriented internal score.
///
/// The score is a tapered blend of a midgame and an endgame evaluation, each the
/// sum of a material + piece-square base and the positional terms (mobility, king
/// safety, passed pawns, pawn structure, bishop pair, rook files), all summed for
/// White minus the same for Black. A final tempo bonus rewards the side to move.
pub fn evaluate(pos: &Position) -> i32 {
    // White-positive running totals for each game phase.
    let mut mg = 0i32;
    let mut eg = 0i32;
    // How much material is left, as a phase 0..=24.
    let mut phase = 0i32;

    for pt in PieceType::ALL {
        let pi = pt.index();
        let mg_val = MG_VALUE[pi];
        let eg_val = EG_VALUE[pi];
        let mg_tbl = MG_PST[pi];
        let eg_tbl = EG_PST[pi];
        let inc = PHASE_INC[pi];

        // White pieces score by their own square; add to the White-positive sum.
        for sq in pos.pieces_cp(Color::White, pt) {
            let i = sq.index();
            mg += mg_val + mg_tbl[i];
            eg += eg_val + eg_tbl[i];
            phase += inc;
        }
        // Black pieces score by the vertically-mirrored square; subtract.
        for sq in pos.pieces_cp(Color::Black, pt) {
            let i = sq.flip_rank().index();
            mg -= mg_val + mg_tbl[i];
            eg -= eg_val + eg_tbl[i];
            phase += inc;
        }
    }

    // --- Positional terms, all accumulated White-positive. ------------------
    // Precompute each side's pawn geometry once (pawn attacks and front spans).
    let white_pawns = PawnInfo::new(pos.pieces_cp(Color::White, PieceType::Pawn), Color::White);
    let black_pawns = PawnInfo::new(pos.pieces_cp(Color::Black, PieceType::Pawn), Color::Black);

    // Each term returns a (mg, eg) pair already relative to White (its White score
    // minus its Black score), so they simply add into the running totals.
    let (m_mg, m_eg) = mobility(pos, &white_pawns, &black_pawns);
    mg += m_mg;
    eg += m_eg;

    let (k_mg, k_eg) = king_safety(pos, &white_pawns, &black_pawns);
    mg += k_mg;
    eg += k_eg;

    let (p_mg, p_eg) = pawn_structure(&white_pawns, &black_pawns);
    mg += p_mg;
    eg += p_eg;

    let (b_mg, b_eg) = bishop_pair(pos);
    mg += b_mg;
    eg += b_eg;

    let (r_mg, r_eg) = rook_files(pos, &white_pawns, &black_pawns);
    mg += r_mg;
    eg += r_eg;

    // Tapered interpolation. Clamp the phase in case of odd material (e.g. after
    // several promotions there can be *more* than 24 worth of pieces on board).
    let phase = phase.min(MAX_PHASE);
    let mut score = (mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE;

    // Tempo: a small bonus for the side to move, applied in its own frame. It is
    // color-symmetric because a mirrored position also flips the side to move.
    let tempo = (TEMPO_MG * phase + TEMPO_EG * (MAX_PHASE - phase)) / MAX_PHASE;
    score += match pos.side_to_move() {
        Color::White => tempo,
        Color::Black => -tempo,
    };

    // Flip into the side-to-move's frame (negamax convention).
    match pos.side_to_move() {
        Color::White => score,
        Color::Black => -score,
    }
}

// ---------------------------------------------------------------------------
// Positional terms. Each returns a White-positive `(mg, eg)` pair: it computes
// the term for White and for Black identically (only the board geometry differs)
// and returns White minus Black, which keeps the whole eval color-symmetric.
// ---------------------------------------------------------------------------

/// **Mobility** — reward each knight/bishop/rook/queen for the number of *safe*
/// squares it attacks: squares not occupied by a friendly piece and not defended
/// by an enemy pawn. More room to manoeuvre is better, on a gently rising table.
fn mobility(pos: &Position, white: &PawnInfo, black: &PawnInfo) -> (i32, i32) {
    let occ = pos.occupied();
    let (mut mg, mut eg) = (0, 0);
    // White gains its own mobility (minus enemy pawn attacks); Black subtracts.
    let (wm, we) = mobility_for(pos, Color::White, occ, black.attacks);
    let (bm, be) = mobility_for(pos, Color::Black, occ, white.attacks);
    mg += wm - bm;
    eg += we - be;
    (mg, eg)
}

/// One color's raw mobility total. `enemy_pawn_attacks` are the squares that side
/// must avoid; own pieces block their own moves.
fn mobility_for(
    pos: &Position,
    color: Color,
    occ: Bitboard,
    enemy_pawn_attacks: Bitboard,
) -> (i32, i32) {
    // Legal "landing" squares: anything not our own and not attacked by a pawn.
    let allowed = !pos.pieces(color) & !enemy_pawn_attacks;
    let (mut mg, mut eg) = (0, 0);

    for sq in pos.pieces_cp(color, PieceType::Knight) {
        let n = (knight_attacks(sq) & allowed).count() as usize;
        mg += MG_KNIGHT_MOB[n];
        eg += EG_KNIGHT_MOB[n];
    }
    for sq in pos.pieces_cp(color, PieceType::Bishop) {
        let n = (bishop_attacks(sq, occ) & allowed).count() as usize;
        mg += MG_BISHOP_MOB[n];
        eg += EG_BISHOP_MOB[n];
    }
    for sq in pos.pieces_cp(color, PieceType::Rook) {
        let n = (rook_attacks(sq, occ) & allowed).count() as usize;
        mg += MG_ROOK_MOB[n];
        eg += EG_ROOK_MOB[n];
    }
    for sq in pos.pieces_cp(color, PieceType::Queen) {
        let n = (queen_attacks(sq, occ) & allowed).count() as usize;
        mg += MG_QUEEN_MOB[n];
        eg += EG_QUEEN_MOB[n];
    }
    (mg, eg)
}

/// **King safety** — a midgame-weighted term with two parts per side: a *pawn
/// shield* bonus for friendly pawns standing on the king's file and its two
/// neighbours just in front of it, and a *king-attack* penalty that grows
/// nonlinearly with the number and weight of enemy pieces hitting the king ring.
fn king_safety(pos: &Position, white: &PawnInfo, black: &PawnInfo) -> (i32, i32) {
    // Each side's danger is computed identically, so White minus Black.
    let w = king_safety_for(pos, Color::White, white);
    let b = king_safety_for(pos, Color::Black, black);
    // King safety is a midgame concern; taper it right down in the endgame.
    (w - b, 0)
}

/// The midgame king-safety score for one color (shield bonus minus attack danger).
fn king_safety_for(pos: &Position, color: Color, own_pawns: &PawnInfo) -> i32 {
    let ksq = pos.king_square(color);
    let ring = king_attacks(ksq);

    // Pawn shield: friendly pawns on the three files around the king, on the two
    // ranks directly ahead of it. We take the king ring, keep only the squares in
    // front, extend one more rank forward, and count friendly pawns landing there.
    let fwd = match color {
        Color::White => Direction::North,
        Color::Black => Direction::South,
    };
    let shield_zone = (ring | ring.shift(fwd)) & !Bitboard::rank_bb(ksq);
    let shield = (shield_zone & own_pawns.pawns).count() as i32;

    // King-attack danger: sum the weight of every enemy knight/bishop/rook/queen
    // that attacks a square in the king ring, then look the total up in the
    // nonlinear safety table.
    let occ = pos.occupied();
    let enemy = !color;
    let mut danger = 0i32;
    for sq in pos.pieces_cp(enemy, PieceType::Knight) {
        if (knight_attacks(sq) & ring).any() {
            danger += KING_ATTACK_WEIGHT[PieceType::Knight.index()];
        }
    }
    for sq in pos.pieces_cp(enemy, PieceType::Bishop) {
        if (bishop_attacks(sq, occ) & ring).any() {
            danger += KING_ATTACK_WEIGHT[PieceType::Bishop.index()];
        }
    }
    for sq in pos.pieces_cp(enemy, PieceType::Rook) {
        if (rook_attacks(sq, occ) & ring).any() {
            danger += KING_ATTACK_WEIGHT[PieceType::Rook.index()];
        }
    }
    for sq in pos.pieces_cp(enemy, PieceType::Queen) {
        if (queen_attacks(sq, occ) & ring).any() {
            danger += KING_ATTACK_WEIGHT[PieceType::Queen.index()];
        }
    }
    let danger = (danger as usize).min(KING_SAFETY_TABLE.len() - 1);

    shield * KING_SHIELD - KING_SAFETY_TABLE[danger]
}

/// **Pawn structure** and **passed pawns** — walk each side's pawns once and
/// apply doubled / isolated / backward penalties and a passed-pawn bonus. Both
/// sides run the identical routine (mirrored by color), returning White minus
/// Black.
fn pawn_structure(white: &PawnInfo, black: &PawnInfo) -> (i32, i32) {
    let (wm, we) = pawn_structure_for(Color::White, white, black);
    let (bm, be) = pawn_structure_for(Color::Black, black, white);
    (wm - bm, we - be)
}

/// One color's pawn-structure score (doubled + isolated + backward + passed).
fn pawn_structure_for(color: Color, own: &PawnInfo, enemy: &PawnInfo) -> (i32, i32) {
    let (mut mg, mut eg) = (0, 0);

    // Doubled: any file carrying two or more of our pawns is penalised once per
    // extra pawn. We count per file so a tripled pawn is punished twice.
    for file in 0..8u8 {
        let file_mask = Bitboard(FILE_A_BB << file);
        let n = (own.pawns & file_mask).count() as i32;
        if n > 1 {
            mg += DOUBLED_PAWN[0] * (n - 1);
            eg += DOUBLED_PAWN[1] * (n - 1);
        }
    }

    // Isolated / backward / passed are per-pawn tests.
    for sq in own.pawns {
        // Isolated: no friendly pawn on either neighbouring file at all.
        let file_mask = Bitboard::file_bb(sq);
        let has_neighbour = (adjacent_files(file_mask) & own.pawns).any();
        if !has_neighbour {
            mg += ISOLATED_PAWN[0];
            eg += ISOLATED_PAWN[1];
        } else if is_backward(color, sq, own, enemy) {
            // Backward (only meaningful when it *has* neighbours to lag behind).
            mg += BACKWARD_PAWN[0];
            eg += BACKWARD_PAWN[1];
        }

        // Passed: no enemy pawn on this file or the two adjacent files anywhere
        // ahead of us. `enemy.front_span` already covers "every square an enemy
        // pawn stops us on"; if our square is not in it, we are a passer.
        if !enemy.front_span.contains(sq) {
            let rel_rank = relative_rank(color, sq) as usize;
            mg += MG_PASSED_PAWN[rel_rank];
            eg += EG_PASSED_PAWN[rel_rank];
        }
    }
    (mg, eg)
}

/// A pawn is **backward** if it cannot be defended by a friendly pawn (no friendly
/// pawn is level with or behind it on an adjacent file) and the square it would
/// advance to is attacked by an enemy pawn — so it is stuck and weak.
fn is_backward(color: Color, sq: Square, own: &PawnInfo, enemy: &PawnInfo) -> bool {
    let fwd = match color {
        Color::White => Direction::North,
        Color::Black => Direction::South,
    };
    // The square directly ahead of the pawn (its push target). If off-board it
    // cannot be backward in any meaningful way.
    let push = match sq.offset(fwd) {
        Some(s) => s,
        None => return false,
    };
    // Are there friendly pawns on adjacent files at or behind this pawn's rank
    // that could later catch up and defend it? Build the "behind and level" span
    // on the adjacent files and test for friendly pawns there.
    let adj = adjacent_files(Bitboard::file_bb(sq));
    let behind = behind_span(color, sq);
    let supported = (adj & (behind | Bitboard::rank_bb(sq)) & own.pawns).any();
    if supported {
        return false;
    }
    // Stuck: an enemy pawn guards the square in front, so it can never advance.
    enemy.attacks.contains(push)
}

/// Every square strictly *behind* `sq` from `color`'s point of view (lower ranks
/// for White, higher for Black), across the whole board.
#[inline]
fn behind_span(color: Color, sq: Square) -> Bitboard {
    let r = sq.rank() as u32;
    match color {
        // White: ranks below this one.
        Color::White => Bitboard((1u64 << (8 * r)) - 1),
        // Black: ranks above this one.
        Color::Black => Bitboard(!0u64 << (8 * (r + 1))),
    }
}

/// **Bishop pair** — a side holding both bishops gets a bonus, worth a touch more
/// in the endgame where the two bishops sweep open boards.
fn bishop_pair(pos: &Position) -> (i32, i32) {
    let w = pos.pieces_cp(Color::White, PieceType::Bishop).more_than_one();
    let b = pos.pieces_cp(Color::Black, PieceType::Bishop).more_than_one();
    let mg = (w as i32 - b as i32) * BISHOP_PAIR[0];
    let eg = (w as i32 - b as i32) * BISHOP_PAIR[1];
    (mg, eg)
}

/// **Rook on an open / semi-open file** — a rook on a file with no pawns at all
/// (open) or with no *friendly* pawns (semi-open) gets a bonus for the pressure
/// it exerts down the file.
fn rook_files(pos: &Position, white: &PawnInfo, black: &PawnInfo) -> (i32, i32) {
    let (wm, we) = rook_files_for(pos, Color::White, white, black);
    let (bm, be) = rook_files_for(pos, Color::Black, black, white);
    (wm - bm, we - be)
}

/// One color's rook-file bonus.
fn rook_files_for(
    pos: &Position,
    color: Color,
    own_pawns: &PawnInfo,
    enemy_pawns: &PawnInfo,
) -> (i32, i32) {
    let (mut mg, mut eg) = (0, 0);
    for sq in pos.pieces_cp(color, PieceType::Rook) {
        let file = Bitboard::file_bb(sq);
        let own_on_file = (file & own_pawns.pawns).any();
        let enemy_on_file = (file & enemy_pawns.pawns).any();
        if !own_on_file && !enemy_on_file {
            mg += ROOK_OPEN_FILE[0];
            eg += ROOK_OPEN_FILE[1];
        } else if !own_on_file {
            mg += ROOK_SEMI_OPEN_FILE[0];
            eg += ROOK_SEMI_OPEN_FILE[1];
        }
    }
    (mg, eg)
}

/// A square's rank counted from `color`'s own back rank (0) to the far edge (7),
/// so a White pawn on rank 5 and a Black pawn on rank 4 both read as relative
/// rank 4 — the geometry both passed-pawn tables index by.
#[inline]
fn relative_rank(color: Color, sq: Square) -> u8 {
    match color {
        Color::White => sq.rank(),
        Color::Black => 7 - sq.rank(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Vertically mirror a FEN's piece placement and swap piece colors, so the
    /// resulting position is the exact color-reflection of the input. A White
    /// knight on c3 becomes a Black knight on c6, etc. Castling rights and the
    /// side to move are swapped too. En-passant is dropped for simplicity (none
    /// of the symmetry test FENs use it).
    ///
    /// This lets us assert `evaluate(pos) == evaluate(mirror(pos))`, which only
    /// holds if the PST orientation is correct for both colors.
    fn mirror_fen(fen: &str) -> String {
        let mut parts = fen.split_whitespace();
        let placement = parts.next().unwrap();
        let side = parts.next().unwrap_or("w");
        let castling = parts.next().unwrap_or("-");

        // Reverse the rank order (rank 8 <-> rank 1) and swap the case of every
        // piece char (White <-> Black).
        let ranks: Vec<&str> = placement.split('/').collect();
        let mirrored_placement = ranks
            .iter()
            .rev()
            .map(|rank| {
                rank.chars()
                    .map(|c| {
                        if c.is_ascii_uppercase() {
                            c.to_ascii_lowercase()
                        } else if c.is_ascii_lowercase() {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        }
                    })
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("/");

        // Swap the side to move.
        let mirrored_side = match side {
            "w" => "b",
            _ => "w",
        };

        // Swap castling-right case (K<->k, Q<->q) so the mirror is exact.
        let mirrored_castling = if castling == "-" {
            "-".to_string()
        } else {
            castling
                .chars()
                .map(|c| {
                    if c.is_ascii_uppercase() {
                        c.to_ascii_lowercase()
                    } else {
                        c.to_ascii_uppercase()
                    }
                })
                .collect::<String>()
        };

        format!("{mirrored_placement} {mirrored_side} {mirrored_castling} - 0 1")
    }

    #[test]
    fn startpos_is_balanced() {
        // The start position is materially and structurally symmetric, so every
        // White-positive term cancels against its Black twin and the *positional*
        // part is exactly 0. What remains is the tempo bonus for the side to move
        // (White here), so the score reads as roughly +tempo rather than dead 0.
        // The tolerance is widened from 5 to 30cp purely to admit that tempo term;
        // it is not loosening a correctness invariant (color symmetry still holds
        // exactly — see `color_symmetry`).
        let score = evaluate(&Position::startpos());
        assert!(
            (0..=30).contains(&score),
            "start position should be ~+tempo (0..30), got {score}"
        );
    }

    #[test]
    fn up_a_queen_is_winning_for_the_mover() {
        // White has an extra queen (Black is missing its queen). Being up a queen
        // should read roughly +900 from the perspective of whoever owns it.

        // White to move, White has the extra queen -> clearly positive.
        let white_up =
            Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let sw = evaluate(&white_up);
        assert!(sw > 400, "White up a queen (White to move) should be clearly +, got {sw}");
        assert!(
            (750..=1150).contains(&sw),
            "White up a queen should be ~+900 (±150ish), got {sw}"
        );

        // Same material imbalance but Black to move: now the mover (Black) is the
        // one DOWN a queen, so the score must be clearly negative.
        let black_to_move =
            Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let sb = evaluate(&black_to_move);
        assert!(sb < -400, "Black to move while down a queen should be clearly -, got {sb}");

        // And when the side that HAS the extra queen is to move, the magnitude is
        // symmetric: a Black-up-a-queen position with Black to move ~ +900 too.
        let black_up =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1").unwrap();
        let sbu = evaluate(&black_up);
        assert!(
            (750..=1150).contains(&sbu),
            "Black up a queen (Black to move) should be ~+900, got {sbu}"
        );
    }

    #[test]
    fn color_symmetry() {
        // A handful of asymmetric middlegame positions. Each must evaluate to the
        // exact same number as its color-mirrored twin — this is the strongest
        // check that the PST orientation is right for both colors.
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",     // Italian-ish
            "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NP3/1BN1B3/PPP1QPPP/R4RK1 w - - 0 1",     // Sicilian mg
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                            // rook endgame
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            let mirror = Position::from_fen(&mirror_fen(fen)).unwrap();
            let a = evaluate(&pos);
            let b = evaluate(&mirror);
            assert_eq!(a, b, "color symmetry broken for {fen}: {a} vs {b}");
        }
    }

    #[test]
    fn piece_values_are_ordered() {
        assert!(piece_value(PieceType::Queen) > piece_value(PieceType::Rook));
        assert!(piece_value(PieceType::Rook) > piece_value(PieceType::Bishop));
        assert!(piece_value(PieceType::Bishop) >= piece_value(PieceType::Knight));
        assert!(piece_value(PieceType::Knight) > piece_value(PieceType::Pawn));
    }

    /// Evaluate a FEN from White's point of view (undo the side-to-move flip), so
    /// two crafted positions can be compared on a single, consistent scale.
    fn white_eval(fen: &str) -> i32 {
        let pos = Position::from_fen(fen).unwrap();
        match pos.side_to_move() {
            Color::White => evaluate(&pos),
            Color::Black => -evaluate(&pos),
        }
    }

    #[test]
    fn passed_pawn_helps_its_owner() {
        // Compare the passed-pawn term directly, in isolation from the rest of the
        // eval, so black's own pawns can't confound the sign. Both positions have a
        // White pawn on e5 and three black pawns; in the first the black pawns are
        // on a7/b7/c7 (nowhere near the e-file) so White's e-pawn is a clean passer,
        // in the second they sit on d7/e7/f7 and blockade it.
        let passer = {
            let pos = Position::from_fen("4k3/ppp5/8/4P3/8/8/8/4K3 w - - 0 1").unwrap();
            let wp = PawnInfo::new(pos.pieces_cp(Color::White, PieceType::Pawn), Color::White);
            let bp = PawnInfo::new(pos.pieces_cp(Color::Black, PieceType::Pawn), Color::Black);
            pawn_structure_for(Color::White, &wp, &bp)
        };
        let blocked = {
            let pos = Position::from_fen("4k3/3ppp2/8/4P3/8/8/8/4K3 w - - 0 1").unwrap();
            let wp = PawnInfo::new(pos.pieces_cp(Color::White, PieceType::Pawn), Color::White);
            let bp = PawnInfo::new(pos.pieces_cp(Color::Black, PieceType::Pawn), Color::Black);
            pawn_structure_for(Color::White, &wp, &bp)
        };
        // The endgame pawn-structure score (index 1) must be strictly higher when
        // the pawn is a passer.
        assert!(
            passer.1 > blocked.1,
            "a passed e-pawn (eg {}) should beat a blockaded one (eg {})",
            passer.1,
            blocked.1
        );
        assert!(
            passer.0 > blocked.0,
            "a passed e-pawn (mg {}) should beat a blockaded one (mg {})",
            passer.0,
            blocked.0
        );
    }

    #[test]
    fn bishop_pair_is_a_bonus() {
        // Same material count (two minor pieces each) but one side has two bishops
        // and the other a bishop + knight. The bishop pair must score higher.
        let with_pair = white_eval("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1");
        let without_pair = white_eval("4k3/8/8/8/8/8/8/2B1KN2 w - - 0 1");
        assert!(
            with_pair > without_pair,
            "two bishops ({with_pair}) should beat bishop+knight ({without_pair})"
        );
    }

    #[test]
    fn exposed_king_scores_worse() {
        // The same White army, but in the first the king is tucked behind a healthy
        // f2/g2/h2 pawn shield while enemy pieces bear down on it; in the second the
        // shield pawns are gone, exposing the king. The exposed king must be worse.
        //
        // Black keeps a queen + rook aimed at the White king in both; only the
        // White shield pawns differ, isolating the king-safety term.
        let sheltered = white_eval("3rr1k1/8/8/8/8/6q1/5PPP/6K1 w - - 0 1");
        let exposed = white_eval("3rr1k1/8/8/8/8/6q1/8/6K1 w - - 0 1");
        assert!(
            sheltered > exposed,
            "a sheltered king ({sheltered}) should score better than an exposed one \
             ({exposed})"
        );
    }

    #[test]
    fn rook_on_open_file_is_better() {
        // Identical material and pawn structure (seven pawns, the d-file empty) in
        // both. Only the rook's file differs: on d1 it sits on the open d-file; on
        // c1 it is boxed behind its own c2 pawn. The open file must score better.
        let open = white_eval("4k3/8/8/8/8/8/PPP1PPPP/3RK3 w - - 0 1");
        let closed = white_eval("4k3/8/8/8/8/8/PPP1PPPP/2R1K3 w - - 0 1");
        assert!(
            open > closed,
            "a rook on the open d-file ({open}) should beat one behind its pawn \
             ({closed})"
        );
    }

    #[test]
    fn mobility_rewards_active_pieces() {
        // A knight in the centre (d4) commands far more squares than one stuck in
        // the corner (a1), so the central knight must score better for White. Both
        // positions are otherwise identical bare-king setups.
        let central = white_eval("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1");
        let cornered = white_eval("4k3/8/8/8/8/8/8/N3K3 w - - 0 1");
        assert!(
            central > cornered,
            "a central knight ({central}) should beat a cornered one ({cornered})"
        );
    }

    #[test]
    fn positional_terms_stay_color_symmetric() {
        // Extra positions chosen to exercise every new term (passed pawns, king
        // safety, rook files, mobility, pawn structure) — each must still equal its
        // color-mirrored twin exactly.
        let fens = [
            "4k3/pp4pp/8/3P4/8/8/PP4PP/4K3 w - - 0 1",        // passed d-pawn
            "3rr1k1/5ppp/8/8/8/6q1/5PPP/3RR1K1 w - - 0 1",    // king-safety heavy
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", // rooks + shields
            "2r2rk1/1bqnbppp/pp1ppn2/8/2P1P3/1PN1BN2/PB1QBPPP/3R1RK1 w - - 0 1", // rich mg
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            let mirror = Position::from_fen(&mirror_fen(fen)).unwrap();
            assert_eq!(
                evaluate(&pos),
                evaluate(&mirror),
                "color symmetry broken for {fen}"
            );
        }
    }
}
