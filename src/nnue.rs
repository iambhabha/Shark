//! NNUE: a small, real "efficiently updatable" neural-network evaluation.
//!
//! This module owns the *feature convention* and the *network shape* for Mythos's
//! learned evaluation. The trainer (`src/bin/train.rs`) reuses the exact same
//! feature function and constants from here via `use mythos::nnue::...`, so the
//! net that is trained and the net that is served can never disagree about what a
//! feature index means.
//!
//! ## Architecture — a perspective ("HalfKP-lite") net
//!
//! * **Input**: 768 binary features = 2 (friendly / enemy) × 6 (piece type) × 64
//!   (square). Each *perspective* color builds its own feature set, with the board
//!   vertically flipped when viewed from Black so "my back rank" is always rank 1.
//! * **Layer 1** `W1` (`[HIDDEN][768]`, row-major) + `b1` (`[HIDDEN]`): the input
//!   feature vector is sparse, so an accumulator is just the sum of the `W1`
//!   columns of the active features, plus the bias.
//! * We build **two** accumulators — one from the side-to-move's perspective, one
//!   from the not-side-to-move's — and concatenate their CReLU activations into a
//!   `2*HIDDEN` "combined" vector (side-to-move half first).
//! * **Layer 2** `W2` (`[2*HIDDEN]`) + `b2` (scalar) reduces that to a single
//!   output, which is scaled to centipawns.
//!
//! CReLU (clamped ReLU) is `x.clamp(0.0, 1.0)`, matching the trainer exactly.

use std::fs;
use std::io::{self, Write};
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

use crate::position::Position;
use crate::types::{Color, Move, MoveType, PieceType, Square};

// ---------------------------------------------------------------------------
// Architecture constants. The trainer imports these so the two stay in lockstep.
// ---------------------------------------------------------------------------

/// Hidden-layer width (per perspective). The combined layer is `2 * HIDDEN`.
pub const HIDDEN: usize = 256;
/// Number of input features: 2 (friendly/enemy) × 6 (piece type) × 64 (square).
pub const NUM_FEATURES: usize = 768;
/// Output scale: the raw network output is multiplied by this to get centipawns,
/// and the training target squashes `score_cp / SCALE` through a sigmoid.
pub const SCALE: f32 = 400.0;

/// File-format magic: the ASCII bytes "NNUE" as a little-endian `u32`.
const MAGIC: u32 = 0x4E4E_5545;

/// Half of the feature space: features 0..384 are "friendly" pieces, 384..768 are
/// "enemy" pieces (from the chosen perspective).
const HALF: usize = NUM_FEATURES / 2; // 384

// ---------------------------------------------------------------------------
// Feature extraction — the single source of truth for the input convention.
// ---------------------------------------------------------------------------

/// Fill `out` with the indices of every active input feature for `pos`, seen from
/// `perspective`. Clears `out` first.
///
/// For a piece of color `c` on square `sq`, seen from perspective color `P`:
///
/// ```text
/// oriented_sq = if P == White { sq.index() } else { sq.index() ^ 56 }  // vflip for Black
/// friendly    = (c == P)
/// feature_idx = (if friendly { 0 } else { 1 }) * 384
///             + piece_type.index() * 64
///             + oriented_sq
/// ```
///
/// The `^ 56` is a vertical flip (`Square::flip_rank`), so Black views the board
/// from its own side: the mapping is symmetric between the two colors.
pub fn active_features(pos: &Position, perspective: Color, out: &mut Vec<usize>) {
    out.clear();
    for i in 0..64 {
        // `from_index(0..64)` is always `Some`, but we match rather than unwrap.
        let sq = match crate::types::Square::from_index(i) {
            Some(s) => s,
            None => continue,
        };
        if let Some(piece) = pos.piece_at(sq) {
            out.push(feature_index(perspective, piece.color, piece.piece_type, sq));
        }
    }
}

/// The input-feature index of a piece of color `c` and type `pt` on square `sq`,
/// seen from perspective color `perspective`. This is the *single source of truth*
/// for the feature convention — both [`active_features`] (from-scratch) and the
/// incremental [`Accumulator`] read their indices from here, so the two paths can
/// never disagree about what a feature means.
///
/// See the module docs / [`active_features`] for the derivation:
///
/// ```text
/// oriented_sq = if perspective == White { sq } else { sq ^ 56 }  // vflip for Black
/// friendly    = (c == perspective)
/// idx         = (if friendly { 0 } else { 1 }) * 384 + pt.index() * 64 + oriented_sq
/// ```
#[inline]
pub fn feature_index(perspective: Color, c: Color, pt: PieceType, sq: Square) -> usize {
    let oriented_sq = if perspective == Color::White {
        sq.index()
    } else {
        sq.index() ^ 56
    };
    let friendly = c == perspective;
    (if friendly { 0 } else { 1 }) * HALF + pt.index() * 64 + oriented_sq
}

// ---------------------------------------------------------------------------
// The network.
// ---------------------------------------------------------------------------

/// The trained weights of the perspective net.
///
/// * `w1`: layer-1 weights, shape `[HIDDEN][NUM_FEATURES]` stored row-major as a
///   flat `Vec<f32>` of length `HIDDEN * NUM_FEATURES`. Element `(j, f)` — the
///   weight from feature `f` into hidden neuron `j` — is at index `j * NUM_FEATURES + f`.
/// * `w1t`: the **transposed** copy of `w1`, shape `[NUM_FEATURES][HIDDEN]`, so each
///   feature's `HIDDEN`-wide column is *contiguous*: `w1t[f * HIDDEN + j] ==
///   w1[j * NUM_FEATURES + f]`. This is derived in memory from `w1` (see
///   [`build_w1t`]) and is what the accumulator add/sub touch, because a contiguous
///   column is SIMD-friendly (a strided gather across `w1` is not). It is *not*
///   part of the on-disk format — `save` only writes `w1`.
/// * `b1`: layer-1 biases, length `HIDDEN`.
/// * `w2`: layer-2 weights, length `2 * HIDDEN` (side-to-move half first).
/// * `b2`: layer-2 bias (scalar).
pub struct Net {
    pub w1: Vec<f32>,
    pub w1t: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: f32,
}

/// Build the transposed feature-transformer weights from the row-major `w1`.
///
/// `w1` is `[HIDDEN][NUM_FEATURES]` (feature `f`'s column strided by `NUM_FEATURES`);
/// the result is `[NUM_FEATURES][HIDDEN]` so that feature `f`'s whole `HIDDEN`-wide
/// column lives contiguously at `out[f * HIDDEN .. f * HIDDEN + HIDDEN]`. The
/// accumulator update then adds/subtracts that contiguous slice, which vectorizes
/// cleanly. Deriving `w1t` keeps the `.nnue` file format unchanged.
fn build_w1t(w1: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w1.len(), HIDDEN * NUM_FEATURES);
    let mut w1t = vec![0.0f32; NUM_FEATURES * HIDDEN];
    for j in 0..HIDDEN {
        let row = j * NUM_FEATURES;
        for f in 0..NUM_FEATURES {
            w1t[f * HIDDEN + j] = w1[row + f];
        }
    }
    w1t
}

impl Net {
    /// A correctly-sized, all-zeros net. Evaluates every position to 0.
    pub fn zeros() -> Net {
        let w1 = vec![0.0; HIDDEN * NUM_FEATURES];
        let w1t = build_w1t(&w1);
        Net {
            w1,
            w1t,
            b1: vec![0.0; HIDDEN],
            w2: vec![0.0; 2 * HIDDEN],
            b2: 0.0,
        }
    }

    /// Rebuild the transposed FT weights (`w1t`) from the current `w1`.
    ///
    /// `w1t` is a derived, in-memory-only copy that the accumulator update reads
    /// (see [`build_w1t`]). It is set automatically by [`Net::zeros`] and
    /// [`Net::from_bytes`]; call this after mutating `w1` directly (e.g. a trainer
    /// updating weights) so the transposed copy stays consistent.
    pub fn rebuild_w1t(&mut self) {
        self.w1t = build_w1t(&self.w1);
    }

    /// Load a net from a file in the binary format written by [`Net::save`].
    pub fn load(path: &str) -> io::Result<Net> {
        let bytes = fs::read(path)?;
        Net::from_bytes(&bytes).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "not a valid Mythos NNUE file (bad magic, dims, or length)",
            )
        })
    }

    /// A short human-readable description of this net's shape, e.g.
    /// `"NNUE 768->256->1"`, for a UCI `info string`.
    pub fn describe(&self) -> String {
        format!("NNUE {NUM_FEATURES}->{HIDDEN}->1")
    }

    /// Parse a net from raw file bytes. Returns `None` if the magic, dimensions,
    /// or total length do not match this architecture.
    pub fn from_bytes(bytes: &[u8]) -> Option<Net> {
        // Header: magic u32, hidden u32, num_features u32.
        let mut off = 0usize;
        let magic = read_u32(bytes, &mut off)?;
        if magic != MAGIC {
            return None;
        }
        let hidden = read_u32(bytes, &mut off)? as usize;
        let num_features = read_u32(bytes, &mut off)? as usize;
        if hidden != HIDDEN || num_features != NUM_FEATURES {
            return None;
        }

        let w1_len = HIDDEN * NUM_FEATURES;
        let b1_len = HIDDEN;
        let w2_len = 2 * HIDDEN;

        let mut w1 = vec![0.0f32; w1_len];
        for w in w1.iter_mut() {
            *w = read_f32(bytes, &mut off)?;
        }
        let mut b1 = vec![0.0f32; b1_len];
        for b in b1.iter_mut() {
            *b = read_f32(bytes, &mut off)?;
        }
        let mut w2 = vec![0.0f32; w2_len];
        for w in w2.iter_mut() {
            *w = read_f32(bytes, &mut off)?;
        }
        let b2 = read_f32(bytes, &mut off)?;

        // Reject trailing garbage: the file must be exactly the expected size.
        if off != bytes.len() {
            return None;
        }

        // Derive the transposed FT weights in memory (see `build_w1t`). The disk
        // format only ever stores `w1`.
        let w1t = build_w1t(&w1);
        Some(Net { w1, w1t, b1, w2, b2 })
    }

    /// Serialize this net to `path` in the binary format [`Net::from_bytes`] reads.
    pub fn save(&self, path: &str) -> io::Result<()> {
        // Guard against a mis-sized net (e.g. hand-built): the writer assumes the
        // canonical dimensions.
        debug_assert_eq!(self.w1.len(), HIDDEN * NUM_FEATURES);
        debug_assert_eq!(self.b1.len(), HIDDEN);
        debug_assert_eq!(self.w2.len(), 2 * HIDDEN);

        let mut buf: Vec<u8> = Vec::with_capacity(
            12 + 4 * (self.w1.len() + self.b1.len() + self.w2.len() + 1),
        );
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&(HIDDEN as u32).to_le_bytes());
        buf.extend_from_slice(&(NUM_FEATURES as u32).to_le_bytes());
        for &w in &self.w1 {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for &b in &self.b1 {
            buf.extend_from_slice(&b.to_le_bytes());
        }
        for &w in &self.w2 {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf.extend_from_slice(&self.b2.to_le_bytes());

        let mut f = fs::File::create(path)?;
        f.write_all(&buf)?;
        f.flush()?;
        Ok(())
    }

    /// Compute one accumulator (length `HIDDEN`) for `pos` from `perspective`:
    /// `acc[j] = b1[j] + Σ_f W1[j][f]` over the active features `f`.
    ///
    /// `scratch` is a reusable feature buffer so callers can avoid re-allocating.
    fn accumulate(&self, pos: &Position, perspective: Color, scratch: &mut Vec<usize>) -> Vec<f32> {
        active_features(pos, perspective, scratch);
        let mut acc = self.b1.clone();
        for &f in scratch.iter() {
            // W1 row-major: neuron j, feature f is at j * NUM_FEATURES + f.
            let mut base = f;
            for a in acc.iter_mut() {
                *a += self.w1[base];
                base += NUM_FEATURES;
            }
        }
        acc
    }

    /// Evaluate `pos`, returning a **side-to-move-relative** score in centipawns
    /// (positive = the side to move is better), clamped to about ±10000.
    ///
    /// This recomputes both accumulators from scratch, so it is `O(pieces)` — fine
    /// for a from-scratch static eval and for the trainer's forward pass. (A real
    /// search would update the accumulators incrementally.)
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let stm = pos.side_to_move();
        let nstm = !stm;

        let mut scratch: Vec<usize> = Vec::with_capacity(32);
        let acc_stm = self.accumulate(pos, stm, &mut scratch);
        let acc_nstm = self.accumulate(pos, nstm, &mut scratch);

        // combined = concat(CReLU(acc_stm), CReLU(acc_nstm)); output = b2 + W2·combined.
        // Route through the same `output` path (AVX2 or scalar) as `evaluate_acc`
        // so `evaluate_acc(refresh(pos)) == evaluate(pos)` stays exact.
        let acc_stm: &[f32; HIDDEN] = (&acc_stm[..]).try_into().expect("accumulate returns HIDDEN");
        let acc_nstm: &[f32; HIDDEN] =
            (&acc_nstm[..]).try_into().expect("accumulate returns HIDDEN");
        let o = self.output(acc_stm, acc_nstm);

        let cp = (o * SCALE).round();
        cp.clamp(-10_000.0, 10_000.0) as i32
    }

    /// Evaluate from a **maintained** [`Accumulator`] instead of recomputing it
    /// from the board. This is the incremental fast path used inside the search:
    /// the accumulator is kept up to date across make/undo, so evaluation is just
    /// the layer-2 output over the two already-summed hidden vectors.
    ///
    /// `stm` is the side to move for the position the accumulator describes.
    /// The result is **numerically identical** to [`Net::evaluate`] on the same
    /// position, because both read the same `W1` columns (via the shared
    /// [`feature_index`]) into the same `b1`-seeded sums and apply the same
    /// CReLU / layer-2 arithmetic — only the *order of summation* into the
    /// accumulator differs, and `refresh` reproduces `evaluate`'s order exactly.
    pub fn evaluate_acc(&self, acc: &Accumulator, stm: Color) -> i32 {
        // Pick the side-to-move accumulator first (W2's first half), then the
        // not-side-to-move one, mirroring `evaluate`'s combined-vector layout.
        let (acc_stm, acc_nstm) = match stm {
            Color::White => (&acc.white, &acc.black),
            Color::Black => (&acc.black, &acc.white),
        };

        let o = self.output(acc_stm, acc_nstm);
        let cp = (o * SCALE).round();
        cp.clamp(-10_000.0, 10_000.0) as i32
    }

    /// The layer-2 output `b2 + Σ w2·CReLU(combined)` over the two `HIDDEN`-wide
    /// perspective vectors (side-to-move half first). Runtime-dispatches to an AVX2
    /// kernel when available, else the scalar reference; the two agree to float
    /// precision (SIMD only reorders the summation).
    #[inline]
    fn output(&self, acc_stm: &[f32; HIDDEN], acc_nstm: &[f32; HIDDEN]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if have_avx2() {
                // SAFETY: guarded by a runtime AVX2 check. `w2` is `2*HIDDEN` long
                // and each accumulator half is exactly `HIDDEN`; the kernel reads
                // them with unaligned 8-wide loads.
                return unsafe { output_avx2(&self.w2, self.b2, acc_stm, acc_nstm) };
            }
        }
        self.output_scalar(acc_stm, acc_nstm)
    }

    /// Scalar reference for [`Net::output`]: interleaves the stm/nstm halves exactly
    /// as the original `evaluate`/`evaluate_acc` loops did, so it is bit-identical
    /// to the pre-SIMD code.
    #[inline]
    fn output_scalar(&self, acc_stm: &[f32; HIDDEN], acc_nstm: &[f32; HIDDEN]) -> f32 {
        let mut o = self.b2;
        for j in 0..HIDDEN {
            o += self.w2[j] * crelu(acc_stm[j]);
            o += self.w2[HIDDEN + j] * crelu(acc_nstm[j]);
        }
        o
    }
}

// ---------------------------------------------------------------------------
// The incremental accumulator — the "efficiently updatable" part of NNUE.
// ---------------------------------------------------------------------------

/// The two layer-1 hidden vectors (one per perspective color), maintained
/// incrementally across make/undo so evaluation never re-scans the board.
///
/// `white[j] = b1[j] + Σ W1[j][f]` over the active features `f` from **White's**
/// perspective; `black` is the same from **Black's** perspective. Because every
/// input feature is binary, adding or removing a piece is just adding or
/// subtracting that feature's `W1` column into both vectors — an `O(HIDDEN)`
/// update per changed piece instead of an `O(pieces * HIDDEN)` rebuild.
#[derive(Clone)]
pub struct Accumulator {
    pub white: [f32; HIDDEN],
    pub black: [f32; HIDDEN],
}

impl Accumulator {
    /// Build an accumulator from scratch for `pos`: seed each perspective from
    /// `b1`, then add the `W1` column of every piece on the board. This is the
    /// reference every incremental update is checked against, and it reproduces
    /// [`Net::evaluate`]'s summation order (iterate squares 0..64, add each
    /// active feature) so `evaluate_acc(refresh(net, pos), stm)` is bit-identical
    /// to `evaluate(pos)`.
    pub fn refresh(net: &Net, pos: &Position) -> Accumulator {
        let mut acc = Accumulator {
            white: [0.0; HIDDEN],
            black: [0.0; HIDDEN],
        };
        acc.white.copy_from_slice(&net.b1);
        acc.black.copy_from_slice(&net.b1);

        for i in 0..64 {
            let sq = match Square::from_index(i) {
                Some(s) => s,
                None => continue,
            };
            if let Some(piece) = pos.piece_at(sq) {
                acc.add_piece(net, piece.color, piece.piece_type, sq);
            }
        }
        acc
    }

    /// Whether two accumulators agree element-wise within a small float epsilon.
    ///
    /// An incrementally-maintained accumulator and a from-scratch [`refresh`] sum
    /// the same `W1` columns but in a different order, so they match only to
    /// IEEE-754 precision — a tiny (~1e-6) drift, never a structural difference.
    /// A failure of this check means a genuine feature-mapping bug in
    /// [`apply_move`](Accumulator::apply_move), not mere float rounding.
    pub fn close_to(&self, other: &Accumulator) -> bool {
        // Generous enough to absorb accumulated float rounding across a deep line,
        // tight enough to catch any real (whole-weight-scale) mismatch.
        const EPS: f32 = 1e-3;
        self.white
            .iter()
            .zip(other.white.iter())
            .chain(self.black.iter().zip(other.black.iter()))
            .all(|(a, b)| (a - b).abs() <= EPS)
    }

    /// Add the `W1` columns of the piece `(color, pt)` on `sq` into **both**
    /// perspective vectors (White vector uses the White-perspective feature index,
    /// Black vector the Black-perspective one).
    #[inline]
    fn add_piece(&mut self, net: &Net, color: Color, pt: PieceType, sq: Square) {
        let wf = feature_index(Color::White, color, pt, sq);
        let bf = feature_index(Color::Black, color, pt, sq);
        add_column(&mut self.white, &net.w1t, wf);
        add_column(&mut self.black, &net.w1t, bf);
    }

    /// Subtract the `W1` columns of the piece `(color, pt)` on `sq` from **both**
    /// perspective vectors — the exact inverse of [`add_piece`](Accumulator::add_piece).
    #[inline]
    fn remove_piece(&mut self, net: &Net, color: Color, pt: PieceType, sq: Square) {
        let wf = feature_index(Color::White, color, pt, sq);
        let bf = feature_index(Color::Black, color, pt, sq);
        sub_column(&mut self.white, &net.w1t, wf);
        sub_column(&mut self.black, &net.w1t, bf);
    }

    /// Produce the accumulator for the position *after* `m` is played, given the
    /// `parent` accumulator (for the position *before* the move) and `pos_before`,
    /// the position **before** the move is made. Only the features touched by the
    /// move are updated, so this is `O(HIDDEN)` per changed piece.
    ///
    /// The feature diff mirrors [`Position::make_move`] exactly:
    ///
    /// * **Normal**: remove the mover from `from`; if `to` holds an enemy piece,
    ///   remove it; add the mover on `to`.
    /// * **Promotion**: remove the pawn from `from`; remove any captured piece on
    ///   `to`; add the promoted piece on `to`.
    /// * **EnPassant**: remove our pawn from `from`, add it on `to`, and remove the
    ///   enemy pawn that stood on `(file(to), rank(from))`.
    /// * **Castling**: `to` is the *rook* square. Move the king to the g/c-file and
    ///   the rook to the f/d-file (kingside if `rook_file > king_file`), same rank.
    pub fn apply_move(
        net: &Net,
        parent: &Accumulator,
        pos_before: &Position,
        m: Move,
    ) -> Accumulator {
        let mut acc = parent.clone();

        let us = pos_before.side_to_move();
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq(); // NB: for castling this is the ROOK's square.

        // The moving piece, read from the pre-move board.
        let moving = pos_before
            .piece_at(from)
            .expect("apply_move from an empty square");

        match m.move_type() {
            MoveType::Normal => {
                acc.remove_piece(net, us, moving.piece_type, from);
                if let Some(cap) = pos_before.piece_at(to) {
                    acc.remove_piece(net, cap.color, cap.piece_type, to);
                }
                acc.add_piece(net, us, moving.piece_type, to);
            }

            MoveType::Promotion => {
                acc.remove_piece(net, us, PieceType::Pawn, from);
                if let Some(cap) = pos_before.piece_at(to) {
                    acc.remove_piece(net, cap.color, cap.piece_type, to);
                }
                acc.add_piece(net, us, m.promotion_type(), to);
            }

            MoveType::EnPassant => {
                // The captured pawn sits on the square with `to`'s file and
                // `from`'s rank (directly "behind" the target).
                let cap_sq = Square::make(to.file(), from.rank());
                acc.remove_piece(net, us, PieceType::Pawn, from);
                acc.add_piece(net, us, PieceType::Pawn, to);
                acc.remove_piece(net, them, PieceType::Pawn, cap_sq);
            }

            MoveType::Castling => {
                // `to` is the rook's square; derive king/rook destinations exactly
                // as `Position::make_move` does.
                let king_from = from;
                let rook_from = to;
                let (king_to, rook_to) = if rook_from.file() > king_from.file() {
                    // King-side: king to g-file, rook to f-file.
                    (
                        Square::make(6, king_from.rank()),
                        Square::make(5, king_from.rank()),
                    )
                } else {
                    // Queen-side: king to c-file, rook to d-file.
                    (
                        Square::make(2, king_from.rank()),
                        Square::make(3, king_from.rank()),
                    )
                };
                acc.remove_piece(net, us, PieceType::King, king_from);
                acc.add_piece(net, us, PieceType::King, king_to);
                acc.remove_piece(net, us, PieceType::Rook, rook_from);
                acc.add_piece(net, us, PieceType::Rook, rook_to);
            }
        }

        acc
    }
}

/// Add the transposed FT column of feature `f` into `acc` (`acc[j] += w1t[j]`).
///
/// `w1t` is the transposed weights `[NUM_FEATURES][HIDDEN]`, so feature `f`'s whole
/// `HIDDEN`-wide column is the *contiguous* slice `w1t[f*HIDDEN .. f*HIDDEN+HIDDEN]`
/// (unlike the strided column of the row-major `w1`). Runtime-dispatches to an AVX2
/// kernel when available, else the scalar loop — the two agree to float precision.
#[inline]
fn add_column(acc: &mut [f32; HIDDEN], w1t: &[f32], f: usize) {
    let col = &w1t[f * HIDDEN..f * HIDDEN + HIDDEN];
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx2() {
            // SAFETY: guarded by a runtime AVX2 check; `col` and `acc` are both
            // exactly `HIDDEN` f32s, and the kernel uses unaligned loads/stores.
            unsafe {
                add_column_avx2(acc, col);
            }
            return;
        }
    }
    add_column_scalar(acc, col);
}

/// Subtract the transposed FT column of feature `f` from `acc` (`acc[j] -= w1t[j]`).
/// The exact inverse of [`add_column`]; same AVX2/scalar dispatch.
#[inline]
fn sub_column(acc: &mut [f32; HIDDEN], w1t: &[f32], f: usize) {
    let col = &w1t[f * HIDDEN..f * HIDDEN + HIDDEN];
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx2() {
            // SAFETY: see `add_column` — runtime-guarded, exact `HIDDEN` lengths.
            unsafe {
                sub_column_avx2(acc, col);
            }
            return;
        }
    }
    sub_column_scalar(acc, col);
}

/// Scalar `acc[j] += col[j]` over the `HIDDEN`-wide contiguous column.
#[inline]
fn add_column_scalar(acc: &mut [f32; HIDDEN], col: &[f32]) {
    for (a, &c) in acc.iter_mut().zip(col.iter()) {
        *a += c;
    }
}

/// Scalar `acc[j] -= col[j]` over the `HIDDEN`-wide contiguous column.
#[inline]
fn sub_column_scalar(acc: &mut [f32; HIDDEN], col: &[f32]) {
    for (a, &c) in acc.iter_mut().zip(col.iter()) {
        *a -= c;
    }
}

/// Attempt to load the default net file, `mythos.nnue`, from the usual places:
/// first alongside the running executable (so a shipped net travels with the
/// binary), then the current working directory (handy during development).
/// Returns the first net that loads, or `None` if neither exists / is valid — in
/// which case the caller stays on the hand-crafted evaluation.
pub fn load_default() -> Option<Net> {
    const DEFAULT_NAME: &str = "mythos.nnue";

    // (a) The directory the current executable lives in.
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
    {
        let candidate = dir.join(DEFAULT_NAME);
        if let Some(path) = candidate.to_str()
            && let Ok(net) = Net::load(path)
        {
            return Some(net);
        }
    }

    // (b) The current working directory.
    if let Ok(net) = Net::load(DEFAULT_NAME) {
        return Some(net);
    }

    None
}

/// Clamped ReLU: the activation used between the two layers.
#[inline]
pub fn crelu(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// AVX2 SIMD kernels (x86_64), selected at runtime.
//
// The hot NNUE math is two shapes: an `acc[j] ±= col[j]` over 256 f32 (the
// accumulator update, run per changed piece), and a CReLU-then-dot-product over
// 512 f32 (the layer-2 output). Both are 8-wide vectorizable with `__m256`.
//
// AVX2 is *not* enabled at compile time (baseline x86-64 target), so we detect it
// once at runtime and route to either a `#[target_feature(enable = "avx2")]`
// kernel or the scalar fallback. The SIMD result only *reorders* the float sums,
// so it agrees with the scalar path to ~1e-6.
// ---------------------------------------------------------------------------

/// Whether the running CPU supports both AVX2 and FMA, detected once and cached.
///
/// The output kernel uses `_mm256_fmadd_ps`, which needs the `fma` feature in
/// addition to `avx2`, so we gate on both. On non-x86_64 this is never called (the
/// callers are behind `#[cfg(target_arch = "x86_64")]`).
#[cfg(target_arch = "x86_64")]
#[inline]
fn have_avx2() -> bool {
    static AVX2_FMA: OnceLock<bool> = OnceLock::new();
    *AVX2_FMA
        .get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"))
}

/// AVX2 `acc[j] += col[j]` over `HIDDEN` (=256) f32 = 32 vector adds.
///
/// # Safety
/// The caller must have verified AVX2 support (via [`have_avx2`]). `acc` is exactly
/// `HIDDEN` f32; `col` must be at least `HIDDEN` f32 (it is a `HIDDEN`-wide column
/// slice). Uses unaligned loads/stores, so no alignment requirement.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn add_column_avx2(acc: &mut [f32; HIDDEN], col: &[f32]) {
    use std::arch::x86_64::*;
    debug_assert!(col.len() >= HIDDEN);
    let a = acc.as_mut_ptr();
    let c = col.as_ptr();
    let mut j = 0;
    while j < HIDDEN {
        // SAFETY: `j` steps by 8 and stops before `HIDDEN`, so every `.add(j)`
        // plus an 8-wide load/store stays within the `HIDDEN`-long `acc`/`col`.
        unsafe {
            let va = _mm256_loadu_ps(a.add(j));
            let vc = _mm256_loadu_ps(c.add(j));
            _mm256_storeu_ps(a.add(j), _mm256_add_ps(va, vc));
        }
        j += 8;
    }
}

/// AVX2 `acc[j] -= col[j]` over `HIDDEN` (=256) f32 = 32 vector subs.
///
/// # Safety
/// Same contract as [`add_column_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn sub_column_avx2(acc: &mut [f32; HIDDEN], col: &[f32]) {
    use std::arch::x86_64::*;
    debug_assert!(col.len() >= HIDDEN);
    let a = acc.as_mut_ptr();
    let c = col.as_ptr();
    let mut j = 0;
    while j < HIDDEN {
        // SAFETY: as in `add_column_avx2` — bounded 8-wide loads/stores.
        unsafe {
            let va = _mm256_loadu_ps(a.add(j));
            let vc = _mm256_loadu_ps(c.add(j));
            _mm256_storeu_ps(a.add(j), _mm256_sub_ps(va, vc));
        }
        j += 8;
    }
}

/// AVX2+FMA layer-2 output: `b2 + Σ w2·CReLU(combined)` over the two `HIDDEN` halves.
///
/// CReLU is `_mm256_max_ps(x, 0)` then `_mm256_min_ps(x, 1)`; the multiply-add into
/// an accumulator vector uses `_mm256_fmadd_ps`. The stm half (dotted with
/// `w2[0..HIDDEN]`) and the nstm half (dotted with `w2[HIDDEN..2*HIDDEN]`) are
/// accumulated into the same lane-vector, then horizontally summed once at the end
/// and added to `b2` — matching the scalar `output_scalar` to float precision.
///
/// # Safety
/// The caller must have verified AVX2+FMA support (via [`have_avx2`]). `w2` must be
/// at least `2 * HIDDEN` f32; each accumulator half is exactly `HIDDEN`. Uses
/// unaligned loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn output_avx2(w2: &[f32], b2: f32, acc_stm: &[f32; HIDDEN], acc_nstm: &[f32; HIDDEN]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert!(w2.len() >= 2 * HIDDEN);

    let w2p = w2.as_ptr();
    let stm = acc_stm.as_ptr();
    let nstm = acc_nstm.as_ptr();

    // SAFETY: `j` steps by 8 and stops before `HIDDEN`; the stm/nstm loads read
    // within the `HIDDEN`-long accumulators, and the `w2` loads read within its
    // `2*HIDDEN`-long buffer (`j` for the first half, `HIDDEN + j` for the second).
    let sum = unsafe {
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let mut sum = _mm256_setzero_ps();
        let mut j = 0;
        while j < HIDDEN {
            // stm half: CReLU(acc_stm[j..]) * w2[j..].
            let xs = _mm256_loadu_ps(stm.add(j));
            let cs = _mm256_min_ps(_mm256_max_ps(xs, zero), one);
            let ws = _mm256_loadu_ps(w2p.add(j));
            sum = _mm256_fmadd_ps(cs, ws, sum);

            // nstm half: CReLU(acc_nstm[j..]) * w2[HIDDEN + j..].
            let xn = _mm256_loadu_ps(nstm.add(j));
            let cn = _mm256_min_ps(_mm256_max_ps(xn, zero), one);
            let wn = _mm256_loadu_ps(w2p.add(HIDDEN + j));
            sum = _mm256_fmadd_ps(cn, wn, sum);

            j += 8;
        }
        sum
    };

    b2 + hsum256_ps(sum)
}

/// Horizontal sum of the 8 lanes of a `__m256`.
///
/// # Safety
/// Caller must have AVX (implied by AVX2). Pure lane arithmetic, no memory access.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // Pure register/lane arithmetic (no memory access): safe inside this
    // `avx2`-enabled fn, so no `unsafe` block is needed.
    // Fold the high 128 into the low 128, then reduce the 4 lanes.
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let mut s = _mm_add_ps(lo, hi); // [a0+a4, a1+a5, a2+a6, a3+a7]
    let shuf = _mm_movehdup_ps(s); // [s1, s1, s3, s3]
    s = _mm_add_ps(s, shuf); // [s0+s1, _, s2+s3, _]
    let hi64 = _mm_movehl_ps(shuf, s); // move s2+s3 into lane 0
    s = _mm_add_ss(s, hi64);
    _mm_cvtss_f32(s)
}

// ---------------------------------------------------------------------------
// Little-endian primitive readers (inverse of `to_le_bytes` used in `save`).
// ---------------------------------------------------------------------------

/// Read a little-endian `u32` at `*off`, advancing `*off`. `None` if out of range.
#[inline]
fn read_u32(bytes: &[u8], off: &mut usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[*off..end]);
    *off = end;
    Some(u32::from_le_bytes(b))
}

/// Read a little-endian `f32` at `*off`, advancing `*off`. `None` if out of range.
#[inline]
fn read_f32(bytes: &[u8], off: &mut usize) -> Option<f32> {
    let end = off.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[*off..end]);
    *off = end;
    Some(f32::from_le_bytes(b))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_net_evaluates_to_zero() {
        let net = Net::zeros();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            assert_eq!(net.evaluate(&pos), 0, "zeros net must score 0 for {fen}");
        }
    }

    #[test]
    fn save_load_round_trips() {
        // Build a net with a few distinct, recognizable values.
        let mut net = Net::zeros();
        net.w1[0] = 0.5;
        net.w1[NUM_FEATURES + 3] = -0.25;
        net.w1[HIDDEN * NUM_FEATURES - 1] = 1.5;
        net.b1[0] = -0.75;
        net.b1[HIDDEN - 1] = 0.125;
        net.w2[0] = 2.0;
        net.w2[2 * HIDDEN - 1] = -2.0;
        net.b2 = 0.333;

        let dir = std::env::temp_dir();
        let path = dir.join("mythos_nnue_roundtrip_test.bin");
        let path_str = path.to_str().unwrap();

        net.save(path_str).unwrap();
        let loaded = Net::load(path_str).unwrap();

        assert_eq!(loaded.w1, net.w1);
        assert_eq!(loaded.b1, net.b1);
        assert_eq!(loaded.w2, net.w2);
        assert_eq!(loaded.b2, net.b2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn start_position_has_32_valid_features_per_perspective() {
        let pos = Position::startpos();
        let mut feats = Vec::new();
        for perspective in [Color::White, Color::Black] {
            active_features(&pos, perspective, &mut feats);
            assert_eq!(feats.len(), 32, "start position has 32 pieces");
            for &f in &feats {
                assert!(f < NUM_FEATURES, "feature {f} out of range");
            }
        }
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut net = Net::zeros();
        net.b2 = 1.0;
        let dir = std::env::temp_dir();
        let path = dir.join("mythos_nnue_badmagic_test.bin");
        let path_str = path.to_str().unwrap();
        net.save(path_str).unwrap();

        let mut bytes = std::fs::read(path_str).unwrap();
        bytes[0] ^= 0xFF; // corrupt the magic
        assert!(Net::from_bytes(&bytes).is_none());
        let _ = std::fs::remove_file(&path);
    }

    // -- Incremental accumulator ------------------------------------------

    /// A deterministic "random-ish" net: every weight is a small, reproducible
    /// value derived from its index by a cheap hash. This gives every hidden
    /// neuron and feature a distinct nonzero weight (so a wrong feature index or
    /// a sign error in an update cannot hide behind a zero), without needing an
    /// RNG crate or a real net file.
    fn pseudo_random_net() -> Net {
        // A splitmix-style scramble mapped into roughly [-0.5, 0.5).
        fn weight(i: usize) -> f32 {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 29;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 32;
            // Take 16 bits -> [0, 65535] -> [-0.5, ~0.5).
            ((x & 0xFFFF) as f32) / 65536.0 - 0.5
        }

        let mut net = Net::zeros();
        for (i, w) in net.w1.iter_mut().enumerate() {
            *w = weight(i);
        }
        for (i, b) in net.b1.iter_mut().enumerate() {
            *b = weight(i + 0x1000_0000);
        }
        for (i, w) in net.w2.iter_mut().enumerate() {
            *w = weight(i + 0x2000_0000);
        }
        net.b2 = weight(0x3000_0000);
        // `w1` was mutated directly, so refresh the derived transposed copy that the
        // accumulator update reads.
        net.rebuild_w1t();
        net
    }

    /// The two accumulators agree element-wise within a tiny float epsilon.
    fn accs_close(a: &Accumulator, b: &Accumulator) -> bool {
        a.close_to(b)
    }

    #[test]
    fn evaluate_acc_matches_from_scratch_evaluate() {
        // The core invariant: evaluating from a freshly refreshed accumulator must
        // give the *identical* centipawn score as the from-scratch `evaluate`.
        let net = pseudo_random_net();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
            "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3",
            "8/P7/8/8/8/8/8/k1K5 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 5 12",
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            let acc = Accumulator::refresh(&net, &pos);
            assert_eq!(
                net.evaluate_acc(&acc, pos.side_to_move()),
                net.evaluate(&pos),
                "acc eval must equal from-scratch eval for {fen}"
            );
        }
    }

    /// Apply a sequence of moves, threading the accumulator incrementally, and
    /// after each move assert the incremental accumulator equals a from-scratch
    /// refresh of the resulting position (and that the eval matches too).
    fn assert_incremental_matches(net: &Net, fen: &str, moves: &[Move]) {
        let mut pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("bad fen {fen}: {e}"));
        let mut acc = Accumulator::refresh(net, &pos);
        // Sanity: the starting accumulator itself agrees with the eval.
        assert_eq!(net.evaluate_acc(&acc, pos.side_to_move()), net.evaluate(&pos));

        for &m in moves {
            // Compute the child accumulator from the pre-move position, then make
            // the move on the scratch board to advance `pos`.
            let child = Accumulator::apply_move(net, &acc, &pos, m);
            pos.make_move(m);

            let fresh = Accumulator::refresh(net, &pos);
            assert!(
                accs_close(&child, &fresh),
                "incremental accumulator drifted after {m} in {fen}"
            );
            assert_eq!(
                net.evaluate_acc(&child, pos.side_to_move()),
                net.evaluate(&pos),
                "incremental eval drifted after {m} in {fen}"
            );
            acc = child;
        }
    }

    #[test]
    fn incremental_normal_and_capture_and_double_push() {
        let net = pseudo_random_net();
        // A full opening line: double pushes (ep targets), a knight develop, a
        // bishop develop, castling, and captures along the way.
        assert_incremental_matches(
            &net,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &[
                Move::normal(Square::E2, Square::E4), // white double push
                Move::normal(Square::D7, Square::D5), // black double push
                Move::normal(Square::E4, Square::D5), // capture (pawn takes pawn)
                Move::normal(Square::G8, Square::F6), // knight develop
                Move::normal(Square::G1, Square::F3), // knight develop
                Move::normal(Square::F6, Square::D5), // knight recaptures pawn
            ],
        );
    }

    #[test]
    fn incremental_en_passant() {
        let net = pseudo_random_net();
        // White pawn on d5, black just played ...c5 (ep target on c6): d5xc6 e.p.
        assert_incremental_matches(
            &net,
            "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3",
            &[Move::en_passant(Square::D5, Square::C6)],
        );
    }

    #[test]
    fn incremental_castling_both_sides() {
        let net = pseudo_random_net();
        // White king-side, then continue and let Black castle king-side too.
        assert_incremental_matches(
            &net,
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            &[
                Move::castling(Square::E1, Square::H1), // white O-O
                Move::castling(Square::E8, Square::H8), // black O-O
            ],
        );
        // White queen-side, then Black queen-side.
        assert_incremental_matches(
            &net,
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            &[
                Move::castling(Square::E1, Square::A1), // white O-O-O
                Move::castling(Square::E8, Square::A8), // black O-O-O
            ],
        );
    }

    #[test]
    fn incremental_promotion_quiet_and_capture() {
        let net = pseudo_random_net();
        // A quiet promotion to a queen.
        assert_incremental_matches(
            &net,
            "k7/P7/8/8/8/8/8/K7 w - - 0 1",
            &[Move::promotion(Square::A7, Square::A8, PieceType::Queen)],
        );
        // A capturing promotion: b7 takes the rook on a8, promoting to a knight.
        assert_incremental_matches(
            &net,
            "r6k/1P6/8/8/8/8/8/K7 w - - 0 1",
            &[Move::promotion(Square::B7, Square::A8, PieceType::Knight)],
        );
    }

    #[test]
    fn incremental_mixed_sequence_on_kiwipete() {
        let net = pseudo_random_net();
        // A rich middlegame with knight moves, a pawn capture, and castling mixed
        // together — the accumulator must stay exact across all of them.
        assert_incremental_matches(
            &net,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[
                Move::normal(Square::D5, Square::E6),   // pawn captures pawn
                Move::normal(Square::E7, Square::E6),   // queen recaptures
                Move::castling(Square::E1, Square::H1), // white O-O
                Move::normal(Square::A6, Square::E2),   // bishop captures bishop
                Move::normal(Square::F3, Square::E2),   // queen recaptures bishop
            ],
        );
    }
}
