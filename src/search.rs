//! Search: given a position, walk the game tree and return the best move.
//!
//! This is the engine's "brain". It mirrors the classic Stockfish search skeleton
//! but stays deliberately compact and readable — this is a learning project, so
//! every technique is documented in one plain sentence where it appears.
//!
//! The core is **negamax alpha-beta** (a single recursive routine that scores a
//! position from the side-to-move's perspective) wrapped in **iterative
//! deepening** (search depth 1, then 2, then 3, ... reusing what we learn each
//! time to order moves better and to always have a good move ready if time runs
//! out). On top of that we layer the standard speed-ups:
//!
//!  * a **transposition table** to reuse results for positions reached by
//!    different move orders,
//!  * **principal variation search** (PVS): assume the first move is best and
//!    prove the rest are worse with a cheap null-window search,
//!  * **quiescence search**: at the leaves, keep resolving captures so we never
//!    evaluate a position in the middle of a trade (the "horizon effect"),
//!  * **move ordering** (TT move, MVV-LVA captures, killers, history) so the
//!    best moves are tried first and alpha-beta prunes as much as possible,
//!  * **draw detection** for the 50-move rule and repetitions.
//!
//! Scores are centipawns from the side-to-move's view (the negamax convention).
//! A checkmate is worth [`MATE`] minus the ply it is delivered at, so a faster
//! mate scores higher; being mated is the negation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::eval::piece_value;
use crate::movegen::generate_legal;
use crate::position::Position;
use crate::see::see_ge;
use crate::tt::{Bound, TranspositionTable};
use crate::types::{Color, Move, MoveType, PieceType};

// ---------------------------------------------------------------------------
// Score constants.
// ---------------------------------------------------------------------------

/// The score of a checkmate delivered at ply 0. A mate `n` plies away scores
/// `MATE - n`, so a shorter mate is always preferred. Kept well below `INFINITY`
/// so window arithmetic (`-beta`, `alpha + 1`, ...) never overflows.
pub const MATE: i32 = 30000;

/// A score larger than any real evaluation — the initial alpha/beta bounds.
pub const INFINITY: i32 = 31000;

/// Any score at or beyond this magnitude is a "mate score" (a forced mate was
/// found), as opposed to an ordinary centipawn evaluation. The margin (`MAX_PLY`)
/// leaves room for the per-ply mate-distance adjustment.
const MATE_IN_MAX: i32 = MATE - MAX_PLY as i32;

/// Hard ceiling on search depth / recursion, sizing the killer and repetition
/// bookkeeping. Chess searches never come close to this in practice.
const MAX_PLY: usize = 128;

/// How deep the quiescence (captures-only) search may extend before we force a
/// static evaluation, guarding against pathological capture chains.
const MAX_QPLY: i32 = 32;

/// Contempt: the score (from the side-to-move's view) returned for a draw by
/// repetition / 50-move rule / stalemate. A small *negative* value means the
/// engine mildly dislikes drawing and prefers to keep playing an equal game.
const DRAW_SCORE: i32 = -10;

/// Per-ply centipawn slack for SEE-based capture pruning in the main search. A
/// capture is pruned at low depth only when its static-exchange value is worse
/// than `-SEE_PRUNE_MARGIN * depth`, so the deeper the node the larger a material
/// loss we are willing to dismiss without searching it.
const SEE_PRUNE_MARGIN: i32 = 100;

/// Bound on the butterfly history score. The gravity update keeps every entry in
/// `[-HISTORY_MAX, HISTORY_MAX]`, so history stays comparable across a game and
/// can never overflow the ordering bands.
const HISTORY_MAX: i32 = 16384;

/// Sentinel stored in [`Searcher::eval_stack`] at nodes where a static eval is
/// meaningless (in check), so the "improving" test can skip them.
const NONE_EVAL: i32 = i32::MIN;

// ---------------------------------------------------------------------------
// Public types.
// ---------------------------------------------------------------------------

/// The constraints a UCI `go` command places on a search: how deep, how long, or
/// how many nodes we are allowed to spend. All fields are optional; `Default`
/// gives an unconstrained (but not infinite) search.
#[derive(Clone, Debug)]
pub struct SearchLimits {
    /// Search exactly this many plies deep and stop.
    pub depth: Option<u32>,
    /// Spend at most this many milliseconds on the move.
    pub movetime_ms: Option<u64>,
    /// White's / Black's remaining clock, in milliseconds.
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    /// White's / Black's increment per move, in milliseconds.
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    /// Moves left until the next time control (for time budgeting).
    pub movestogo: Option<u32>,
    /// Stop after searching this many nodes.
    pub nodes: Option<u64>,
    /// Search until explicitly told to stop (no time or depth cap).
    pub infinite: bool,
}

impl Default for SearchLimits {
    // Spelled out (rather than `#[derive(Default)]`) so the "unconstrained"
    // baseline is explicit and easy to read.
    #[allow(clippy::derivable_impls)]
    fn default() -> SearchLimits {
        SearchLimits {
            depth: None,
            movetime_ms: None,
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            movestogo: None,
            nodes: None,
            infinite: false,
        }
    }
}

/// What a completed search hands back: the move to play plus diagnostic detail.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// The best move found (or [`Move::NONE`] if the position has no legal move).
    pub best_move: Move,
    /// Its score, in centipawns, from the root side-to-move's perspective.
    pub score: i32,
    /// The depth of the last fully completed iteration.
    pub depth: u32,
    /// Total nodes visited across the whole search.
    pub nodes: u64,
    /// The principal variation: the expected line of best play.
    pub pv: Vec<Move>,
}

// ---------------------------------------------------------------------------
// The searcher.
// ---------------------------------------------------------------------------

/// Holds everything that persists between and within searches: the transposition
/// table and the heuristic tables (killers / history), plus the per-search
/// bookkeeping (node counter, deadline, stop flag, repetition stack).
pub struct Searcher {
    /// Cache of previously searched positions, shared across the whole game.
    tt: TranspositionTable,
    /// Two "killer" quiet moves per ply that recently caused a beta cutoff.
    killers: [[Move; 2]; MAX_PLY],
    /// Butterfly history: a signed `[from][to]` score for quiet moves, raised on
    /// a cutoff and lowered (malus) when a sibling cut off instead. Kept bounded
    /// by a "gravity" update so it self-normalizes.
    history: [[i32; 64]; 64],
    /// Static eval recorded per ply, so a node can tell whether its side's
    /// position is *improving* versus two plies ago (a cheap trend signal that
    /// tunes reductions and pruning).
    eval_stack: [i32; MAX_PLY],
    /// Positions seen along the current root-to-leaf line, for repetition draws.
    repetitions: Vec<u64>,
    /// Nodes visited in the current search.
    nodes: u64,
    /// Absolute time to stop by, if a time budget applies.
    deadline: Option<Instant>,
    /// Node cap from the limits, if any.
    node_limit: Option<u64>,
    /// Set once the search must abort (time up, stop flag, node cap).
    stopped: bool,
    /// The learned NNUE evaluator, if one has been loaded. When present, static
    /// evaluations use it instead of the hand-crafted eval; when `None`, we fall
    /// back to [`crate::eval::evaluate`].
    net: Option<crate::nnue::Net>,
    /// The incremental NNUE accumulator stack, one entry per ply along the current
    /// root-to-leaf line. Only maintained when `net` is `Some`: the root is
    /// [`refresh`](crate::nnue::Accumulator::refresh)ed once, then each make pushes
    /// the child accumulator and each undo pops it, so `static_eval` reads the top
    /// instead of rebuilding from the board. Empty when running on the HCE.
    acc_stack: Vec<crate::nnue::Accumulator>,
}

impl Searcher {
    /// Create a searcher with a transposition table of `tt_size_mb` megabytes.
    pub fn new(tt_size_mb: usize) -> Searcher {
        Searcher {
            tt: TranspositionTable::new(tt_size_mb),
            killers: [[Move::NONE; 2]; MAX_PLY],
            history: [[0; 64]; 64],
            eval_stack: [0; MAX_PLY],
            repetitions: Vec::with_capacity(MAX_PLY),
            nodes: 0,
            deadline: None,
            node_limit: None,
            stopped: false,
            net: None,
            acc_stack: Vec::with_capacity(MAX_PLY + 2),
        }
    }

    /// Install (or clear) the NNUE evaluator. With `Some(net)` every static eval
    /// runs through the network; with `None` the search reverts to the
    /// hand-crafted evaluation.
    pub fn set_net(&mut self, net: Option<crate::nnue::Net>) {
        self.net = net;
    }

    /// Whether an NNUE evaluator is currently loaded.
    pub fn has_net(&self) -> bool {
        self.net.is_some()
    }

    /// The static evaluation used everywhere in the search: the NNUE net if one is
    /// loaded, otherwise the hand-crafted evaluation. Both return a centipawn score
    /// from the side-to-move's perspective, so callers are agnostic to the source.
    ///
    /// With a net loaded, we evaluate from the **maintained** accumulator on top of
    /// [`acc_stack`](Searcher::acc_stack) — kept in sync with `pos` by the push/pop
    /// around every make/undo — which is numerically identical to a from-scratch
    /// `net.evaluate(pos)` but far cheaper. The `debug_assert` proves that identity
    /// in test/debug builds.
    fn static_eval(&self, pos: &Position) -> i32 {
        match &self.net {
            Some(n) => {
                let acc = self
                    .acc_stack
                    .last()
                    .expect("acc_stack must be seeded when a net is loaded");
                // In debug builds, prove the maintained accumulator has not drifted
                // from a from-scratch rebuild. We compare the *hidden vectors*
                // within a float epsilon rather than the rounded i32 eval: the two
                // summation orders (square-order rebuild vs. move-order updates) are
                // equal only to IEEE-754 precision (~1e-7 here), so on a rounding
                // boundary the final centipawn value may differ by 1, but the
                // accumulator itself must stay tight. A large delta would mean a real
                // feature-mapping bug in `apply_move`.
                debug_assert!(
                    crate::nnue::Accumulator::close_to(acc, &crate::nnue::Accumulator::refresh(n, pos)),
                    "incremental accumulator diverged from the from-scratch rebuild"
                );
                n.evaluate_acc(acc, pos.side_to_move())
            }
            None => crate::eval::evaluate(pos),
        }
    }

    /// Push the child accumulator for playing `m`, computed from `pos` (the
    /// position **before** the move). A no-op on the HCE. Call this immediately
    /// *before* `pos.make_move(m)`, and pair it with [`pop_accumulator`] after the
    /// matching `undo_move`.
    #[inline]
    fn push_accumulator(&mut self, pos: &Position, m: Move) {
        if let Some(net) = &self.net {
            let top = self
                .acc_stack
                .last()
                .expect("acc_stack must be seeded when a net is loaded");
            let child = crate::nnue::Accumulator::apply_move(net, top, pos, m);
            self.acc_stack.push(child);
        }
    }

    /// Push a duplicate of the current accumulator for a null move (no piece moves,
    /// so the features are unchanged). A no-op on the HCE. Pair with
    /// [`pop_accumulator`] after `undo_null_move`.
    #[inline]
    fn push_null_accumulator(&mut self) {
        if self.net.is_some() {
            let top = self
                .acc_stack
                .last()
                .expect("acc_stack must be seeded when a net is loaded")
                .clone();
            self.acc_stack.push(top);
        }
    }

    /// Pop the accumulator pushed for the move (or null move) we are undoing. A
    /// no-op on the HCE. Must balance every [`push_accumulator`] /
    /// [`push_null_accumulator`] on every code path.
    #[inline]
    fn pop_accumulator(&mut self) {
        if self.net.is_some() {
            self.acc_stack.pop();
        }
    }

    /// Forget everything learned so far — used for the UCI `ucinewgame` command
    /// so a fresh game does not inherit stale table entries.
    pub fn clear(&mut self) {
        self.tt.clear();
        self.killers = [[Move::NONE; 2]; MAX_PLY];
        self.history = [[0; 64]; 64];
        self.eval_stack = [0; MAX_PLY];
        self.repetitions.clear();
    }

    // -----------------------------------------------------------------------
    // Top-level driver: iterative deepening.
    // -----------------------------------------------------------------------

    /// Search `root` under `limits`, returning the best move once done.
    ///
    /// `stop` is a flag another thread (the UCI loop) can set to force an early
    /// halt; we also stop on our own time / depth / node budget. The returned
    /// result always reflects the last *fully completed* iteration, never a
    /// half-searched one.
    pub fn search(
        &mut self,
        root: &Position,
        limits: &SearchLimits,
        stop: &Arc<AtomicBool>,
    ) -> SearchResult {
        let mut pos = root.clone();

        // Reset per-search state (heuristics persist across searches on purpose).
        self.nodes = 0;
        self.stopped = false;
        self.node_limit = limits.nodes;
        self.repetitions.clear();
        self.deadline = compute_deadline(root.side_to_move(), limits);

        // Seed the incremental NNUE accumulator for the root. Every make along the
        // search pushes a child accumulator and every undo pops it, so the top of
        // the stack always describes the position currently on the board. (No-op on
        // the HCE, where the stack stays empty.)
        self.acc_stack.clear();
        if let Some(net) = &self.net {
            self.acc_stack
                .push(crate::nnue::Accumulator::refresh(net, &pos));
        }

        let start = Instant::now();
        let max_depth = limits.depth.unwrap_or(MAX_PLY as u32).min(MAX_PLY as u32);

        // Whatever happens, we must be able to return *some* legal move.
        let mut best = SearchResult {
            best_move: Move::NONE,
            score: 0,
            depth: 0,
            nodes: 0,
            pv: Vec::new(),
        };

        // Seed with any legal move so a stop before depth 1 finishes still
        // returns something playable.
        {
            let legal = generate_legal(&mut pos);
            if legal.is_empty() {
                // The game is already over at the root: report the terminal
                // score (checkmate if we're in check, else stalemate = draw) and
                // no move. A terminal draw at the root is a true 0 (contempt only
                // biases *choices* between playable moves, never the final verdict
                // of an already-over game).
                best.score = if pos.in_check() { -MATE } else { 0 };
                best.nodes = self.nodes;
                return best;
            }
            best.best_move = legal[0];
        }

        // Iterative deepening: search progressively deeper. Each completed
        // iteration refines the best move and warms the TT for the next one.
        for depth in 1..=max_depth {
            let mut pv = Vec::new();
            let score = self.aspiration_search(&mut pos, depth as i32, best.score, &mut pv, stop);

            // If the search was aborted mid-iteration, discard its (unreliable)
            // result and keep the last completed depth's answer.
            if self.stopped {
                break;
            }

            // Commit this iteration's result.
            best.score = score;
            best.depth = depth;
            best.nodes = self.nodes;
            if let Some(m) = pv.first() {
                best.best_move = *m;
            }
            best.pv = pv;

            // Emit a UCI info line for this completed depth.
            let elapsed = start.elapsed();
            print_info(depth, best.score, self.nodes, elapsed, &best.pv);

            // No point searching deeper than a forced mate we already found.
            if is_mate_score(best.score) {
                break;
            }
            // Respect an explicit depth cap and the stop conditions.
            if self.check_stop(stop) {
                break;
            }
        }

        best.nodes = self.nodes;
        best
    }

    /// Run one iterative-deepening iteration through an **aspiration window**.
    ///
    /// Once earlier depths have given us a reliable score, most positions barely
    /// change from one depth to the next, so we search the root with a narrow
    /// window centred on `prev_score` (`prev ± delta`) instead of the full
    /// `(-INF, INF)`. A narrow window prunes far more, buying us deeper searches.
    /// If reality falls outside the guess — the search returns `<= alpha`
    /// (fail-low) or `>= beta` (fail-high) — we widen that side and re-search the
    /// *same* depth; once the window gets wide we fall back to a full window so we
    /// always get an exact, trustworthy score for this depth.
    fn aspiration_search(
        &mut self,
        pos: &mut Position,
        depth: i32,
        prev_score: i32,
        pv: &mut Vec<Move>,
        stop: &Arc<AtomicBool>,
    ) -> i32 {
        // Shallow depths are cheap and their scores too jumpy to guess a tight
        // window for; also skip aspiration if the previous score is already a
        // mate (a mate window must stay full so the mate is never clipped away).
        if depth < 5 || is_mate_score(prev_score) {
            return self.negamax(pos, depth, -INFINITY, INFINITY, 0, pv, true, true, stop);
        }

        // Start with a tight window and widen on whichever side keeps failing.
        let mut delta = 25;
        let mut alpha = (prev_score - delta).max(-INFINITY);
        let mut beta = (prev_score + delta).min(INFINITY);

        loop {
            let score = self.negamax(pos, depth, alpha, beta, 0, pv, true, true, stop);

            // A stop mid-iteration makes the score meaningless; the caller will
            // discard it and keep the previous depth's answer (including its
            // best move), so we just hand it back and bail.
            if self.stopped {
                return score;
            }

            // A mate popped out of a narrow window: re-search with a full window
            // so the mate distance is scored exactly and never hidden.
            if is_mate_score(score) {
                return self.negamax(pos, depth, -INFINITY, INFINITY, 0, pv, true, true, stop);
            }

            if score <= alpha {
                // Fail-low: the true score is below our guess. Keep beta anchored
                // near the guess and drop alpha, doubling the margin each retry.
                beta = (alpha + beta) / 2;
                alpha = (score - delta).max(-INFINITY);
            } else if score >= beta {
                // Fail-high: the true score is above our guess. Push beta up.
                beta = (score + delta).min(INFINITY);
            } else {
                // The score landed inside the window: it is exact, we are done.
                return score;
            }

            // Widen for the next attempt; once the window is very wide, opening it
            // fully is cheaper than repeated re-searches — and guarantees a result.
            delta += delta / 2;
            if delta > 600 {
                alpha = -INFINITY;
                beta = INFINITY;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Negamax with alpha-beta (fail-soft).
    // -----------------------------------------------------------------------

    /// Score the position `depth` plies deep, only caring about scores inside the
    /// `(alpha, beta)` window (anything outside is provably irrelevant to the
    /// parent). Returns the best score for the side to move; fills `pv` with the
    /// best line. `ply` is the distance from the root (used for mate scoring).
    ///
    /// `is_pv` marks a principal-variation node (a full-window search); the pruning
    /// heuristics below are only applied in non-PV nodes, where a slightly-wrong
    /// answer can be tolerated. `can_null` is `false` when the parent just played a
    /// null move, so we never pass twice in a row.
    // The wide signature (window, ply, pv, flags, stop) is inherent to a search node.
    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        pos: &mut Position,
        mut depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        pv: &mut Vec<Move>,
        is_pv: bool,
        can_null: bool,
        stop: &Arc<AtomicBool>,
    ) -> i32 {
        pv.clear();

        // (1) Poll the clock/stop flag every so often so we can bail out promptly
        // without checking on every single node.
        self.nodes += 1;
        if self.nodes.is_multiple_of(2048) && self.check_stop(stop) {
            self.stopped = true;
        }
        if self.stopped {
            return 0; // an ignored value; the caller discards aborted results.
        }

        let is_root = ply == 0;

        // (2) Draw detection: the 50-move rule and repetitions are immediate
        // draws (score 0). We never claim a draw at the root, so we always have a
        // move to return.
        if !is_root {
            if pos.halfmove_clock() >= 100 {
                return DRAW_SCORE;
            }
            if self.is_repetition(pos) {
                return DRAW_SCORE;
            }
        }

        // Clamp recursion so the killer / pv indexing stays in bounds.
        if ply >= MAX_PLY - 1 {
            return self.static_eval(pos);
        }

        // (3) Transposition-table probe. A stored result searched at least as deep
        // as we need can end the search here (respecting its bound); either way we
        // remember its move to try first. Mate scores are stored relative to the
        // storing node, so we translate back into this node's frame.
        let key = pos.key();
        let mut tt_move = Move::NONE;
        if let Some(entry) = self.tt.probe(key) {
            tt_move = entry.best_move;
            if !is_root && entry.depth as i32 >= depth {
                let score = score_from_tt(entry.score, ply);
                match entry.bound {
                    Bound::Exact => return score,
                    Bound::Lower if score >= beta => return score,
                    Bound::Upper if score <= alpha => return score,
                    _ => {}
                }
            }
        }

        // Check extension: if we are in check, search one ply deeper so we resolve
        // the threat instead of stopping the count in a forcing line.
        let in_check = pos.in_check();
        if in_check {
            depth += 1;
        }

        // (4) At the horizon, hand off to quiescence search so we only ever score
        // "quiet" positions with no pending captures.
        if depth <= 0 {
            return self.quiescence(pos, alpha, beta, ply, stop);
        }

        // (4a) Forward pruning. These heuristics only fire in a non-PV node that
        // is not in check and whose window is a plain (non-mate) one: they trade a
        // little accuracy for a lot of speed, so we keep them well away from the
        // exact PV and from forced-mate lines. We compute the static eval once
        // here and reuse it for both the node-level pruning below and the
        // move-level pruning (LMP / futility) in the loop.
        let non_mate_window = beta.abs() < MATE_IN_MAX && alpha.abs() < MATE_IN_MAX;
        let prunable = !is_pv && !in_check && non_mate_window;
        // Static eval, used both for the forward-pruning heuristics below and for
        // the "improving" trend. It is meaningless while in check, so we skip it
        // there (and such nodes count as not improving).
        let static_eval = if in_check { NONE_EVAL } else { self.static_eval(pos) };
        self.eval_stack[ply] = static_eval;
        // Are we better off than two plies ago (same side to move)? If so, a
        // fail-low is less likely, so we can prune a touch harder / reduce a touch
        // more; if not, we stay cautious.
        let improving = !in_check
            && ply >= 2
            && self.eval_stack[ply - 2] != NONE_EVAL
            && static_eval > self.eval_stack[ply - 2];
        if prunable {
            // (4b) Reverse futility pruning (a.k.a. static null move): if our
            // static eval is already so far above beta that even giving back
            // `margin` per remaining ply would still fail high, we assume this node
            // fails high and return early without searching a single move.
            const RFP_MARGIN: i32 = 90;
            // When improving, shave one depth off the margin so we prune a little
            // more readily; the trend says this node is unlikely to fail low.
            let rfp_depth = depth - improving as i32;
            if depth <= 6 && static_eval - RFP_MARGIN * rfp_depth >= beta {
                return static_eval;
            }

            // (4c) Null-move pruning: let the opponent move twice. If passing our
            // turn and searching to a reduced depth *still* fails high, the real
            // position is almost certainly winning, so we prune. We require depth,
            // a static eval already at/above beta, non-pawn material for the side
            // to move (so we are not in zugzwang, where passing would help), and
            // that the parent did not itself pass (no two nulls in a row).
            if can_null && depth >= 3 && static_eval >= beta && self.has_non_pawn_material(pos) {
                let r = 3 + depth / 3; // search reduction for the null search.
                self.push_null_accumulator();
                let undo = pos.make_null_move();
                let mut child_pv: Vec<Move> = Vec::new();
                let score = -self.negamax(
                    pos,
                    depth - r - 1,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    &mut child_pv,
                    false, // the null search is never a PV node.
                    false, // and it may not immediately pass again.
                    stop,
                );
                pos.undo_null_move(undo);
                self.pop_accumulator();

                if self.stopped {
                    return 0;
                }
                // A fail-high proves nothing tactical (the opponent got a free
                // move), so never propagate a mate score out of it — clamp to beta.
                if score >= beta {
                    return beta;
                }
            }
        }

        // (5) Generate legal moves. No moves means the game is over right here:
        // checkmate if we are in check, otherwise stalemate (a draw).
        let moves = generate_legal(pos);
        if moves.is_empty() {
            return if in_check {
                -(MATE - ply as i32)
            } else {
                DRAW_SCORE
            };
        }

        // (6) Order the moves so the most promising are searched first — this is
        // what makes alpha-beta prune effectively.
        let mut scored = self.order_moves(pos, &moves, tt_move, ply);

        let orig_alpha = alpha;
        let mut best_score = -INFINITY;
        let mut best_move = Move::NONE;
        let mut child_pv: Vec<Move> = Vec::new();

        // Quiet moves we actually searched, so that when a *later* quiet move
        // causes a cutoff we can give these a history malus. A fixed stack buffer
        // keeps this allocation-free in the hot loop.
        let mut quiets_tried: [Move; 64] = [Move::NONE; 64];
        let mut n_quiets = 0usize;

        // Push our key so children can detect a repetition back to this node.
        self.repetitions.push(key);

        for i in 0..scored.len() {
            // Selection sort: pull the best-scored remaining move to the front.
            let mut best_idx = i;
            for j in (i + 1)..scored.len() {
                if scored[j].1 > scored[best_idx].1 {
                    best_idx = j;
                }
            }
            scored.swap(i, best_idx);
            let m = scored[i].0;

            // Is this a quiet move that is a candidate for late-move reduction?
            // (We decide before making the move so we can still read the board.)
            let quiet = is_quiet(pos, m);
            let is_killer = m == self.killers[ply][0] || m == self.killers[ply][1];

            // (6a) SEE pruning of losing captures. Read on the pre-move board: at
            // low depth a capture that loses material by static exchange is very
            // unlikely to be best, and the deeper we are the larger a loss we are
            // willing to write off. We only apply this in a prunable node, after
            // the first move, and never to a promotion (large forcing swing).
            if prunable
                && i > 0
                && depth <= 6
                && !quiet
                && m.move_type() != MoveType::Promotion
                && !see_ge(pos, m, -SEE_PRUNE_MARGIN * depth)
            {
                continue; // skip the badly-losing capture without searching it.
            }

            // Update the incremental accumulator for this move *before* making it
            // (we need the pre-move board to read the mover / captured piece).
            self.push_accumulator(pos, m);
            let undo = pos.make_move(m);
            // Does the move give check? (Read after making it: it's the opponent
            // who is now potentially in check.) We never reduce checking moves.
            let gives_check = pos.in_check();

            // (6b) Move-level pruning of clearly-hopeless *quiet* moves. Only in a
            // prunable node (non-PV, not in check, non-mate window), only once the
            // first move has already set a real `best_score`, and never for a move
            // that captures, promotes, gives check, or is a killer — those can
            // still swing the score. When a move qualifies we undo it and skip it.
            if prunable && i > 0 && quiet && !is_killer && !gives_check {
                // (6c) Late move pruning (move-count pruning): deep enough into the
                // ordered move list at low depth, the remaining quiets are almost
                // never best, so we stop searching them entirely. We prune more
                // aggressively (halve the move count) when the position is not
                // improving, since a fail-low there is more likely.
                const LMP: [usize; 7] = [0, 4, 8, 12, 20, 30, 45];
                if depth <= 6 {
                    let base = LMP[depth as usize];
                    let lmp_limit = if improving { base } else { (base + 1) / 2 };
                    if i >= lmp_limit {
                        pos.undo_move(m, undo);
                        self.pop_accumulator();
                        continue;
                    }
                }

                // (6c') History pruning: a quiet move with a clearly poor history
                // score at low depth is very unlikely to be best — skip it.
                if depth <= 3
                    && self.history[m.from_sq().index()][m.to_sq().index()] < -2000 * depth
                {
                    pos.undo_move(m, undo);
                    self.pop_accumulator();
                    continue;
                }

                // (6d) Frontier futility pruning: at very low depth, if the static
                // eval plus a generous per-ply margin still cannot reach alpha,
                // this quiet move is extremely unlikely to raise it, so skip it.
                if depth <= 4 && static_eval + 100 + 100 * depth <= alpha {
                    pos.undo_move(m, undo);
                    self.pop_accumulator();
                    continue;
                }
            }

            // (7) Principal Variation Search: the first move is searched with the
            // full window; every later move is first probed with a null window
            // (we only ask "is it worse than what we have?"). If that surprises us
            // by raising alpha, we re-search it properly with the full window.
            let score = if i == 0 {
                // The presumed best move: full window, and a PV child.
                -self.negamax(
                    pos,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    &mut child_pv,
                    is_pv,
                    true,
                    stop,
                )
            } else {
                // (7a) Late Move Reductions: unlikely-good late quiet moves are
                // first searched shallower with a null window. Only reduce moves
                // that are quiet, not a killer, deep enough, late enough in the
                // ordering, and neither escaping nor giving check.
                let reduction = if depth >= 3
                    && i >= 3
                    && quiet
                    && !is_killer
                    && !in_check
                    && !gives_check
                {
                    // A gentle log-based reduction, clamped so we never search
                    // below depth 1.
                    let mut r = (0.75 + (depth as f64).ln() * (i as f64).ln() / 2.25) as i32;
                    // History-based: reduce less for quiet moves with a good track
                    // record, more for ones with a bad one.
                    r -= self.history[m.from_sq().index()][m.to_sq().index()] / 8192;
                    // Reduce one extra ply when our position is not improving.
                    if !improving {
                        r += 1;
                    }
                    r.clamp(1, depth - 2)
                } else {
                    0
                };

                // (7b) Null-window probe, possibly at the reduced depth.
                let mut s = -self.negamax(
                    pos,
                    depth - 1 - reduction,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    &mut child_pv,
                    false,
                    true,
                    stop,
                );
                // If a *reduced* search unexpectedly beat alpha, it may have been
                // pruned too aggressively — re-search it at full depth (still a
                // null window) to get an honest verdict.
                if reduction > 0 && s > alpha {
                    s = -self.negamax(
                        pos,
                        depth - 1,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        &mut child_pv,
                        false,
                        true,
                        stop,
                    );
                }
                // If the null-window search lands inside the window, it might be a
                // new PV move: re-search with the full window to score it exactly.
                if s > alpha && s < beta {
                    s = -self.negamax(
                        pos,
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        &mut child_pv,
                        is_pv,
                        true,
                        stop,
                    );
                }
                s
            };

            pos.undo_move(m, undo);
            self.pop_accumulator();

            if self.stopped {
                self.repetitions.pop();
                return 0;
            }

            // (8) Track the best move and tighten alpha.
            if score > best_score {
                best_score = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    // Rebuild the PV: this move followed by the child's best line.
                    pv.clear();
                    pv.push(m);
                    pv.extend_from_slice(&child_pv);
                }
            }

            // Beta cutoff: this move is so good the opponent would never allow the
            // position, so we can stop searching siblings (fail-high). Reward the
            // move that cut off and penalize the earlier quiets that didn't.
            if alpha >= beta {
                if quiet {
                    self.record_killer(m, ply);
                    let bonus = (depth * depth).min(HISTORY_MAX);
                    self.update_history(m, bonus);
                    for &qm in &quiets_tried[..n_quiets] {
                        self.update_history(qm, -bonus);
                    }
                }
                break;
            }

            // Searched a quiet move that did not cut off: remember it so a later
            // cutoff can apply the history malus above.
            if quiet && n_quiets < quiets_tried.len() {
                quiets_tried[n_quiets] = m;
                n_quiets += 1;
            }
        }

        self.repetitions.pop();

        // (9) Store the result. The bound tells future probes how to trust it:
        // Lower if we failed high (>= beta), Upper if no move beat alpha, else
        // Exact. Mate scores are made relative to this node before storing.
        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > orig_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(
            key,
            best_move,
            score_to_tt(best_score, ply),
            depth as i16,
            bound,
        );

        best_score
    }

    // -----------------------------------------------------------------------
    // Quiescence search.
    // -----------------------------------------------------------------------

    /// At a leaf, keep searching only captures (and promotions) until the
    /// position is "quiet", so we never evaluate mid-trade. This tames the
    /// horizon effect: a static eval taken right after we grab a pawn — but
    /// before the recapture — would be badly wrong.
    fn quiescence(
        &mut self,
        pos: &mut Position,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        stop: &Arc<AtomicBool>,
    ) -> i32 {
        self.nodes += 1;
        if self.nodes.is_multiple_of(2048) && self.check_stop(stop) {
            self.stopped = true;
        }
        if self.stopped {
            return 0;
        }

        // The "stand-pat" score: we may simply choose not to capture. If even
        // doing nothing already beats beta, the opponent won't enter this line.
        let stand_pat = self.static_eval(pos);
        if ply as i32 >= MAX_QPLY {
            return stand_pat;
        }
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        // In check we must examine *every* evasion, so delta pruning is disabled
        // for this node (skipping a capture could skip the only way out of check).
        let in_check = pos.in_check();

        // Only consider forcing moves: captures, en-passant, and promotions.
        // SEE pruning: outside of check, a capture that loses material by static
        // exchange (`see_ge(pos, m, 0) == false`) is almost never worth resolving
        // in quiescence — it only deepens the tree to confirm the loss — so we
        // skip it. Promotions are always kept (their material swing is large and
        // forcing), and while in check we must try every evasion, so no capture is
        // pruned there.
        let moves = generate_legal(pos);
        let mut scored: Vec<(Move, i32)> = Vec::new();
        for m in &moves {
            if !is_capture_or_promo(pos, m) {
                continue;
            }
            if !in_check && m.move_type() != MoveType::Promotion && !see_ge(pos, m, 0) {
                continue; // skip a static-exchange-losing capture.
            }
            scored.push((m, mvv_lva(pos, m)));
        }

        let mut best_score = stand_pat;
        for i in 0..scored.len() {
            // Selection sort by MVV-LVA so the fattest captures come first.
            let mut best_idx = i;
            for j in (i + 1)..scored.len() {
                if scored[j].1 > scored[best_idx].1 {
                    best_idx = j;
                }
            }
            scored.swap(i, best_idx);
            let m = scored[i].0;

            // Delta pruning: if even winning this capture outright — the victim's
            // value on top of our stand-pat, plus a safety margin — still falls
            // short of alpha, the capture cannot improve our score, so skip it.
            // We never prune while in check (evasions are mandatory) nor a
            // promotion (its large value swing is not captured by the victim).
            const DELTA_MARGIN: i32 = 200;
            if !in_check && m.move_type() != MoveType::Promotion {
                let victim = captured_value(pos, m);
                if stand_pat + victim + DELTA_MARGIN < alpha {
                    continue;
                }
            }

            self.push_accumulator(pos, m);
            let undo = pos.make_move(m);
            let score = -self.quiescence(pos, -beta, -alpha, ply + 1, stop);
            pos.undo_move(m, undo);
            self.pop_accumulator();

            if self.stopped {
                return 0;
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                }
            }
            if alpha >= beta {
                break; // fail-high: opponent avoids this line.
            }
        }

        best_score
    }

    // -----------------------------------------------------------------------
    // Move ordering.
    // -----------------------------------------------------------------------

    /// Attach an ordering score to every move so the search can try the most
    /// promising first. The bands, highest first:
    ///
    ///  1. the TT / previous-PV move,
    ///  2. **winning / equal captures** (`see_ge(pos, m, 0)`) plus promotions,
    ///     ranked among themselves by MVV-LVA,
    ///  3. the two killer quiet moves,
    ///  4. ordinary quiet moves, ranked by the history heuristic,
    ///  5. **losing captures** (SEE `< 0`), pushed below every quiet so the
    ///     search only reaches them once the safe options are exhausted; ordered
    ///     among themselves by MVV-LVA so the least-bad is tried first.
    ///
    /// Splitting captures by SEE is the payoff: a queen grab that just hangs the
    /// queen back no longer jumps ahead of a solid developing move.
    fn order_moves(
        &self,
        pos: &Position,
        moves: &crate::movegen::MoveList,
        tt_move: Move,
        ply: usize,
    ) -> Vec<(Move, i32)> {
        // Ordering-score bands, chosen so they never overlap. History scores are
        // bounded well inside `[0, KILLER_BONUS)` in practice, and the losing-
        // capture band sits far below zero so it always sorts last.
        const TT_BONUS: i32 = 1_000_000;
        const GOOD_CAPTURE_BONUS: i32 = 100_000;
        const KILLER_BONUS: i32 = 80_000;
        const BAD_CAPTURE_BONUS: i32 = -100_000;

        let killer0 = self.killers[ply][0];
        let killer1 = self.killers[ply][1];

        let mut out: Vec<(Move, i32)> = Vec::with_capacity(moves.len());
        for m in moves {
            let score = if m == tt_move {
                // (1) The move the TT (or the previous iteration's PV) liked most.
                TT_BONUS
            } else if is_capture_or_promo(pos, m) {
                // (2)/(5) A capture or promotion. A queen (or capturing) promotion
                // is virtually always winning, so we treat any promotion and any
                // SEE-non-losing capture as a "good" capture; SEE-losing captures
                // drop to the bottom band. MVV-LVA is kept as the intra-band tie-
                // break so the fattest victim / cheapest attacker still leads.
                let good = m.move_type() == MoveType::Promotion || see_ge(pos, m, 0);
                if good {
                    GOOD_CAPTURE_BONUS + mvv_lva(pos, m)
                } else {
                    BAD_CAPTURE_BONUS + mvv_lva(pos, m)
                }
            } else if m == killer0 || m == killer1 {
                // (3) A quiet move that caused a cutoff at this ply before.
                KILLER_BONUS
            } else {
                // (4) Any other quiet move, scored by the history heuristic.
                let from = m.from_sq().index();
                let to = m.to_sq().index();
                self.history[from][to]
            };
            out.push((m, score));
        }
        out
    }

    /// Remember a quiet move that caused a beta cutoff at `ply`: it is often good
    /// in sibling positions too. We keep the two most recent, newest first.
    fn record_killer(&mut self, m: Move, ply: usize) {
        if self.killers[ply][0] != m {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = m;
        }
    }

    /// Nudge a quiet move's history toward `bonus` (positive to reward, negative
    /// for malus). The gravity term `h * |bonus| / HISTORY_MAX` pulls large scores
    /// back toward zero, so entries self-normalize and stay in
    /// `[-HISTORY_MAX, HISTORY_MAX]` without ever overflowing.
    fn update_history(&mut self, m: Move, bonus: i32) {
        let from = m.from_sq().index();
        let to = m.to_sq().index();
        let b = bonus.clamp(-HISTORY_MAX, HISTORY_MAX);
        let h = &mut self.history[from][to];
        *h += b - *h * b.abs() / HISTORY_MAX;
    }

    // -----------------------------------------------------------------------
    // Repetition & stop helpers.
    // -----------------------------------------------------------------------

    /// Has this exact position occurred earlier in the current search line since
    /// the last irreversible move? We scan back over the reversible window
    /// (`halfmove_clock` plies) in steps of two — only same-side-to-move
    /// positions can repeat. A single repetition is treated as a draw for
    /// simplicity, which is safe inside a search.
    fn is_repetition(&self, pos: &Position) -> bool {
        let key = pos.key();
        let reversible = pos.halfmove_clock() as usize;
        let stack = &self.repetitions;
        let mut i = stack.len();
        // Look back at most `reversible` plies; positions two apart share the
        // side to move.
        let mut count = 0;
        while i >= 2 && count < reversible {
            i -= 2;
            count += 2;
            if stack[i] == key {
                return true;
            }
        }
        false
    }

    /// Does the side to move have at least one knight, bishop, rook, or queen?
    /// Null-move pruning needs this: in a king-and-pawns endgame, passing can be
    /// *better* than any real move (zugzwang), which would make the null search
    /// lie. Requiring non-pawn material sidesteps that trap.
    fn has_non_pawn_material(&self, pos: &Position) -> bool {
        let us = pos.side_to_move();
        pos.pieces_cp(us, PieceType::Knight).any()
            || pos.pieces_cp(us, PieceType::Bishop).any()
            || pos.pieces_cp(us, PieceType::Rook).any()
            || pos.pieces_cp(us, PieceType::Queen).any()
    }

    /// Whether the search must stop now: the stop flag, the time deadline, or the
    /// node budget has been reached.
    fn check_stop(&self, stop: &Arc<AtomicBool>) -> bool {
        if stop.load(Ordering::Relaxed) {
            return true;
        }
        if let Some(limit) = self.node_limit
            && self.nodes >= limit
        {
            return true;
        }
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            return true;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Free helper functions.
// ---------------------------------------------------------------------------

/// Turn the time-control limits into an absolute stop time, or `None` when there
/// is no clock to respect (infinite / depth-only / nodes-only searches).
///
/// The allocation is deliberately simple and safe: use `movetime` verbatim if
/// given, otherwise spend about `time/20 + inc/2` of our remaining clock, clamped
/// so we never blow the whole clock or return an impossibly tiny budget.
fn compute_deadline(side: Color, limits: &SearchLimits) -> Option<Instant> {
    let now = Instant::now();

    if let Some(mt) = limits.movetime_ms {
        // A small safety margin so we report the move before the flag falls.
        let budget = mt.saturating_sub(5).max(1);
        return Some(now + Duration::from_millis(budget));
    }

    if limits.infinite {
        return None;
    }

    let (time, inc) = match side {
        Color::White => (limits.wtime, limits.winc.unwrap_or(0)),
        Color::Black => (limits.btime, limits.binc.unwrap_or(0)),
    };

    // No clock given for this side: rely on the depth/nodes caps (or MAX_PLY).
    let time = time?;

    // Roughly time/20 + inc/2, leave a safety margin, and clamp to a sane floor.
    let mut budget = time / 20 + inc / 2;
    // Never use more than ~40% of the remaining time on a single move.
    budget = budget.min(time.saturating_mul(2) / 5);
    // Keep at least a few ms so we always complete depth 1.
    budget = budget.max(5);
    Some(now + Duration::from_millis(budget))
}

/// Whether `score` represents a forced mate (as opposed to a normal eval).
fn is_mate_score(score: i32) -> bool {
    score.abs() >= MATE_IN_MAX
}

/// Adjust a mate score from "relative to the current node" into "relative to the
/// storing node" on its way *into* the TT: a mate looks one ply further away from
/// the root than from here, so we add `ply` to the distance.
fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        score + ply as i32
    } else if score <= -MATE_IN_MAX {
        score - ply as i32
    } else {
        score
    }
}

/// The inverse of [`score_to_tt`]: pull a stored mate score back into the current
/// node's frame on the way *out* of the TT.
fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        score - ply as i32
    } else if score <= -MATE_IN_MAX {
        score + ply as i32
    } else {
        score
    }
}

/// Does `m` capture an enemy piece (including en passant)?
fn is_capture(pos: &Position, m: Move) -> bool {
    if m.move_type() == MoveType::EnPassant {
        return true;
    }
    pos.piece_at(m.to_sq()).is_some()
}

/// Is `m` a "forcing" move worth resolving in quiescence — a capture or a
/// promotion?
fn is_capture_or_promo(pos: &Position, m: Move) -> bool {
    is_capture(pos, m) || m.move_type() == MoveType::Promotion
}

/// Is `m` a quiet move (neither a capture nor a promotion)? Only quiet cutoffs
/// feed the killer / history tables.
fn is_quiet(pos: &Position, m: Move) -> bool {
    !is_capture_or_promo(pos, m)
}

/// The centipawn value of the piece `m` captures (a pawn for en passant, 0 for a
/// non-capturing move). Used by quiescence delta pruning to bound the best-case
/// material gain of a capture.
fn captured_value(pos: &Position, m: Move) -> i32 {
    if m.move_type() == MoveType::EnPassant {
        return piece_value(PieceType::Pawn);
    }
    match pos.piece_at(m.to_sq()) {
        Some(p) => piece_value(p.piece_type),
        None => 0,
    }
}

/// Most-Valuable-Victim / Least-Valuable-Attacker score for a capture: prefer
/// grabbing a big piece with a small one. En-passant always takes a pawn.
fn mvv_lva(pos: &Position, m: Move) -> i32 {
    let attacker = pos
        .piece_at(m.from_sq())
        .map(|p| p.piece_type)
        .unwrap_or(PieceType::Pawn);
    let victim = if m.move_type() == MoveType::EnPassant {
        PieceType::Pawn
    } else {
        match pos.piece_at(m.to_sq()) {
            Some(p) => p.piece_type,
            None => PieceType::Pawn, // a bare promotion; treat as a pawn victim.
        }
    };
    // Victim dominates; subtract a small slice of the attacker so cheaper
    // attackers of the same victim sort first.
    piece_value(victim) * 16 - piece_value(attacker)
}

/// Print a single UCI `info` line for a completed iteration. Mate scores are
/// reported as `mate N` (moves), everything else as `cp` (centipawns).
fn print_info(depth: u32, score: i32, nodes: u64, elapsed: Duration, pv: &[Move]) {
    let ms = elapsed.as_millis() as u64;
    let nps = if ms > 0 { nodes * 1000 / ms } else { nodes };

    let mut pv_str = String::new();
    for (i, m) in pv.iter().enumerate() {
        if i > 0 {
            pv_str.push(' ');
        }
        pv_str.push_str(&m.to_string());
    }

    if is_mate_score(score) {
        // Convert plies-to-mate into moves-to-mate, keeping the sign.
        let mate_plies = MATE - score.abs();
        let mate_moves = (mate_plies + 1) / 2;
        let signed = if score > 0 { mate_moves } else { -mate_moves };
        println!(
            "info depth {depth} score mate {signed} nodes {nodes} nps {nps} time {ms} pv {pv_str}"
        );
    } else {
        println!(
            "info depth {depth} score cp {score} nodes {nodes} nps {nps} time {ms} pv {pv_str}"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    /// Run a fixed-depth search with no time cap and an already-clear stop flag,
    /// so results are fully deterministic and fast. Used by every test below.
    fn search_depth(fen: &str, depth: u32) -> SearchResult {
        let pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("bad fen {fen}: {e}"));
        let mut searcher = Searcher::new(16);
        let limits = SearchLimits {
            depth: Some(depth),
            ..SearchLimits::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        searcher.search(&pos, &limits, &stop)
    }

    /// Is `m` a legal move in `pos`?
    fn is_legal(pos: &mut Position, m: Move) -> bool {
        generate_legal(pos).contains(m)
    }

    /// Play `m` and report whether the resulting position is checkmate (the side
    /// to move is in check and has no legal reply).
    fn move_is_checkmate(fen: &str, m: Move) -> bool {
        let mut pos = Position::from_fen(fen).unwrap();
        assert!(is_legal(&mut pos, m), "move {m} is not legal in {fen}");
        let undo = pos.make_move(m);
        let mated = pos.in_check() && generate_legal(&mut pos).is_empty();
        pos.undo_move(m, undo);
        mated
    }

    #[test]
    fn startpos_returns_legal_reasonable_move() {
        let res = search_depth("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4);
        // A real move must come back.
        assert!(!res.best_move.is_none(), "startpos must return a move");
        // It must be legal.
        let mut pos = Position::startpos();
        assert!(is_legal(&mut pos, res.best_move), "best move must be legal");
        // The start position is balanced, so the score must be small.
        assert!(res.score.abs() < 100, "startpos score should be near 0, got {}", res.score);
    }

    #[test]
    fn finds_mate_in_one() {
        // White: Kg6, Qh1, Black: Kg8. Qh1-a8 is mate (Qa8#): the queen checks
        // along the 8th rank while the king covers f7/g7/h7, so the black king
        // has no escape. (Verified: it is the unique mate here.)
        let fen = "6k1/8/6K1/8/8/8/8/7Q w - - 0 1";
        let res = search_depth(fen, 2);
        // The score must read as a mate.
        assert!(
            res.score >= MATE - 10,
            "mate-in-1 must return a mate score, got {}",
            res.score
        );
        // And the move it returns must actually deliver checkmate.
        assert!(
            move_is_checkmate(fen, res.best_move),
            "returned move {} is not checkmate in {fen}",
            res.best_move
        );
    }

    #[test]
    fn finds_mate_in_two() {
        // A clean, legal mate-in-2: 1.Rf7! (threatening Qe8#) and Black cannot
        // parry both the back-rank and the queen. The score at depth 4 (= 2 full
        // moves) must read as a mate. Rf6-f7 is the key move.
        let fen = "r5rk/5p1p/5R2/4Q3/8/8/7P/7K w - - 0 1";
        let res = search_depth(fen, 4);
        assert!(
            res.score >= MATE - 10,
            "mate-in-2 must return a mate score at depth 4, got {}",
            res.score
        );
        // The first move must be legal.
        let mut pos = Position::from_fen(fen).unwrap();
        assert!(is_legal(&mut pos, res.best_move));
    }

    #[test]
    fn wins_a_free_queen() {
        // White to move; the black queen on d5 is completely undefended and can be
        // taken for free by the pawn on e4. A depth-3 search must grab it.
        let fen = "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1";
        let res = search_depth(fen, 3);
        assert_eq!(
            res.best_move,
            Move::normal(Square::E4, Square::D5),
            "should capture the free queen, played {} instead",
            res.best_move
        );
        // The position starts near -900 for White (down a queen); grabbing the
        // queen must swing the score to clearly positive.
        assert!(res.score > 50, "winning a queen should score positive, got {}", res.score);
    }

    #[test]
    fn finds_a_deeper_mate() {
        // A basic two-rook "ladder" mate: with two rooks and the enemy king cut
        // off, White forces mate in a few moves. This checks that the new pruning
        // (reverse-futility / null-move / LMR) never hides a forced mate — the
        // score must read as a mate and the first move must be legal.
        let fen = "4k3/8/8/8/8/8/R7/1R2K3 w - - 0 1";
        let res = search_depth(fen, 8);
        assert!(
            res.score >= MATE - 20,
            "a forced-mate position must return a mate score, got {}",
            res.score
        );
        let mut pos = Position::from_fen(fen).unwrap();
        assert!(is_legal(&mut pos, res.best_move), "mate move must be legal");
    }

    #[test]
    fn wins_a_piece_through_pruning() {
        // A simple tactic: the black rook on h4 is completely undefended, so the
        // white bishop on f2 grabs it for free with Bxh4. Reverse-futility /
        // null-move / LMR must not blind the search to this clean win of material.
        let fen = "4k3/8/8/8/7r/8/5B2/4K3 w - - 0 1";
        let res = search_depth(fen, 6);
        assert_eq!(
            res.best_move,
            Move::normal(Square::F2, Square::H4),
            "should capture the free rook, played {} instead",
            res.best_move
        );
        // Winning a whole rook for nothing must swing the score clearly positive
        // (well above a pawn — the exact number tapers in a bare endgame).
        assert!(res.score > 250, "winning a rook should score positive, got {}", res.score);
    }

    #[test]
    fn midgame_search_returns_legal_move_without_exploding() {
        // A rich middlegame position searched to a fixed depth. The only thing we
        // assert is that the search terminates cleanly and returns a legal move —
        // i.e. the new pruning does not panic, loop, or return garbage.
        let fen = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 0 1";
        let res = search_depth(fen, 6);
        assert!(!res.best_move.is_none(), "must return a move");
        let mut pos = Position::from_fen(fen).unwrap();
        assert!(is_legal(&mut pos, res.best_move), "best move must be legal");
        assert!(res.nodes > 0, "search must have visited nodes");
    }

    #[test]
    fn search_is_deterministic() {
        let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1";
        let a = search_depth(fen, 4);
        let b = search_depth(fen, 4);
        assert_eq!(a.best_move, b.best_move, "same position must give same move");
        assert_eq!(a.score, b.score, "same position must give same score");
    }

    #[test]
    fn checkmated_position_scores_mated() {
        // Fool's-mate final position: white is checkmated, White to move.
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
        let res = search_depth(fen, 3);
        // No legal move: the searcher returns NONE and a mated score.
        assert!(res.best_move.is_none(), "checkmated side has no move");
        assert!(
            res.score <= -(MATE - 10),
            "a mated position should score very negative, got {}",
            res.score
        );
    }

    #[test]
    fn stalemate_position_scores_zero() {
        // Classic stalemate: Black to move, king on a8 has no legal move and is
        // NOT in check. Score must be 0 and no move returned.
        let fen = "k7/8/1Q6/2K5/8/8/8/8 b - - 0 1";
        let res = search_depth(fen, 3);
        assert!(res.best_move.is_none(), "stalemated side has no move");
        assert_eq!(res.score, 0, "stalemate must score 0, got {}", res.score);
    }

    #[test]
    fn aspiration_windows_still_find_mate_in_two() {
        // The same mate-in-2 as above, but searched deep enough (>= depth 5) to
        // exercise the aspiration-window path. A narrow window must never hide the
        // mate: the score must still read as a mate and the key move stay legal.
        let fen = "r5rk/5p1p/5R2/4Q3/8/8/7P/7K w - - 0 1";
        let res = search_depth(fen, 7);
        assert!(
            res.score >= MATE - 10,
            "aspiration search must still return the mate score, got {}",
            res.score
        );
        let mut pos = Position::from_fen(fen).unwrap();
        assert!(is_legal(&mut pos, res.best_move), "mate move must be legal");
    }

    #[test]
    fn aspiration_deep_search_matches_full_window_on_a_tactic() {
        // Winning the free rook (Bxh4) must still be found once the aspiration
        // window is in play (depth 8 >= 5), confirming the narrow window does not
        // clip a clearly-winning capture.
        let fen = "4k3/8/8/8/7r/8/5B2/4K3 w - - 0 1";
        let res = search_depth(fen, 8);
        assert_eq!(
            res.best_move,
            Move::normal(Square::F2, Square::H4),
            "aspiration search should still grab the free rook, played {} instead",
            res.best_move
        );
    }

    #[test]
    fn deep_midgame_search_returns_legal_move_without_panic() {
        // A couple of rich middlegame positions searched to depth 8 — deep enough
        // that aspiration windows, LMP, futility and delta pruning all fire. The
        // search must terminate cleanly and return a legal, non-null move each time
        // (i.e. none of the new pruning panics, loops, or returns garbage).
        let fens = [
            "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete
        ];
        for fen in fens {
            let res = search_depth(fen, 8);
            assert!(!res.best_move.is_none(), "must return a move for {fen}");
            let mut pos = Position::from_fen(fen).unwrap();
            assert!(
                is_legal(&mut pos, res.best_move),
                "best move must be legal for {fen}"
            );
            assert!(res.nodes > 0, "search must have visited nodes for {fen}");
        }
    }
}
