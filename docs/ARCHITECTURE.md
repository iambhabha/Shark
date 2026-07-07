# Mythos architecture

A complete tour of how Mythos works, subsystem by subsystem. If you want to
*train* the neural network specifically, see **[TRAINING.md](TRAINING.md)**; to
make the engine stronger, see **[IMPROVING.md](IMPROVING.md)**.

## The big picture

A chess engine answers one question — *"what is the best move?"* — by
**searching** the tree of possible moves and **evaluating** the positions at the
leaves. Everything in Mythos serves that loop:

```
   UCI command / browser move
              │
              ▼
        ┌───────────┐   asks for legal moves   ┌────────────────┐
        │  Search   │ ───────────────────────▶ │ Move generation │
        │           │ ◀─────────────────────── │  (legal, fast)  │
        │  (picks   │      list of moves        └────────────────┘
        │   a move) │
        │           │   scores leaf positions   ┌────────────────┐
        │           │ ───────────────────────▶ │  Evaluation     │
        │           │ ◀─────────────────────── │  HCE  or  NNUE  │
        └───────────┘         a number          └────────────────┘
              │
              ▼
         best move → bestmove / plays in the browser
```

Everything below the search (board, move generation, evaluation) is pure,
stateless machinery. The search is where the "thinking" happens.

---

## 1. Board representation — `types.rs`, `bitboard.rs`, `position.rs`

**Bitboards.** A `Bitboard` is a single `u64`; bit *i* is set when square *i*
(0 = a1 … 63 = h8) is "in the set". "All white pawns", "all squares a rook
attacks", "the squares between two pieces" — all are bitboards. Rust gives the
hardware bit instructions for free (`count_ones` = popcount,
`trailing_zeros` = find-first-bit), so bit manipulation is fast and portable.

**Position** (`position.rs`) holds the full game state: a piece array + bitboards
per color and per piece type, side to move, castling rights, the en-passant
square, move clocks, and the **Zobrist hash** (`zobrist.rs`) — a 64-bit fingerprint
of the position, updated incrementally as pieces move. The hash is what the
transposition table and repetition detection key on.

`make_move` / `undo_move` apply and reverse a move, updating every bitboard, the
piece array, and the hash. They handle the tricky cases: captures, en passant,
castling (king + rook move together), and promotions.

## 2. Move generation — `attacks.rs`, `movegen.rs`

**Attacks.** Knight/king/pawn attacks are simple precomputed tables. Sliding
pieces (bishop/rook/queen) use **magic bitboards**: a perfect-hash trick that
turns "given the occupied squares, what does this rook attack?" into one
multiply + shift + table lookup.

**Legal move generation** (`movegen.rs`) is *fast and exact*. Instead of making
every move and checking whether the king is left in check, it computes the
**checkers** and **pinned pieces** once, then decides legality directly:
- If the king is in double check, only king moves are legal.
- In single check, moves must capture the checker or block it.
- A pinned piece may only move along its pin ray.
- Everything else is legal without any make/undo.

This is verified by **`perft`** (`perft.rs`) — it counts the exact number of leaf
nodes at each depth and compares against published values (start position at
depth 6 = `119,060,324`). If a single move-generation rule is wrong, perft
mismatches. Perft is the engine's correctness bedrock.

## 3. Search — `search.rs` (the brain)

The search is **negamax alpha-beta with iterative deepening**. It searches to
depth 1, then 2, then 3… reusing what it learned each time, until it runs out of
time. Key components:

- **Alpha-beta pruning** — skips branches that can't affect the result. The single
  most important idea; it turns an impossible search into a feasible one.
- **Principal Variation Search (PVS)** — assumes the first (best-ordered) move is
  best and searches the rest with a cheap "null window", re-searching only if one
  surprises us.
- **Quiescence search** — at the leaves, keep searching *captures* until the
  position is quiet, so the engine doesn't stop in the middle of a trade (the
  "horizon effect").
- **Transposition table** (`tt.rs`) — a big hash map remembering results of
  positions already searched, so identical positions (reached different ways)
  aren't re-searched.
- **Move ordering** — searching the best move first makes alpha-beta far more
  effective. Order: TT move → good captures (by **SEE**, `see.rs`) → killer moves →
  history heuristic → the rest.

**Pruning & reductions** (why it can look so deep, so fast):

| Technique | Idea |
| --- | --- |
| Null-move pruning | "If I skip my turn and I'm *still* winning, this line is too good — prune." |
| Late move reductions (LMR) | Search unlikely (late-ordered) moves shallower first. |
| Late move pruning | At low depth, just skip the very last quiet moves. |
| Futility / reverse futility | If the static score is far above/below the window, prune. |
| Delta pruning | In quiescence, skip captures that can't raise alpha. |
| Aspiration windows | Search around the previous score with a narrow window; widen on failure. |

**Time management** decides how long to think from the clock/increment or a
`movetime`. The search runs on its **own thread**, so the engine still responds to
`stop` and `isready` while thinking.

## 4. Evaluation — `eval.rs`, `nnue.rs`

The evaluation turns a leaf position into a single number (centipawns, from the
side-to-move's view). Mythos has two:

**Hand-crafted evaluation (HCE, `eval.rs`)** — always available, no files needed.
It is **tapered** (blends a midgame and endgame score by how much material is
left) and sums: material + piece-square tables (PeSTO), mobility, king safety,
passed pawns, pawn structure (doubled/isolated/backward), bishop pair, rook on
open file, and a tempo bonus.

**NNUE (`nnue.rs`)** — an optional neural network loaded from `mythos.nnue`.
It is a `768 → 256 → 1` **perspective network**. Its first layer (the
"accumulator") is maintained **incrementally**: instead of recomputing it from
scratch at every leaf, only the features of the piece that moved are updated on
make/undo. That is the "efficiently updatable" in NNUE. See
**[TRAINING.md](TRAINING.md)** for how a net is produced and the exact format.

The search calls one `static_eval(pos)` that dispatches to the NNUE if a net is
loaded, otherwise the HCE.

## 5. Interfaces — `uci.rs`, `bin/webserver.rs`, `web/index.html`

**UCI** (`uci.rs`) is the text protocol every chess GUI speaks: the GUI sends
`position …` and `go …`; the engine replies with `info …` lines and a `bestmove`.
The search runs on a worker thread so `stop`/`isready` stay responsive. Options
like `Hash`, `UseNNUE`, and `EvalFile` are set with `setoption`.

**Browser UI.** `bin/webserver.rs` is a tiny standard-library HTTP server (no
dependencies) that serves `web/index.html` and one JSON endpoint. It is
**stateless** — the browser sends the full move list each time; the server
rebuilds the position, is the referee (via the crate's legal-move generator), and
is the opponent (via the search). The page is self-contained vanilla JS with a
click-to-move board, promotion picker, undo, board flip, and a size slider.

## 6. Tooling — `bin/selfplay.rs`, `bin/datagen.rs`

**`selfplay`** is Mythos's own miniature "Fishtest": it plays two engine binaries
against each other over many games (varied openings, colors swapped), referees
with the crate, and reports the score and estimated Elo gap with an error margin.
**Every strength change is validated here before it is kept** — see
[CONTRIBUTING.md](../CONTRIBUTING.md).

**`datagen`** generates NNUE training data by self-play. See
[TRAINING.md](TRAINING.md).

## How it all fits together

1. A GUI or the browser sends a position and a `go`.
2. The **search** runs iterative deepening on its thread.
3. At each node it asks **move generation** for legal moves and orders them.
4. It recurses (with all the pruning), and at the leaves calls **evaluation**.
5. The **transposition table** caches results across the whole search.
6. When time runs out, it returns the best move from the last completed depth.

Every subsystem is independently testable — which is why the whole thing can be
trusted: `perft` proves move generation, unit tests prove the pieces, and
`selfplay` proves that each change actually makes the engine *stronger*.
