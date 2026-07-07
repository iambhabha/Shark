#!/usr/bin/env python
"""
GPU trainer for Mythos's NNUE, using PyTorch (CUDA).

It reads the self-play data produced by `datagen` (lines `FEN | score_cp | result`)
and trains a (768 -> 256)x2 -> 1 perspective network whose feature convention and
on-disk format EXACTLY match the Rust inference in `src/nnue.rs`, so the exported
`.nnue` file loads straight into the engine.

Feature index (must match src/nnue.rs):
    oriented_sq = sq            if perspective == White
                = sq ^ 56       if perspective == Black
    friendly    = (piece_color == perspective)
    idx = (0 if friendly else 1) * 384 + piece_type_index * 64 + oriented_sq
    piece_type_index: P=0 N=1 B=2 R=3 Q=4 K=5

.nnue file format (little-endian):
    u32 magic = 0x4E4E5545 ("NNUE"), u32 hidden = 256, u32 num_features = 768,
    w1 [256*768] f32 (row-major j*768+f), b1 [256] f32, w2 [512] f32, b2 f32

Usage:
    python train_nnue.py <data_file> <out.nnue> [--epochs 40] [--lr 1e-3]
                         [--batch 16384] [--lambda 0.7] [--val 0.02]
"""

import argparse
import math
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 256
NUM_FEATURES = 768
PAD = NUM_FEATURES          # padding feature index (contributes zero)
MAX_FEATS = 32              # at most 32 pieces on the board
SCALE = 400.0               # must match src/nnue.rs SCALE
MAGIC = 0x4E4E5545

PT = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def parse_fen(fen):
    """Return (list of (is_white, pt_index, square), stm_is_white). square 0=a1..63=h8."""
    parts = fen.split()
    board, stm = parts[0], parts[1]
    pieces = []
    rank, file = 7, 0
    for ch in board:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            is_white = ch.isupper()
            pieces.append((is_white, PT[ch.lower()], rank * 8 + file))
            file += 1
    return pieces, (stm == "w")


def feature_row(pieces, persp_white, out_row):
    """Fill out_row (len MAX_FEATS, prefilled with PAD) with feature indices for a perspective."""
    for i, (is_white, pt, sq) in enumerate(pieces):
        oriented = sq if persp_white else (sq ^ 56)
        friendly = (is_white == persp_white)
        out_row[i] = (0 if friendly else 1) * 384 + pt * 64 + oriented


def load_dataset(path, lam):
    """Parse the data file into padded feature-index arrays + targets."""
    print(f"Reading {path} ...", flush=True)
    with open(path, "r") as f:
        lines = [ln for ln in f if ln and ln[0] != "#" and "|" in ln]
    n = len(lines)
    print(f"  {n:,} samples; extracting features ...", flush=True)

    stm = np.full((n, MAX_FEATS), PAD, dtype=np.int32)
    nstm = np.full((n, MAX_FEATS), PAD, dtype=np.int32)
    y = np.empty(n, dtype=np.float32)

    t0 = time.time()
    w = 0
    for ln in lines:
        a, b, c = ln.split("|")
        fen = a.strip()
        try:
            score = float(b.strip())
            result = float(c.strip())
            pieces, stm_white = parse_fen(fen)
        except Exception:
            continue
        if len(pieces) > MAX_FEATS:
            continue
        feature_row(pieces, stm_white, stm[w])
        feature_row(pieces, not stm_white, nstm[w])
        # Target: blend the (side-to-move) search score and the game result.
        y[w] = lam * (1.0 / (1.0 + math.exp(-score / SCALE))) + (1.0 - lam) * result
        w += 1
        if w % 200000 == 0:
            print(f"    {w:,}/{n:,} ({time.time()-t0:.0f}s)", flush=True)

    stm, nstm, y = stm[:w], nstm[:w], y[:w]
    print(f"  parsed {w:,} usable samples in {time.time()-t0:.0f}s", flush=True)
    return stm, nstm, y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # +1 embedding row for the PAD index (kept at zero, ignored in the sum).
        self.ft = nn.EmbeddingBag(NUM_FEATURES + 1, HIDDEN, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.out = nn.Linear(2 * HIDDEN, 1)
        nn.init.uniform_(self.ft.weight, -0.1, 0.1)
        with torch.no_grad():
            self.ft.weight[PAD].zero_()
        nn.init.uniform_(self.out.weight, -0.1, 0.1)
        nn.init.zeros_(self.out.bias)

    def forward(self, stm_idx, nstm_idx):
        acc_stm = (self.ft(stm_idx) + self.ft_bias).clamp(0.0, 1.0)
        acc_nstm = (self.ft(nstm_idx) + self.ft_bias).clamp(0.0, 1.0)
        x = torch.cat([acc_stm, acc_nstm], dim=1)
        return self.out(x).squeeze(1)


def export_nnue(model, path):
    ft_w = model.ft.weight.detach().cpu().numpy()          # [769, 256]
    b1 = model.ft_bias.detach().cpu().numpy()              # [256]
    out_w = model.out.weight.detach().cpu().numpy()[0]     # [512]
    b2 = float(model.out.bias.detach().cpu().numpy()[0])
    # w1[j*768 + f] = ft_w[f][j]  -> transpose the real-feature rows.
    w1 = ft_w[:NUM_FEATURES, :].T.reshape(-1)              # [256*768], order j*768+f
    with open(path, "wb") as fp:
        fp.write(struct.pack("<III", MAGIC, HIDDEN, NUM_FEATURES))
        fp.write(w1.astype("<f4").tobytes())
        fp.write(b1.astype("<f4").tobytes())
        fp.write(out_w.astype("<f4").tobytes())
        fp.write(struct.pack("<f", b2))
    print(f"  wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.02)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training on CPU (slow).", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}", flush=True)

    stm, nstm, y = load_dataset(args.data, args.lam)
    n = len(y)
    if n == 0:
        print("No samples parsed — aborting.", flush=True)
        sys.exit(1)

    # Keep the whole dataset on the GPU if it fits (it easily does here).
    stm_t = torch.from_numpy(stm).to(device, dtype=torch.long)
    nstm_t = torch.from_numpy(nstm).to(device, dtype=torch.long)
    y_t = torch.from_numpy(y).to(device)

    n_val = int(n * args.val)
    perm = torch.randperm(n, device=device)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_idx[torch.randperm(train_idx.numel(), device=device)]
        total, seen = 0.0, 0
        t0 = time.time()
        for i in range(0, order.numel(), args.batch):
            b = order[i:i + args.batch]
            pred = model(stm_t[b], nstm_t[b])
            loss = F.mse_loss(torch.sigmoid(pred), y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * b.numel()
            seen += b.numel()
        # Validation loss.
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for i in range(0, val_idx.numel(), args.batch):
                b = val_idx[i:i + args.batch]
                pred = model(stm_t[b], nstm_t[b])
                vloss += F.mse_loss(torch.sigmoid(pred), y_t[b]).item() * b.numel()
            vloss = vloss / max(1, val_idx.numel())
        print(
            f"epoch {epoch:3d}/{args.epochs}  train {total/max(1,seen):.5f}  "
            f"val {vloss:.5f}  ({time.time()-t0:.1f}s)",
            flush=True,
        )
        if epoch % 5 == 0 or epoch == args.epochs:
            export_nnue(model, args.out)

    export_nnue(model, args.out)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
