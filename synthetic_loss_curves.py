"""
Generate training-loss curves for the synthetic tasks (appendix figure).

Trains all six attention variants on each of the five tasks, logs per-step training
loss over an extended (default 10-epoch) budget, averages over seeds, and renders a
5-panel figure (one panel per task, six variant curves each). Reuses the exact model
and variants from synthetic_tasks.py.

This is a visualization-only run (longer than the 2-epoch Table-1 budget) at a single
representative config, so convergence differences between variants are legible.

Usage:
    conda run -n kv python synthetic_loss_curves.py --device cuda:3
    conda run -n kv python synthetic_loss_curves.py --device cpu --seeds 1   # quick
"""

import os
import csv
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_tasks import (Encoder, ModelCfg, make_dataset, lr_lambda_factory,
                             VARIANTS, TASKS)

# Okabe-Ito colorblind-safe hues (6 of 8), + distinct line styles as secondary
# encoding so variant identity never relies on color alone (print / CVD safe).
STYLE = {
    "QKV":        ("#0072B2", "-"),    # blue
    "Q=K!=V":     ("#E69F00", "--"),   # orange
    "(Q=K!=V)+":  ("#009E73", "-."),   # bluish green
    "Q!=K=V":     ("#CC79A7", ":"),    # reddish purple
    "Q=K=V":      ("#D55E00", "--"),   # vermillion
    "(Q=K=V)+":   ("#56B4E9", "-."),   # sky blue
}


def loss_curve(variant, task, seed, device, epochs, n_train, batch, d, L, H, T):
    """Train one model, return the per-step training loss list."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x, y = make_dataset(n_train, T, task, g)
    cfg = ModelCfg(n_embd=d, n_layer=L, n_head=H, seq_len=T, variant=variant)
    model = Encoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = math.ceil(n_train / batch) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(steps, 0.05))

    losses = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_train, generator=g)
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx].to(device), y[idx].to(device)
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
    return losses


def rolling(a, w):
    if w <= 1 or len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=64)
    ap.add_argument("--out", type=str, default="synthetic_loss_curves")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  config: d{args.embd} L{args.layers} H{args.heads} "
          f"T{args.seqlen}  epochs={args.epochs}  seeds={args.seeds}", flush=True)

    steps_per_epoch = math.ceil(args.n_train / args.batch_size)

    # curves[(variant, task)] = mean per-step loss over seeds
    curves = {}
    for task in TASKS:
        for variant in VARIANTS:
            runs = []
            for s in range(args.seeds):
                runs.append(loss_curve(variant, task, s, device, args.epochs,
                                       args.n_train, args.batch_size,
                                       args.embd, args.layers, args.heads, args.seqlen))
            m = min(len(r) for r in runs)
            curves[(variant, task)] = np.mean([r[:m] for r in runs], axis=0)
            print(f"  {task:8} {variant:11} done (final loss "
                  f"{curves[(variant, task)][-1]:.3f})", flush=True)

    # ---- save raw curves to CSV ----
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "variant", "step", "loss_mean"])
        for (variant, task), c in curves.items():
            for step, v in enumerate(c):
                w.writerow([task, variant, step, f"{v:.5f}"])

    # ---- plot: 2x3 grid, 5 task panels + 1 legend cell ----
    win = max(1, steps_per_epoch // 3)   # light smoothing for legibility
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True)
    axes = axes.ravel()
    for ax, task in zip(axes, TASKS):
        for variant in VARIANTS:
            c = rolling(curves[(variant, task)], win)
            xs = np.arange(len(c)) / steps_per_epoch
            color, ls = STYLE[variant]
            ax.plot(xs, c, color=color, linestyle=ls, linewidth=1.8, label=variant)
        ax.set_title(task, fontsize=11)
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    # legend in the 6th cell
    legend_ax = axes[5]
    legend_ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", frameon=False,
                     fontsize=11, title="Attention variant")

    fig.suptitle(f"Synthetic-task training loss  (d={args.embd}, L={args.layers}, "
                 f"H={args.heads}, T={args.seqlen}; mean of {args.seeds} seeds)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{args.out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    print(f"Wrote {args.out}.png / .pdf / .csv", flush=True)


if __name__ == "__main__":
    main()
