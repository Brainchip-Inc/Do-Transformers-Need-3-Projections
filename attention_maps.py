"""
Render sample self-attention maps for the synthetic tasks (appendix figure).

For each attention variant and task, trains a small encoder, then visualizes the
last-layer self-attention map (softmax weights, averaged over heads) for one fixed
sample input. Produces a variants x tasks grid of heatmaps.

Reuses the model and variants from synthetic_tasks.py. Expected patterns:
  COPY / SUB  -> diagonal (each position attends to itself)
  REVERSE     -> anti-diagonal (position i attends to n-1-i)
  SWAP        -> two off-diagonal blocks
  SORT        -> content-dependent routing
The symmetric variants (Q=K..., Q=K=V) produce visibly symmetric maps unless the
(X)+ positional injection is applied.

Usage:
    conda run -n kv python attention_maps.py --device cuda:2
    conda run -n kv python attention_maps.py --device cpu --epochs 8   # quicker
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_tasks import Encoder, ModelCfg, make_dataset, VARIANTS, TASKS

# tasks to show as columns (SUB omitted — visually identical diagonal to COPY)
SHOW_TASKS = ["REVERSE", "SORT", "SWAP", "COPY"]


def train(variant, task, seed, device, epochs, n_train, batch, d, L, H, T):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x, y = make_dataset(n_train, T, task, g)
    model = Encoder(ModelCfg(n_embd=d, n_layer=L, n_head=H, seq_len=T, variant=variant)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_train, generator=g)
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            _, loss = model(x[idx].to(device), y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


@torch.no_grad()
def attention_map(model, sample, device):
    """Last-layer self-attention (softmax), averaged over heads, for one sample [T]."""
    model.eval()
    ids = sample.unsqueeze(0).to(device)                 # [1, T]
    oh = torch.nn.functional.one_hot(ids, 10).float()
    pos = torch.arange(ids.size(1), device=device)
    h = model.embed(oh) + model.pos_emb(pos)[None]
    for blk in model.blocks[:-1]:
        h = blk(h)
    last = model.blocks[-1]
    _, attn = last.attn(last.ln1(h), return_attn=True)   # [1, H, T, T]
    return attn[0].mean(0).cpu().numpy()                 # [T, T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=16)   # small T -> legible maps
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="attention_maps")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  config d{args.embd} L{args.layers} H{args.heads} "
          f"T{args.seqlen}  epochs={args.epochs}", flush=True)

    # one fixed sample input per task (same across variants for comparability)
    g = torch.Generator().manual_seed(args.seed + 999)
    samples = {t: torch.randint(0, 10, (args.seqlen,), generator=g) for t in SHOW_TASKS}

    variants = list(VARIANTS)
    maps = {}
    for task in SHOW_TASKS:
        for variant in variants:
            model = train(variant, task, args.seed, device, args.epochs,
                          args.n_train, args.batch_size, args.embd, args.layers,
                          args.heads, args.seqlen)
            maps[(variant, task)] = attention_map(model, samples[task], device)
            print(f"  {task:8} {variant:11} done", flush=True)

    # ---- grid: rows = variants, cols = tasks ----
    nr, nc = len(variants), len(SHOW_TASKS)
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.3 * nr))
    im = None
    for r, variant in enumerate(variants):
        for c, task in enumerate(SHOW_TASKS):
            ax = axes[r, c]
            im = ax.imshow(maps[(variant, task)], cmap="viridis", vmin=0,
                           interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(task, fontsize=11)
            if c == 0:
                ax.set_ylabel(variant, fontsize=10, rotation=90, labelpad=8)

    fig.suptitle(f"Last-layer self-attention maps (heads averaged; d={args.embd}, "
                 f"L={args.layers}, H={args.heads}, T={args.seqlen})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])
    cax = fig.add_axes([0.945, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cax, label="attention weight")
    fig.savefig(f"{args.out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    print(f"Wrote {args.out}.png / .pdf", flush=True)


if __name__ == "__main__":
    main()
