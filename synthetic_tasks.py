"""
Synthetic-reasoning benchmark for QKV projection-sharing variants (paper Table 1).

Reproduces the five synthetic sequence tasks (REVERSE / SORT / SUB / SWAP / COPY)
across six attention variants:

    QKV          separate Q, K, V (baseline)
    Q=K!=V       shared query/key, separate value        (symmetric map K Kᵀ)
    (Q=K!=V)+    same, plus 2D positional injection       (breaks symmetry)
    Q!=K=V       separate query, shared key/value         (the paper's headline variant)
    Q=K=V        single projection for all three          (symmetric map K Kᵀ)
    (Q=K=V)+     same, plus 2D positional injection

Model is a single Transformer ENCODER (non-causal) doing per-token classification
over the 10 symbols {0..9}. Metric is token accuracy on a held-out test set,
averaged over the paper's config sweep (embed dim x layers x heads x seq len) and 3 seeds.

This is a faithful-but-modernized reproduction: it follows the paper's Section 4.1 setup
(one-hot inputs, Adam lr 1e-3, cosine schedule with warmup, grad clip 5, m=10, 2 epochs,
the {32,64,256} x {2,4} x {2,4} x {16,64,128} sweep, 3 runs averaged) but uses pre-norm
blocks, learned positional embeddings, and seeded reproducibility. Dataset size (unspecified
in the paper) defaults to 10k train / 2k test sequences per run and is configurable.

Usage:
    python synthetic_tasks.py --quick                 # fast smoke test (minutes)
    python synthetic_tasks.py --check-symmetry        # verify (X)+ breaks attn symmetry
    python synthetic_tasks.py                          # full sweep (single device)
    python synthetic_tasks.py --shard 0/4 --device cuda:0   # one shard of a 4-way split
    python synthetic_tasks.py --merge                  # merge shard CSVs -> results table
"""

import os
import csv
import math
import time
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# TASKS
# ============================================================================

TASKS = ["REVERSE", "SORT", "SUB", "SWAP", "COPY"]
NUM_SYMBOLS = 10  # values 0..9, one-hot dim


def make_targets(x: torch.Tensor, task: str) -> torch.Tensor:
    """Given input lists x [B, T] of ints in 0..9, return the target lists [B, T]."""
    if task == "COPY":
        return x.clone()
    if task == "REVERSE":
        return torch.flip(x, dims=[1])
    if task == "SORT":
        return torch.sort(x, dim=1).values
    if task == "SUB":
        return (NUM_SYMBOLS - 1) - x  # 9 - x
    if task == "SWAP":
        assert x.size(1) % 2 == 0, "SWAP needs even sequence length"
        half = x.size(1) // 2
        return torch.cat([x[:, half:], x[:, :half]], dim=1)
    raise ValueError(f"unknown task {task}")


def make_dataset(n: int, seq_len: int, task: str, generator: torch.Generator):
    """Generate n random lists and their task targets. Returns (inputs, targets) [n, T]."""
    x = torch.randint(0, NUM_SYMBOLS, (n, seq_len), generator=generator)
    y = make_targets(x, task)
    return x, y


# ============================================================================
# 2D POSITIONAL ENCODING (for the (X)+ variants)
# ============================================================================

def sincos_1d(length: int, dim: int) -> torch.Tensor:
    """Standard 1D sinusoidal features, robust to odd dim. Returns [length, dim]."""
    pe = torch.zeros(length, dim)
    position = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / max(dim, 1)))
    pe[:, 0::2] = torch.sin(position * div[: pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe


def build_2d_sincos(n: int, m: int) -> torch.Tensor:
    """
    Fixed 2D sinusoidal positional tensor P [n, n, m].
    P[i, j] = concat(rowfeat(i), colfeat(j)), so P[i, j] != P[j, i] in general.
    This asymmetry is what lets the (X)+ variants break the symmetric K Kᵀ attention map.
    """
    half = m // 2
    row = sincos_1d(n, half)          # depends on row index i
    col = sincos_1d(n, m - half)      # depends on col index j
    P = torch.zeros(n, n, m)
    P[:, :, :half] = row.unsqueeze(1).expand(n, n, half)
    P[:, :, half:] = col.unsqueeze(0).expand(n, n, m - half)
    return P


# ============================================================================
# ATTENTION (all six variants)
# ============================================================================

# canonical variant name -> (projection sharing, uses 2D pos injection)
VARIANTS = {
    "QKV":        ("none", False),
    "Q=K!=V":     ("qk",   False),
    "(Q=K!=V)+":  ("qk",   True),
    "Q!=K=V":     ("kv",   False),
    "Q=K=V":      ("qkv",  False),
    "(Q=K=V)+":   ("qkv",  True),
    "QVV(3)":     ("qvv3", False),
}


class SharedProjAttention(nn.Module):
    """Multi-head encoder self-attention with configurable projection sharing.

    share:
        none -> separate c_q, c_k, c_v
        qk   -> q = k = c_qk(x); separate c_v          (symmetric scores)
        kv   -> separate c_q; k = v = c_kv(x)
        qkv  -> single c_qkv(x) for all three          (symmetric scores)
        qvv3 -> q = c_q2(c_q1(x)) (depth-2 factored); k = v = c_v(x). Same
                asymmetric Q!=K=V attention, but Q is a composition of two
                learned matrices, so param count matches standard QKV
                (Gao & Xu 2026's QVV(3): the two Q matrices + one V matrix =
                3 learned matrices, same total as W^Q, W^K, W^V). c_q1/c_q2
                have no nonlinearity between them so they fuse into a single
                matrix at inference (see `fuse_qvv3`), making inference cost
                identical to the plain kv-share variant.
    plus: inject 2D positional encoding into the score map (see build_2d_sincos).
    """

    def __init__(self, n_embd, n_head, share, plus, pos_dim=10, max_len=128):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.share = share
        self.plus = plus

        if share == "none":
            self.c_q = nn.Linear(n_embd, n_embd)
            self.c_k = nn.Linear(n_embd, n_embd)
            self.c_v = nn.Linear(n_embd, n_embd)
        elif share == "qk":
            self.c_qk = nn.Linear(n_embd, n_embd)
            self.c_v = nn.Linear(n_embd, n_embd)
        elif share == "kv":
            self.c_q = nn.Linear(n_embd, n_embd)
            self.c_kv = nn.Linear(n_embd, n_embd)
        elif share == "qkv":
            self.c_qkv = nn.Linear(n_embd, n_embd)
        elif share == "qvv3":
            self.c_q1 = nn.Linear(n_embd, n_embd)
            self.c_q2 = nn.Linear(n_embd, n_embd)
            self.c_v = nn.Linear(n_embd, n_embd)
        else:
            raise ValueError(share)

        self.c_proj = nn.Linear(n_embd, n_embd)

        if plus:
            # 1x1 conv over the m channels of (scores broadcast + P) collapsing m -> 1.
            # A 1x1 conv is a per-channel linear combo; since conv is linear it factors as
            #   out[i,j] = scores[i,j]*sum(w) + sum_c w_c*P[i,j,c] + b,
            # which we compute directly (memory-light, mathematically identical).
            self.register_buffer("pos2d", build_2d_sincos(max_len, pos_dim))
            self.pos_w = nn.Parameter(torch.randn(pos_dim) * 0.02)
            self.pos_b = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_scores=False, return_attn=False):
        B, T, C = x.size()

        if self.share == "none":
            q, k, v = self.c_q(x), self.c_k(x), self.c_v(x)
        elif self.share == "qk":
            qk = self.c_qk(x); q, k, v = qk, qk, self.c_v(x)
        elif self.share == "kv":
            q = self.c_q(x); kv = self.c_kv(x); k, v = kv, kv
        elif self.share == "qvv3":
            q = self.c_q2(self.c_q1(x)); kv = self.c_v(x); k, v = kv, kv
        else:  # qkv
            s = self.c_qkv(x); q, k, v = s, s, s

        def split(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,hd]

        q, k, v = split(q), split(k), split(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,T,T]

        if self.plus:
            P = self.pos2d[:T, :T, :]                      # [T,T,m]
            pos_term = torch.einsum("ijm,m->ij", P, self.pos_w)  # [T,T]
            scores = scores * self.pos_w.sum() + pos_term + self.pos_b

        # encoder: no causal mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                        # [B,H,T,hd]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        if return_scores:
            return out, scores  # raw pre-softmax score map
        if return_attn:
            return out, attn    # post-softmax attention weights [B,H,T,T]
        return out

    @torch.no_grad()
    def fuse_qvv3(self):
        """Collapse c_q1, c_q2 into a single c_q Linear (exact, since there is no
        nonlinearity between them): y = W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2).
        Returns a new 'kv'-share module with identical forward output, matching the
        paper's claim that QVV(3) has no extra inference cost over Q!=K=V."""
        assert self.share == "qvv3"
        fused = SharedProjAttention(self.n_embd, self.n_head, "kv", self.plus)
        W1, b1 = self.c_q1.weight, self.c_q1.bias
        W2, b2 = self.c_q2.weight, self.c_q2.bias
        fused.c_q.weight.copy_(W2 @ W1)
        fused.c_q.bias.copy_(W2 @ b1 + b2)
        fused.c_kv.weight.copy_(self.c_v.weight)
        fused.c_kv.bias.copy_(self.c_v.bias)
        fused.c_proj.load_state_dict(self.c_proj.state_dict())
        return fused


# ============================================================================
# ENCODER MODEL
# ============================================================================

@dataclass
class ModelCfg:
    n_embd: int
    n_layer: int
    n_head: int
    seq_len: int
    variant: str
    pos_dim: int = 10
    n_inner_mult: int = 4
    max_len: int = 128


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        share, plus = VARIANTS[cfg.variant]
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SharedProjAttention(cfg.n_embd, cfg.n_head, share, plus,
                                        pos_dim=cfg.pos_dim, max_len=cfg.max_len)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_inner_mult * cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_inner_mult * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):
    """One-hot inputs -> linear embed + learned pos emb -> N blocks -> per-token 10-way head."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(NUM_SYMBOLS, cfg.n_embd)       # one-hot(10) -> d
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.n_embd)  # learned positional
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, NUM_SYMBOLS)

    def forward(self, x_ids, targets=None):
        B, T = x_ids.size()
        oh = F.one_hot(x_ids, NUM_SYMBOLS).float()
        pos = torch.arange(T, device=x_ids.device)
        h = self.embed(oh) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(self.ln_f(h))                     # [B,T,10]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, NUM_SYMBOLS), targets.reshape(-1))
        return logits, loss


# ============================================================================
# TRAIN / EVAL A SINGLE RUN
# ============================================================================

def lr_lambda_factory(total_steps, warmup_frac):
    warmup = max(1, int(warmup_frac * total_steps))

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


def run_one(variant, task, n_embd, n_layer, n_head, seq_len, seed,
            device, epochs=2, n_train=10000, n_test=2000, batch_size=128,
            lr=1e-3, grad_clip=5.0, warmup_frac=0.05, pos_dim=10):
    """Train one model on one task/config/seed. Returns test token accuracy."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    x_tr, y_tr = make_dataset(n_train, seq_len, task, g)
    x_te, y_te = make_dataset(n_test, seq_len, task, g)
    x_te, y_te = x_te.to(device), y_te.to(device)

    cfg = ModelCfg(n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                   seq_len=seq_len, variant=variant, pos_dim=pos_dim)
    model = Encoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total_steps, warmup_frac))

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_train, generator=g)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_tr[idx].to(device), y_tr[idx].to(device)
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()

    # eval: token accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, x_te.size(0), batch_size):
            xb, yb = x_te[i:i + batch_size], y_te[i:i + batch_size]
            logits, _ = model(xb)
            pred = logits.argmax(-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total


# ============================================================================
# SWEEP DRIVER
# ============================================================================

def build_jobs(quick=False):
    """Return the full list of (variant, task, embed, layers, heads, seqlen, seed) jobs."""
    if quick:
        embeds, layers, heads, seqlens, seeds = [32], [2], [2], [16], [0]
    else:
        embeds = [32, 64, 256]
        layers = [2, 4]
        heads = [2, 4]
        seqlens = [16, 64, 128]
        seeds = [0, 1, 2]
    jobs = []
    for variant in VARIANTS:
        for task in TASKS:
            for e in embeds:
                for L in layers:
                    for H in heads:
                        if e % H != 0:
                            continue
                        for T in seqlens:
                            for s in seeds:
                                jobs.append((variant, task, e, L, H, T, s))
    return jobs


def aggregate(csv_paths, out_md):
    """Read per-run CSVs, average accuracy per (variant, task), write the Table 1 grid."""
    from collections import defaultdict
    acc = defaultdict(list)
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                acc[(row["variant"], row["task"])].append(float(row["accuracy"]))

    lines = ["| Variant | " + " | ".join(TASKS) + " | Avg |",
             "|" + "---|" * (len(TASKS) + 2)]
    for variant in VARIANTS:
        cells = []
        per_task_means = []
        for task in TASKS:
            vals = acc.get((variant, task), [])
            if vals:
                m = sum(vals) / len(vals)
                per_task_means.append(m)
                cells.append(f"{m:.3f}")
            else:
                cells.append("-")
        avg = f"{sum(per_task_means) / len(per_task_means):.3f}" if per_task_means else "-"
        lines.append(f"| {variant} | " + " | ".join(cells) + f" | {avg} |")

    table = "\n".join(lines)
    with open(out_md, "w") as f:
        f.write("# Synthetic-tasks results (token accuracy, averaged over sweep)\n\n")
        f.write(table + "\n")
    print(table)
    print(f"\nWrote {out_md}")


def check_symmetry(device):
    """Sanity check on the raw score map (pre-softmax):
    symmetric-projection variants produce a symmetric map (K Kᵀ); the (X)+ variants
    inject asymmetry via the 2D positional encoding. We set the pos weights to a fixed
    nonzero vector to emulate a trained state (at init they are ~0 and inject nothing).
    """
    x = torch.randint(0, NUM_SYMBOLS, (2, 16), device=device)
    for base, plus in [("Q=K!=V", "(Q=K!=V)+"), ("Q=K=V", "(Q=K=V)+")]:
        for v in (base, plus):
            torch.manual_seed(0)
            m = Encoder(ModelCfg(n_embd=32, n_layer=1, n_head=2, seq_len=16, variant=v))
            m = m.to(device).eval()
            attn0 = m.blocks[0].attn
            if attn0.plus:
                torch.manual_seed(1)
                attn0.pos_w.data = torch.randn_like(attn0.pos_w)  # emulate trained weights
            oh = F.one_hot(x, NUM_SYMBOLS).float()
            pos = torch.arange(16, device=device)
            h = m.embed(oh) + m.pos_emb(pos)[None]
            _, scores = attn0(m.blocks[0].ln1(h), return_scores=True)
            asym = (scores - scores.transpose(-1, -2)).abs().mean().item()
            print(f"{v:>12}: mean|S - Sᵀ| = {asym:.6f}")
    print("(base ~0 = symmetric score map; + clearly nonzero = asymmetry injected)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny sweep for a smoke test")
    ap.add_argument("--check-symmetry", action="store_true")
    ap.add_argument("--merge", action="store_true", help="merge shard CSVs into results table")
    ap.add_argument("--shard", type=str, default="0/1", help="i/N split of the job list")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--variant", type=str, default=None, help="restrict to one variant")
    ap.add_argument("--task", type=str, default=None, help="restrict to one task")
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--out-md", type=str, default="synthetic_results.md")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    if args.check_symmetry:
        check_symmetry(device)
        return

    if args.merge:
        shard_csvs = [f"synthetic_results_shard{i}.csv" for i in range(64)]
        existing = [p for p in shard_csvs if os.path.exists(p)]
        if os.path.exists("synthetic_results.csv"):
            existing.append("synthetic_results.csv")
        aggregate(existing, args.out_md)
        return

    i, N = (int(v) for v in args.shard.split("/"))
    jobs = build_jobs(quick=args.quick)
    if args.variant:
        jobs = [j for j in jobs if j[0] == args.variant]
    if args.task:
        jobs = [j for j in jobs if j[1] == args.task]
    jobs = jobs[i::N]

    out_csv = args.out_csv or (f"synthetic_results_shard{i}.csv" if N > 1
                               else "synthetic_results.csv")
    print(f"device={device}  shard={i}/{N}  jobs={len(jobs)}  -> {out_csv}")

    epochs = 1 if args.quick else args.epochs
    n_train = 512 if args.quick else args.n_train
    n_test = 256 if args.quick else args.n_test

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "task", "n_embd", "n_layer", "n_head",
                    "seq_len", "seed", "accuracy"])
        t0 = time.time()
        for n, (variant, task, e, L, H, T, s) in enumerate(jobs):
            try:
                acc = run_one(variant, task, e, L, H, T, s, device,
                              epochs=epochs, n_train=n_train, n_test=n_test,
                              batch_size=args.batch_size)
            except Exception as ex:
                # keep a long unattended run alive if a single job errors
                # (e.g. transient OOM if a neighbor grabs GPU memory)
                print(f"[{n+1}/{len(jobs)}] ERROR {variant} {task} "
                      f"d{e} L{L} H{H} T{T} s{s}: {repr(ex)[:120]}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            w.writerow([variant, task, e, L, H, T, s, f"{acc:.4f}"])
            f.flush()
            if (n + 1) % 10 == 0 or args.quick:
                el = time.time() - t0
                print(f"[{n+1}/{len(jobs)}] {variant} {task} "
                      f"d{e} L{L} H{H} T{T} s{s} -> acc {acc:.4f} "
                      f"({el:.0f}s, {el/(n+1):.1f}s/run)", flush=True)

    if N == 1:
        aggregate([out_csv], args.out_md)
    else:
        print(f"Shard done. After all shards finish, run: python {os.path.basename(__file__)} --merge")


if __name__ == "__main__":
    main()
