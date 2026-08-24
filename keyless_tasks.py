"""
Synthetic diagnostics for keyless_peper/paper.tex's Track 1 (Dynamic Keyless
Attention): tests H1.1 (static keyless attention underperforms full QKV
specifically on tasks needing content-addressed routing) and H1.2 (a dynamic,
query-conditioned routing operator recovers most of that gap while keeping the
KV-cache-reduction property, since it depends only on the current query, not on
cached history).

Self-contained rather than extending synthetic_tasks.py's SharedProjAttention/
VARIANTS: that file and its "share" config are the K=V tying paper's
Table-1 reproduction harness, used by a dozen other analysis scripts in this
repo -- adding unrelated variants there risks polluting its own sweep/reuse
semantics. Only NUM_SYMBOLS and lr_lambda_factory (pure utilities) are reused.

Two tasks, deliberately not in synthetic_tasks.py's TASKS list (REVERSE/SORT/
SUB/SWAP/COPY probe routing generally, not content-addressed recall):

  INDUCTION: a bigram (A, B) appears once early in the sequence; when A recurs
    later, the target at the position right after that later A is B (Olsson
    et al. 2022's induction-head diagnostic). Only that one position is scored.

  ASSOC_RECALL: n_pairs key-value pairs k_1 v_1 ... k_n v_n (unique keys, one
    query position appending a repeated k_i); target at the query position is
    v_i (the associated value). Standard single-query associative-recall
    diagnostic (Based/Zoology line of work). Only the query position is scored.

Three attention variants, each a standalone module (not sharing weights/state
with synthetic_tasks.py):
  QKV            -- separate Q, K, V (baseline, full expressivity)
  Keyless        -- no K at all; S_ij = (x_i W_Q) R (x_j W_V)^T, R fixed
                    per head (the paper's static keyless baseline)
  DynamicKeyless -- R(x_i) = U_R diag(g(x_i W_Q)) V_R^T, rank-r, query-
                    conditioned (the paper's Track 1 proposal)

Usage:
    python keyless_tasks.py --quick     # fast smoke test
    python keyless_tasks.py             # full run, all variants x tasks
"""

import csv
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_tasks import NUM_SYMBOLS, lr_lambda_factory

TASKS = ["INDUCTION", "ASSOC_RECALL"]
VARIANTS = ["QKV", "Keyless", "DynamicKeyless"]
IGNORE_INDEX = -100


# ============================================================================
# TASKS
# ============================================================================

def make_induction_dataset(n, seq_len, generator):
    """[n, seq_len] random sequences with an inserted (A, B) induction pair.
    Target/mask are [n, seq_len]; only the position right after the SECOND A
    is scored (target = B), everything else is IGNORE_INDEX.

    Filler positions are resampled to avoid A: without this, a random filler
    slot coincidentally equal to A (expected ~seq_len/NUM_SYMBOLS times per
    sequence) makes "which earlier A" ambiguous -- confirmed empirically as
    the reason a first version of this task never let even the QKV baseline
    exceed ~40% regardless of training budget (more epochs made it worse, a
    sign of genuine task ambiguity rather than underfitting)."""
    assert seq_len >= 6, "need room for two A's, a B, and some filler"
    x = torch.randint(0, NUM_SYMBOLS, (n, seq_len), generator=generator)
    y = torch.full((n, seq_len), IGNORE_INDEX, dtype=torch.long)
    for i in range(n):
        a = torch.randint(0, NUM_SYMBOLS, (1,), generator=generator).item()
        b = torch.randint(0, NUM_SYMBOLS, (1,), generator=generator).item()
        # first A at an early position (leaves room for a gap + second A + filler)
        pos_a1 = torch.randint(0, seq_len // 3, (1,), generator=generator).item()
        # second A strictly after the first bigram, with room for one more position
        pos_a2 = torch.randint(pos_a1 + 2, seq_len - 1, (1,), generator=generator).item()
        for p in range(seq_len):
            if p in (pos_a1, pos_a2):
                continue
            while x[i, p].item() == a:
                x[i, p] = torch.randint(0, NUM_SYMBOLS, (1,), generator=generator).item()
        x[i, pos_a1] = a
        x[i, pos_a1 + 1] = b
        x[i, pos_a2] = a
        y[i, pos_a2 + 1] = b
    return x, y


def make_assoc_recall_dataset(n, n_pairs, generator):
    """[n, 2*n_pairs + 1] sequences: k_1 v_1 ... k_n v_n, then a query position
    repeating one k_i. Target/mask are the same shape; only the final (query)
    position is scored (target = v_i)."""
    seq_len = 2 * n_pairs + 1
    x = torch.zeros(n, seq_len, dtype=torch.long)
    y = torch.full((n, seq_len), IGNORE_INDEX, dtype=torch.long)
    for i in range(n):
        keys = torch.randperm(NUM_SYMBOLS, generator=generator)[:n_pairs]
        values = torch.randint(0, NUM_SYMBOLS, (n_pairs,), generator=generator)
        x[i, 0:2 * n_pairs:2] = keys
        x[i, 1:2 * n_pairs:2] = values
        q_idx = torch.randint(0, n_pairs, (1,), generator=generator).item()
        x[i, -1] = keys[q_idx]
        y[i, -1] = values[q_idx]
    return x, y


def make_dataset(task, n, seq_len_or_pairs, generator):
    if task == "INDUCTION":
        return make_induction_dataset(n, seq_len_or_pairs, generator)
    if task == "ASSOC_RECALL":
        return make_assoc_recall_dataset(n, seq_len_or_pairs, generator)
    raise ValueError(task)


# ============================================================================
# ATTENTION VARIANTS
# ============================================================================

def _split_heads(t, n_head, head_dim):
    B, T, _ = t.size()
    return t.view(B, T, n_head, head_dim).transpose(1, 2)  # [B,H,T,hd]


def _merge_heads(t):
    B, H, T, hd = t.size()
    return t.transpose(1, 2).contiguous().view(B, T, H * hd)


class QKVAttentionSmall(nn.Module):
    """Standard separate-Q/K/V multi-head encoder self-attention (baseline)."""

    def __init__(self, n_embd, n_head, **kw):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.c_q = nn.Linear(n_embd, n_embd)
        self.c_k = nn.Linear(n_embd, n_embd)
        self.c_v = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        q = _split_heads(self.c_q(x), self.n_head, self.head_dim)
        k = _split_heads(self.c_k(x), self.n_head, self.head_dim)
        v = _split_heads(self.c_v(x), self.n_head, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out = _merge_heads(torch.matmul(attn, v))
        return self.c_proj(out)


class KeylessAttention(nn.Module):
    """Static keyless attention: no K projection. Per-head bilinear coupling
    S_ij = q_i^T R v_j (R fixed, learned, shared across all inputs) replaces
    the usual q_i^T k_j. Only V needs caching at inference (matches the
    paper's 50% KV-cache-reduction claim -- R and Q don't require a per-token
    cache)."""

    def __init__(self, n_embd, n_head, **kw):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.c_q = nn.Linear(n_embd, n_embd)
        self.c_v = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.R = nn.Parameter(torch.randn(n_head, self.head_dim, self.head_dim) * 0.02)

    def forward(self, x):
        B, T, C = x.size()
        q = _split_heads(self.c_q(x), self.n_head, self.head_dim)
        v = _split_heads(self.c_v(x), self.n_head, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        # S[b,h,i,j] = q[b,h,i,:] @ R[h] @ v[b,h,j,:]
        qR = torch.einsum("bhid,hde->bhie", q, self.R)
        scores = torch.einsum("bhie,bhje->bhij", qR, v) * scale
        attn = F.softmax(scores, dim=-1)
        out = _merge_heads(torch.matmul(attn, v))
        return self.c_proj(out)


class DynamicKeylessAttention(nn.Module):
    """Dynamic keyless attention (Track 1): R(x_i) = U_R diag(g(q_i)) V_R^T,
    a rank-`rank` operator conditioned on the CURRENT query only (so the
    cache-reduction property survives -- nothing here depends on cached
    history). Efficient equivalent form used below:
        a_i = q_i @ U_R                      [rank]
        gate_i = g(q_i)                      [rank]
        b_j = v_j @ V_R                      [rank]
        S_ij = (a_i * gate_i) . b_j
    which follows directly from substituting R(x_i) into q_i^T R(x_i) v_j and
    regrouping -- see module docstring's derivation in the plan."""

    def __init__(self, n_embd, n_head, rank=16, **kw):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.head_dim, self.rank = n_head, n_embd // n_head, rank
        self.c_q = nn.Linear(n_embd, n_embd)
        self.c_v = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.U_R = nn.Parameter(torch.randn(n_head, self.head_dim, rank) * 0.02)
        self.V_R = nn.Parameter(torch.randn(n_head, self.head_dim, rank) * 0.02)
        self.gate = nn.Linear(self.head_dim, rank)  # shared across heads, lightweight

    def forward(self, x):
        B, T, C = x.size()
        q = _split_heads(self.c_q(x), self.n_head, self.head_dim)
        v = _split_heads(self.c_v(x), self.n_head, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        a = torch.einsum("bhid,hdr->bhir", q, self.U_R)   # [B,H,T,r]
        gate = self.gate(q)                                # [B,H,T,r]
        b = torch.einsum("bhjd,hdr->bhjr", v, self.V_R)    # [B,H,T,r]
        ag = a * gate
        scores = torch.einsum("bhir,bhjr->bhij", ag, b) * scale
        attn = F.softmax(scores, dim=-1)
        out = _merge_heads(torch.matmul(attn, v))
        return self.c_proj(out)


class LowRankKeyAttention(nn.Module):
    """Diagnostic, not part of the paper draft: a genuinely INDEPENDENT but
    low-rank K, c_k: n_embd -> n_head*rank (much cheaper to cache than a full
    K, but -- unlike both keyless variants above -- computed from x_j
    directly rather than routed through v_j). Tests whether independence
    from V, not extra capacity, is what keyless attention is missing: both
    keyless variants plateau at ~0.35 on ASSOC_RECALL regardless of rank
    (8-64) or depth (2-4 layers), which rules out a capacity explanation."""

    def __init__(self, n_embd, n_head, rank=16, **kw):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.head_dim, self.rank = n_head, n_embd // n_head, rank
        self.c_q = nn.Linear(n_embd, n_head * rank)
        self.c_k = nn.Linear(n_embd, n_head * rank)
        self.c_v = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        q = _split_heads(self.c_q(x), self.n_head, self.rank)
        k = _split_heads(self.c_k(x), self.n_head, self.rank)
        v = _split_heads(self.c_v(x), self.n_head, self.head_dim)
        scale = 1.0 / math.sqrt(self.rank)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out = _merge_heads(torch.matmul(attn, v))
        return self.c_proj(out)


ATTN_CLASSES = {"QKV": QKVAttentionSmall, "Keyless": KeylessAttention,
                 "DynamicKeyless": DynamicKeylessAttention,
                 "LowRankKey": LowRankKeyAttention}


# ============================================================================
# ENCODER MODEL
# ============================================================================

@dataclass
class ModelCfg:
    n_embd: int
    n_layer: int
    n_head: int
    max_len: int
    variant: str
    rank: int = 16
    n_inner_mult: int = 4


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = ATTN_CLASSES[cfg.variant](cfg.n_embd, cfg.n_head, rank=cfg.rank)
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
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(NUM_SYMBOLS, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, NUM_SYMBOLS)

    def forward(self, x_ids):
        B, T = x_ids.size()
        oh = F.one_hot(x_ids, NUM_SYMBOLS).float()
        pos = torch.arange(T, device=x_ids.device)
        h = self.embed(oh) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))  # [B,T,NUM_SYMBOLS]


# ============================================================================
# TRAIN / EVAL
# ============================================================================

def masked_accuracy(logits, targets):
    mask = targets != IGNORE_INDEX
    pred = logits.argmax(-1)
    correct = ((pred == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / max(total, 1)


def run_one(variant, task, n_embd, n_layer, n_head, seq_len_or_pairs, seed, device,
            rank=16, epochs=5, n_train=10000, n_test=2000, batch_size=128,
            lr=1e-3, grad_clip=5.0, warmup_frac=0.05):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    x_tr, y_tr = make_dataset(task, n_train, seq_len_or_pairs, g)
    x_te, y_te = make_dataset(task, n_test, seq_len_or_pairs, g)
    x_te, y_te = x_te.to(device), y_te.to(device)

    max_len = x_tr.size(1)
    cfg = ModelCfg(n_embd=n_embd, n_layer=n_layer, n_head=n_head, max_len=max_len,
                   variant=variant, rank=rank)
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
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, NUM_SYMBOLS), yb.reshape(-1),
                                    ignore_index=IGNORE_INDEX)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, x_te.size(0), batch_size):
            xb, yb = x_te[i:i + batch_size], y_te[i:i + batch_size]
            logits = model(xb)
            mask = yb != IGNORE_INDEX
            pred = logits.argmax(-1)
            correct += ((pred == yb) & mask).sum().item()
            total += mask.sum().item()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--induction-seq-len", type=int, default=32)
    ap.add_argument("--assoc-n-pairs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", type=str, default="keyless_tasks_results.csv")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    if args.quick:
        args.epochs, args.n_train, args.n_test = 1, 512, 256

    seq_params = {"INDUCTION": args.induction_seq_len, "ASSOC_RECALL": args.assoc_n_pairs}

    rows = []
    for task in TASKS:
        for variant in VARIANTS:
            acc = run_one(variant, task, args.n_embd, args.n_layer, args.n_head,
                          seq_params[task], args.seed, device, rank=args.rank,
                          epochs=args.epochs, n_train=args.n_train, n_test=args.n_test)
            row = {"task": task, "variant": variant, "accuracy": round(acc, 4)}
            rows.append(row)
            print(row, flush=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "variant", "accuracy"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
