"""
LLM counterpart of the content-probe half of kv_addressing_content_probe.py: does a
closed-form ridge-regression linear probe from k_j/v_j activations recover the
identity of token j, on the 300M FineWeb-Edu QKV teacher? (The addressing-probe half
doesn't transfer here -- causal LM has no single ground-truth "source position" per
output position the way the synthetic tasks' REVERSE/SORT/SWAP do.)

If Corollary 1 (theory/kv_tying_theory.tex) held as originally stated -- K sacrifices
content for addressing -- V should probe noticeably better than K. The synthetic/
vision result (Section 4.2) was the opposite: K and V probe equally well everywhere,
which is why the paper reads the mechanism as basis-mismatch rather than content-loss.
This checks whether that also holds at LLM scale.

Adapted for a 50304-token vocabulary: naively materializing an [N, vocab_size]
one-hot target matrix doesn't fit in memory at this scale, so X^T Y is accumulated
via index_add_ over per-token columns instead of a dense one-hot matmul.

Usage:
    conda run -n kv python kv_content_probe_llm.py --device cuda:0
"""

import csv
import argparse

import torch
from transformers import AutoTokenizer

from distillation_llm import load_teacher, TEACHER_CKPT
from transformer_KQV_300M_fineweb import get_dataloader

RESULTS_CSV = "kv_content_probe_llm_results.csv"
PROBE_RIDGE = 0.1  # fraction of X^T X's mean diagonal, not an absolute value -- see below


def capture_kv_activations(model, input_ids, n_embd):
    acts = {i: {} for i in range(len(model.h))}
    handles = []
    for i, block in enumerate(model.h):
        def mk(i):
            def hook(_mod, _inp, out):
                q, k, v = out.split(n_embd, dim=2)
                acts[i]["k"] = k.detach().reshape(-1, n_embd)
                acts[i]["v"] = v.detach().reshape(-1, n_embd)
            return hook
        handles.append(block.attn.c_attn.register_forward_hook(mk(i)))
    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()
    return acts


def ridge_probe_accuracy(feat, labels, vocab_size, ridge_frac=PROBE_RIDGE, train_frac=0.8):
    """Closed-form ridge regression from feat [N,d] to one-hot(labels) [N,vocab_size],
    fit on a train split, evaluated (argmax accuracy) on the held-out split. X^T Y is
    built via index_add_ (summing feature columns into their label's column) instead
    of a dense [N, vocab_size] one-hot matmul, which would be ~50k x too large here.

    Features are standardized (mean/std from the train split only) before fitting, and
    the ridge penalty is set relative to X^T X's own scale (ridge_frac * mean diagonal)
    rather than a fixed absolute value: this model's K/V activations are severely
    rank-deficient (X^T X's condition number is ~1e37-1e39 even after standardization
    -- a small number of directions carry almost all the variance), so a fixed
    ridge=1e-2 is negligible at that eigenvalue scale and the closed-form solve
    degenerates into float32 noise (observed directly: one layer's raw-ridge K-probe
    accuracy collapsed to 0.13 against every neighboring layer's ~0.8-0.9, purely from
    numerical instability, not a real representational difference). Scaling the ridge
    to the data's own diagonal energy (checked to bring the regularized condition
    number down to ~1e3-1e4, safely within float32's solve precision) fixes this."""
    N = feat.size(0)
    n_train = int(N * train_frac)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0), device="cpu")
    perm = perm.to(feat.device)
    tr, te = perm[:n_train], perm[n_train:]

    Xtr, Xte = feat[tr], feat[te]
    ytr, yte = labels[tr], labels[te]

    mean = Xtr.mean(dim=0, keepdim=True)
    std = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    ones_tr = torch.ones(Xtr.size(0), 1, device=feat.device, dtype=feat.dtype)
    Xtr1 = torch.cat([Xtr, ones_tr], dim=1)
    d = Xtr1.size(1)

    XtX = Xtr1.T @ Xtr1
    ridge = ridge_frac * XtX.diagonal().mean()
    A = XtX + ridge * torch.eye(d, device=feat.device, dtype=feat.dtype)
    XtY = torch.zeros(d, vocab_size, device=feat.device, dtype=feat.dtype)
    XtY.index_add_(1, ytr, Xtr1.T)
    Wb = torch.linalg.solve(A, XtY)  # [d, vocab_size]

    ones_te = torch.ones(Xte.size(0), 1, device=feat.device, dtype=feat.dtype)
    Xte1 = torch.cat([Xte, ones_te], dim=1)
    pred = (Xte1 @ Wb).argmax(dim=-1)
    return (pred == yte).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--teacher-ckpt", type=str, default=TEACHER_CKPT)
    ap.add_argument("--val-data", type=str, default="./fineweb_edu_validation")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    teacher, model_config, _ = load_teacher(args.teacher_ckpt, device)
    teacher.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    loader, _ = get_dataloader(
        args.val_data, tokenizer, args.batch_size, model_config.n_positions,
        num_workers=2, prefetch_factor=2, shuffle=False)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    labels = input_ids.reshape(-1)
    print(f"{labels.numel()} probe examples ({args.batch_size} sequences x "
          f"{model_config.n_positions} positions)", flush=True)

    acts = capture_kv_activations(teacher, input_ids, model_config.n_embd)

    rows = []
    for layer in range(len(teacher.h)):
        k_acc = ridge_probe_accuracy(acts[layer]["k"], labels, model_config.vocab_size)
        v_acc = ridge_probe_accuracy(acts[layer]["v"], labels, model_config.vocab_size)
        row = {"layer": layer, "k_content_probe_acc": round(k_acc, 4),
               "v_content_probe_acc": round(v_acc, 4)}
        rows.append(row)
        print(row, flush=True)

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "k_content_probe_acc", "v_content_probe_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
