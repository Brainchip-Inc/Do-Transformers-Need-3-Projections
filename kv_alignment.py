"""
K->V alignment (reviewer-requested "decisive experiment" #2): fit a linear map
(or orthogonal Procrustes rotation) from a teacher's K activations to its V
activations on held-out TRAIN-split data, then test zero-shot (no retraining)
whether that learned correction closes the keep_k gap -- a direct test of the
"basis mismatch" hypothesis (Section 4.2 of theory/kv_tying_theory.tex), rather
than an inference from indirect evidence (the linear probe / random-baseline /
distillation-speed arguments already in the paper).

For each teacher block, fits (on a *train*-split sample, evaluated on the
existing held-out *test* split, so this isn't just memorizing the test set):
    linear:     A*, b* = argmin_{A,b} sum_j || A k_j + b - v_j ||^2   (ridge)
    procrustes: R*      = argmin_{R^T R = I} || K R - V ||_F           (SVD)

Then evaluates two new zero-shot surgery modes (monkey-patched forward, exactly
like kv_addressing_payload_swap.py -- no new model class, no retraining):
    keep_k_aligned:    c_kv(x) := A* @ c_k(x) + b*   (learned affine correction)
    keep_k_procrustes: c_kv(x) := c_k(x) @ R*         (learned rotation only)
against the existing keep_k / keep_v zero-shot numbers.

Usage:
    conda run -n torch_env python kv_alignment.py
"""

import csv
import types

import torch
import torch.nn.functional as F

from synthetic_tasks import Encoder, ModelCfg, make_dataset, TASKS
from distillation_synthetic import CKPT_DIR as SYN_CKPT_DIR, SYN_CFG, evaluate as syn_evaluate
import vision_tasks as V
from vision_tasks import ViT, ViTConfig, CLASSIFICATION, evaluate as vis_evaluate, _loader
from kv_similarity_analysis import capture_kv_activations

DEVICE = torch.device("cpu")
RIDGE_FRAC = 0.1


def fit_linear_map(K, V_, ridge_frac=RIDGE_FRAC):
    """Closed-form ridge regression K -> V (standardized K for numerical safety,
    then folded back into a single (weight, bias) pair operating on raw K --
    see kv_content_probe_llm.py for why standardization + relative ridge matters
    once activation scales differ across projections)."""
    mean = K.mean(0, keepdim=True)
    std = K.std(0, keepdim=True).clamp_min(1e-6)
    Kn = (K - mean) / std
    ones = torch.ones(Kn.size(0), 1)
    K1 = torch.cat([Kn, ones], dim=1)
    XtX = K1.T @ K1
    ridge = ridge_frac * XtX.diagonal().mean()
    A = XtX + ridge * torch.eye(K1.size(1))
    Wb = torch.linalg.solve(A, K1.T @ V_)  # [d+1, d_out]
    W, b = Wb[:-1], Wb[-1]
    # fold standardization into a single affine map on RAW k: v = k @ W_raw + b_raw
    W_raw = W / std.T                      # [d, d_out]
    b_raw = (b - (mean / std) @ W).squeeze(0)  # [d_out]
    return W_raw, b_raw


def fit_procrustes(K, V_):
    """Orthogonal Procrustes: R* = argmin_{R^T R=I} ||K R - V||_F, via SVD of
    K^T V (standardized first, same reasoning as fit_linear_map)."""
    mean_k, std_k = K.mean(0, keepdim=True), K.std(0, keepdim=True).clamp_min(1e-6)
    mean_v, std_v = V_.mean(0, keepdim=True), V_.std(0, keepdim=True).clamp_min(1e-6)
    Kn = (K - mean_k) / std_k
    Vn = (V_ - mean_v) / std_v
    M = Kn.T @ Vn
    U, _, Wt = torch.linalg.svd(M)
    R = U @ Wt  # [d, d]
    return R, mean_k, std_k, mean_v, std_v


def aligned_forward_factory(kind, params):
    """Returns a forward(self, x) that replicates SharedProjAttention.forward
    (share='none') but with a single shared kv := transform(c_k(x)) used for
    BOTH scoring and payload -- the same "one shared projection" shape as
    keep_k/keep_v surgery, just with a learned correction instead of an
    identity copy."""
    if kind == "linear":
        W_raw, b_raw = params

        def transform(k):
            return k @ W_raw + b_raw
    else:  # procrustes
        R, mean_k, std_k, mean_v, std_v = params

        def transform(k):
            kn = (k - mean_k) / std_k
            vn = kn @ R
            return vn * std_v + mean_v

    def forward(self, x):
        assert self.share == "none"
        B, T, C = x.size()
        q = self.c_q(x)
        kv = transform(self.c_k(x))

        def split(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, kv_h = split(q), split(kv)

        import math
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, kv_h.transpose(-2, -1)) * scale
        if self.plus:
            P = self.pos2d[:T, :T, :]
            pos_term = torch.einsum("ijm,m->ij", P, self.pos_w)
            scores = scores * self.pos_w.sum() + pos_term + self.pos_b
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, kv_h)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)
    return forward


def apply_alignment(model, fit_x, kind):
    """Fits per-block K->V maps on fit_x (a train-split batch) and monkey-patches
    each block's attention forward to use the learned correction. No parameters
    are modified -- purely a forward-time transform, still zero-shot/no-retraining."""
    acts = capture_kv_activations(model, fit_x, model.blocks)
    for i, blk in enumerate(model.blocks):
        K, V_ = acts[i]["k"], acts[i]["v"]
        if kind == "linear":
            params = fit_linear_map(K, V_)
        else:
            params = fit_procrustes(K, V_)
        blk.attn.forward = types.MethodType(aligned_forward_factory(kind, params), blk.attn)


def load_existing_row(csv_path, key_col, key_val, mode):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row[key_col] == key_val and row["mode"] == mode:
                return row
    return None


def run_synthetic():
    rows = []
    for task in TASKS:
        path = f"{SYN_CKPT_DIR}/synthetic_{task.lower()}_qkv.pt"
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ModelCfg(**d["config"])

        g = torch.Generator().manual_seed(0)
        x_tr, _ = make_dataset(SYN_CFG["n_train"], cfg.seq_len, task, g)
        x_te, y_te = make_dataset(SYN_CFG["n_test"], cfg.seq_len, task, g)
        fit_x = x_tr[:2000]  # a train-split sample is plenty for a per-layer closed-form fit

        accs = {}
        for kind in ["linear", "procrustes"]:
            teacher = Encoder(cfg).to(DEVICE).eval()
            teacher.load_state_dict(d["model_state_dict"])
            apply_alignment(teacher, fit_x, kind)
            accs[kind] = round(syn_evaluate(teacher, x_te, y_te, SYN_CFG["batch_size"]), 4)

        kk = load_existing_row("distillation_synthetic_results.csv", "task", task, "keep_k")
        kv = load_existing_row("distillation_synthetic_results.csv", "task", task, "keep_v")
        row = {
            "task": task,
            "keep_k": float(kk["zero_shot_surgery_acc"]) if kk else None,
            "keep_k_aligned": accs["linear"],
            "keep_k_procrustes": accs["procrustes"],
            "keep_v": float(kv["zero_shot_surgery_acc"]) if kv else None,
        }
        rows.append(row)
        print(row, flush=True)
    return rows


def run_vision():
    rows = []
    for ds in CLASSIFICATION:
        path = f"checkpoints/vision_{ds}_qkv.pt"
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ViTConfig(**d["config"])

        train_ds = V.get_classification_dataset(ds, "./vision_data", True)
        fit_loader = _loader(train_ds, 512, 2, False)
        fit_x, _ = next(iter(fit_loader))
        fit_x = fit_x.to(DEVICE)

        test_ds = V.get_classification_dataset(ds, "./vision_data", False)
        test_loader = _loader(test_ds, 256, 2, False)

        accs = {}
        for kind in ["linear", "procrustes"]:
            teacher = ViT(cfg).to(DEVICE).eval()
            teacher.load_state_dict(d["model_state_dict"])
            apply_alignment(teacher, fit_x, kind)
            accs[kind] = round(vis_evaluate(teacher, test_loader, DEVICE), 4)

        kk = load_existing_row("distillation_vision_results.csv", "dataset", ds, "keep_k")
        kv = load_existing_row("distillation_vision_results.csv", "dataset", ds, "keep_v")
        row = {
            "dataset": ds,
            "keep_k": float(kk["zero_shot_surgery_acc"]) if kk else None,
            "keep_k_aligned": accs["linear"],
            "keep_k_procrustes": accs["procrustes"],
            "keep_v": float(kv["zero_shot_surgery_acc"]) if kv else None,
        }
        rows.append(row)
        print(row, flush=True)
    return rows


def main():
    print("=== Synthetic tasks ===", flush=True)
    syn_rows = run_synthetic()
    print("\n=== Vision datasets ===", flush=True)
    vis_rows = run_vision()

    fieldnames = ["domain", "task_or_dataset", "keep_k", "keep_k_aligned",
                  "keep_k_procrustes", "keep_v"]
    with open("kv_alignment_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in syn_rows:
            w.writerow({"domain": "synthetic", "task_or_dataset": r["task"], **{k: r[k] for k in fieldnames[2:]}})
        for r in vis_rows:
            w.writerow({"domain": "vision", "task_or_dataset": r["dataset"], **{k: r[k] for k in fieldnames[2:]}})
    print("\nWrote kv_alignment_results.csv")


if __name__ == "__main__":
    main()
