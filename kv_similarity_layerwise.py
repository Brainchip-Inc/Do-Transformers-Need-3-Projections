"""
Follow-up to kv_similarity_analysis.py, answering two reviewer questions about the
K-V CKA measurement in theory/kv_tying_theory.tex:

  1. "Averaged over layers" -- what's the layer-wise pattern? Both task models here
     have exactly 2 attention layers, so this reports CKA at layer 0 vs. layer 1
     separately for all 9 tasks/datasets, instead of only the average.
  2. "Have you considered other similarity measures beyond (linear) CKA?" -- adds
     RBF-kernel CKA (Kornblith et al. 2019's other standard variant) as a robustness
     check on the same activations, and reports whether it changes the qualitative
     ranking or the correlation with collapse severity.

Reuses the exact checkpoints, held-out data, and activation-capture logic from
kv_similarity_analysis.py -- only the reporting differs.

Usage:
    conda run -n torch_env python kv_similarity_layerwise.py
"""

import csv
import torch

from synthetic_tasks import Encoder, ModelCfg, make_dataset, TASKS
import vision_tasks as V
from vision_tasks import ViT, ViTConfig, CLASSIFICATION
from kv_similarity_analysis import capture_kv_activations, linear_cka

CKPT_DIR = "checkpoints"
# CPU by design, not just OOM avoidance: RBF-CKA's N x N gram matrix (N = a few
# thousand held-out tokens) doesn't fit alongside the live multi-day GPU training
# run in outputs_qkv_baseline_10B, and these 2-layer/256-dim models are cheap enough
# that CPU forward passes cost seconds, not minutes.
DEVICE = torch.device("cpu")
MAX_ROWS_FOR_RBF = 2000  # subsample: RBF-CKA's gram matrix is O(N^2) memory/compute


def rbf_cka(X, Y, sigma_frac=1.0):
    """RBF-kernel CKA (Kornblith et al. 2019, Eq. 4-6): same HSIC-based statistic as
    linear CKA, but with the linear kernel K(x,y)=x.y replaced by an RBF kernel
    K(x,y)=exp(-||x-y||^2 / (2*sigma^2)). sigma is set to sigma_frac times the median
    pairwise distance (the paper's default heuristic). Subsamples rows to
    MAX_ROWS_FOR_RBF first since the gram matrix is O(N^2)."""
    if X.size(0) > MAX_ROWS_FOR_RBF:
        idx = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))[:MAX_ROWS_FOR_RBF]
        X, Y = X[idx], Y[idx]

    def gram(A):
        sq = (A * A).sum(1, keepdim=True)
        d2 = sq + sq.T - 2 * (A @ A.T)
        d2 = d2.clamp(min=0)
        med = d2[d2 > 0].median().sqrt()
        sigma = sigma_frac * med
        return torch.exp(-d2 / (2 * sigma ** 2 + 1e-12))

    def center(K):
        n = K.size(0)
        H = torch.eye(n, device=K.device) - 1.0 / n
        return H @ K @ H

    Kx, Ky = center(gram(X)), center(gram(Y))
    hsic = (Kx * Ky).sum()
    normx = (Kx * Kx).sum().sqrt()
    normy = (Ky * Ky).sum().sqrt()
    return (hsic / (normx * normy + 1e-12)).item()


def analyze_synthetic():
    rows = []
    for task in TASKS:
        path = f"{CKPT_DIR}/synthetic_{task.lower()}_qkv.pt"
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ModelCfg(**d["config"])
        model = Encoder(cfg).to(DEVICE).eval()
        model.load_state_dict(d["model_state_dict"])

        g = torch.Generator().manual_seed(0)
        x_te, _ = make_dataset(1000, cfg.seq_len, task, g)
        x_te = x_te.to(DEVICE)

        acts = capture_kv_activations(model, x_te, model.blocks)
        for i in range(len(model.blocks)):
            k, v = acts[i]["k"], acts[i]["v"]
            row = {"domain": "synthetic", "name": task, "layer": i,
                   "linear_cka": round(linear_cka(k, v), 4),
                   "rbf_cka": round(rbf_cka(k, v), 4)}
            rows.append(row)
            print(row, flush=True)
    return rows


def analyze_vision():
    rows = []
    for ds in CLASSIFICATION:
        path = f"{CKPT_DIR}/vision_{ds}_qkv.pt"
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ViTConfig(**d["config"])
        model = ViT(cfg).to(DEVICE).eval()
        model.load_state_dict(d["model_state_dict"])

        test_ds = V.get_classification_dataset(ds, "./vision_data", False)
        loader = V._loader(test_ds, 256, 2, False)
        xb, _ = next(iter(loader))
        xb = xb.to(DEVICE)

        acts = capture_kv_activations(model, xb, model.blocks)
        for i in range(len(model.blocks)):
            k, v = acts[i]["k"], acts[i]["v"]
            row = {"domain": "vision", "name": ds, "layer": i,
                   "linear_cka": round(linear_cka(k, v), 4),
                   "rbf_cka": round(rbf_cka(k, v), 4)}
            rows.append(row)
            print(row, flush=True)
    return rows


def main():
    rows = analyze_synthetic() + analyze_vision()
    with open("kv_similarity_layerwise_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "name", "layer", "linear_cka", "rbf_cka"])
        w.writeheader()
        w.writerows(rows)
    print("\nWrote kv_similarity_layerwise_results.csv")


if __name__ == "__main__":
    main()
