"""
Representational-similarity analysis of the trained K and V projections in the QKV
teacher checkpoints -- the empirical half of the theoretical account of why post-hoc
K=V surgery (distillation_synthetic.py / distillation_vision.py) collapses some tasks
and not others.

For each task/dataset's QKV checkpoint, captures c_k(x) and c_v(x) activations (via
forward hooks) on held-out data at every attention layer, then reports:
  - weight cosine similarity between the flattened (W_k, b_k) and (W_v, b_v) parameters
  - linear CKA (Kornblith et al. 2019) between the K and V activation matrices
averaged over layers. Both are high when K and V converge to near-redundant functions
(safe to tie post-hoc) and low when they specialize into distinct roles (unsafe to tie).

Usage:
    conda run -n torch_env python kv_similarity_analysis.py
"""

import csv
import torch
import torch.nn as nn

from synthetic_tasks import Encoder, ModelCfg, make_dataset, TASKS
import vision_tasks as V
from vision_tasks import ViT, ViTConfig, CLASSIFICATION

CKPT_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_cka(X, Y):
    """Linear CKA between two [N, d] activation matrices (Kornblith et al. 2019)."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (Y.T @ X).norm() ** 2
    normx = (X.T @ X).norm()
    normy = (Y.T @ Y).norm()
    return (hsic / (normx * normy + 1e-12)).item()


def weight_cosine(lin_k, lin_v):
    wk = torch.cat([lin_k.weight.flatten(), lin_k.bias.flatten()])
    wv = torch.cat([lin_v.weight.flatten(), lin_v.bias.flatten()])
    return torch.nn.functional.cosine_similarity(wk, wv, dim=0).item()


def capture_kv_activations(model, x, blocks):
    """Run a forward pass, capturing each block's c_k(x)/c_v(x) output via hooks."""
    acts = {i: {} for i in range(len(blocks))}
    handles = []
    for i, blk in enumerate(blocks):
        def mk(i, name):
            def hook(_mod, _inp, out):
                acts[i][name] = out.detach().reshape(-1, out.size(-1))
            return hook
        handles.append(blk.attn.c_k.register_forward_hook(mk(i, "k")))
        handles.append(blk.attn.c_v.register_forward_hook(mk(i, "v")))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return acts


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
        cos_per_layer = [weight_cosine(b.attn.c_k, b.attn.c_v) for b in model.blocks]
        cka_per_layer = [linear_cka(acts[i]["k"], acts[i]["v"]) for i in range(len(model.blocks))]
        rows.append({
            "domain": "synthetic", "name": task,
            "weight_cos_sim": round(sum(cos_per_layer) / len(cos_per_layer), 4),
            "activation_cka": round(sum(cka_per_layer) / len(cka_per_layer), 4),
        })
        print(rows[-1], flush=True)
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
        cos_per_layer = [weight_cosine(b.attn.c_k, b.attn.c_v) for b in model.blocks]
        cka_per_layer = [linear_cka(acts[i]["k"], acts[i]["v"]) for i in range(len(model.blocks))]
        rows.append({
            "domain": "vision", "name": ds,
            "weight_cos_sim": round(sum(cos_per_layer) / len(cos_per_layer), 4),
            "activation_cka": round(sum(cka_per_layer) / len(cka_per_layer), 4),
        })
        print(rows[-1], flush=True)
    return rows


def main():
    rows = analyze_synthetic() + analyze_vision()
    with open("kv_similarity_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "name", "weight_cos_sim", "activation_cka"])
        w.writeheader()
        w.writerows(rows)
    print("\nWrote kv_similarity_results.csv")


if __name__ == "__main__":
    main()
