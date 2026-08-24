"""
Characterizes the K->V correction maps from kv_alignment.py (reviewer question:
"the linear/Procrustes correction works, but what does it actually do to the
representation?"), on the synthetic/vision QKV teachers.

For each teacher, per layer, on held-out TEST data disjoint from the data the
maps were fit on (kv_alignment.py fits on a train-split sample):
  - baseline activation CKA(K, V)               (already known, from
                                                   kv_similarity_results.csv)
  - CKA(linear-corrected K, V)   -- does the fix make K look more like V?
  - CKA(Procrustes-corrected K, V)
  - ||A* - "closest same-shape scaled identity"||_F, reported instead as the
    mean/std of A*'s singular values (close to 1 <=> near-orthogonal, i.e. the
    unconstrained linear fit essentially rediscovers a near-rotation; spread
    away from 1 <=> real rescaling was needed, not just a change of basis)

Usage:
    conda run -n torch_env python kv_correction_analysis.py
"""

import csv

import torch

from synthetic_tasks import Encoder, ModelCfg, make_dataset, TASKS
from distillation_synthetic import CKPT_DIR as SYN_CKPT_DIR, SYN_CFG
import vision_tasks as V
from vision_tasks import ViT, ViTConfig, CLASSIFICATION, _loader
from kv_similarity_analysis import capture_kv_activations, linear_cka
from kv_alignment import fit_linear_map, fit_procrustes

DEVICE = torch.device("cpu")


def analyze_one(teacher, fit_x, test_x, blocks):
    fit_acts = capture_kv_activations(teacher, fit_x, blocks)
    test_acts = capture_kv_activations(teacher, test_x, blocks)

    rows = []
    for i in range(len(blocks)):
        K_fit, V_fit = fit_acts[i]["k"], fit_acts[i]["v"]
        K_te, V_te = test_acts[i]["k"], test_acts[i]["v"]

        baseline_cka = linear_cka(K_te, V_te)

        W_raw, b_raw = fit_linear_map(K_fit, V_fit)
        K_lin = K_te @ W_raw + b_raw
        lin_cka = linear_cka(K_lin, V_te)
        sv = torch.linalg.svdvals(W_raw)

        R, mean_k, std_k, mean_v, std_v = fit_procrustes(K_fit, V_fit)
        K_proc = (((K_te - mean_k) / std_k) @ R) * std_v + mean_v
        proc_cka = linear_cka(K_proc, V_te)

        rows.append({
            "layer": i,
            "cka_baseline": round(baseline_cka, 4),
            "cka_after_linear": round(lin_cka, 4),
            "cka_after_procrustes": round(proc_cka, 4),
            "linear_map_singular_values_mean": round(sv.mean().item(), 4),
            "linear_map_singular_values_std": round(sv.std().item(), 4),
        })
    return rows


def run_synthetic():
    all_rows = []
    for task in TASKS:
        path = f"{SYN_CKPT_DIR}/synthetic_{task.lower()}_qkv.pt"
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ModelCfg(**d["config"])
        teacher = Encoder(cfg).to(DEVICE).eval()
        teacher.load_state_dict(d["model_state_dict"])

        g = torch.Generator().manual_seed(0)
        x_tr, _ = make_dataset(SYN_CFG["n_train"], cfg.seq_len, task, g)
        x_te, _ = make_dataset(SYN_CFG["n_test"], cfg.seq_len, task, g)
        fit_x, test_x = x_tr[:2000], x_te  # same fit slice as kv_alignment.py; distinct test data

        for row in analyze_one(teacher, fit_x, test_x, teacher.blocks):
            row_full = {"domain": "synthetic", "task_or_dataset": task, **row}
            all_rows.append(row_full)
            print(row_full, flush=True)
    return all_rows


def run_vision():
    all_rows = []
    for ds in CLASSIFICATION:
        path = f"checkpoints/vision_{ds}_qkv.pt"
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ViTConfig(**d["config"])
        teacher = ViT(cfg).to(DEVICE).eval()
        teacher.load_state_dict(d["model_state_dict"])

        train_ds = V.get_classification_dataset(ds, "./vision_data", True)
        fit_x, _ = next(iter(_loader(train_ds, 512, 2, False)))
        test_ds = V.get_classification_dataset(ds, "./vision_data", False)
        test_x, _ = next(iter(_loader(test_ds, 512, 2, False)))

        for row in analyze_one(teacher, fit_x, test_x, teacher.blocks):
            row_full = {"domain": "vision", "task_or_dataset": ds, **row}
            all_rows.append(row_full)
            print(row_full, flush=True)
    return all_rows


def main():
    print("=== Synthetic tasks ===", flush=True)
    syn_rows = run_synthetic()
    print("\n=== Vision datasets ===", flush=True)
    vis_rows = run_vision()

    fieldnames = ["domain", "task_or_dataset", "layer", "cka_baseline",
                  "cka_after_linear", "cka_after_procrustes",
                  "linear_map_singular_values_mean", "linear_map_singular_values_std"]
    with open("kv_correction_analysis_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(syn_rows + vis_rows)
    print("\nWrote kv_correction_analysis_results.csv")


if __name__ == "__main__":
    main()
