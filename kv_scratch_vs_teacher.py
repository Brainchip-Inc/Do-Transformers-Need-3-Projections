"""
Reviewer question: "did you analyze the from-scratch Q!=K=V representations?"
Nobody has -- this compares the scratch model's single learned c_kv projection
to the teacher's independent c_k and c_v, on held-out data, via activation CKA.

If joint training's c_kv converges to something close to the teacher's V (or K),
that would suggest joint training just "picks a side"; if it's roughly equidistant
from both (or more similar to neither than K and V are to each other), that would
suggest joint training finds a genuinely different synthesis rather than
approximating either independently-trained projection.

Usage:
    conda run -n torch_env python kv_scratch_vs_teacher.py
"""

import csv

import torch

from synthetic_tasks import Encoder, ModelCfg, make_dataset, TASKS
from distillation_synthetic import CKPT_DIR as SYN_CKPT_DIR, SYN_CFG
import vision_tasks as V
from vision_tasks import ViT, ViTConfig, CLASSIFICATION, _loader
from kv_similarity_analysis import linear_cka

DEVICE = torch.device("cpu")


def capture_teacher_kv(model, x, blocks):
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


def capture_scratch_kv(model, x, blocks):
    acts = {i: {} for i in range(len(blocks))}
    handles = []
    for i, blk in enumerate(blocks):
        def mk(i):
            def hook(_mod, _inp, out):
                acts[i]["kv"] = out.detach().reshape(-1, out.size(-1))
            return hook
        handles.append(blk.attn.c_kv.register_forward_hook(mk(i)))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return acts


def analyze(teacher, scratch, x, blocks_t, blocks_s):
    t_acts = capture_teacher_kv(teacher, x, blocks_t)
    s_acts = capture_scratch_kv(scratch, x, blocks_s)
    rows = []
    for i in range(len(blocks_t)):
        K, V_ = t_acts[i]["k"], t_acts[i]["v"]
        KV = s_acts[i]["kv"]
        rows.append({
            "layer": i,
            "cka_teacher_k_v": round(linear_cka(K, V_), 4),
            "cka_scratch_vs_teacher_k": round(linear_cka(KV, K), 4),
            "cka_scratch_vs_teacher_v": round(linear_cka(KV, V_), 4),
        })
    return rows


def run_synthetic():
    all_rows = []
    for task in TASKS:
        d_t = torch.load(f"{SYN_CKPT_DIR}/synthetic_{task.lower()}_qkv.pt", map_location=DEVICE, weights_only=False)
        d_s = torch.load(f"{SYN_CKPT_DIR}/synthetic_{task.lower()}_qkv_kv.pt", map_location=DEVICE, weights_only=False)
        cfg_t = ModelCfg(**d_t["config"])
        cfg_s = ModelCfg(**d_s["config"])
        teacher = Encoder(cfg_t).to(DEVICE).eval()
        teacher.load_state_dict(d_t["model_state_dict"])
        scratch = Encoder(cfg_s).to(DEVICE).eval()
        scratch.load_state_dict(d_s["model_state_dict"])

        g = torch.Generator().manual_seed(0)
        _, _ = make_dataset(SYN_CFG["n_train"], cfg_t.seq_len, task, g)
        x_te, _ = make_dataset(SYN_CFG["n_test"], cfg_t.seq_len, task, g)

        for row in analyze(teacher, scratch, x_te, teacher.blocks, scratch.blocks):
            row_full = {"domain": "synthetic", "task_or_dataset": task, **row}
            all_rows.append(row_full)
            print(row_full, flush=True)
    return all_rows


def run_vision():
    all_rows = []
    for ds in CLASSIFICATION:
        d_t = torch.load(f"checkpoints/vision_{ds}_qkv.pt", map_location=DEVICE, weights_only=False)
        d_s = torch.load(f"checkpoints/vision_{ds}_qkv_kv.pt", map_location=DEVICE, weights_only=False)
        cfg_t = ViTConfig(**d_t["config"])
        cfg_s = ViTConfig(**d_s["config"])
        teacher = ViT(cfg_t).to(DEVICE).eval()
        teacher.load_state_dict(d_t["model_state_dict"])
        scratch = ViT(cfg_s).to(DEVICE).eval()
        scratch.load_state_dict(d_s["model_state_dict"])

        test_ds = V.get_classification_dataset(ds, "./vision_data", False)
        x_te, _ = next(iter(_loader(test_ds, 512, 2, False)))

        for row in analyze(teacher, scratch, x_te, teacher.blocks, scratch.blocks):
            row_full = {"domain": "vision", "task_or_dataset": ds, **row}
            all_rows.append(row_full)
            print(row_full, flush=True)
    return all_rows


def main():
    print("=== Synthetic tasks ===", flush=True)
    syn_rows = run_synthetic()
    print("\n=== Vision datasets ===", flush=True)
    vis_rows = run_vision()

    fieldnames = ["domain", "task_or_dataset", "layer", "cka_teacher_k_v",
                  "cka_scratch_vs_teacher_k", "cka_scratch_vs_teacher_v"]
    with open("kv_scratch_vs_teacher_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(syn_rows + vis_rows)
    print("\nWrote kv_scratch_vs_teacher_results.csv")


if __name__ == "__main__":
    main()
