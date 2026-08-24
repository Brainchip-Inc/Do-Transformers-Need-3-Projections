"""
LLM counterpart of kv_correction_analysis.py: characterizes what the K->V
correction maps from kv_alignment_llm.py actually do to the representation,
on held-out data disjoint from both the fit batch and the select batch used
to pick the linear map's ridge strength.

Usage:
    conda run -n kv python kv_correction_analysis_llm.py --device cuda:0
"""

import csv
import argparse

import torch
from transformers import AutoTokenizer

from distillation_llm import load_teacher, TEACHER_CKPT
from transformer_KQV_300M_fineweb import get_dataloader, enable_sdpa_attention
from kv_alignment_llm import (
    capture_kv_activations, fit_linear_map, fit_procrustes, select_ridge_frac_by_ppl,
)
from kv_similarity_llm import linear_cka


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--teacher-ckpt", type=str, default=TEACHER_CKPT)
    ap.add_argument("--train-data", type=str, default="./fineweb_edu_train")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    teacher, model_config, _ = load_teacher(args.teacher_ckpt, device)
    teacher.eval()
    enable_sdpa_attention(teacher)
    n_embd = model_config.n_embd

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    loader, _ = get_dataloader(
        args.train_data, tokenizer, args.batch_size, model_config.n_positions,
        num_workers=2, prefetch_factor=2, shuffle=False)
    it = iter(loader)
    fit_ids = next(it)["input_ids"].to(device)
    # mirrors kv_alignment_llm.py exactly: 8 disjoint batches for ridge selection,
    # then one more genuinely separate batch as this script's own held-out test set
    select_batches = []
    for _ in range(8):
        b = next(it)
        select_batches.append((b["input_ids"].to(device), b["labels"].to(device)))
    test_ids = next(it)["input_ids"].to(device)

    acts = capture_kv_activations(teacher, fit_ids, n_embd)
    test_acts = capture_kv_activations(teacher, test_ids, n_embd)

    print("Selecting linear map's ridge_frac (same procedure as kv_alignment_llm.py)...", flush=True)
    ridge_frac = select_ridge_frac_by_ppl(teacher, acts, n_embd, select_batches, device)
    print(f"-> selected ridge_frac={ridge_frac}\n", flush=True)

    rows = []
    for i in range(len(teacher.h)):
        K_fit, V_fit = acts[i]["k"], acts[i]["v"]
        K_te, V_te = test_acts[i]["k"], test_acts[i]["v"]

        baseline_cka = linear_cka(K_te, V_te)

        W_raw, b_raw = fit_linear_map(K_fit, V_fit, ridge_frac=ridge_frac)
        K_lin = K_te @ W_raw + b_raw
        lin_cka = linear_cka(K_lin, V_te)
        sv = torch.linalg.svdvals(W_raw)

        R, mean_k, std_k, mean_v, std_v = fit_procrustes(K_fit, V_fit)
        K_proc = (((K_te - mean_k) / std_k) @ R) * std_v + mean_v
        proc_cka = linear_cka(K_proc, V_te)

        row = {
            "layer": i,
            "cka_baseline": round(baseline_cka, 4),
            "cka_after_linear": round(lin_cka, 4),
            "cka_after_procrustes": round(proc_cka, 4),
            "linear_map_singular_values_mean": round(sv.mean().item(), 4),
            "linear_map_singular_values_std": round(sv.std().item(), 4),
        }
        rows.append(row)
        print(row, flush=True)

    with open("kv_correction_analysis_llm_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote kv_correction_analysis_llm_results.csv")


if __name__ == "__main__":
    main()
