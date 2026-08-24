"""
LLM counterpart of kv_scratch_vs_teacher.py: compares the from-scratch
Q!=K=V model's single learned projection (checkpoints_llm/qkv_keqv_300m_fineweb_edu.pt,
GPT_QKV_KEqualsV's c_k -- there is no separate c_v in that architecture, v:=k by
construction) to the QKV teacher's independent K and V, on held-out data.

Usage:
    conda run -n kv python kv_scratch_vs_teacher_llm.py --device cuda:0
"""

import csv
import argparse

import torch
from transformers import AutoTokenizer

from distillation_llm import load_teacher, TEACHER_CKPT, load_scratch_ceiling, SCRATCH_CKPT
from transformer_KQV_300M_fineweb import get_dataloader
from kv_similarity_llm import linear_cka


def capture_teacher_kv(model, input_ids, n_embd):
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


def capture_scratch_k(model, input_ids, n_embd):
    acts = {i: {} for i in range(len(model.h))}
    handles = []
    for i, block in enumerate(model.h):
        def mk(i):
            def hook(_mod, _inp, out):
                acts[i]["k"] = out.detach().reshape(-1, n_embd)
            return hook
        handles.append(block.attn.c_k.register_forward_hook(mk(i)))
    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--teacher-ckpt", type=str, default=TEACHER_CKPT)
    ap.add_argument("--scratch-ckpt", type=str, default=SCRATCH_CKPT)
    ap.add_argument("--val-data", type=str, default="./fineweb_edu_validation")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    teacher, model_config, _ = load_teacher(args.teacher_ckpt, device)
    teacher.eval()
    scratch, _, _ = load_scratch_ceiling(args.scratch_ckpt, device)
    scratch.eval()
    n_embd = model_config.n_embd

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    loader, _ = get_dataloader(
        args.val_data, tokenizer, args.batch_size, model_config.n_positions,
        num_workers=2, prefetch_factor=2, shuffle=False)
    input_ids = next(iter(loader))["input_ids"].to(device)

    t_acts = capture_teacher_kv(teacher, input_ids, n_embd)
    s_acts = capture_scratch_k(scratch, input_ids, n_embd)

    rows = []
    for i in range(len(teacher.h)):
        K, V_ = t_acts[i]["k"], t_acts[i]["v"]
        scratch_k = s_acts[i]["k"]
        row = {
            "layer": i,
            "cka_teacher_k_v": round(linear_cka(K, V_), 4),
            "cka_scratch_vs_teacher_k": round(linear_cka(scratch_k, K), 4),
            "cka_scratch_vs_teacher_v": round(linear_cka(scratch_k, V_), 4),
        }
        rows.append(row)
        print(row, flush=True)

    with open("kv_scratch_vs_teacher_llm_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote kv_scratch_vs_teacher_llm_results.csv")


if __name__ == "__main__":
    main()
