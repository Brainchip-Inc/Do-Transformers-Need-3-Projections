"""
LLM counterpart of kv_similarity_analysis.py: activation-level linear CKA between
the trained K and V projections, computed on the 300M FineWeb-Edu QKV teacher
(outputs_qkv_baseline_10B/final_model.pt) instead of the synthetic/vision QKV
checkpoints. Same measurement (Kornblith et al. 2019 linear CKA, averaged over
layers), just adapted to this model's fused c_attn projection (Q/K/V packed into
one Linear, split by row/output-slice) instead of separate c_k/c_v Linears.

Appends a ("llm", "fineweb_edu_300m", ...) row to kv_similarity_results.csv,
alongside the existing synthetic/vision rows -- see kv_cka_correlation.py for how
this is plotted (as a reference line, not folded into the accuracy-drop
correlation, since the LLM's collapse severity is only measurable in perplexity,
not on the same 0-1 accuracy-drop scale as the other 9 points).

Usage:
    conda run -n kv python kv_similarity_llm.py --device cuda:0
"""

import csv
import argparse

import torch
from transformers import AutoTokenizer

from distillation_llm import load_teacher, TEACHER_CKPT
from transformer_KQV_300M_fineweb import get_dataloader

RESULTS_CSV = "kv_similarity_results.csv"


def linear_cka(X, Y):
    """Linear CKA between two [N, d] activation matrices (Kornblith et al. 2019).
    Copied from kv_similarity_analysis.py rather than imported: that module pulls
    in vision_tasks -> torchvision, which isn't installed in the `kv` conda env
    used for the LLM checkpoints."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (Y.T @ X).norm() ** 2
    normx = (X.T @ X).norm()
    normy = (Y.T @ Y).norm()
    return (hsic / (normx * normy + 1e-12)).item()


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


def weight_cosine_kv(c_attn, n_embd):
    w_k = c_attn.weight.data[n_embd:2 * n_embd]
    b_k = c_attn.bias.data[n_embd:2 * n_embd]
    w_v = c_attn.weight.data[2 * n_embd:3 * n_embd]
    b_v = c_attn.bias.data[2 * n_embd:3 * n_embd]
    wk = torch.cat([w_k.flatten(), b_k.flatten()])
    wv = torch.cat([w_v.flatten(), b_v.flatten()])
    return torch.nn.functional.cosine_similarity(wk, wv, dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--teacher-ckpt", type=str, default=TEACHER_CKPT)
    ap.add_argument("--val-data", type=str, default="./fineweb_edu_validation")
    ap.add_argument("--batch-size", type=int, default=8)
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

    acts = capture_kv_activations(teacher, input_ids, model_config.n_embd)
    cos_per_layer = [weight_cosine_kv(b.attn.c_attn, model_config.n_embd) for b in teacher.h]
    cka_per_layer = [linear_cka(acts[i]["k"], acts[i]["v"]) for i in range(len(teacher.h))]

    row = {
        "domain": "llm", "name": "fineweb_edu_300m",
        "weight_cos_sim": round(sum(cos_per_layer) / len(cos_per_layer), 4),
        "activation_cka": round(sum(cka_per_layer) / len(cka_per_layer), 4),
    }
    print(row, flush=True)
    print("Per-layer CKA:", [round(c, 4) for c in cka_per_layer], flush=True)

    rows = []
    try:
        with open(RESULTS_CSV) as f:
            rows = [r for r in csv.DictReader(f) if r["domain"] != "llm"]
    except FileNotFoundError:
        pass
    rows.append(row)
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "name", "weight_cos_sim", "activation_cka"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
