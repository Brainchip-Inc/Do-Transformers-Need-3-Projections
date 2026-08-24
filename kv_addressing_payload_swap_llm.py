"""
LLM counterpart of kv_addressing_payload_swap.py: the only genuinely new cell of
the addressing/payload decomposition -- A(Q,V)K (scores from V, payload from K)
-- evaluated zero-shot (no retraining) on the 300M FineWeb-Edu QKV teacher.

A(Q,K)V is the teacher itself (already known: val PPL 20.82); A(Q,K)K and
A(Q,V)V are exactly what keep_k/keep_v surgery already measured (13345.19 and
7347.14 PPL respectively, from distill_llm_{keep_k,keep_v}.log's zero-shot line)
-- see kv_addressing_payload_swap.py's module docstring for why tying c_kv to a
single shared projection collapses those two cells onto the existing surgery
numbers. Only A(Q,V)K needs a new eval.

Usage:
    conda run -n kv python kv_addressing_payload_swap_llm.py --device cuda:0
"""

import types
import math
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from transformer_KQV_300_M import evaluate
from distillation_llm import load_teacher, TEACHER_CKPT
from transformer_KQV_300M_fineweb import get_dataloader

# already-known cells (see module docstring)
TEACHER_PPL = 20.8183          # A(Q,K)V
KEEP_K_ZERO_SHOT_PPL = 13345.19  # A(Q,K)K
KEEP_V_ZERO_SHOT_PPL = 7347.14   # A(Q,V)V


def swapped_forward(self, x):
    """A(Q,V)K: split the fused c_attn output into (q, k, v) as usual, but use
    v for addressing and k for the transported payload -- roles swapped
    relative to QKVAttention.forward's own A(Q,K)V."""
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    k_role = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # addressing from V
    v_role = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # payload from K
    out = F.scaled_dot_product_attention(
        q, k_role, v_role, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    out = self.c_proj(out)
    out = self.resid_dropout(out)
    return out


def apply_swap(model):
    for block in model.h:
        block.attn.forward = types.MethodType(swapped_forward, block.attn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--teacher-ckpt", type=str, default=TEACHER_CKPT)
    ap.add_argument("--val-data", type=str, default="./fineweb_edu_validation")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=64)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    teacher, model_config, teacher_val = load_teacher(args.teacher_ckpt, device)
    teacher.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    val_loader, _ = get_dataloader(
        args.val_data, tokenizer, args.batch_size, model_config.n_positions,
        num_workers=2, prefetch_factor=2, shuffle=False)

    # sanity check: unmodified teacher forward must reproduce its own known val PPL
    baseline = evaluate(teacher, val_loader, args.eval_batches, device)
    print(f"A(Q,K)V sanity check: recomputed val_ppl={baseline['perplexity']:.2f} "
          f"vs checkpoint's stored {teacher_val['perplexity']:.2f}", flush=True)
    assert abs(baseline["perplexity"] - teacher_val["perplexity"]) < 2.0, \
        "A(Q,K)V baseline mismatch -- swap harness bug"

    apply_swap(teacher)
    swap_metrics = evaluate(teacher, val_loader, args.eval_batches, device)
    swap_ppl = swap_metrics["perplexity"]

    print(f"\nA(Q,K)V (teacher)   PPL = {TEACHER_PPL:.2f}")
    print(f"A(Q,K)K (keep_k)    PPL = {KEEP_K_ZERO_SHOT_PPL:.2f}")
    print(f"A(Q,V)V (keep_v)    PPL = {KEEP_V_ZERO_SHOT_PPL:.2f}")
    print(f"A(Q,V)K (new)       PPL = {swap_ppl:.2f}")


if __name__ == "__main__":
    main()
