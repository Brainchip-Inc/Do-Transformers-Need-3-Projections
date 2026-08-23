"""
K=V tying study on the 300M-param FineWeb-Edu LLM: post-hoc weight surgery +
distillation recovery. LLM counterpart of distillation_vision.py /
distillation_synthetic.py -- same two questions, same surgery/KD mechanics,
applied to the causal-LM checkpoints trained by transformer_KQV_300M_fineweb.py
instead of the ViT/Encoder classifiers.

  (a) Post-hoc surgery: take the *trained* QKV teacher
      (outputs_qkv_baseline_10B/final_model.pt) and force K=V after the fact
      (no retraining) by dropping one of the teacher's fused c_attn Q/K/V
      slices and reusing the other (keep_k, keep_v) or averaging them (avg).
      Compare zero-shot val perplexity against the QKV teacher and the
      from-scratch Q!=K=V ceiling (checkpoints_llm/qkv_keqv_300m_fineweb_edu.pt).

  (b) Distillation recovery: fine-tune each surgically-tied student against
      the QKV teacher's soft logits (+ hard-label CE) for a fixed token
      budget and see how much of the gap opened by (a) is recovered relative
      to the from-scratch Q!=K=V ceiling.

Run in the `kv` conda env (same one used for transformer_KQV_300M_fineweb.py):
    conda run -n kv python distillation_llm.py --mode keep_k --smoke-test --device cuda:0
    conda run -n kv python distillation_llm.py --mode keep_k --device cuda:0

    # multi-GPU (DDP), one process per GPU, launched via torchrun:
    conda run -n kv torchrun --standalone --nproc_per_node=2 distillation_llm.py --mode keep_k
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import csv
import math
import types
import contextlib
import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

from transformer_KQV_300_M import (
    ModelConfig, TrainingConfig, GPT, TransformerBlock, count_parameters, get_lr,
    AverageMeter, evaluate,
)
from transformer_KV_1_300_M import GPT_QKV_KEqualsV
from transformer_KQV_300M_fineweb import (
    LocalTextDataset, get_dataloader, enable_sdpa_attention,
    enable_gradient_checkpointing, make_smoke_dataset,
)

TEACHER_CKPT = "outputs_qkv_baseline_10B/final_model.pt"
SCRATCH_CKPT = "checkpoints_llm/qkv_keqv_300m_fineweb_edu.pt"
MODES = ["keep_k", "keep_v", "avg"]


def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def load_teacher(path, device):
    d = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**d["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(_strip_module_prefix(d["model_state_dict"]))
    return model, cfg, d["val_metrics"]


def load_scratch_ceiling(path, device):
    d = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**d["model_config"])
    model = GPT_QKV_KEqualsV(cfg).to(device)
    model.load_state_dict(_strip_module_prefix(d["model_state_dict"]))
    return model, cfg, d["val_metrics"]


def surgically_tie_kv(teacher, model_config, mode, device):
    """Build a Q!=K=V (GPT_QKV_KEqualsV) student from a trained QKV teacher,
    with no retraining (see distillation_vision.py's surgically_tie_kv for the
    full rationale). The teacher's attention fuses Q/K/V into one c_attn
    Linear (shape (3*n_embd, n_embd)); split it by row into the three
    per-projection blocks that transformer_KQV_300_M.QKVAttention.forward
    reads via qkv.split(n_embd, dim=2)."""
    student = GPT_QKV_KEqualsV(model_config).to(device)
    # loads wte/wpe/ln_1/mlp/ln_2/c_proj/ln_f verbatim; attn.c_q/c_k have no
    # match in the teacher's state dict (which has attn.c_attn) and stay random.
    student.load_state_dict(teacher.state_dict(), strict=False)

    n_embd = model_config.n_embd
    for t_blk, s_blk in zip(teacher.h, student.h):
        t_attn, s_attn = t_blk.attn, s_blk.attn
        q_w, k_w, v_w = t_attn.c_attn.weight.data.split(n_embd, dim=0)
        q_b, k_b, v_b = t_attn.c_attn.bias.data.split(n_embd, dim=0)
        s_attn.c_q.weight.data.copy_(q_w)
        s_attn.c_q.bias.data.copy_(q_b)
        if mode == "keep_k":
            s_attn.c_k.weight.data.copy_(k_w)
            s_attn.c_k.bias.data.copy_(k_b)
        elif mode == "keep_v":
            s_attn.c_k.weight.data.copy_(v_w)
            s_attn.c_k.bias.data.copy_(v_b)
        elif mode == "avg":
            s_attn.c_k.weight.data.copy_((k_w + v_w) / 2)
            s_attn.c_k.bias.data.copy_((k_b + v_b) / 2)
        else:
            raise ValueError(mode)
    return student


def enable_sdpa_attention_keqv(model):
    """Q!=K=V counterpart of transformer_KQV_300M_fineweb.enable_sdpa_attention:
    same fused-kernel swap, but the manual math it replaces reuses c_k's
    output as v (see QKVAttention_KEqualsV.forward) instead of splitting a
    fused c_attn."""
    def sdpa_forward(self, x):
        B, T, C = x.size()
        q, k = self.c_q(x), self.c_k(x)
        v = k
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out
    for block in model.h:
        block.attn.forward = types.MethodType(sdpa_forward, block.attn)


def kd_loss(student_logits, teacher_logits, labels, temperature):
    """KL(student || teacher) over the next-token distribution, masked to the
    same positions the model's own internal CE scores (shift by one, ignore
    -100 padding targets) so padded tokens don't pollute the recovery signal."""
    s_shift = student_logits[..., :-1, :]
    t_shift = teacher_logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    mask = (shift_labels != -100)
    kd_per_tok = F.kl_div(
        F.log_softmax(s_shift / temperature, dim=-1),
        F.softmax(t_shift / temperature, dim=-1),
        reduction="none",
    ).sum(-1)
    denom = mask.sum().clamp(min=1)
    return (kd_per_tok * mask).sum() / denom * (temperature ** 2)


def distill(student, teacher, train_loader, train_sampler, val_loader, eval_batches,
            device, total_steps, grad_accum, distributed, lr, min_lr, warmup_steps,
            temperature, alpha, n_eval_points, is_main, log_prefix=""):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    lr_config = TrainingConfig(learning_rate=lr, min_lr=min_lr, warmup_steps=warmup_steps)
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    eval_every = max(1, total_steps // n_eval_points)
    ppls = []

    student.train()
    loss_meter = AverageMeter()
    if train_sampler is not None:
        train_sampler.set_epoch(0)
    train_iter = iter(train_loader)
    epoch = 0
    step = 0
    while step < total_steps:
        optimizer.zero_grad()
        for i in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            is_last_microbatch = i == grad_accum - 1
            sync_ctx = (contextlib.nullcontext() if not distributed or is_last_microbatch
                        else student.no_sync())
            with sync_ctx:
                with autocast(dtype=torch.bfloat16):
                    s_logits, ce = student(input_ids, labels=labels)
                    with torch.no_grad():
                        t_logits, _ = teacher(input_ids)
                    kd = kd_loss(s_logits, t_logits, labels, temperature)
                    loss = (alpha * ce + (1 - alpha) * kd) / grad_accum
                scaler.scale(loss).backward()
            loss_meter.update(loss.item() * grad_accum)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        lr_now = get_lr(step, lr_config, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        scaler.step(optimizer)
        scaler.update()
        if distributed:
            dist.broadcast(scaler._scale, src=0)
            dist.broadcast(scaler._growth_tracker, src=0)
        step += 1

        if is_main and step % 10 == 0:
            print(f"{log_prefix} step {step}/{total_steps} loss={loss_meter.avg:.4f} "
                  f"grad={grad_norm:.2f} lr={lr_now:.2e}", flush=True)
            loss_meter.reset()

        if step % eval_every == 0 or step == total_steps:
            if is_main:
                val_metrics = evaluate(student, val_loader, eval_batches, device)
                ppls.append(val_metrics["perplexity"])
                print(f"{log_prefix} [EVAL] step {step}/{total_steps} "
                      f"val_ppl={val_metrics['perplexity']:.2f}", flush=True)
            if distributed:
                dist.barrier()
            student.train()

    while len(ppls) < n_eval_points:
        ppls.append(ppls[-1] if ppls else float("nan"))
    # Keep the tail, not the head: the final step's eval (the true end-of-budget
    # recovery reading) must survive even when rounding produces more than
    # n_eval_points evals (e.g. eval_every=1 for a tiny total_steps).
    return ppls[-n_eval_points:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=MODES)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--teacher-ckpt", type=str, default=TEACHER_CKPT)
    ap.add_argument("--scratch-ckpt", type=str, default=SCRATCH_CKPT)
    ap.add_argument("--total-tokens", type=int, default=500_000_000)
    ap.add_argument("--micro-batch-size", type=int, default=4)
    ap.add_argument("--eval-points", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--min-lr", type=float, default=5e-6)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on hard-label CE vs KD")
    ap.add_argument("--train-data", type=str, default="./fineweb_edu_train")
    ap.add_argument("--val-data", type=str, default="./fineweb_edu_validation")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--prefetch-factor", type=int, default=6)
    ap.add_argument("--eval-tokens", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--out-csv", type=str, default=None,
                     help="default: distill_llm_{mode}_results.csv")
    args = ap.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device) if args.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    torch.manual_seed(args.seed)

    train_data, val_data = args.train_data, args.val_data
    if args.smoke_test:
        train_data, val_data = "./smoke_fineweb_train", "./smoke_fineweb_validation"
        make_smoke_dataset(train_data)
        make_smoke_dataset(val_data, n=40)

    if is_main:
        print(f"device={device} world_size={world_size} mode={args.mode}", flush=True)
        print("Loading teacher + scratch ceiling checkpoints...", flush=True)
    teacher, model_config, teacher_val = load_teacher(args.teacher_ckpt, device)
    _, _, scratch_val = load_scratch_ceiling(args.scratch_ckpt, device)
    teacher.eval()
    enable_sdpa_attention(teacher)

    student = surgically_tie_kv(teacher, model_config, args.mode, device)
    enable_sdpa_attention_keqv(student)
    enable_gradient_checkpointing(student)
    if is_main:
        params = count_parameters(student)
        print(f"Student parameters: {params['total_M']:.2f}M", flush=True)

    reference_tpb = 6 * 24  # matches the reference micro_batch_size=6/grad_accum=24 baseline
    assert reference_tpb % args.micro_batch_size == 0, \
        f"micro_batch_size {args.micro_batch_size} must divide {reference_tpb}"
    grad_accum = reference_tpb // args.micro_batch_size
    if world_size > 1:
        assert grad_accum % world_size == 0, \
            f"gradient_accumulation {grad_accum} must divide evenly by world_size {world_size}"
        grad_accum //= world_size

    total_tokens = args.total_tokens
    num_workers, prefetch_factor, eval_tokens = args.num_workers, args.prefetch_factor, args.eval_tokens
    if args.smoke_test:
        tpb = args.micro_batch_size * model_config.n_positions * grad_accum
        total_tokens = tpb * (world_size if world_size > 1 else 1) * 3
        num_workers, prefetch_factor = 0, None
        eval_tokens = args.micro_batch_size * model_config.n_positions * 2
        args.eval_points = min(args.eval_points, 2)

    if distributed:
        student = DDP(student, device_ids=[local_rank], broadcast_buffers=False)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print("Loading data...", flush=True)
    train_loader, train_sampler = get_dataloader(
        train_data, tokenizer, args.micro_batch_size, model_config.n_positions,
        num_workers, prefetch_factor, shuffle=True, world_size=world_size, rank=rank, seed=args.seed)
    val_loader = None
    if is_main:
        val_loader, _ = get_dataloader(
            val_data, tokenizer, args.micro_batch_size, model_config.n_positions,
            num_workers, prefetch_factor, shuffle=False)

    tokens_per_batch = args.micro_batch_size * model_config.n_positions * grad_accum * world_size
    total_steps = max(1, total_tokens // tokens_per_batch)
    eval_batches = max(1, eval_tokens // (args.micro_batch_size * model_config.n_positions))
    if is_main:
        print(f"tokens_per_batch={tokens_per_batch:,} total_steps={total_steps:,}", flush=True)

    zero_shot_ppl = None
    if is_main:
        zs_metrics = evaluate(student, val_loader, eval_batches, device)
        zero_shot_ppl = zs_metrics["perplexity"]
        print(f"[{args.mode}] zero-shot surgery val_ppl={zero_shot_ppl:.2f}", flush=True)
    if distributed:
        dist.barrier()

    distill_ppls = distill(
        student, teacher, train_loader, train_sampler, val_loader, eval_batches, device,
        total_steps, grad_accum, distributed, args.lr, args.min_lr, args.warmup_steps,
        args.temperature, args.alpha, args.eval_points, is_main, log_prefix=f"[{args.mode}]")

    if is_main:
        row = {
            "mode": args.mode,
            "teacher_ppl": round(teacher_val["perplexity"], 4),
            "scratch_qkv_kv_ppl": round(scratch_val["perplexity"], 4),
            "zero_shot_surgery_ppl": round(zero_shot_ppl, 4),
        }
        for i, ppl in enumerate(distill_ppls):
            row[f"distill_point{i + 1}_ppl"] = round(ppl, 4)
        out_csv = args.out_csv or f"distill_llm_{args.mode}_results.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"[{args.mode}] {row}", flush=True)
        print(f"Wrote {out_csv}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
