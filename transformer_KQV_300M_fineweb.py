"""
Single-GPU training script for the 300M QKV (full, separate Q/K/V) baseline on
FineWeb-Edu, built to exactly match the existing Q!=K=V checkpoint
(checkpoints_llm/qkv_keqv_300m_fineweb_edu.pt) for a fair comparison: same model
hyperparameters, same optimizer/schedule/token budget/seed, same data source
(FineWeb-Edu sample-10BT, see download_fineweb_edu.py). The GPT/ModelConfig/
TrainingConfig classes are imported unmodified from transformer_KQV_300_M.py;
only the training harness differs from that reference file:

  - optional multi-GPU DDP (see Usage below) to cut wall-clock on this box's two
    A30s. The global effective batch (tokens_per_batch = micro_batch_size *
    n_positions * gradient_accumulation * world_size) is kept equal to the
    reference run's regardless of world_size -- gradient_accumulation scales
    inversely with world_size -- so total_steps and the LR schedule are
    identical to a single-GPU run; DDP only parallelizes the same schedule
    across more GPUs, it doesn't change what's being trained.
  - reads FineWeb-Edu (./fineweb_edu_train, ./fineweb_edu_validation) instead of
    local SlimPajama.
  - fixes the padding-loss-masking bug documented on the Q!=K=V model card
    ("short documents were previously scored on their padding"): the reference
    script's LocalSlimPajamaDataset pads short documents with the EOS token and
    then uses the padded sequence as its own label with no masking, so every
    padding position after the first is scored as a next-token prediction
    target. LocalTextDataset here keeps exactly one EOS as a legitimate
    "document ends here" target (standard practice) and masks everything after
    it to ignore_index=-100.
  - adds checkpoint-based resume (absent from the reference script) since an
    unattended ~45h run without it risks losing most of a multi-day GPU
    allocation to a single crash; training config/data/schedule are otherwise
    unchanged, so a resume continues the same run in every way that affects the
    comparison (it only reshuffles the data loader, which is immaterial for LM
    pretraining).
  - checkpoints use the exact same top-level keys as the existing checkpoint
    (step, epoch, tokens_seen, model_state_dict, val_metrics, model_config,
    train_config) so downstream tooling (surgery/distillation) loads either
    checkpoint interchangeably.
  - enables gradient checkpointing by default: the reference script's
    use_flash_attention=True config assumes flash-attn is installed (it isn't,
    here), and standard attention's O(T^2) score matrix across 20 layers OOMs
    a 24GB GPU at the reference micro_batch_size=6. Checkpointing is applied by
    monkey-patching each TransformerBlock instance's bound forward method
    (not by wrapping it in a new submodule), so state-dict keys are byte-for-
    byte identical to the unwrapped model and still interchangeable with the
    existing checkpoint.
  - enables fused SDPA attention by default: same causal scaled-dot-product-
    attention computation as the reference file's manual matmul+softmax
    fallback (used whenever flash_attn isn't installed), just routed through
    torch.nn.functional.scaled_dot_product_attention's fused CUDA kernel
    instead of materializing the full T*T score matrix -- ~4x faster here,
    no flash_attn pip package needed. Also monkey-patched onto the instance,
    so state-dict keys are unaffected.

Usage:
    conda run -n kv python transformer_KQV_300M_fineweb.py --device cuda:0
    conda run -n kv python transformer_KQV_300M_fineweb.py --smoke-test   # a few steps, synthetic data

    # multi-GPU (DDP), one process per GPU, launched via torchrun:
    conda run -n kv torchrun --standalone --nproc_per_node=2 transformer_KQV_300M_fineweb.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import contextlib
import glob
import math
import time
import types
import argparse
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset as HFDataset

from transformer_KQV_300_M import (
    ModelConfig, TrainingConfig, GPT, TransformerBlock, count_parameters, get_lr,
    AverageMeter, evaluate,
)


def enable_gradient_checkpointing(model):
    """Monkey-patch each block's bound forward method to checkpoint through
    torch.utils.checkpoint, trading recompute for activation memory. Patches
    the instance's forward attribute directly (no wrapper submodule), so
    model.state_dict() keys are unaffected."""
    def checkpointed_forward(self, x):
        return grad_checkpoint(TransformerBlock.forward, self, x, use_reentrant=False)
    for block in model.h:
        block.forward = types.MethodType(checkpointed_forward, block)


def enable_sdpa_attention(model):
    """Monkey-patch each QKVAttention instance's forward to route through
    torch's fused scaled_dot_product_attention instead of the reference
    file's manual matmul+softmax fallback (used whenever the flash_attn pip
    package isn't installed, which it isn't here). Same causal scaled-dot-
    product-attention computation, just the fused CUDA kernel instead of
    materializing the full T*T score matrix -- no flash_attn install needed.
    Patches the instance's forward attribute directly (no wrapper submodule),
    so model.state_dict() keys are unaffected."""
    def sdpa_forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out
    for block in model.h:
        block.attn.forward = types.MethodType(sdpa_forward, block.attn)


class LocalTextDataset(Dataset):
    """On-the-fly tokenizing text dataset with the padding-loss-masking fix
    (see module docstring). Expects a 'text' column, as produced by
    download_fineweb_edu.py."""

    def __init__(self, data_path, tokenizer, seq_length, max_examples=None):
        print(f"Loading dataset from {data_path}...", flush=True)
        self.dataset = load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        if max_examples is not None:
            self.dataset = self.dataset.select(range(min(max_examples, len(self.dataset))))
        print(f"Loaded {len(self.dataset):,} examples", flush=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        tokens = self.tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=self.seq_length + 1)
        orig_len = len(tokens)
        if orig_len < self.seq_length + 1:
            tokens = tokens + [self.tokenizer.eos_token_id] * (self.seq_length + 1 - orig_len)

        input_ids = torch.tensor(tokens[:self.seq_length], dtype=torch.long)
        labels = input_ids.clone()
        if orig_len <= self.seq_length:
            keep_upto = min(orig_len + 1, self.seq_length)  # keep one EOS as a real target
            labels[keep_upto:] = -100
        return {"input_ids": input_ids, "labels": labels}


def get_dataloader(data_path, tokenizer, batch_size, seq_length, num_workers, prefetch_factor,
                    shuffle, max_examples=None, world_size=1, rank=0, seed=0):
    dataset = LocalTextDataset(data_path, tokenizer, seq_length, max_examples)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                      shuffle=shuffle, seed=seed, drop_last=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(shuffle and sampler is None),
        sampler=sampler, num_workers=num_workers, pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0, drop_last=True,
    )
    return loader, sampler


def make_smoke_dataset(path, n=200, seq_chars=6000):
    if os.path.isdir(path):
        return
    texts = [("The quick brown fox jumps over the lazy dog. " * 40)[:seq_chars]
             if i % 2 == 0 else "Short doc. " for i in range(n)]
    HFDataset.from_dict({"text": texts}).save_to_disk(path)


def latest_checkpoint(output_dir):
    ckpts = sorted(glob.glob(f"{output_dir}/checkpoint_step_*.pt"),
                    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    return ckpts[-1] if ckpts else None


def train(args):
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
        device = torch.device(args.device)
    is_main = rank == 0
    torch.manual_seed(args.seed)

    train_data, val_data, output_dir = args.train_data, args.val_data, args.output_dir
    if args.smoke_test:
        # never collide with the real (possibly in-progress) download directories
        train_data, val_data, output_dir = "./smoke_fineweb_train", "./smoke_fineweb_validation", "./smoke_outputs"

    model_config = ModelConfig()
    train_config = TrainingConfig(
        train_data_path=train_data, val_data_path=val_data,
        output_dir=output_dir,
        warmup_steps=20, save_interval=5000, eval_interval=500, log_interval=5,
        eval_tokens=1_000_000, clearml_task="GPT-300M-QKV-Baseline-10B",
    )
    if args.micro_batch_size is not None:
        # Without flash-attn, standard attention materializes an O(T^2) score matrix
        # per layer that doesn't fit at the reference micro_batch_size=6 on a 24GB GPU.
        # tokens_per_batch (= micro_batch_size * n_positions * gradient_accumulation)
        # must stay fixed at the reference value for the LR schedule/total_steps/
        # effective batch size to match the Q!=K=V run exactly, so grad_accum scales
        # inversely with micro_batch_size.
        reference_tpb = train_config.micro_batch_size * train_config.gradient_accumulation
        assert reference_tpb % args.micro_batch_size == 0, \
            f"micro_batch_size {args.micro_batch_size} must divide {reference_tpb}"
        train_config.gradient_accumulation = reference_tpb // args.micro_batch_size
        train_config.micro_batch_size = args.micro_batch_size
    if world_size > 1:
        # Keep the GLOBAL effective batch (tokens_per_batch * world_size) equal to the
        # single-GPU reference value: each rank does 1/world_size of the accumulation,
        # and the all-reduced gradient at sync time covers the same global batch the
        # single-GPU run would have accumulated alone. This preserves total_steps and
        # the LR schedule exactly -- DDP parallelizes the same run, doesn't grow it.
        assert train_config.gradient_accumulation % world_size == 0, \
            (f"gradient_accumulation {train_config.gradient_accumulation} must divide "
             f"evenly by world_size {world_size}")
        train_config.gradient_accumulation //= world_size
    if args.total_tokens is not None:
        train_config.total_tokens = args.total_tokens
    max_examples = None
    if args.smoke_test:
        make_smoke_dataset(train_data)
        make_smoke_dataset(val_data, n=40)
        tpb = train_config.micro_batch_size * model_config.n_positions * train_config.gradient_accumulation
        train_config.total_tokens = tpb * 3
        train_config.eval_interval = 2
        train_config.save_interval = 2
        train_config.eval_tokens = train_config.micro_batch_size * model_config.n_positions * 2
        train_config.num_workers = 0
        train_config.prefetch_factor = None

    os.makedirs(train_config.output_dir, exist_ok=True)

    if is_main:
        print("=" * 80); print("INITIALIZING MODEL"); print("=" * 80, flush=True)
    model = GPT(model_config).to(device)
    params = count_parameters(model)
    if is_main:
        print(f"Total parameters: {params['total_M']:.2f}M", flush=True)
    if args.sdpa_attention:
        enable_sdpa_attention(model)
        if is_main:
            print("Fused SDPA attention enabled", flush=True)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
        if is_main:
            print("Gradient checkpointing enabled", flush=True)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2), weight_decay=train_config.weight_decay)

    step, tokens_seen, epoch = 0, 0, 0
    resume_path = None if args.smoke_test else latest_checkpoint(train_config.output_dir)
    if resume_path:
        if is_main:
            print(f"Resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step, tokens_seen, epoch = ckpt["step"], ckpt["tokens_seen"], ckpt["epoch"]

    if distributed:
        # Wrapped only after the raw model's state dict is loaded above, so
        # checkpoint keys never pick up DDP's "module." prefix; saving/loading
        # later goes through raw_model (below) to keep that format unchanged.
        # broadcast_buffers=False: the model's only buffer is QKVAttention's
        # causal mask (registered but unused once enable_sdpa_attention patches
        # attention forward) -- a fixed constant derived only from n_positions,
        # identical on every rank by construction, never learned or mutated.
        # DDP's default (True) would broadcast it from rank 0 on every single
        # forward() call, which (a) wastes bandwidth every training step and
        # (b) deadlocks evaluate(), where only rank 0 calls forward(): rank 0
        # blocks on the buffer-broadcast collective while every other rank has
        # already moved on to a different collective (the next barrier).
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    raw_model = model.module if distributed else model

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print("=" * 80); print("LOADING DATA"); print("=" * 80, flush=True)
    train_loader, train_sampler = get_dataloader(
        train_config.train_data_path, tokenizer, train_config.micro_batch_size,
        model_config.n_positions, train_config.num_workers, train_config.prefetch_factor,
        shuffle=True, max_examples=max_examples, world_size=world_size, rank=rank, seed=args.seed)
    val_loader = None
    if is_main:
        val_loader, _ = get_dataloader(
            train_config.val_data_path, tokenizer, train_config.micro_batch_size,
            model_config.n_positions, train_config.num_workers, train_config.prefetch_factor,
            shuffle=False, max_examples=max_examples)

    tokens_per_batch = (train_config.micro_batch_size * model_config.n_positions
                        * train_config.gradient_accumulation * world_size)
    total_steps = train_config.total_tokens // tokens_per_batch
    eval_batches = max(1, train_config.eval_tokens // (train_config.micro_batch_size * model_config.n_positions))

    if is_main:
        print(f"Tokens per batch: {tokens_per_batch:,}", flush=True)
        print(f"Total steps: {total_steps:,}", flush=True)

    model.train()
    loss_meter = AverageMeter()
    start_time = time.time()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)

    while step < total_steps:
        optimizer.zero_grad()
        for i in range(train_config.gradient_accumulation):
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
            # Only all-reduce gradients on the last microbatch of the accumulation
            # window -- syncing every microbatch would be correct but wastes a
            # collective per microbatch instead of one per optimizer step.
            is_last_microbatch = i == train_config.gradient_accumulation - 1
            sync_ctx = (contextlib.nullcontext() if not distributed or is_last_microbatch
                        else model.no_sync())
            with sync_ctx:
                with autocast(dtype=torch.bfloat16):
                    _, loss = model(input_ids, labels=labels)
                loss = loss / train_config.gradient_accumulation
                scaler.scale(loss).backward()
            loss_meter.update(loss.item() * train_config.gradient_accumulation)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        if grad_norm > 100.0:
            if is_main:
                print(f"WARNING: grad norm {grad_norm:.2f} too high, skipping update", flush=True)
            optimizer.zero_grad()
            scaler.update()
            if distributed:
                dist.broadcast(scaler._scale, src=0)
                dist.broadcast(scaler._growth_tracker, src=0)
            continue

        lr = get_lr(step, train_config, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        if distributed:
            # Each rank's GradScaler tracks inf/nan overflow independently off its
            # own data shard; without this, a rank-local overflow could desync the
            # scale factor across ranks, and DDP would then all-reduce gradients
            # scaled by different factors on each rank on the next step -- silently
            # corrupting the averaged gradient. Rebroadcast rank 0's scale state
            # every step to keep all ranks' scalers identical.
            dist.broadcast(scaler._scale, src=0)
            dist.broadcast(scaler._growth_tracker, src=0)

        step += 1
        tokens_seen += tokens_per_batch

        if step % train_config.log_interval == 0:
            avg_loss = loss_meter.avg
            if distributed:
                # Each rank's loss_meter only reflects its own data shard;
                # average across ranks so the logged number reflects the
                # full global batch, matching what a single-GPU run would print.
                loss_tensor = torch.tensor(avg_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss = loss_tensor.item()
            if is_main:
                elapsed = time.time() - start_time
                print(f"Step {step}/{total_steps} | Epoch {epoch} | Loss {avg_loss:.4f} | "
                      f"PPL {math.exp(avg_loss):.2f} | LR {lr:.2e} | "
                      f"Tok/s {tokens_seen / elapsed:.0f} | Grad {grad_norm:.2f}", flush=True)
            loss_meter.reset()

        if step % train_config.eval_interval == 0:
            if is_main:
                val_metrics = evaluate(model, val_loader, eval_batches, device)
                print(f"[EVAL] step {step} val_loss={val_metrics['loss']:.4f} "
                      f"val_ppl={val_metrics['perplexity']:.2f}", flush=True)
            if distributed:
                dist.barrier()

        if step % train_config.save_interval == 0:
            if is_main:
                ckpt_path = f"{train_config.output_dir}/checkpoint_step_{step}.pt"
                torch.save({
                    "step": step, "epoch": epoch, "tokens_seen": tokens_seen,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": asdict(model_config), "train_config": asdict(train_config),
                }, ckpt_path)
                print(f"[CHECKPOINT] saved {ckpt_path}", flush=True)
            if distributed:
                dist.barrier()

    if is_main:
        print("=" * 80); print("FINAL EVALUATION"); print("=" * 80, flush=True)
        val_metrics = evaluate(model, val_loader, eval_batches, device)
        print(f"Final val loss={val_metrics['loss']:.4f} ppl={val_metrics['perplexity']:.2f}", flush=True)

        final_path = f"{train_config.output_dir}/final_model.pt"
        torch.save({
            "step": step, "epoch": epoch, "tokens_seen": tokens_seen,
            "model_state_dict": raw_model.state_dict(), "val_metrics": val_metrics,
            "model_config": asdict(model_config), "train_config": asdict(train_config),
        }, final_path)
        print(f"Saved {final_path}. Total time: {(time.time() - start_time) / 3600:.2f}h", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train-data", default="./fineweb_edu_train")
    ap.add_argument("--val-data", default="./fineweb_edu_validation")
    ap.add_argument("--output-dir", default="./outputs_qkv_baseline_10B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--micro-batch-size", type=int, default=None,
                     help="override micro_batch_size, scaling gradient_accumulation inversely "
                          "so tokens_per_batch/total_steps/LR schedule stay unchanged "
                          "(extra safety margin on top of --gradient-checkpointing if still needed)")
    ap.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                     action="store_true", default=True,
                     help="checkpoint each block's activations (default on; no flash-attn here)")
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                     action="store_false")
    ap.add_argument("--sdpa-attention", dest="sdpa_attention",
                     action="store_true", default=True,
                     help="use torch's fused scaled_dot_product_attention instead of the "
                          "reference file's manual matmul+softmax fallback (default on; "
                          "same computation, no flash_attn pip package needed)")
    ap.add_argument("--no-sdpa-attention", dest="sdpa_attention", action="store_false")
    ap.add_argument("--total-tokens", type=int, default=None,
                     help="override total_tokens (e.g. for a quick throughput benchmark)")
    ap.add_argument("--smoke-test", action="store_true", help="a few steps on synthetic data")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
