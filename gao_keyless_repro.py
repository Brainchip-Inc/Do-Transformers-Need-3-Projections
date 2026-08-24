"""
Small-scale reproduction of Gao & Xu, "Keyless Attention: Value-Space Routing
and Value-Only Caching for Efficient Transformers" (arXiv:2606.21848) --
perplexity/LM comparison only (their smallest model: GPT-2, 12 layers, 1024
hidden, 8 heads, ~280M params), matching their reported setup: a 30M-token
subset of WikiText-103 (see download_wikitext103.py), AdamW (lr=1e-4,
weight_decay=0.01), linear schedule with 5% warmup, 10 epochs.

Keyless attention (their Eq. 2): softmax(X W^Q W^R (X W^V)^T / sqrt(dk)) X W^V
-- replaces the K projection with a per-head routing matrix W^R applied to Q,
then scored against V directly. Implemented here via a single
scaled_dot_product_attention call with V passed as BOTH the "key" and "value"
argument (after Q has been transformed by R) -- mathematically identical to
the paper's formula, and reuses the fused kernel instead of a manual
matmul+softmax (same trick already used for the K=V-tied variant in
distillation_llm.py::enable_sdpa_attention_keqv).

Reuses ModelConfig/TrainingConfig/GPT/evaluate/AverageMeter/count_parameters
from transformer_KQV_300_M.py (generic, attention-independent) and
LocalTextDataset/get_dataloader/enable_sdpa_attention from
transformer_KQV_300M_fineweb.py; only the Keyless-specific attention/block/
model classes and the linear LR schedule (Gao uses linear decay for GPT-2,
not this repo's usual cosine) are new.

Usage:
    conda run -n kv python gao_keyless_repro.py --variant QKV --device cuda:0
    conda run -n kv python gao_keyless_repro.py --variant Keyless --device cuda:0
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import math
import time
import types
import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

from transformer_KQV_300_M import (
    ModelConfig, TrainingConfig, GPT, MLP, count_parameters, AverageMeter, evaluate,
)
from transformer_KQV_300M_fineweb import (
    LocalTextDataset, get_dataloader, enable_sdpa_attention, enable_gradient_checkpointing,
)

TRAIN_DATA = "./wikitext103_train"
VAL_DATA = "./wikitext103_validation"


# ============================================================================
# KEYLESS ATTENTION (Gao & Xu, arXiv:2606.21848, Eq. 2)
# ============================================================================

class KeylessAttention(nn.Module):
    """s_ij = (x_i W^Q W^R)(x_j W^V)^T / sqrt(dk), causal. W^R is a per-head
    routing matrix (not low-rank -- the paper's rank constraint comes from
    W^Q W^R's composition, d x dk, not from W^R itself)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd, self.n_head = config.n_embd, config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.attn_pdrop

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        self.R = nn.Parameter(torch.randn(self.n_head, self.head_dim, self.head_dim)
                               * config.initializer_range)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        qR = torch.einsum("bhid,hde->bhie", q, self.R)  # q transformed by the routing matrix

        # softmax(qR @ v^T / sqrt(dk)) @ v -- SDPA with v as both "key" and "value"
        # computes exactly this (same trick as enable_sdpa_attention_keqv).
        out = F.scaled_dot_product_attention(
            qR, v, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out


class KeylessBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = KeylessAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT_Keyless(nn.Module):
    """Same structure as transformer_KQV_300_M.GPT, KeylessAttention in place
    of QKVAttention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([KeylessBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        if config.tie_word_embeddings:
            self.lm_head = lambda x: F.linear(x, self.wte.weight)
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, labels=None):
        device = input_ids.device
        b, t = input_ids.size()
        tok_emb = self.wte(input_ids)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                ignore_index=-100)
        return logits, loss


def enable_sdpa_attention_keyless(model):
    """No-op in effect (KeylessAttention already uses SDPA directly), kept
    only so the calling code can treat both variants uniformly."""
    pass


def get_lr_linear(step, config, total_steps):
    """Gao & Xu's schedule for GPT-2: linear decay with 5% linear warmup
    (this repo's other scripts default to cosine; matching the paper's own
    choice here rather than reusing get_lr)."""
    warmup_steps = config.warmup_steps
    if step < warmup_steps:
        return config.learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return config.learning_rate + (config.min_lr - config.learning_rate) * progress


def build_model(variant, model_config, device):
    if variant == "QKV":
        model = GPT(model_config).to(device)
    elif variant == "Keyless":
        model = GPT_Keyless(model_config).to(device)
    else:
        raise ValueError(variant)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, required=True, choices=["QKV", "Keyless"])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--micro-batch-size", type=int, default=8)
    ap.add_argument("--gradient-accumulation", type=int, default=4)
    ap.add_argument("--n-positions", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir or f"./outputs_gao_repro_{args.variant.lower()}"
    os.makedirs(output_dir, exist_ok=True)

    # Gao & Xu's smallest model: GPT-2, 12 layers, 1024 hidden, 8 heads (~280M params)
    model_config = ModelConfig(
        vocab_size=50304, n_positions=args.n_positions, n_layer=12, n_embd=1024,
        n_head=8, n_inner=4096, use_flash_attention=False,
    )
    train_config = TrainingConfig(
        train_data_path=TRAIN_DATA, val_data_path=VAL_DATA, output_dir=output_dir,
        learning_rate=1e-4, min_lr=1e-4 * 0.1, weight_decay=0.01,
        micro_batch_size=args.micro_batch_size, gradient_accumulation=args.gradient_accumulation,
        eval_interval=200, save_interval=100000, log_interval=10,
        eval_tokens=1_000_000, clearml_task=f"Gao-Keyless-Repro-{args.variant}",
    )

    tpb = train_config.micro_batch_size * model_config.n_positions * train_config.gradient_accumulation
    if args.smoke_test:
        train_config.train_data_path = TRAIN_DATA  # reuse real (small) data, just fewer steps
        train_config.total_tokens = tpb * 3
        train_config.eval_interval = 2
        train_config.eval_tokens = train_config.micro_batch_size * model_config.n_positions * 2
        args.epochs = None
    else:
        # 10 epochs over the ~30M-token WikiText-103 subset (see download_wikitext103.py)
        train_config.total_tokens = 30_000_000 * args.epochs
    train_config.warmup_steps = 0  # set below once total_steps is known (needs tpb)

    print(f"device={device} variant={args.variant}", flush=True)
    model = build_model(args.variant, model_config, device)
    params = count_parameters(model)
    print(f"Total parameters: {params['total_M']:.2f}M", flush=True)

    if args.variant == "QKV":
        enable_sdpa_attention(model)
    enable_gradient_checkpointing(model) if args.variant == "QKV" else None
    if args.variant == "Keyless":
        # KeylessBlock/GPT_Keyless share TransformerBlock's forward shape
        # (x + attn(ln1(x)); x + mlp(ln2(x))), so the generic checkpointing
        # patch works unchanged -- see enable_gradient_checkpointing's own
        # docstring for why this duck-typing is safe.
        enable_gradient_checkpointing(model)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2), weight_decay=train_config.weight_decay)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading data...", flush=True)
    train_loader, _ = get_dataloader(
        train_config.train_data_path, tokenizer, train_config.micro_batch_size,
        model_config.n_positions, train_config.num_workers, train_config.prefetch_factor,
        shuffle=True, seed=args.seed)
    val_loader, _ = get_dataloader(
        train_config.val_data_path, tokenizer, train_config.micro_batch_size,
        model_config.n_positions, train_config.num_workers, train_config.prefetch_factor,
        shuffle=False)

    tokens_per_batch = train_config.micro_batch_size * model_config.n_positions * train_config.gradient_accumulation
    total_steps = max(1, train_config.total_tokens // tokens_per_batch)
    train_config.warmup_steps = max(1, int(0.05 * total_steps))  # Gao & Xu: 5% linear warmup
    eval_batches = max(1, train_config.eval_tokens // (train_config.micro_batch_size * model_config.n_positions))
    print(f"tokens_per_batch={tokens_per_batch:,} total_steps={total_steps:,} "
          f"warmup_steps={train_config.warmup_steps}", flush=True)

    model.train()
    loss_meter = AverageMeter()
    start_time = time.time()
    train_iter = iter(train_loader)
    step, epoch = 0, 0
    while step < total_steps:
        optimizer.zero_grad()
        for i in range(train_config.gradient_accumulation):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                train_iter = iter(train_loader)
                batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast(dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)
            loss = loss / train_config.gradient_accumulation
            scaler.scale(loss).backward()
            loss_meter.update(loss.item() * train_config.gradient_accumulation)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        lr = get_lr_linear(step, train_config, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        step += 1

        if step % train_config.log_interval == 0:
            print(f"[{args.variant}] step {step}/{total_steps} epoch {epoch} "
                  f"loss={loss_meter.avg:.4f} ppl={math.exp(loss_meter.avg):.2f} "
                  f"lr={lr:.2e} grad={grad_norm:.2f}", flush=True)
            loss_meter.reset()

        if step % train_config.eval_interval == 0 or step == total_steps:
            val_metrics = evaluate(model, val_loader, eval_batches, device)
            print(f"[{args.variant}] [EVAL] step {step}/{total_steps} "
                  f"val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['perplexity']:.2f}",
                  flush=True)

    print("=" * 60, flush=True)
    val_metrics = evaluate(model, val_loader, eval_batches, device)
    print(f"[{args.variant}] FINAL val_loss={val_metrics['loss']:.4f} "
          f"val_ppl={val_metrics['perplexity']:.2f}. Total time: "
          f"{(time.time() - start_time) / 3600:.2f}h", flush=True)
    torch.save({
        "step": step, "epoch": epoch, "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics, "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }, f"{output_dir}/final_model.pt")
    print(f"Saved {output_dir}/final_model.pt", flush=True)


if __name__ == "__main__":
    main()
