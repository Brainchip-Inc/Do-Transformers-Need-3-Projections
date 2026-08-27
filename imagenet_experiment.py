"""
ImageNet-1k ViT-S/16 comparison across attention conditions (paper design doc,
paper/main.tex): QKV (baseline), Q-K=V, Q=K=V+, QVV(3), Q=K-V+, and Q=K=V (no +).
Reuses the same ViT / SharedProjAttention implementation as vision_tasks.py
(MNIST/CIFAR/TinyImageNet), with a DeiT-style training recipe (RandAugment,
Mixup/CutMix, stochastic depth, label smoothing) added on top, since that recipe
is required to train a ViT from scratch on ImageNet-1k without ImageNet-21k
pretraining.

Condition -> internal variant name (see synthetic_tasks.VARIANTS):
    QKV       -> "QKV"        separate Q, K, V
    Q-K=V     -> "Q!=K=V"     separate Q; shared K=V (headline variant, halves cache)
    Q=K=V+    -> "(Q=K=V)+"   single projection for Q/K/V, 2D positional correction
    QVV(3)    -> "QVV(3)"     depth-2-factored Q; shared K=V; matches QKV param count
    Q=K-V+    -> "(Q=K!=V)+"  shared Q=K; separate V; 2D positional correction
    Q=K=V (no +) -> "Q=K=V"   single projection for Q/K/V, no positional correction

Run environment: torch_env (torch 2.7 + torchvision 0.22 + timm 1.0.22), confirmed
working on this machine's RTX 2080 Ti (11 GB) GPUs.

Usage:
    # smoke test: tiny subset, 1 epoch, verifies the pipeline + reports img/sec
    conda run -n torch_env python imagenet_experiment.py --variant QKV --smoke-test --device cuda:0

    # full run (one condition per GPU, run these four in parallel)
    conda run -n torch_env python imagenet_experiment.py --variant QKV      --device cuda:0
    conda run -n torch_env python imagenet_experiment.py --variant "Q!=K=V" --device cuda:1
    conda run -n torch_env python imagenet_experiment.py --variant "(Q=K=V)+" --device cuda:2
    conda run -n torch_env python imagenet_experiment.py --variant "QVV(3)" --device cuda:3

    # resume an interrupted run
    conda run -n torch_env python imagenet_experiment.py --variant QKV --device cuda:0 --resume

    # benchmark inference throughput / peak memory only (no training)
    conda run -n torch_env python imagenet_experiment.py --variant "QVV(3)" --bench-only --device cuda:0

    # 2-GPU DDP: split a batch-256 run as 128+128 across two physical GPUs, e.g. 0 and 3,
    # when batch-256 doesn't fit in a single 2080 Ti's 11 GB. --batch-size is PER GPU;
    # global/effective batch and LR scaling match a single-GPU --batch-size 256 run.
    CUDA_VISIBLE_DEVICES=0,3 conda run -n torch_env torchrun --standalone --nproc_per_node=2 \\
        imagenet_experiment.py --variant "(Q=K!=V)+" --batch-size 128 --out-dir checkpoints_imagenet_bs256
"""

import os
import csv
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from vision_tasks import ViT, ViTConfig

# ============================================================================
# CONSTANTS
# ============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

CONDITIONS = ["QKV", "Q!=K=V", "(Q=K=V)+", "QVV(3)", "(Q=K!=V)+", "Q=K=V", "Q=K!=V"]
DISPLAY_NAME = {"QKV": "QKV", "Q!=K=V": "Q-K=V", "(Q=K=V)+": "Q=K=V+", "QVV(3)": "QVV(3)",
                "(Q=K!=V)+": "Q=K-V+", "Q=K=V": "Q=K=V (no +)", "Q=K!=V": "Q=K-V (no +)"}

# ViT-S/16
VIT_S16 = dict(patch=16, n_embd=384, n_layer=12, n_head=6)


_TAG_OVERRIDE = {
    # Stripping non-alnum chars would otherwise collapse each of these onto the
    # same tag as its '+' (positional-correction) counterpart, clobbering that
    # checkpoint/CSV: 'Q=K=V' -> '(Q=K=V)+', 'Q=K!=V' -> '(Q=K!=V)+'.
    "Q=K=V": "Q_K_V_noplus",
    "Q=K!=V": "Q_K__V_noplus",
}


def tag(variant):
    """Filesystem-safe tag for a variant name, e.g. 'Q!=K=V' -> 'Q_K_V'."""
    if variant in _TAG_OVERRIDE:
        return _TAG_OVERRIDE[variant]
    return "".join(c if c.isalnum() else "_" for c in variant).strip("_")


# ============================================================================
# DATA
# ============================================================================

def build_transforms(train):
    if train:
        return T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0),
                                interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9,
                          interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25),
        ])
    return T.Compose([
        T.Resize(256, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_imagenet(data_root, train):
    split_dir = "train" if train else "validation"
    return torchvision.datasets.ImageFolder(os.path.join(data_root, split_dir),
                                            transform=build_transforms(train))


# ============================================================================
# TRAIN / EVAL
# ============================================================================

def lr_lambda_factory(total_steps, warmup_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    correct1 = correct5 = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * y.size(0)
        top5 = logits.topk(5, dim=-1).indices
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(dim=-1).sum().item()
        total += y.size(0)
    return loss_sum / total, correct1 / total, correct5 / total


def build_model(variant, device):
    cfg = ViTConfig(image_size=IMAGE_SIZE, channels=3, patch=VIT_S16["patch"],
                    n_classes=1000, n_embd=VIT_S16["n_embd"], n_layer=VIT_S16["n_layer"],
                    n_head=VIT_S16["n_head"], variant=variant, dropout=0.0, drop_path=0.1)
    return ViT(cfg).to(device)


@torch.no_grad()
def benchmark_inference(model, device, batch_size=256, iters=20, warmup=5):
    model.eval()
    x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for _ in range(iters):
        model(x)
    torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    imgs_per_sec = (batch_size * iters) / elapsed
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
    return imgs_per_sec, peak_mem_mb


def ddp_is_active():
    """True when launched via torchrun with more than one process."""
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def save_checkpoint(path, model, opt, scaler, epoch, best_top1, best_epoch):
    torch.save({
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_top1": best_top1,
        "best_epoch": best_epoch,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, required=True, choices=CONDITIONS)
    ap.add_argument("--data-root", type=str, default="/data/scratch/sbruers/imagenet")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--base-lr", type=float, default=5e-4, help="LR at batch size 512 (DeiT default); scaled linearly by batch_size/512")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out-dir", type=str, default="checkpoints_imagenet")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny subset, 1 epoch, verifies the pipeline and reports img/sec")
    ap.add_argument("--smoke-subset", type=int, default=2000,
                    help="train-subset size for --smoke-test (larger -> better throughput estimate)")
    ap.add_argument("--bench-only", action="store_true",
                    help="skip training; just report inference throughput/peak memory")
    args = ap.parse_args()

    ddp = ddp_is_active()
    if ddp:
        local_rank, rank, world_size = ddp_setup()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device(args.device)
    is_main = rank == 0

    os.makedirs(args.out_dir, exist_ok=True)
    variant_tag = tag(args.variant)
    ckpt_path = os.path.join(args.out_dir, f"vit_s16_imagenet_{variant_tag}.pt")
    csv_path = os.path.join(args.out_dir, f"vit_s16_imagenet_{variant_tag}.csv")

    if is_main:
        eff_batch = args.batch_size * world_size
        print(f"variant={args.variant} ({DISPLAY_NAME[args.variant]})  device={device}  "
              f"world_size={world_size}  per_gpu_batch={args.batch_size}  "
              f"effective_batch={eff_batch}  out={ckpt_path}", flush=True)

    model = build_model(args.variant, device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"params={n_params/1e6:.2f}M", flush=True)

    if args.bench_only:
        imgs_per_sec, peak_mem_mb = benchmark_inference(model, device)
        print(f"inference throughput: {imgs_per_sec:.1f} img/s, "
              f"peak memory: {peak_mem_mb:.1f} MB (batch=256, fp32)")
        if args.variant == "QVV(3)":
            fused_model = build_model(args.variant, device)
            fused_model.load_state_dict(model.state_dict())
            for blk in fused_model.blocks:
                blk.attn = blk.attn.fuse_qvv3().to(device)
            imgs_per_sec_f, peak_mem_f = benchmark_inference(fused_model, device)
            print(f"QVV(3) fused (Q1,Q2 -> single Q, matches Q-K=V cost): "
                  f"{imgs_per_sec_f:.1f} img/s, {peak_mem_f:.1f} MB")
        return

    if is_main:
        print("loading ImageNet-1k...", flush=True)
    train_ds = get_imagenet(args.data_root, train=True)
    val_ds = get_imagenet(args.data_root, train=False)
    if is_main:
        print(f"train={len(train_ds)} val={len(val_ds)} classes={len(train_ds.classes)}", flush=True)

    if args.smoke_test:
        train_ds = torch.utils.data.Subset(train_ds, range(min(args.smoke_subset, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(args.smoke_subset // 2, len(val_ds))))
        args.epochs = 1

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.workers > 0)
    # Validation only ever runs on rank 0 (see the eval call below), against the full val_ds.
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)

    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                     mode="batch", label_smoothing=0.1, num_classes=1000)
    train_criterion = SoftTargetCrossEntropy()

    raw_model = model  # unwrapped module; checkpoints are always saved/loaded from this
    if ddp:
        model = DDP(raw_model, device_ids=[local_rank])

    # args.batch_size is PER GPU; scale LR by the effective global batch (batch_size * world_size)
    # so a 2-GPU DDP run at --batch-size 128 matches a single-GPU --batch-size 256 run exactly.
    lr = args.base_lr * (args.batch_size * world_size) / 512
    opt = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total_steps, warmup_steps))
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    best_top1, best_epoch = 0.0, -1
    if args.resume and os.path.exists(ckpt_path):
        d = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(d["model_state_dict"])
        opt.load_state_dict(d["opt_state_dict"])
        scaler.load_state_dict(d["scaler_state_dict"])
        start_epoch = d["epoch"] + 1
        best_top1, best_epoch = d["best_top1"], d["best_epoch"]
        for _ in range(start_epoch * steps_per_epoch):
            sched.step()
        if is_main:
            print(f"resumed from epoch {d['epoch']} (best_top1={best_top1:.4f} @ {best_epoch})",
                  flush=True)

    if is_main:
        write_header = not (os.path.exists(csv_path) and args.resume)
        csv_file = open(csv_path, "a" if args.resume else "w", newline="")
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_top1", "val_top5",
                             "epoch_time_sec", "lr"])
            csv_file.flush()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss_sum, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y_soft = mixup_fn(x, y)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = train_criterion(logits, y_soft)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            train_loss_sum += loss.item() * y.size(0)
            n_seen += y.size(0)

        train_loss = train_loss_sum / n_seen
        epoch_time = time.time() - t0

        if is_main:
            val_loss, val_top1, val_top5 = evaluate(raw_model, val_loader, device, amp=True)
            if val_top1 > best_top1:
                best_top1, best_epoch = val_top1, epoch
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_top1:.4f}",
                             f"{val_top5:.4f}", f"{epoch_time:.1f}", f"{sched.get_last_lr()[0]:.6f}"])
            csv_file.flush()
            save_checkpoint(ckpt_path, raw_model, opt, scaler, epoch, best_top1, best_epoch)
            print(f"[{args.variant}] epoch {epoch+1}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"top1={val_top1:.4f}  top5={val_top5:.4f}  "
                  f"best_top1={best_top1:.4f}@{best_epoch}  ({epoch_time:.0f}s)", flush=True)
        if ddp:
            dist.barrier()

    if is_main:
        csv_file.close()
        print(f"done. best_top1={best_top1:.4f} @ epoch {best_epoch}, "
              f"final_val_top1={val_top1:.4f}", flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
