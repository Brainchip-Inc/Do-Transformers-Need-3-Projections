"""
K=V tying study on the vision-classification tasks: post-hoc weight surgery + distillation
recovery. Vision counterpart of distillation_synthetic.py -- same two questions, same
surgery/KD mechanics, applied to the ViT classifiers instead of the sequence Encoder.

  (a) Post-hoc surgery: take a *trained* QKV ViT checkpoint and force K=V after the fact
      (no retraining) by dropping one projection and reusing the other (keep_k, keep_v) or
      averaging the two Linear layers' weights (avg). Compare zero-shot top-1 accuracy
      against the QKV teacher and the from-scratch Q!=K=V checkpoint
      (checkpoints/vision_{dataset}_{qkv,qkv_kv}.pt, see export_checkpoints.py).

  (b) Distillation recovery: fine-tune each surgically-tied student for a few epochs
      against the QKV teacher's soft logits (+ hard-label CE) and see how much of the gap
      opened by (a) is recovered relative to the from-scratch Q!=K=V ceiling.

Run in an sm_61-compatible env WITH torchvision (vision_tasks.py needs it), e.g. torch_env:
    conda run -n torch_env python distillation_vision.py --quick --device cuda:0
    conda run -n torch_env python distillation_vision.py --device cuda:0
"""

import os
import csv
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_tasks import lr_lambda_factory
import vision_tasks as V
from vision_tasks import ViT, ViTConfig, CLASSIFICATION, evaluate, _loader

CKPT_DIR = "checkpoints"
TEACHER_TAG = "qkv"
SCRATCH_KV_TAG = "qkv_kv"
MODES = ["keep_k", "keep_v", "avg"]
DATASETS = list(CLASSIFICATION)


def load_checkpoint(dataset, tag, device):
    path = os.path.join(CKPT_DIR, f"vision_{dataset}_{tag}.pt")
    d = torch.load(path, map_location=device, weights_only=False)
    cfg = ViTConfig(**d["config"])
    model = ViT(cfg).to(device)
    model.load_state_dict(d["model_state_dict"])
    return model, cfg, d["test_accuracy"]


def surgically_tie_kv(teacher, mode, device):
    """Build a Q!=K=V ViT student from a trained QKV teacher, with no retraining
    (see distillation_synthetic.py's surgically_tie_kv for the full rationale)."""
    student_cfg = ViTConfig(**{**vars(teacher.cfg), "variant": "Q!=K=V"})
    student = ViT(student_cfg).to(device)
    # loads patch_embed/cls/pos_emb/ln_f/head/c_q/c_proj/mlp/layernorms verbatim;
    # c_kv has no match in the teacher's state dict (which has c_k/c_v) and stays random.
    student.load_state_dict(teacher.state_dict(), strict=False)

    for t_blk, s_blk in zip(teacher.blocks, student.blocks):
        t_attn, s_attn = t_blk.attn, s_blk.attn
        if mode == "keep_k":
            s_attn.c_kv.weight.data.copy_(t_attn.c_k.weight.data)
            s_attn.c_kv.bias.data.copy_(t_attn.c_k.bias.data)
        elif mode == "keep_v":
            s_attn.c_kv.weight.data.copy_(t_attn.c_v.weight.data)
            s_attn.c_kv.bias.data.copy_(t_attn.c_v.bias.data)
        elif mode == "avg":
            s_attn.c_kv.weight.data.copy_((t_attn.c_k.weight.data + t_attn.c_v.weight.data) / 2)
            s_attn.c_kv.bias.data.copy_((t_attn.c_k.bias.data + t_attn.c_v.bias.data) / 2)
        else:
            raise ValueError(mode)
    return student


def distill(student, teacher, train_loader, test_loader, device, epochs,
            lr, temperature, alpha, log_prefix=""):
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(steps_per_epoch * epochs, warmup_frac=0.1))

    accs = []
    for ep in range(epochs):
        student.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                t_logits = teacher(xb)
            s_logits = student(xb)
            ce = F.cross_entropy(s_logits, yb)
            kd = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=-1),
                F.softmax(t_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            loss = alpha * ce + (1 - alpha) * kd
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
            sched.step()
        acc = evaluate(student, test_loader, device)
        accs.append(acc)
        print(f"{log_prefix} epoch {ep + 1}/{epochs} -> test acc {acc:.4f}", flush=True)
    return accs


def run_dataset_mode(dataset, mode, device, args):
    teacher, cfg, teacher_acc = load_checkpoint(dataset, TEACHER_TAG, device)
    _, _, scratch_kv_acc = load_checkpoint(dataset, SCRATCH_KV_TAG, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    train_ds = V.get_classification_dataset(dataset, args.data_root, True)
    test_ds = V.get_classification_dataset(dataset, args.data_root, False)
    if args.quick:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(512, len(train_ds))))
        test_ds = Subset(test_ds, range(min(512, len(test_ds))))
    train_loader = _loader(train_ds, args.batch_size, args.workers, True)
    test_loader = _loader(test_ds, args.batch_size, args.workers, False)

    student = surgically_tie_kv(teacher, mode, device)
    zero_shot_acc = evaluate(student, test_loader, device)

    accs = distill(student, teacher, train_loader, test_loader, device,
                    epochs=args.epochs, lr=args.lr, temperature=args.temperature,
                    alpha=args.alpha, log_prefix=f"[{dataset} {mode}]")

    row = {
        "dataset": dataset, "mode": mode,
        "teacher_qkv_acc": teacher_acc,
        "scratch_qkv_kv_acc": scratch_kv_acc,
        "zero_shot_surgery_acc": round(zero_shot_acc, 4),
    }
    for i, a in enumerate(accs):
        row[f"distill_epoch{i + 1}_acc"] = round(a, 4)
    print(f"[{dataset} {mode}] teacher={teacher_acc:.4f} scratch_kv={scratch_kv_acc:.4f} "
          f"zero_shot={zero_shot_acc:.4f} final_distill={accs[-1]:.4f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None, choices=DATASETS)
    ap.add_argument("--mode", type=str, default=None, choices=MODES)
    ap.add_argument("--epochs", type=int, default=5, help="distillation fine-tune epochs")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on hard-label CE vs KD")
    ap.add_argument("--data-root", type=str, default="./vision_data")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="1 dataset x 1 mode, 1 epoch, subset")
    ap.add_argument("--out-csv", type=str, default="distillation_vision_results.csv")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    datasets = [args.dataset] if args.dataset else DATASETS
    modes = [args.mode] if args.mode else MODES
    if args.quick:
        datasets, modes, args.epochs = datasets[:1], modes[:1], 1

    print(f"device={device}  datasets={datasets}  modes={modes}  epochs={args.epochs}", flush=True)

    rows = [run_dataset_mode(dataset, mode, device, args)
            for dataset in datasets for mode in modes]

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
