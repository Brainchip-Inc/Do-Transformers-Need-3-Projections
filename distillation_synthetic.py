"""
K=V tying study on the synthetic tasks: post-hoc weight surgery + distillation recovery.

Two questions, both scoped to the 5 synthetic tasks (REVERSE/SORT/SUB/SWAP/COPY) where we
already have trained QKV and Q!=K=V (K=V, trained-from-scratch) checkpoints
(see export_checkpoints.py, checkpoints/synthetic_{task}_{qkv,qkv_kv}.pt):

  (a) Post-hoc surgery: take a *trained QKV* checkpoint and force K=V after the fact, with
      no retraining, by either dropping one projection and reusing the other (keep-K,
      keep-V) or averaging the two projections' weights (avg). Since c_k/c_v are both
      plain Linear layers, averaging weights == averaging outputs, so a single "avg" mode
      covers both readings of "average the two". Compare the resulting zero-shot accuracy
      against the QKV teacher and the from-scratch Q!=K=V checkpoint.

  (b) Distillation recovery: fine-tune each surgered student for a few epochs against the
      QKV teacher's soft targets (+ hard-label CE) and see how much of the gap (a) opens up
      is recovered, and how that compares to training Q!=K=V from scratch.

Usage:
    python distillation_synthetic.py --quick                  # fast smoke test, 1 task
    python distillation_synthetic.py --device cuda:0           # full run, all tasks x modes
    python distillation_synthetic.py --task REVERSE --mode avg # restrict
"""

import os
import csv
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_tasks import Encoder, ModelCfg, make_dataset, TASKS, lr_lambda_factory

CKPT_DIR = "checkpoints"
TEACHER_TAG = "qkv"        # trained-from-scratch QKV checkpoint (surgery source / KD teacher)
SCRATCH_KV_TAG = "qkv_kv"  # trained-from-scratch Q!=K=V checkpoint (reference ceiling)

MODES = ["keep_k", "keep_v", "avg"]

SYN_CFG = dict(n_embd=256, n_layer=2, n_head=4, seq_len=64,
               n_train=10000, n_test=2000, batch_size=128)


def load_checkpoint(task, tag, device):
    path = os.path.join(CKPT_DIR, f"synthetic_{task.lower()}_{tag}.pt")
    d = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelCfg(**d["config"])
    model = Encoder(cfg).to(device)
    model.load_state_dict(d["model_state_dict"])
    return model, cfg, d["test_accuracy"]


def surgically_tie_kv(teacher, mode, device):
    """Build a Q!=K=V ("kv" share) student from a trained QKV teacher, with no retraining.

    c_q and everything outside attention (embed, pos_emb, mlp, layernorms, c_proj, head)
    are copied verbatim. c_kv is built from the teacher's independently-trained c_k / c_v:
        keep_k -> c_kv := c_k   (drop V's projection, reuse K's)
        keep_v -> c_kv := c_v   (drop K's projection, reuse V's)
        avg    -> c_kv := mean(c_k, c_v)  (weights and bias averaged elementwise)
    """
    student_cfg = ModelCfg(**{**vars(teacher.cfg), "variant": "Q!=K=V"})
    student = Encoder(student_cfg).to(device)
    # loads embed/pos_emb/ln_f/head/c_q/c_proj/mlp/layernorms verbatim; c_kv has no match
    # in the teacher's state dict (which has c_k/c_v instead) and stays randomly init'd.
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


@torch.no_grad()
def evaluate(model, x_te, y_te, batch_size=128):
    model.eval()
    correct, total = 0, 0
    for i in range(0, x_te.size(0), batch_size):
        xb, yb = x_te[i:i + batch_size], y_te[i:i + batch_size]
        logits, _ = model(xb)
        pred = logits.argmax(-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total


def distill(student, teacher, x_tr, y_tr, x_te, y_te, device, epochs, batch_size,
            lr, temperature, alpha, seed, log_prefix=""):
    """Fine-tune `student` against `teacher`'s soft targets + hard labels.

    loss = alpha * CE(student, hard_labels)
         + (1 - alpha) * T^2 * KLDiv(log_softmax(student/T), softmax(teacher/T))
    Returns list of per-epoch test accuracies.
    """
    g = torch.Generator().manual_seed(seed)
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    n_train = x_tr.size(0)
    steps_per_epoch = math.ceil(n_train / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(steps_per_epoch * epochs, warmup_frac=0.1))

    accs = []
    for ep in range(epochs):
        student.train()
        perm = torch.randperm(n_train, generator=g)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_tr[idx].to(device), y_tr[idx].to(device)
            with torch.no_grad():
                t_logits, _ = teacher(xb)
            s_logits, _ = student(xb)
            ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), yb.reshape(-1))
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
        acc = evaluate(student, x_te, y_te, batch_size)
        accs.append(acc)
        print(f"{log_prefix} epoch {ep + 1}/{epochs} -> test acc {acc:.4f}", flush=True)
    return accs


def run_task_mode(task, mode, device, args):
    teacher, cfg, teacher_acc = load_checkpoint(task, TEACHER_TAG, device)
    _, _, scratch_kv_acc = load_checkpoint(task, SCRATCH_KV_TAG, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    g = torch.Generator().manual_seed(0)  # same seed as export_checkpoints.py's data gen
    x_tr, y_tr = make_dataset(SYN_CFG["n_train"], cfg.seq_len, task, g)
    x_te, y_te = make_dataset(SYN_CFG["n_test"], cfg.seq_len, task, g)
    x_te, y_te = x_te.to(device), y_te.to(device)

    student = surgically_tie_kv(teacher, mode, device)
    zero_shot_acc = evaluate(student, x_te, y_te, SYN_CFG["batch_size"])

    accs = distill(student, teacher, x_tr, y_tr, x_te, y_te, device,
                    epochs=args.epochs, batch_size=SYN_CFG["batch_size"],
                    lr=args.lr, temperature=args.temperature, alpha=args.alpha,
                    seed=1, log_prefix=f"[{task} {mode}]")

    row = {
        "task": task, "mode": mode,
        "teacher_qkv_acc": teacher_acc,
        "scratch_qkv_kv_acc": scratch_kv_acc,
        "zero_shot_surgery_acc": round(zero_shot_acc, 4),
    }
    for i, a in enumerate(accs):
        row[f"distill_epoch{i + 1}_acc"] = round(a, 4)
    print(f"[{task} {mode}] teacher={teacher_acc:.4f} scratch_kv={scratch_kv_acc:.4f} "
          f"zero_shot={zero_shot_acc:.4f} final_distill={accs[-1]:.4f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--task", type=str, default=None, choices=TASKS)
    ap.add_argument("--mode", type=str, default=None, choices=MODES)
    ap.add_argument("--epochs", type=int, default=5, help="distillation fine-tune epochs")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on hard-label CE vs KD")
    ap.add_argument("--quick", action="store_true", help="1 task x 1 mode, 1 epoch")
    ap.add_argument("--out-csv", type=str, default="distillation_synthetic_results.csv")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    tasks = [args.task] if args.task else TASKS
    modes = [args.mode] if args.mode else MODES
    if args.quick:
        tasks, modes, args.epochs = tasks[:1], modes[:1], 1

    print(f"device={device}  tasks={tasks}  modes={modes}  epochs={args.epochs}", flush=True)

    rows = []
    for task in tasks:
        for mode in modes:
            rows.append(run_task_mode(task, mode, device, args))

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
