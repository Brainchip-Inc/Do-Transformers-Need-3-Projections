"""
Addressing/payload decomposition (reviewer-requested "decisive experiment" #1):
separately vary which projection produces attention *scores* (addressing) vs.
which produces the *transported* vectors (payload) on the synthetic/vision QKV
teachers, with no retraining.

There are 4 cells, A(q-source, payload-source):
    A(Q,K)V  -- the QKV teacher itself (standard attention)
    A(Q,K)K  -- exactly keep_k surgery: tying c_kv := c_k reuses c_k for BOTH
                roles, so scores are unchanged (A(Q,K)) and the payload becomes
                K instead of V
    A(Q,V)V  -- exactly keep_v surgery, by the same argument with c_v
    A(Q,V)K  -- the only genuinely new cell: scores from V, payload from K

So this script only needs to compute A(Q,V)K; the other three are read
straight out of the existing result CSVs (distillation_synthetic_results.csv /
distillation_vision_results.csv), not recomputed, since they're already the
exact same teacher checkpoints and held-out data.

Usage:
    conda run -n torch_env python kv_addressing_payload_swap.py
"""

import csv
import types

import torch
import torch.nn.functional as F

from synthetic_tasks import Encoder, ModelCfg, make_dataset, TASKS
from distillation_synthetic import CKPT_DIR as SYN_CKPT_DIR, SYN_CFG, evaluate as syn_evaluate
import vision_tasks as V
from vision_tasks import ViT, ViTConfig, CLASSIFICATION, evaluate as vis_evaluate, _loader

DEVICE = torch.device("cpu")  # tiny models; keep off the GPU the live training run needs


def swapped_forward(self, x):
    """A(Q,V)K: SharedProjAttention.forward's exact math (share='none' only,
    the only case with independent c_q/c_k/c_v), but scores come from c_v(x)
    and the transported payload comes from c_k(x) -- roles swapped relative to
    the teacher's own forward."""
    assert self.share == "none", "swap only defined for independent c_q/c_k/c_v"
    B, T, C = x.size()
    q = self.c_q(x)
    k_role = self.c_v(x)  # addressing now comes from V
    v_role = self.c_k(x)  # payload now comes from K

    def split(t):
        return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    q, k_role, v_role = split(q), split(k_role), split(v_role)

    import math
    scale = 1.0 / math.sqrt(self.head_dim)
    scores = torch.matmul(q, k_role.transpose(-2, -1)) * scale
    if self.plus:
        P = self.pos2d[:T, :T, :]
        pos_term = torch.einsum("ijm,m->ij", P, self.pos_w)
        scores = scores * self.pos_w.sum() + pos_term + self.pos_b
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_role)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.c_proj(out)


def apply_swap(model):
    for blk in model.blocks:
        blk.attn.forward = types.MethodType(swapped_forward, blk.attn)


def load_existing_row(csv_path, key_col, key_val, mode):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row[key_col] == key_val and row["mode"] in (mode, "any"):
                return row
    return None


def run_synthetic():
    rows = []
    for task in TASKS:
        path = f"{SYN_CKPT_DIR}/synthetic_{task.lower()}_qkv.pt"
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ModelCfg(**d["config"])
        teacher = Encoder(cfg).to(DEVICE).eval()
        teacher.load_state_dict(d["model_state_dict"])

        g = torch.Generator().manual_seed(0)  # same seed as distillation_synthetic.py
        _, _ = make_dataset(SYN_CFG["n_train"], cfg.seq_len, task, g)
        x_te, y_te = make_dataset(SYN_CFG["n_test"], cfg.seq_len, task, g)

        # sanity check: unmodified forward must reproduce the known teacher accuracy
        baseline_acc = syn_evaluate(teacher, x_te, y_te, SYN_CFG["batch_size"])
        assert abs(baseline_acc - d["test_accuracy"]) < 2e-3, \
            f"{task}: baseline A(Q,K)V mismatch, got {baseline_acc} vs {d['test_accuracy']}"

        apply_swap(teacher)
        swap_acc = syn_evaluate(teacher, x_te, y_te, SYN_CFG["batch_size"])

        kk = load_existing_row("distillation_synthetic_results.csv", "task", task, "keep_k")
        kv = load_existing_row("distillation_synthetic_results.csv", "task", task, "keep_v")
        row = {
            "task": task,
            "A(Q,K)V_teacher": d["test_accuracy"],
            "A(Q,K)K_keep_k": float(kk["zero_shot_surgery_acc"]) if kk else None,
            "A(Q,V)V_keep_v": float(kv["zero_shot_surgery_acc"]) if kv else None,
            "A(Q,V)K_new": round(swap_acc, 4),
        }
        rows.append(row)
        print(row, flush=True)
    return rows


def run_vision():
    rows = []
    for ds in CLASSIFICATION:
        path = f"checkpoints/vision_{ds}_qkv.pt"
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ViTConfig(**d["config"])
        teacher = ViT(cfg).to(DEVICE).eval()
        teacher.load_state_dict(d["model_state_dict"])

        test_ds = V.get_classification_dataset(ds, "./vision_data", False)
        loader = _loader(test_ds, 256, 2, False)

        baseline_acc = vis_evaluate(teacher, loader, DEVICE)
        assert abs(baseline_acc - d["test_accuracy"]) < 2e-3, \
            f"{ds}: baseline A(Q,K)V mismatch, got {baseline_acc} vs {d['test_accuracy']}"

        apply_swap(teacher)
        swap_acc = vis_evaluate(teacher, loader, DEVICE)

        kk = load_existing_row("distillation_vision_results.csv", "dataset", ds, "keep_k")
        kv = load_existing_row("distillation_vision_results.csv", "dataset", ds, "keep_v")
        row = {
            "dataset": ds,
            "A(Q,K)V_teacher": d["test_accuracy"],
            "A(Q,K)K_keep_k": float(kk["zero_shot_surgery_acc"]) if kk else None,
            "A(Q,V)V_keep_v": float(kv["zero_shot_surgery_acc"]) if kv else None,
            "A(Q,V)K_new": round(swap_acc, 4),
        }
        rows.append(row)
        print(row, flush=True)
    return rows


def main():
    print("=== Synthetic tasks ===", flush=True)
    syn_rows = run_synthetic()
    print("\n=== Vision datasets ===", flush=True)
    vis_rows = run_vision()

    fieldnames = ["domain", "task_or_dataset", "A(Q,K)V_teacher", "A(Q,K)K_keep_k",
                  "A(Q,V)V_keep_v", "A(Q,V)K_new"]
    with open("kv_addressing_payload_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in syn_rows:
            w.writerow({"domain": "synthetic", "task_or_dataset": r["task"], **{k: r[k] for k in fieldnames[2:]}})
        for r in vis_rows:
            w.writerow({"domain": "vision", "task_or_dataset": r["dataset"], **{k: r[k] for k in fieldnames[2:]}})
    print("\nWrote kv_addressing_payload_results.csv")


if __name__ == "__main__":
    main()
