"""
Reviewer-requested extensions to the post-hoc K=V surgery study (zero-shot only, no
distillation -- this answers "where does the merge want to sit between K and V, and is
merging-after-training specifically about K/V incompatibility or just any post-hoc
architecture edit" without needing new training runs):

  (a) alpha-interpolation sweep: c_kv(alpha) = alpha*W_K + (1-alpha)*W_V (and the
      corresponding bias blend), swept over alpha in {0, 0.1, ..., 1.0}. alpha=0 is
      keep_v, alpha=1 is keep_k, alpha=0.5 is the paper's avg -- this fills in the
      9 points between those three already-reported ones and shows whether the
      zero-shot optimum sits nearer K, nearer V, or at the midpoint.

  (b) random-projection baseline: c_kv initialized fresh (default nn.Linear init,
      independent of the teacher's trained K/V) instead of built from them. If
      zero-shot accuracy for keep_k/keep_v/avg is close to this random baseline, the
      damage would just be "any post-hoc architecture edit is catastrophic," not
      specifically a K/V incompatibility; if keep_k/keep_v/avg clear it by a wide
      margin (as the already-reported avg numbers suggest), that supports the paper's
      account that *which* projection you keep matters, not just that you changed one.

Both reuse the exact surgically_tie_kv / checkpoint / test-set construction from
distillation_synthetic.py and distillation_vision.py -- only which c_kv gets built
differs, and nothing here is trained.

Usage:
    conda run -n torch_env python kv_alpha_sweep.py
"""

import csv
import statistics
import torch

from synthetic_tasks import Encoder, ModelCfg, make_dataset, TASKS
import distillation_synthetic as DS

import vision_tasks as V
from vision_tasks import ViT, ViTConfig, CLASSIFICATION, evaluate as vision_evaluate, _loader
import distillation_vision as DV

DEVICE = torch.device("cpu")
ALPHAS = [round(0.1 * i, 1) for i in range(11)]  # 0.0 (keep_v) ... 1.0 (keep_k)
N_RANDOM_SEEDS = 5  # reviewer request: report random-baseline variability, not just its mean


def build_alpha_student(teacher, alpha, cfg_cls, model_cls):
    """alpha=1 -> keep_k, alpha=0 -> keep_v, alpha=0.5 -> avg. None of these three special
    cases are handled differently; this is just the general blend they're all a point of."""
    student_cfg = cfg_cls(**{**vars(teacher.cfg), "variant": "Q!=K=V"})
    student = model_cls(student_cfg).to(DEVICE)
    student.load_state_dict(teacher.state_dict(), strict=False)
    for t_blk, s_blk in zip(teacher.blocks, student.blocks):
        t_attn, s_attn = t_blk.attn, s_blk.attn
        s_attn.c_kv.weight.data.copy_(
            alpha * t_attn.c_k.weight.data + (1 - alpha) * t_attn.c_v.weight.data)
        s_attn.c_kv.bias.data.copy_(
            alpha * t_attn.c_k.bias.data + (1 - alpha) * t_attn.c_v.bias.data)
    return student


def build_random_student(teacher, cfg_cls, model_cls, seed=0):
    """c_kv left at its fresh nn.Linear init (independent of the trained teacher)."""
    student_cfg = cfg_cls(**{**vars(teacher.cfg), "variant": "Q!=K=V"})
    torch.manual_seed(seed)
    student = model_cls(student_cfg).to(DEVICE)
    student.load_state_dict(teacher.state_dict(), strict=False)  # c_kv stays fresh-init
    return student


def synthetic_sweep():
    rows = []
    for task in TASKS:
        teacher, cfg, teacher_acc = DS.load_checkpoint(task, DS.TEACHER_TAG, DEVICE)
        _, _, scratch_acc = DS.load_checkpoint(task, DS.SCRATCH_KV_TAG, DEVICE)
        teacher.eval()

        g = torch.Generator().manual_seed(0)
        x_tr, y_tr = make_dataset(DS.SYN_CFG["n_train"], cfg.seq_len, task, g)
        x_te, y_te = make_dataset(DS.SYN_CFG["n_test"], cfg.seq_len, task, g)

        for alpha in ALPHAS:
            student = build_alpha_student(teacher, alpha, ModelCfg, Encoder)
            acc = DS.evaluate(student, x_te, y_te, DS.SYN_CFG["batch_size"])
            rows.append({"domain": "synthetic", "name": task, "alpha": alpha,
                         "mode": "alpha", "acc": round(acc, 4),
                         "teacher_acc": teacher_acc, "scratch_acc": scratch_acc})
            print(rows[-1], flush=True)

        rand_accs = [DS.evaluate(build_random_student(teacher, ModelCfg, Encoder, seed=s),
                                  x_te, y_te, DS.SYN_CFG["batch_size"])
                     for s in range(N_RANDOM_SEEDS)]
        rows.append({"domain": "synthetic", "name": task, "alpha": "random",
                     "mode": "random", "acc": round(statistics.mean(rand_accs), 4),
                     "acc_std": round(statistics.stdev(rand_accs), 4),
                     "acc_all_seeds": ";".join(f"{a:.4f}" for a in rand_accs),
                     "teacher_acc": teacher_acc, "scratch_acc": scratch_acc})
        print(rows[-1], flush=True)
    return rows


def vision_sweep():
    rows = []
    for ds in CLASSIFICATION:
        teacher, cfg, teacher_acc = DV.load_checkpoint(ds, DV.TEACHER_TAG, DEVICE)
        _, _, scratch_acc = DV.load_checkpoint(ds, DV.SCRATCH_KV_TAG, DEVICE)
        teacher.eval()

        test_ds = V.get_classification_dataset(ds, "./vision_data", False)
        test_loader = _loader(test_ds, 256, 2, False)

        for alpha in ALPHAS:
            student = build_alpha_student(teacher, alpha, ViTConfig, ViT)
            acc = vision_evaluate(student, test_loader, DEVICE)
            rows.append({"domain": "vision", "name": ds, "alpha": alpha,
                         "mode": "alpha", "acc": round(acc, 4),
                         "teacher_acc": teacher_acc, "scratch_acc": scratch_acc})
            print(rows[-1], flush=True)

        rand_accs = [vision_evaluate(build_random_student(teacher, ViTConfig, ViT, seed=s),
                                      test_loader, DEVICE) for s in range(N_RANDOM_SEEDS)]
        rows.append({"domain": "vision", "name": ds, "alpha": "random",
                     "mode": "random", "acc": round(statistics.mean(rand_accs), 4),
                     "acc_std": round(statistics.stdev(rand_accs), 4),
                     "acc_all_seeds": ";".join(f"{a:.4f}" for a in rand_accs),
                     "teacher_acc": teacher_acc, "scratch_acc": scratch_acc})
        print(rows[-1], flush=True)
    return rows


def main():
    rows = synthetic_sweep() + vision_sweep()
    with open("kv_alpha_sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "name", "alpha", "mode", "acc",
                                          "acc_std", "acc_all_seeds",
                                          "teacher_acc", "scratch_acc"],
                            restval="")
        w.writeheader()
        w.writerows(rows)
    print("\nWrote kv_alpha_sweep_results.csv")


if __name__ == "__main__":
    main()
