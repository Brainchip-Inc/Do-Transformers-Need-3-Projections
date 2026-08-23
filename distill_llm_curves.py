"""
Training-loss and validation-perplexity curves for the LLM-scale K=V distillation
runs (distillation_llm.py), parsed directly from distill_llm_{mode}.log. Appendix
counterpart to distillation_curves_full.pdf's per-epoch synthetic/vision curves --
same idea (full recovery trajectory, not just table endpoints), adapted to this
run's step-based logging and much larger zero-shot-to-recovered dynamic range
(hence the log-scale PPL axis).

Usage:
    python3 distill_llm_curves.py
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["keep_k", "keep_v", "avg"]
COLORS = {"keep_k": "#c0623b", "keep_v": "#3b6fa0", "avg": "#4a9b57"}
ZERO_SHOT_PPL = {"keep_k": 13345.19, "keep_v": 7347.14, "avg": 6430.36}
TEACHER_PPL = 20.8183
SCRATCH_CEILING_PPL = 21.2798
TOTAL_STEPS = 1695

STEP_RE = re.compile(r"^\[(\w+)\] step (\d+)/\d+ loss=([\d.]+)")
EVAL_RE = re.compile(r"^\[(\w+)\] \[EVAL\] step (\d+)/\d+ val_ppl=([\d.]+)")


def parse_log(mode):
    steps, losses, eval_steps, eval_ppls = [], [], [], []
    with open(f"distill_llm_{mode}.log") as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                steps.append(int(m.group(2)))
                losses.append(float(m.group(3)))
                continue
            m = EVAL_RE.match(line)
            if m:
                eval_steps.append(int(m.group(2)))
                eval_ppls.append(float(m.group(3)))
    return steps, losses, eval_steps, eval_ppls


fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(9.5, 3.6))

for mode in MODES:
    steps, losses, eval_steps, eval_ppls = parse_log(mode)
    pct = [100 * s / TOTAL_STEPS for s in steps]
    ax_loss.plot(pct, losses, color=COLORS[mode], linewidth=1.2, label=mode)

    eval_pct = [0.0] + [100 * s / TOTAL_STEPS for s in eval_steps]
    eval_y = [ZERO_SHOT_PPL[mode]] + eval_ppls
    ax_ppl.plot(eval_pct, eval_y, color=COLORS[mode], marker="o", markersize=4,
                linewidth=1.5, label=mode)

ax_loss.set_xlabel("% of 500M-token distillation budget")
ax_loss.set_ylabel("Training loss (CE + KD)")
ax_loss.legend(fontsize=8, frameon=False)

ax_ppl.set_yscale("log")
ax_ppl.axhline(TEACHER_PPL, color="gray", linestyle=":", linewidth=1,
               label=f"QKV teacher ({TEACHER_PPL:.1f})")
ax_ppl.axhline(SCRATCH_CEILING_PPL, color="gray", linestyle="--", linewidth=1,
               label=f"Q≠K=V scratch ({SCRATCH_CEILING_PPL:.1f})")
ax_ppl.set_xlabel("% of 500M-token distillation budget")
ax_ppl.set_ylabel("Validation perplexity (log scale)")
ax_ppl.legend(fontsize=7, frameon=False)

fig.tight_layout()
fig.savefig("distill_llm_curves.pdf")
fig.savefig("distill_llm_curves.png", dpi=150)
print("wrote distill_llm_curves.{pdf,png}")
