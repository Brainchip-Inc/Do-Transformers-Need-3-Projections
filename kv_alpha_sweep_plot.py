"""Plot the alpha-interpolation sweep (kv_alpha_sweep_results.csv) as one panel per
task/dataset: zero-shot accuracy vs. alpha = weight on K (alpha=0 is keep_v, alpha=1
is keep_k, alpha=0.5 is avg), with the random-projection baseline shown as a shaded
mean +/- 1 std band over 5 random seeds (not a single number -- reviewer request: the
baseline itself has sampling variance and that variance is the point).

Usage:
    conda run -n torch_env python kv_alpha_sweep_plot.py
"""

import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("kv_alpha_sweep_results.csv")))
by_name = {}
for r in rows:
    by_name.setdefault(r["name"], {"alpha": [], "acc": [], "random": None, "random_std": None})
    if r["alpha"] == "random":
        by_name[r["name"]]["random"] = float(r["acc"])
        by_name[r["name"]]["random_std"] = float(r["acc_std"])
    else:
        by_name[r["name"]]["alpha"].append(float(r["alpha"]))
        by_name[r["name"]]["acc"].append(float(r["acc"]))

syn_order = ["REVERSE", "SORT", "SUB", "SWAP", "COPY"]
vis_order = ["mnist", "fmnist", "cifar10", "cifar100"]


def plot_random_band(ax, d):
    mu, sd = d["random"], d["random_std"]
    ax.axhspan(mu - sd, mu + sd, color="gray", alpha=0.25, linewidth=0, label="random c_kv (mean±1 std, 5 seeds)")
    ax.axhline(mu, color="gray", linestyle="--", linewidth=1)


fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
for j, name in enumerate(syn_order):
    d = by_name[name]
    ax = axes[0, j]
    ax.plot(d["alpha"], d["acc"], marker="o", markersize=3, color="#9467bd")
    plot_random_band(ax, d)
    ax.set_title(name, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=8)
for j, name in enumerate(vis_order):
    d = by_name[name]
    ax = axes[1, j]
    ax.plot(d["alpha"], d["acc"], marker="o", markersize=3, color="#9467bd", label="alpha blend")
    plot_random_band(ax, d)
    ax.set_title(name.upper(), fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=8)
axes[1, 4].axis("off")

axes[0, 0].set_ylabel("Synthetic\nzero-shot accuracy", fontsize=9)
axes[1, 0].set_ylabel("Vision\nzero-shot accuracy", fontsize=9)
for ax in list(axes[0, :]) + list(axes[1, :4]):
    ax.set_xlabel(r"$\alpha$ (0=keep_v, 1=keep_k)", fontsize=8)

handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.06), fontsize=10)
fig.suptitle(r"Zero-shot accuracy vs. $c_{kv} = \alpha W_K + (1-\alpha) W_V$, "
             "vs. a random (untrained) $c_{kv}$ baseline", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("kv_alpha_sweep.pdf")
fig.savefig("kv_alpha_sweep.png", dpi=150)
print("Wrote kv_alpha_sweep.pdf / .png")
