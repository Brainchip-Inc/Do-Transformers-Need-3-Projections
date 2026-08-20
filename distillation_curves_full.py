"""
Full per-epoch distillation-recovery curves for every (task/dataset, surgery-mode)
configuration in theory/kv_tying_theory.tex -- the paper references these curves
("still improving at epoch 5", "already plateauing by epoch 8-9") but only ever
shows the endpoint accuracies in its tables. This renders all of them.

Sources (already-completed runs, no retraining):
    distillation_synthetic_results.csv       -- 5 tasks x 3 modes, 5 epochs
    distillation_vision_results.csv          -- 4 datasets x 3 modes, 5 epochs
    distillation_vision_cifar10_10ep.csv      -- CIFAR-10, 3 modes, 10 epochs
    distillation_vision_cifar100_10ep.csv     -- CIFAR-100, 3 modes, 10 epochs
(the two _10ep files supersede the 5-epoch CIFAR rows in distillation_vision_results.csv)

Usage:
    conda run -n torch_env python distillation_curves_full.py
"""

import csv
import matplotlib.pyplot as plt

MODE_COLOR = {"keep_k": "#d62728", "keep_v": "#1f77b4", "avg": "#2ca02c"}


def load(path, key_col):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        name = r[key_col]
        epoch_cols = sorted(
            [c for c in r if c.startswith("distill_epoch")],
            key=lambda c: int(c.replace("distill_epoch", "").replace("_acc", "")))
        epochs = [int(c.replace("distill_epoch", "").replace("_acc", "")) for c in epoch_cols]
        accs = [float(r[c]) for c in epoch_cols]
        out.setdefault(name, {})[r["mode"]] = (epochs, accs, float(r["scratch_qkv_kv_acc"]))
    return out


def plot_panel(ax, title, mode_curves, ceiling):
    for mode, (epochs, accs, _) in sorted(mode_curves.items()):
        ax.plot(epochs, accs, marker="o", markersize=3, label=mode, color=MODE_COLOR[mode])
    ax.axhline(ceiling, color="gray", linestyle="--", linewidth=1, label="scratch ceiling")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=8)


def main():
    synthetic = load("distillation_synthetic_results.csv", "task")
    vision5 = load("distillation_vision_results.csv", "dataset")
    cifar10_10 = load("distillation_vision_cifar10_10ep.csv", "dataset")
    cifar100_10 = load("distillation_vision_cifar100_10ep.csv", "dataset")
    # 10-epoch CIFAR runs supersede the 5-epoch rows for those two datasets
    vision = dict(vision5)
    vision["cifar10"] = cifar10_10["cifar10"]
    vision["cifar100"] = cifar100_10["cifar100"]

    syn_order = ["REVERSE", "SORT", "SUB", "SWAP", "COPY"]
    vis_order = ["mnist", "fmnist", "cifar10", "cifar100"]

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    for j, name in enumerate(syn_order):
        ceiling = next(iter(synthetic[name].values()))[2]
        plot_panel(axes[0, j], name, synthetic[name], ceiling)
    for j, name in enumerate(vis_order):
        ceiling = next(iter(vision[name].values()))[2]
        plot_panel(axes[1, j], name.upper(), vision[name], ceiling)
    axes[1, 4].axis("off")

    axes[0, 0].set_ylabel("Synthetic\ntoken accuracy", fontsize=9)
    axes[1, 0].set_ylabel("Vision\ntop-1 accuracy", fontsize=9)
    for ax in list(axes[0, :]) + list(axes[1, :4]):
        ax.set_xlabel("distillation epoch", fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.06), fontsize=10)
    fig.suptitle("Distillation recovery curves: all (task/dataset, surgery-mode) configurations",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("distillation_curves_full.pdf")
    fig.savefig("distillation_curves_full.png", dpi=150)
    print("Wrote distillation_curves_full.pdf / .png")


if __name__ == "__main__":
    main()
