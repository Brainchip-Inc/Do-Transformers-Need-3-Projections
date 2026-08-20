import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

syn = {
    "REVERSE": (1.0, 0.1000, 0.1226), "SORT": (0.9975, 0.5830, 0.7170),
    "SUB": (1.0, 1.0000, 1.0000), "SWAP": (1.0, 0.0879, 0.1223),
    "COPY": (1.0, 1.0000, 1.0000),
}
vis = {
    "mnist": (0.9814, 0.1143, 0.1511), "fmnist": (0.8874, 0.0835, 0.1047),
    "cifar10": (0.6959, 0.0760, 0.1525), "cifar100": (0.4374, 0.0067, 0.0214),
}

cka = {}
with open("kv_similarity_results.csv") as f:
    for row in csv.DictReader(f):
        cka[(row["domain"], row["name"])] = float(row["activation_cka"])

names, xs, ys, domains = [], [], [], []
for name, (teacher, kk, kv) in syn.items():
    drop = teacher - (kk + kv) / 2
    names.append(name); xs.append(cka[("synthetic", name)]); ys.append(drop); domains.append("synthetic")
for name, (teacher, kk, kv) in vis.items():
    drop = teacher - (kk + kv) / 2
    names.append(name); xs.append(cka[("vision", name.lower())]); ys.append(drop); domains.append("vision")

xs, ys = np.array(xs), np.array(ys)
r = np.corrcoef(xs, ys)[0, 1]
print(f"Pearson r (CKA vs collapse severity) = {r:.4f}, n={len(xs)}")
for n, x, y, d in zip(names, xs, ys, domains):
    print(f"  {d:9s} {n:10s} CKA={x:.4f}  collapse={y:.4f}")

fig, ax = plt.subplots(figsize=(4.6, 3.6))
for d, marker, color in [("synthetic", "o", "#3b6fa0"), ("vision", "^", "#c0623b")]:
    mask = [dd == d for dd in domains]
    ax.scatter(xs[mask], ys[mask], marker=marker, color=color, s=60, label=d, zorder=3)
label_offset = {"REVERSE": (4, 8), "SWAP": (4, -12), "SUB": (-28, 8), "COPY": (4, -12)}
for n, x, y in zip(names, xs, ys):
    ax.annotate(n, (x, y), fontsize=7, xytext=label_offset.get(n, (4, 3)), textcoords="offset points")
m, b = np.polyfit(xs, ys, 1)
xline = np.linspace(xs.min() - 0.03, xs.max() + 0.03, 10)
ax.plot(xline, m * xline + b, "--", color="gray", linewidth=1, zorder=1, label=f"fit (r={r:.2f})")
ax.set_xlabel("Activation linear CKA(K, V)")
ax.set_ylabel("Accuracy drop under K=V surgery")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("kv_cka_vs_collapse.pdf")
fig.savefig("kv_cka_vs_collapse.png", dpi=150)
print("wrote kv_cka_vs_collapse.{pdf,png}")
