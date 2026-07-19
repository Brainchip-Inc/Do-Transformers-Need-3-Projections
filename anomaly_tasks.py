"""
Set Anomaly Detection for the QKV projection-sharing variants (paper Table 2, "Anomaly").

Task: given a SET of 10 images (9 from one CIFAR-100 class + 1 from a different class),
find the odd one out. Following the paper, images are encoded with a frozen ImageNet-pretrained
ResNet34 (512-d features), and a permutation-equivariant set transformer (no positional
embedding) scores each element; the argmax is the predicted anomaly. Metric: odd-one-out accuracy.

Reuses the six attention variants from synthetic_tasks.py (SharedProjAttention / VARIANTS), so
attention semantics match the synthetic and vision experiments exactly.

ResNet34 features are extracted once and cached to vision_data/cifar100_r34_{train,test}.pt.

Usage:
    conda run -n torch_env python anomaly_tasks.py --quick --device cuda:2
    conda run -n torch_env python anomaly_tasks.py --device cuda:2          # 24-run minimal sweep
    conda run -n torch_env python anomaly_tasks.py --merge                  # -> anomaly_results.md
"""

import os
import csv
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from synthetic_tasks import SharedProjAttention, VARIANTS

FEAT_DIM = 512          # ResNet34 penultimate
SET_SIZE = 10           # 9 normal + 1 anomaly
NUM_CLASSES = 100       # CIFAR-100

# ============================================================================
# RESNET34 FEATURE EXTRACTION (cached)
# ============================================================================

def _extract_features(data_root, train, device):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet
    ])
    ds = torchvision.datasets.CIFAR100(data_root, train=train, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=6, pin_memory=True)

    net = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net = net.to(device).eval()

    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                f = net(x.to(device))
            feats.append(f.float().cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def get_features(data_root, split, device):
    """Return (features [N,512], labels [N]) for split in {train,test}, cached to disk."""
    path = os.path.join(data_root, f"cifar100_r34_{split}.pt")
    if os.path.exists(path):
        d = torch.load(path)
        return d["feats"], d["labels"]
    print(f"Extracting ResNet34 features for CIFAR-100 {split}...", flush=True)
    feats, labels = _extract_features(data_root, split == "train", device)
    torch.save({"feats": feats, "labels": labels}, path)
    print(f"Cached {feats.shape} to {path}", flush=True)
    return feats, labels


# ============================================================================
# SET DATASET (sampled on the fly, seeded)
# ============================================================================

def build_class_index(labels):
    idx = {}
    for i, c in enumerate(labels.tolist()):
        idx.setdefault(c, []).append(i)
    return idx


def sample_sets(feats, class_idx, n_sets, gen):
    """n_sets sets of SET_SIZE feature vectors (9 normal + 1 anomaly). Returns (X, y).

    X: [n_sets, SET_SIZE, 512], y: [n_sets] anomaly position.
    """
    classes = list(class_idx.keys())
    C = len(classes)
    X = torch.empty(n_sets, SET_SIZE, FEAT_DIM)
    y = torch.empty(n_sets, dtype=torch.long)
    for s in range(n_sets):
        ci = classes[torch.randint(C, (1,), generator=gen).item()]
        aj = classes[torch.randint(C, (1,), generator=gen).item()]
        while aj == ci:
            aj = classes[torch.randint(C, (1,), generator=gen).item()]
        normal_pool, anom_pool = class_idx[ci], class_idx[aj]
        ni = [normal_pool[torch.randint(len(normal_pool), (1,), generator=gen).item()]
              for _ in range(SET_SIZE - 1)]
        anom = anom_pool[torch.randint(len(anom_pool), (1,), generator=gen).item()]
        pos = torch.randint(SET_SIZE, (1,), generator=gen).item()
        members = ni[:pos] + [anom] + ni[pos:]
        X[s] = feats[torch.tensor(members)]
        y[s] = pos
    return X, y


# ============================================================================
# SET ENCODER (permutation-equivariant; reuses the 6 attention variants)
# ============================================================================

class SetBlock(nn.Module):
    def __init__(self, d, n_head, share, plus):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = SharedProjAttention(d, n_head, share, plus, pos_dim=10, max_len=SET_SIZE)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SetAnomalyModel(nn.Module):
    """No positional embedding -> permutation-equivariant over the set."""

    def __init__(self, d, n_layer, n_head, variant):
        super().__init__()
        share, plus = VARIANTS[variant]
        self.embed = nn.Linear(FEAT_DIM, d)
        self.blocks = nn.ModuleList([SetBlock(d, n_head, share, plus) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x, targets=None):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(self.ln_f(h)).squeeze(-1)   # [B, SET_SIZE]
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss


# ============================================================================
# TRAIN / EVAL ONE RUN
# ============================================================================

def run_one(variant, n_embd, n_layer, n_head, seed, device, feats_tr, idx_tr,
            feats_te, idx_te, epochs=20, n_train=10000, n_test=2000, batch_size=128, lr=1e-3):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Xtr, ytr = sample_sets(feats_tr, idx_tr, n_train, g)
    Xte, yte = sample_sets(feats_te, idx_te, n_test, g)
    Xte, yte = Xte.to(device), yte.to(device)

    model = SetAnomalyModel(n_embd, n_layer, n_head, variant).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_train, generator=g)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            _, loss = model(Xtr[idx].to(device), ytr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, Xte.size(0), batch_size):
            logits, _ = model(Xte[i:i + batch_size])
            correct += (logits.argmax(-1) == yte[i:i + batch_size]).sum().item()
    return correct / Xte.size(0)


# ============================================================================
# SWEEP DRIVER
# ============================================================================

def build_jobs(quick=False):
    if quick:
        embeds, layers, heads, seeds = [64], [2], [2], [0]
    else:
        embeds, layers, heads, seeds = [64, 256], [2], [2, 4], [0]  # minimal grid: 4 cfg/variant
    jobs = []
    for variant in VARIANTS:
        for e in embeds:
            for L in layers:
                for H in heads:
                    if e % H:
                        continue
                    for s in seeds:
                        jobs.append((variant, e, L, H, s))
    return jobs


def aggregate(csv_path, out_md):
    from collections import defaultdict
    acc = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            acc[row["variant"]].append(float(row["accuracy"]))
    lines = ["| Variant | Anomaly (odd-one-out acc) |", "|---|---|"]
    for v in VARIANTS:
        vals = acc.get(v, [])
        lines.append(f"| {v} | {sum(vals)/len(vals):.3f} |" if vals else f"| {v} | - |")
    table = "\n".join(lines)
    with open(out_md, "w") as f:
        f.write("# Set Anomaly Detection results (odd-one-out accuracy)\n\n" + table + "\n")
    print(table)
    print(f"\nWrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--data-root", type=str, default="./vision_data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out-csv", type=str, default="anomaly_results.csv")
    ap.add_argument("--out-md", type=str, default="anomaly_results.md")
    args = ap.parse_args()

    if args.merge:
        aggregate(args.out_csv, args.out_md)
        return

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    feats_tr, lab_tr = get_features(args.data_root, "train", device)
    feats_te, lab_te = get_features(args.data_root, "test", device)
    idx_tr, idx_te = build_class_index(lab_tr), build_class_index(lab_te)

    jobs = build_jobs(quick=args.quick)
    epochs = 2 if args.quick else args.epochs
    n_train, n_test = (1000, 500) if args.quick else (10000, 2000)
    print(f"device={device}  jobs={len(jobs)}  epochs={epochs}  -> {args.out_csv}", flush=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_embd", "n_layer", "n_head", "seed", "accuracy"])
        for n, (variant, e, L, H, s) in enumerate(jobs):
            try:
                acc = run_one(variant, e, L, H, s, device, feats_tr, idx_tr, feats_te, idx_te,
                              epochs=epochs, n_train=n_train, n_test=n_test)
            except Exception as ex:
                print(f"[{n+1}/{len(jobs)}] ERROR {variant} d{e} H{H}: {repr(ex)[:120]}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            w.writerow([variant, e, L, H, s, f"{acc:.4f}"])
            f.flush()
            print(f"[{n+1}/{len(jobs)}] {variant} d{e} L{L} H{H} s{s} -> acc {acc:.4f}", flush=True)

    aggregate(args.out_csv, args.out_md)


if __name__ == "__main__":
    main()
