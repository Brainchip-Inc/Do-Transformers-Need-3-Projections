"""
Vision-classification benchmark for QKV projection-sharing variants (paper Table 2,
anomaly detection excluded).

Trains a Vision Transformer (ViT) from scratch across the six attention variants on:
    MNIST, FashionMNIST, CIFAR-10, CIFAR-100   (classification sweep, averaged)
    TinyImageNet                                (large ViT, reported separately)

The six variants and their (X)+ 2D positional injection are imported verbatim from
`synthetic_tasks.py` (SharedProjAttention / VARIANTS), so the attention semantics are
byte-for-byte identical to the synthetic-tasks experiments. Metric is top-1 accuracy.

This follows the paper's Section 4.2 setup — ViT classifier, cross-entropy, Adam +
MultiStepLR, the {4,7}x{1e-3,1e-4}x{64,256,512}x{2,4}x{2,4} sweep with 2 runs each, and
per-dataset epoch budgets (MNIST/FMNIST 20, CIFAR-10 40, CIFAR-100 50) — with faithful-but-
modernized choices (pre-norm blocks, learned positional embeddings, light train-time
augmentation for CIFAR/TinyImageNet, seeded runs). All knobs are CLI flags.

Run environment: use an sm_61-compatible conda env WITH torchvision — e.g. `torch_env`
(torch 2.7 + torchvision 0.22) — on a free GPU. The `kv` env used for synthetic_tasks has
no torchvision.

The sweep size is controlled by --grid:
    minimal (default) : 4 configs per (variant,dataset), ~99 runs total — fast ranking check (~a day)
    full              : paper-faithful grid, ~2,310 runs — weeks of GPU time

Usage:
    conda run -n torch_env python vision_tasks.py --quick --device cuda:2
    conda run -n torch_env python vision_tasks.py --device cuda:2               # minimal sweep (default)
    conda run -n torch_env python vision_tasks.py --grid full --shard 0/2 --device cuda:2
    conda run -n torch_env python vision_tasks.py --dataset cifar10 --device cuda:2
    conda run -n torch_env python vision_tasks.py --tiny --device cuda:2        # TinyImageNet
    conda run -n torch_env python vision_tasks.py --merge

TinyImageNet is not in torchvision. Download + unzip once:
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P <data_root>
    unzip <data_root>/tiny-imagenet-200.zip -d <data_root>
The loader reorganizes the val split into class folders automatically on first use.
"""

import os
import csv
import math
import time
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms

# reuse the EXACT attention variants from the synthetic-tasks code
from synthetic_tasks import SharedProjAttention, VARIANTS

# ============================================================================
# DATASETS
# ============================================================================

# name -> (channels, native image size, num classes)
CLASSIFICATION = {
    "mnist":    (1, 28, 10),
    "fmnist":   (1, 28, 10),
    "cifar10":  (3, 32, 10),
    "cifar100": (3, 32, 100),
}
# per-dataset epoch budgets from the paper
EPOCHS = {"mnist": 20, "fmnist": 20, "cifar10": 40, "cifar100": 50}
# average column in the paper excludes TinyImageNet (and anomaly, which we skip)
AVG_DATASETS = ["mnist", "fmnist", "cifar10", "cifar100"]

_NORM = {
    "mnist":    ((0.1307,), (0.3081,)),
    "fmnist":   ((0.2860,), (0.3530,)),
    "cifar10":  ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def build_transforms(name, train):
    mean, std = _NORM[name]
    tfm = []
    if train and name in ("cifar10", "cifar100"):
        tfm += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    tfm += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(tfm)


def get_classification_dataset(name, data_root, train):
    tfm = build_transforms(name, train)
    if name == "mnist":
        return torchvision.datasets.MNIST(data_root, train=train, download=True, transform=tfm)
    if name == "fmnist":
        return torchvision.datasets.FashionMNIST(data_root, train=train, download=True, transform=tfm)
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(data_root, train=train, download=True, transform=tfm)
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(data_root, train=train, download=True, transform=tfm)
    raise ValueError(name)


# --- TinyImageNet (not in torchvision) ---------------------------------------

TINY_MEAN, TINY_STD = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)


def _prepare_tinyimagenet_val(root):
    """Reorganize tiny-imagenet-200/val/images + val_annotations.txt into class folders."""
    val_dir = os.path.join(root, "val")
    img_dir = os.path.join(val_dir, "images")
    ann = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.isdir(img_dir) or not os.path.exists(ann):
        return  # already reorganized (or not downloaded yet)
    import shutil
    with open(ann) as f:
        for line in f:
            fname, wnid = line.split("\t")[:2]
            cls_dir = os.path.join(val_dir, wnid)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(img_dir, fname)
            if os.path.exists(src):
                shutil.move(src, os.path.join(cls_dir, fname))
    shutil.rmtree(img_dir, ignore_errors=True)


def get_tinyimagenet_dataset(data_root, train, image_size=224):
    root = os.path.join(data_root, "tiny-imagenet-200")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"TinyImageNet not found at {root}. Download with:\n"
            f"  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P {data_root}\n"
            f"  unzip {data_root}/tiny-imagenet-200.zip -d {data_root}")
    split = "train" if train else "val"
    if not train:
        _prepare_tinyimagenet_val(root)
    aug = [transforms.RandomHorizontalFlip()] if train else []
    tfm = transforms.Compose([transforms.Resize((image_size, image_size))] + aug +
                             [transforms.ToTensor(), transforms.Normalize(TINY_MEAN, TINY_STD)])
    # train layout is train/<wnid>/images/*.JPEG -> point ImageFolder one level in via loader
    folder = os.path.join(root, split)
    return _TinyImageFolder(folder, transform=tfm)


class _TinyImageFolder(torch.utils.data.Dataset):
    """ImageFolder that tolerates the train/<wnid>/images/ extra nesting."""

    def __init__(self, folder, transform):
        from torchvision.datasets.folder import default_loader
        self.loader, self.transform = default_loader, transform
        wnids = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}
        self.samples = []
        for w in wnids:
            cdir = os.path.join(folder, w)
            imgs = os.path.join(cdir, "images")
            scan = imgs if os.path.isdir(imgs) else cdir
            for fn in os.listdir(scan):
                if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(scan, fn), self.class_to_idx[w]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(self.loader(path)), label


# ============================================================================
# VISION TRANSFORMER (reuses SharedProjAttention for the six variants)
# ============================================================================

class DropPath(nn.Module):
    """Stochastic depth: drops the whole residual branch per-sample at rate `p`
    during training, scaling surviving branches by 1/(1-p) to keep the expectation
    unchanged (standard DeiT/timm formulation)."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep)
        return x * mask / keep


@dataclass
class ViTConfig:
    image_size: int
    channels: int
    patch: int
    n_classes: int
    n_embd: int
    n_layer: int
    n_head: int
    variant: str
    pos_dim: int = 50
    n_inner_mult: int = 4
    dropout: float = 0.1
    drop_path: float = 0.0  # max stochastic-depth rate (linearly scaled across layers)


class PatchEmbed(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.patch = cfg.patch
        self.grid = math.ceil(cfg.image_size / cfg.patch)  # pad up if not divisible
        self.num_patches = self.grid ** 2
        self.proj = nn.Conv2d(cfg.channels, cfg.n_embd, kernel_size=cfg.patch, stride=cfg.patch)

    def forward(self, x):
        target = self.grid * self.patch
        ph, pw = target - x.shape[2], target - x.shape[3]
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        x = self.proj(x)                       # [B, d, grid, grid]
        return x.flatten(2).transpose(1, 2)    # [B, num_patches, d]


class ViTBlock(nn.Module):
    def __init__(self, cfg: ViTConfig, share, plus, max_len, drop_path=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SharedProjAttention(cfg.n_embd, cfg.n_head, share, plus,
                                        pos_dim=cfg.pos_dim, max_len=max_len)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_inner_mult * cfg.n_embd),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.n_inner_mult * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )
        # stochastic depth: drop the whole residual branch per-sample at rate drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        n = self.patch_embed.num_patches + 1                    # + cls token
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.pos_emb = nn.Parameter(torch.randn(1, n, cfg.n_embd) * 0.02)
        self.drop = nn.Dropout(cfg.dropout)
        share, plus = VARIANTS[cfg.variant]
        # linearly scale stochastic-depth rate across layers (0 -> cfg.drop_path), DeiT-style
        dpr = [cfg.drop_path * i / max(1, cfg.n_layer - 1) for i in range(cfg.n_layer)]
        self.blocks = nn.ModuleList([ViTBlock(cfg, share, plus, max_len=n, drop_path=dpr[i])
                                     for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.n_classes)

    def forward(self, x):
        B = x.size(0)
        h = self.patch_embed(x)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos_emb
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h[:, 0])              # classify from cls token


# ============================================================================
# TRAIN / EVAL ONE RUN
# ============================================================================

def _loader(ds, batch_size, workers, train):
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                      pin_memory=True, drop_last=train, persistent_workers=workers > 0)


@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def run_one(variant, dataset, patch, lr, n_embd, n_layer, n_head, run_idx, device,
            epochs=None, data_root="./vision_data", batch_size=128, workers=4,
            image_size=None, subset=None, tiny=False):
    """Train one ViT on one dataset/config. Returns test top-1 accuracy."""
    torch.manual_seed(run_idx)

    # TinyImageNet uses a large ViT that needs mixed precision to fit in 11 GB
    amp = tiny
    if tiny:
        channels, native, n_classes = 3, 224, 200
        train_ds = get_tinyimagenet_dataset(data_root, True, image_size or 224)
        test_ds = get_tinyimagenet_dataset(data_root, False, image_size or 224)
        img = image_size or 224
        E = epochs or 30
    else:
        channels, native, n_classes = CLASSIFICATION[dataset]
        train_ds = get_classification_dataset(dataset, data_root, True)
        test_ds = get_classification_dataset(dataset, data_root, False)
        img = image_size or native
        E = epochs if epochs is not None else EPOCHS[dataset]

    if subset:
        train_ds = Subset(train_ds, range(min(subset, len(train_ds))))
        test_ds = Subset(test_ds, range(min(subset, len(test_ds))))

    cfg = ViTConfig(image_size=img, channels=channels, patch=patch, n_classes=n_classes,
                    n_embd=n_embd, n_layer=n_layer, n_head=n_head, variant=variant)
    model = ViT(cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(0.5 * E), int(0.75 * E)], gamma=0.1)

    train_loader = _loader(train_ds, batch_size, workers, True)
    test_loader = _loader(test_ds, batch_size, workers, False)

    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    for _ in range(E):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

    return evaluate(model, test_loader, device, amp=amp)


# ============================================================================
# SWEEP DRIVER
# ============================================================================

def build_jobs(datasets, grid="minimal", quick=False):
    """(variant, dataset, patch, lr, embed, layers, heads, run) for the classification sweep.

    grid="full"    : paper-faithful — 96 runs per (variant,dataset), ~2,304 total. Weeks of GPU.
    grid="minimal" : 4 configs per (variant,dataset) (embed{64,256} x heads{2,4}, 1 run),
                     ~96 classification runs total — a fast ranking sanity check (~a day).
    """
    if quick:
        patches, lrs, embeds, layers, heads, runs = [7], [1e-3], [64], [2], [2], [0]
    elif grid == "minimal":
        patches = [4]
        lrs = [1e-3]
        embeds = [64, 256]
        layers = [2]
        heads = [2, 4]
        runs = [0]
    else:  # full (paper-faithful)
        patches = [4, 7]
        lrs = [1e-3, 1e-4]
        embeds = [64, 256, 512]
        layers = [2, 4]
        heads = [2, 4]
        runs = [0, 1]
    jobs = []
    for variant in VARIANTS:
        for ds in datasets:
            for p in patches:
                for lr in lrs:
                    for e in embeds:
                        for L in layers:
                            for H in heads:
                                if e % H != 0:
                                    continue
                                for r in runs:
                                    jobs.append((variant, ds, p, lr, e, L, H, r))
    return jobs


def aggregate(csv_paths, out_md):
    from collections import defaultdict
    acc = defaultdict(list)
    datasets_seen = set()
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                acc[(row["variant"], row["dataset"])].append(float(row["accuracy"]))
                datasets_seen.add(row["dataset"])

    cols = [d for d in AVG_DATASETS if d in datasets_seen]
    extra = [d for d in sorted(datasets_seen) if d not in AVG_DATASETS]  # e.g. tinyimagenet
    header = ["Variant"] + cols + (["Average"] if cols else []) + extra
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for variant in VARIANTS:
        cells, means = [], []
        for d in cols:
            vals = acc.get((variant, d), [])
            if vals:
                m = sum(vals) / len(vals); means.append(m); cells.append(f"{m:.3f}")
            else:
                cells.append("-")
        avg = [f"{sum(means)/len(means):.3f}" if means else "-"] if cols else []
        for d in extra:
            vals = acc.get((variant, d), [])
            cells.append(f"{sum(vals)/len(vals):.3f}" if vals else "-")
        row_cells = cells[:len(cols)] + avg + cells[len(cols):]
        lines.append(f"| {variant} | " + " | ".join(row_cells) + " |")

    table = "\n".join(lines)
    with open(out_md, "w") as f:
        f.write("# Vision-tasks results (top-1 accuracy, averaged over sweep)\n\n")
        f.write("Average excludes TinyImageNet (and anomaly, which is out of scope).\n\n")
        f.write(table + "\n")
    print(table)
    print(f"\nWrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (1 epoch, subset)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="run TinyImageNet (large ViT) instead of the sweep")
    ap.add_argument("--grid", choices=["minimal", "full"], default="minimal",
                    help="minimal (~96 runs, default) or full paper-faithful grid (~2,304 runs)")
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None, help="restrict classification sweep to one dataset")
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override per-dataset epoch budget")
    ap.add_argument("--data-root", type=str, default="./vision_data")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tiny-variants", type=str, default="QKV,Q=K!=V,Q=K=V",
                    help="comma list; paper reports TinyImageNet for these three")
    ap.add_argument("--tiny-epochs", type=int, default=30)
    ap.add_argument("--tiny-batch", type=int, default=32,
                    help="TinyImageNet batch size (32 fits the big ViT in 11 GB with AMP)")
    ap.add_argument("--tiny-runs", type=int, default=1, help="repeats per TinyImageNet variant")
    ap.add_argument("--tiny-out", type=str, default="vision_results_tiny.csv",
                    help="per-process CSV for TinyImageNet (use distinct names when running in parallel)")
    ap.add_argument("--out-md", type=str, default="vision_results.md")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    if args.merge:
        import glob
        existing = [f"vision_results_shard{i}.csv" for i in range(64)]
        existing += ["vision_results.csv"]
        existing += glob.glob("vision_results_tiny*.csv")
        existing = [p for p in dict.fromkeys(existing) if os.path.exists(p)]
        aggregate(existing, args.out_md)
        return

    # ---- TinyImageNet path (separate, large ViT, run twice) ----
    if args.tiny:
        variants = [v.strip() for v in args.tiny_variants.split(",")] if not args.variant else [args.variant]
        out_csv = args.tiny_out
        print(f"device={device}  TinyImageNet  variants={variants}  -> {out_csv}")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "dataset", "patch", "lr", "n_embd", "n_layer",
                        "n_head", "run", "accuracy"])
            for variant in variants:
                for r in ([0] if args.quick else list(range(args.tiny_runs))):
                    try:
                        acc = run_one(variant, "tinyimagenet", 16, 1e-4, 768, 12, 12, r, device,
                                      epochs=(1 if args.quick else args.tiny_epochs),
                                      data_root=args.data_root, batch_size=args.tiny_batch,
                                      workers=args.workers, subset=(256 if args.quick else None),
                                      tiny=True)
                    except Exception as ex:
                        print(f"ERROR tiny {variant} run{r}: {repr(ex)[:160]}", flush=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    w.writerow([variant, "tinyimagenet", 16, 1e-4, 768, 12, 12, r, f"{acc:.4f}"])
                    f.flush()
                    print(f"tiny {variant} run{r} -> acc {acc:.4f}", flush=True)
        return

    # ---- classification sweep ----
    datasets = [args.dataset] if args.dataset else list(CLASSIFICATION)
    i, N = (int(v) for v in args.shard.split("/"))
    jobs = build_jobs(datasets, grid=args.grid, quick=args.quick)
    if args.variant:
        jobs = [j for j in jobs if j[0] == args.variant]
    jobs = jobs[i::N]

    out_csv = f"vision_results_shard{i}.csv" if N > 1 else "vision_results.csv"
    print(f"device={device}  shard={i}/{N}  datasets={datasets}  jobs={len(jobs)}  -> {out_csv}",
          flush=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "dataset", "patch", "lr", "n_embd", "n_layer",
                    "n_head", "run", "accuracy"])
        t0 = time.time()
        for n, (variant, ds, p, lr, e, L, H, r) in enumerate(jobs):
            try:
                acc = run_one(variant, ds, p, lr, e, L, H, r, device,
                              epochs=(1 if args.quick else args.epochs),
                              data_root=args.data_root, batch_size=args.batch_size,
                              workers=args.workers, subset=(512 if args.quick else None))
            except Exception as ex:
                print(f"[{n+1}/{len(jobs)}] ERROR {variant} {ds} p{p} lr{lr} d{e} "
                      f"L{L} H{H} r{r}: {repr(ex)[:140]}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            w.writerow([variant, ds, p, lr, e, L, H, r, f"{acc:.4f}"])
            f.flush()
            el = time.time() - t0
            print(f"[{n+1}/{len(jobs)}] {variant} {ds} p{p} lr{lr} d{e} L{L} H{H} r{r} "
                  f"-> acc {acc:.4f} ({el:.0f}s)", flush=True)

    if N == 1 and not args.dataset:
        aggregate([out_csv], args.out_md)
    else:
        print(f"Done. After all shards/datasets finish: python {os.path.basename(__file__)} --merge")


if __name__ == "__main__":
    main()
