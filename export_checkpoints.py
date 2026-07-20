"""
Train and export Q!=K=V (the paper's headline K=V variant) checkpoints for the
synthetic and vision-classification tasks, for release on the Hugging Face Hub.

Produces one checkpoint per synthetic task (REVERSE/SORT/SUB/SWAP/COPY) and one per
vision-classification dataset (MNIST/FMNIST/CIFAR-10/CIFAR-100), each a dict with the
model state, the config needed to rebuild the model, and the test accuracy. Reuses the
models and the SharedProjAttention variant from synthetic_tasks.py / vision_tasks.py.

Run in an sm_61-compatible env with torchvision (e.g. torch_env):
    conda run -n torch_env python export_checkpoints.py --device cuda:2
"""

import os
import argparse

import torch

from synthetic_tasks import Encoder, ModelCfg, make_dataset, TASKS
import vision_tasks as V

CKPT_DIR = "checkpoints"
# filename tag per variant: qkv = baseline, qkv_kv = headline Q!=K=V (K=V)
VARIANT_TAG = {"QKV": "qkv", "Q!=K=V": "qkv_kv"}

# representative configs
SYN_CFG = dict(n_embd=256, n_layer=2, n_head=4, seq_len=64, epochs=10,
               n_train=10000, n_test=2000, batch_size=128, lr=1e-3)
VIS_CFG = dict(patch=4, n_embd=256, n_layer=2, n_head=4, lr=1e-3, batch_size=128)


def export_synthetic(device, variant):
    import math
    import torch.nn as nn
    from synthetic_tasks import lr_lambda_factory
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = VARIANT_TAG[variant]
    c = SYN_CFG
    for task in TASKS:
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        x, y = make_dataset(c["n_train"], c["seq_len"], task, g)
        xte, yte = make_dataset(c["n_test"], c["seq_len"], task, g)
        cfg = ModelCfg(n_embd=c["n_embd"], n_layer=c["n_layer"], n_head=c["n_head"],
                       seq_len=c["seq_len"], variant=variant)
        model = Encoder(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
        steps = math.ceil(c["n_train"] / c["batch_size"]) * c["epochs"]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(steps, 0.05))
        model.train()
        for _ in range(c["epochs"]):
            perm = torch.randperm(c["n_train"], generator=g)
            for i in range(0, c["n_train"], c["batch_size"]):
                idx = perm[i:i + c["batch_size"]]
                _, loss = model(x[idx].to(device), y[idx].to(device))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(xte.to(device))[0].argmax(-1)
            acc = (pred == yte.to(device)).float().mean().item()
        path = os.path.join(CKPT_DIR, f"synthetic_{task.lower()}_{tag}.pt")
        torch.save({"task": "synthetic-" + task, "variant": variant,
                    "model": "Encoder", "config": vars(cfg),
                    "test_accuracy": round(acc, 4),
                    "model_state_dict": model.state_dict()}, path)
        print(f"[synthetic] {task}: acc {acc:.4f} -> {path}", flush=True)


def export_vision(device, variant):
    import torch.nn.functional as F
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = VARIANT_TAG[variant]
    c = VIS_CFG
    for ds in V.CLASSIFICATION:
        ch, native, ncls = V.CLASSIFICATION[ds]
        E = V.EPOCHS[ds]
        torch.manual_seed(0)
        tr = V.get_classification_dataset(ds, "./vision_data", True)
        te = V.get_classification_dataset(ds, "./vision_data", False)
        cfg = V.ViTConfig(image_size=native, channels=ch, patch=c["patch"], n_classes=ncls,
                          n_embd=c["n_embd"], n_layer=c["n_layer"], n_head=c["n_head"], variant=variant)
        model = V.ViT(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.5 * E), int(0.75 * E)], gamma=0.1)
        tl = V._loader(tr, c["batch_size"], 4, True)
        for _ in range(E):
            model.train()
            for xb, yb in tl:
                loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        acc = V.evaluate(model, V._loader(te, c["batch_size"], 4, False), device)
        path = os.path.join(CKPT_DIR, f"vision_{ds}_{tag}.pt")
        torch.save({"task": "vision-" + ds, "variant": variant,
                    "model": "ViT", "config": {k: getattr(cfg, k) for k in vars(cfg)},
                    "test_accuracy": round(acc, 4),
                    "model_state_dict": model.state_dict()}, path)
        print(f"[vision] {ds}: acc {acc:.4f} -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--variant", default="Q!=K=V", choices=list(VARIANT_TAG),
                    help="QKV (baseline) or Q!=K=V (headline)")
    ap.add_argument("--only", choices=["synthetic", "vision", "both"], default="both")
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  variant={args.variant}  -> {CKPT_DIR}/", flush=True)
    if args.only in ("synthetic", "both"):
        export_synthetic(device, args.variant)
    if args.only in ("vision", "both"):
        export_vision(device, args.variant)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
