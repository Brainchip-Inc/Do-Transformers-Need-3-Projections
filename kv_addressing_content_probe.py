"""
Direct test of the K=addressing / V=content mechanism behind Corollary 1 in
theory/kv_tying_theory.tex, on the 5 synthetic-task QKV checkpoints (reviewer request:
"measure addressing quality from K and content reconstruction from V independently,
rather than inferring both from a single CKA number").

For each task's trained QKV teacher, on held-out data:

  Addressing quality: does argmax_j attn[i,j] (the model's *actual* attention weights,
  captured by re-running each block's attention with return_attn=True on its true
  layer-normed input) match the ground-truth source position the task requires at
  output position i? (COPY/SUB: self; REVERSE: T-1-i; SWAP: swap-halves; SORT:
  argsort(x), which is per-sample and data-dependent.) Reported per layer.

  Content recoverability: fit a closed-form ridge-regression linear probe from k_j
  (resp. v_j) activations to one-hot(x_j) on a train split of the held-out data,
  report classification accuracy (argmax of the probe's prediction) on a held-out
  test split. If Corollary 1 is right, V should probe better than K on the
  routing-dependent tasks (REVERSE/SORT/SWAP) and the two should be close on the
  pointwise tasks (SUB/COPY), where there's no pressure to specialize either way.

Usage:
    conda run -n torch_env python kv_addressing_content_probe.py
"""

import csv
import torch
import torch.nn.functional as F

from synthetic_tasks import Encoder, ModelCfg, make_dataset, make_targets, TASKS, NUM_SYMBOLS

CKPT_DIR = "checkpoints"
DEVICE = torch.device("cpu")  # tiny models; keep off the GPU the live training run needs
N_HELD_OUT = 1000
PROBE_RIDGE = 1e-2


def ground_truth_source_idx(x, task):
    """[B, T] source position each output position i should route from. For SORT
    this is per-sample (argsort); for the others it's the same for every sample."""
    B, T = x.shape
    if task in ("COPY", "SUB"):
        idx = torch.arange(T).unsqueeze(0).expand(B, T)
    elif task == "REVERSE":
        idx = torch.arange(T - 1, -1, -1).unsqueeze(0).expand(B, T)
    elif task == "SWAP":
        half = T // 2
        idx = torch.cat([torch.arange(half, T), torch.arange(0, half)]).unsqueeze(0).expand(B, T)
    elif task == "SORT":
        idx = torch.argsort(x, dim=1, stable=True)
    else:
        raise ValueError(task)
    return idx


def capture_layer_attn_and_kv(model, x_ids):
    """Re-run each block's attention with return_attn=True on its true (layer-normed)
    input, captured via a forward hook on ln1. Returns, per layer: attention weights
    [B,H,T,T] (head-averaged) and the k/v activations [B*T, d]."""
    ln1_inputs = {}
    handles = []
    for i, blk in enumerate(model.blocks):
        def mk(i):
            def hook(_mod, _inp, out):
                ln1_inputs[i] = out.detach()
            return hook
        handles.append(blk.ln1.register_forward_hook(mk(i)))

    with torch.no_grad():
        model(x_ids)
    for h in handles:
        h.remove()

    out = {}
    for i, blk in enumerate(model.blocks):
        xin = ln1_inputs[i]
        with torch.no_grad():
            _, attn = blk.attn(xin, return_attn=True)   # [B,H,T,T]
            k = blk.attn.c_k(xin).reshape(-1, xin.size(-1))
            v = blk.attn.c_v(xin).reshape(-1, xin.size(-1))
        out[i] = {"attn": attn.mean(dim=1), "k": k, "v": v}  # head-average for addressing
    return out


def addressing_accuracy(attn, tgt_idx):
    """attn: [B,T,T] head-averaged attention weights. tgt_idx: [B,T] ground-truth
    source position per output position. Returns fraction where argmax matches."""
    pred = attn.argmax(dim=-1)  # [B,T]
    return (pred == tgt_idx).float().mean().item()


def ridge_probe_accuracy(feat, labels, ridge=PROBE_RIDGE, train_frac=0.8):
    """Closed-form ridge regression from feat [N,d] to one-hot(labels) [N,C], fit on
    a train split, evaluated (argmax classification accuracy) on the held-out split."""
    N = feat.size(0)
    n_train = int(N * train_frac)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    tr, te = perm[:n_train], perm[n_train:]

    Xtr, Xte = feat[tr], feat[te]
    Ytr = F.one_hot(labels[tr], NUM_SYMBOLS).float()
    ytest = labels[te]

    Xtr1 = torch.cat([Xtr, torch.ones(Xtr.size(0), 1)], dim=1)
    d = Xtr1.size(1)
    A = Xtr1.T @ Xtr1 + ridge * torch.eye(d)
    Wb = torch.linalg.solve(A, Xtr1.T @ Ytr)  # [d, C]

    Xte1 = torch.cat([Xte, torch.ones(Xte.size(0), 1)], dim=1)
    pred = (Xte1 @ Wb).argmax(dim=-1)
    return (pred == ytest).float().mean().item()


def main():
    rows = []
    for task in TASKS:
        path = f"{CKPT_DIR}/synthetic_{task.lower()}_qkv.pt"
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ModelCfg(**d["config"])
        model = Encoder(cfg).to(DEVICE).eval()
        model.load_state_dict(d["model_state_dict"])

        g = torch.Generator().manual_seed(0)
        x, _ = make_dataset(N_HELD_OUT, cfg.seq_len, task, g)
        tgt_idx = ground_truth_source_idx(x, task)  # [B, T]

        acts = capture_layer_attn_and_kv(model, x)
        x_flat = x.reshape(-1)  # token identity per (sample, position), flattened to match k/v

        for layer in range(len(model.blocks)):
            addr_acc = addressing_accuracy(acts[layer]["attn"], tgt_idx)
            k_probe = ridge_probe_accuracy(acts[layer]["k"], x_flat)
            v_probe = ridge_probe_accuracy(acts[layer]["v"], x_flat)
            row = {"task": task, "layer": layer,
                   "addressing_acc": round(addr_acc, 4),
                   "k_content_probe_acc": round(k_probe, 4),
                   "v_content_probe_acc": round(v_probe, 4)}
            rows.append(row)
            print(row, flush=True)

    with open("kv_addressing_content_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "layer", "addressing_acc",
                                          "k_content_probe_acc", "v_content_probe_acc"])
        w.writeheader()
        w.writerows(rows)
    print("\nWrote kv_addressing_content_results.csv")


if __name__ == "__main__":
    main()
