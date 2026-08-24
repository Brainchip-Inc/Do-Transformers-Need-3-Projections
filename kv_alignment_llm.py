"""
LLM counterpart of kv_alignment.py: fit a linear map / orthogonal Procrustes
rotation from the 300M FineWeb-Edu QKV teacher's K activations to its V
activations (per layer, on a held-out TRAIN-split batch), then evaluate
zero-shot (no retraining) validation perplexity with that learned correction
in place of keep_k's literal copy -- same numerical-stability approach as
kv_content_probe_llm.py (standardize, ridge scaled to the data's own energy),
since this model's K activations are severely rank-deficient in raw units.

Usage:
    conda run -n kv python kv_alignment_llm.py --device cuda:0
"""

import types
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from transformer_KQV_300_M import evaluate
from distillation_llm import load_teacher, TEACHER_CKPT
from transformer_KQV_300M_fineweb import get_dataloader, enable_sdpa_attention

RIDGE_FRAC = 0.1
KEEP_K_ZERO_SHOT_PPL = 13345.19
KEEP_V_ZERO_SHOT_PPL = 7347.14


def capture_kv_activations(model, input_ids, n_embd):
    acts = {i: {} for i in range(len(model.h))}
    handles = []
    for i, block in enumerate(model.h):
        def mk(i):
            def hook(_mod, _inp, out):
                q, k, v = out.split(n_embd, dim=2)
                acts[i]["k"] = k.detach().reshape(-1, n_embd)
                acts[i]["v"] = v.detach().reshape(-1, n_embd)
            return hook
        handles.append(block.attn.c_attn.register_forward_hook(mk(i)))
    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()
    return acts


RIDGE_CANDIDATES = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def fit_linear_map(K, V_, ridge_frac=RIDGE_FRAC):
    mean = K.mean(0, keepdim=True)
    std = K.std(0, keepdim=True).clamp_min(1e-6)
    Kn = (K - mean) / std
    ones = torch.ones(Kn.size(0), 1, device=K.device, dtype=K.dtype)
    K1 = torch.cat([Kn, ones], dim=1)
    XtX = K1.T @ K1
    ridge = ridge_frac * XtX.diagonal().mean()
    A = XtX + ridge * torch.eye(K1.size(1), device=K.device, dtype=K.dtype)
    Wb = torch.linalg.solve(A, K1.T @ V_)
    W, b = Wb[:-1], Wb[-1]
    W_raw = W / std.T
    b_raw = (b - (mean / std) @ W).squeeze(0)
    return W_raw, b_raw


def fit_procrustes(K, V_):
    mean_k, std_k = K.mean(0, keepdim=True), K.std(0, keepdim=True).clamp_min(1e-6)
    mean_v, std_v = V_.mean(0, keepdim=True), V_.std(0, keepdim=True).clamp_min(1e-6)
    Kn = (K - mean_k) / std_k
    Vn = (V_ - mean_v) / std_v
    M = Kn.T @ Vn
    U, _, Wt = torch.linalg.svd(M)
    R = U @ Wt
    return R, mean_k, std_k, mean_v, std_v


def aligned_forward_factory(kind, params, n_embd):
    if kind == "linear":
        W_raw, b_raw = params

        def transform(k):
            return k @ W_raw + b_raw
    else:
        R, mean_k, std_k, mean_v, std_v = params

        def transform(k):
            kn = (k - mean_k) / std_k
            return (kn @ R) * std_v + mean_v

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(n_embd, dim=2)
        kv = transform(k.reshape(-1, n_embd)).reshape(B, T, n_embd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv_h = kv.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, kv_h, kv_h, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out
    return forward


def apply_alignment(model, acts, n_embd, kind, ridge_frac=RIDGE_FRAC):
    """acts must come from capture_kv_activations on the model's UNMODIFIED
    forward -- calling this repeatedly on the same model with freshly
    recaptured activations would contaminate the fit with the previous call's
    already-patched attention output. Capture once, reuse across calls."""
    for i, block in enumerate(model.h):
        K, V_ = acts[i]["k"], acts[i]["v"]
        if kind == "linear":
            params = fit_linear_map(K, V_, ridge_frac=ridge_frac)
        else:
            params = fit_procrustes(K, V_)
        block.attn.forward = types.MethodType(aligned_forward_factory(kind, params, n_embd), block.attn)


def eval_perplexity_on_batches(model, batches, device):
    """Same computation as transformer_KQV_300_M.evaluate, but over a fixed,
    pre-materialized list of (input_ids, labels) batches instead of a
    DataLoader + max_batches -- needed because re-enumerating a DataLoader
    restarts it from the beginning each time, which would make a "skip the
    fit batch, then take the next N" select-set impossible to reuse across
    the ridge sweep."""
    import math
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for input_ids, labels in batches:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)
            n = input_ids.numel()
            total_loss += loss.item() * n
            total_tokens += n
    if was_training:
        model.train()
    avg_loss = total_loss / total_tokens
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}


def select_ridge_frac_by_ppl(model, acts, n_embd, select_batches, device,
                              candidates=RIDGE_CANDIDATES):
    """Picks a single global ridge_frac (same for every layer) by actual
    downstream perplexity on a SEPARATE select-set of held-out training
    documents -- never the real reported validation set.

    This replaces an earlier reconstruction-MSE-based selector (fit vs. a
    disjoint CV batch of K/V activations), which turned out to be measuring
    the wrong thing: checked directly, cross-batch reconstruction error
    ||A k - v||^2 monotonically favors LOWER ridge at every layer, yet
    ridge_frac=0.1 is exactly the setting that blew val PPL up to 38747 (vs.
    13345 for plain keep_k). A map that reconstructs V accurately in MSE
    terms is not the same as one that's useful to the frozen downstream
    computation -- the same disconnect the paper's content probe already
    found between K's linear decodability and its downstream uselessness.
    Selecting directly by (proxy) downstream perplexity avoids this trap."""
    best_frac, best_ppl = candidates[0], float("inf")
    for rf in candidates:
        for i, block in enumerate(model.h):
            K, V_ = acts[i]["k"], acts[i]["v"]
            params = fit_linear_map(K, V_, ridge_frac=rf)
            block.attn.forward = types.MethodType(
                aligned_forward_factory("linear", params, n_embd), block.attn)
        ppl = eval_perplexity_on_batches(model, select_batches, device)["perplexity"]
        print(f"  ridge_frac={rf}: select-set val_ppl={ppl:.2f}", flush=True)
        if ppl < best_ppl:
            best_ppl, best_frac = ppl, rf
    return best_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--teacher-ckpt", type=str, default=TEACHER_CKPT)
    ap.add_argument("--train-data", type=str, default="./fineweb_edu_train")
    ap.add_argument("--val-data", type=str, default="./fineweb_edu_validation")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--select-batches", type=int, default=8,
                     help="held-out train batches used only for ridge selection")
    ap.add_argument("--eval-batches", type=int, default=64)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    teacher, model_config, _ = load_teacher(args.teacher_ckpt, device)
    teacher.eval()
    enable_sdpa_attention(teacher)
    n_embd = model_config.n_embd

    fit_loader, _ = get_dataloader(
        args.train_data, tokenizer, args.batch_size, model_config.n_positions,
        num_workers=2, prefetch_factor=2, shuffle=False)
    fit_iter = iter(fit_loader)
    fit_ids = next(fit_iter)["input_ids"].to(device)
    # the NEXT `select_batches` batches (disjoint documents from fit_ids, and
    # from the real held-out validation set) -- used only to pick the linear
    # map's ridge strength by actual downstream perplexity, never the real
    # reported validation metric (see select_ridge_frac_by_ppl's docstring)
    select_batches = []
    for _ in range(args.select_batches):
        b = next(fit_iter)
        select_batches.append((b["input_ids"].to(device), b["labels"].to(device)))

    val_loader, _ = get_dataloader(
        args.val_data, tokenizer, args.batch_size, model_config.n_positions,
        num_workers=2, prefetch_factor=2, shuffle=False)

    # capture once, from the model's unmodified forward -- reused throughout
    acts = capture_kv_activations(teacher, fit_ids, n_embd)

    print("Selecting linear map's ridge_frac by select-set perplexity...", flush=True)
    ridge_frac = select_ridge_frac_by_ppl(teacher, acts, n_embd, select_batches, device)
    print(f"-> selected ridge_frac={ridge_frac}\n", flush=True)

    results = {}
    for kind, kw in [("linear", {"ridge_frac": ridge_frac}), ("procrustes", {})]:
        apply_alignment(teacher, acts, n_embd, kind, **kw)
        metrics = evaluate(teacher, val_loader, args.eval_batches, device)
        results[kind] = metrics["perplexity"]
        print(f"keep_k_{kind}: val_ppl={metrics['perplexity']:.2f}", flush=True)
        torch.cuda.empty_cache()

    print(f"\nkeep_k              PPL = {KEEP_K_ZERO_SHOT_PPL:.2f}")
    print(f"keep_k_aligned       PPL = {results['linear']:.2f}  (ridge_frac={ridge_frac})")
    print(f"keep_k_procrustes    PPL = {results['procrustes']:.2f}")
    print(f"keep_v               PPL = {KEEP_V_ZERO_SHOT_PPL:.2f}")


if __name__ == "__main__":
    main()
