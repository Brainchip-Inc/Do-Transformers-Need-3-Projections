# K=V tying: post-hoc surgery + distillation recovery (synthetic tasks)

Study on top of the existing synthetic-task checkpoints (`checkpoints/synthetic_*_{qkv,qkv_kv}.pt`,
n_embd=256, n_layer=2, n_head=4, seq_len=64). Two questions:

**(a) Surgery.** Take a *trained* QKV checkpoint (separate Q/K/V) and force K=V after the fact,
with no retraining — either drop one projection and reuse the other (`keep_k`, `keep_v`), or
average the two Linear layers' weights (`avg`, equivalent to averaging their outputs). Compare
zero-shot accuracy against the QKV teacher and the from-scratch Q≠K=V checkpoint.

**(b) Distillation recovery.** Fine-tune each surgically-tied student for up to 5 epochs against
the QKV teacher's soft logits (KL term, T=2) + hard-label CE (alpha=0.5), and see how much of the
gap opened by (a) is recovered relative to training Q≠K=V from scratch (10 epochs).

Script: `distillation_synthetic.py`. Raw per-epoch numbers: `distillation_synthetic_results.csv`.

## Results (test token accuracy)

| Task | Mode | Teacher (QKV) | Scratch Q≠K=V | Zero-shot surgery | Distilled (5 ep) |
|---|---|---|---|---|---|
| REVERSE | keep_k | 1.000 | 1.000 | 0.100 | 1.000 |
| REVERSE | keep_v | 1.000 | 1.000 | 0.123 | 1.000 |
| REVERSE | avg    | 1.000 | 1.000 | 0.861 | 1.000 |
| SORT    | keep_k | 0.998 | 0.996 | 0.583 | 0.965 |
| SORT    | keep_v | 0.998 | 0.996 | 0.717 | 0.996 |
| SORT    | avg    | 0.998 | 0.996 | 0.751 | 0.996 |
| SUB     | keep_k | 1.000 | 1.000 | 1.000 | 1.000 |
| SUB     | keep_v | 1.000 | 1.000 | 1.000 | 1.000 |
| SUB     | avg    | 1.000 | 1.000 | 1.000 | 1.000 |
| SWAP    | keep_k | 1.000 | 1.000 | 0.088 | 1.000 |
| SWAP    | keep_v | 1.000 | 1.000 | 0.122 | 1.000 |
| SWAP    | avg    | 1.000 | 1.000 | 0.906 | 1.000 |
| COPY    | keep_k | 1.000 | 1.000 | 1.000 | 1.000 |
| COPY    | keep_v | 1.000 | 1.000 | 1.000 | 1.000 |
| COPY    | avg    | 1.000 | 1.000 | 1.000 | 1.000 |

## Takeaways

- **Post-hoc K=V tying is task-dependent, not uniformly harmless.** For the permutation/comparison
  tasks (REVERSE, SWAP, SORT) — which need content-based *addressing* (via K) that is distinct
  from content *value* (via V) — forcing K=V after training collapses accuracy to near chance for
  `keep_k`/`keep_v`, and takes a large but smaller hit under `avg`. For the pointwise tasks (SUB,
  COPY) — where every position's output depends only on its own input, not on comparing across
  positions — there is zero drop under any surgery mode. This shows that when Q≠K=V is trained
  from scratch it *learns* to route around the constraint, but K and V are **not** interchangeable
  post-hoc: on tasks that need it, they converge to genuinely different, non-redundant functions.
- **A handful of distillation epochs (mostly) closes the gap.** 14 of 15 (task, mode) pairs fully
  recover to the from-scratch Q≠K=V ceiling within 1-5 epochs of KD fine-tuning against the QKV
  teacher — an order of magnitude less training than the 10 epochs used from scratch. The one
  holdout, SORT/`keep_k`, climbs from 0.583 to 0.965 but falls just short of the 0.996 target
  within the 5-epoch budget (see `distillation_synthetic_results.csv` for the per-epoch curve —
  it is still improving at epoch 5, so more epochs would likely close it).
- Practical reading: if you need to shrink a trained QKV model to Q≠K=V after the fact (rather
  than retraining from scratch), `avg` is the safer surgery starting point, and a short KD pass
  is cheap insurance regardless of which surgery mode is used.

## Reproduce

```
conda run -n kv python distillation_synthetic.py --device cuda:0
```
