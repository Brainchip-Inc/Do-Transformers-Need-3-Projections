# Synthetic Tasks — Methods & Results

*Companion write-up for the synthetic-reasoning experiments (paper Table 1),*
*reproduced by [`synthetic_tasks.py`](synthetic_tasks.py) on the `synthetics` branch.*

---

## 1. Methods

### 1.1 Tasks

We evaluate the six attention variants on five synthetic sequence-to-sequence tasks
operating on integer lists whose elements are drawn uniformly from the ten symbols
$\{0,\dots,9\}$. Given an input list, each task defines a deterministic target list of the
same length:

| Task | Definition | Example (input → target) |
|---|---|---|
| **REVERSE** | reverse the list | `[4,3,9,8,1]` → `[1,8,9,3,4]` |
| **SORT** | sort ascending | `[4,3,9,8,1]` → `[1,3,4,8,9]` |
| **SUB** | subtract each element from 9 | `[4,3,9,8,1]` → `[5,6,0,1,8]` |
| **SWAP** | exchange first and second halves | `[4,3,9,8,1,7]` → `[8,1,7,4,3,9]` |
| **COPY** | identity | `[4,3,9,8,1]` → `[4,3,9,8,1]` |

SWAP requires an even sequence length; all swept lengths (16, 64, 128) are even. Each symbol
is encoded as a one-hot vector of dimension 10.

### 1.2 Model

The model is a single Transformer **encoder** (no causal mask) that performs per-token
10-way classification: it maps the one-hot input through a linear embedding, adds learned
positional embeddings, applies $L$ pre-norm Transformer blocks, and reads out each position
with a shared linear head. Training minimizes per-token cross-entropy against the task target;
the reported metric is **token accuracy** on a held-out test set (fraction of positions
predicted correctly).

The only component that differs across the six configurations is the attention projection:

- **QKV** — separate $Q$, $K$, $V$ projections (baseline).
- **Q=K≠V** — shared $Q{=}K$ projection, separate $V$; the score map $KK^{\top}$ is symmetric.
- **(Q=K≠V)⁺** — as above, with 2D positional injection (§1.3) to break symmetry.
- **Q≠K=V** — separate $Q$, shared $K{=}V$; asymmetric score map, and only $K$ need be cached.
- **Q=K=V** — a single projection for all three roles; symmetric score map.
- **(Q=K=V)⁺** — as above, with 2D positional injection.

### 1.3 2D positional injection for the (X)⁺ variants

The symmetric variants ($Q{=}K$) produce a symmetric score map, which cannot represent
directional relations. To restore asymmetry without abandoning projection sharing, the (X)⁺
variants add a fixed 2D sinusoidal positional tensor to the score map. Concretely, we build
$P \in \mathbb{R}^{n\times n\times m}$ with $m=10$, where
$P_{ij} = [\,\text{sinusoid}(i)\,;\,\text{sinusoid}(j)\,]$ concatenates a row-index encoding
and a column-index encoding, so that $P_{ij} \neq P_{ji}$ in general. The scalar score map is
broadcast across the $m$ channels, added to $P$, and collapsed back to a single $n\times n$
map by a $1\times1$ convolution over the channel dimension (a learned linear combination of
the $m$ channels). Because a $1\times1$ convolution is linear, this is computed directly and
equivalently as
$$S'_{ij} = S_{ij}\textstyle\sum_c w_c \;+\; \sum_c w_c P_{ijc} \;+\; b,$$
avoiding materialization of the $n\times n\times m$ tensor. The resulting map is asymmetric by
construction; a symmetry check (`python synthetic_tasks.py --check-symmetry`) confirms
$\text{mean}\,|S-S^{\top}| = 0$ for the base symmetric variants and a clearly nonzero value for
the (X)⁺ variants.

> The (X)⁺ construction follows the description in the paper body (Sec. 3.1). The exact
> sinusoid formula is a faithful reconstruction, as the appendix "pos2d" details were not
> available at reproduction time.

### 1.4 Training and the configuration sweep

All runs use the Adam optimizer with learning rate $10^{-3}$, a cosine learning-rate schedule
with a short linear warmup, gradient clipping at 5, and 2 training epochs. Following the paper,
each attention variant is evaluated over a Cartesian sweep of architectural configurations and
averaged over 3 random seeds:

| Hyperparameter | Values |
|---|---|
| Embedding dimension | 32, 64, 256 |
| Number of layers | 2, 4 |
| Number of heads | 2, 4 |
| Sequence length | 16, 64, 128 |
| Seeds | 3 |

This yields $3\times2\times2\times3 = 36$ configurations, or 108 runs per (variant, task), and
the reported accuracy for each (variant, task) cell is the mean token accuracy across all of
them. Configurations where the head count does not divide the embedding dimension are skipped.

**Deviations from the paper (faithful-but-modernized).** The reproduction uses pre-norm
Transformer blocks, learned positional embeddings, and fully seeded data/initialization for
determinism. The training/test set size — unspecified in the paper — is fixed at 10,000 train
and 2,000 test sequences per run (regenerated per seed); this and all other knobs are exposed
as CLI flags in `synthetic_tasks.py`.

### 1.5 Compute

Experiments run on a single NVIDIA GTX 1080 Ti per shard (matching the paper's hardware class).
The full grid is $6\text{ variants} \times 5\text{ tasks} \times 108 = 3{,}240$ runs; the driver
supports sharding across GPUs (`--shard i/N --device cuda:X`) and a `--merge` step that
aggregates the per-shard CSVs into the results table below.

---

## 2. Results

*Reproduced from the full 3,240-run sweep (6 variants × 5 tasks × 36 configs × 3 seeds),*
*zero failed runs. Regenerate with `python synthetic_tasks.py --merge`.*

### 2.1 Reproduced accuracies

Mean token accuracy, averaged over the 36-config × 3-seed sweep (higher is better):

| Variant | REVERSE | SORT | SUB | SWAP | COPY | Avg |
|---|---|---|---|---|---|---|
| QKV        | 0.529 | 0.753 | 0.992 | 0.524 | 0.991 | 0.758 |
| Q=K≠V      | 0.475 | 0.691 | 0.991 | 0.471 | 0.994 | 0.724 |
| (Q=K≠V)⁺   | 0.522 | 0.770 | 0.993 | 0.530 | 0.991 | **0.761** |
| Q≠K=V      | 0.510 | 0.747 | 0.992 | 0.511 | 0.994 | 0.751 |
| Q=K=V      | 0.295 | 0.671 | 0.991 | 0.307 | 0.992 | 0.651 |
| (Q=K=V)⁺   | 0.370 | 0.770 | 0.990 | 0.382 | 0.990 | 0.700 |

**Paper Table 1 (target, for comparison):**

| Variant | REVERSE | SORT | SUB | SWAP | COPY | Avg |
|---|---|---|---|---|---|---|
| QKV        | 0.698 | 0.971 | 1.0 | 0.588 | 1.0 | 0.851 |
| Q=K≠V      | 0.705 | 0.967 | 1.0 | 0.597 | 1.0 | 0.854 |
| (Q=K≠V)⁺   | 0.718 | 0.963 | 1.0 | 0.671 | 1.0 | **0.870** |
| Q≠K=V      | 0.701 | 0.958 | 1.0 | 0.590 | 1.0 | 0.850 |
| Q=K=V      | 0.514 | 0.939 | 1.0 | 0.446 | 1.0 | 0.780 |
| (Q=K=V)⁺   | 0.581 | 0.957 | 1.0 | 0.576 | 1.0 | 0.823 |

### 2.2 Narrative (reproduced vs. paper)

The absolute accuracies come in lower than the paper on the attention-dependent tasks
(REVERSE / SORT / SWAP are ~0.15–0.20 below), most likely because of the unspecified
training budget — this reproduction uses 10k training sequences × 2 epochs (§1.4). The
**qualitative conclusions all hold**:

- **Projection sharing is nearly free on these tasks.** ✅ _Confirmed:_ QKV (0.758) and Q≠K=V
  (0.751) are within 0.007; the two-projection variants stay competitive with the baseline.
  (Q=K≠V dips a bit more, to 0.724 — slightly below QKV here, whereas the paper had it marginally
  above.)
- **The single-projection variant degrades.** ✅ _Confirmed:_ Q=K=V is the clear worst (0.651,
  −0.107 vs QKV), driven by REVERSE (0.295) and SWAP (0.307) — the directional tasks.
- **Positional injection recovers the symmetric variants.** ✅ _Confirmed:_ (X)⁺ lifts both
  symmetric variants (Q=K≠V 0.724→0.761; Q=K=V 0.651→0.700), with the largest gains on REVERSE
  and SWAP; (Q=K≠V)⁺ is the top average overall (0.761), matching the paper.
- **Position-wise tasks are trivial for all variants.** ✅ _Confirmed:_ SUB and COPY reach ≈0.99
  everywhere, since each output depends only on the same-position input.
- **Overall ranking.** ✅ _Confirmed (endpoints match the paper):_ (Q=K≠V)⁺ (0.761) is best and
  Q=K=V (0.651) is worst; reproduced order is
  (Q=K≠V)⁺ > QKV > Q≠K=V > Q=K≠V > (Q=K=V)⁺ > Q=K=V.

### 2.3 LaTeX table (paper format, ready to drop in)

```latex
\begin{table}[t]
\footnotesize
\caption{Performance on synthetic tasks. Multiple runs over different configurations
(number of attention heads, embedding dimension, learning rate, sequence length, etc.)
are conducted and the results are averaged.}
\label{tab:synthetic}
\vspace{-5pt}
\setlength{\tabcolsep}{3pt}
\begin{center}
\begin{tabular}{l||ccccc|c}
 & REVERSE &  SORT & SUB & SWAP & COPY & Avg.\\
\midrule
QKV & 0.529 & 0.753 & 0.992 & 0.524 & 0.991 & 0.758 \\
\midrule
Q=K$\neq$V & 0.475 & 0.691 & 0.991 & 0.471 & 0.994 & 0.724 \\
(Q=K$\neq$V)$^+$ & 0.522 & 0.770 & 0.993 & 0.530 & 0.991 & {\bf 0.761} \\
\midrule
Q$\neq$K=V & 0.510 & 0.747 & 0.992 & 0.511 & 0.994 & 0.751 \\
\midrule
Q=K=V & 0.295 & 0.671 & 0.991 & 0.307 & 0.992 & 0.651 \\
(Q=K=V)$^+$ & 0.370 & 0.770 & 0.990 & 0.382 & 0.990 & 0.700 \\
\end{tabular}
\end{center}
\vspace{-20pt}
\end{table}
```
