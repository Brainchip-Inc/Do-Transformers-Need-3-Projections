---
license: mit
library_name: pytorch
tags:
  - transformer
  - attention
  - projection-sharing
  - qkv
  - kv-cache
  - language-modeling
  - gpt
---

# Do Transformers Need Three Projections? — LLM checkpoint (Q≠K=V, 300M)

A 300M-parameter GPT-style language model trained from scratch with the **Q≠K=V** attention
variant from the ICML 2026 paper [*"Do Transformers Need Three Projections? A Systematic Study
of QKV Variants"*](https://huggingface.co/BrainChip-AI/do-transformers-need-3-projections)
(Kayyam, Madan Gopal, Lewis; BrainChip Inc.) — separate query projection, **shared key/value**
(`V = K`, no value weights at all). Only **K** needs to be cached during autoregressive
generation, halving KV-cache memory versus standard attention.

This card covers the **language-modeling** result. The paper's original synthetic-task and
vision-classification checkpoints are on a
[separate card](https://huggingface.co/BrainChip-AI/do-transformers-need-3-projections).

## ⚠️ Scope of this checkpoint — please read

This is **not** the paper's official SlimPajama LM run described in the
[code repository](https://github.com/Brainchip-Inc/Do-Transformers-Need-3-Projections) README
(that one reports "~3% perplexity degradation" vs. a matched QKV baseline). This checkpoint is a
**later, independent reproduction** on a different dataset, with only the Q≠K=V variant trained —
**there is no matched QKV/K/KV baseline run under the same conditions to compare against.** Do
not read the numbers below as reproducing or contradicting the paper's headline claim.

- **Dataset**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  (`sample-10BT`), not SlimPajama — SlimPajama was no longer resolvable on the HF Hub at
  training time.
- **Training script**: an updated copy of
  [`transformer_KV_1_300_M.py`](https://github.com/Brainchip-Inc/Do-Transformers-Need-3-Projections/blob/main/transformer_KV_1_300_M.py)
  (class `GPT_QKV_KEqualsV`) with a padding-loss masking fix (short documents were previously
  scored on their padding). The model architecture is identical to the file linked above; only
  the training loop's loss masking changed. The fixed training script itself was not preserved
  outside the training instance.

## Results

Single GPU, 10B tokens (FineWeb-Edu `sample-10BT`), 33,908 steps, ~45 hours.

| Metric | Value |
|---|---|
| Final validation loss | 3.058 |
| Final validation perplexity | **21.28** |
| Bits per character | 4.41 |

Validation PPL every 500 steps follows a standard power-law decay (281.7 → 21.28 across the run).

**Generation quality caveat**: sample generations use `temperature=0.8, top_k=50` with no seed
and no repetition penalty — outputs vary run-to-run and can fall into repetition loops. Treat
generated text as illustrative only; validation perplexity is the reliable metric here.

## What's included

| File | Description |
|---|---|
| `checkpoints/qkv_keqv_300m_fineweb_edu.pt` | Final model weights only (no optimizer state), `{step, epoch, tokens_seen, model_state_dict, val_metrics, model_config, train_config}` |
| `checkpoints/model_config.yaml` | Architecture hyperparameters |
| `checkpoints/train_config.yaml` | Optimizer / schedule hyperparameters |

## Model configuration

```yaml
n_layer: 20
n_embd: 1024
n_head: 16
n_inner: 4096
n_positions: 2048
vocab_size: 50304
tie_word_embeddings: true
```

284.5M parameters (checkpoint metadata rounds this to "300M" after the paper's naming
convention for this scale tier). Tokenizer: GPT-2 (`AutoTokenizer.from_pretrained("gpt2")`).

## Usage

The model class lives in the paper's code repository (not bundled in this HF repo):

```bash
git clone https://github.com/Brainchip-Inc/Do-Transformers-Need-3-Projections
pip install torch huggingface_hub
```

```python
import torch
from huggingface_hub import hf_hub_download
from transformer_KV_1_300_M import ModelConfig, GPT_QKV_KEqualsV

path = hf_hub_download(
    "BrainChip-AI/qkv-fineweb-300m",
    "checkpoints/qkv_keqv_300m_fineweb_edu.pt",
)
ckpt = torch.load(path, map_location="cpu", weights_only=False)

model = GPT_QKV_KEqualsV(ModelConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(ckpt["val_metrics"])  # {'loss': 3.058, 'perplexity': 21.28, 'bpc': 4.41}
```

## Citation

```bibtex
@inproceedings{kayyam2026qkv,
  title={Do Transformers Need Three Projections? A Systematic Study of {QKV} Variants},
  author={Kayyam, Ali and Madan Gopal, Anusha and Lewis, M Anthony},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026},
  series={PMLR},
  volume={306}
}
```

## License

MIT (see the [code repository](https://github.com/Brainchip-Inc/Do-Transformers-Need-3-Projections)).
