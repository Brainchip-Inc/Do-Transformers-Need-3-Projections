---
license: mit
library_name: pytorch
tags:
  - transformer
  - attention
  - projection-sharing
  - qkv
  - kv-cache
  - image-classification
  - sequence-modeling
---

# Do Transformers Need Three Projections? — checkpoints

Reference checkpoints from the ICML 2026 paper *"Do Transformers Need Three Projections? A
Systematic Study of QKV Variants"* (Kayyam, Madan Gopal, Lewis; BrainChip Inc.). This release
pairs the standard **QKV baseline** with the paper's headline **Q≠K=V** variant so the two can be
compared directly on matched architectures.

- **QKV** — standard attention with separate query, key, and value projections (baseline).
- **Q≠K=V** — separate query, **shared key and value** (`V = K`). Only **K** is cached during
  autoregressive generation → **50% KV-cache reduction**, while attention stays asymmetric
  (unlike the symmetric `Q=K` variants).

The point of the pairing: Q≠K=V matches the QKV baseline in accuracy at half the KV cache. The
full study (all six projection-sharing variants) is in the
[code repository](https://github.com/Brainchip-Inc/Do-Transformers-Need-3-Projections).

## What's included

18 trained-from-scratch checkpoints — QKV (`*_qkv.pt`) and Q≠K=V (`*_qkv_kv.pt`) for each task.
Each is a dict: `{task, variant, model, config, test_accuracy, model_state_dict}`.

**Synthetic sequence tasks** (single Transformer encoder, token accuracy):

| Task | QKV (`*_qkv.pt`) | Q≠K=V (`*_qkv_kv.pt`) |
|---|---|---|
| REVERSE | 1.000 | 1.000 |
| SORT | 0.998 | 0.996 |
| SUB | 1.000 | 1.000 |
| SWAP | 1.000 | 1.000 |
| COPY | 1.000 | 1.000 |

Files: `checkpoints/synthetic_<task>_qkv.pt` and `checkpoints/synthetic_<task>_qkv_kv.pt`.

**Vision classification** (ViT trained from scratch, top-1 accuracy):

| Dataset | QKV (`*_qkv.pt`) | Q≠K=V (`*_qkv_kv.pt`) |
|---|---|---|
| MNIST | 0.981 | 0.978 |
| FashionMNIST | 0.887 | 0.882 |
| CIFAR-10 | 0.696 | 0.698 |
| CIFAR-100 | 0.437 | 0.445 |

Files: `checkpoints/vision_<dataset>_qkv.pt` and `checkpoints/vision_<dataset>_qkv_kv.pt`.

Across every task the two variants are within ~0.005 of each other — Q≠K=V matches the QKV
baseline while halving the KV cache.

## Usage

The model definitions live in the GitHub repository. Clone it, then load a checkpoint:

```bash
git clone https://github.com/Brainchip-Inc/Do-Transformers-Need-3-Projections
pip install torch torchvision huggingface_hub
```

```python
import torch
from huggingface_hub import hf_hub_download

REPO = "BrainChip-AI/do-transformers-need-3-projections"

# --- synthetic task model (swap _qkv <-> _qkv_kv for baseline vs Q!=K=V) ---
from synthetic_tasks import Encoder, ModelCfg
path = hf_hub_download(REPO, "checkpoints/synthetic_reverse_qkv_kv.pt")
ckpt = torch.load(path, map_location="cpu")
model = Encoder(ModelCfg(**ckpt["config"]))
model.load_state_dict(ckpt["model_state_dict"]); model.eval()

# --- vision (ViT) model ---
from vision_tasks import ViT, ViTConfig
path = hf_hub_download(REPO, "checkpoints/vision_cifar10_qkv_kv.pt")
ckpt = torch.load(path, map_location="cpu")
vit = ViT(ViTConfig(**ckpt["config"]))
vit.load_state_dict(ckpt["model_state_dict"]); vit.eval()
```

## Training

- **Synthetic**: single encoder, embedding dim 256, 2 layers, 4 heads, sequence length 64,
  one-hot inputs, Adam (lr 1e-3), cross-entropy, gradient clip 5, 10 epochs.
- **Vision**: ViT, patch 4, embedding dim 256, 2 layers, 4 heads, Adam (lr 1e-3) with MultiStepLR,
  cross-entropy; 20 epochs (MNIST/FMNIST), 40 (CIFAR-10), 50 (CIFAR-100); CIFAR uses random-crop +
  horizontal-flip augmentation. Trained from scratch on a single NVIDIA GTX 1080 Ti.

These are reproduction checkpoints; absolute numbers may differ slightly from the paper's tables
(training-budget and augmentation details). See the repository for the full methodology and all
six attention variants.

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

MIT (see the code repository).
