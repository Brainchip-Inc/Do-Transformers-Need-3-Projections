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

# Do Transformers Need Three Projections? — Q≠K=V checkpoints

Reference checkpoints for the **Q≠K=V** attention variant (separate query, **shared key and
value**) from the ICML 2026 paper *"Do Transformers Need Three Projections? A Systematic Study
of QKV Variants"* (Kayyam, Madan Gopal, Lewis; BrainChip Inc.).

Q≠K=V is the paper's **headline variant**: it removes the value projection and reuses the key as
the value, so only **K** needs to be cached during autoregressive generation — a **50% KV-cache
reduction** — while keeping attention asymmetric (unlike the symmetric Q=K variants). See the
[paper / code repository](https://github.com/Brainchip-Inc/Do-Transformers-Need-3-Projections)
for the full study across all six projection-sharing variants.

## What's included

Nine trained-from-scratch Q≠K=V checkpoints (PyTorch `state_dict` + rebuild config + test accuracy):

**Synthetic sequence tasks** (single Transformer encoder, token accuracy):

| Checkpoint | Task | Accuracy |
|---|---|---|
| `checkpoints/synthetic_reverse_qkv_kv.pt` | REVERSE | 1.000 |
| `checkpoints/synthetic_sort_qkv_kv.pt` | SORT | 0.996 |
| `checkpoints/synthetic_sub_qkv_kv.pt` | SUB | 1.000 |
| `checkpoints/synthetic_swap_qkv_kv.pt` | SWAP | 1.000 |
| `checkpoints/synthetic_copy_qkv_kv.pt` | COPY | 1.000 |

**Vision classification** (ViT trained from scratch, top-1 accuracy):

| Checkpoint | Dataset | Accuracy |
|---|---|---|
| `checkpoints/vision_mnist_qkv_kv.pt` | MNIST | 0.978 |
| `checkpoints/vision_fmnist_qkv_kv.pt` | FashionMNIST | 0.882 |
| `checkpoints/vision_cifar10_qkv_kv.pt` | CIFAR-10 | 0.698 |
| `checkpoints/vision_cifar100_qkv_kv.pt` | CIFAR-100 | 0.445 |

Each checkpoint is a dict: `{task, variant, model, config, test_accuracy, model_state_dict}`.

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

# --- synthetic task model ---
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
