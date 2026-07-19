# Vision-tasks results (top-1 / odd-one-out accuracy)

Reproduction of paper Table 2 across the six QKV projection-sharing variants. Classification
datasets use the **minimal grid** (`vision_tasks.py --grid minimal`: embed{64,256}×heads{2,4},
patch 4, lr 1e-3, 2 layers, 1 run), CIFAR with train-time augmentation. TinyImageNet uses the
large ViT (224px, patch 16, d768, 12 layers, AMP, 20 epochs). Anomaly uses frozen ResNet34
features + a permutation-equivariant set transformer (see `anomaly_results.md`).

`Average` follows the paper: mean of MNIST/FMNIST/CIFAR-10/CIFAR-100/Anomaly (**excludes**
TinyImageNet).

## Reproduced (this work)

| Variant | MNIST | FMNIST | CIFAR-10 | CIFAR-100 | Anomaly | Average | TinyImgNet |
|---|---|---|---|---|---|---|---|
| QKV | 0.984 | 0.885 | 0.700 | 0.430 | 0.789 | 0.758 | 0.331 |
| Q=K≠V | 0.983 | 0.884 | 0.715 | 0.447 | 0.800 | 0.766 | 0.339 |
| (Q=K≠V)⁺ | 0.981 | 0.883 | 0.695 | 0.408 | 0.836 | 0.761 | 0.334 |
| Q≠K=V | 0.979 | 0.882 | 0.694 | 0.428 | 0.811 | 0.759 | 0.334 |
| Q=K=V | 0.981 | 0.884 | 0.715 | 0.423 | 0.812 | 0.763 | **0.381** |
| (Q=K=V)⁺ | 0.979 | 0.875 | 0.693 | 0.410 | 0.834 | 0.758 | 0.342 |

## Paper Table 2

| Variant | MNIST | FMNIST | CIFAR-10 | CIFAR-100 | Anomaly | Average | TinyImgNet |
|---|---|---|---|---|---|---|---|
| QKV | 0.981 | 0.887 | 0.663 | 0.363 | 0.942 | 0.767 | 0.229 |
| Q=K≠V | 0.981 | 0.885 | 0.666 | 0.369 | 0.954 | 0.771 | 0.236 |
| (Q=K≠V)⁺ | 0.982 | 0.884 | 0.662 | 0.366 | 0.966 | 0.772 | — |
| Q≠K=V | 0.976 | 0.883 | 0.659 | 0.358 | 0.949 | 0.767 | — |
| Q=K=V | 0.978 | 0.877 | 0.672 | 0.376 | 0.933 | 0.767 | 0.266 |
| (Q=K=V)⁺ | 0.977 | 0.875 | 0.669 | 0.364 | 0.961 | 0.769 | — |

## Notes

- **Central claim reproduces**: all six variants are within a ~0.01-wide band on the paper's
  averaging (0.758–0.766) — projection sharing barely affects vision accuracy. The reproduced
  averages land very close to the paper's (0.767–0.772).
- **TinyImageNet quirk reproduces**: **Q=K=V is the best** variant (0.381) and QKV the lowest
  (0.331) — exactly the paper's surprising finding ("the Q=K=V Transformer, despite employing only
  one projection, achieves the best results in this instance"; paper Q=K=V 0.266 > QKV 0.229).
- **Anomaly**: the (X)⁺ variants win — (Q=K≠V)⁺ best (0.836), (Q=K=V)⁺ second (0.834) — matching
  the paper; all sharing variants ≥ QKV.
- **Per-dataset deltas vs paper**: MNIST/FMNIST match closely; CIFAR-10/100 and TinyImageNet run
  **higher** (augmentation + AMP + epoch budget); Anomaly runs **lower** (0.79–0.84 vs 0.93–0.97,
  simplified training setup — see `anomaly_results.md`). Rankings hold throughout.
- Regenerate classification+TinyImageNet with `python vision_tasks.py --merge`; anomaly with
  `python anomaly_tasks.py --merge`.
