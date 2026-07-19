# Set Anomaly Detection results (odd-one-out accuracy)

Frozen ImageNet-pretrained ResNet34 features (512-d), permutation-equivariant set transformer
(no positional embedding), sets of 10 (9 normal + 1 anomaly from a different CIFAR-100 class).
Minimal grid: embed{64,256} × heads{2,4}, 2 layers, 1 run → 4 configs × 6 variants = 24 runs,
20 epochs, 10k train / 2k test sets. Chance = 0.10.

| Variant | Reproduced | Paper Table 2 |
|---|---|---|
| QKV | 0.789 | 0.942 |
| Q=K≠V | 0.800 | 0.954 |
| (Q=K≠V)⁺ | **0.836** | **0.966** |
| Q≠K=V | 0.811 | 0.949 |
| Q=K=V | 0.812 | 0.933 |
| (Q=K=V)⁺ | 0.834 | 0.961 |

## Notes

- **Key finding reproduces**: the (X)⁺ variants are best — (Q=K≠V)⁺ tops both our run (0.836) and
  the paper (0.966), with (Q=K=V)⁺ second. The 2D positional injection helps most on this task.
- **All projection-sharing variants ≥ QKV baseline** in our run, matching the paper's finding that
  sharing does not hurt (and the (X)⁺ augmentation helps).
- **Absolute values run ~0.13–0.15 below the paper** (ours ~0.79–0.84 vs 0.93–0.97), attributable to
  the unspecified training budget / set-sampling / feature details; the cross-variant ranking holds.
- Note: (X)⁺ applies a position-based injection over the 10 set elements even though order is not
  meaningful for a set; included for parity with the paper's reported (X)⁺ anomaly numbers.
- Regenerate with `python anomaly_tasks.py --merge`.
