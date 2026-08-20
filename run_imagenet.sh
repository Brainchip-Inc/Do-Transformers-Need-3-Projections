#!/bin/bash
# Launch the ImageNet-1k ViT-S/16 comparison: one condition per GPU, in parallel.
# workers=6 per job (24 cores / 4 concurrent jobs) to avoid oversubscribing the CPU
# during JPEG decode. batch-size=128 (256 OOMs at 11 GB with this recipe's AMP memory
# footprint; 128 leaves ~3 GB headroom, peak measured at 7.4 GB).
set -e
cd "$(dirname "$0")"

conda run --no-capture-output -n torch_env python -u imagenet_experiment.py --variant "QKV" \
    --device cuda:0 --batch-size 128 --workers 6 --epochs 100 \
    > imagenet_gpu0_qkv.log 2>&1 &

conda run --no-capture-output -n torch_env python -u imagenet_experiment.py --variant "Q!=K=V" \
    --device cuda:1 --batch-size 128 --workers 6 --epochs 100 \
    > imagenet_gpu1_qkv_eq_v.log 2>&1 &

conda run --no-capture-output -n torch_env python -u imagenet_experiment.py --variant "(Q=K=V)+" \
    --device cuda:2 --batch-size 128 --workers 6 --epochs 100 \
    > imagenet_gpu2_qeqkeqv.log 2>&1 &

conda run --no-capture-output -n torch_env python -u imagenet_experiment.py --variant "QVV(3)" \
    --device cuda:3 --batch-size 128 --workers 6 --epochs 100 \
    > imagenet_gpu3_qvv3.log 2>&1 &

wait
