#!/bin/bash
# Sequential post-hoc K=V surgery + distillation recovery for the 300M FineWeb-Edu
# LLM, one mode at a time, using both A30s via DDP (matches how the QKV baseline
# itself was trained). 500M-token distillation budget per mode (~5% of the 10B-token
# pretraining budget, matching the ratio used in the vision/synthetic tables).
set -e
cd "$(dirname "$0")"

for mode in keep_k keep_v avg; do
  echo "=== distillation_llm.py --mode $mode ==="
  conda run --no-capture-output -n kv torchrun --standalone --nproc_per_node=2 \
    distillation_llm.py --mode "$mode" --total-tokens 500000000 \
    > "distill_llm_${mode}.log" 2>&1
done
