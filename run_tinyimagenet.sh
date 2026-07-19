#!/usr/bin/env bash
# TinyImageNet: all 6 variants, 20 epochs, across 4 GPUs (2 GPUs run 2 variants each),
# fully detached (survive session disconnect). ~18h wall-clock. AMP, batch 32.
cd /home/akayyam/Do-Transformers-Need-3-Projections
EP=20

launch () {  # $1=device  $2=variants  $3=tag
  setsid nohup conda run -n torch_env --no-capture-output python vision_tasks.py \
      --tiny --tiny-variants "$2" --tiny-epochs $EP --tiny-batch 32 --workers 3 \
      --device cuda:$1 --tiny-out "vision_results_tiny_$3.csv" \
      > tiny_gpu$1.log 2>&1 < /dev/null &
}

launch 0 "QKV,(Q=K=V)+"      g0
launch 1 "Q=K!=V,(Q=K!=V)+"  g1
launch 2 "Q!=K=V"            g2
launch 3 "Q=K=V"             g3
echo "launched TinyImageNet: 6 variants / 4 GPUs, $EP epochs"
