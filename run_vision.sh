#!/usr/bin/env bash
# Launch the minimal vision classification sweep, 4-way sharded across GPUs 0-3,
# fully detached (survive session disconnect). Datasets must be pre-downloaded.
cd /home/akayyam/Do-Transformers-Need-3-Projections
for i in 0 1 2 3; do
  setsid nohup conda run -n torch_env --no-capture-output python vision_tasks.py \
      --grid minimal --shard $i/4 --device cuda:$i --workers 3 \
      > vision_gpu$i.log 2>&1 < /dev/null &
done
echo "launched 4 minimal-grid shards on cuda:0-3"
