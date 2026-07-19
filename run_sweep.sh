#!/usr/bin/env bash
# Launch both synthetic-tasks sweep shards fully detached (survive session disconnect).
cd /home/akayyam/Do-Transformers-Need-3-Projections
setsid nohup conda run -n kv --no-capture-output python synthetic_tasks.py \
    --shard 0/2 --device cuda:2 > sweep_gpu2.log 2>&1 < /dev/null &
setsid nohup conda run -n kv --no-capture-output python synthetic_tasks.py \
    --shard 1/2 --device cuda:3 > sweep_gpu3.log 2>&1 < /dev/null &
echo "launched shards on cuda:2 and cuda:3"
