#!/bin/bash
# Member-parallel run: one training process per GPU, each owning N / n_gpus members (reppo_llm/train.py). If any
# process fails, the others are stopped instead of waiting on the store.
# Usage: bash scripts/launch.sh <gpus, e.g. 1,2,3,4,5> conf/math.yaml conf/math_<model>.yaml [conf/overlay.yaml ...]
#        [key=value ...]
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
IFS=, read -ra GPUS <<< "$1"
shift
PORT=$(python -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
for r in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$r]} REPPO_RANK=$r REPPO_WORLD=${#GPUS[@]} REPPO_PORT=$PORT python -m reppo_llm.train "$@" &
done
while [ -n "$(jobs -rp)" ]; do
    wait -n || { kill $(jobs -rp) 2>/dev/null; exit 1; }
done
