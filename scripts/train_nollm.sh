#!/bin/bash

set -euo pipefail
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3,4,5,6

NUM_GPUS="${AEROVLA_NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching no-LLM ablation with DDP on $NUM_GPUS GPUs..."
    exec python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
        ./src/train_aerovla_nollm.py "$@"
fi

echo "Launching no-LLM ablation on one GPU..."
exec python ./src/train_aerovla_nollm.py "$@"
