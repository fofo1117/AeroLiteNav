#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,7}"
PYTHON_BIN="${AEROVLA_PYTHON:-python}"
NUM_GPUS="${AEROVLA_NUM_GPUS:-$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')}"
if [ "$NUM_GPUS" -gt 1 ]; then
    exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
        ./src/train_aerovla_crossattn.py "$@"
fi
exec "$PYTHON_BIN" ./src/train_aerovla_crossattn.py "$@"
