#!/bin/bash

set -euo pipefail
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5,6}"

PYTHON_BIN="${AEROVLA_PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

echo "Using Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
if ! "$PYTHON_BIN" -c 'import torch, transformers, timm, accelerate, safetensors' >/dev/null; then
    echo "ERROR: AeroVLA Python dependencies are incompatible or incomplete." >&2
    echo "Activate the documented Python 3.10 aero_vla environment, or set:" >&2
    echo "  AEROVLA_PYTHON=/path/to/aero_vla/bin/python bash scripts/train_nollm.sh ..." >&2
    echo "Then install the pinned requirements with that same interpreter:" >&2
    echo "  /path/to/python -m pip install -r requirements.txt" >&2
    exit 1
fi

NUM_GPUS="${AEROVLA_NUM_GPUS:-$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')}"
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching no-LLM ablation with DDP on $NUM_GPUS GPUs..."
    exec "$PYTHON_BIN" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
        ./src/train_aerovla_nollm.py "$@"
fi

echo "Launching no-LLM ablation on one GPU..."
exec "$PYTHON_BIN" ./src/train_aerovla_nollm.py "$@"
