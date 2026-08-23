#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

NUM_GPUS="${AEROVLA_NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training on $NUM_GPUS GPUs (DDP)..."
    exec python -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NUM_GPUS" \
        ./src/train_aerovla.py "$@"
fi

echo "Launching training on a single GPU..."
exec python ./src/train_aerovla.py "$@"
