#!/bin/bash
set -euo pipefail
PROJECT_ROOT="."
MODEL_DIR="${AEROVLA_MODEL_DIR:-$PROJECT_ROOT/checkpoints/aero_vla_crossattn_d768_l6}"
EXP_NAME=$(basename "$MODEL_DIR")
TASK_ID="${AEROVLA_TASK_ID:-seen_valset/NYCEnvironmentMegapa}"
CATEGORY=$(echo "$TASK_ID" | cut -d'/' -f1)
MAP_NAME=$(echo "$TASK_ID" | cut -d'/' -f2)
TEST_JSON="$PROJECT_ROOT/data/uav_dataset/${CATEGORY}_splits/${MAP_NAME}.json"
SAVE_DIR="$PROJECT_ROOT/eval_results/${EXP_NAME}/${CATEGORY}/${MAP_NAME}"
GPU_ID="${AEROVLA_GPU_ID:-0}"
PORT=$((30000 + GPU_ID * 5000))
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=$GPU_ID python -u "$PROJECT_ROOT/src/vlnce_src/eval_aerovla.py" \
    --run_type eval --name AeroVLA_CrossAttn_Eval --model_variant crossattn \
    --gpu_id "$GPU_ID" --simulator_tool_port "$PORT" --DDP_MASTER_PORT 80005 \
    --batchSize 1 --maxWaypoints 200 --dataset_path "$PROJECT_ROOT/dataset_raw/" \
    --eval_save_path "$SAVE_DIR" --model_path "$MODEL_DIR" --eval_json_path "$TEST_JSON" \
    --map_spawn_area_json_path "$PROJECT_ROOT/data/meta/map_spawnarea_info.json" \
    --object_name_json_path "$PROJECT_ROOT/data/meta/object_description.json" "$@"
