#!/bin/bash
set -euo pipefail
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${AEROVLA_GPU_ID:-0}"
MODEL_DIR="${AEROVLA_MODEL_DIR:-./checkpoints/aero_vla_lite_s1_t3_d384_l4}"
TASK_ID="${AEROVLA_TASK_ID:-seen_valset/NYCEnvironmentMegapa}"
SAFETY_DEPTH_M="${AEROVLA_SAFETY_DEPTH_M:-3.0}"
python -u ./src/vlnce_src/eval_aerovla.py \
    --run_type eval --name AeroVLA_Lite_Eval --model_variant lite \
    --model_path "$MODEL_DIR" \
    --eval_save_path "./eval_results/aero_vla_lite/$TASK_ID" \
    --eval_json_path "./data/uav_dataset/$TASK_ID.json" \
    --dataset_path ./dataset_raw \
    --activate_maps "${TASK_ID##*/}" \
    --safety_depth_threshold_m "$SAFETY_DEPTH_M" \
    --batchSize 1 --gpu_id 0 --simulator_tool_port 30000 "$@"
