#!/bin/bash

# --- 実行モード設定 ---
VIS_MODE="flow" # flow or detection

# --- 基本設定 ---
GPU_IDS=0  
MODEL="rvt" # rvt, yolox_lstm
DATASET="SEVD"
DT_MS=33
T_BIN=10
CHANNEL=20
REPR_TYPE="stacked_histogram"  # stacked_histogram, voxel_grid 
NORM="norm"

# --- パス設定 ---
DATA_DIR="/path/to/dataset/"
CKPT_PATH="/path/to/checkpoint/model.ckpt"
OUTPUT_PATH="./video/output_${VIS_MODE}.mp4"

# --- 可視化フラグ ---
SHOW_GT=true
SHOW_PRED=false

# --- モードに応じた条件分岐 ---
if [ "$VIS_MODE" == "flow" ]; then
    SCRIPT_NAME="visualize_flow.py"
    TRAIN_TASK="optical_flow"
    USE_FLOW=true
    USE_BOX=false
elif [ "$VIS_MODE" == "detection" ]; then
    SCRIPT_NAME="visualize_detection.py"
    TRAIN_TASK="detection"
    USE_FLOW=false
    USE_BOX=true
else
    echo "Invalid VIS_MODE: $VIS_MODE. Use 'flow' or 'detection'."
    exit 1
fi

# --- 表現名の生成 ---
REPR_NAME="${REPR_TYPE}_dt=${DT_MS}_nbins=${T_BIN}"
if [ "$REPR_TYPE" == "voxel_grid" ]; then
    REPR_NAME="${REPR_NAME}_${NORM}"
fi

echo "------------------------------------------"
echo "Mode: ${VIS_MODE} -> Running ${SCRIPT_NAME}"
echo "Model: ${MODEL} | Task: ${TRAIN_TASK}"
echo "Checkpoint: ${CKPT_PATH}"
echo "------------------------------------------"

# --- 実行コマンド ---
CUDA_VISIBLE_DEVICES=${GPU_IDS} python3 ${SCRIPT_NAME} \
model=${MODEL} \
+train_task=${TRAIN_TASK} \
output_path=${OUTPUT_PATH} \
gt=${SHOW_GT} \
pred=${SHOW_PRED} \
dataset=${DATASET} \
dataset.path=${DATA_DIR} \
ckpt_path=${CKPT_PATH} \
dataset.ev_repr_name="'${REPR_NAME}'" \
dataset.use_flow=${USE_FLOW} \
dataset.use_box=${USE_BOX}