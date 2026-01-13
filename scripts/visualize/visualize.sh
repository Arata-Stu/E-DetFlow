#!/bin/bash

# --- 実行モード設定 ---
# オプション: flow, detection, combined
VIS_MODE="combined" 

# --- 基本設定 ---
GPU_IDS=0  
MODEL="rvt" # rvt, yolox_lstm
DATASET="SEVD"
DT_MS=33
T_BIN=10
CHANNEL=20
REPR_TYPE="stacked_histogram"  # stacked_histogram, voxel_grid 
NORM="norm"
APPLY_MASK=true  # true or false

DATASET_MODE='test' # 'train' or 'val' or 'test'

# --- パス設定 ---
DATA_DIR="/path/to/dataset/"
CKPT_PATH="/path/to/checkpoint/model.ckpt"
OUTPUT_PATH="./video/output_${VIS_MODE}.mp4"

# --- 可視化フラグ ---
SHOW_GT=true
SHOW_PRED=true # 統合モードでは通常両方見たいので true 推奨

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
elif [ "$VIS_MODE" == "combined" ]; then
    # 以前作成した統合可視化スクリプト名
    SCRIPT_NAME="visualize_flow_det.py" 
    TRAIN_TASK="flow_and_detection"
    USE_FLOW=true
    USE_BOX=true
else
    echo "Invalid VIS_MODE: $VIS_MODE. Use 'flow', 'detection', or 'combined'."
    exit 1
fi

# --- 表現名の生成 ---
REPR_NAME="${REPR_TYPE}_dt=${DT_MS}_nbins=${T_BIN}"
if [ "$REPR_TYPE" == "voxel_grid" ]; then
    REPR_NAME="${REPR_NAME}_${NORM}"
fi

echo "------------------------------------------------"
echo "Mode: ${VIS_MODE} -> Running ${SCRIPT_NAME}"
echo "Model: ${MODEL} | Task: ${TRAIN_TASK}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Flags: Flow=${USE_FLOW}, Box=${USE_BOX}"
echo "------------------------------------------------"

# --- 実行コマンド ---
CUDA_VISIBLE_DEVICES=${GPU_IDS} python3 ${SCRIPT_NAME} \
model=${MODEL} \
+train_task=${TRAIN_TASK} \
output_path=${OUTPUT_PATH} \
gt=${SHOW_GT} \
pred=${SHOW_PRED} \
dataset=${DATASET} \
dataset.path=${DATA_DIR} \
dataset_mode="${DATASET_MODE}" \
ckpt_path=${CKPT_PATH} \
dataset.ev_repr_name="'${REPR_NAME}'" \
dataset.use_flow=${USE_FLOW} \
dataset.use_box=${USE_BOX} \
apply_mask=${APPLY_MASK}