#!/bin/bash

# GPU設定
GPU_IDS=0  
MODEL="rvt"  # rvt, yolox_lstm
DATASET="SEVD"  # gen1 gen4 VGA SEVD
BATCH_SIZE_PER_GPU=12
TRAIN_WORKERS_PER_GPU=8
EVAL_WORKERS_PER_GPU=4
DT_MS=50
T_BIN=10
CHANNEL=20
SEQUENCE_LENGTH=5
DOWNSAMPLE=true  # true false
REPR_TYPE="stacked_histogram"  # stacked_histogram voxel_grid 
NORM="norm"

DATA_DIR="/path/to/dataset/"
GROUP=""

CKPT="/path/to/checkpoint.ckpt"
TRAIN_TASK="detection" # detection optical_flow

if [ "$TRAIN_TASK" == "detection" ]; then
    PROJECT_PREFIX="E-Det"
elif [ "$TRAIN_TASK" == "optical_flow" ]; then
    PROJECT_PREFIX="E-Flow"
else
    echo "Unsupported TRAIN_TASK: $TRAIN_TASK"
    exit 1
fi
PROJECT="${PROJECT_PREFIX}_${MODEL}_${DATASET}"

REPR_NAME="${REPR_TYPE}_dt=${DT_MS}_nbins=${T_BIN}"

if [ "$REPR_TYPE" == "voxel_grid" ]; then
    REPR_NAME="${REPR_NAME}_${NORM}"
fi

echo "Generated Representation Name: ${REPR_NAME}"

python3 validation.py \
dataset=${DATASET} \
model=${MODEL} \
dataset.path=${DATA_DIR} \
dataset.ev_repr_name="'${REPR_NAME}'" \
dataset.sequence_length=${SEQUENCE_LENGTH} \
hardware.gpus=${GPU_IDS} \
model.backbone.input_channels=${CHANNEL} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} \
hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU} \
batch_size.train=${BATCH_SIZE_PER_GPU} \
batch_size.eval=${BATCH_SIZE_PER_GPU} \
dataset.downsample_by_factor_2=${DOWNSAMPLE} \
checkpoint=${CKPT} \
+train_task=${TRAIN_TASK}