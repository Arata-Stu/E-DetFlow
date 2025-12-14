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
PROJECT="E-Det_${MODEL}_${DATASET}"
DOWNSAMPLE=true  # true false

DATA_DIR="/path/to/dataset/"
GROUP=""

python3 train.py \
dataset=${DATASET} \
model=${MODEL} \
dataset.path=${DATA_DIR} \
dataset.ev_repr_name="'stacked_histogram_dt=${DT_MS}_nbins=${T_BIN}'" \
dataset.sequence_length=${SEQUENCE_LENGTH} \
hardware.gpus=${GPU_IDS} \
model.backbone.input_channels=${CHANNEL} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} \
hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU} \
batch_size.train=${BATCH_SIZE_PER_GPU} \
batch_size.eval=${BATCH_SIZE_PER_GPU} \
wandb.project_name=${PROJECT} \
wandb.group_name=${GROUP} \
dataset.downsample_by_factor_2=${DOWNSAMPLE} 
