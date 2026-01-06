#!/bin/bash

# ==========================================
# 設定エリア (環境に合わせて書き換えてください)
# ==========================================
DATASET_DIR="/path/to/dataset"
OUTPUT_DIR="/path/to/output"
NUM_PROCESSORS=5
NUM_WORKERS_FLOW=10

# 共通の構成ファイル
REP_YAML="conf_preprocess/representation/stacked_hist.yaml"
EXT_YAML="conf_preprocess/extraction/const_duration.yaml"
FILTER_YAML="conf_preprocess/filter_DSEC.yaml"
IGNORE_YAML="conf_preprocess/ignore_DSEC.yaml"

# ==========================================
# モード判定
# ==========================================
MODE=$1

if [ "$MODE" == "det" ]; then
    echo "Starting DSEC-Detection preprocessing..."
    SPLIT_YAML="conf_preprocess/split_DSEC-det.yaml"

elif [ "$MODE" == "flow" ]; then
    echo "Starting DSEC-Flow preprocessing..."
    SPLIT_YAML="conf_preprocess/split_DSEC-flow.yaml"
    
    # Flowの場合のみ必要なフォーマット変換を実行
    echo "Step 1: Running 1_format_flow..."
    python3 1_format_flow ${DATASET_DIR} \
        --num_workers ${NUM_WORKERS_FLOW} \
        --config ${SPLIT_YAML}
else
    echo "Usage: ./preprocess.sh [det|flow]"
    exit 1
fi

# ==========================================
# 共通のプリプロセス実行 (Step 2)
# ==========================================
echo "Step 2: Running 2_preprocess_dataset..."
python3 2_preprocess_dataset ${DATASET_DIR} ${OUTPUT_DIR} \
    ${SPLIT_YAML} \
    ${REP_YAML} \
    ${EXT_YAML} \
    ${FILTER_YAML} \
    -d DSEC \
    -np ${NUM_PROCESSORS} \
    --ignore_yaml ${IGNORE_YAML}

echo "Done!"