#!/bin/bash

# エラーが発生したら即停止
set -e

# --- 設定項目 ---
INPUT_DIR="/mnt/ssd-4tb/dataset/carla/"
OUTPUT_DIR="/mnt/ssd-4tb/dataset/carla_preprocessed/"

# 固定のコンフィグ
SPLIT_YAML="conf_preprocess/split_SEVD.yaml"
EXTRACTION_YAML="conf_preprocess/extraction/const_duration.yaml"
FILTER_YAML="conf_preprocess/filter_SEVD.yaml"

# 表現手法のリスト (YAMLファイル名から .yaml を除いたもの)
REPRESENTATIONS=(
    "stacked_hist_interpolated"
    "stacked_hist"
    "voxel_grid"
)

# --- 処理ループ ---
for REPR in "${REPRESENTATIONS[@]}"; do
    
    # 表現用YAMLのパスを構築
    REPR_YAML="conf_preprocess/representation/${REPR}.yaml"
    
    echo "=========================================================="
    echo " Start processing: ${REPR}"
    echo " Config: ${REPR_YAML}"
    echo "=========================================================="

    # ファイル存在チェック
    if [ ! -f "$REPR_YAML" ]; then
        echo "❌ Error: Configuration file not found: $REPR_YAML"
        exit 1
    fi

    # Pythonスクリプト実行
    python3 5_preprocess_dataset.py \
        "$INPUT_DIR" \
        "$OUTPUT_DIR" \
        "$SPLIT_YAML" \
        "$REPR_YAML" \
        "$EXTRACTION_YAML" \
        "$FILTER_YAML" \
        -d SEVD \
        --downsample \
        -np 5

    echo "✅ Finished: ${REPR}"
    echo ""
done

echo " All representations processed successfully!"
