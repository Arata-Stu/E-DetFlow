#!/bin/bash

# エラーが発生したら即停止
set -e

USE_FILTER=false

# --- 設定項目 ---
INPUT_DIR="/mnt/ssd-4tb/dataset/carla/"

if [ "$USE_FILTER" = true ]; then
    OUTPUT_DIR="/mnt/ssd-4tb/dataset/carla_preprocessed_filtered/"
else
    OUTPUT_DIR="/mnt/ssd-4tb/dataset/carla_preprocessed/"
fi

SPLIT_YAML="conf_preprocess/split_SEVD.yaml"
EXTRACTION_YAML="conf_preprocess/extraction/const_duration.yaml"
FILTER_YAML="conf_preprocess/filter_SEVD.yaml"

REPRESENTATIONS=(
    # "stacked_hist_interpolated"
    "stacked_hist"
    "voxel_grid"
)

# --- 処理ループ ---
for REPR in "${REPRESENTATIONS[@]}"; do
    
    REPR_YAML="conf_preprocess/representation/${REPR}.yaml"
    
    echo "=========================================================="
    echo " Start processing: ${REPR}"
    echo " Config: ${REPR_YAML}"
    echo "=========================================================="

    if [ ! -f "$REPR_YAML" ]; then
        echo "❌ Error: Configuration file not found: $REPR_YAML"
        exit 1
    fi

    # ---------------------------------------------------------
    # 【ここが安全な実装のポイント】
    # コマンドの引数を配列として定義します
    # ---------------------------------------------------------
    CMD_ARGS=(
        "$INPUT_DIR"
        "$OUTPUT_DIR"
        "$SPLIT_YAML"
        "$REPR_YAML"
        "$EXTRACTION_YAML"
        "$FILTER_YAML"
        -d SEVD
        --downsample
        -np 5
    )

    # 条件に応じて配列に追加 (append) します
    if [ "$USE_FILTER" = true ]; then
        CMD_ARGS+=("--filtered_label")
    fi

    # デバッグ用: 実際に実行されるコマンドを表示（必要に応じてコメントアウト）
    echo "Executing: python3 5_preprocess_dataset.py ${CMD_ARGS[*]}"

    # 配列を展開して実行
    # "${CMD_ARGS[@]}" と書くことで、スペース区切りなどが正確に処理されます
    python3 5_preprocess_dataset.py "${CMD_ARGS[@]}"

    echo "✅ Finished: ${REPR}"
    echo ""
done

echo " All representations processed successfully!"