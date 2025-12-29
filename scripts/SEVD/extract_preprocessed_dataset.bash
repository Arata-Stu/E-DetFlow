#!/bin/bash

# 引数チェック
if [ "$#" -ne 2 ]; then
    echo "使用法: $0 <入力ルートディレクトリ> <出力先ディレクトリ>"
    echo "例: $0 /data/carla_dataset /data/extracted_dataset"
    exit 1
fi

SRC_ROOT=$(realpath "$1")
DST_ROOT=$(realpath "$2")

# 入力ディレクトリが存在するか確認
if [ ! -d "$SRC_ROOT" ]; then
    echo "エラー: 入力ディレクトリが存在しません: $SRC_ROOT"
    exit 1
fi

# 抽出対象のディレクトリ名
TARGET_NAMES=("events" "optical_flow_processed" "labels")

echo "=========================================="
echo "コピー元: $SRC_ROOT"
echo "コピー先: $DST_ROOT"
echo "抽出対象: ${TARGET_NAMES[*]}"
echo "=========================================="

# 抽出対象のリストを作成
echo "コピー対象をスキャン中..."
# findで対象ディレクトリをすべて検索し、一時ファイルに保存
LIST_FILE=$(mktemp)
for name in "${TARGET_NAMES[@]}"; do
    find "$SRC_ROOT" -type d -name "$name" >> "$LIST_FILE"
done

# 見つかった数を確認
COUNT=$(wc -l < "$LIST_FILE")
if [ "$COUNT" -eq 0 ]; then
    echo "対象ディレクトリが見つかりませんでした。"
    rm "$LIST_FILE"
    exit 0
fi

echo "$COUNT 個のディレクトリが見つかりました。"
echo "------------------------------------------"
echo "プレビュー (最初の5件):"
head -n 5 "$LIST_FILE"
echo "..."
echo "------------------------------------------"

# ユーザーに確認
read -p "構造を維持してコピーを開始しますか？ (y/N): " CONFIRM

if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
    echo "実行中..."

    while IFS= read -r src_dir; do
        # 入力ルートからの相対パスを取得
        # 例: /data/carla/seq1/events -> seq1/events
        REL_PATH="${src_dir#$SRC_ROOT/}"
        
        # 出力先のフルパスを作成
        DEST_DIR="$DST_ROOT/$REL_PATH"
        
        # 親ディレクトリを作成してコピー
        mkdir -p "$(dirname "$DEST_DIR")"
        cp -r "$src_dir" "$DEST_DIR"
        
        echo "Copied: $REL_PATH"
    done < "$LIST_FILE"

    echo "------------------------------------------"
    echo "完了しました。出力先を確認してください: $DST_ROOT"
else
    echo "キャンセルしました。"
fi

rm "$LIST_FILE"