#!/bin/bash

# 引数チェック
if [ -z "$1" ]; then
    echo "使用法: $0 <データセットのルートディレクトリ>"
    echo "例: $0 /data/carla_dataset"
    exit 1
fi

TARGET_ROOT="$1"

# ディレクトリが存在するか確認
if [ ! -d "$TARGET_ROOT" ]; then
    echo "エラー: 指定されたディレクトリが存在しません: $TARGET_ROOT"
    exit 1
fi

echo "=========================================="
echo "検索対象ディレクトリ: $TARGET_ROOT"
echo "削除対象: 名前が 'events' のディレクトリすべて"
echo "=========================================="
echo ""

# 削除対象のリストアップ (Dry Run)
# -type d: ディレクトリのみ
# -name "events": 名前が "events" に完全一致するもの
FOUND_DIRS=$(find "$TARGET_ROOT" -type d -name "events")

if [ -z "$FOUND_DIRS" ]; then
    echo "削除対象の 'events' ディレクトリは見つかりませんでした。"
    exit 0
fi

# 見つかったディレクトリを表示
echo "以下のディレクトリが見つかりました:"
echo "$FOUND_DIRS"
echo ""
echo "------------------------------------------"
echo "合計: $(echo "$FOUND_DIRS" | wc -l) 個のディレクトリ"
echo "------------------------------------------"

# ユーザーに確認
read -p "これらをすべて削除してよろしいですか？ (y/N): " CONFIRM

if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
    echo ""
    echo "削除を実行中..."
    
    # 実際に削除を実行
    # xargs rm -rf を使って一括削除
    echo "$FOUND_DIRS" | xargs rm -rf
    
    echo "完了しました。"
else
    echo "キャンセルしました。"
fi