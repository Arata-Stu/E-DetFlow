#!/bin/bash

# 引数チェック
if [ -z "$1" ]; then
    echo "使用法: $0 <データセットのルートディレクトリ>"
    echo "例: $0 /data/dsec_dataset"
    exit 1
fi

TARGET_ROOT="$1"

# trainとtestのパスを設定
TRAIN_DIR="${TARGET_ROOT%/}/train"
TEST_DIR="${TARGET_ROOT%/}/test"

# 探索対象のディレクトリが存在するか確認し、リストを作成
SEARCH_PATHS=""
if [ -d "$TRAIN_DIR" ]; then
    SEARCH_PATHS="$TRAIN_DIR"
fi

if [ -d "$TEST_DIR" ]; then
    SEARCH_PATHS="$SEARCH_PATHS $TEST_DIR"
fi

# どちらも存在しない場合はエラー
if [ -z "$SEARCH_PATHS" ]; then
    echo "エラー: '$TARGET_ROOT' 内に train または test ディレクトリが見つかりません。"
    exit 1
fi

echo "=========================================="
echo "探索ベースパス: $TARGET_ROOT"
echo "探索サブパス: $(echo $SEARCH_PATHS)"
echo "削除対象: 名前が 'optical_flow_processed' のディレクトリすべて"
echo "=========================================="
echo ""

# 削除対象のリストアップ (Dry Run)
# 複数のパス（train と test）をfindに渡す
FOUND_DIRS=$(find $SEARCH_PATHS -type d -name "optical_flow_processed" 2>/dev/null)

if [ -z "$FOUND_DIRS" ]; then
    echo "削除対象の 'optical_flow_processed' ディレクトリは見つかりませんでした。"
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
    
    # xargs を使って安全に削除を実行
    # ディレクトリ名にスペースが含まれる可能性を考慮して -d '\n' を指定
    echo "$FOUND_DIRS" | xargs -d '\n' rm -rf
    
    echo "完了しました。"
else
    echo "キャンセルしました。"
fi