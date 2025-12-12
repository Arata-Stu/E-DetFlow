#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-./datasets}"
OUT_FILE="${2:-split_SEVD.yaml}"
VAL_RATIO=0.15
TEST_RATIO=0.15

# bcコマンドの存在チェック（計算に使用するため）
if ! command -v bc &> /dev/null; then
    echo "❌ 'bc' コマンドが見つかりません。インストールしてください (例: sudo apt install bc)"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
  echo "❌ ディレクトリが存在しません: $DATASET_ROOT"
  exit 1
fi

echo "📂 データセットルート: $DATASET_ROOT"
echo "📊 出力ファイル: $OUT_FILE"
echo "==============================================="

: > "$OUT_FILE"

# 1. 全シーンディレクトリを取得
mapfile -t all_scenes < <(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
if [ ${#all_scenes[@]} -eq 0 ]; then
  echo "❌ シーンディレクトリが見つかりません。"
  exit 1
fi

# 2. ユニークなTown名を抽出
mapfile -t unique_towns < <(printf "%s\n" "${all_scenes[@]}" | grep -o "Town[0-9][0-9]" | sort -u)

if [ ${#unique_towns[@]} -eq 0 ]; then
  echo "❌ ディレクトリ名から 'TownXX' が見つかりませんでした。"
  exit 1
fi

echo "🏙️  検出されたTown一覧: ${unique_towns[*]}"

# 3. Town単位でシャッフルして分割
shuffled_towns=($(printf "%s\n" "${unique_towns[@]}" | shuf))
total_towns=${#shuffled_towns[@]}

val_count=$(printf "%.0f" "$(echo "$total_towns * $VAL_RATIO" | bc)")
test_count=$(printf "%.0f" "$(echo "$total_towns * $TEST_RATIO" | bc)")

# 最低数の保証
if [ "$total_towns" -ge 3 ]; then
    [ "$val_count" -eq 0 ] && val_count=1
    [ "$test_count" -eq 0 ] && test_count=1
fi

train_towns=("${shuffled_towns[@]:0:$((total_towns - val_count - test_count))}")
val_towns=("${shuffled_towns[@]:$((total_towns - val_count - test_count)):$val_count}")
test_towns=("${shuffled_towns[@]:$((total_towns - test_count)):$test_count}")

echo "🎯 Town割り当て結果:"
echo "   Train Towns: ${train_towns[*]}"
echo "   Val Towns  : ${val_towns[*]}"
echo "   Test Towns : ${test_towns[*]}"

# 4. splitごとにYAML出力
declare -A TOWN_GROUPS=(
  ["train"]="${train_towns[*]}"
  ["val"]="${val_towns[*]}"
  ["test"]="${test_towns[*]}"
)

for split in train val test; do
  echo "${split}:" >> "$OUT_FILE"
  
  target_towns=(${TOWN_GROUPS[$split]})
  

  for town in "${target_towns[@]}"; do
    for scene_path in "${all_scenes[@]}"; do
      if [[ "$scene_path" == *"$town"* ]]; then
        
        scene_name=$(basename "$scene_path")
        
        mapfile -t subs < <(find "$scene_path" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9][0-9]$' | sort)
        
        if [ ${#subs[@]} -eq 0 ]; then
           echo "  - ${scene_name}" >> "$OUT_FILE"
        else
           for sub in "${subs[@]}"; do
             sub_name=$(basename "$sub")
             echo "  - ${scene_name}/${sub_name}" >> "$OUT_FILE"
           done
        fi
      fi
    done
  done
done

echo "✅ YAML 出力完了 → ${OUT_FILE}"