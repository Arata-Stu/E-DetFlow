#!/usr/bin/env python3
import sys
sys.path.append("../")
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from utils.directory import SequenceDir

# ==========================================
# 設定
# ==========================================
# 既知のクラスリスト（表示順序の固定用）
KNOWN_CLASSES = [
    'car', 'van', 'truck', 
    'pedestrian', 
    'cyclist', 'bicycle', 'motorcycle', 
    'misc'
]

# ==========================================
# 解析ロジック
# ==========================================

def analyze_sequence(seq: SequenceDir):
    """
    1つのシーケンス内の全ラベルファイルを読み込み、統計情報を返す
    """
    dvs_dir = seq.dvs_dir
    if not dvs_dir.exists():
        return None

    # txtファイル取得
    label_files = sorted(list(dvs_dir.glob("dvs-*.txt")))
    if not label_files:
        return None

    # カウンター初期化
    class_counts = Counter()
    occlusion_counts = Counter({0:0, 1:0, 2:0, 3:0})
    total_boxes = 0

    for txt_file in label_files:
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                parts = line.split(' ')
                if len(parts) < 15: continue # KITTI形式の最低長チェック

                # --- 1. Class Count ---
                obj_type = parts[0].lower()
                class_counts[obj_type] += 1

                # --- 2. Occlusion Count ---
                try:
                    # KITTI format index 2 is 'occluded'
                    occluded = int(parts[2])
                    occlusion_counts[occluded] += 1
                except ValueError:
                    # int変換できない場合などは無視（あるいはunknownとしてカウント）
                    pass
                
                total_boxes += 1

    return {
        "seq_id": seq.root.name,
        "town": seq.root.parent.name,
        "class_counts": class_counts,
        "occlusion_counts": occlusion_counts,
        "total_boxes": total_boxes
    }

def print_stats(stats_list):
    """
    集計結果を表形式で出力する
    """
    if not stats_list:
        print("No labels found.")
        return

    # 全体集計用
    grand_total_class = Counter()
    grand_total_occ = Counter()
    grand_total_boxes = 0

    print("\n" + "="*80)
    print(f"{'Town/Seq':<20} | {'Total':<8} || {'Occ=0':<6} {'Occ=1':<6} {'Occ=2':<6} {'Occ=3':<6} || {'Classes (Top 3)'}")
    print("-" * 80)

    for stat in stats_list:
        name = f"{stat['town']}/{stat['seq_id']}"
        total = stat['total_boxes']
        occ = stat['occlusion_counts']
        cls = stat['class_counts']

        # 全体集計に加算
        grand_total_boxes += total
        grand_total_occ.update(occ)
        grand_total_class.update(cls)

        # クラスの上位3つを表示用に文字列化
        top_classes = cls.most_common(3)
        cls_str = ", ".join([f"{k}:{v}" for k, v in top_classes])

        print(f"{name:<20} | {total:<8} || {occ[0]:<6} {occ[1]:<6} {occ[2]:<6} {occ[3]:<6} || {cls_str}")

    print("=" * 80)
    print("GRAND TOTAL SUMMARY")
    print("-" * 80)
    print(f"Total Bounding Boxes: {grand_total_boxes}")
    
    print("\n[Occlusion Distribution]")
    for i in range(4):
        count = grand_total_occ[i]
        percent = (count / grand_total_boxes * 100) if grand_total_boxes > 0 else 0
        print(f"  Occluded {i}: {count:>8} ({percent:.1f}%)")

    print("\n[Class Distribution]")
    # 既知のクラス順で表示、それ以外はその他として表示
    for c in KNOWN_CLASSES:
        if grand_total_class[c] > 0:
            print(f"  {c:<12}: {grand_total_class[c]:>8}")
            del grand_total_class[c] # 表示済みは削除
    
    # 未定義のクラスがあれば表示（デバッグ用）
    if len(grand_total_class) > 0:
        print("\n  [Unknown Classes Found!]")
        for c, count in grand_total_class.items():
            print(f"  {c:<12}: {count:>8}")
    print("=" * 80)


# ==========================================
# メイン探索ロジック
# ==========================================
def process_dataset(root_dir: Path):
    town_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and "Town" in d.name])

    if not town_dirs:
        print(f"No 'Town' directories found in {root_dir}")
        return

    all_sequence_stats = []

    # プログレスバーで表示
    pbar = tqdm(total=0, desc="Analyzing")
    
    # 事前にタスク数を数えるのが面倒なので、発見次第更新するスタイル
    for town in town_dirs:
        part_dirs = sorted([d for d in town.iterdir() if d.is_dir() and d.name.isdigit()])
        
        # Town直下シーケンスの場合
        if not part_dirs:
            seq = SequenceDir(town)
            stats = analyze_sequence(seq)
            if stats:
                all_sequence_stats.append(stats)
            continue

        # 分割シーケンスの場合
        for part in part_dirs:
            seq = SequenceDir(part)
            stats = analyze_sequence(seq)
            if stats:
                all_sequence_stats.append(stats)
                pbar.set_description(f"Analyzing {town.name}/{part.name}")
                pbar.update(1)
    
    pbar.close()

    # 結果表示
    print_stats(all_sequence_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze class and occlusion distribution in KITTI labels.")
    parser.add_argument("input_dir", type=str, help="Dataset Root Directory")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
    else:
        process_dataset(input_path)