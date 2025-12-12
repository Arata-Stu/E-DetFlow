#!/usr/bin/env python3
import sys
sys.path.append("../")
import argparse
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from utils.directory import SequenceDir

# ==========================================
# 1. データ構造 & モーション解析クラス
# ==========================================
GNSS_FULL_DTYPE = np.dtype([
    ('timestamp', '<f8'),
    ('lat', '<f8'),
    ('lon', '<f8')
])

class MotionAnalyzer:
    @staticmethod
    def parse_gnss_trajectory(file_path: Path):
        """速度計算用に緯度経度のみを抽出"""
        if not file_path.exists(): return None
        data = []
        pattern = re.compile(r"timestamp=([0-9.]+),\s*lat=([0-9.-]+),\s*lon=([0-9.-]+)")
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        data.append((
                            float(match.group(1)), # timestamp
                            float(match.group(2)), # lat
                            float(match.group(3))  # lon
                        ))
        except Exception: return None
        if not data: return None
        return np.array(data, dtype=GNSS_FULL_DTYPE)

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000.0 
        d_lat = np.radians(lat2 - lat1)
        d_lon = np.radians(lon2 - lon1)
        a = np.sin(d_lat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    @staticmethod
    def calculate_speeds(gnss_data):
        if len(gnss_data) < 2: return np.zeros(len(gnss_data))
        
        d_dist = MotionAnalyzer.haversine_distance(
            gnss_data['lat'][:-1], gnss_data['lon'][:-1],
            gnss_data['lat'][1:], gnss_data['lon'][1:]
        )
        d_time = gnss_data['timestamp'][1:] - gnss_data['timestamp'][:-1]
        d_time[d_time == 0] = 1e-6 
        
        speed_kmh = (d_dist / d_time) * 3.6
        return np.concatenate(([0], speed_kmh))

    @staticmethod
    def get_static_mask(gnss_data, threshold_kmh, min_duration_sec):
        speeds = MotionAnalyzer.calculate_speeds(gnss_data)
        
        is_candidate = speeds < threshold_kmh
        padded = np.concatenate(([False], is_candidate, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        final_static_mask = np.zeros_like(is_candidate, dtype=bool)
        
        for s, e in zip(starts, ends):
            t_start = gnss_data['timestamp'][s]
            t_end = gnss_data['timestamp'][min(e - 1, len(gnss_data) - 1)]
            duration = t_end - t_start
            
            if duration >= min_duration_sec:
                final_static_mask[s:e] = True
        
        avg_speed = np.mean(speeds) if len(speeds) > 0 else 0
        return final_static_mask, avg_speed

# ==========================================
# 2. ヘルパー関数
# ==========================================
def parse_gnss_timestamps(gnss_file_path: Path):
    frame_to_ts = {}
    pattern = re.compile(r"frame=(\d+),\s*timestamp=([0-9.]+)")
    try:
        with open(gnss_file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    frame_id = int(match.group(1))
                    frame_to_ts[frame_id] = float(match.group(2))
    except Exception: return None
    return frame_to_ts

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

# ==========================================
# 3. 解析ロジック
# ==========================================
def analyze_sequence_detailed(seq: SequenceDir, threshold_kmh, min_duration):
    if not seq.gnss_file.exists() or not seq.dvs_dir.exists():
        return None

    # 1. GNSS解析
    gnss_traj = MotionAnalyzer.parse_gnss_trajectory(seq.gnss_file)
    if gnss_traj is None or len(gnss_traj) < 2:
        return None

    static_mask, avg_spd = MotionAnalyzer.get_static_mask(gnss_traj, threshold_kmh, min_duration)
    
    # 2. 同期情報
    frame_map_sec = parse_gnss_timestamps(seq.gnss_file)
    if not frame_map_sec:
        return None

    # 3. ラベル走査
    label_files = sorted(list(seq.dvs_dir.glob("dvs-*.txt")), key=lambda p: natural_sort_key(p.name))
    
    seq_total_counts = Counter()
    seq_removed_counts = Counter()

    valid_files = []
    file_timestamps = []
    
    for txt_file in label_files:
        match = re.search(r"dvs-(\d+)\.txt", txt_file.name)
        if match:
            fid = int(match.group(1))
            if fid in frame_map_sec:
                valid_files.append(txt_file)
                file_timestamps.append(frame_map_sec[fid])

    if not valid_files:
        return None

    # 一括補間
    interp_vals = np.interp(file_timestamps, gnss_traj['timestamp'], static_mask.astype(float))
    is_static_frame = interp_vals > 0.5

    # クラスごとカウント
    for i, txt_file in enumerate(valid_files):
        is_static = is_static_frame[i]
        
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if not parts: continue
                obj_type = parts[0].lower()
                
                seq_total_counts[obj_type] += 1
                if is_static:
                    seq_removed_counts[obj_type] += 1

    total_boxes = sum(seq_total_counts.values())
    removed_boxes = sum(seq_removed_counts.values())

    return {
        "seq_id": seq.root.name,
        "town": seq.root.parent.name,
        "avg_speed": avg_spd,
        "total_counts": seq_total_counts,
        "removed_counts": seq_removed_counts,
        "total_boxes": total_boxes,
        "removed_boxes": removed_boxes
    }

def print_detailed_stats(stats_list, threshold_kmh, min_duration):
    if not stats_list:
        print("No valid sequences found.")
        return

    print("\n" + "="*95)
    print(f" MOTION FILTER SIMULATION (Threshold: < {threshold_kmh} km/h, Duration: >= {min_duration} s)")
    print("-" * 95)
    print(f"{'Town/Seq':<30} | {'Class':<12} | {'Total':<8} | {'Removed':<8} | {'Ratio':<8}")
    print("-" * 95)

    grand_total_counts = Counter()
    grand_removed_counts = Counter()

    for s in stats_list:
        grand_total_counts.update(s['total_counts'])
        grand_removed_counts.update(s['removed_counts'])
        
        seq_name = f"{s['town']}/{s['seq_id']}"
        
        # まず全体の集計行を表示
        total = s['total_boxes']
        removed = s['removed_boxes']
        ratio = (removed / total * 100) if total > 0 else 0.0
        
        # 全体行の表示
        print(f"{seq_name:<30} | {'[ALL]':<12} | {total:<8} | {removed:<8} | {ratio:>6.1f}%")

        # クラスごとの内訳を表示
        # そのシーケンスに存在するクラスのみを表示
        classes = sorted(s['total_counts'].keys())
        
        for cls in classes:
            c_total = s['total_counts'][cls]
            c_removed = s['removed_counts'][cls]
            c_ratio = (c_removed / c_total * 100) if c_total > 0 else 0.0
            
            c_ratio_str = f"{c_ratio:>6.1f}%"
            # 削除率が高い場合は強調
            if c_ratio > 80:
                c_ratio_str += " !!"
            elif c_ratio > 0:
                c_ratio_str += "  "

            # シーケンス名は空欄にして見やすくする
            print(f"{'':<30} | {cls:<12} | {c_total:<8} | {c_removed:<8} | {c_ratio_str}")
        
        # シーケンス間の区切り線
        print("-" * 95)

    print("=" * 95)
    print(" SUMMARY (Global Class-wise)")
    print("-" * 95)
    print(f"{'Class Name':<15} | {'Total':<10} | {'Removed':<10} | {'Kept':<10} | {'Removal Rate'}")
    print("-" * 95)

    all_classes = sorted(grand_total_counts.keys())
    for cls in all_classes:
        total = grand_total_counts[cls]
        removed = grand_removed_counts[cls]
        kept = total - removed
        rate = (removed / total * 100) if total > 0 else 0.0
        print(f"{cls:<15} | {total:<10} | {removed:<10} | {kept:<10} | {rate:>6.1f}%")
    
    print("=" * 95)

# ==========================================
# 4. メイン探索ロジック
# ==========================================
def process_dataset(root_dir: Path, args):
    town_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and "Town" in d.name])

    if not town_dirs:
        print(f"No 'Town' directories found in {root_dir}")
        return

    all_stats = []
    pbar = tqdm(total=0, desc="Analyzing Sequences")

    for town in town_dirs:
        part_dirs = sorted([d for d in town.iterdir() if d.is_dir() and d.name.isdigit()])
        
        if not part_dirs:
            seq = SequenceDir(town)
            res = analyze_sequence_detailed(seq, args.threshold, args.duration)
            if res: all_stats.append(res)
            continue

        for part in part_dirs:
            seq = SequenceDir(part)
            res = analyze_sequence_detailed(seq, args.threshold, args.duration)
            if res:
                all_stats.append(res)
                pbar.set_description(f"Analyzing {town.name}/{part.name}")
                pbar.update(1)

    pbar.close()
    print_detailed_stats(all_stats, args.threshold, args.duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detailed per-sequence, per-class debug tool.")
    parser.add_argument("input_dir", type=str, help="Dataset Root Directory")
    parser.add_argument("--threshold", type=float, default=5.0, help="Simulate Speed threshold in km/h")
    parser.add_argument("--duration", type=float, default=1.0, help="Simulate Min duration in seconds")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
    else:
        process_dataset(input_path, args)