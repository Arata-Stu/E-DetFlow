#!/usr/bin/env python3
import argparse
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.directory import SequenceDir

# ==========================================
# 1. 設定 & 定数
# ==========================================
CLASS_MAP = {
    'car': 0, 'van': 0, 'truck': 0,
    'pedestrian': 1,
    'cyclist': 2, 'bicycle': 2, 'motorcycle': 2,
    'misc': 3,
}

# Prophesee形式の構造化配列定義
DTYPE = np.dtype([
    ('t', '<i8'),       # timestamp (us)
    ('x', '<f4'),       # bbox x (left)
    ('y', '<f4'),       # bbox y (top)
    ('w', '<f4'),       # width
    ('h', '<f4'),       # height
    ('class_id', '<u1') # class id
])

GNSS_FULL_DTYPE = np.dtype([
    ('timestamp', '<f8'), # sec
    ('lat', '<f8'),
    ('lon', '<f8')
])

# ==========================================
# 2. モーション解析クラス 
# ==========================================
class MotionAnalyzer:
    @staticmethod
    def parse_gnss_trajectory(file_path: Path):
        """速度計算用に緯度経度のみを抽出"""
        if not file_path.exists(): return None
        data = []
        # frame=..., timestamp=..., lat=..., lon=..., alt=...
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
        R = 6371000.0 # 地球半径 (m)
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
        d_time[d_time == 0] = 1e-6 # ゼロ除算防止
        
        speed_kmh = (d_dist / d_time) * 3.6
        # サイズ合わせのため先頭に0を追加
        return np.concatenate(([0], speed_kmh))

    @staticmethod
    def get_static_mask(gnss_data, threshold_kmh, min_duration_sec):
        """速度と継続時間から静止区間のマスク(True=静止)を作成"""
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
                
        return final_static_mask

# ==========================================
# 3. ヘルパー関数: パース処理
# ==========================================
def parse_gnss_timestamps(gnss_file_path: Path):
    """
    ラベル同期用: frame -> timestamp (microseconds) の辞書を作成
    """
    frame_to_ts = {}
    pattern = re.compile(r"frame=(\d+),\s*timestamp=([0-9.]+)")
    
    try:
        with open(gnss_file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    frame_id = int(match.group(1))
                    timestamp_sec = float(match.group(2))
                    timestamp_us = int(timestamp_sec * 1e6)
                    frame_to_ts[frame_id] = timestamp_us
    except Exception as e:
        tqdm.write(f"    [Error] GNSS読み込みエラー: {gnss_file_path.name}: {e}")
        return None
        
    return frame_to_ts

def parse_kitti_line(line, timestamp_us, misc_counter):
    """KITTI形式の1行をパースし、Occludedフィルタを適用"""
    parts = line.strip().split(' ')
    # --- Occlusion Filter ---
    try:
        occluded = int(parts[2])
        # occluded 2 (largely occluded) または 3 (unknown) は除外
        if occluded in [2, 3]:
            return None
    except (IndexError, ValueError):
        pass
    # ------------------------

    obj_type = parts[0].lower()
    
    class_id = CLASS_MAP.get(obj_type, CLASS_MAP['misc'])
    
    if class_id == 3: # misc
        misc_counter['count'] += 1
        misc_counter['types'].add(obj_type)

    bbox_xmin = float(parts[4])
    bbox_ymin = float(parts[5])
    bbox_xmax = float(parts[6])
    bbox_ymax = float(parts[7])

    x = bbox_xmin
    y = bbox_ymin
    w = bbox_xmax - bbox_xmin
    h = bbox_ymax - bbox_ymin

    return (timestamp_us, x, y, w, h, class_id)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

# ==========================================
# 4. コアロジック: ラベル生成 (SequenceDir対応)
# ==========================================
def create_labels_from_sequence(seq: SequenceDir, args):
    """
    SequenceDir を受け取り、root/labels フォルダ内にラベルnpyを生成
    args.filter_static がTrueなら静止区間を削除する
    """
    dvs_dir = seq.dvs_dir
    labels_dir = seq.root / "labels"
    
    # 出力ファイル名の決定
    filename = "labels_bbox_filtered.npy" if args.filter_static else "labels_bbox.npy"
    output_path = labels_dir / filename
    
    if not dvs_dir.exists():
        return
    
    labels_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return

    gnss_path = seq.gnss_file
    if not gnss_path.exists():
        return

    # ラベルファイルリスト取得
    label_files = sorted(list(dvs_dir.glob("dvs-*.txt")), key=lambda p: natural_sort_key(p.name))
    if not label_files:
        return

    # 1. GNSS読み込み (同期用)
    frame_map = parse_gnss_timestamps(gnss_path)
    if not frame_map:
        return

    # 2. KITTIラベルパース & リスト化
    all_labels = []
    misc_stats = {'count': 0, 'types': set()}
    missing_ts_count = 0

    for txt_file in label_files:
        match = re.search(r"dvs-(\d+)\.txt", txt_file.name)
        if not match: continue
        
        frame_id = int(match.group(1))
        if frame_id not in frame_map:
            missing_ts_count += 1
            continue
            
        ts_us = frame_map[frame_id]

        with open(txt_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                label_data = parse_kitti_line(line, ts_us, misc_stats)
                if label_data is None: continue
                all_labels.append(label_data)

    if not all_labels:
        return

    # 3. NumPy配列化
    structured_array = np.array(all_labels, dtype=DTYPE)
    structured_array.sort(order='t')

    # ==========================================
    # ★ フィルタリング処理 (Optional)
    # ==========================================
    original_count = len(structured_array)
    
    if args.filter_static:
        # GNSS軌跡データを読み込み
        gnss_traj = MotionAnalyzer.parse_gnss_trajectory(gnss_path)
        
        if gnss_traj is not None and len(gnss_traj) > 1:
            # 静止マスクを作成 (GNSS時間軸)
            static_mask_gnss = MotionAnalyzer.get_static_mask(
                gnss_traj, args.threshold, args.duration
            )
            
            # ラベルの時間軸に補間 (0=Moving, 1=Static)
            label_ts_sec = structured_array['t'].astype(np.float64) / 1e6
            interp_static = np.interp(label_ts_sec, gnss_traj['timestamp'], static_mask_gnss.astype(float))
            
            # 0.5以上を静止とみなして削除
            is_static_label = interp_static > 0.5
            structured_array = structured_array[~is_static_label]

    # 保存処理
    np.save(str(output_path), structured_array)

    # 結果レポート
    filtered_count = len(structured_array)
    status_str = "Filtered" if args.filter_static else "Raw"
    
    msg = f"  ✅ Saved [{status_str}]: {output_path.name} ({filtered_count} labels)"
    
    if args.filter_static:
        removed = original_count - filtered_count
        if removed > 0:
            msg += f" [Removed {removed} static]"

    if misc_stats['count'] > 0:
        msg += f" (Misc: {misc_stats['count']})"
    if missing_ts_count > 0:
        msg += f" [Warn] {missing_ts_count} frames missing timestamps."
    
    tqdm.write(msg)


# ==========================================
# 5. 探索ロジック
# ==========================================
def process_dataset(root_dir: Path, args):
    town_dirs = sorted([
        d for d in root_dir.iterdir() 
        if d.is_dir() and "Town" in d.name
    ])

    if not town_dirs:
        print(f"No 'Town' directories found in {root_dir}")
        return

    print(f"Found {len(town_dirs)} scenes.")
    if args.filter_static:
        print(f"Filter ON: Removing stops (> {args.duration}s, < {args.threshold}km/h)")

    for town in tqdm(town_dirs, desc="Scenes"):
        part_dirs = sorted([
            d for d in town.iterdir() 
            if d.is_dir() and d.name.isdigit()
        ])
        
        if not part_dirs:
            seq = SequenceDir(town)
            if seq.dvs_dir.exists():
                create_labels_from_sequence(seq, args)
            continue

        for part in part_dirs:
            seq = SequenceDir(part)
            create_labels_from_sequence(seq, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Dataset Root Directory")
    
    # フィルタリング用引数
    parser.add_argument("--filter_static", action="store_true", help="Enable static object filtering based on GNSS speed")
    parser.add_argument("--threshold", type=float, default=0.0, help="Speed threshold in km/h (default: 5.0)")
    parser.add_argument("--duration", type=float, default=1.0, help="Min duration in seconds to consider static (default: 1.0)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
    else:
        process_dataset(input_path, args)
        print("\nDone.")