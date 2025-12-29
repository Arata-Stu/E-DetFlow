#!/usr/bin/env python3
import argparse
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

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
        pattern = re.compile(r"timestamp=([0-9.]+),\s*lat=([0-9.-]+),\s*lon=([0-9.-]+)")
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        data.append((
                            float(match.group(1)),
                            float(match.group(2)),
                            float(match.group(3))
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
            if (t_end - t_start) >= min_duration_sec:
                final_static_mask[s:e] = True
        return final_static_mask

# ==========================================
# 3. ヘルパー関数: パース処理
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
                    timestamp_us = int(float(match.group(2)) * 1e6)
                    frame_to_ts[frame_id] = timestamp_us
    except Exception as e:
        tqdm.write(f"    [Error] GNSS読み込みエラー: {gnss_file_path.name}: {e}")
        return None
    return frame_to_ts

def parse_kitti_line(line, timestamp_us, misc_counter):
    parts = line.strip().split(' ')
    try:
        occluded = int(parts[2])
        if occluded in [2, 3]: return None
    except (IndexError, ValueError): pass

    obj_type = parts[0].lower()
    class_id = CLASS_MAP.get(obj_type, CLASS_MAP['misc'])
    if class_id == 3:
        misc_counter['count'] += 1
        misc_counter['types'].add(obj_type)

    bbox_xmin, bbox_ymin = float(parts[4]), float(parts[5])
    bbox_xmax, bbox_ymax = float(parts[6]), float(parts[7])
    return (timestamp_us, bbox_xmin, bbox_ymin, bbox_xmax - bbox_xmin, bbox_ymax - bbox_ymin, class_id)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def _print_debug_info(root_path, frame_map, missing_frames):
    gnss_ids = sorted(frame_map.keys())
    min_gnss, max_gnss = (min(gnss_ids), max(gnss_ids)) if gnss_ids else ("None", "None")
    tqdm.write(f"  📂 Location : {root_path}")
    tqdm.write(f"  ❌ Missing  : {len(missing_frames)} frames")
    tqdm.write(f"  🔍 Details  : First 5 missing IDs -> {missing_frames[:5]} ...")
    tqdm.write(f"  📡 GNSS Data: Range [{min_gnss} ~ {max_gnss}] (Total {len(gnss_ids)} records)")
    tqdm.write("-" * 60)

# ==========================================
# 4. コアロジック: ラベル生成
# ==========================================
def create_labels_from_sequence(seq: SequenceDir, args, output_base: Path = None):
    dvs_dir = seq.dvs_dir
    
    # --- 出力ディレクトリの決定 ---
    if output_base:
        # 入力ルートからの相対パスを取得して出力先に結合
        rel_path = seq.root.relative_to(Path(args.input_dir))
        labels_dir = output_base / rel_path / "labels"
    else:
        labels_dir = seq.root / "labels"
    
    filename = "labels_bbox_filtered.npy" if args.filter_static else "labels_bbox.npy"
    output_path = labels_dir / filename
    
    if not dvs_dir.exists(): return
    if output_path.exists(): return

    gnss_path = seq.gnss_file
    if not gnss_path.exists():
        tqdm.write(f"[Skip] GNSS file missing in: {seq.root}")
        return

    label_files = sorted(list(dvs_dir.glob("dvs-*.txt")), key=lambda p: natural_sort_key(p.name))
    if not label_files: return

    frame_map = parse_gnss_timestamps(gnss_path)
    if not frame_map: return

    all_labels = []
    misc_stats = {'count': 0, 'types': set()}
    missing_frames = [] 

    for txt_file in label_files:
        match = re.search(r"dvs-(\d+)\.txt", txt_file.name)
        if not match: continue
        frame_id = int(match.group(1))
        if frame_id not in frame_map:
            missing_frames.append(frame_id)
            continue
        ts_us = frame_map[frame_id]
        with open(txt_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                label_data = parse_kitti_line(line, ts_us, misc_stats)
                if label_data: all_labels.append(label_data)

    if not all_labels:
        if missing_frames: _print_debug_info(seq.root, frame_map, missing_frames)
        return

    structured_array = np.array(all_labels, dtype=DTYPE)
    structured_array.sort(order='t')

    # 静止フィルタ
    if args.filter_static:
        gnss_traj = MotionAnalyzer.parse_gnss_trajectory(gnss_path)
        if gnss_traj is not None and len(gnss_traj) > 1:
            static_mask = MotionAnalyzer.get_static_mask(gnss_traj, args.threshold, args.duration)
            label_ts_sec = structured_array['t'].astype(np.float64) / 1e6
            interp_static = np.interp(label_ts_sec, gnss_traj['timestamp'], static_mask.astype(float))
            structured_array = structured_array[interp_static <= 0.5]

    if len(structured_array) > 0:
        labels_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), structured_array)
        tqdm.write(f"Saved: {output_path}")

# ==========================================
# 5. 探索ロジック
# ==========================================
def process_dataset(root_dir: Path, args):
    # --- YAML設定からパスを取得 ---
    try:
        conf = OmegaConf.load(args.config)
    except Exception as e:
        print(f"Error loading config: {e}"); return

    rel_paths = []
    for split in conf.keys():
        if conf[split] is not None:
            rel_paths.extend(list(conf[split]))
    unique_rel_paths = list(dict.fromkeys(rel_paths))

    output_base = Path(args.output_dir) if args.output_dir else None

    target_sequences = []
    for rel_p in unique_rel_paths:
        full_p = root_dir / rel_p
        if full_p.exists() and full_p.is_dir():
            seq = SequenceDir(full_p)
            if seq.dvs_dir.exists():
                target_sequences.append(seq)
    
    if not target_sequences:
        print("No valid sequences found."); return

    print(f"Found {len(target_sequences)} sequences from config.")
    if args.filter_static:
        print(f"Filter ON: (> {args.duration}s, < {args.threshold}km/h)")

    for seq in tqdm(target_sequences, desc="BBox"):
        create_labels_from_sequence(seq, args, output_base)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--config", type=str, required=True, help="YAML split config") # 追加
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--filter_static", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=1.0)
    
    args = parser.parse_args()
    input_path = Path(args.input_dir)
    if input_path.exists():
        process_dataset(input_path, args)
        print("\nDone.")