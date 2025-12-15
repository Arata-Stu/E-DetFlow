#!/usr/bin/env python3
import argparse
import re
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from utils.directory import SequenceDir

# ==========================================
# 1. Helper Functions
# ==========================================
def parse_gnss_timestamps(gnss_file_path: Path):
    """GNSSログから frame -> timestamp (microseconds) の辞書を作成"""
    frame_to_ts = {}
    pattern = re.compile(r"frame=(\d+),\s*timestamp=([0-9.]+)")
    
    if not gnss_file_path.exists():
        return None

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
        tqdm.write(f"Error parsing GNSS: {e}")
        return None
        
    return frame_to_ts

def load_flow_from_npz(npz_path: Path):
    """npzファイルからFlowデータをロードする"""
    try:
        with np.load(npz_path) as data:
            # 一般的なキーを探索
            for key in ['flow', 'arr_0', 'data']:
                if key in data:
                    return data[key]
            # キーが見つからない場合、最初のキーを使用
            keys = list(data.keys())
            if keys:
                return data[keys[0]]
    except Exception as e:
        tqdm.write(f"Error loading {npz_path.name}: {e}")
    return None

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

# ==========================================
# 2. Core Logic
# ==========================================
def aggregate_optical_flow(seq: SequenceDir, args):
    """
    Optical Flowを集約し、必要に応じてダウンサンプルを行う
    """
    # 保存先ディレクトリ
    output_dir = seq.root / "optical_flow_processed"
    
    # 出力ファイル名の切り替え
    suffix = "_ds.npy" if args.downsample else ".npy"
    output_path = output_dir / f"optical_flow_synced{suffix}"
    
    if not seq.optical_flow_dir.exists():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        tqdm.write(f"[Skip] Already exists: {output_path.name}")
        return

    # GNSS読み込み
    frame_map = parse_gnss_timestamps(seq.gnss_file)
    if not frame_map:
        tqdm.write(f"[Skip] GNSS missing or invalid: {seq.root.name}")
        return

    # ファイルリスト取得
    flow_files = sorted(list(seq.optical_flow_dir.glob("*.npz")), key=lambda p: natural_sort_key(p.name))
    valid_data = []
    
    desc = f"Processing ({'Half' if args.downsample else 'Full'}) {seq.root.name}"
    
    for npz_file in tqdm(flow_files, desc=desc, leave=False):
        try:
            frame_id = int(npz_file.stem)
        except ValueError:
            continue

        if frame_id not in frame_map:
            continue

        ts_us = frame_map[frame_id]
        flow_data = load_flow_from_npz(npz_file)
        
        if flow_data is None:
            continue
            
        # 形状チェック (H, W, 2)
        if flow_data.ndim != 3 or flow_data.shape[2] != 2:
            continue

        # ダウンサンプル処理
        if args.downsample:
            h, w, _ = flow_data.shape
            new_h, new_w = h // 2, w // 2
            
            # 1. 解像度変更 (Resolution)
            flow_data = cv2.resize(flow_data, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 2. 値のスケーリング (Magnitude)
            flow_data = flow_data * 0.5

        valid_data.append((ts_us, flow_data))

    if not valid_data:
        tqdm.write(f"🚫 [Warn] No valid flow data found for {seq.root.name}")
        return

    # 配列結合
    valid_data.sort(key=lambda x: x[0])
    
    timestamps = np.array([x[0] for x in valid_data], dtype=np.int64)
    flows = np.stack([x[1] for x in valid_data], axis=0).astype(np.float32)

    # 保存
    save_data = {
        'timestamps': timestamps,
        'flow': flows
    }
    
    np.save(str(output_path), save_data)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    tqdm.write(f"✅ Saved: {output_path.name} | Shape: {flows.shape} | Size: {size_mb:.2f} MB")

# ==========================================
# 3. Main Loop
# ==========================================
def process_dataset(root_dir: Path, args):
    town_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and "Town" in d.name])

    if not town_dirs:
        print(f"No 'Town' directories found in {root_dir}")
        return

    print(f"Options: Downsample={args.downsample}")

    for town in tqdm(town_dirs, desc="Total Progress"):
        part_dirs = sorted([d for d in town.iterdir() if d.is_dir() and d.name.isdigit()])
        
        # Partディレクトリがない場合 (Town直下)
        if not part_dirs:
            seq = SequenceDir(town)
            aggregate_optical_flow(seq, args)
            continue

        # Partディレクトリがある場合
        for part in part_dirs:
            seq = SequenceDir(part)
            aggregate_optical_flow(seq, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Dataset Root Directory")
    parser.add_argument("--downsample", action="store_true", 
                        help="Downsample flow by 1/2 (resizes image AND scales flow magnitude)")
    
    args = parser.parse_args()
    input_path = Path(args.input_dir)
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
    else:
        process_dataset(input_path, args)
        print("\nDone.")