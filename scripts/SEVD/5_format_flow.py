#!/usr/bin/env python3
import argparse
import re
import numpy as np
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm

from utils.directory import SequenceDir

# ==========================================
# 1. Helper Functions
# ==========================================
def parse_gnss_timestamps(gnss_file_path: Path):
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
    try:
        with np.load(npz_path) as data:
            for key in ['flow', 'arr_0', 'data']:
                if key in data:
                    return data[key]
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
    output_dir = seq.root / "optical_flow_processed"
    suffix = "_ds.h5" if args.downsample else ".h5"
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
    
    # 処理対象リスト作成
    target_list = []
    for npz_file in flow_files:
        try:
            frame_id = int(npz_file.stem)
        except ValueError:
            continue
            
        if frame_id in frame_map:
            ts_us = frame_map[frame_id]
            target_list.append((ts_us, npz_file))
            
    if not target_list:
        tqdm.write(f"🚫 [Warn] No valid flow data found for {seq.root.name}")
        return

    target_list.sort(key=lambda x: x[0])
    num_frames = len(target_list)

    # 形状決定
    _, first_path = target_list[0]
    sample_flow = load_flow_from_npz(first_path)
    if sample_flow is None:
        return

    orig_h, orig_w, c = sample_flow.shape
    
    if args.downsample:
        final_h, final_w = orig_h // 2, orig_w // 2
    else:
        final_h, final_w = orig_h, orig_w

    # HDF5書き込み
    desc = f"Writing H5 ({'Half' if args.downsample else 'Full'}) {seq.root.name}"
    compression_args = {"compression": "gzip", "compression_opts": 4}
    
    try:
        with h5py.File(str(output_path), 'w') as h5f:
            dset_flow = h5f.create_dataset(
                'flow', 
                shape=(num_frames, final_h, final_w, 2), 
                dtype='float32', 
                chunks=(1, final_h, final_w, 2),
                **compression_args
            )
            
            dset_ts = h5f.create_dataset(
                'timestamps', 
                shape=(num_frames,), 
                dtype='int64'
            )

            for i, (ts_us, npz_file) in enumerate(tqdm(target_list, desc=desc, leave=False)):
                flow_data = load_flow_from_npz(npz_file)
                
                if flow_data is None:
                    flow_data = np.zeros((orig_h, orig_w, 2), dtype=np.float32)
                
                if args.downsample:
                    flow_data = cv2.resize(flow_data, (final_w, final_h), interpolation=cv2.INTER_AREA)
                    flow_data = flow_data * 0.5
                
                dset_flow[i] = flow_data
                dset_ts[i] = ts_us

        size_mb = output_path.stat().st_size / (1024 * 1024)
        tqdm.write(f"✅ Saved: {output_path.name} | Frames: {num_frames} | Size: {size_mb:.2f} MB")

    except Exception as e:
        tqdm.write(f"❌ Error writing H5 for {seq.root.name}: {e}")
        if output_path.exists():
            output_path.unlink()

# ==========================================
# 3. Main Loop
# ==========================================
def process_dataset(root_dir: Path, args):
    town_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and "Town" in d.name])

    if not town_dirs:
        print(f"No 'Town' directories found in {root_dir}")
        return

    print(f"Options: Downsample={args.downsample} (Output format: HDF5)")

    for town in tqdm(town_dirs, desc="Total Progress"):
        part_dirs = sorted([d for d in town.iterdir() if d.is_dir() and d.name.isdigit()])
        
        if not part_dirs:
            seq = SequenceDir(town)
            aggregate_optical_flow(seq, args)
            continue

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