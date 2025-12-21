#!/usr/bin/env python3
import argparse
import re
import numpy as np
import cv2
import h5py
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from functools import partial

from utils.directory import SequenceDir

# --- 定数設定 ---
COLORS_RGB = {
    'Sky':        (70, 130, 180),
    'Vegetation': (107, 142, 35),
    'Water':      (45, 60, 150)
}
IGNORE_LABELS = ['Sky']

# --- ヘルパー関数群 ---

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
                    timestamp_us = int(float(match.group(2)) * 1e6)
                    frame_to_ts[frame_id] = timestamp_us
    except Exception as e:
        return None
    return frame_to_ts

def load_flow_from_npz(npz_path: Path):
    try:
        with np.load(npz_path) as data:
            for key in ['flow', 'arr_0', 'data']:
                if key in data: return data[key]
            keys = list(data.keys())
            if keys: return data[keys[0]]
    except Exception:
        pass
    return None

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def generate_mask_with_semantics_color(flow, sem_seg_path, height, width, max_flow=400, min_flow=0.1):
    mag = np.linalg.norm(flow, axis=2)
    mask_valid = (mag < max_flow) & (mag >= min_flow)
    
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    dest_x, dest_y = x_grid + flow[..., 0], y_grid + flow[..., 1]
    mask_oob = (dest_x >= 0) & (dest_x < width) & (dest_y >= 0) & (dest_y < height)
    mask_valid &= mask_oob

    if sem_seg_path and sem_seg_path.exists():
        sem_img = cv2.imread(str(sem_seg_path), cv2.IMREAD_COLOR)
        if sem_img is not None:
            if sem_img.shape[:2] != (height, width):
                sem_img = cv2.resize(sem_img, (width, height), interpolation=cv2.INTER_NEAREST)
            for label in IGNORE_LABELS:
                if label in COLORS_RGB:
                    is_target = np.all(sem_img == COLORS_RGB[label][::-1], axis=2)
                    mask_valid &= (~is_target)
    return mask_valid.astype(np.uint8)

# --- メイン処理関数 ---

def aggregate_optical_flow(seq_path: Path, args):
    """
    1つのシーケンスディレクトリを処理する。
    並列ワーカーから呼び出されるため、内部でのtqdmは避けるか簡略化する。
    """
    seq = SequenceDir(seq_path)
    output_dir = seq.root / "optical_flow_processed"
    suffix = "_ds.h5" if args.downsample else ".h5"
    
    final_path = output_dir / f"optical_flow_synced{suffix}"
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    # 1. 完了チェックとゴミ掃除
    if final_path.exists():
        return f"[Skip] {seq.root.name}"

    if tmp_path.exists():
        tmp_path.unlink() # 途中のファイルがあれば削除して再試行

    if not seq.optical_flow_dir.exists():
        return f"[Error] No flow dir in {seq.root.name}"

    # 2. 前準備（GNSS/ファイルリスト）
    frame_map = parse_gnss_timestamps(seq.gnss_file)
    if not frame_map:
        return f"[Error] GNSS invalid: {seq.root.name}"

    flow_files = sorted(list(seq.optical_flow_dir.glob("*.npz")), key=lambda p: natural_sort_key(p.name))
    target_list = []
    for npz_file in flow_files:
        try:
            f_id = int(npz_file.stem)
            if f_id in frame_map:
                target_list.append((frame_map[f_id], npz_file, seq.sem_seg_dir / f"{f_id}.png"))
        except ValueError: continue
    
    if not target_list:
        return f"[Warn] No valid data: {seq.root.name}"

    target_list.sort(key=lambda x: x[0])
    num_frames = len(target_list)

    # 3. H5ファイル書き出し (Atomic Write)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_flow = load_flow_from_npz(target_list[0][1])
    if sample_flow is None: return f"[Error] Load failed: {seq.root.name}"
    
    orig_h, orig_w = sample_flow.shape[:2]
    final_h, final_w = (orig_h // 2, orig_w // 2) if args.downsample else (orig_h, orig_w)
    comp = {"compression": "gzip", "compression_opts": 4}

    try:
        with h5py.File(str(tmp_path), 'w') as h5f:
            d_flow = h5f.create_dataset('flow', (num_frames, final_h, final_w, 2), dtype='f4', chunks=(1, final_h, final_w, 2), **comp)
            d_valid = h5f.create_dataset('valid', (num_frames, final_h, final_w), dtype='u1', chunks=(1, final_h, final_w), **comp)
            d_ts = h5f.create_dataset('timestamps', (num_frames,), dtype='i8')

            for i, (ts, npz_p, sem_p) in enumerate(target_list):
                data = load_flow_from_npz(npz_p)
                if data is None: data = np.zeros((orig_h, orig_w, 2), dtype=np.float32)
                
                if args.downsample:
                    data = cv2.resize(data, (final_w, final_h), interpolation=cv2.INTER_AREA) * 0.5
                
                mask = generate_mask_with_semantics_color(data, sem_p, final_h, final_w)
                
                d_flow[i], d_valid[i], d_ts[i] = data, mask, ts

        # 正常終了したらリネーム
        tmp_path.rename(final_path)
        return f"✅ Saved: {seq.root.name}"

    except Exception as e:
        if tmp_path.exists(): tmp_path.unlink()
        return f"❌ Failed: {seq.root.name} ({e})"

# --- 並列実行制御 ---

def process_dataset_parallel(root_dir: Path, args):
    town_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and "Town" in d.name])
    
    # 処理対象シーケンスのリストアップ
    all_seq_paths = []
    for town in town_dirs:
        part_dirs = sorted([d for d in town.iterdir() if d.is_dir() and d.name.isdigit()])
        if not part_dirs:
            all_seq_paths.append(town)
        else:
            all_seq_paths.extend(part_dirs)

    print(f"Total sequences: {len(all_seq_paths)} | Downsample: {args.downsample}")
    print(f"Workers: {args.num_workers or mp.cpu_count()} (Method: spawn)")

    ctx = mp.get_context('spawn')
    func = partial(aggregate_optical_flow, args=args)

    results = []
    with ctx.Pool(processes=args.num_workers) as pool:
        # imap_unordered で進捗を表示
        for res in tqdm(pool.imap_unordered(func, all_seq_paths), total=len(all_seq_paths), desc="Processing"):
            if "✅" not in res and "[Skip]" not in res:
                tqdm.write(res) # エラーや警告があれば表示

if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Dataset Root Directory")
    parser.add_argument("--downsample", action="store_true", help="Downsample flow by 1/2")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers (default: CPU count)")
    
    args = parser.parse_args()
    input_path = Path(args.input_dir)
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
    else:
        process_dataset_parallel(input_path, args)
        print("\nFinished.")