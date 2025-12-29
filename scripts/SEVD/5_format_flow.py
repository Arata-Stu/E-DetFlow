#!/usr/bin/env python3
import argparse
import re
import numpy as np
import cv2
import h5py
import hdf5plugin
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from functools import partial
from omegaconf import OmegaConf

from utils.directory import SequenceDir
from utils.preprocessing import _blosc_opts  

# --- 定数設定 ---
COLORS_RGB = {
    'Sky':        (70, 130, 180),
    'Vegetation': (107, 142, 35),
    'Water':      (45, 60, 150)
}
IGNORE_LABELS = ['Sky']


def parse_gnss_timestamps(gnss_file_path: Path):
    frame_to_ts = {}
    pattern = re.compile(r"frame=(\d+),\s*timestamp=([0-9.]+)")
    if not gnss_file_path.exists(): return None
    try:
        with open(gnss_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    frame_id = int(match.group(1))
                    timestamp_us = int(float(match.group(2)) * 1e6)
                    frame_to_ts[frame_id] = timestamp_us
    except Exception: return None
    return frame_to_ts

def load_flow_from_npz(npz_path: Path):
    try:
        with np.load(npz_path) as data:
            for key in ['flow', 'arr_0', 'data']:
                if key in data: return data[key]
            keys = list(data.keys())
            if keys: return data[keys[0]]
    except Exception: pass
    return None

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def generate_mask_with_debug(flow, sem_seg_path, height, width, max_flow=400, min_flow=0.1):
    stats = {"too_small": 0, "too_large": 0, "oob": 0, "semantics": 0}
    mag = np.linalg.norm(flow, axis=2)
    mask_small, mask_large = mag < min_flow, mag >= max_flow
    stats["too_small"], stats["too_large"] = np.sum(mask_small), np.sum(mask_large)
    mask_valid = ~(mask_small | mask_large)
    
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    dest_x, dest_y = x_grid + flow[..., 0], y_grid + flow[..., 1]
    mask_oob = (dest_x < 0) | (dest_x >= width) | (dest_y < 0) | (dest_y >= height)
    stats["oob"] = np.sum(mask_oob & mask_valid)
    mask_valid &= (~mask_oob)

    if sem_seg_path and sem_seg_path.exists():
        sem_img = cv2.imread(str(sem_seg_path), cv2.IMREAD_COLOR)
        if sem_img is not None:
            if sem_img.shape[:2] != (height, width):
                sem_img = cv2.resize(sem_img, (width, height), interpolation=cv2.INTER_NEAREST)
            for label in IGNORE_LABELS:
                if label in COLORS_RGB:
                    is_target = np.all(sem_img == COLORS_RGB[label][::-1], axis=2)
                    stats["semantics"] += np.sum(is_target & mask_valid)
                    mask_valid &= (~is_target)
    return mask_valid.astype(np.uint8), stats, mag.max()

# --- メイン処理関数 ---

def aggregate_optical_flow(seq_path: Path, args):
    seq = SequenceDir(seq_path)
    
    # --- 【修正】出力ディレクトリの決定ロジック ---
    if args.output_dir:
        # 入力ルートからの相対パスを取得して出力ルートに結合
        rel_path = seq.root.relative_to(Path(args.input_dir))
        output_dir = Path(args.output_dir) / rel_path / "optical_flow_processed"
    else:
        output_dir = seq.root / "optical_flow_processed"
    
    suffix = "_ds.h5" if args.downsample else ".h5"
    final_path = output_dir / f"optical_flow_synced{suffix}"
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    if final_path.exists(): return f"[Skip] {seq.root.name}"
    if not seq.optical_flow_dir.exists(): return f"[Error] No flow dir: {seq.root.name}"

    frame_map = parse_gnss_timestamps(seq.gnss_file)
    if not frame_map: return f"[Error] GNSS invalid: {seq.root.name}"

    flow_files = sorted(list(seq.optical_flow_dir.glob("*.npz")), key=lambda p: natural_sort_key(p.name))
    target_list = []
    for npz_file in flow_files:
        try:
            f_id = int(npz_file.stem)
            if f_id in frame_map:
                target_list.append((frame_map[f_id], npz_file, seq.sem_seg_dir / f"{f_id}.png"))
        except ValueError: continue
    
    if not target_list: return f"[Warn] No valid data: {seq.root.name}"

    num_frames = len(target_list)
    output_dir.mkdir(parents=True, exist_ok=True) # parents=Trueにして親階層も作成
    
    sample_flow = load_flow_from_npz(target_list[0][1])
    orig_h, orig_w = sample_flow.shape[:2]
    if args.orig_size: orig_w, orig_h = args.orig_size

    final_h, final_w = (orig_h // 2, orig_w // 2) if args.downsample else (orig_h, orig_w)
    agg_stats = {"too_small": 0, "too_large": 0, "oob": 0, "semantics": 0}
    max_mag_in_seq = 0.0

    try:
        with h5py.File(str(tmp_path), 'w') as h5f:
            d_flow = h5f.create_dataset('flow', (num_frames, final_h, final_w, 2), dtype='f4',
                                        chunks=(1, final_h, final_w, 2), **_blosc_opts(complevel=1, shuffle='byte'))
            d_valid = h5f.create_dataset('valid', (num_frames, final_h, final_w), dtype='u1',
                                         chunks=(1, final_h, final_w), **_blosc_opts(complevel=1, shuffle='byte'))
            d_ts = h5f.create_dataset('timestamps', (num_frames,), dtype='i8')

            for i, (ts, npz_p, sem_p) in enumerate(target_list):
                data = load_flow_from_npz(npz_p)
                if data is None: data = np.zeros((orig_h, orig_w, 2), dtype=np.float32)
                data[..., 0] *= orig_w
                data[..., 1] *= orig_h
                if args.downsample:
                    data = cv2.resize(data, (final_w, final_h), interpolation=cv2.INTER_AREA) * 0.5
                
                mask, frame_stats, frame_max_mag = generate_mask_with_debug(data, sem_p, final_h, final_w)
                for k in agg_stats: agg_stats[k] += frame_stats[k]
                max_mag_in_seq = max(max_mag_in_seq, frame_max_mag)
                d_flow[i], d_valid[i], d_ts[i] = data, mask, ts

        tmp_path.rename(final_path)
        total_px = num_frames * final_h * final_w
        valid_pct = max(0, (total_px - sum(agg_stats.values())) / total_px * 100)
        return f"✅ {seq.root.name} -> {final_path.name} | Valid: {valid_pct:.1f}%"

    except Exception as e:
        if tmp_path.exists(): tmp_path.unlink()
        return f"❌ Failed: {seq.root.name} ({e})"

def process_dataset_parallel(root_dir: Path, args):
    try: conf = OmegaConf.load(args.config)
    except Exception as e:
        print(f"Error loading config: {e}"); return

    rel_paths = []
    for split in conf.keys():
        if conf[split] is not None: rel_paths.extend(list(conf[split]))
    unique_rel_paths = list(dict.fromkeys(rel_paths))

    all_seq_paths = []
    for rel_p in unique_rel_paths:
        full_p = root_dir / rel_p
        if full_p.exists() and full_p.is_dir(): all_seq_paths.append(full_p)
    
    if not all_seq_paths:
        print("No valid sequences found."); return

    print(f"Total: {len(all_seq_paths)} | Output: {args.output_dir or 'Same as input'}")
    ctx = mp.get_context('spawn')
    func = partial(aggregate_optical_flow, args=args)

    with ctx.Pool(processes=args.num_workers) as pool:
        pbar = tqdm(pool.imap_unordered(func, all_seq_paths), total=len(all_seq_paths), desc="Processing")
        for res in pbar: tqdm.write(res)

if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None, help="Output Root Directory") 
    parser.add_argument("--downsample", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--orig_size", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    
    args = parser.parse_args()
    input_path = Path(args.input_dir)
    if input_path.exists():
        process_dataset_parallel(input_path, args)
        print("\nFinished.")