#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import numba
import multiprocessing as mp
from omegaconf import OmegaConf

from utils.directory import SequenceDir

# ==========================================
# 1. Numba JIT 関数 (ダウンサンプル計算用)
# ==========================================
@numba.jit(nopython=True, cache=True)
def _filter_events_resize_jit(x, y, p, mask, change_map, fx, fy):
    for i in range(len(x)):
        x_l = x[i] // fx
        y_l = y[i] // fy
        if y_l >= change_map.shape[0] or x_l >= change_map.shape[1]:
            continue
        change_map[y_l, x_l] += p[i] * 1.0 / (fx * fy)
        if np.abs(change_map[y_l, x_l]) >= 1:
            mask[i] = True
            change_map[y_l, x_l] -= p[i]
    return mask

# ==========================================
# 2. ダウンサンプル・ラッパー関数
# ==========================================
def apply_downsampling_half(x, y, p, t, width, height):
    fx, fy = 2, 2
    out_w, out_h = width // fx, height // fy
    change_map = np.zeros((out_h, out_w), dtype="float32")
    mask = np.zeros(len(x), dtype="bool")
    p_signed = 2 * p.astype(np.int8) - 1
    _filter_events_resize_jit(x, y, p_signed, mask, change_map, fx, fy)

    x_new = (x[mask] // fx).astype(x.dtype)
    y_new = (y[mask] // fy).astype(y.dtype)
    p_new = ((p_signed[mask] + 1) // 2).astype(np.uint8)
    return x_new, y_new, p_new, t[mask]

# ==========================================
# 3. ヘルパー関数
# ==========================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

# ==========================================
# 4. 変換ロジック (ワーカー用)
# ==========================================
def convert_single_sequence(dir_path: Path, args):
    seq = SequenceDir(dir_path)
    output_filename = "events_ds.h5" if args.downsample else "events.h5"

    # --- 出力パスの決定 ---
    if args.output_dir:
        # 入力ルートからの相対パスを取得して出力ルートに結合
        rel_path = dir_path.relative_to(Path(args.input_dir))
        output_dir = Path(args.output_dir) / rel_path / "events"
    else:
        output_dir = dir_path / "events"
    
    output_path = output_dir / output_filename
    
    if not seq.dvs_dir.exists(): return None 
    if output_path.exists(): return f"Skipped: {seq.root.name}"

    npz_files = sorted(list(seq.dvs_dir.glob("*.npz")), key=lambda p: natural_sort_key(p.name))
    if not npz_files: return None

    all_x, all_y, all_t, all_p = [], [], [], []
    for file_path in npz_files:
        try:
            with np.load(file_path) as data:
                key = 'dvs_events' if 'dvs_events' in data else 'events'
                if key in data:
                    all_x.append(data[key]['x'])
                    all_y.append(data[key]['y'])
                    all_t.append(data[key]['t'])
                    all_p.append(data[key]['pol']) 
        except Exception: pass

    if not all_x: return None

    x_full, y_full, t_full, p_full = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_t), np.concatenate(all_p)

    if args.downsample:
        x_full, y_full, p_full, t_full = apply_downsampling_half(
            x_full.astype(np.int32), y_full.astype(np.int32), p_full, t_full, args.width, args.height
        )

    t_micro = (t_full // 1000).astype(np.uint64)
    
    # 書き出し
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        grp = f.create_group("events")
        grp.create_dataset("x", data=x_full.astype(np.uint16), dtype='uint16')
        grp.create_dataset("y", data=y_full.astype(np.uint16), dtype='uint16')
        grp.create_dataset("t", data=t_micro, dtype='uint64')
        grp.create_dataset("p", data=p_full.astype(np.uint8), dtype='uint8')
        
        f_w, f_h = (args.width // 2, args.height // 2) if args.downsample else (args.width, args.height)
        grp.attrs['width'], grp.attrs['height'] = f_w, f_h

    return f"Saved: {seq.root.name} (Events: {len(t_micro)})"

# ==========================================
# 5. マルチプロセス用ラッパー
# ==========================================
def _worker_task(payload):
    path, args = payload
    try:
        return convert_single_sequence(path, args)
    except Exception as e:
        return f"Error in {path.name}: {e}"

# ==========================================
# 6. メイン探索ロジック
# ==========================================
def process_dataset(args):
    root_dir = Path(args.input_dir)
    
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

    target_paths = []
    print("Scanning directories based on config...")
    for rel_p in unique_rel_paths:
        full_p = root_dir / rel_p
        if full_p.exists() and full_p.is_dir():
            seq = SequenceDir(full_p)
            if seq.dvs_dir.exists():
                target_paths.append(full_p)
        else:
            print(f"[Skip] Path not found: {full_p}")

    total_tasks = len(target_paths)
    if total_tasks == 0:
        print("No valid sequences found."); return

    print(f"Total: {total_tasks} | Output: {args.output_dir or 'Same as input'}")
    
    # マルチプロセス実行
    task_args = [(p, args) for p in target_paths]
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.num_workers if args.num_workers > 0 else None) as pool:
        results = list(tqdm(pool.imap_unordered(_worker_task, task_args), total=total_tasks, desc="Events"))

    # Summary
    saved = sum(1 for r in results if r and r.startswith("Saved"))
    print(f"\nSummary: Processed {saved} sequences.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--config", type=str, required=True, help="YAML split config") # 追加
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--downsample", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--num_workers", type=int, default=0)
    
    args = parser.parse_args()
    if Path(args.input_dir).exists():
        process_dataset(args)