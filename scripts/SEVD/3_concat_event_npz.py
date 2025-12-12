#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import numba
import multiprocessing as mp

from utils.directory import SequenceDir

# ==========================================
# 1. Numba JIT 関数 (ダウンサンプル計算用)
# ==========================================
@numba.jit(nopython=True, cache=True)
def _filter_events_resize_jit(x, y, p, mask, change_map, fx, fy):
    """
    JITコンパイルされた積分発火ロジック
    """
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
    out_w = width // fx
    out_h = height // fy
    
    change_map = np.zeros((out_h, out_w), dtype="float32")
    mask = np.zeros(len(x), dtype="bool")

    p_signed = 2 * p.astype(np.int8) - 1
    
    _filter_events_resize_jit(x, y, p_signed, mask, change_map, fx, fy)

    x_new = (x[mask] // fx).astype(x.dtype)
    y_new = (y[mask] // fy).astype(y.dtype)
    t_new = t[mask]
    
    p_filtered_signed = p_signed[mask]
    p_new = ((p_filtered_signed + 1) // 2).astype(np.uint8)

    return x_new, y_new, p_new, t_new

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
    """
    1つのシーケンスディレクトリを処理する関数
    マルチプロセスで呼び出されるため、tqdmの表示は最小限にするか、戻り値で制御します。
    """
    # プロセス内でSequenceDirを初期化 (Pickleエラー回避のためパスで受け取る)
    seq = SequenceDir(dir_path)

    if args.downsample:
        output_filename = "events_ds.h5"
    else:
        output_filename = "events.h5"

    source_dir = seq.dvs_dir
    output_dir = seq.root / "events"
    output_path = output_dir / output_filename
    
    # スキップ条件
    if not source_dir.exists():
        return None 
    if output_path.exists():
        return f"Skipped (Exists): {seq.root.name}"

    npz_files = sorted(list(source_dir.glob("*.npz")), key=lambda p: natural_sort_key(p.name))
    if not npz_files:
        return None

    # データ読み込み処理
    all_x, all_y, all_t, all_p = [], [], [], []

    for file_path in npz_files:
        try:
            with np.load(file_path) as data:
                key = 'dvs_events' if 'dvs_events' in data else 'events'
                if key in data:
                    events = data[key]
                    all_x.append(events['x'])
                    all_y.append(events['y'])
                    all_t.append(events['t'])
                    all_p.append(events['pol']) 
        except Exception:
            pass

    if not all_x:
        return None

    x_full = np.concatenate(all_x)
    y_full = np.concatenate(all_y)
    t_full = np.concatenate(all_t)
    p_full = np.concatenate(all_p)

    # ダウンサンプル処理
    if args.downsample:
        x_full, y_full, p_full, t_full = apply_downsampling_half(
            x_full.astype(np.int32), 
            y_full.astype(np.int32), 
            p_full, 
            t_full, 
            args.width, 
            args.height
        )

    # 保存処理
    t_micro = (t_full // 1000).astype(np.uint64)
    p_uint8 = p_full.astype(np.uint8)
    x_uint16 = x_full.astype(np.uint16)
    y_uint16 = y_full.astype(np.uint16)

    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        grp = f.create_group("events")
        grp.create_dataset("x", data=x_uint16, dtype='uint16')
        grp.create_dataset("y", data=y_uint16, dtype='uint16')
        grp.create_dataset("t", data=t_micro,  dtype='uint64')
        grp.create_dataset("p", data=p_uint8,  dtype='uint8')
        
        if args.downsample:
            grp.attrs['width'] = args.width // 2
            grp.attrs['height'] = args.height // 2
        else:
            grp.attrs['width'] = args.width
            grp.attrs['height'] = args.height

    return f"Saved: {seq.root.name} (Events: {len(t_micro)})"

# ==========================================
# 5. マルチプロセス用ラッパー
# ==========================================
def _worker_task(payload):
    """
    Pool.imap等に渡すためのラッパー関数。
    引数をアンパックして処理関数へ渡す。
    """
    path, args = payload
    try:
        result = convert_single_sequence(path, args)
        return result
    except Exception as e:
        return f"Error in {path.name}: {e}"

# ==========================================
# 6. メイン探索ロジック
# ==========================================
def process_dataset(args):
    root_dir = Path(args.input_dir)
    town_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and "Town" in d.name])

    if not town_dirs:
        print(f"No 'Town' directories found in {root_dir}")
        return

    # 1. 処理対象のディレクトリパスを全て収集する
    target_paths = []
    print("Scanning directories...")
    
    for town in town_dirs:
        part_dirs = sorted([d for d in town.iterdir() if d.is_dir() and d.name.isdigit()])
        
        if not part_dirs:
            # 分割されていない場合 (Town直下)
            seq = SequenceDir(town)
            if seq.dvs_dir.exists():
                target_paths.append(town)
        else:
            # 分割されている場合 (Town/01, Town/02...)
            for part in part_dirs:
                seq = SequenceDir(part)
                if seq.dvs_dir.exists():
                    target_paths.append(part)

    total_tasks = len(target_paths)
    print(f"Found {total_tasks} sequences to process.")
    
    if total_tasks == 0:
        return

    if args.downsample:
        print(f"Option: Downsampling enabled (1/2 scale). Input assume: {args.width}x{args.height}")

    # 2. SpawnコンテキストでPoolを作成して並列処理実行
    num_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 2)
    print(f"Starting parallel processing with {num_workers} workers (spawn method)...")

    # 引数リストの作成 (パスと設定をペアにする)
    task_args = [(p, args) for p in target_paths]

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        # imap_unorderedを使って完了したものから順次処理
        # tqdmで進捗を表示
        results = list(tqdm(
            pool.imap_unordered(_worker_task, task_args),
            total=total_tasks,
            desc="Processing",
            unit="seq"
        ))

    # 3. 結果の要約表示（オプション）
    saved_count = sum(1 for r in results if r and r.startswith("Saved"))
    skipped_count = sum(1 for r in results if r and r.startswith("Skipped"))
    error_count = sum(1 for r in results if r and r.startswith("Error"))
    
    print("\nSummary:")
    print(f"  Processed: {saved_count}")
    print(f"  Skipped:   {skipped_count}")
    print(f"  Errors:    {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Dataset Root Directory")
    
    # オプション
    parser.add_argument("--downsample", action="store_true", help="Enable 1/2 spatial downsampling")
    parser.add_argument("--width", type=int, default=1280, help="Original input width (default: 1280)")
    parser.add_argument("--height", type=int, default=960, help="Original input height (default: 960)")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers (default: CPU_COUNT - 2)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
    else:
        process_dataset(args)
        print("\nDone.")