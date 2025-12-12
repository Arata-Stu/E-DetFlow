#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import numba

from utils.directory import SequenceDir


# ==========================================
# 1. Numba JIT 関数 (ダウンサンプル計算用)
# ==========================================
@numba.jit(nopython=True, cache=True)
def _filter_events_resize_jit(x, y, p, mask, change_map, fx, fy):
    """
    JITコンパイルされた積分発火ロジック
    p は {-1, 1} の符号付き整数または浮動小数点数であること
    """
    for i in range(len(x)):
        x_l = x[i] // fx
        y_l = y[i] // fy
        
        # 配列外参照ガード
        if y_l >= change_map.shape[0] or x_l >= change_map.shape[1]:
            continue

        # 積分 (Integrate)
        # p[i]がfloatならそのまま、intならキャストして計算
        change_map[y_l, x_l] += p[i] * 1.0 / (fx * fy)

        # 発火判定 (Fire)
        # 閾値 1.0 または -1.0 を超えたらイベント通過
        if np.abs(change_map[y_l, x_l]) >= 1:
            mask[i] = True
            change_map[y_l, x_l] -= p[i] # 残差を残してリセット

    return mask

# ==========================================
# 2. ダウンサンプル・ラッパー関数
# ==========================================
def apply_downsampling_half(x, y, p, t, width, height):
    """
    イベントデータを1/2スケールにダウンサンプルする
    input:
      p: bool or {0, 1} array
    output:
      x, y, p, t (p is converted back to {0, 1})
    """
    fx, fy = 2, 2 # 1/2スケール
    out_w = width // fx
    out_h = height // fy
    
    # マップ初期化
    change_map = np.zeros((out_h, out_w), dtype="float32")
    mask = np.zeros(len(x), dtype="bool")

    # Polarity変換: Boolean/uint8 {0, 1} -> Signed int8 {-1, 1}
    # True(1) -> 2*1 - 1 = 1
    # False(0) -> 2*0 - 1 = -1
    p_signed = 2 * p.astype(np.int8) - 1
    
    # Numba関数呼び出し (p_signed を渡す)
    _filter_events_resize_jit(x, y, p_signed, mask, change_map, fx, fy)

    # マスク適用 & 座標変換
    x_new = (x[mask] // fx).astype(x.dtype)
    y_new = (y[mask] // fy).astype(y.dtype)
    t_new = t[mask]
    
    # Polarityを {-1, 1} から {0, 1} に戻す
    # -1 -> 0, 1 -> 1
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
# 4. 変換ロジック
# ==========================================
def convert_dvs_dir_to_h5(seq: SequenceDir, args):
    """
    SequenceDir内のDVSデータを読み込み、必要ならダウンサンプルしてH5保存
    """

    if args.downsample:
        output_filename = "events_ds.h5"
    else:
        output_filename = "events.h5"

    source_dir = seq.dvs_dir
    output_dir = seq.root / "events"
    output_path = output_dir / output_filename
    
    if not source_dir.exists():
        return
    if output_path.exists():
        return

    npz_files = sorted(list(source_dir.glob("*.npz")), key=lambda p: natural_sort_key(p.name))
    if not npz_files:
        return

    display_path = f"{seq.root.parent.name}/{seq.root.name}"
    process_msg = f"Processing: {display_path}"
    if args.downsample:
        process_msg += " [Downsampling 1/2]"
    tqdm.write(process_msg)

    all_x, all_y, all_t, all_p = [], [], [], []

    # --- 1. データ読み込み ---
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
        return

    x_full = np.concatenate(all_x)
    y_full = np.concatenate(all_y)
    t_full = np.concatenate(all_t)
    p_full = np.concatenate(all_p)

    # --- 2. ダウンサンプル処理 (Optional) ---
    if args.downsample:
        # ダウンサンプル実行
        # 関数内部で bool -> {-1, 1} 計算 -> {0, 1} 復元まで行う
        x_full, y_full, p_full, t_full = apply_downsampling_half(
            x_full.astype(np.int32), 
            y_full.astype(np.int32), 
            p_full, # BooleanでもOK
            t_full, 
            args.width, 
            args.height
        )

    # --- 3. データ型変換 & 保存 ---
    # 時間単位: ナノ秒(CARLA) -> マイクロ秒
    t_micro = (t_full // 1000).astype(np.uint64)
    
    # Polarity: Booleanの場合もあるので、念のため astype(uint8) で 0, 1 に正規化
    # ダウンサンプル済みの場合は既に uint8(0,1) になっていますが、未処理の場合はここで変換
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
        
        # 解像度情報の保存
        if args.downsample:
            grp.attrs['width'] = args.width // 2
            grp.attrs['height'] = args.height // 2
        else:
            grp.attrs['width'] = args.width
            grp.attrs['height'] = args.height

    tqdm.write(f"  ✅ Saved: {output_path} (Events: {len(t_micro)})")


# ==========================================
# 5. メイン探索ロジック
# ==========================================
def process_dataset(args):
    root_dir = Path(args.input_dir)
    town_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and "Town" in d.name])

    if not town_dirs:
        print(f"No 'Town' directories found in {root_dir}")
        return

    print(f"Found {len(town_dirs)} scenes.")
    if args.downsample:
        print(f"Option: Downsampling enabled (1/2 scale). Input assume: {args.width}x{args.height}")

    for town in tqdm(town_dirs, desc="Scenes"):
        part_dirs = sorted([d for d in town.iterdir() if d.is_dir() and d.name.isdigit()])
        
        if not part_dirs:
            seq = SequenceDir(town)
            if seq.dvs_dir.exists():
                convert_dvs_dir_to_h5(seq, args)
            continue

        for part in part_dirs:
            seq = SequenceDir(part)
            if seq.dvs_dir.exists():
                convert_dvs_dir_to_h5(seq, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Dataset Root Directory")
    
    # オプション
    parser.add_argument("--downsample", action="store_true", help="Enable 1/2 spatial downsampling")
    parser.add_argument("--width", type=int, default=1280, help="Original input width (default: 1280)")
    parser.add_argument("--height", type=int, default=960, help="Original input height (default: 960)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
    else:
        process_dataset(args)
        print("\nDone.")