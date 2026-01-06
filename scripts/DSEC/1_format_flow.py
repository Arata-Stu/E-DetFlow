import os
import argparse
import numpy as np
import cv2
import h5py
import hdf5plugin
import multiprocessing as mp
import imageio.v3 as imageio
from pathlib import Path
from tqdm import tqdm
from functools import partial
from omegaconf import OmegaConf

# 前処理スクリプトで定義されている圧縮設定
def _blosc_opts(complevel=1, shuffle='byte'):
    shuffle_map = {'none': 0, 'byte': 1, 'bit': 2}
    return dict(
        compression=32001, 
        compression_opts=(0, 0, 0, 0, complevel, shuffle_map[shuffle], 5), 
        chunks=True
    )

def load_dsec_flow_png(path: Path):
    """DSEC仕様の16bit PNGからFlowとValidマスクを復元"""
    try:
        img = imageio.imread(path).astype(np.float32)
        flow_x = (img[..., 0] - 2**15) / 128.0
        flow_y = (img[..., 1] - 2**15) / 128.0
        valid = (img[..., 2] > 0).astype(np.uint8)
        flow = np.stack([flow_x, flow_y], axis=-1)
        return flow, valid
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

def process_dsec_flow_sequence(seq_path: Path, args):
    seq_name = seq_path.name
    
    # 出力パスの設定: train/test階層を無視し、シーケンス名直下に保存
    if args.output_dir:
        output_root = Path(args.output_dir) / seq_name
        output_dir = output_root / "optical_flow_processed"
    else:
        # args.output_dirが指定されない場合は、入力シーケンス内に作成
        output_dir = seq_path / "optical_flow_processed"
    
    filename = "optical_flow_synced_ds.h5" if args.downsample else "optical_flow_synced.h5"
    final_path = output_dir / filename
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    if final_path.exists():
        return f" [Skip] {seq_name}"

    flow_dir = seq_path / "flow" / "forward"
    ts_path = seq_path / "flow" / "forward_timestamps.txt"
    
    if not flow_dir.exists() or not ts_path.exists():
        return f" [Error] Missing flow/ts: {seq_name}"

    png_files = sorted(list(flow_dir.glob("*.png")))
    try:
        raw_ts = np.loadtxt(ts_path, delimiter=',', dtype=np.int64)
        target_timestamps = raw_ts[:, 1] # end_us を使用
    except Exception as e:
        return f" [Error] TS Load Failed: {seq_name} ({e})"

    if len(png_files) != len(target_timestamps):
        return f" [Error] Count Mismatch: {seq_name} PNGs({len(png_files)}) != TS({len(target_timestamps)})"

    num_frames = len(png_files)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_flow, _ = load_dsec_flow_png(png_files[0])
    if sample_flow is None: return f" [Error] Load Failed: {seq_name}"
    
    orig_h, orig_w = sample_flow.shape[:2]
    final_h, final_w = (orig_h // 2, orig_w // 2) if args.downsample else (orig_h, orig_w)

    try:
        with h5py.File(str(tmp_path), 'w') as h5f:
            d_flow = h5f.create_dataset('flow', (num_frames, final_h, final_w, 2), dtype='f4',
                                        chunks=(1, final_h, final_w, 2), **_blosc_opts(complevel=1, shuffle='byte'))
            d_valid = h5f.create_dataset('valid', (num_frames, final_h, final_w), dtype='u1',
                                         chunks=(1, final_h, final_w), **_blosc_opts(complevel=1, shuffle='byte'))
            d_ts = h5f.create_dataset('timestamps', data=target_timestamps, dtype='i8')

            for i, png_p in enumerate(png_files):
                flow, valid = load_dsec_flow_png(png_p)
                
                if args.downsample:
                    flow = cv2.resize(flow, (final_w, final_h), interpolation=cv2.INTER_AREA) * 0.5
                    valid = cv2.resize(valid, (final_w, final_h), interpolation=cv2.INTER_NEAREST)
                
                d_flow[i], d_valid[i] = flow, valid

        tmp_path.rename(final_path)
        return f"✅ {seq_name} -> {filename} ({num_frames} frames)"

    except Exception as e:
        if tmp_path.exists(): tmp_path.unlink()
        return f"❌ Failed: {seq_name} ({str(e)})"

def process_dataset(args):
    root_dir = Path(args.input_dir)
    try: 
        conf = OmegaConf.load(args.config)
    except Exception as e:
        print(f"Error loading config: {e}"); return

    # YAMLからシーケンスパスを抽出
    rel_paths = []
    for split in conf.keys():
        if conf[split] is not None:
            rel_paths.extend(list(conf[split]))
    unique_rel_paths = list(dict.fromkeys(rel_paths))

    # train/test 構造に対応したパス探索
    all_seq_paths = []
    for p in unique_rel_paths:
        path_in_train = root_dir / "train" / p
        path_in_test = root_dir / "test" / p
        
        if path_in_train.exists():
            all_seq_paths.append(path_in_train)
        elif path_in_test.exists():
            all_seq_paths.append(path_in_test)
        else:
            print(f"⚠️ Sequence not found in train/test: {p}")
    
    if not all_seq_paths:
        print("No valid sequences found."); return

    print(f"Total sequences found: {len(all_seq_paths)}")
    num_procs = args.num_workers if args.num_workers is not None else mp.cpu_count()
    print(f"Running with {num_procs} processes")

    ctx = mp.get_context('spawn')
    worker_func = partial(process_dsec_flow_sequence, args=args)
    
    with ctx.Pool(processes=num_procs) as pool:
        results = tqdm(pool.imap_unordered(worker_func, all_seq_paths), 
                       total=len(all_seq_paths), 
                       desc="DSEC Flow Aggregation")
        for res in results:
            tqdm.write(res)

if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Convert DSEC Flow PNGs to Synced H5")
    parser.add_argument("input_dir", type=str, help="Root directory (containing train/ and test/)")
    parser.add_argument("--config", type=str, required=True, help="YAML split config")
    parser.add_argument("--output_dir", type=str, default=None, help="Output Root (Flat structure)") 
    parser.add_argument("--downsample", action="store_true", help="1/2 downsampling")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    
    args = parser.parse_args()
    if Path(args.input_dir).exists():
        process_dataset(args)
        print("\nAll finished.")
    else:
        print(f"Input path not found: {args.input_dir}")