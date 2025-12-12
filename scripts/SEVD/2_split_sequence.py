#!/usr/bin/env python3
import argparse
import shutil
import math
from pathlib import Path

from utils.directory import SequenceDir

def zpad(i: int, n: int) -> str:
    """ゼロ埋めヘルパー (例: 1 -> '01', 10 -> '10')"""
    width = max(2, len(str(n)))
    return f"{i:0{width}d}"


def split_text_by_lines(file_path: Path, part_dirs: list[Path], move: bool, dry: bool):
    """行単位でテキストを分割 (GNSS, IMU等)"""
    is_in_subdir = file_path.parent.name in ["gnss", "imu"]
    
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    n = len(part_dirs)
    chunk_size = math.ceil(len(lines) / n)

    print(f"  📄 Splitting {file_path.name} (Lines)...")
    for i, pdir in enumerate(part_dirs, start=1):
        chunk = lines[(i - 1) * chunk_size: i * chunk_size]
        if not chunk: continue
        
        # 出力先パスの決定 (gnssフォルダ内なら、分割先でもgnssフォルダを作る)
        if is_in_subdir:
            out_dir = pdir / file_path.parent.name
        else:
            out_dir = pdir
            
        out_file = out_dir / file_path.name
        
        if not dry:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as fw:
                fw.writelines(chunk)
                
    if move and not dry:
        file_path.unlink()
        # 親ディレクトリが空なら削除
        if is_in_subdir and not any(file_path.parent.iterdir()):
             file_path.parent.rmdir()


def split_text_by_commas(file_path: Path, part_dirs: list[Path], move: bool, dry: bool):
    """カンマ区切りで値を分割 (Steering等)"""
    content = file_path.read_text().strip().strip(",")
    values = [v.strip() for v in content.split(",") if v.strip()]
    
    n = len(part_dirs)
    chunk_size = math.ceil(len(values) / n)

    print(f"  📄 Splitting {file_path.name} (Commas)...")
    for i, pdir in enumerate(part_dirs, start=1):
        chunk = values[(i - 1) * chunk_size: i * chunk_size]
        if not chunk: continue
        
        out_file = pdir / file_path.name
        if not dry:
            pdir.mkdir(parents=True, exist_ok=True)
            out_file.write_text(", ".join(chunk))
            
    if move and not dry:
        file_path.unlink()


def split_frame_dir(sensor_dir: Path, part_dirs: list[Path], move: bool, dry: bool):
    """センサーディレクトリ内のファイルを分割"""
    # 隠しファイル以外を取得
    files = sorted([f for f in sensor_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
    
    # メタデータとデータファイルを分離
    data_files = [f for f in files if "metadata" not in f.name.lower()]
    meta_files = [f for f in files if "metadata" in f.name.lower()]

    if not data_files:
        return

    n = len(part_dirs)
    chunk_size = math.ceil(len(data_files) / n)

    print(f"  📁 Splitting {sensor_dir.name} ({len(data_files)} files)...")

    for i, pdir in enumerate(part_dirs, start=1):
        target_dir = pdir / sensor_dir.name
        
        # データの移動/コピー
        chunk = data_files[(i - 1) * chunk_size: i * chunk_size]
        for f in chunk:
            dest = target_dir / f.name
            if not dry:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest)) if move else shutil.copy2(f, dest)
        
        # メタデータのコピー (全パーツに配置)
        for meta in meta_files:
            dest_meta = target_dir / meta.name
            if not dry:
                target_dir.mkdir(parents=True, exist_ok=True)
                if not dest_meta.exists():
                    shutil.copy2(meta, dest_meta)

    # 元ディレクトリの削除
    if move and not dry:
        shutil.rmtree(sensor_dir)


def process_scene(scene_path: Path, n_split: int, dry: bool):
    print(f"\n=== Processing Scene: {scene_path.name} ===")
    
    # SequenceDirクラスを使ってディレクトリ構造を把握
    seq = SequenceDir(scene_path)
    
    # 分割先のディレクトリリストを作成
    part_dirs = [scene_path / zpad(i, n_split) for i in range(1, n_split + 1)]
    
    if not dry:
        for d in part_dirs:
            d.mkdir(exist_ok=True)

    # 1. 単一ファイル (GNSS, IMU, Steering) の分割
    # SequenceDirのプロパティから「存在するファイル」と「分割タイプ」を取得
    for file_path, split_type in seq.target_line_files:
        if split_type == 'line':
            split_text_by_lines(file_path, part_dirs, move=True, dry=dry)
        elif split_type == 'comma':
            split_text_by_commas(file_path, part_dirs, move=True, dry=dry)

    # 2. センサーディレクトリ (RGB, DVS...) の分割
    # SequenceDirのプロパティから「存在するディレクトリ」を取得
    for sensor_dir in seq.target_sensor_dirs:
        split_frame_dir(sensor_dir, part_dirs, move=True, dry=dry)

    # 3. 掃除
    if not dry:
        for d in sorted(scene_path.iterdir()):
            if d.is_dir() and d not in part_dirs and not any(d.iterdir()):
                d.rmdir()


def list_scenes(input_dir: Path):
    if "Town" in input_dir.name and input_dir.is_dir():
        return [input_dir]
    return sorted([d for d in input_dir.iterdir() if d.is_dir() and "Town" in d.name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=str, help="Path to data root or scene dir")
    ap.add_argument("--n_split", type=int, default=10, help="Number of splits")
    ap.add_argument("--dry-run", action="store_true", help="No changes")
    args = ap.parse_args()

    input_path = Path(args.input_dir)
    scenes = list_scenes(input_path)
    
    if not scenes:
        print(f"No scenes found in {input_path}")
        return

    for scene in scenes:
        process_scene(scene, args.n_split, args.dry_run)
        
    print("\n✅ Done.")


if __name__ == "__main__":
    main()