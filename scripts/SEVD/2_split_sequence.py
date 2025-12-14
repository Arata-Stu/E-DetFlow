#!/usr/bin/env python3
import argparse
import shutil
import math
import re
from pathlib import Path
from typing import List, Tuple, Optional

from utils.directory import SequenceDir


def zpad(i: int, n: int) -> str:
    """ゼロ埋めヘルパー"""
    width = max(2, len(str(n)))
    return f"{i:0{width}d}"


def extract_id_from_text(line: str) -> Optional[int]:
    """テキスト行から 'frame=1234' を抽出する"""
    match = re.search(r"frame=(\d+)", line)
    if match:
        return int(match.group(1))
    return None


def extract_id_from_filename(filename: str) -> Optional[int]:
    """ファイル名から末尾の数値を抽出する (例: dvs-100.png -> 100)"""
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[-1])
    return None


def split_text_by_frame_id(file_path: Path, batch_ranges: List[Tuple[int, int, Path]], dry: bool):
    """GNSS/IMUなど 'frame=XXXX' を含むテキストファイルをIDに基づいて分割"""
    print(f"  📄 Parsing & Splitting {file_path.name} (by Frame ID)...")

    with open(file_path, "r") as f:
        lines = f.readlines()

    batch_buffer = {pdir: [] for _, _, pdir in batch_ranges}
    orphaned = 0

    for line in lines:
        fid = extract_id_from_text(line)
        if fid is None:
            continue

        target_dir = None
        for min_id, max_id, pdir in batch_ranges:
            if min_id <= fid <= max_id:
                target_dir = pdir
                break
        
        if target_dir:
            batch_buffer[target_dir].append(line)
        else:
            orphaned += 1

    for _, _, pdir in batch_ranges:
        lines_to_write = batch_buffer[pdir]
        if not lines_to_write:
            continue

        if file_path.parent.name in ["gnss", "imu"]:
            out_dir = pdir / file_path.parent.name
        else:
            out_dir = pdir

        out_file = out_dir / file_path.name
        
        if not dry:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as fw:
                fw.writelines(lines_to_write)
    
    if orphaned > 0:
        print(f"    ⚠️  {orphaned} lines were out of RGB range and skipped.")


def split_text_by_ratio(file_path: Path, batch_ranges: List[Tuple[int, int, Path]], total_rgb_frames: int, split_type: str, dry: bool):
    """IDを持たないテキストデータ（Steering等）を、RGBの分割比率に合わせて分割"""
    print(f"  📄 Splitting {file_path.name} (by Ratio)...")
    
    content = file_path.read_text().strip()
    if split_type == 'comma':
        items = [v.strip() for v in content.strip(",").split(",") if v.strip()]
        separator = ", "
    else:
        items = content.splitlines()
        separator = "\n"

    total_items = len(items)
    current_idx = 0
    
    for min_id, max_id, pdir in batch_ranges:
        n_frames_in_batch = (max_id - min_id) + 1
        ratio = n_frames_in_batch / total_rgb_frames
        n_items = math.ceil(total_items * ratio)
        
        chunk = items[current_idx : current_idx + n_items]
        current_idx += n_items
        
        if not chunk: continue

        out_file = pdir / file_path.name
        if not dry:
            pdir.mkdir(parents=True, exist_ok=True)
            if split_type == 'comma':
                out_file.write_text(separator.join(chunk))
            else:
                with open(out_file, "w") as fw:
                    for line in chunk:
                        fw.write(line + "\n")


def split_sensor_dir(sensor_dir: Path, batch_ranges: List[Tuple[int, int, Path]], dry: bool):
    """画像やNPZファイルをIDに基づいて分割"""
    print(f"  📁 Splitting {sensor_dir.name} (by Frame ID)...")
    
    all_files = [f for f in sensor_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    
    data_files = [f for f in all_files if "metadata" not in f.name.lower()]
    meta_files = [f for f in all_files if "metadata" in f.name.lower()]

    moved_count = 0
    
    for f in data_files:
        fid = extract_id_from_filename(f.name)
        if fid is None:
            continue

        target_dir = None
        for min_id, max_id, pdir in batch_ranges:
            if min_id <= fid <= max_id:
                target_dir = pdir
                break
        
        if target_dir:
            dest_dir = target_dir / sensor_dir.name
            dest_file = dest_dir / f.name
            
            if not dry:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest_file))
            moved_count += 1

    for meta in meta_files:
        for _, _, pdir in batch_ranges:
            dest_dir = pdir / sensor_dir.name
            dest_meta = dest_dir / meta.name
            if not dry:
                dest_dir.mkdir(parents=True, exist_ok=True)
                if not dest_meta.exists():
                    shutil.copy2(meta, dest_meta)
    
    if not dry and moved_count == len(data_files):
        shutil.rmtree(sensor_dir) 


def process_scene(scene_path: Path, n_split: int, dry: bool):
    print(f"\n=== Processing Scene: {scene_path.name} ===")
    
    seq = SequenceDir(scene_path)
    
    rgb_indices = seq.get_frame_indices()
    if not rgb_indices:
        print("  ⚠️ No RGB frames found. Skipping.")
        return

    total_frames = len(rgb_indices)
    
    chunk_size = math.ceil(total_frames / n_split)
    
    print(f"  Total RGB Frames: {total_frames} -> Split into {n_split} parts (approx {chunk_size} frames/part)")
    
    part_dirs = [scene_path / zpad(i, n_split) for i in range(1, n_split + 1)]
    if not dry:
        for d in part_dirs:
            d.mkdir(exist_ok=True)

    batch_ranges = []
    for i, pdir in enumerate(part_dirs):
        batch_ids = rgb_indices[i * chunk_size : (i + 1) * chunk_size]
        
        if not batch_ids: 
            continue
        
        min_id, max_id = batch_ids[0], batch_ids[-1]
        batch_ranges.append((min_id, max_id, pdir))

    for file_path, split_type in seq.target_line_files:
        is_id_based = file_path.name in ["gnss.txt", "imu.txt"]
        
        if is_id_based:
            split_text_by_frame_id(file_path, batch_ranges, dry)
        else:
            split_text_by_ratio(file_path, batch_ranges, total_frames, split_type, dry)

    for sensor_dir in seq.target_sensor_dirs:
        split_sensor_dir(sensor_dir, batch_ranges, dry)

    if not dry:
        for d in sorted(scene_path.iterdir()):
            if d in part_dirs: continue
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()


def list_scenes(input_dir: Path):
    if "Town" in input_dir.name and input_dir.is_dir():
        return [input_dir]
    return sorted([d for d in input_dir.iterdir() if d.is_dir() and "Town" in d.name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=str, help="Path to data root or scene dir")
    ap.add_argument("--n_split", type=int, default=10, help="Number of splits (default: 10)")
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