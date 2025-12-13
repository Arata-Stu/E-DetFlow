#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

def restore_text_lines(scene_path: Path, parts: list[Path], rel_path: str, dry: bool):
    """行単位で分割されたテキストファイルを結合して復元 (GNSS, IMU等)"""
    target_file = scene_path / rel_path
    print(f"  📄 Restoring Lines: {target_file.name}")

    if not dry:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        # ファイルを新規作成（上書き）モードで開き、順番に追記
        with open(target_file, "w") as outfile:
            for pdir in parts:
                part_file = pdir / rel_path
                if part_file.exists():
                    with open(part_file, "r") as infile:
                        shutil.copyfileobj(infile, outfile)
    
    # 元の分割ファイルを削除（移動扱いにするため）
    if not dry:
        for pdir in parts:
            part_file = pdir / rel_path
            if part_file.exists():
                part_file.unlink()

def restore_text_commas(scene_path: Path, parts: list[Path], filename: str, dry: bool):
    """カンマ区切りで分割されたファイルを結合して復元 (Steering等)"""
    target_file = scene_path / filename
    print(f"  📄 Restoring Commas: {filename}")

    all_values = []
    
    # 順番に読み込んで値をリスト化
    for pdir in parts:
        part_file = pdir / filename
        if part_file.exists():
            content = part_file.read_text().strip().strip(",")
            if content:
                values = [v.strip() for v in content.split(",") if v.strip()]
                all_values.extend(values)

    if not dry:
        if all_values:
            target_file.write_text(", ".join(all_values))

    # 元の分割ファイルを削除
    if not dry:
        for pdir in parts:
            part_file = pdir / filename
            if part_file.exists():
                part_file.unlink()

def restore_sensor_dir(scene_path: Path, parts: list[Path], dir_name: str, dry: bool):
    """センサーディレクトリ内のファイルを移動して復元"""
    target_dir = scene_path / dir_name
    print(f"  📁 Restoring Dir: {dir_name}")

    if not dry:
        target_dir.mkdir(exist_ok=True)

    for pdir in parts:
        part_sensor_dir = pdir / dir_name
        if not part_sensor_dir.exists():
            continue

        # ファイルを移動
        for f in part_sensor_dir.iterdir():
            if f.is_file():
                dest = target_dir / f.name
                if not dry:
                    # メタデータなどでファイルが既に存在する場合は上書き (shutil.moveは上書きエラーになる場合があるため注意)
                    if dest.exists():
                        dest.unlink() 
                    shutil.move(str(f), str(dest))
        
        # 空になったディレクトリを削除
        if not dry:
            part_sensor_dir.rmdir()

def process_restore_scene(scene_path: Path, dry: bool):
    print(f"\n=== Restoring Scene: {scene_path.name} ===")
    
    # 数字のみのディレクトリ (01, 02...) を取得してソート
    parts = sorted([d for d in scene_path.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))
    
    if not parts:
        print("  No split parts found.")
        return

    # 最初のパート(01)の中身を見て、復元すべき対象をリストアップ
    first_part = parts[0]
    
    # 1. ディレクトリ (Sensor data)
    sensor_dirs = [d.name for d in first_part.iterdir() if d.is_dir() and d.name not in ["gnss", "imu"]]
    for s_dir in sensor_dirs:
        restore_sensor_dir(scene_path, parts, s_dir, dry)

    # 2. GNSS/IMU フォルダ内のテキスト (Line split)
    # gnss/gnss.txt や imu/imu.txt などを探す
    for special_dir in ["gnss", "imu"]:
        part_subdir = first_part / special_dir
        if part_subdir.exists():
            for f in part_subdir.iterdir():
                if f.suffix == ".txt":
                    rel_path = f"{special_dir}/{f.name}"
                    restore_text_lines(scene_path, parts, rel_path, dry)
            # 処理後に空なら削除
            if not dry:
                for p in parts:
                    sub = p / special_dir
                    if sub.exists() and not any(sub.iterdir()):
                        sub.rmdir()

    # 3. ルートにあるテキストファイル (Comma split or Line split)
    # 元のスクリプトの挙動から推測: ルートにあるtxtでカンマ区切りっぽいもの
    # steering.txt はカンマ区切りと仮定。それ以外は安全のためLine結合にするか、個別指定。
    root_files = [f.name for f in first_part.iterdir() if f.is_file()]
    for fname in root_files:
        if "steering" in fname.lower() and fname.endswith(".txt"):
            restore_text_commas(scene_path, parts, fname, dry)
        elif fname.endswith(".txt"):
            # その他のテキストファイルは行結合とみなす（安全策）
            restore_text_lines(scene_path, parts, fname, dry)

    # 4. 空になった分割ディレクトリ(01, 02...)を削除
    if not dry:
        for p in parts:
            if p.exists() and not any(p.iterdir()):
                print(f"  🗑 Removing empty part: {p.name}")
                p.rmdir()
            elif p.exists():
                print(f"  ⚠️ Part {p.name} is not empty, skipping deletion.")

def list_scenes(input_dir: Path):
    if "Town" in input_dir.name and input_dir.is_dir():
        return [input_dir]
    return sorted([d for d in input_dir.iterdir() if d.is_dir() and "Town" in d.name])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=str, help="Path to data root or scene dir")
    ap.add_argument("--dry-run", action="store_true", help="No changes, just check")
    args = ap.parse_args()

    input_path = Path(args.input_dir)
    scenes = list_scenes(input_path)
    
    if not scenes:
        print(f"No scenes found in {input_path}")
        return

    for scene in scenes:
        process_restore_scene(scene, args.dry_run)
        
    print("\n✅ Restoration Complete.")

if __name__ == "__main__":
    main()