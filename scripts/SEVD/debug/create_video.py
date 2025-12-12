#!/usr/bin/env python3
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path

from utils.directory import SequenceDir



# ==========================================
# 1. KITTI BBox Helper
# ==========================================
class KittiObject:
    def __init__(self, line):
        parts = line.strip().split(' ')
        # 想定フォーマット: type truncated occluded alpha xmin ymin xmax ymax h w l x y z ry ID
        # 最低でも15要素あることを期待（IDがある場合）
        self.type = parts[0]
        # BBox coordinates (0-based pixels)
        self.xmin = float(parts[4])
        self.ymin = float(parts[5])
        self.xmax = float(parts[6])
        self.ymax = float(parts[7])
        
        # 末尾にIDがあると仮定 (以前の会話に基づく)
        # IDがない標準フォーマットの場合は ID=0 とする
        if len(parts) > 15: 
            self.track_id = parts[-1] 
        else:
            self.track_id = "N/A"

def draw_bboxes(frame, label_path):
    """画像フレームにBBoxを描画する"""
    if not label_path.exists():
        return frame
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                obj = KittiObject(line)

                # 座標を整数に変換
                p1 = (int(obj.xmin), int(obj.ymin))
                p2 = (int(obj.xmax), int(obj.ymax))

                # 緑色の矩形を描画
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

                # ラベルとIDを描画
                label_text = f"{obj.type}:{obj.track_id}"
                cv2.putText(frame, label_text, (p1[0], max(0, p1[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    except Exception as e:
        # 描画エラーはログに出すが処理は止めない
        # print(f"BBox drawing error: {e}")
        pass
    return frame


# ==========================================
# 2. 動画生成ロジック (SequenceDir 対応版)
# ==========================================
def create_video_for_sensor(seq: SequenceDir, sensor_type: str, output_dir: Path, fps: float):
    """
    SequenceDir と センサータイプ('rgb', 'dvs', 'depth'...) を指定して動画を作成
    """
    
    # センサータイプに応じた パス取得関数とディレクトリ を設定
    if sensor_type == 'rgb':
        target_dir = seq.rgb_dir
        get_img = seq.get_rgb_image_path
        get_lbl = seq.get_rgb_label_path
    elif sensor_type == 'dvs':
        target_dir = seq.dvs_dir
        get_img = seq.get_dvs_image_path
        get_lbl = seq.get_dvs_label_path
    elif sensor_type == 'depth':
        target_dir = seq.depth_dir
        get_img = seq.get_depth_image_path
        get_lbl = lambda x: Path("dummy") # Depthには通常ラベルがない
    elif sensor_type == 'optical_flow':
        target_dir = seq.optical_flow_dir
        get_img = seq.get_optical_flow_vis_path # 可視化画像(.png)を取得
        get_lbl = lambda x: Path("dummy")
    elif sensor_type == 'semantic_segmentation':
        target_dir = seq.sem_seg_dir
        get_img = lambda idx: seq.sem_seg_dir / f"{idx}.png" # クラスにメソッドがない場合の補完
        get_lbl = lambda x: Path("dummy")
    elif sensor_type == 'instance_segmentation':
        target_dir = seq.ins_seg_dir
        get_img = lambda idx: seq.ins_seg_dir / f"{idx}.png"
        get_lbl = lambda x: Path("dummy")
    else:
        return

    # ディレクトリが存在しない場合はスキップ
    if not target_dir.exists():
        return

    # フレームリスト取得
    # SequenceDir.get_frame_indices() は RGBディレクトリ依存の実装になっているため、
    # もしRGBがない場合（DVSのみなど）は、対象ディレクトリからインデックスを作成するフォールバックを行う
    frame_indices = seq.get_frame_indices()
    if not frame_indices and target_dir.exists():
        # フォールバック: 対象ディレクトリ内のPNGからIDを抽出
        frame_indices = sorted([
            int(p.name.split('-')[-1].split('.')[0]) if '-' in p.name else int(p.stem)
            for p in target_dir.glob("*.png") if p.name[0].isdigit() or "dvs-" in p.name
        ])

    if not frame_indices:
        return

    # 出力パス設定
    # 例: output_base/Town01/01/dvs_camera-front.mp4
    video_filename = f"{target_dir.name}.mp4"
    output_path = output_dir / video_filename
    
    if output_path.exists():
        # 既に存在する場合はスキップしたい場合ここを有効化
        # return 
        pass

    output_dir.mkdir(parents=True, exist_ok=True)

    # 最初のフレームを読んで解像度決定
    first_img_path = get_img(frame_indices[0])
    first_frame = cv2.imread(str(first_img_path))
    if first_frame is None:
        return
    height, width, layers = first_frame.shape

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # プログレスバー表示
    desc_str = f"Creating {sensor_type.upper()}: {output_dir.name}/{video_filename}"
    
    for idx in tqdm(frame_indices, desc=desc_str, leave=False, ncols=100):
        # 画像パス取得
        img_path = get_img(idx)
        
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # ラベルがあれば描画
        lbl_path = get_lbl(idx)
        if lbl_path.exists():
            frame = draw_bboxes(frame, lbl_path)

        out.write(frame)

    out.release()


# ==========================================
# 3. シーケンス処理ループ
# ==========================================
def process_sequence_dir(seq: SequenceDir, base_output_dir: Path, fps: float):
    """
    1つのSequenceDir（Town01/01/など）を受け取り、RGB/DVS等の動画を作成
    """
    # 出力先の構造を維持: base_output_dir / TownXX / 01 / ...
    # seq.root が "path/to/Town01/01" だとすると
    # output は "output_videos/Town01/01/" にしたい
    
    # 親ディレクトリ名(Town01) と カレントディレクトリ名(01) を取得
    parent_name = seq.root.parent.name
    current_name = seq.root.name
    
    # 階層構造に応じて出力先を決定
    if "Town" in parent_name:
        # .../Town01/01 のパターン
        final_output_dir = base_output_dir / parent_name / current_name
    else:
        # .../Town01 (直下) のパターン
        final_output_dir = base_output_dir / current_name

    # 各センサーについて動画作成を試みる
    sensors = [
        'rgb', 'dvs', 'depth', 'optical_flow', 
        'semantic_segmentation', 'instance_segmentation'
    ]
    
    for sensor in sensors:
        create_video_for_sensor(seq, sensor, final_output_dir, fps)


# ==========================================
# 4. メイン探索ロジック
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="SequenceDirを使用して動画を生成します")
    parser.add_argument("input_dir", type=str, help="データセットルート または Townディレクトリ")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS")
    parser.add_argument("--output_dir", type=str, default="videos", help="出力先ディレクトリ")
    
    args = parser.parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir).resolve()

    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    # 探索ロジック (Town -> Part -> SequenceDir)
    # これまでのスクリプトと同様のロジック
    town_dirs = []
    if "Town" in input_path.name:
        town_dirs = [input_path]
    else:
        town_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and "Town" in d.name])

    print(f"Found {len(town_dirs)} scenes.")
    print(f"Output Directory: {output_path}")

    for town in tqdm(town_dirs, desc="Total Progress"):
        # パートフォルダ (01, 02...) を探す
        part_dirs = sorted([d for d in town.iterdir() if d.is_dir() and d.name.isdigit()])
        
        if part_dirs:
            # スプリットされている場合
            for part in part_dirs:
                seq = SequenceDir(part)
                process_sequence_dir(seq, output_path, args.fps)
        else:
            # スプリットされていない場合 (Town直下)
            seq = SequenceDir(town)
            process_sequence_dir(seq, output_path, args.fps)

    print("\n✅ All videos generated.")

if __name__ == "__main__":
    main()