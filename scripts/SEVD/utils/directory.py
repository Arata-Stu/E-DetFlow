from pathlib import Path
from typing import Union, List, Tuple
import numpy as np

class BaseDirectory:
    def __init__(self, root: Union[str, Path]):
        if not isinstance(root, Path):
            root = Path(root)
        self.root = root

    def exists(self):
        return self.root.is_dir()

class SequenceDir(BaseDirectory):
    def __init__(self, root: Union[str, Path]):
        super().__init__(root)

        # ディレクトリパスの定義
        self.rgb_dir = self.root / "rgb_camera-front"
        self.depth_dir = self.root / "depth_camera-front"
        self.dvs_dir = self.root / "dvs_camera-front"
        self.optical_flow_dir = self.root / "optical_flow-front"
        self.sem_seg_dir = self.root / "semantic_segmentation_camera-front"
        self.ins_seg_dir = self.root / "instance_segmentation_camera-front"
        
        # 単一ファイル系（センサディレクトリ外）
        self.gnss_file = self.root / "gnss" / "gnss.txt"
        self.imu_file = self.root / "imu" / "imu.txt"
        self.steering_true_file = self.root / "steering_true.txt"
        self.steering_norm_file = self.root / "steering_norm.txt"  
    
    @property
    def target_sensor_dirs(self) -> List[Path]:
        dirs = [
            self.rgb_dir, self.depth_dir, self.dvs_dir,
            self.optical_flow_dir, self.sem_seg_dir, self.ins_seg_dir
        ]
        return [d for d in dirs if d.exists()]

    @property
    def target_line_files(self) -> List[Tuple[Path, str]]:
        targets = []
        if self.gnss_file.exists(): targets.append((self.gnss_file, 'line'))
        if self.imu_file.exists(): targets.append((self.imu_file, 'line'))
        if self.steering_true_file.exists(): targets.append((self.steering_true_file, 'comma'))
        if self.steering_norm_file.exists(): targets.append((self.steering_norm_file, 'comma'))
        return targets

    def get_frame_indices(self) -> List[int]:
        """
        RGBディレクトリを基準にして、存在するフレーム番号(ID)のソート済みリストを返す
        例: [10081, 10082, ..., 10289]
        """
        if not self.rgb_dir.exists():
            return []

        indices = [
            int(p.stem) for p in self.rgb_dir.glob("*.png") 
            if p.stem.isdigit()
        ]
        return sorted(indices)

    def get_rgb_image_path(self, frame_id: int) -> Path:
        return self.rgb_dir / f"{frame_id}.png"

    def get_rgb_label_path(self, frame_id: int) -> Path:
        """RGB用のバウンディングボックス情報(.txt)"""
        return self.rgb_dir / f"{frame_id}.txt"

    def get_rgb_metadata_path(self) -> Path:
        return self.rgb_dir / "rgb_camera_metadata.txt"

    def get_dvs_image_path(self, frame_id: int) -> Path:
        """積算画像 (dvs-XXXXX.png)"""
        return self.dvs_dir / f"dvs-{frame_id}.png"

    def get_dvs_npz_path(self, frame_id: int) -> Path:
        """イベントデータ (dvs-XXXXX-xytp.npz)"""
        return self.dvs_dir / f"dvs-{frame_id}-xytp.npz"

    def get_dvs_label_path(self, frame_id: int) -> Path:
        """DVS用のバウンディングボックス情報 (dvs-XXXXX.txt)"""
        return self.dvs_dir / f"dvs-{frame_id}.txt"

    def get_depth_image_path(self, frame_id: int) -> Path:
        return self.depth_dir / f"{frame_id}.png"

    def get_optical_flow_npz_path(self, frame_id: int) -> Path:
        return self.optical_flow_dir / f"{frame_id}.npz"

    def get_optical_flow_vis_path(self, frame_id: int) -> Path:
        return self.optical_flow_dir / f"{frame_id}.png"

    def load_npz(self, path: Path):
        """npzファイルをロードして返すヘルパー"""
        if path.exists():
            return np.load(path)
        return None

    def read_gnss(self):
        """GNSSテキストを読み込む簡易実装例"""
        if self.gnss_file.exists():
            with open(self.gnss_file, 'r') as f:
                return f.readlines()
        return []