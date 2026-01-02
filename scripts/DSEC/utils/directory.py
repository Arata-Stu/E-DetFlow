import h5py
import hdf5plugin
import numpy as np
import imageio.v3 as imageio 
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple

class SequenceDir:
    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        
        # --- ディレクトリ定義 ---
        self.rgb_dir = self.root / "images" / "left" / "distorted" 
        self.events_dir = self.root / "events" / "left"
        self.objects_dir = self.root / "object_detections" / "left"
        self.flow_root = self.root / "flow" 
        
        # --- ファイル・パス定義 ---
        self.events_h5_path = self.events_dir / "events.h5"
        self.rectify_map_path = self.events_dir / "rectify_map.h5"
        self.tracks_npy_path = self.objects_dir / "tracks.npy"
        self.image_timestamps_path = self.root / "images" / "left" / "image_timestamps.txt"

        # Flow用
        self.flow_forward_dir = self.flow_root / "forward"
        self.flow_backward_dir = self.flow_root / "backward"
        self.flow_forward_ts_path = self.flow_root / "forward_timestamps.txt"
        self.flow_backward_ts_path = self.flow_root / "backward_timestamps.txt"

    def load_flow_timestamps(self, forward: bool = True) -> np.ndarray:
        """Flowの各ファイルに対応する (start_us, end_us) のリストをロード"""
        ts_path = self.flow_forward_ts_path if forward else self.flow_backward_ts_path
        if ts_path.exists():
            return np.loadtxt(ts_path, delimiter=',', dtype=np.int64)
        return np.array([])

    def get_flow_indices(self, forward: bool = True) -> List[int]:
        """FlowディレクトリにあるPNGファイル名（ID）をソートして取得"""
        target_dir = self.flow_forward_dir if forward else self.flow_backward_dir
        if not target_dir.exists():
            return []
        indices = [int(p.stem) for p in target_dir.glob("*.png") if p.stem.isdigit()]
        return sorted(indices)

    def load_flow_png(self, index: int, forward: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        DSEC仕様の16bit PNGからFlowデータを復元する
        Returns: (flow_x, flow_y, valid_mask)
        """
        target_dir = self.flow_forward_dir if forward else self.flow_backward_dir
        path = target_dir / f"{index:06d}.png"
        
        if not path.exists():
            raise FileNotFoundError(f"Flow file not found: {path}")

        img = imageio.imread(path) 
        img = img.astype(np.float32)

        # 1st channel (R): x, 2nd channel (G): y, 3rd channel (B): valid
        flow_x = (img[..., 0] - 2**15) / 128.0
        flow_y = (img[..., 1] - 2**15) / 128.0
        valid = img[..., 2].astype(bool)

        return flow_x, flow_y, valid

    def get_event_data_by_ms(self, start_ms: int, end_ms: int) -> Dict[str, np.ndarray]:
        if not self.events_h5_path.exists(): return {}
        with h5py.File(self.events_h5_path, 'r') as f:
            ms_to_idx = f['ms_to_idx'][:]
            t_offset = f['t_offset'][()]
            idx_start = ms_to_idx[start_ms]; idx_end = ms_to_idx[end_ms]
            events = f['events']
            t = events['t'][idx_start:idx_end].astype(np.int64) + t_offset
            return {'t': t, 'x': events['x'][idx_start:idx_end], 'y': events['y'][idx_start:idx_end], 'p': events['p'][idx_start:idx_end]}

    def load_tracks(self) -> Optional[np.ndarray]:
        if self.tracks_npy_path.exists(): return np.load(self.tracks_npy_path)
        return None