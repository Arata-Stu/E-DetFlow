from pathlib import Path
from typing import Any, List, Optional

import h5py
try:
    import hdf5plugin
except ImportError:
    pass
import numpy as np
import torch
from torchdata.datapipes.map import MapDataPipe

from data.genx_utils.labels import ObjectLabelFactory, ObjectLabels
from data.utils.spatial import get_original_hw
from data.utils.types import DatasetType
from utils.timers import TimerDummy as Timer


def get_event_representation_dir(path: Path, ev_representation_name: str) -> Path:
    ev_repr_dir = path / 'event_representations_v2' / ev_representation_name
    assert ev_repr_dir.is_dir(), f'{ev_repr_dir}'
    return ev_repr_dir


def get_objframe_idx_2_repr_idx(path: Path, ev_representation_name: str) -> np.ndarray:
    ev_repr_dir = get_event_representation_dir(path=path, ev_representation_name=ev_representation_name)
    objframe_idx_2_repr_idx = np.load(str(ev_repr_dir / 'objframe_idx_2_repr_idx.npy'))
    return objframe_idx_2_repr_idx


class SequenceBase(MapDataPipe):
    """
    Structure example of a sequence:
    .
    ├── event_representations_v2
    │ └── ev_representation_name
    │     ├── event_representations.h5
    │     ├── flow_ground_truth.h5
    │     ├── objframe_idx_2_repr_idx.npy
    │     └── timestamps_us.npy
    └── labels_v2
        ├── labels.npz
        └── timestamps_us.npy
    """

    def __init__(self,
                 path: Path,
                 ev_representation_name: str,
                 sequence_length: int,
                 dataset_type: DatasetType,
                 downsample_by_factor_2: bool,
                 only_load_end_labels: bool):
        assert sequence_length >= 1
        assert path.is_dir()
        assert dataset_type in {DatasetType.GEN1, DatasetType.GEN4, DatasetType.VGA, DatasetType.SEVD}, f'{dataset_type} not implemented'

        self.only_load_end_labels = only_load_end_labels

        ev_repr_dir = get_event_representation_dir(path=path, ev_representation_name=ev_representation_name)

        labels_dir = path / 'labels_v2'
        assert labels_dir.is_dir()

        height, width = get_original_hw(dataset_type)
        self.seq_len = sequence_length

        if dataset_type in {DatasetType.GEN1, DatasetType.GEN4, DatasetType.VGA}:
            ds_factor_str = '_ds2_nearest' if downsample_by_factor_2 else ''
        elif dataset_type == DatasetType.SEVD:
            ds_factor_str = ''
            
        self.ev_repr_file = ev_repr_dir / f'event_representations{ds_factor_str}.h5'
        assert self.ev_repr_file.exists(), f'{str(self.ev_repr_file)=}'

        self.flow_file = ev_repr_dir / 'flow_ground_truth.h5'
        self.has_flow = self.flow_file.exists()

        self.has_valid = False
        if self.has_flow:
            # Check if 'valid' dataset exists in the file
            try:
                with h5py.File(str(self.flow_file), 'r') as h5f:
                    if 'valid' in h5f:
                        self.has_valid = True
            except Exception as e:
                print(f"Warning: Failed to check valid mask in {self.flow_file}: {e}")

        with Timer(timer_name='prepare labels'):
            label_data = np.load(str(labels_dir / 'labels.npz'))
            objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
            labels = label_data['labels']
            label_factory = ObjectLabelFactory.from_structured_array(
                object_labels=labels,
                objframe_idx_2_label_idx=objframe_idx_2_label_idx,
                input_size_hw=(height, width),
                downsample_factor=2 if downsample_by_factor_2 else None)
            self.label_factory = label_factory

        with Timer(timer_name='load objframe_idx_2_repr_idx'):
            self.objframe_idx_2_repr_idx = get_objframe_idx_2_repr_idx(
                path=path, ev_representation_name=ev_representation_name)
        with Timer(timer_name='construct repr_idx_2_objframe_idx'):
            self.repr_idx_2_objframe_idx = dict(zip(self.objframe_idx_2_repr_idx,
                                                    range(len(self.objframe_idx_2_repr_idx))))

    def _get_labels_from_repr_idx(self, repr_idx: int) -> Optional[ObjectLabels]:
        objframe_idx = self.repr_idx_2_objframe_idx.get(repr_idx, None)
        return None if objframe_idx is None else self.label_factory[objframe_idx]

    def _get_event_repr_torch(self, start_idx: int, end_idx: int) -> List[torch.Tensor]:
        assert end_idx > start_idx
        with h5py.File(str(self.ev_repr_file), 'r') as h5f:
            ev_repr = h5f['data'][start_idx:end_idx]
        ev_repr = torch.from_numpy(ev_repr)
        if ev_repr.dtype != torch.uint8:
            ev_repr = torch.asarray(ev_repr, dtype=torch.float32)
        ev_repr = torch.split(ev_repr, 1, dim=0)
        # remove first dim that is always 1 due to how torch.split works
        ev_repr = [x[0] for x in ev_repr]
        return ev_repr
    
    def _get_flow_torch(self, start_idx: int, end_idx: int) -> List[torch.Tensor]:
        assert end_idx > start_idx
        if not self.has_flow:
            raise FileNotFoundError(f"Flow file not found: {self.flow_file}")

        with h5py.File(str(self.flow_file), 'r') as h5f:
            flow_data = h5f['flow'][start_idx:end_idx]
        
        flow_tensor = torch.from_numpy(flow_data) # (N, H, W, 2) 
        flow_tensor = flow_tensor.permute(0, 3, 1, 2).float() # (N, H, W, 2) -> (N, 2, H, W)
        flow_list = torch.split(flow_tensor, 1, dim=0)
        flow_list = [x[0] for x in flow_list]
        
        return flow_list
    
    def _get_valid_torch(self, start_idx: int, end_idx: int) -> List[torch.Tensor]:
        """
        validマスクを取得する関数
        Returns: List of tensors with shape (1, H, W), float32 (0.0 or 1.0)
        """
        assert end_idx > start_idx
        if not self.has_valid:
            raise FileNotFoundError(f"Valid mask not found in {self.flow_file}")

        with h5py.File(str(self.flow_file), 'r') as h5f:
            valid_data = h5f['valid'][start_idx:end_idx] # (N, H, W) uint8

        valid_tensor = torch.from_numpy(valid_data)
        valid_tensor = valid_tensor.unsqueeze(1).float() # (N, H, W) -> (N, 1, H, W)
        
        valid_list = torch.split(valid_tensor, 1, dim=0)
        valid_list = [x[0] for x in valid_list]
        
        return valid_list

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError
