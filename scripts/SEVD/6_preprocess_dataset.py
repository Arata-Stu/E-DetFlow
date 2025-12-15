import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys

sys.path.append('../..')
from typing import Any, Dict, List, Optional, Tuple, Union
import weakref

import h5py
try:
    import hdf5plugin
except ImportError:
    pass
from numba import jit
import numpy as np
from omegaconf import OmegaConf, DictConfig, MISSING
import torch
from tqdm import tqdm

from utils.preprocessing import _blosc_opts
from data.utils.representations import (
    RepresentationBase,
    StackedHistogram,
    MixedDensityEventStack,
    StackedHistogramInterpolated,
    VoxelGrid,
)


# ==========================================
# 設定クラス & Enum
# ==========================================

class DataKeys(Enum):
    InNPY = auto()
    InH5 = auto()
    InFlowH5 = auto()  
    OutLabelDir = auto()
    OutEvReprDir = auto()
    SplitType = auto()

class SplitType(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

split_name_2_type = {
    'train': SplitType.TRAIN,
    'val': SplitType.VAL,
    'test': SplitType.TEST,
}

# デフォルト解像度 (SEVD)
dataset_2_height = {'SEVD': 960}
dataset_2_width = {'SEVD': 1280}


class NoLabelsException(Exception):
    ...

class AggregationType(Enum):
    COUNT = auto()
    DURATION = auto()

aggregation_2_string = {
    AggregationType.DURATION: 'dt',
    AggregationType.COUNT: 'ne',
}

# --- Config Dataclasses ---
@dataclass
class FilterConf:
    apply_psee_bbox_filter: bool = MISSING
    apply_faulty_bbox_filter: bool = MISSING

@dataclass
class EventWindowExtractionConf:
    method: AggregationType = MISSING
    value: int = MISSING
    ts_step_ev_repr_ms: int = MISSING

@dataclass
class StackedHistogramConf:
    name: str = MISSING
    nbins: int = MISSING
    count_cutoff: Optional[int] = MISSING
    event_window_extraction: EventWindowExtractionConf = field(default_factory=EventWindowExtractionConf)
    fastmode: bool = True

@dataclass
class MixedDensityEventStackConf:
    name: str = MISSING
    nbins: int = MISSING
    count_cutoff: Optional[int] = MISSING
    event_window_extraction: EventWindowExtractionConf = field(default_factory=EventWindowExtractionConf)

@dataclass
class StackedHistogramInterpolatedConf:
    name: str = MISSING
    nbins: int = MISSING
    count_cutoff: Optional[int] = MISSING
    event_window_extraction: EventWindowExtractionConf = field(default_factory=EventWindowExtractionConf)

@dataclass
class VoxelGridConf:
    name: str = MISSING
    nbins: int = MISSING
    normalize: bool = True
    event_window_extraction: EventWindowExtractionConf = field(default_factory=EventWindowExtractionConf)

name_2_structured_config = {
    'stacked_histogram': StackedHistogramConf,
    'mixeddensity_stack': MixedDensityEventStackConf,
    'stacked_histogram_interpolated': StackedHistogramInterpolatedConf,
    'voxel_grid': VoxelGridConf,
}

# ==========================================
# Factory Classes
# ==========================================
class EventRepresentationFactory(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def create(self, height: int, width: int) -> Any:
        ...

class StackedHistogramFactory(EventRepresentationFactory):
    @property
    def name(self) -> str:
        extraction = self.config.event_window_extraction
        return f'{self.config.name}_{aggregation_2_string[extraction.method]}={extraction.value}_nbins={self.config.nbins}'
    def create(self, height: int, width: int) -> StackedHistogram:
        return StackedHistogram(bins=self.config.nbins, height=height, width=width, count_cutoff=self.config.count_cutoff, fastmode=self.config.fastmode)

class MixedDensityStackFactory(EventRepresentationFactory):
    @property
    def name(self) -> str:
        extraction = self.config.event_window_extraction
        cutoff_str = f'_cutoff={self.config.count_cutoff}' if self.config.count_cutoff is not None else ''
        return f'{self.config.name}_{aggregation_2_string[extraction.method]}={extraction.value}_nbins={self.config.nbins}{cutoff_str}'
    def create(self, height: int, width: int) -> MixedDensityEventStack:
        return MixedDensityEventStack(bins=self.config.nbins, height=height, width=width, count_cutoff=self.config.count_cutoff)

class StackedHistogramInterpolatedFactory(EventRepresentationFactory):
    @property
    def name(self) -> str:
        extraction = self.config.event_window_extraction
        return f'{self.config.name}_{aggregation_2_string[extraction.method]}={extraction.value}_nbins={self.config.nbins}'
    def create(self, height: int, width: int) -> StackedHistogramInterpolated:
        return StackedHistogramInterpolated(bins=self.config.nbins, height=height, width=width, count_cutoff=self.config.count_cutoff)

class VoxelGridFactory(EventRepresentationFactory):
    @property
    def name(self) -> str:
        extraction = self.config.event_window_extraction
        norm_str = '_norm' if self.config.normalize else ''
        return f'{self.config.name}_{aggregation_2_string[extraction.method]}={extraction.value}_nbins={self.config.nbins}{norm_str}'
    def create(self, height: int, width: int) -> VoxelGrid:
        return VoxelGrid(bins=self.config.nbins, height=height, width=width, normalize=self.config.normalize)

name_2_ev_repr_factory = {
    'stacked_histogram': StackedHistogramFactory,
    'mixeddensity_stack': MixedDensityStackFactory,
    'stacked_histogram_interpolated': StackedHistogramInterpolatedFactory,
    'voxel_grid': VoxelGridFactory,
}

def get_configuration(ev_repr_yaml_config: Path, extraction_yaml_config: Path) -> DictConfig:
    config = OmegaConf.load(ev_repr_yaml_config)
    event_window_extraction_config = OmegaConf.load(extraction_yaml_config)
    event_window_extraction_config = OmegaConf.merge(OmegaConf.structured(EventWindowExtractionConf), event_window_extraction_config)
    config.event_window_extraction = event_window_extraction_config
    config_schema = OmegaConf.structured(name_2_structured_config[config.name])
    config = OmegaConf.merge(config_schema, config)
    return config

class H5Writer:
    def __init__(self, outfile: Path, key: str, ev_repr_shape: Tuple, numpy_dtype: np.dtype):
        assert len(ev_repr_shape) == 3
        self.h5f = h5py.File(str(outfile), 'w')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
        self.key = key
        self.numpy_dtype = numpy_dtype

        maxshape = (None,) + ev_repr_shape
        chunkshape = (1,) + ev_repr_shape
        self.maxshape = maxshape
        self.h5f.create_dataset(key, dtype=self.numpy_dtype.name, shape=chunkshape, chunks=chunkshape,
                                maxshape=maxshape, **_blosc_opts(complevel=1, shuffle='byte'))
        self.t_idx = 0

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalizer()
    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()
    def close(self):
        self.h5f.close()
    def get_current_length(self):
        return self.t_idx
    def add_data(self, data: np.ndarray):
        assert data.dtype == self.numpy_dtype
        assert data.shape == self.maxshape[1:]
        new_size = self.t_idx + 1
        self.h5f[self.key].resize(new_size, axis=0)
        self.h5f[self.key][self.t_idx:new_size] = data
        self.t_idx = new_size

class H5Reader:
    def __init__(self, h5_file: Path, dataset: str = 'SEVD'):
        assert h5_file.exists()
        assert h5_file.suffix == '.h5'
        
        self.h5f = h5py.File(str(h5_file), 'r')
        self._finalizer = weakref.finalize(self, self._close_callback, self.h5f)
        self.is_open = True

        ev_grp = self.h5f['events']
        if 'height' in ev_grp.attrs:
            self.height = int(ev_grp.attrs['height'])
            self.width = int(ev_grp.attrs['width'])
        elif 'height' in ev_grp:
            self.height = ev_grp['height'][()].item()
            self.width = ev_grp['width'][()].item()
        else:
            # フォールバック
            self.height = dataset_2_height.get(dataset, 480)
            self.width = dataset_2_width.get(dataset, 640)

        self.all_times = None

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalizer()
    @staticmethod
    def _close_callback(h5f: h5py.File):
        h5f.close()
    def close(self):
        self.h5f.close()
        self.is_open = False
    def get_height_and_width(self) -> Tuple[int, int]:
        return self.height, self.width
    @property
    def time(self) -> np.ndarray:
        assert self.is_open
        if self.all_times is None:
            self.all_times = np.asarray(self.h5f['events']['t'])
            self._correct_time(self.all_times)
        return self.all_times
    @staticmethod
    @jit(nopython=True)
    def _correct_time(time_array: np.ndarray):
        assert time_array[0] >= 0
        time_last = 0
        for idx, time in enumerate(time_array):
            if time < time_last:
                time_array[idx] = time_last
            else:
                time_last = time

    def get_event_slice(self, idx_start: int, idx_end: int, convert_2_torch: bool = True):
        assert self.is_open
        assert idx_end >= idx_start
        ev_data = self.h5f['events']
        x_array = np.asarray(ev_data['x'][idx_start:idx_end], dtype='int64')
        y_array = np.asarray(ev_data['y'][idx_start:idx_end], dtype='int64')
        p_array = np.asarray(ev_data['p'][idx_start:idx_end], dtype='int64')
        p_array = np.clip(p_array, a_min=0, a_max=None)
        t_array = np.asarray(self.time[idx_start:idx_end], dtype='int64')
        
        ev_data_dict = dict(
            x=x_array if not convert_2_torch else torch.from_numpy(x_array),
            y=y_array if not convert_2_torch else torch.from_numpy(y_array),
            p=p_array if not convert_2_torch else torch.from_numpy(p_array),
            t=t_array if not convert_2_torch else torch.from_numpy(t_array),
            height=self.height,
            width=self.width,
        )
        return ev_data_dict

class FlowReader:
    def __init__(self, h5_file: Path):
        self.h5_file = h5_file
        self.h5f = None
        self.timestamps = None
        
    def __enter__(self):
        self.h5f = h5py.File(str(self.h5_file), 'r')
        self.timestamps = np.asarray(self.h5f['timestamps'], dtype='int64')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5f:
            self.h5f.close()

    def get_nearest_flow(self, target_ts_us: int):
        """指定時刻に最も近いFlowフレームを取得"""
        idx = np.searchsorted(self.timestamps, target_ts_us, side="left")
        
        # 境界チェックと最近傍探索
        if idx == 0:
            best_idx = 0
        elif idx == len(self.timestamps):
            best_idx = len(self.timestamps) - 1
        else:
            dt_prev = abs(self.timestamps[idx-1] - target_ts_us)
            dt_curr = abs(self.timestamps[idx] - target_ts_us)
            best_idx = idx - 1 if dt_prev < dt_curr else idx
            
        return self.h5f['flow'][best_idx]


# ==========================================
# Filtering Logic
# ==========================================

def prophesee_bbox_filter(labels: np.ndarray, dataset_type: str) -> np.ndarray:
    if dataset_type == 'SEVD':
        min_box_diag = 60
        min_box_side = 20
    else:
        raise NotImplementedError
    
    w_lbl = labels['w']
    h_lbl = labels['h']
    diag_ok = w_lbl ** 2 + h_lbl ** 2 >= min_box_diag ** 2
    side_ok = (w_lbl >= min_box_side) & (h_lbl >= min_box_side)
    keep = diag_ok & side_ok
    return labels[keep]

def conservative_bbox_filter(labels: np.ndarray) -> np.ndarray:
    w_lbl = labels['w']
    h_lbl = labels['h']
    min_box_side = 5
    side_ok = (w_lbl >= min_box_side) & (h_lbl >= min_box_side)
    return labels[side_ok]

def remove_faulty_huge_bbox_filter(labels: np.ndarray, dataset_type: str) -> np.ndarray:
    w_lbl = labels['w']
    max_width = (9 * dataset_2_width[dataset_type]) // 10
    side_ok = (w_lbl <= max_width)
    return labels[side_ok]

def crop_to_fov_filter(labels: np.ndarray, dataset_type: str) -> np.ndarray:
    frame_height = dataset_2_height[dataset_type]
    frame_width = dataset_2_width[dataset_type]
    
    x_left = labels['x']
    y_top = labels['y']
    x_right = x_left + labels['w']
    y_bottom = y_top + labels['h']
    x_left_cropped = np.clip(x_left, a_min=0, a_max=frame_width - 1)
    y_top_cropped = np.clip(y_top, a_min=0, a_max=frame_height - 1)
    x_right_cropped = np.clip(x_right, a_min=0, a_max=frame_width - 1)
    y_bottom_cropped = np.clip(y_bottom, a_min=0, a_max=frame_height - 1)

    w_cropped = x_right_cropped - x_left_cropped
    h_cropped = y_bottom_cropped - y_top_cropped

    labels['x'] = x_left_cropped
    labels['y'] = y_top_cropped
    labels['w'] = w_cropped
    labels['h'] = h_cropped
    keep = (labels['w'] > 0) & (labels['h'] > 0)
    return labels[keep]

def apply_filters(labels: np.ndarray, split_type: SplitType, filter_cfg: DictConfig, dataset_type: str = 'SEVD') -> np.ndarray:
    labels = crop_to_fov_filter(labels=labels, dataset_type=dataset_type)
    if filter_cfg.apply_psee_bbox_filter:
        labels = prophesee_bbox_filter(labels=labels, dataset_type=dataset_type)
    else:
        labels = conservative_bbox_filter(labels=labels)
    if split_type == SplitType.TRAIN and filter_cfg.apply_faulty_bbox_filter:
        labels = remove_faulty_huge_bbox_filter(labels=labels, dataset_type=dataset_type)
    return labels

def get_base_delta_ts_for_labels_us(unique_label_ts_us: np.ndarray, dataset_type: str = 'SEVD') -> int:
    diff_us = np.diff(unique_label_ts_us)
    median_diff_us = np.median(diff_us)
    hz = int(np.rint(10 ** 6 / median_diff_us))
    
    if hz >= 50: # 60Hz
        return int(6 * median_diff_us) # approx 100ms
    elif hz >= 25: # 30Hz
        return int(3 * median_diff_us)
    else: # 20Hz or others
        return int(2 * median_diff_us)

# ==========================================
# Processing Logic
# ==========================================

def save_labels(out_labels_dir: Path, labels_per_frame: List[np.ndarray], frame_timestamps_us: np.ndarray, match_if_exists: bool = True) -> None:
    assert len(labels_per_frame) == len(frame_timestamps_us)
    labels_v2 = list()
    objframe_idx_2_label_idx = list()
    start_idx = 0
    for labels, timestamp in zip(labels_per_frame, frame_timestamps_us):
        objframe_idx_2_label_idx.append(start_idx)
        labels_v2.append(labels)
        start_idx += len(labels)
    labels_v2 = np.concatenate(labels_v2)

    outfile_labels = out_labels_dir / 'labels.npz'
    if outfile_labels.exists() and match_if_exists:
        return 
    else:
        np.savez(str(outfile_labels), labels=labels_v2, objframe_idx_2_label_idx=objframe_idx_2_label_idx)

    out_labels_ts_file = out_labels_dir / 'timestamps_us.npy'
    if not out_labels_ts_file.exists():
        np.save(str(out_labels_ts_file), frame_timestamps_us)

def labels_and_ev_repr_timestamps(npy_file: Path, split_type: SplitType, filter_cfg: DictConfig, align_t_ms: int, ts_step_ev_repr_ms: int, dataset_type: str):
    sequence_labels = np.load(str(npy_file))
    sequence_labels = apply_filters(labels=sequence_labels, split_type=split_type, filter_cfg=filter_cfg, dataset_type=dataset_type)
    if sequence_labels.size == 0:
        raise NoLabelsException

    unique_ts_us = np.unique(np.asarray(sequence_labels['t'], dtype='int64'))
    base_delta_ts_labels_us = get_base_delta_ts_for_labels_us(unique_label_ts_us=unique_ts_us, dataset_type=dataset_type)
    align_t_us = align_t_ms * 1000
    delta_t_us = ts_step_ev_repr_ms * 1000

    unique_ts_idx_first = np.searchsorted(unique_ts_us, align_t_us, side='left')
    
    frame_timestamps_us = [unique_ts_us[unique_ts_idx_first]]
    num_ev_reprs_between_frame_ts = []
    
    ts_step_frame_ms = 100 
    for unique_ts_idx in range(unique_ts_idx_first + 1, len(unique_ts_us)):
        reference_time = frame_timestamps_us[-1]
        ts = unique_ts_us[unique_ts_idx]
        diff_to_ref = ts - reference_time
        base_delta_count = round(diff_to_ref / base_delta_ts_labels_us)
        diff_to_ref_rounded = base_delta_count * base_delta_ts_labels_us
        
        if np.abs(diff_to_ref - diff_to_ref_rounded) <= 2000:
            frame_timestamps_us.append(ts)
            num_ev_reprs_between_frame_ts.append(base_delta_count * (ts_step_frame_ms // ts_step_ev_repr_ms))
            
    frame_timestamps_us = np.asarray(frame_timestamps_us, dtype='int64')
    start_indices_per_label = np.searchsorted(sequence_labels['t'], frame_timestamps_us, side='left')
    end_indices_per_label = np.searchsorted(sequence_labels['t'], frame_timestamps_us, side='right')

    labels_per_frame = []
    for idx_start, idx_end in zip(start_indices_per_label, end_indices_per_label):
        labels_per_frame.append(sequence_labels[idx_start:idx_end])

    ev_repr_timestamps_us_end = list(reversed(range(frame_timestamps_us[0], 0, -delta_t_us)))[1:-1]
    for idx, (num_ev_repr_between, frame_ts_us_start, frame_ts_us_end) in enumerate(zip(num_ev_reprs_between_frame_ts, frame_timestamps_us[:-1], frame_timestamps_us[1:])):
        new_edge_timestamps = np.asarray(np.linspace(frame_ts_us_start, frame_ts_us_end, num_ev_repr_between + 1), dtype='int64').tolist()
        if idx != len(num_ev_reprs_between_frame_ts) - 1:
            new_edge_timestamps = new_edge_timestamps[:-1]
        ev_repr_timestamps_us_end.extend(new_edge_timestamps)
    
    if len(frame_timestamps_us) == 1:
         ev_repr_timestamps_us_end.append(frame_timestamps_us[0])
         
    ev_repr_timestamps_us_end = np.asarray(ev_repr_timestamps_us_end, dtype='int64')
    frameidx_2_repridx = np.searchsorted(ev_repr_timestamps_us_end, frame_timestamps_us, side='left')

    return labels_per_frame, frame_timestamps_us, ev_repr_timestamps_us_end, frameidx_2_repridx

def write_event_representations(in_h5_file: Path, ev_out_dir: Path, dataset: str, event_representation: RepresentationBase, ev_repr_num_events: Optional[int], ev_repr_delta_ts_ms: Optional[int], ev_repr_timestamps_us: np.ndarray, overwrite_if_exists: bool = False) -> None:
    ev_outfile = ev_out_dir / "event_representations.h5"
    if ev_outfile.exists() and not overwrite_if_exists:
        return
    ev_outfile_in_progress = ev_outfile.parent / (ev_outfile.stem + '_in_progress' + ev_outfile.suffix)
    if ev_outfile_in_progress.exists():
        os.remove(ev_outfile_in_progress)
    
    ev_repr_shape = tuple(event_representation.get_shape())
    ev_repr_dtype = event_representation.get_numpy_dtype()
    
    with H5Reader(in_h5_file, dataset=dataset) as h5_reader, \
            H5Writer(ev_outfile_in_progress, key='data', ev_repr_shape=ev_repr_shape, numpy_dtype=ev_repr_dtype) as h5_writer:
        
        h_reader, w_reader = h5_reader.get_height_and_width()
        assert (h_reader, w_reader) == ev_repr_shape[-2:], f"Mismatch: H5({h_reader}x{w_reader}) vs Repr({ev_repr_shape[-2:]})"

        ev_ts_us = h5_reader.time
        end_indices = np.searchsorted(ev_ts_us, ev_repr_timestamps_us, side='right')
        
        if ev_repr_num_events is not None:
            start_indices = np.maximum(end_indices - ev_repr_num_events, 0)
        else:
            start_indices = np.searchsorted(ev_ts_us, ev_repr_timestamps_us - ev_repr_delta_ts_ms * 1000, side='left')

        for idx_start, idx_end in zip(start_indices, end_indices):
            ev_window = h5_reader.get_event_slice(idx_start=idx_start, idx_end=idx_end)
            ev_repr = event_representation.construct(x=ev_window['x'], y=ev_window['y'], pol=ev_window['p'], time=ev_window['t'])
            h5_writer.add_data(ev_repr.numpy())
            
    os.rename(ev_outfile_in_progress, ev_outfile)

def write_synced_flow(flow_h5_path: Path, out_dir: Path, target_timestamps_us: np.ndarray):
    """
    イベント表現のタイムスタンプに合わせて、Optical Flowを同期して保存する
    """
    out_flow_file = out_dir / "flow_ground_truth.h5"
    if out_flow_file.exists():
        return

    if not flow_h5_path.exists():
        # Flowがない場合はスキップ (警告は出す)
        print(f"Warning: Flow file not found: {flow_h5_path}")
        return

    with FlowReader(flow_h5_path) as flow_reader:
        # Flowデータの形状を取得
        try:
            sample_flow = flow_reader.h5f['flow'][0]
        except Exception as e:
            print(f"Error reading flow data from {flow_h5_path}: {e}")
            return
            
        h, w, c = sample_flow.shape
        num_frames = len(target_timestamps_us)
        
        # 出力用H5作成 (in_progressを使用)
        out_flow_file_in_progress = out_dir / (out_flow_file.stem + '_in_progress' + out_flow_file.suffix)
        
        with h5py.File(str(out_flow_file_in_progress), 'w') as f_out:
            dset = f_out.create_dataset('flow', shape=(num_frames, h, w, c), 
                                      dtype='float32', chunks=(1, h, w, c), 
                                      compression="gzip", compression_opts=4)
            dset_ts = f_out.create_dataset('timestamps', data=target_timestamps_us)

            for i, ts in enumerate(target_timestamps_us):
                synced_flow = flow_reader.get_nearest_flow(ts)
                dset[i] = synced_flow

    os.rename(out_flow_file_in_progress, out_flow_file)


def process_sequence(dataset: str,
                     filter_cfg: DictConfig,
                     event_representation: RepresentationBase,
                     ev_repr_num_events: Optional[int],
                     ev_repr_delta_ts_ms: Optional[int],
                     ts_step_ev_repr_ms: int,
                     sequence_data: Dict[DataKeys, Union[Path, SplitType]]):
    
    in_npy_file = sequence_data[DataKeys.InNPY]
    in_h5_file = sequence_data[DataKeys.InH5]
    in_flow_h5_file = sequence_data[DataKeys.InFlowH5] # 追加
    out_labels_dir = sequence_data[DataKeys.OutLabelDir]
    out_ev_repr_dir = sequence_data[DataKeys.OutEvReprDir]
    split_type = sequence_data[DataKeys.SplitType]
    
    if not in_npy_file.exists():
        return

    try:
        labels_per_frame, frame_timestamps_us, ev_repr_timestamps_us, frameidx2repridx = \
            labels_and_ev_repr_timestamps(
                npy_file=in_npy_file,
                split_type=split_type,
                filter_cfg=filter_cfg,
                align_t_ms=100,
                ts_step_ev_repr_ms=ts_step_ev_repr_ms,
                dataset_type=dataset)
    except NoLabelsException:
        print(f"No labels for {in_npy_file}, skipping.")
        return
    except ValueError as e:
        error_msg = f"!!! ERROR in file: {in_npy_file} (Seq: {in_npy_file.parent.parent.name}) !!! -> {e}"
        raise ValueError(error_msg) from e

    # Labels Saving
    save_labels(out_labels_dir=out_labels_dir, labels_per_frame=labels_per_frame, frame_timestamps_us=frame_timestamps_us)
    
    # Metadata Saving
    frameidx2repridx_file = out_ev_repr_dir / 'objframe_idx_2_repr_idx.npy'
    if not frameidx2repridx_file.exists():
        np.save(str(frameidx2repridx_file), frameidx2repridx)
    timestamps_file = out_ev_repr_dir / 'timestamps_us.npy'
    if not timestamps_file.exists():
        np.save(str(timestamps_file), ev_repr_timestamps_us)

    # Event Representation Saving
    write_event_representations(in_h5_file=in_h5_file,
                                ev_out_dir=out_ev_repr_dir,
                                dataset=dataset,
                                event_representation=event_representation,
                                ev_repr_num_events=ev_repr_num_events,
                                ev_repr_delta_ts_ms=ev_repr_delta_ts_ms,
                                ev_repr_timestamps_us=ev_repr_timestamps_us,
                                overwrite_if_exists=False)

    write_synced_flow(
        flow_h5_path=in_flow_h5_file,
        out_dir=out_ev_repr_dir,
        target_timestamps_us=ev_repr_timestamps_us
    )


# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('target_dir')
    parser.add_argument('split_yaml', help='Path to split definition yaml file')
    parser.add_argument('ev_repr_yaml_config', help='Path to event representation yaml config file')
    parser.add_argument('extraction_yaml_config', help='Path to event window extraction yaml config file')
    parser.add_argument('bbox_filter_yaml_config', help='Path to bbox filter yaml config file')
    parser.add_argument('-d', '--dataset', default='SEVD', help='SEVD') 
    parser.add_argument('--downsample', action='store_true', help="Use events_ds.h5 and 1/2 resolution")
    parser.add_argument('-np', '--num_processes', type=int, default=1)
    parser.add_argument('--filtered_label', action='store_true', help="Use pre-filtered labels (if available)")
    parser.add_argument('--ignore_yaml', default=None, help='Path to ignore definition yaml file')
    args = parser.parse_args()

    # Config読み込み
    config = get_configuration(ev_repr_yaml_config=Path(args.ev_repr_yaml_config),
                               extraction_yaml_config=Path(args.extraction_yaml_config))

    bbox_filter_yaml_config = Path(args.bbox_filter_yaml_config)
    filter_cfg = OmegaConf.load(str(bbox_filter_yaml_config))
    filter_cfg = OmegaConf.merge(OmegaConf.structured(FilterConf), filter_cfg)
    split_config = OmegaConf.load(str(args.split_yaml))

    # ==========================================
    # Ignore List (除外リスト) の読み込み
    # ==========================================
    dirs_to_ignore = {}
    if args.ignore_yaml:
        ignore_yaml_path = Path(args.ignore_yaml)
        if ignore_yaml_path.exists():
            print(f"Loading ignore list from: {ignore_yaml_path}")
            dirs_to_ignore = OmegaConf.load(ignore_yaml_path)
        else:
            print(f"Warning: Ignore YAML provided but not found: {ignore_yaml_path}")

    dataset_input_path = Path(args.input_dir)
    target_dir = Path(args.target_dir)

    # --- Step 1: 解像度と対象ファイル名の決定 ---
    
    # 基本解像度
    base_h = dataset_2_height.get(args.dataset, 960)
    base_w = dataset_2_width.get(args.dataset, 1280)
    
    # フラグによる分岐
    if args.downsample:
        height = base_h // 2
        width = base_w // 2
        target_h5_filename = "events_ds.h5"
        # Flowファイル名の決定 (downsample時)
        target_flow_filename = "optical_flow_synced_ds.h5"
        print(f"Mode: [Downsample ON] Using '{target_h5_filename}' and '{target_flow_filename}' (Size: {width}x{height})")
    else:
        height = base_h
        width = base_w
        target_h5_filename = "events.h5"
        # Flowファイル名の決定 (通常時)
        target_flow_filename = "optical_flow_synced.h5"
        print(f"Mode: [Standard] Using '{target_h5_filename}' and '{target_flow_filename}' (Size: {width}x{height})")

    if args.filtered_label:
        target_label_suffix = '_filtered'
    else:
        target_label_suffix = ''

    # Factory作成
    ev_repr_factory = name_2_ev_repr_factory[config.name](config)
    ev_repr = ev_repr_factory.create(height=height, width=width)
    ev_repr_string = ev_repr_factory.name
    
    print(f"Event Representation: {ev_repr_string}")

    # --- Step 2: シーケンスリスト作成 ---
    seq_data_list = list()
    for split_name in ['train', 'val', 'test']:
        if split_name not in split_config or not split_config[split_name]:
            continue

        sequence_ids = split_config[split_name]
        split_out_dir = target_dir / split_name
        os.makedirs(split_out_dir, exist_ok=True)

        for sequence_id in sequence_ids:
            if args.dataset in dirs_to_ignore:
                if sequence_id in dirs_to_ignore[args.dataset]:
                    print(f"Ignoring sequence (Blacklisted): {sequence_id}")
                    continue

            seq_dir = dataset_input_path / sequence_id
            
            npy_file = seq_dir / "labels" / f"labels_bbox{target_label_suffix}.npy"
            h5f_path = seq_dir / "events" / target_h5_filename
            flow_h5_path = seq_dir / "optical_flow_processed" / target_flow_filename

            if not npy_file.exists():
                continue
            
            if not h5f_path.exists():
                continue

            # 出力先
            out_seq_path = split_out_dir / sequence_id
            out_labels_path = out_seq_path / 'labels_v2'
            os.makedirs(out_labels_path, exist_ok=True)
            
            out_ev_repr_path = out_seq_path / 'event_representations_v2' / ev_repr_string
            os.makedirs(out_ev_repr_path, exist_ok=True)

            sequence_data = {
                DataKeys.InNPY: npy_file,
                DataKeys.InH5: h5f_path,
                DataKeys.InFlowH5: flow_h5_path,
                DataKeys.OutLabelDir: out_labels_path,
                DataKeys.OutEvReprDir: out_ev_repr_path,
                DataKeys.SplitType: split_name_2_type[split_name],
            }
            seq_data_list.append(sequence_data)

    print(f"Target Sequences: {len(seq_data_list)}")

    # --- Step 3: 処理実行 ---
    ev_repr_num_events = None
    ev_repr_delta_ts_ms = None
    if config.event_window_extraction.method == AggregationType.COUNT:
        ev_repr_num_events = config.event_window_extraction.value
        ts_step_ev_repr_ms = config.event_window_extraction.ts_step_ev_repr_ms
    else:
        ev_repr_delta_ts_ms = config.event_window_extraction.value
        ts_step_ev_repr_ms = config.event_window_extraction.ts_step_ev_repr_ms

    if args.num_processes > 1:
        chunksize = 1
        func = partial(process_sequence,
                       args.dataset,
                       filter_cfg,
                       ev_repr,
                       ev_repr_num_events,
                       ev_repr_delta_ts_ms,
                       ts_step_ev_repr_ms)
        
        with get_context('spawn').Pool(args.num_processes) as pool:
            with tqdm(total=len(seq_data_list), desc='sequences') as pbar:
                for _ in pool.imap_unordered(func, iterable=seq_data_list, chunksize=chunksize):
                    pbar.update()
    else:
        for entry in tqdm(seq_data_list, desc='sequences'):
            process_sequence(dataset=args.dataset,
                             filter_cfg=filter_cfg,
                             event_representation=ev_repr,
                             ev_repr_num_events=ev_repr_num_events,
                             ev_repr_delta_ts_ms=ev_repr_delta_ts_ms,
                             ts_step_ev_repr_ms=ts_step_ev_repr_ms,
                             sequence_data=entry)