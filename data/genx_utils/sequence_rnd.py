import torch
from pathlib import Path

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.sequence_base import SequenceBase
from data.utils.types import DataType, DatasetType, LoaderDataDictGenX
from utils.timers import TimerDummy as Timer


class SequenceForRandomAccess(SequenceBase):
    def __init__(self,
                 path: Path,
                 ev_representation_name: str,
                 sequence_length: int,
                 dataset_type: DatasetType,
                 downsample_by_factor_2: bool,
                 only_load_end_labels: bool,
                 use_flow: bool = False):  
        super().__init__(path=path,
                         ev_representation_name=ev_representation_name,
                         sequence_length=sequence_length,
                         dataset_type=dataset_type,
                         downsample_by_factor_2=downsample_by_factor_2,
                         only_load_end_labels=only_load_end_labels,
                         use_flow=use_flow)  

        self.start_idx_offset = None
        for objframe_idx, repr_idx in enumerate(self.objframe_idx_2_repr_idx):
            if repr_idx - self.seq_len + 1 >= 0:
                self.start_idx_offset = objframe_idx
                break
        if self.start_idx_offset is None:
            self.start_idx_offset = len(self.label_factory)

        self.length = len(self.label_factory) - self.start_idx_offset
        assert len(self.label_factory) == len(self.objframe_idx_2_repr_idx)

        self._only_load_labels = False

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        corrected_idx = index + self.start_idx_offset
        labels_repr_idx = self.objframe_idx_2_repr_idx[corrected_idx]

        end_idx = labels_repr_idx + 1
        start_idx = end_idx - self.seq_len
        assert start_idx >= 0

        # labels
        labels = list()
        for repr_idx in range(start_idx, end_idx):
            if self.only_load_end_labels and repr_idx < end_idx - 1:
                labels.append(None)
            else:
                labels.append(self._get_labels_from_repr_idx(repr_idx))
        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)
        
        if self._only_load_labels:
            return {DataType.OBJLABELS_SEQ: sparse_labels}

        # event representations
        with Timer(timer_name='read ev reprs'):
            ev_repr = self._get_event_repr_torch(start_idx=start_idx, end_idx=end_idx)

        # Output dictionary
        out = {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: True,
            DataType.IS_PADDED_MASK: [False] * len(ev_repr),
        }

        if self.use_flow:
            # flow
            if self.has_flow:
                with Timer(timer_name='read flow'):
                    flow = self._get_flow_torch(start_idx=start_idx, end_idx=end_idx)
                assert len(flow) == len(ev_repr)
            else:
                h, w = ev_repr[0].shape[-2:]
                zero_flow = torch.zeros((2, h, w), dtype=torch.float32)
                flow = [zero_flow] * len(ev_repr)
            
            # valid mask
            if self.has_valid:
                with Timer(timer_name='read valid'):
                    valid = self._get_valid_torch(start_idx=start_idx, end_idx=end_idx)
                assert len(valid) == len(ev_repr)
            else:
                h, w = ev_repr[0].shape[-2:]
                ones_valid = torch.ones((1, h, w), dtype=torch.float32)
                valid = [ones_valid] * len(ev_repr)
            
            out[DataType.FLOW] = flow
            out[DataType.VALID] = valid

        assert len(sparse_labels) == len(ev_repr)
        return out

    def is_only_loading_labels(self) -> bool:
        return self._only_load_labels

    def only_load_labels(self):
        self._only_load_labels = True

    def load_everything(self):
        self._only_load_labels = False