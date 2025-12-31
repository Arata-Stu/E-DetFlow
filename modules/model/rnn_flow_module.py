from typing import Any, Optional, Tuple, Union, Dict, List
from warnings import warn

import lightning.pytorch as pl
import torch
import torch as th
from omegaconf import DictConfig
from lightning.pytorch.utilities.types import STEP_OUTPUT
from utils.evaluation.optical_flow.eval import compute_flow_metrics 

from data.utils.types import DataType, LstmStates, DatasetSamplingMode
from models.flow.flow_estimator import FlowEstimator
from utils.padding import InputPadderFromShape
from modules.utils.detection import BackboneFeatureSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches


class ModelModule(pl.LightningModule):
    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config
        self.mdl_config = full_config.model

        self.in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=self.in_res_hw)

        self.mdl = FlowEstimator(self.mdl_config)

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }

    def setup(self, stage: Optional[str] = None) -> None:
        self.mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
        self.mode_2_batch_size: Dict[Mode, Optional[int]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        
        if stage == 'fit':  # train + val
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False
            
        elif stage == 'validate':
            mode = Mode.VAL
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
            
        elif stage == 'test':
            mode = Mode.TEST
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError

    def forward(self,
                event_tensor: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_flow: bool = True,
                flow_gt: Optional[th.Tensor] = None,      
                valid_mask: Optional[th.Tensor] = None): 

        return self.mdl(x=event_tensor,
                        previous_states=previous_states,
                        retrieve_flow=retrieve_flow,
                        flow_gt=flow_gt,                 
                        valid_mask=valid_mask)

    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch['worker_id']

    def get_data_from_batch(self, batch: Any):
        return batch['data']

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        mode = Mode.TRAIN
        self.started_training = True
        
        ev_tensor_sequence = data[DataType.EV_REPR]
        flow_tensor_sequence = data[DataType.FLOW]
        valid_mask_sequence = data[DataType.VALID]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = ev_tensor_sequence[0].shape[0]
        
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        
        flow_targets = list()
        flow_valid_masks = list()
        all_batch_indices = torch.arange(batch_size, device=self.device, dtype=torch.long)

        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx].to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)

            token_masks = None
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors,
                                                                  previous_states=prev_states,
                                                                  token_mask=token_masks)
            prev_states = states

            backbone_feature_selector.add_backbone_features(
                backbone_features=backbone_features,
                selected_indices=all_batch_indices
            )

            current_flows = flow_tensor_sequence[tidx].to(dtype=self.dtype)
            current_valid_masks = valid_mask_sequence[tidx].to(dtype=self.dtype)
            flow_targets.append(current_flows)
            flow_valid_masks.append(current_valid_masks)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        predictions, losses = self.mdl.forward_flow(
            backbone_features=selected_backbone_features,
            flow_gt=flow_targets,         
            valid_mask=flow_valid_masks   
        )

        assert losses is not None
        assert 'loss_flow' in losses

        output = {'loss': losses['loss_flow']}

        prefix = f'{mode_2_string[mode]}/'
        self.log_dict({f'{prefix}{k}': v for k, v in losses.items()}, 
                      on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=True)

        return output

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        ev_tensor_sequence = data[DataType.EV_REPR]
        flow_tensor_sequence = data[DataType.FLOW]
        valid_mask_sequence = data[DataType.VALID]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = ev_tensor_sequence[0].shape[0]
        
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        
        flow_targets = list()
        flow_valid_masks = list()
        all_batch_indices = torch.arange(batch_size, device=self.device, dtype=torch.long)
        
        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or \
                                  (self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM)
            
            ev_tensors = ev_tensor_sequence[tidx].to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors, previous_states=prev_states)
            prev_states = states

            if collect_predictions:
                backbone_feature_selector.add_backbone_features(
                    backbone_features=backbone_features,
                    selected_indices=all_batch_indices
                )
                
                current_flows = flow_tensor_sequence[tidx].to(dtype=self.dtype)
                current_valid_masks = valid_mask_sequence[tidx].to(dtype=self.dtype)
                flow_targets.append(current_flows)
                flow_valid_masks.append(current_valid_masks)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        predictions, losses = self.mdl.forward_flow(
            backbone_features=selected_backbone_features,
            flow_gt=flow_targets,         
            valid_mask=flow_valid_masks   
        )

        orig_h, orig_w = flow_targets[0].shape[-2:]
        predictions = predictions[..., :orig_h, :orig_w]

        full_gt = torch.cat(flow_targets, dim=0).to(predictions.device)
        full_mask = torch.cat(flow_valid_masks, dim=0).to(predictions.device)

        metrics = compute_flow_metrics(predictions, full_gt, full_mask)

        prefix = f'{mode_2_string[mode]}'
        
        # ログ出力処理
        if metrics:
            for k, v in metrics.items():
                val = v.mean() if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                self.log(f'{prefix}/{k}', val, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        if losses is not None and 'loss_flow' in losses:
            self.log(f'{prefix}/loss', losses['loss_flow'], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        return None

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.VAL)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST)


    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = th.optim.AdamW(self.mdl.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}