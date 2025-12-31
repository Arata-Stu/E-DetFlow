from typing import Any, Optional, Tuple, Union, Dict
from warnings import warn

import numpy as np
import lightning.pytorch as pl
import torch
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig
from lightning.pytorch.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode
from models.EFDNet import EFDNet
from models.detection.yolox.utils import postprocess
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from utils.evaluation.optical_flow.eval import compute_flow_metrics
from utils.padding import InputPadderFromShape
from modules.utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches


class ModelModule(pl.LightningModule):
    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config
        self.mdl_config = full_config.model

        self.in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=self.in_res_hw)

        self.mdl = EFDNet(self.mdl_config)

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }

        self.flow_loss_weight = self.full_config.training.get('flow_loss_weight', 1.0)
        self.det_loss_weight = self.full_config.training.get('det_loss_weight', 1.0)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_name = self.full_config.dataset.name
        self.mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
        self.mode_2_batch_size: Dict[Mode, Optional[int]] = {}
        self.mode_2_psee_evaluator: Dict[Mode, Optional[PropheseeEvaluator]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)

        if stage == 'fit':  # training + validation
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            self.mode_2_psee_evaluator[Mode.VAL] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            
            if self.train_metrics_config.compute:
                self.mode_2_psee_evaluator[Mode.TRAIN] = PropheseeEvaluator(
                    dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)

            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False

        elif stage == 'validate':
            mode = Mode.VAL
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None

        elif stage == 'test':
            mode = Mode.TEST
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError

    def forward(self,
                x: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                flow_gt: Optional[th.Tensor] = None,
                valid_mask: Optional[th.Tensor] = None,
                det_targets: Optional[th.Tensor] = None) -> \
            Tuple[Union[Dict[str, th.Tensor], None], Union[Dict[str, th.Tensor], None], LstmStates]:
        return self.mdl(x=x,
                        previous_states=previous_states,
                        retrieve_detections=retrieve_detections,
                        flow_gt=flow_gt,
                        valid_mask=valid_mask,
                        det_targets=det_targets)

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
        step = self.trainer.global_step
        
        ev_tensor_sequence = data[DataType.EV_REPR]        # [Seq, Batch, C, H, W]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]   # List[ObjectLabels] len=Seq
        flow_tensor_sequence = data[DataType.FLOW]         # [Seq, Batch, 2, H, W]
        valid_mask_sequence = data[DataType.VALID]         # [Seq, Batch, 1, H, W]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        
        backbone_feature_selector = BackboneFeatureSelector()
        
        det_labels_flat = []    # Detectionラベル (蓄積用)
        flow_targets_flat = []  # Flowターゲット (蓄積用)
        flow_masks_flat = []    # Flowマスク (蓄積用)

        # --- 時間方向ループ ---
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx].to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            
            token_masks = None
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors,
                                                                  previous_states=prev_states,
                                                                  token_mask=token_masks)
            prev_states = states

            current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            
            if len(current_labels) > 0:
                
                backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                selected_indices=valid_batch_indices)
                
                det_labels_flat.extend(current_labels)
                current_flow = flow_tensor_sequence[tidx].to(dtype=self.dtype)
                current_mask = valid_mask_sequence[tidx].to(dtype=self.dtype)
                
                selected_flow = current_flow[valid_batch_indices]  # -> [Valid_N, 2, H, W]
                selected_mask = current_mask[valid_batch_indices]  # -> [Valid_N, 1, H, W]
                
                flow_targets_flat.append(selected_flow)
                flow_masks_flat.append(selected_mask)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)

        total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        log_dict = {}

        if len(det_labels_flat) > 0:
            
            batched_backbone_features = backbone_feature_selector.get_batched_backbone_features()
            
            batched_det_targets = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=det_labels_flat, format_='yolox')
            batched_det_targets = batched_det_targets.to(dtype=self.dtype)
            
            batched_flow_gt = torch.cat(flow_targets_flat, dim=0)
            batched_flow_mask = torch.cat(flow_masks_flat, dim=0)

            outputs, losses = self.mdl.forward_heads(
                backbone_features=batched_backbone_features,
                det_targets=batched_det_targets,
                flow_gt=batched_flow_gt,
                valid_mask=batched_flow_mask
            )

            if losses is not None:
                # Detection Loss
                if 'loss' in losses:
                    l_det = losses['loss']
                    total_loss += self.det_loss_weight * l_det
                    log_dict[f'{mode_2_string[mode]}/loss_det'] = l_det.detach()
                
                # Flow Loss
                if 'loss_flow' in losses:
                    l_flow = losses['loss_flow']
                    total_loss += self.flow_loss_weight * l_flow
                    log_dict[f'{mode_2_string[mode]}/loss_flow'] = l_flow.detach()

        # ログ記録
        log_dict[f'{mode_2_string[mode]}/loss_total'] = total_loss.detach()
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        return {'loss': total_loss}

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        flow_tensor_sequence = data[DataType.FLOW]
        valid_mask_sequence = data[DataType.VALID]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        batch_size = len(sparse_obj_labels[0])
        
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        
        backbone_feature_selector = BackboneFeatureSelector()
        obj_labels_list = list()
        flow_gt_flat = list()
        flow_mask_flat = list()

        # --- 1. 時間方向ループ: 抽出処理 ---
        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or \
                                  (self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM)

            ev_tensors = ev_tensor_sequence[tidx].to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors, previous_states=prev_states)
            prev_states = states

            if collect_predictions:
                current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                
                if len(current_labels) > 0:
                    backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                    selected_indices=valid_batch_indices)
                    obj_labels_list.extend(current_labels)
                    
                    current_flow = flow_tensor_sequence[tidx].to(dtype=self.dtype)
                    current_mask = valid_mask_sequence[tidx].to(dtype=self.dtype)
                    flow_gt_flat.append(current_flow[valid_batch_indices])
                    flow_mask_flat.append(current_mask[valid_batch_indices])

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)

        if len(obj_labels_list) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}

        # --- 2. Headの一括実行 ---
        batched_features = backbone_feature_selector.get_batched_backbone_features()
        outputs, _ = self.mdl.forward_heads(backbone_features=batched_features)

        # --- 3. Detectionの評価器へデータ追加 ---
        det_preds = outputs.get('detection', None)
        if det_preds is not None:
            pred_processed = postprocess(prediction=det_preds,
                                         num_classes=self.mdl_config.head.num_classes,
                                         conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                         nms_thre=self.mdl_config.postprocess.nms_threshold)
            
            loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels_list, pred_processed)
            
            if self.mode_2_psee_evaluator[mode] is not None:
                self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
                self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)

        # --- 4. Flowの一括評価とログ出力 ---
        flow_preds = outputs.get('flow', None)
        if flow_preds is not None and len(flow_gt_flat) > 0:
            batched_flow_gt = torch.cat(flow_gt_flat, dim=0).to(flow_preds.device)
            batched_flow_mask = torch.cat(flow_mask_flat, dim=0).to(flow_preds.device)
            
            orig_h, orig_w = flow_gt_flat[0].shape[-2:]
            flow_preds = flow_preds[..., :orig_h, :orig_w]

            metrics = compute_flow_metrics(flow_preds, batched_flow_gt, batched_flow_mask)
            
            if metrics:
                prefix = f'{mode_2_string[mode]}/'
                for k, v in metrics.items():
                    val = v.mean() if isinstance(v, torch.Tensor) else v
                    self.log(f'{prefix}{k}', val, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        return {ObjDetOutput.SKIP_VIZ: False}

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.VAL)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST)

    def run_psee_evaluator(self, mode: Mode):
        psee_evaluator = self.mode_2_psee_evaluator[mode]
        batch_size = self.mode_2_batch_size[mode]
        hw_tuple = self.mode_2_hw[mode]
        
        if psee_evaluator is None:
            return

        if psee_evaluator.has_data():
            metrics = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0],
                                                     img_width=hw_tuple[1])
            
            prefix = f'{mode_2_string[mode]}/'
            step = self.trainer.global_step
            log_dict = {}
            
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                else:
                    value = v
                log_dict[f'{prefix}{k}'] = value.to(self.device) 

            # Distributed Trainingの同期
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                for k, v in log_dict.items():
                    dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0:
                        log_dict[k] /= dist.get_world_size()

            if self.trainer.is_global_zero:
                add_hack = 2
                self.logger.log_metrics(metrics=log_dict, step=step + add_hack)

            psee_evaluator.reset_buffer()

    def on_train_epoch_end(self) -> None:
        pass 

    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        if self.started_training and self.mode_2_psee_evaluator[mode] is not None:
            if self.mode_2_psee_evaluator[mode].has_data():
                self.run_psee_evaluator(mode=mode)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        if self.mode_2_psee_evaluator[mode] is not None:
             if self.mode_2_psee_evaluator[mode].has_data():
                self.run_psee_evaluator(mode=mode)

    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = th.optim.AdamW(self.mdl.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
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
        
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "strict": True,
                "name": 'learning_rate',
            }
        }