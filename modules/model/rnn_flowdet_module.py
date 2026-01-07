from typing import Any, Optional, Tuple, Union, Dict, List
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

        self.mode_2_flow_metrics_buffer: Dict[Mode, List[Dict[str, torch.Tensor]]] = {
            Mode.VAL: [],
            Mode.TEST: [],
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
        
        ev_tensor_sequence = data[DataType.EV_REPR]        # [Seq, Batch, C, H, W]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]   # List[ObjectLabels] len=Seq
        flow_tensor_sequence = data[DataType.FLOW]         # [Seq, Batch, 2, H, W]
        valid_mask_sequence = data[DataType.VALID]         # [Seq, Batch, 1, H, W]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        batch_size = ev_tensor_sequence[0].shape[0]
        
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        
        # --- セレクターとバッファの初期化 ---
        flow_selector = BackboneFeatureSelector() # Flow用 (全フレーム・全バッチ)
        all_batch_indices = torch.arange(batch_size, device=self.device)
        
        det_labels_flat = []      # 有効なDetectionラベル
        det_indices_in_flat = []  # 全サンプル中のどこにDetラベルがあるか
        flow_targets_list = []    # Flowターゲット
        flow_masks_list = []      # Flowマスク

        # --- 1. 時間方向ループ (Backbone特徴抽出) ---
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx].to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            
            token_masks = None
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])

            backbone_features, states = self.mdl.forward_backbone(
                x=ev_tensors, previous_states=prev_states, token_mask=token_masks
            )
            prev_states = states

            # A. Flow用: 全件を蓄積
            flow_selector.add_backbone_features(backbone_features, all_batch_indices)
            flow_targets_list.append(flow_tensor_sequence[tidx].to(dtype=self.dtype))
            flow_masks_list.append(valid_mask_sequence[tidx].to(dtype=self.dtype))

            # B. Detection用: ラベルがあるインデックスを記録
            current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            if len(current_labels) > 0:
                # 全件フラット化した際のインデックス位置を特定
                offset = tidx * batch_size
                for v_idx in valid_batch_indices:
                    det_indices_in_flat.append(offset + v_idx)
                det_labels_flat.extend(current_labels)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)

        # --- 2. 共有FPNの実行 (全フレーム一括) ---
        all_backbone_feats = flow_selector.get_batched_backbone_features()
        all_fpn_feats = self.mdl.forward_fpn(all_backbone_feats)

        total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        log_dict = {}
        prefix = f'{mode_2_string[mode]}/'

        # --- 3. Flow Headの計算 (全サンプル) ---
        batched_flow_gt = torch.cat(flow_targets_list, dim=0)
        batched_flow_mask = torch.cat(flow_masks_list, dim=0)
        
        _, flow_losses = self.mdl.forward_flow_head(
            all_fpn_feats, flow_gt=batched_flow_gt, valid_mask=batched_flow_mask
        )
        
        if flow_losses and 'loss_flow' in flow_losses:
            l_flow = flow_losses['loss_flow']
            total_loss += self.flow_loss_weight * l_flow
            log_dict[f'{prefix}loss_flow'] = l_flow.detach()

        # --- 4. Detection Headの計算 (スライスして実行) ---
        if len(det_labels_flat) > 0:
            # FPN出力をラベルがある位置だけスライス (自動微分は維持されます)
            # YOLOX Headが期待する各スケールの特徴マップをスライス
            valid_fpn_feats = [f[det_indices_in_flat] for f in all_fpn_feats]
            
            batched_det_targets = ObjectLabels.get_labels_as_batched_tensor(det_labels_flat, format_='yolox')
            batched_det_targets = batched_det_targets.to(dtype=self.dtype)

            _, det_losses = self.mdl.forward_det_head(valid_fpn_feats, det_targets=batched_det_targets)
            
            if det_losses and 'loss' in det_losses:
                l_det = det_losses['loss']
                total_loss += self.det_loss_weight * l_det
                log_dict[f'{prefix}loss_det'] = l_det.detach()

        # --- 5. ログ記録と返却 ---
        log_dict[f'{prefix}loss_total'] = total_loss.detach()
        self.log_dict(log_dict, on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=True)

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
        batch_size = ev_tensor_sequence[0].shape[0]
        
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        
        # --- セレクターとインデックス管理の初期化 ---
        flow_selector = BackboneFeatureSelector() # Flow用 (全バッチ・評価対象全フレーム)
        all_batch_indices = torch.arange(batch_size, device=self.device)
        
        obj_labels_list = list()
        flow_gt_list = list()
        flow_mask_list = list()
        det_indices_in_flat = [] # FPN出力のどこがDetection対象か

        # --- 1. 時間方向ループ: Backbone特徴抽出 ---
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
                # A. Flow用に全バッチの特徴量とGTを保存
                flow_selector.add_backbone_features(backbone_features=backbone_features,
                                                    selected_indices=all_batch_indices)
                flow_gt_list.append(flow_tensor_sequence[tidx].to(dtype=self.dtype))
                flow_mask_list.append(valid_mask_sequence[tidx].to(dtype=self.dtype))

                # B. Detection用にラベルがある箇所のインデックスを記録
                current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                if len(current_labels) > 0:
                    # これまでに収集したFlowフレーム数に基づいてオフセットを計算
                    current_frame_offset = (len(flow_gt_list) - 1) * batch_size
                    for v_idx in valid_batch_indices:
                        det_indices_in_flat.append(current_frame_offset + v_idx)
                    obj_labels_list.extend(current_labels)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)

        # 予測対象が1つもない場合はスキップ
        if len(flow_gt_list) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}

        # --- 2. 共有FPNの実行 (全予測対象フレーム一括) ---
        all_backbone_feats = flow_selector.get_batched_backbone_features()
        all_fpn_feats = self.mdl.forward_fpn(all_backbone_feats)

        # --- 3. Flowタスクの評価 ---
        flow_preds, _ = self.mdl.forward_flow_head(all_fpn_feats)
        
        batched_flow_gt = torch.cat(flow_gt_list, dim=0).to(flow_preds.device)
        batched_flow_mask = torch.cat(flow_mask_list, dim=0).to(flow_preds.device)
        orig_h, orig_w = flow_gt_list[0].shape[-2:]
        flow_preds = flow_preds[..., :orig_h, :orig_w]

        flow_metrics = compute_flow_metrics(flow_preds, batched_flow_gt, batched_flow_mask)
        if flow_metrics:
            self.mode_2_flow_metrics_buffer[mode].append({k: v.detach().cpu() for k, v in flow_metrics.items()})

        # --- 4. Detectionタスクの評価 (スライスして実行) ---
        if len(obj_labels_list) > 0:
            # ラベルがある場所だけFPN出力をスライス
            valid_fpn_feats = [f[det_indices_in_flat] for f in all_fpn_feats]
            det_preds, _ = self.mdl.forward_det_head(valid_fpn_feats)
            
            pred_processed = postprocess(prediction=det_preds,
                                         num_classes=self.mdl_config.head.detection.num_classes,
                                         conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                         nms_thre=self.mdl_config.postprocess.nms_threshold)
            
            loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels_list, pred_processed)
            if self.mode_2_psee_evaluator[mode] is not None:
                self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
                self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)

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
            warn(f'psee_evaluator is None in {mode=}', UserWarning, stacklevel=2)
            return
        
        prefix = f'{mode_2_string[mode]}/'
        step = self.trainer.global_step
        log_dict = {}

        # --- Flowのバッファがあれば集計してlog_dictに追加 ---
        if self.mode_2_flow_metrics_buffer[mode]:
            for k in self.mode_2_flow_metrics_buffer[mode][0].keys():
                vals = [m[k] for m in self.mode_2_flow_metrics_buffer[mode]]
                log_dict[f'{prefix}{k}'] = torch.stack(vals).mean().to(self.device)
            self.mode_2_flow_metrics_buffer[mode].clear()

        # --- Detectionの評価 ---
        if psee_evaluator.has_data():
            metrics = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0], img_width=hw_tuple[1])
            if metrics:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)): value = torch.tensor(v)
                    elif isinstance(v, np.ndarray): value = torch.from_numpy(v)
                    else: value = v
                    log_dict[f'{prefix}{k}'] = value.to(self.device)
            psee_evaluator.reset_buffer()

        if log_dict:
            # 1. Lightning標準ログ
            self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            
            # 2. WandBへの直接ログ（APとFlowを同じStepで強制同期）
            if self.trainer.is_global_zero:
                add_hack = 2
                self.logger.log_metrics(metrics=log_dict, step=step + add_hack)

    def on_train_epoch_end(self) -> None:
        pass 

    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        if self.started_training and self.mode_2_psee_evaluator[mode] is not None:
            self.run_psee_evaluator(mode=mode)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        if self.mode_2_psee_evaluator[mode] is not None:
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