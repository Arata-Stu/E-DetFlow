from typing import Dict, Optional, Tuple, Union

import torch as th
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ...recurrent_backbone import build_recurrent_backbone
from .build import build_yolox_fpn, build_yolox_head, build_flow_head 
from utils.timers import TimerDummy as CudaTimer
from data.utils.types import BackboneFeatures, LstmStates


class YoloXFlowDetector(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        heads_cfg = model_cfg.head

        # --- 1. 共有バックボーン ---
        self.backbone = build_recurrent_backbone(backbone_cfg)

        # --- 2. 検出ブランチ ---
        det_in_channels = self.backbone.get_stage_dims(fpn_cfg.in_stages)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=det_in_channels)
        strides = self.backbone.get_strides(fpn_cfg.in_stages)
        self.yolox_head = build_yolox_head(heads_cfg.detection, in_channels=det_in_channels, strides=strides) 
        self.flow_head = build_flow_head(heads_cfg.flow, in_channels=det_in_channels)

    def forward_backbone(self,
                         x: th.Tensor,
                         previous_states: Optional[LstmStates] = None,
                         token_mask: Optional[th.Tensor] = None) -> \
            Tuple[BackboneFeatures, LstmStates]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features, states = self.backbone(x, previous_states, token_mask)
        return backbone_features, states

    def forward_detect(self,
                       backbone_features: BackboneFeatures,
                       targets: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="FPN"):
            det_features = {k: v for k, v in backbone_features.items() if k in self.fpn.in_features}
            fpn_features = self.fpn(det_features)
        if self.training:
            assert targets is not None
            with CudaTimer(device=device, timer_name="HEAD + Loss"):
                outputs, losses = self.yolox_head(fpn_features, targets)
            return outputs, losses
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs, losses = self.yolox_head(fpn_features)
        assert losses is None
        return outputs, losses

    def forward_flow(self,
                     backbone_features: BackboneFeatures,
                     targets: Optional[th.Tensor] = None) -> Tuple[th.Tensor, Optional[Dict[str, th.Tensor]]]:
        """ステージ3の特徴量からフローを予測し、学習時には損失も返す"""
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="Flow Head"):
            flow_input = backbone_features[3]
            flow_output, flow_losses = self.flow_head(flow_input, targets=targets)
        return flow_output, flow_losses

    def forward(self,
                x: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                retrieve_flow: bool = True, 
                targets: Optional[th.Tensor] = None,
                flow_targets: Optional[th.Tensor] = None) -> Dict[str, Union[th.Tensor, Dict, LstmStates, None]]:
        
        backbone_features, states = self.forward_backbone(x, previous_states)
        
        det_outputs, det_losses = None, None
        flow_outputs, flow_losses = None, None

        if retrieve_detections:
            det_outputs, det_losses = self.forward_detect(backbone_features=backbone_features, targets=targets)

        if retrieve_flow:
            flow_outputs, flow_losses = self.forward_flow(backbone_features=backbone_features, targets=flow_targets)

        total_losses = {}
        if det_losses:
            total_losses.update(det_losses)
        if flow_losses:
            total_losses.update(flow_losses)
            
        return {
            "detections": det_outputs,
            "flow": flow_outputs,
            "losses": total_losses if total_losses else None,
            "states": states,
        }