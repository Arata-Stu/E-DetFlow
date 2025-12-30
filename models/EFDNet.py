from typing import Dict, Optional, Tuple, Union

import torch as th
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from .detection.recurrent_backbone import build_recurrent_backbone
from .detection.yolox_extension.models.build import build_yolox_fpn, build_yolox_head 
from .flow.build import build_flow_head
from utils.timers import TimerDummy as CudaTimer

from data.utils.types import BackboneFeatures, LstmStates


class EFDNet(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        self.backbone = build_recurrent_backbone(backbone_cfg)
        in_channels = self.backbone.get_stage_dims(fpn_cfg.in_stages)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)

        strides = self.backbone.get_strides(fpn_cfg.in_stages)
        self.flow_head = build_flow_head(head_cfg.flow, in_channels=in_channels)
        self.det_head = build_yolox_head(head_cfg.detection, in_channels=in_channels, strides=strides)

    def forward_backbone(self,
                         x: th.Tensor,
                         previous_states: Optional[LstmStates] = None,
                         token_mask: Optional[th.Tensor] = None) -> \
            Tuple[BackboneFeatures, LstmStates]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features, states = self.backbone(x, previous_states, token_mask)
        return backbone_features, states

    def forward_heads(self,
                      backbone_features: BackboneFeatures,
                      flow_gt: Optional[th.Tensor] = None,
                      valid_mask: Optional[th.Tensor] = None,
                      det_targets: Optional[th.Tensor] = None) -> \
            Tuple[Dict[str, th.Tensor], Union[Dict[str, th.Tensor], None]]:
        
        device = next(iter(backbone_features.values())).device
        
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)

        outputs = {}
        losses = {} if self.training else None

        # --- Flow Branch ---
        with CudaTimer(device=device, timer_name="Head: Flow"):
            if self.training:
                flow_out, flow_loss_dict = self.flow_head(
                    fpn_features, flow_gt=flow_gt, valid_mask=valid_mask
                )
                if flow_loss_dict is not None:
                    losses.update(flow_loss_dict)
            else:
                flow_out, _ = self.flow_head(fpn_features)
            
            outputs['flow'] = flow_out

        # --- Detection Branch ---
        with CudaTimer(device=device, timer_name="Head: Detection"):
            if self.training:
                det_out, det_loss_dict = self.det_head(
                    fpn_features, det_targets
                )
                if det_loss_dict is not None:
                    losses.update(det_loss_dict)
            else:
                det_out, _ = self.det_head(fpn_features)
            
            outputs['detection'] = det_out

        return outputs, losses

    def forward(self,
                x: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                flow_gt: Optional[th.Tensor] = None,
                valid_mask: Optional[th.Tensor] = None,
                det_targets: Optional[th.Tensor] = None) -> \
            Tuple[Union[Dict[str, th.Tensor], None], Union[Dict[str, th.Tensor], None], LstmStates]:
        
        backbone_features, states = self.forward_backbone(x, previous_states)
        
        outputs, losses = None, None
        
        if not retrieve_detections:
            if self.training:
                assert flow_gt is None and det_targets is None
            return outputs, losses, states

        outputs, losses = self.forward_heads(
            backbone_features=backbone_features,
            flow_gt=flow_gt,
            valid_mask=valid_mask,
            det_targets=det_targets
        )

        return outputs, losses, states