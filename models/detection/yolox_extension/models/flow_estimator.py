from typing import Dict, Optional, Tuple, Union

import torch as th
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ...recurrent_backbone import build_recurrent_backbone
from .build import build_yolox_fpn, build_flow_head
from utils.timers import TimerDummy as CudaTimer

from data.utils.types import BackboneFeatures, LstmStates


class FlowEstimator(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        self.backbone = build_recurrent_backbone(backbone_cfg)

        in_channels = self.backbone.get_stage_dims(fpn_cfg.in_stages)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)

        self.flow_head = build_flow_head(head_cfg.flow, in_channels=in_channels)

    def forward_backbone(self,
                         x: th.Tensor,
                         previous_states: Optional[LstmStates] = None,
                         token_mask: Optional[th.Tensor] = None) -> \
            Tuple[BackboneFeatures, LstmStates]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features, states = self.backbone(x, previous_states, token_mask)
        return backbone_features, states

    def forward_flow(self,
                     backbone_features: BackboneFeatures,
                     flow_gt: Optional[th.Tensor] = None,
                     valid_mask: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)
        
        if self.training:
            assert flow_gt is not None
            with CudaTimer(device=device, timer_name="HEAD + Loss"):
                outputs, losses = self.flow_head(fpn_features, flow_gt=flow_gt, valid_mask=valid_mask)
            return outputs, losses
        
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs, losses = self.flow_head(fpn_features)
        assert losses is None
        return outputs, losses

    def forward(self,
                x: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_flow: bool = True,
                flow_gt: Optional[th.Tensor] = None,
                valid_mask: Optional[th.Tensor] = None) -> \
            Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
        
        backbone_features, states = self.forward_backbone(x, previous_states)
        outputs, losses = None, None
        
        if not retrieve_flow:
            assert flow_gt is None
            return outputs, losses, states
            
        outputs, losses = self.forward_flow(
            backbone_features=backbone_features, 
            flow_gt=flow_gt, 
            valid_mask=valid_mask
        )
        return outputs, losses, states