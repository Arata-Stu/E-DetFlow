from typing import Dict, Optional, Tuple, Union, Any

import torch as th
from omegaconf import DictConfig

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

    def forward_fpn(self, backbone_features: BackboneFeatures) -> Any:
        """FPNの計算を独立させたメソッド"""
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="FPN"):
            return self.fpn(backbone_features)

    def forward_flow_head(self, 
                          fpn_features: Any, 
                          flow_gt: Optional[th.Tensor] = None, 
                          valid_mask: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Optional[Dict[str, th.Tensor]]]:
        """Flowヘッドのみを計算するメソッド"""
        device = fpn_features[0].device if isinstance(fpn_features, (list, tuple)) else next(iter(fpn_features.values())).device
        with CudaTimer(device=device, timer_name="Head: Flow"):
            if self.training:
                return self.flow_head(fpn_features[0], flow_gt=flow_gt, valid_mask=valid_mask)
            else:
                return self.flow_head(fpn_features[0])

    def forward_det_head(self, 
                         fpn_features: Any, 
                         det_targets: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Optional[Dict[str, th.Tensor]]]:
        """Detectionヘッドのみを計算するメソッド"""
        device = fpn_features[0].device if isinstance(fpn_features, (list, tuple)) else next(iter(fpn_features.values())).device
        with CudaTimer(device=device, timer_name="Head: Detection"):
            if self.training:
                return self.det_head(fpn_features, det_targets)
            else:
                return self.det_head(fpn_features)

    def forward_heads(self,
                      backbone_features: BackboneFeatures,
                      flow_gt: Optional[th.Tensor] = None,
                      valid_mask: Optional[th.Tensor] = None,
                      det_targets: Optional[th.Tensor] = None) -> \
            Tuple[Dict[str, th.Tensor], Union[Dict[str, th.Tensor], None]]:
        """既存の統合メソッド（後方互換性のため維持）"""
        
        fpn_features = self.forward_fpn(backbone_features)

        outputs = {}
        losses = {} if self.training else None

        # Flow実行
        flow_out, flow_loss_dict = self.forward_flow_head(fpn_features, flow_gt, valid_mask)
        outputs['flow'] = flow_out
        if self.training and flow_loss_dict:
            losses.update(flow_loss_dict)

        # Detection実行
        det_out, det_loss_dict = self.forward_det_head(fpn_features, det_targets)
        outputs['detection'] = det_out
        if self.training and det_loss_dict:
            losses.update(det_loss_dict)

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
        
        if not retrieve_detections:
            return None, None, states

        outputs, losses = self.forward_heads(
            backbone_features=backbone_features,
            flow_gt=flow_gt,
            valid_mask=valid_mask,
            det_targets=det_targets
        )

        return outputs, losses, states