import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from ...yolox.models.network_blocks import BaseConv 

def sequence_loss(
    flow_preds: List[torch.Tensor], 
    flow_gt: torch.Tensor, 
    valid_mask: Optional[torch.Tensor] = None, 
    gamma: float = 0.8
) -> Dict[str, torch.Tensor]:
    if flow_gt is None or flow_gt.shape[0] == 0:
        return {"loss_flow": torch.tensor(0.0, device=flow_preds[0].device)}

    n_predictions = len(flow_preds)
    total_loss = 0.0
    epsilon = 1e-8
    loss_weights = [gamma**(n_predictions - 1 - i) for i in range(n_predictions)]

    if valid_mask is None:
        valid_mask = torch.ones_like(flow_gt[:, 0:1, ...])

    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)

    for i, pred_flow in enumerate(flow_preds):
        b, _, h, w = pred_flow.shape
        
        gt_downsampled = F.interpolate(flow_gt, size=(h, w), mode='bilinear', align_corners=False)
        mask_downsampled = F.interpolate(valid_mask.float(), size=(h, w), mode='bilinear', align_corners=False)
        
        flow_diff = pred_flow - gt_downsampled
        loss_map = torch.sqrt(flow_diff.pow(2).sum(dim=1, keepdim=True) + epsilon)

        weighted_loss = loss_map * mask_downsampled
        num_valid_pixels = mask_downsampled.sum() + epsilon
        loss = weighted_loss.sum() / num_valid_pixels
        
        total_loss += loss_weights[i] * loss

    return {"loss_flow": total_loss}


class FlowHead(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int, int, int], 
        act: str = "silu",
    ):
        super().__init__()
        
        stage3_channels = in_channels[0]
        
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            BaseConv(in_channels=stage3_channels, out_channels=stage3_channels // 2, ksize=3, stride=1, act=act)
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            BaseConv(in_channels=stage3_channels // 2, out_channels=stage3_channels // 4, ksize=3, stride=1, act=act)
        )
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            BaseConv(in_channels=stage3_channels // 4, out_channels=stage3_channels // 8, ksize=3, stride=1, act=act)
        )
        
        self.flow_pred_4x = nn.Conv2d(stage3_channels // 2, 2, 3, 1, 1)
        self.flow_pred_2x = nn.Conv2d(stage3_channels // 4, 2, 3, 1, 1)
        self.flow_pred_1x = nn.Conv2d(stage3_channels // 8, 2, 3, 1, 1)

    def forward(
        self, 
        x: torch.Tensor, 
        flow_gt: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        
        features_4x = self.upsample1(x)
        features_2x = self.upsample2(features_4x)
        features_1x = self.upsample3(features_2x)

        flow_4x = self.flow_pred_4x(features_4x)
        flow_2x = self.flow_pred_2x(features_2x)
        flow_1x = self.flow_pred_1x(features_1x)

        losses = None
        if self.training:
            assert flow_gt is not None
            
            all_flow_preds = [flow_4x, flow_2x, flow_1x]
            
            losses = sequence_loss(
                flow_preds=all_flow_preds, 
                flow_gt=flow_gt, 
                valid_mask=valid_mask
            )
        
        return flow_1x, losses