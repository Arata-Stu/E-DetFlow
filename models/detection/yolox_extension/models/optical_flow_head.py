import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List

from ...yolox.models.network_blocks import BaseConv 

def sequence_loss(flow_preds: List[torch.Tensor], flow_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    予測されたフローのリスト（マルチスケール）と正解フローから損失を計算する。
    RAFTで使われている加重L1 + Charbonnier損失を参考に実装。
    """
    if flow_gt is None or flow_gt.shape[0] == 0:
        return {"loss_flow": torch.tensor(0.0, device=flow_preds[0].device)}

    n_predictions = len(flow_preds)
    total_loss = 0.0
    epsilon = 1e-8
    loss_weights = [0.8**(n_predictions - 1 - i) for i in range(n_predictions)]

    for i, pred_flow in enumerate(flow_preds):
        b, _, h, w = pred_flow.shape
        gt_downsampled = nn.functional.interpolate(flow_gt, size=(h, w), mode='bilinear', align_corners=False)
        
        flow_diff = pred_flow - gt_downsampled
        
        loss = torch.sqrt(flow_diff.pow(2).sum(dim=1) + epsilon).mean()
        
        total_loss += loss_weights[i] * loss

    return {"loss_flow": total_loss}


class FlowHead(nn.Module):
    """
    バックボーンの単一ステージ特徴マップからオプティカルフローを予測するデコーダヘッド。
    連続したアップサンプリングにより、特徴マップを元の解像度に戻し、フローを推定する。
    """
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

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        
        features_4x = self.upsample1(x)
        features_2x = self.upsample2(features_4x)
        features_1x = self.upsample3(features_2x)

        flow_4x = self.flow_pred_4x(features_4x)
        flow_2x = self.flow_pred_2x(features_2x)
        flow_1x = self.flow_pred_1x(features_1x)

        losses = None
        if self.training:
            assert targets is not None, "Targets must be provided during training."
            all_flow_preds = [flow_1x, flow_2x, flow_4x]
            losses = sequence_loss(all_flow_preds, targets)
        
        return flow_1x, losses