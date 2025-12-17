import torch

def compute_flow_metrics(flow_pred: torch.Tensor, flow_gt: torch.Tensor, valid: torch.Tensor, max_flow: float = 1000.0):
    """
    EPE, AE, 1PE, 3PE を計算して返します。
    
    Args:
        flow_pred: (B, 2, H, W)
        flow_gt:   (B, 2, H, W)
        valid:     (B, 2, H, W) or (B, 1, H, W)
        max_flow:  これ以上の大きさのGTは無視する
        
    Returns:
        metrics: Dict (Tensor scalars)
            - EPE: Endpoint Error (Average)
            - AE:  Angular Error (Average degrees)
            - 1PE: 1-Pixel Error Rate (%) -> EPE > 1px の割合
            - 3PE: 3-Pixel Error Rate (%) -> EPE > 3px の割合
    """
    
    # --- 1. Mask Handling ---
    if valid.shape[1] >= 2:
        valid_mask = valid[:, 1] 
    else:
        valid_mask = valid.squeeze(1)

    # --- 2. Filtering ---
    # GTの大きさが異常に大きいものや、マスクが無効なものを除外
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid_mask = (valid_mask >= 0.5) & (mag < max_flow)

    # 有効なピクセルがない場合のガード
    if valid_mask.sum() == 0:
        return {
            'EPE': torch.tensor(0.0, device=flow_pred.device),
            'AE':  torch.tensor(0.0, device=flow_pred.device),
            '1PE': torch.tensor(0.0, device=flow_pred.device),
            '3PE': torch.tensor(0.0, device=flow_pred.device)
        }

    # マスクを使って有効画素のみ抽出 (Flatten)
    # (B, 2, H, W) -> (N, 2) where N is number of valid pixels
    pred_flat = flow_pred.permute(0, 2, 3, 1)[valid_mask]
    gt_flat = flow_gt.permute(0, 2, 3, 1)[valid_mask]

    # --- 3. EPE (Endpoint Error) ---
    # ユークリッド距離: ||u_pred - u_gt||
    diff = pred_flat - gt_flat
    epe_all = torch.norm(diff, p=2, dim=1) # (N,)
    mean_epe = epe_all.mean()

    # --- 4. AE (Angular Error) ---
    # フローを時間軸1の3Dベクトル (u, v, 1) とみなして角度を計算
    # cos(theta) = (v1 . v2) / (|v1| * |v2|)
    # dot product: u1*u2 + v1*v2 + 1*1
    dot_product = torch.sum(pred_flat * gt_flat, dim=1) + 1.0

    # norms: sqrt(u^2 + v^2 + 1)
    norm_pred = torch.sqrt(torch.sum(pred_flat**2, dim=1) + 1.0)
    norm_gt   = torch.sqrt(torch.sum(gt_flat**2, dim=1) + 1.0)

    # cosの値が数値誤差で1.0を超えないようにclampする
    cos_val = torch.clamp(dot_product / (norm_pred * norm_gt + 1e-8), -1.0, 1.0)
    ae_all = torch.acos(cos_val) # ラジアン

    mean_ae = ae_all.mean() * (180.0 / 3.14159265) # 度数法に変換 (degrees)

    # --- 5. 1PE / 3PE (N-Pixel Error Rate) ---
    # Error Rateなので「閾値を超えた(間違った)ピクセルの割合」と定義します
    # Accuracy(正解率)にしたい場合は不等号を逆にしてください
    one_pe = (epe_all > 1.0).float().mean() * 100.0
    three_pe = (epe_all > 3.0).float().mean() * 100.0

    metrics = {
        'EPE': mean_epe,
        'AE':  mean_ae,
        '1PE': one_pe,
        '3PE': three_pe
    }
    
    return metrics