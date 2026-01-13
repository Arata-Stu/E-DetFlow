import os
import cv2
import torch
import numpy as np
import lightning as pl
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra

from data.utils.types import DatasetMode, DataType
from models.detection.yolox.utils import postprocess
from modules.utils.detection import RNNStates
from modules.utils.fetch import fetch_data_module, fetch_model_module
from utils.padding import InputPadderFromShape
from vis_utils import dataset2size, ev_repr_to_img, flow_to_image, draw_bboxes_with_id, dataset2labelmap

def visualize_combined(ev_tensors: torch.Tensor, 
                       pred_bboxes: torch.Tensor, 
                       flow_pred: torch.Tensor, 
                       flow_gt: torch.Tensor,
                       valid_mask: torch.Tensor,
                       dataset_name: str):
    """
    左：イベント+BBox, 中：予測フロー, 右：GTフロー を結合する
    """
    # 1. 左画面：イベント + BBox (Detection)
    ev_img = ev_repr_to_img(ev_tensors.detach().cpu().numpy())
    ev_img = cv2.cvtColor(ev_img, cv2.COLOR_RGB2BGR)
    if pred_bboxes is not None and pred_bboxes[0] is not None:
        # BBoxの描画 (vis_utilsに定義されていると仮定)
        ev_img = draw_bboxes_with_id(ev_img, pred_bboxes[0].detach().cpu().numpy(), dataset_name)

    # 2. 中央画面：予測フロー
    flow_pred_uv = flow_pred.detach().cpu().numpy().transpose(1, 2, 0)
    flow_pred_img = flow_to_image(flow_pred_uv, convert_to_bgr=True)
    
    # 3. 右画面：GTフロー + マスク適用
    flow_gt_uv = flow_gt.detach().cpu().numpy().transpose(1, 2, 0)
    flow_gt_img = flow_to_image(flow_gt_uv, convert_to_bgr=True)
    mask_np = valid_mask.detach().cpu().numpy().squeeze()
    flow_gt_img[mask_np == 0] = 0

    # 横に結合
    combined_img = np.hstack((ev_img, flow_pred_img, flow_gt_img))
    return combined_img

def create_combined_video(data: pl.LightningDataModule, 
                          model_module: pl.LightningModule, 
                          ckpt_path: str, 
                          output_path: str, 
                          fps: int, 
                          num_sequence: int):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_size = dataset2size[data.dataset_name] # (W, H)
    video_size = (orig_size[0] * 3, orig_size[1])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_size)

    # チェックポイントのロード
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_module.load_state_dict(ckpt['state_dict'])
    model_module.to(device).eval()
    
    # データローダーの準備 (Validationを使用)
    data.setup('validate')
    data_loader = data.val_dataloader()

    rnn_state = RNNStates()
    input_padder = InputPadderFromShape(model_module.in_res_hw)
    num_classes = len(set(dataset2labelmap[data.dataset_name].values()))
    sequence_count = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating Combined Video"):
            data_batch = batch["data"]
            ev_repr = data_batch[DataType.EV_REPR]
            flow_gt_seq = data_batch[DataType.FLOW]
            valid_mask_seq = data_batch[DataType.VALID]
            is_first_sample = data_batch[DataType.IS_FIRST_SAMPLE]

            rnn_state.reset(worker_id=0, indices_or_bool_tensor=is_first_sample)
            prev_states = rnn_state.get_states(worker_id=0)

            if is_first_sample.any():
                sequence_count += 1
                if sequence_count > num_sequence: break

            for tidx in range(len(ev_repr)):
                # 入力準備
                ev_tensors = ev_repr[tidx].to(device).to(torch.float32)
                ev_padded = input_padder.pad_tensor_ev_repr(ev_tensors)
                
                # --- モデル推論 (ModelModuleのロジックに従う) ---
                # 1. Backbone
                backbone_feats, states = model_module.mdl.forward_backbone(ev_padded, prev_states)
                prev_states = states
                # 2. FPN
                fpn_feats = model_module.mdl.forward_fpn(backbone_feats)
                # 3. Flow Head
                flow_preds, _ = model_module.mdl.forward_flow_head(fpn_feats)
                # 4. Detection Head
                det_preds, _ = model_module.mdl.forward_det_head(fpn_feats)

                # --- 後処理 ---
                # Flowのクロップ
                orig_h, orig_w = ev_tensors.shape[-2:]
                flow_pred_cropped = flow_preds[0, :, :orig_h, :orig_w]
                
                # DetectionのNMS
                pred_bboxes = postprocess(
                    prediction=det_preds,
                    num_classes=num_classes,
                    conf_thre=0.1, # 調整可能
                    nms_thre=0.45
                )

                # 可視化と書き込み
                img = visualize_combined(
                    ev_tensors[0],
                    pred_bboxes,
                    flow_pred_cropped,
                    flow_gt_seq[tidx][0].to(device),
                    valid_mask_seq[tidx][0].to(device),
                    data.dataset_name
                )
                video_writer.write(img)

    video_writer.release()
    print(f"Video saved to: {output_path}")

@hydra.main(config_path="../../config", config_name="visualize")
def main(cfg: DictConfig):
    data = fetch_data_module(cfg)
    model = fetch_model_module(cfg)
    create_combined_video(data, model, cfg.ckpt_path, cfg.output_path, cfg.fps, cfg.num_sequence)

if __name__ == '__main__':
    main()