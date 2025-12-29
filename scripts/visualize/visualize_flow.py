import os
import sys
sys.path.append('../../')

import cv2
import hydra
import lightning as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from config.modifier import dynamically_modify_train_config
from data.utils.types import DatasetMode, DataType
from modules.utils.detection import RNNStates  
from modules.utils.fetch import fetch_data_module, fetch_model_module
from utils.padding import InputPadderFromShape
from vis_utils import dataset2size, ev_repr_to_img, flow_to_image


def visualize_flow(ev_tensors: torch.Tensor, 
                   flow_pred: torch.Tensor, 
                   flow_gt: torch.Tensor,
                   valid_mask: torch.Tensor):
    """
    イベント、予測、GTを横に結合。
    valid_maskの次元が (1, H, W) の場合を考慮して squeeze 处理を行う。
    """
    # 1. イベント画像の作成 (H, W, 3)
    ev_img = ev_repr_to_img(ev_tensors.detach().cpu().numpy())
    ev_img = cv2.cvtColor(ev_img, cv2.COLOR_RGB2BGR)

    # valid_mask が (1, H, W) の場合、(H, W) に変換
    mask_np = valid_mask.detach().cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze(0) 
    mask_np = mask_np.astype(np.uint8)

    # 2. 予測フローのカラー化
    flow_pred_uv = flow_pred.detach().cpu().numpy().transpose(1, 2, 0)
    flow_pred_img = flow_to_image(flow_pred_uv, convert_to_bgr=True)
    
    flow_pred_img[mask_np == 0] = 0

    # 3. GTフローのカラー化
    flow_gt_uv = flow_gt.detach().cpu().numpy().transpose(1, 2, 0)
    flow_gt_img = flow_to_image(flow_gt_uv, convert_to_bgr=True)
    
    flow_gt_img[mask_np == 0] = 0

    # 横に結合
    combined_img = np.hstack((ev_img, flow_pred_img, flow_gt_img))
    return combined_img


def create_video_flow(data: pl.LightningDataModule, 
                      model: pl.LightningModule, 
                      ckpt_path: str, 
                      output_path: str, 
                      fps: int, 
                      num_sequence: int, 
                      dataset_mode: DatasetMode):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3枚並べるため、幅を3倍にする
    orig_size = dataset2size[data.dataset_name] # (W, H)
    video_size = (orig_size[0] * 3, orig_size[1])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_size)

    # データセットの準備
    if dataset_mode == "test":
        data.setup('test')
        data_loader = data.test_dataloader()
    else:
        data.setup('validate')
        data_loader = data.val_dataloader()

    # モデルのロードと評価モード設定
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()

    rnn_state = RNNStates()
    input_padder = InputPadderFromShape(model.in_res_hw)
    sequence_count = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing Flow Video"):
            data_batch = batch["data"]
            ev_repr = data_batch[DataType.EV_REPR]
            flow_gt_seq = data_batch[DataType.FLOW]
            valid_mask_seq = data_batch[DataType.VALID] 
            is_first_sample = data_batch[DataType.IS_FIRST_SAMPLE]

            # RNN状態のリセット
            rnn_state.reset(worker_id=0, indices_or_bool_tensor=is_first_sample)
            prev_states = rnn_state.get_states(worker_id=0)

            if is_first_sample.any():
                sequence_count += 1
                if sequence_count > num_sequence:
                    break

            sequence_len = len(ev_repr)
            for tidx in range(sequence_len):
                # 入力準備
                ev_tensors = ev_repr[tidx].to(torch.float32).to(device)
                ev_tensors_padded = input_padder.pad_tensor_ev_repr(ev_tensors)
                
                flow_pred, _, states = model.forward(ev_tensors_padded, prev_states)
                prev_states = states
                
                # パディング解除のためのサイズ取得
                orig_h, orig_w = ev_tensors.shape[-2:]
                
                # クロップ処理
                flow_pred_cropped = flow_pred[0, :, :orig_h, :orig_w]
                flow_gt_cropped = flow_gt_seq[tidx][0].to(device)
                valid_mask_cropped = valid_mask_seq[tidx][0].to(device)

                if tidx == 0:
                    mask_max = valid_mask_cropped.max().item()
                    mask_min = valid_mask_cropped.min().item()
                    mask_mean = valid_mask_cropped.float().mean().item()
                    print(f"DEBUG: Mask stats -> Max: {mask_max}, Min: {mask_min}, Mean: {mask_mean}")
# ----------------

                img = visualize_flow(
                    ev_tensors[0], 
                    flow_pred_cropped, 
                    flow_gt_cropped, 
                    valid_mask_cropped
                )
                video_writer.write(img)
                
    video_writer.release()
    print(f"\nVideo successfully saved to: {output_path}")

@hydra.main(config_path="../../config", config_name="visualize", version_base="1.2")
def main(cfg: DictConfig):
    dynamically_modify_train_config(cfg)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    # 出力先ディレクトリの準備
    dir_name = os.path.dirname(cfg.output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # DataModule と Model の取得
    data = fetch_data_module(config=cfg)
    model = fetch_model_module(config=cfg)
    
    create_video_flow(
        data=data, 
        model=model, 
        ckpt_path=cfg.ckpt_path, 
        output_path=cfg.output_path, 
        fps=cfg.fps, 
        num_sequence=cfg.num_sequence, 
        dataset_mode=cfg.dataset_mode
    )


if __name__ == '__main__':
    main()