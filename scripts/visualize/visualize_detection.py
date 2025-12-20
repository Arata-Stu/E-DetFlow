import os
import sys
sys.path.append('../../')

import cv2
import hydra
import lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from config.modifier import dynamically_modify_train_config
from data.genx_utils.labels import ObjectLabels
from data.utils.types import DatasetMode, DataType
from models.detection.yolox.utils import postprocess
from modules.utils.detection import RNNStates
from modules.utils.fetch import fetch_data_module, fetch_model_module
from utils.padding import InputPadderFromShape
from vis_utils import dataset2labelmap, dataset2size, draw_bboxes_with_id, ev_repr_to_img


def visualize_detection(video_writer: cv2.VideoWriter, 
                        ev_tensors: torch.Tensor, 
                        labels_yolox: torch.Tensor, 
                        pred_processed: torch.Tensor, 
                        dataset_name: str):

    img = ev_repr_to_img(ev_tensors.squeeze().cpu().numpy())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Ground Truth の描画
    if labels_yolox is not None and labels_yolox[0] is not None:
        labels_gt = labels_yolox.cpu().numpy()[0]
        img = draw_bboxes_with_id(img, labels_gt, dataset_name)

    # 予測結果の描画
    if pred_processed is not None and pred_processed[0] is not None:
        pred_res = pred_processed[0].detach().cpu().numpy()
        img = draw_bboxes_with_id(img, pred_res, dataset_name)

    video_writer.write(img)


def create_video_detection(data: pl.LightningDataModule, 
                           model: pl.LightningModule, 
                           ckpt_path: str, 
                           show_gt: bool, 
                           show_pred: bool, 
                           output_path: str, 
                           fps: int, 
                           num_sequence: int, 
                           dataset_mode: DatasetMode):  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_size = dataset2size[data.dataset_name]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, data_size)

    # データセットのセットアップ
    if dataset_mode == "train":
        data.setup('fit')
        data_loader = data.train_dataloader()
    elif dataset_mode == "val":
        data.setup('validate')
        data_loader = data.val_dataloader()
    elif dataset_mode == "test":
        data.setup('test')
        data_loader = data.test_dataloader()
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

    label_map = dataset2labelmap[data.dataset_name]
    num_classes = len(set(label_map.values()))

    # モデルの準備
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
    
    model.to(device)
    model.eval()

    rnn_state = RNNStates()
    input_padder = InputPadderFromShape(model.in_res_hw)
    sequence_count = 0

    # 推論ループ
    with torch.no_grad():  # メモリ節約と高速化のために必須
        for batch in tqdm(data_loader, desc="Processing Video"):
            data_batch = batch["data"]

            ev_repr = data_batch[DataType.EV_REPR]
            labels = data_batch[DataType.OBJLABELS_SEQ]
            is_first_sample = data_batch[DataType.IS_FIRST_SAMPLE]

            # 新しいシーケンスの開始時にRNN状態をリセット
            rnn_state.reset(worker_id=0, indices_or_bool_tensor=is_first_sample)
            prev_states = rnn_state.get_states(worker_id=0)

            if is_first_sample.any():
                sequence_count += 1
                if sequence_count > num_sequence:
                    break

            sequence_len = len(ev_repr)
            for tidx in range(sequence_len):
                ev_tensors = ev_repr[tidx].to(torch.float32).to(device)

                # Ground Truthの取得
                labels_yolox = None
                if show_gt:
                    current_labels, _ = labels[tidx].get_valid_labels_and_batch_indices()
                    if len(current_labels) > 0:
                        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(
                            obj_label_list=current_labels, format_='yolox')

                # モデル推論
                pred_processed = None
                if show_pred:
                    ev_tensors_padded = input_padder.pad_tensor_ev_repr(ev_tensors)
                    # RVT等のモデル前方計算
                    predictions, _, states = model.forward(
                        event_tensor=ev_tensors_padded, 
                        previous_states=prev_states
                    )
                    prev_states = states
                    rnn_state.save_states_and_detach(worker_id=0, states=prev_states)
                    
                    # 後処理（NMS等）
                    pred_processed = postprocess(
                        prediction=predictions, 
                        num_classes=num_classes, 
                        conf_thre=0.1, 
                        nms_thre=0.45
                    )

                # 可視化実行
                visualize_detection(video_writer, ev_tensors, labels_yolox, pred_processed, data.dataset_name)

    video_writer.release()
    print(f"\nSuccessfully saved video to: {output_path}")


@hydra.main(config_path="../../config", config_name="visualize", version_base="1.2")
def main(cfg: DictConfig):
    # 設定の解決
    dynamically_modify_train_config(cfg)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    # 出力先ディレクトリの作成
    dir_name = os.path.dirname(cfg.output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # モジュールの取得
    data = fetch_data_module(config=cfg)
    model = fetch_model_module(config=cfg)
        
    # 動画生成実行
    create_video_detection(
        data=data, 
        model=model, 
        ckpt_path=cfg.ckpt_path, 
        show_gt=cfg.gt, 
        show_pred=cfg.pred, 
        output_path=cfg.output_path, 
        fps=cfg.fps, 
        num_sequence=cfg.num_sequence, 
        dataset_mode=cfg.dataset_mode
    )


if __name__ == '__main__':
    main()