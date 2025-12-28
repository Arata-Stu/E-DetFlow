from omegaconf import DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint

def get_ckpt_callback(config: DictConfig) -> ModelCheckpoint:
    train_task = config.train_task  # 'detection' or 'optical_flow'
    model_name = config.model.name

    prefix = 'val'
    
    # タスクに応じてメトリクスとモードを切り替え
    if train_task == 'detection':
        metric = 'AP'
        mode = 'max'  # 精度は高い方が良い
    elif train_task == 'optical_flow':
        metric = 'EPE'
        mode = 'min'  # エラー（EPE）は低い方が良い
    else:
        # 他のタスク（例：'multitask'）がある場合はここに追加
        raise NotImplementedError(f"Task {train_task} is not supported.")

    ckpt_callback_monitor = prefix + '/' + metric
    filename_monitor_str = prefix + '_' + metric

    ckpt_filename = 'epoch_{epoch:03d}-step_{step}-' + filename_monitor_str + '_{' + ckpt_callback_monitor + ':.2f}'
    
    cktp_callback = ModelCheckpoint(
        monitor=ckpt_callback_monitor,
        filename=ckpt_filename,
        auto_insert_metric_name=False, 
        save_top_k=1,
        mode=mode,
        every_n_epochs=config.logging.ckpt_every_n_epochs,
        save_last=True,
        verbose=True)
    
    cktp_callback.CHECKPOINT_NAME_LAST = 'last_epoch_{epoch:03d}-step_{step}'
    return cktp_callback