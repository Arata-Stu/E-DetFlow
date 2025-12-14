import lightning as pl
from omegaconf import DictConfig
from modules.data.data_module import DataModule 
from modules.model.rnn_module import  ModelModule as rnn_det_module

def fetch_model_module(config: DictConfig) -> pl.LightningModule:
    model_str = config.model.name
    if model_str in {'RVT', 'YOLOX_LSTM'}:
        return rnn_det_module(config)
    raise NotImplementedError

def fetch_data_module(config: DictConfig) -> pl.LightningDataModule:
    batch_size_train = config.batch_size.train
    batch_size_eval = config.batch_size.eval
    num_workers_generic = config.hardware.get('num_workers', None)
    num_workers_train = config.hardware.num_workers.get('train', num_workers_generic)
    num_workers_eval = config.hardware.num_workers.get('eval', num_workers_generic)
    dataset_str = config.dataset.name
    if dataset_str in {'gen1', 'gen4', 'VGA', 'SEVD'}:
        return DataModule(config.dataset,
                                num_workers_train=num_workers_train,
                                num_workers_eval=num_workers_eval,
                                batch_size_train=batch_size_train,
                                batch_size_eval=batch_size_eval)
    raise NotImplementedError
