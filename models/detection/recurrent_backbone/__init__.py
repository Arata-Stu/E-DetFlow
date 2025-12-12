from omegaconf import DictConfig

from .maxvit_rnn import RNNDetector as MaxViTRNNDetector
from .darknet_rnn import CSPDarknetLSTM as CNNDetector


def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'MaxViTRNN':
        return MaxViTRNNDetector(backbone_cfg)
    elif name == 'CSPDarknetLSTM':
        return CNNDetector(backbone_cfg)
    else:
        raise NotImplementedError
