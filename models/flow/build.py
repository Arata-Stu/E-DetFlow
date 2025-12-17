from omegaconf import OmegaConf, DictConfig

from .optical_flow_head import FlowHead

def build_flow_head(flow_head_cfg: DictConfig, in_channels: int):
    flow_head_cfg_dict = OmegaConf.to_container(flow_head_cfg, resolve=True, throw_on_missing=True)
    flow_head_cfg_dict.pop('name')
    flow_head_cfg_dict.update({"in_channels": in_channels})
    return FlowHead(**flow_head_cfg_dict)

