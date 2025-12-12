#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

# --- PyTorch 2.0 compile ---
try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ..yolox.models.network_blocks import BaseConv, CSPLayer, DWConv, Focus, SPPBottleneck
from ...layers.rnn import DWSConvLSTM2d  
from .base import BaseDetector 


class CSPDarknetLSTMStage(nn.Module):
    """
    CSPDarknet-LSTMの1ステージを構成するモジュール。
    CNNブロック + LSTM のペアをカプセル化する。
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_blocks: int,
        is_spp_stage: bool,
        depthwise: bool,
        act: str,
        lstm_cfg: DictConfig, # DictConfigとして型付け
        enable_masking: bool,
    ):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        
        # --- CNN Block ---
        if is_spp_stage:
            self.cnn_block = nn.Sequential(
                Conv(dim_in, dim_out, 3, 2, act=act),
                SPPBottleneck(dim_out, dim_out, activation=act),
                CSPLayer(dim_out, dim_out, n=n_blocks, shortcut=False, depthwise=depthwise, act=act),
            )
        else:
            self.cnn_block = nn.Sequential(
                Conv(dim_in, dim_out, 3, 2, act=act),
                CSPLayer(dim_out, dim_out, n=n_blocks, depthwise=depthwise, act=act),
            )
            
        # --- LSTM Block ---
        self.lstm = DWSConvLSTM2d(
            dim=dim_out,
            dws_conv=lstm_cfg.dws_conv,
            dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
            dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
            cell_update_dropout=lstm_cfg.get('drop_cell_update', 0)
        )
        
        # --- Mask Token ---
        self.mask_token = nn.Parameter(torch.zeros(1, dim_out, 1, 1), requires_grad=True) if enable_masking else None
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]], token_mask: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.cnn_block(x)
        if token_mask is not None:
            assert self.mask_token is not None, "This stage does not have a mask token."
            x = x.masked_fill(token_mask, self.mask_token)
        h_c_tuple = self.lstm(x, prev_state)
        x = h_c_tuple[0]
        return x, h_c_tuple


class CSPDarknetLSTM(BaseDetector): # BaseDetectorを継承
    """
    複数の`CSPDarknetLSTMStage`を束ね、全体のバックボーンを構築する。
    `mdl_config`から設定を読み込む。
    """
    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        # ====== Config ======
        in_channels = mdl_config.input_channels
        depth = mdl_config.depth
        width = mdl_config.width
        depthwise = mdl_config.depthwise
        act = mdl_config.act
        self.out_features = tuple(mdl_config.out_features)
        lstm_cfg = mdl_config.lstm
        enable_masking = mdl_config.enable_masking
        assert self.out_features, "Please provide output features in config."
        
        # ====== Compile if requested ======
        compile_cfg = mdl_config.get('compile', None)
        if compile_cfg is not None:
            compile_mdl = compile_cfg.enable
            if compile_mdl and th_compile is not None:
                compile_args = OmegaConf.to_container(compile_cfg.args, resolve=True, throw_on_missing=True)
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print('Could not compile backbone because torch.compile is not available')
        
        # ====== Network Architecture ======
        base_channels = int(width * 64)
        base_depth = max(round(depth * 3), 1)
        
        self.strides: Dict[int, int] = {}
        self.stage_dims: Dict[int, int] = {}
        
        # --- Stage 1 (Stem) ---
        self.stem = Focus(in_channels, base_channels, ksize=3, act=act)
        self.lstm_stem = DWSConvLSTM2d(base_channels)
        self.strides[1] = 2
        self.stage_dims[1] = base_channels
        
        # --- Stages 2 to 5 ---
        self.stages = nn.ModuleList()
        stage_configs = [
            [base_channels,     base_channels * 2,  base_depth],      # Stage 2
            [base_channels * 2, base_channels * 4,  base_depth * 3],  # Stage 3
            [base_channels * 4, base_channels * 8,  base_depth * 3],  # Stage 4
            [base_channels * 8, base_channels * 16, base_depth],      # Stage 5
        ]
        
        current_stride = self.strides[1]
        for i, (dim_in, dim_out, n_blocks) in enumerate(stage_configs):
            stage_num = i + 2
            current_stride *= 2
            self.strides[stage_num] = current_stride
            self.stage_dims[stage_num] = dim_out
            
            stage = CSPDarknetLSTMStage(
                dim_in=dim_in,
                dim_out=dim_out,
                n_blocks=n_blocks,
                is_spp_stage=(stage_num == 5),
                depthwise=depthwise,
                act=act,
                lstm_cfg=lstm_cfg,
                enable_masking=(stage_num == 2 and enable_masking),
            )
            self.stages.append(stage)
        self.num_stages = len(self.stages) + 1 

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(self.stage_dims[s] for s in stages)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(self.strides[s] for s in stages)

    def forward(self, x: torch.Tensor, prev_states: Optional[list] = None, token_mask: Optional[torch.Tensor] = None) \
            -> Tuple[Dict[int, torch.Tensor], list]:
        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages
        
        outputs: Dict[int, torch.Tensor] = {}
        states: list = []
        
        # Stage 1
        x = self.stem(x)
        h_c_stem = self.lstm_stem(x, prev_states[0])
        x = h_c_stem[0]
        outputs[1] = x
        states.append(h_c_stem)
        
        # Stages 2 to 5
        for i, stage in enumerate(self.stages):
            stage_num = i + 2
            current_mask = token_mask if stage_num == 2 else None
            x, h_c_stage = stage(x, prev_states[i+1], token_mask=current_mask)
            outputs[stage_num] = x
            states.append(h_c_stage)
            
        filtered_outputs = {k: v for k, v in outputs.items() if k in self.out_features}
        return filtered_outputs, states