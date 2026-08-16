#!/usr/bin/env python3
"""
Helper modules for BEV-only token predictor training pipeline.
Simplified version - only BEV-related modules, no LiDAR.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVEmbed(nn.Module):
    """
    Encode 3-channel satellite BEV into D-dimensional feature maps.

    改进：
    1. 添加LayerNorm以稳定特征幅度
    2. 改进权重初始化，确保特征不会被压制
    3. 添加残差连接（如果输入输出维度匹配）
    """

    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d_model // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, d_model, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 添加LayerNorm以稳定特征幅度
        self.norm = nn.GroupNorm(num_groups=min(32, d_model), num_channels=d_model)

        # 改进权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重，确保特征有合理的幅度"""
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        # 应用GroupNorm以稳定特征幅度
        feat = self.norm(feat)
        return feat


class SatGlobalCrossAttn(nn.Module):
    """
    Optional global cross-attention from front-view tokens to satellite memory.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, cond_seq: torch.Tensor, sat_mem: torch.Tensor) -> torch.Tensor:
        q = self.ln_q(cond_seq)
        kv = self.ln_kv(sat_mem)
        out, _ = self.mha(q, kv, kv, need_weights=False)
        return cond_seq + out


class FrontViewSelfAttn(nn.Module):
    """
    Light self-attention on front-view token conditions (B, L, D).
    """

    def __init__(self, d_model: int, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.ln(seq)
        x = self.encoder(x)
        return x


__all__ = [
    "BEVEmbed",
    "SatGlobalCrossAttn",
    "FrontViewSelfAttn",
]
