"""Frozen dense-ground readout interfaces for the unified BEV latent."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BEVHeightDecoder(nn.Module):
    """Convolutional coarse-to-fine relative-height readout.

    This is intentionally named by its actual contract rather than DPT: it
    consumes one BEV feature map, builds a convolutional pyramid by pooling,
    and predicts one relative-height value per BEV cell.  Stage A trains one
    instance from dense-ground ``Z*``; every Stage-B/eval branch reuses those
    exact frozen weights.
    """

    def __init__(self, latent_channels: int = 64, width: int = 64):
        super().__init__()
        if width % 8 != 0:
            raise ValueError(f"width must be divisible by 8, got {width}")
        self.latent_channels = int(latent_channels)
        self.width = int(width)
        self.stem = _ConvBlock(latent_channels, width)
        self.stage2 = _ConvBlock(width, width * 2)
        self.stage3 = _ConvBlock(width * 2, width * 4)
        self.stage4 = _ConvBlock(width * 4, width * 4)
        self.proj4 = nn.Conv2d(width * 4, width, 1)
        self.proj3 = nn.Conv2d(width * 4, width, 1)
        self.proj2 = nn.Conv2d(width * 2, width, 1)
        self.fuse2 = _ConvBlock(width * 2, width)
        self.fuse1 = _ConvBlock(width * 2, width)
        self.out = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )

    @staticmethod
    def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=reference.shape[-2:], mode="bilinear", align_corners=False,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected (B,{self.latent_channels},H,W), got {tuple(latent.shape)}"
            )
        s1 = self.stem(latent)
        s2 = self.stage2(F.avg_pool2d(s1, 2))
        s3 = self.stage3(F.avg_pool2d(s2, 2))
        s4 = self.stage4(F.avg_pool2d(s3, 2))
        f4 = self.proj4(s4)
        f3 = self.proj3(s3) + self._resize_like(f4, s3)
        f2 = self.fuse2(torch.cat([
            self.proj2(s2), self._resize_like(f3, s2),
        ], dim=1))
        f1 = self.fuse1(torch.cat([s1, self._resize_like(f2, s1)], dim=1))
        return self.out(f1)


def freeze_module(module: nn.Module) -> nn.Module:
    """Put a module in inference mode and permanently exclude its parameters."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module

