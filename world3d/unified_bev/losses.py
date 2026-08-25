"""Observation-aware losses for unified BEV latent recovery."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _expanded_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(-3)
    if mask.ndim != reference.ndim:
        raise ValueError(
            f"mask rank {mask.ndim} cannot broadcast to reference rank {reference.ndim}"
        )
    if mask.shape[:-3] != reference.shape[:-3] or mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is incompatible with {tuple(reference.shape)}"
        )
    if mask.shape[-3] == 1 and reference.shape[-3] != 1:
        mask = mask.expand(*mask.shape[:-3], reference.shape[-3], *mask.shape[-2:])
    elif mask.shape[-3] != reference.shape[-3]:
        raise ValueError(
            f"mask channels {mask.shape[-3]} do not match reference channels {reference.shape[-3]}"
        )
    return mask.to(reference.dtype)


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 mean over supported values, with a safe differentiable zero.

    Empty regions are expected for some tiles/Ns settings.  Returning
    ``pred.sum() * 0`` avoids NaNs while retaining a valid autograd graph.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shapes differ: {pred.shape} vs {target.shape}")
    weights = _expanded_mask(mask, pred)
    denom = weights.sum()
    if not bool(denom > 0):
        return pred.sum() * 0.0
    values = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return (values * weights).sum() / denom


def low_frequency(x: torch.Tensor, scale: int = 8) -> torch.Tensor:
    """Average-pooled low-frequency image/layout representation."""
    if x.ndim < 4:
        raise ValueError("low_frequency expects (...,C,H,W)")
    if scale < 1:
        raise ValueError(f"scale must be >=1, got {scale}")
    kernel = min(int(scale), x.shape[-2], x.shape[-1])
    if kernel == 1:
        return x
    leading = x.shape[:-3]
    pooled = F.avg_pool2d(x.reshape(-1, *x.shape[-3:]), kernel_size=kernel, stride=kernel)
    return pooled.reshape(*leading, *pooled.shape[-3:])


def low_frequency_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: int = 8,
) -> torch.Tensor:
    """Full-image low-frequency layout/color loss."""
    return F.smooth_l1_loss(low_frequency(pred, scale), low_frequency(target, scale))


def high_frequency(x: torch.Tensor, scale: int = 8) -> torch.Tensor:
    """Residual after removing the pooled-and-upsampled low-frequency image."""
    low = low_frequency(x, scale)
    if low.shape[-2:] != x.shape[-2:]:
        leading = low.shape[:-3]
        low = F.interpolate(
            low.reshape(-1, *low.shape[-3:]), size=x.shape[-2:],
            mode="bilinear", align_corners=False,
        ).reshape(*leading, low.shape[-3], *x.shape[-2:])
    return x - low


def high_frequency_masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: int = 8,
) -> torch.Tensor:
    """High-frequency appearance loss only where ground evidence supports it."""
    return masked_smooth_l1(high_frequency(pred, scale), high_frequency(target, scale), mask)
