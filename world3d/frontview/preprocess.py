"""
Front-view on-the-fly preprocessor for pipeline use (no offline generation):
- Estimate per-frame sky crop from LiDAR v-percentiles
- Keep full width; snap (W,H) to multiples of `patch` (default 16)
- Apply crop+resize to RGB and mask tensors
- Adjust intrinsics K -> K'
- Utility to build FrontRaySampler aligned with the token grid

Typical outcome on KITTI image_02 (1242x375): crop sky -> target 1248x224 => 14x78 token grid.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from world3d.geometry.calib_utils import adjust_intrinsics_after_crop_resize
from world3d.geometry.front_rays import FrontRaySampler


@dataclass
class FrontViewParams:
    crop_top: int
    crop_bottom: int
    target_w: int
    target_h: int
    rows: int
    cols: int
    K_prime: torch.Tensor  # (3,3)


def _round_to_multiple(x: int, m: int, mode: str = 'nearest') -> int:
    if mode == 'down':
        return (int(x) // m) * m
    if mode == 'up':
        return ((int(x) + m - 1) // m) * m
    return int(round(int(x) / m) * m)


def estimate_crop_and_target(
    K: torch.Tensor,
    T_cam_lidar: torch.Tensor,
    pts_velo: np.ndarray,
    img_w: int,
    img_h: int,
    patch: int = 16,
    v_percentiles: Tuple[float, float] = (2.0, 98.0),
    force_target_wh: Optional[Tuple[int, int]] = None,
    conservative_crop: bool = False,
    fixed_crop_ratio: Optional[float] = None,
) -> Tuple[int, int, int, int, torch.Tensor]:
    """
    Estimate per-frame crop and target size. Returns (crop_top, crop_bottom, target_w, target_h, K').
    If force_target_wh is provided (e.g., (1248, 224)), we keep width and adjust height accordingly to multiples of patch.

    Args:
        conservative_crop: If True, use less aggressive cropping to preserve more foreground content
        fixed_crop_ratio: If provided, crop this fraction from the top (e.g., 0.2 = crop top 20%)
    """
    assert pts_velo.ndim == 2 and pts_velo.shape[1] >= 3

    # Project LiDAR to image using K, T
    R = T_cam_lidar[:3, :3].cpu().numpy()
    t = T_cam_lidar[:3, 3].cpu().numpy().reshape(3, 1)
    P = torch.zeros(3, 4, dtype=K.dtype)
    P[:3, :3] = K
    P = P.cpu().numpy()
    pts = pts_velo[:, :3].T  # (3,N)
    pts_cam = (R @ pts) + t  # (3,N)
    xyz1 = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]), dtype=pts_cam.dtype)])
    uvw = P @ xyz1
    u = uvw[0] / np.maximum(uvw[2], 1e-6)
    v = uvw[1] / np.maximum(uvw[2], 1e-6)
    z = uvw[2]
    mask = (z > 0) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    v_valid = v[mask]

    # Determine crop_top based on method
    if fixed_crop_ratio is not None:
        # Fixed ratio cropping (e.g., crop top 15% for KITTI)
        crop_top = _round_to_multiple(int(img_h * fixed_crop_ratio), patch, mode='down')
    elif v_valid.size == 0:
        # Fallback: if no points, don't crop
        crop_top = 0
    else:
        # LiDAR-based adaptive cropping
        if conservative_crop:
            # More conservative: use 5th percentile instead of 2nd, and add safety margin
            v_lo = float(np.percentile(v_valid, max(5.0, v_percentiles[0])))
            safety_margin = patch * 2  # Add 2 patch rows as safety margin
            crop_top = max(0, _round_to_multiple(int(math.floor(v_lo)) - safety_margin, patch, mode='down'))
        else:
            # Original aggressive cropping
            v_lo = float(np.percentile(v_valid, v_percentiles[0]))
            crop_top = max(0, _round_to_multiple(int(math.floor(v_lo)), patch, mode='down'))

    crop_bottom = 0  # keep to bottom
    crop_h = img_h - crop_top - crop_bottom
    if crop_h < patch * 8:
        crop_top = max(0, img_h - patch * 8)
        crop_h = img_h - crop_top

    # Decide target size
    if force_target_wh is not None:
        target_w, target_h = force_target_wh
        # snap to multiples of patch just in case
        target_w = _round_to_multiple(target_w, patch, 'nearest')
        target_h = _round_to_multiple(target_h, patch, 'nearest')
    else:
        target_w = _round_to_multiple(img_w, patch, 'nearest')  # keep full width
        target_h = _round_to_multiple(crop_h, patch, 'nearest')

    # Adjust intrinsics after crop+resize
    Kp = adjust_intrinsics_after_crop_resize(
        K, orig_w=img_w, orig_h=img_h,
        crop_top=crop_top, crop_bottom=crop_bottom,
        out_w=target_w, out_h=target_h,
    )

    return crop_top, crop_bottom, target_w, target_h, Kp


def apply_crop_resize_tensor(x: torch.Tensor, crop_top: int, crop_bottom: int, out_w: int, out_h: int) -> torch.Tensor:
    """
    Apply vertical crop (keep full width) then resize to (out_w,out_h).
    x: (B,C,H,W) or (C,H,W). Returns same rank.
    """
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    top = crop_top
    bottom = H - crop_bottom
    x = x[:, :, top:bottom, :]
    x = F.interpolate(x, size=(out_h, out_w), mode='bilinear', align_corners=False)
    return x.squeeze(0) if single else x


def build_front_sampler(
    Kp: torch.Tensor,
    T_cam_lidar: torch.Tensor,
    rows: int,
    cols: int,
    grid_cfg: dict,
    near_m: float = 2.0,
    far_m: float = 80.0,
    N: int = 16,
    compose: str = 'soft',
    beta: float = 8.0,
) -> FrontRaySampler:
    """Create a FrontRaySampler aligned to the (rows x cols) token grid."""
    return FrontRaySampler(Kp, T_cam_lidar, grid_cfg, Hp=rows, Wp=cols,
                           near_m=near_m, far_m=far_m, N=N, compose=compose, beta=beta)


def compute_params_for_frame(
    K: torch.Tensor,
    T_cam_lidar: torch.Tensor,
    pts_velo: np.ndarray,
    img_w: int,
    img_h: int,
    patch: int = 16,
    force_kitii_1248x224: bool = True,
    conservative_crop: bool = False,
    fixed_crop_ratio: Optional[float] = None,
    target_patch_grid: Optional[tuple] = None,  # (width_patches, height_patches)
) -> FrontViewParams:
    """
    Convenience wrapper for KITTI: produce 1248x224 (14x78) by default when feasible.

    Args:
        conservative_crop: Use less aggressive sky cropping to preserve more foreground
        fixed_crop_ratio: If provided, crop this fraction from top (e.g., 0.15 for KITTI)
        target_patch_grid: If provided, force specific patch grid (width_patches, height_patches)
    """
    if target_patch_grid is not None:
        # Force specific patch grid size
        width_patches, height_patches = target_patch_grid
        force_wh = (width_patches * patch, height_patches * patch)
    else:
        force_wh = (1248, 224) if force_kitii_1248x224 else None

    crop_top, crop_bottom, target_w, target_h, Kp = estimate_crop_and_target(
        K, T_cam_lidar, pts_velo, img_w, img_h, patch=patch, force_target_wh=force_wh,
        conservative_crop=conservative_crop, fixed_crop_ratio=fixed_crop_ratio
    )
    rows, cols = target_h // patch, target_w // patch
    return FrontViewParams(crop_top, crop_bottom, target_w, target_h, rows, cols, Kp)

