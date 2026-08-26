"""World-defined geometry targets and georeferenced satellite resampling.

Height is ``surface_height_p90_world_z - z_datum_m``.  ``z_datum_m`` is the
median world-Z of LiDAR optical centers in the scene, not a per-chunk
quantile.  Cells without LiDAR evidence stay unknown and are never labelled
free space.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .data import SAT_M_PER_PX
from .geometry import image_uv_to_grid, sparse_depth_zbuffer


def bev_cell_centers(origin_xy: Tensor, tile_size_m: float, resolution_m: float) -> Tensor:
    """South-up pixel-center XY.  Returns ``[B,H,W,2]`` world metres."""
    if origin_xy.ndim != 2 or origin_xy.shape[-1] != 2:
        raise ValueError(f"origin_xy must be [B,2], got {tuple(origin_xy.shape)}")
    size = int(round(float(tile_size_m) / float(resolution_m)))
    device, dtype = origin_xy.device, origin_xy.dtype
    off = (torch.arange(size, device=device, dtype=dtype) + 0.5) * float(resolution_m)
    x = origin_xy[:, 0, None, None] + off[None, None, :]
    y = origin_xy[:, 1, None, None] + off[None, :, None]
    return torch.stack([x.expand(-1, size, size), y.expand(-1, size, size)], dim=-1)


def satellite_pixel_centers(sat_center_xy: Tensor, height_px: int, width_px: int,
                            sat_m_per_px: float = SAT_M_PER_PX) -> Tuple[Tensor, Tensor]:
    """Pixel-center world XY of a north-up vehicle-centered satellite crop.

    Column 0 is west, row 0 is north.  Returns ``(x_of_col, y_of_row)``.
    """
    dtype, device = sat_center_xy.dtype, sat_center_xy.device
    col = torch.arange(width_px, device=device, dtype=dtype)
    row = torch.arange(height_px, device=device, dtype=dtype)
    x = sat_center_xy[:, 0, None] + (col - (width_px / 2.0 - 0.5)) * float(sat_m_per_px)
    y = sat_center_xy[:, 1, None] + ((height_px / 2.0 - 0.5) - row) * float(sat_m_per_px)
    return x, y


def georeferenced_satellite_resample(
    sat: Tensor,
    sat_center_xy: Tensor,
    origin_xy: Tensor,
    *,
    tile_size_m: float = 100.0,
    resolution_m: float = 0.5,
    sat_m_per_px: float = SAT_M_PER_PX,
) -> Tensor:
    """Sample a north-up 512x512 satellite onto a south-up world BEV grid.

    Mapping uses pixel centers.  Any output cell whose source coordinate
    leaves the asset raises; v1 tiles must sit inside one 100.352 m crop.
    """
    if sat.ndim != 4 or sat.shape[1] != 3:
        raise ValueError(f"sat must be [B,3,H,W], got {tuple(sat.shape)}")
    if sat_center_xy.shape != origin_xy.shape:
        raise ValueError("sat_center_xy and origin_xy must share shape [B,2]")
    b, _, hs, ws = sat.shape
    size = int(round(float(tile_size_m) / float(resolution_m)))
    centers = bev_cell_centers(origin_xy, tile_size_m, resolution_m)
    wx, wy = centers[..., 0], centers[..., 1]
    col = (wx - sat_center_xy[:, 0, None, None]) / float(sat_m_per_px) + (ws / 2.0 - 0.5)
    row = (sat_center_xy[:, 1, None, None] - wy) / float(sat_m_per_px) + (hs / 2.0 - 0.5)
    if bool((col < -1e-6).any() or (col > ws - 1 + 1e-6).any()
            or (row < -1e-6).any() or (row > hs - 1 + 1e-6).any()):
        raise RuntimeError(
            "world BEV leaves the source satellite asset; choose an anchor whose "
            "512x512 crop fully covers the 100 m tile"
        )
    grid = image_uv_to_grid(torch.stack([col, row], dim=-1), (ws, hs))
    sampled = F.grid_sample(sat, grid, mode="bilinear", align_corners=False)
    if sampled.shape[-2:] != (size, size):
        raise RuntimeError(f"resampled satellite {tuple(sampled.shape)} != [{b},3,{size},{size}]")
    return sampled


def satellite_mapping_error_px(
    sat_center_xy: Tensor,
    origin_xy: Tensor,
    *,
    tile_size_m: float = 100.0,
    resolution_m: float = 0.5,
    sat_h: int = 512,
    sat_w: int = 512,
    sat_m_per_px: float = SAT_M_PER_PX,
) -> Tensor:
    """Absolute source-pixel error of the four corners and the tile center."""
    size = int(round(float(tile_size_m) / float(resolution_m)))
    centers = bev_cell_centers(origin_xy, tile_size_m, resolution_m)
    samples = torch.stack([
        centers[:, 0, 0],
        centers[:, 0, -1],
        centers[:, -1, 0],
        centers[:, -1, -1],
        centers[:, size // 2, size // 2],
    ], dim=1)
    col = (samples[..., 0] - sat_center_xy[:, 0:1]) / float(sat_m_per_px) + (sat_w / 2.0 - 0.5)
    row = (sat_center_xy[:, 1:1 + 1] - samples[..., 1]) / float(sat_m_per_px) + (sat_h / 2.0 - 0.5)
    x_src, y_src = satellite_pixel_centers(sat_center_xy, sat_h, sat_w, sat_m_per_px)
    # invert: nearest analytic pixel-center should reconstruct world XY
    recon_x = sat_center_xy[:, 0:1] + (col - (sat_w / 2.0 - 0.5)) * float(sat_m_per_px)
    recon_y = sat_center_xy[:, 1:1 + 1] + ((sat_h / 2.0 - 0.5) - row) * float(sat_m_per_px)
    err_m = torch.stack([(recon_x - samples[..., 0]).abs(), (recon_y - samples[..., 1]).abs()], dim=-1)
    return err_m / float(sat_m_per_px)


def lidar_optical_center_world(T_world_cam: np.ndarray, T_cam_velo: np.ndarray) -> np.ndarray:
    T_world_velo = np.asarray(T_world_cam, dtype=np.float64) @ np.asarray(T_cam_velo, dtype=np.float64)
    return T_world_velo[:3, 3].copy()


def z_datum_from_centers(centers_z: Sequence[float]) -> float:
    values = np.asarray(list(centers_z), dtype=np.float64)
    if values.size == 0:
        raise ValueError("z_datum requires at least one LiDAR optical center")
    return float(np.median(values))


def accumulate_lidar_surface(
    points_world: np.ndarray,
    origin_xy: np.ndarray,
    *,
    tile_size_m: float,
    resolution_m: float,
    quantile: float = 0.9,
    min_height_m: float = -2.0,
    max_height_m: float = 40.0,
) -> Dict[str, np.ndarray]:
    """Bin static world points onto the south-up BEV grid.

    Returns height (world Z p90), count, and a valid mask.  Empty cells stay
    unknown (valid=False) and are not labelled free.
    """
    size = int(round(float(tile_size_m) / float(resolution_m)))
    height = np.full((size, size), np.nan, dtype=np.float64)
    count = np.zeros((size, size), dtype=np.int32)
    if points_world.size == 0:
        return {"height_world_z": height, "count": count, "valid": np.zeros((size, size), dtype=bool)}
    xy = np.asarray(points_world[:, :2], dtype=np.float64)
    z = np.asarray(points_world[:, 2], dtype=np.float64)
    local = (xy - np.asarray(origin_xy, dtype=np.float64)[None, :]) / float(resolution_m)
    col = np.floor(local[:, 0]).astype(np.int64)
    row = np.floor(local[:, 1]).astype(np.int64)
    inside = (col >= 0) & (col < size) & (row >= 0) & (row < size) & np.isfinite(z)
    row, col, z = row[inside], col[inside], z[inside]
    if row.size == 0:
        return {"height_world_z": height, "count": count, "valid": np.zeros((size, size), dtype=bool)}
    flat = row * size + col
    order = np.argsort(flat)
    flat_s = flat[order]
    z_s = z[order]
    splits = np.flatnonzero(np.diff(flat_s)) + 1
    groups = np.split(z_s, splits)
    keys = np.unique(flat_s)
    for key, zs in zip(keys, groups):
        r, c = divmod(int(key), size)
        count[r, c] = zs.size
        height[r, c] = np.quantile(zs, quantile)
    valid = count > 0
    height = np.clip(height, min_height_m, max_height_m)
    return {"height_world_z": height, "count": count, "valid": valid}


def log_normalize_density(count: np.ndarray) -> np.ndarray:
    count = np.asarray(count, dtype=np.float64)
    positive = count[count > 0]
    ref = float(np.quantile(positive, 0.99)) if positive.size else 1.0
    ref = max(ref, 1.0)
    density = np.zeros_like(count, dtype=np.float32)
    density[count > 0] = (np.log1p(count[count > 0]) / np.log1p(ref)).astype(np.float32)
    return np.clip(density, 0.0, 1.0)


def height_minus_datum(height_world_z: np.ndarray, z_datum_m: float) -> np.ndarray:
    out = np.asarray(height_world_z, dtype=np.float32) - float(z_datum_m)
    out[~np.isfinite(height_world_z)] = 0.0
    return out


def query_sparse_depth(
    points_world: torch.Tensor,
    K: torch.Tensor,
    T_world_cam: torch.Tensor,
    image_size: Tuple[int, int],
) -> Tuple[Tensor, Tensor]:
    return sparse_depth_zbuffer(points_world, T_world_cam, K, image_size)
