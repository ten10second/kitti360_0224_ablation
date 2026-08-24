"""Geometry primitives for the unified KITTI-360 BEV latent pipeline.

All transforms use column vectors and are named ``T_dst_src``: ``p_dst =
T_dst_src @ p_src``.  The satellite image is north-up, with east increasing
to the right and north increasing upward in the world frame.  BEV rasters
(splat output, decoder sampling grid, satellite crops after the vertical
flip in the satellite encoder) use row 0 = min world y (south edge) and
column 0 = min world x (west edge).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Invert a batch of rigid transforms."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    out = torch.zeros_like(T)
    out[..., :3, :3] = R.transpose(-1, -2)
    out[..., :3, 3:4] = -R.transpose(-1, -2) @ t
    out[..., 3, 3] = 1.0
    return out


def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply ``T`` to points with shape ``(..., N, 3)``."""
    return points @ T[..., :3, :3].transpose(-1, -2) + T[..., :3, 3].unsqueeze(-2)


def project_points_to_image(
    points_world: torch.Tensor,
    T_world_cam: torch.Tensor,
    K: torch.Tensor,
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points into a rectified camera.

    Returns ``(uv, depth, valid)``.  ``image_size`` is ``(width, height)``.
    """
    W, H = image_size
    T_cam_world = se3_inverse(T_world_cam)
    points_cam = transform_points(T_cam_world, points_world)
    z = points_cam[..., 2]
    uvw = points_cam @ K.transpose(-1, -2)
    uv = uvw[..., :2] / z.unsqueeze(-1).clamp_min(1e-6)
    valid = (
        (z > 1e-3)
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < W)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < H)
    )
    return uv, z, valid


def sparse_depth_zbuffer(
    points_world: torch.Tensor,
    T_world_cam: torch.Tensor,
    K: torch.Tensor,
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a sparse metric depth image with a nearest-point z-buffer."""
    W, H = image_size
    uv, depth, valid = project_points_to_image(points_world, T_world_cam, K, image_size)
    depth_img = torch.zeros((H, W), dtype=points_world.dtype, device=points_world.device)
    if valid.any():
        xy = uv[valid].long()
        d = depth[valid]
        # Sorting ascending makes the first write the closest point.  The
        # loop is intentionally simple: this runs only for target supervision.
        order = torch.argsort(d)
        xy = xy[order]
        d = d[order]
        flat = xy[:, 1] * W + xy[:, 0]
        seen = torch.zeros(H * W, dtype=torch.bool, device=points_world.device)
        keep = ~seen[flat]
        seen[flat[keep]] = True
        depth_img.view(-1)[flat[keep]] = d[keep]
    return depth_img, depth_img > 0


def bev_grid_from_world_xy(
    xy_m: torch.Tensor,
    origin_xy: torch.Tensor,
    tile_size_m: float,
) -> torch.Tensor:
    """grid_sample coordinates for world XY on a BEV raster.

    Row 0 is the min-y (south) edge and column 0 the min-x (west) edge,
    matching :func:`bilinear_splat`.  ``xy_m`` has shape ``(..., 2)``;
    ``origin_xy`` broadcasts against its leading dimension.
    """
    denom = float(tile_size_m)
    gx = (xy_m[..., 0] - origin_xy[..., 0]) / denom * 2 - 1
    gy = (xy_m[..., 1] - origin_xy[..., 1]) / denom * 2 - 1
    return torch.stack([gx, gy], dim=-1)


def image_uv_to_grid(uv: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """Convert image pixel-center coordinates to ``grid_sample`` coordinates.

    ``uv=(0,0)`` and ``uv=(W-1,H-1)`` denote the centers of the first and
    last image pixels.  The formula matches ``align_corners=False`` and does
    not depend on the sampled feature-map resolution.
    """
    W, H = image_size
    gx = 2.0 * (uv[..., 0] + 0.5) / float(W) - 1.0
    gy = 2.0 * (uv[..., 1] + 0.5) / float(H) - 1.0
    return torch.stack([gx, gy], dim=-1)


def bilinear_splat(
    values: torch.Tensor,
    xy_m: torch.Tensor,
    valid: torch.Tensor,
    *,
    origin_xy: torch.Tensor,
    resolution_m: float,
    height: int,
    width: int,
    point_weights: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splat point values onto a BEV grid with bilinear weights.

    Args:
        values: ``(B, N, P, C)`` point features.
        xy_m: ``(B, N, P, 2)`` world XY coordinates.
        valid: ``(B, N, P)`` point validity mask.
        origin_xy: ``(B, 2)`` world coordinate of the raster's south-west
            corner; cell ``r`` covers ``[origin + r*res, origin + (r+1)*res)``
            and its center is ``origin + (r + 0.5) * res`` (pixel-center
            convention, matching ``grid_sample(align_corners=False)``).

    Returns:
        ``bev`` with shape ``(B,C,H,W)`` and a floating coverage/count map
        with shape ``(B,1,H,W)``.
    """
    B, N, P, C = values.shape
    device, dtype = values.device, values.dtype
    # Pixel-center convention: subtract 0.5 cells so that a point at a cell
    # center lands exactly on that cell, matching the decoder's grid_sample.
    local = (xy_m - origin_xy[:, None, None, :]) / float(resolution_m) - 0.5
    gx, gy = local[..., 0], local[..., 1]
    x0, y0 = torch.floor(gx).long(), torch.floor(gy).long()
    wx, wy = gx - x0.float(), gy - y0.float()

    acc = torch.zeros((B, C, height * width), dtype=dtype, device=device)
    count = torch.zeros((B, 1, height * width), dtype=dtype, device=device)
    for dx, dy, weight in (
        (0, 0, (1 - wx) * (1 - wy)),
        (1, 0, wx * (1 - wy)),
        (0, 1, (1 - wx) * wy),
        (1, 1, wx * wy),
    ):
        xi, yi = x0 + dx, y0 + dy
        ok = valid & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        w = weight * ok.to(dtype) if point_weights is None else weight * ok.to(dtype) * point_weights.to(dtype)
        index = (yi.clamp(0, height - 1) * width + xi.clamp(0, width - 1)).reshape(B, -1)
        weighted = (values * w[..., None]).reshape(B, -1, C).transpose(1, 2)
        acc.scatter_add_(2, index[:, None, :].expand(-1, C, -1), weighted)
        count.scatter_add_(2, index[:, None, :], w.reshape(B, 1, -1))
    bev = acc / count.clamp_min(1e-6)
    return bev.view(B, C, height, width), count.view(B, 1, height, width)


def height_statistics(
    points_world: torch.Tensor,
    points_valid: torch.Tensor,
    origin_xy: torch.Tensor,
    resolution_m: float,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-cell bilinear-weighted mean and variance of point heights.

    ``bilinear_splat`` already returns weighted means (sum / weight), so the
    count map must NOT divide them again.
    """
    h = points_world[..., 2:3]
    stats, _ = bilinear_splat(
        torch.cat([h, h * h], dim=-1),
        points_world[..., :2],
        points_valid,
        origin_xy=origin_xy,
        resolution_m=resolution_m,
        height=height,
        width=width,
    )
    h_mean = stats[..., :1, :, :]
    h_var = (stats[..., 1:, :, :] - h_mean * h_mean).clamp_min(0)
    return h_mean, h_var


def ray_distance_to_camera_z(distance: torch.Tensor, dirs_cam: torch.Tensor) -> torch.Tensor:
    """Convert metric distance along a unit camera ray to camera-axis depth."""
    return distance * dirs_cam[..., 2]


def render_volume(
    sigma: torch.Tensor,
    rgb: torch.Tensor,
    depths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard alpha compositing for sampled column-field values."""
    deltas = depths[..., 1:] - depths[..., :-1]
    tail = torch.full_like(deltas[..., :1], 1e3)
    deltas = torch.cat([deltas, tail], dim=-1)
    alpha = 1.0 - torch.exp(-F.softplus(sigma) * deltas)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-8], dim=-1), dim=-1
    )[..., :-1]
    weights = alpha * trans
    rgb_out = (weights[..., None] * rgb).sum(dim=-2)
    depth_out = (weights * depths).sum(dim=-1)
    opacity = weights.sum(dim=-1)
    return rgb_out, depth_out, opacity
