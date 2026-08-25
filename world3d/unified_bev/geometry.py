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
        w = weight * ok.to(dtype)
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


def relative_height_map(
    points_world: torch.Tensor,
    points_valid: torch.Tensor,
    origin_xy: torch.Tensor,
    resolution_m: float,
    height: int,
    width: int,
    *,
    quantile: float = 0.1,
    min_height_m: float = -2.0,
    max_height_m: float = 30.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one translation-invariant relative-height target per BEV cell.

    World-Z in KITTI-360 is an absolute map altitude.  A geometry readout
    should instead predict height relative to the local road/ground level.
    The ground anchor is therefore estimated independently for every sample
    from the low quantile of *covered* cell means.  Empty cells never enter
    the quantile and are explicitly reset to zero after subtraction.

    Returns:
        ``height_relative``: ``(B,1,H,W)`` in meters.
        ``valid_mask``: bool ``(B,1,H,W)`` marking cells with observations.
        ``ground_z``: ``(B,1,1,1)`` absolute world-Z anchor in meters.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0,1], got {quantile}")
    if min_height_m >= max_height_m:
        raise ValueError("min_height_m must be smaller than max_height_m")
    h_abs, count = bilinear_splat(
        points_world[..., 2:3],
        points_world[..., :2],
        points_valid,
        origin_xy=origin_xy,
        resolution_m=resolution_m,
        height=height,
        width=width,
    )
    valid_mask = count > 0
    ground_z = h_abs.new_zeros((h_abs.shape[0], 1, 1, 1))
    for batch_index in range(h_abs.shape[0]):
        covered = h_abs[batch_index, 0][valid_mask[batch_index, 0]]
        if covered.numel() > 0:
            ground_z[batch_index, 0, 0, 0] = torch.quantile(
                covered.float(), quantile,
            ).to(h_abs.dtype)
    height_relative = (h_abs - ground_z).clamp(min=min_height_m, max=max_height_m)
    height_relative = torch.where(valid_mask, height_relative, torch.zeros_like(height_relative))
    return height_relative, valid_mask, ground_z


def observation_partition(
    sparse_support: torch.Tensor,
    dense_geometry_support: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic observed and geometry-completion BEV regions.

    ``M_obs`` is exactly the sparse-ground support. ``M_fill`` contains cells
    with a dense geometry target but no sparse-ground support.  The masks are
    boolean and disjoint by construction; no learned confidence is involved.
    """
    if sparse_support.shape != dense_geometry_support.shape:
        raise ValueError(
            "support masks must have the same shape, got "
            f"{tuple(sparse_support.shape)} and {tuple(dense_geometry_support.shape)}"
        )
    observed = sparse_support.bool()
    fill = dense_geometry_support.bool() & ~observed
    return observed, fill


def geometry_supervision_support(
    dense_geometry_support: torch.Tensor,
    height_target_valid: torch.Tensor,
) -> torch.Tensor:
    """Cells where both the dense lift and metric height target are valid.

    The frozen-height loss cannot supervise a VGGT-covered cell without a
    LiDAR relative-height label, and it should not headline a LiDAR-only cell
    that the dense reference encoder itself did not support.  Keeping this
    intersection explicit prevents those two notions of "dense" from being
    conflated in Stage B and evaluation.
    """
    if dense_geometry_support.shape != height_target_valid.shape:
        raise ValueError(
            "dense geometry and height-target masks must have the same shape, got "
            f"{tuple(dense_geometry_support.shape)} and {tuple(height_target_valid.shape)}"
        )
    return dense_geometry_support.bool() & height_target_valid.bool()


def target_pixels_supported_by_bev(
    depth_z: torch.Tensor,
    depth_valid: torch.Tensor,
    K: torch.Tensor,
    T_world_cam: torch.Tensor,
    origin_xy: torch.Tensor,
    tile_size_m: float,
    bev_support: torch.Tensor,
) -> torch.Tensor:
    """Project target teacher-depth pixels into a BEV observation mask.

    The returned ``(B,1,H,W)`` bool mask identifies target RGB pixels whose
    backprojected world XY lies in a sparse-ground-supported BEV cell.  This
    is the provenance mask used to restrict high-frequency appearance loss;
    pixels without valid teacher depth are deliberately unsupported.
    """
    if K.ndim == 4:
        batch_size, target_views = K.shape[:2]
        if depth_z.shape[:2] != (batch_size, target_views):
            raise ValueError("multi-target depth and K dimensions differ")
        support = bev_support[:, None].expand(-1, target_views, -1, -1, -1).reshape(
            batch_size * target_views, *bev_support.shape[1:]
        )
        origin = origin_xy[:, None].expand(-1, target_views, -1).reshape(-1, 2)
        result = target_pixels_supported_by_bev(
            depth_z.reshape(-1, *depth_z.shape[-2:]),
            depth_valid.reshape(-1, *depth_valid.shape[-2:]),
            K.reshape(-1, 3, 3), T_world_cam.reshape(-1, 4, 4), origin,
            tile_size_m, support,
        )
        return result.reshape(batch_size, target_views, *result.shape[1:])
    if depth_z.ndim == 4 and depth_z.shape[1] == 1:
        depth_z = depth_z[:, 0]
    if depth_valid.ndim == 4 and depth_valid.shape[1] == 1:
        depth_valid = depth_valid[:, 0]
    if depth_z.ndim != 3 or depth_valid.shape != depth_z.shape:
        raise ValueError("depth_z and depth_valid must have shape (B,H,W)")
    if bev_support.ndim != 4 or bev_support.shape[1] != 1:
        raise ValueError("bev_support must have shape (B,1,Hb,Wb)")
    B, H, W = depth_z.shape
    device, dtype = depth_z.device, depth_z.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype) + 0.5,
        torch.arange(W, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    pixels = pixels.reshape(1, H * W, 3).expand(B, -1, -1)
    rays_cam = pixels @ torch.linalg.inv(K.to(dtype)).transpose(-1, -2)
    points_cam = rays_cam * depth_z.reshape(B, H * W, 1)
    points_world = (
        points_cam @ T_world_cam[:, :3, :3].to(dtype).transpose(-1, -2)
        + T_world_cam[:, None, :3, 3].to(dtype)
    )
    grid = bev_grid_from_world_xy(
        points_world[..., :2].reshape(B, H, W, 2),
        origin_xy[:, None, None, :],
        tile_size_m,
    )
    sampled = F.grid_sample(
        bev_support.to(dtype), grid, mode="nearest", padding_mode="zeros",
        align_corners=False,
    )
    finite_depth = torch.isfinite(depth_z) & (depth_z > 1e-3)
    inside = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) \
        & (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    return (sampled > 0.5) & depth_valid[:, None].bool() & finite_depth[:, None] & inside[:, None]


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
