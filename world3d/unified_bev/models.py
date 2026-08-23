"""Small, explicit models for the two-stage unified BEV latent experiment."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import (
    bev_grid_from_world_xy,
    bilinear_splat,
    height_statistics,
    image_uv_to_grid,
    ray_distance_to_camera_z,
    render_volume,
)


class ImageFeatureEncoder(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, channels, 5, stride=2, padding=2), nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroundBEVEncoder(nn.Module):
    """Lift sparse metric LiDAR-supported image features into a canonical BEV."""

    def __init__(self, latent_channels: int = 64, bev_height: int = 128, bev_width: int = 128,
                 context_blocks: int = 4):
        super().__init__()
        self.latent_channels = latent_channels
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.image_encoder = ImageFeatureEncoder(latent_channels)
        # Input: aggregated feature + coverage + height mean/var + log count.
        in_ch = latent_channels + 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, latent_channels, 3, padding=1),
            nn.GroupNorm(8, latent_channels), nn.GELU(),
        )
        self.blocks = nn.ModuleList()
        for _ in range(context_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
                nn.GroupNorm(8, latent_channels), nn.GELU(),
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
                nn.GroupNorm(8, latent_channels),
            ))
        self.out = nn.Sequential(nn.GELU(), nn.Conv2d(latent_channels, latent_channels, 3, padding=1))

    def forward(
        self,
        images: torch.Tensor,
        points_world: torch.Tensor,
        points_uv: torch.Tensor,
        points_valid: torch.Tensor,
        origin_xy: torch.Tensor,
        resolution_m: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Z, coverage)``.

        ``images`` is ``(B,N,3,H,W)``; point tensors are ``(B,N,P,...)``.
        The source set is aggregated by sum/count, so source ordering cannot
        change the result.
        """
        B, N, _, H, W = images.shape
        feat = self.image_encoder(images.reshape(B * N, 3, H, W))
        _, C, _, _ = feat.shape
        uv = points_uv.reshape(B * N, -1, 2)
        grid = image_uv_to_grid(uv, (W, H)).view(B * N, 1, -1, 2)
        sampled = F.grid_sample(feat, grid, mode="bilinear", align_corners=False)
        sampled = sampled.squeeze(2).transpose(1, 2).reshape(B, N, -1, C)
        bev, count = bilinear_splat(
            sampled,
            points_world[..., :2],
            points_valid,
            origin_xy=origin_xy,
            resolution_m=resolution_m,
            height=self.bev_height,
            width=self.bev_width,
        )
        # Height statistics are deterministic geometry channels: splat mean
        # and variance so the column decoder can reason about vertical
        # spread instead of only average height.  bilinear_splat already
        # returns weighted means; no second division by the count.
        h_mean, h_var = height_statistics(
            points_world, points_valid, origin_xy, resolution_m,
            self.bev_height, self.bev_width,
        )
        coverage = (count > 0).to(bev.dtype)
        x = torch.cat([bev, coverage, h_mean * coverage, h_var * coverage,
                       (count + 1).log() / 5.0], dim=1)
        z = self.stem(x)
        for block in self.blocks:
            z = z + block(z)
        z = self.out(z)
        return z, coverage


class SatelliteBEVEncoder(nn.Module):
    """Encode a north-up, vehicle-centered satellite crop directly on BEV cells."""

    def __init__(self, latent_channels: int = 64, bev_height: int = 128, bev_width: int = 128):
        super().__init__()
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, latent_channels, 3, padding=1), nn.GroupNorm(8, latent_channels), nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1), nn.GELU(),
        )

    def forward(self, sat: torch.Tensor, tile_size_m: float, sat_m_per_px: float) -> torch.Tensor:
        crop = satellite_bev_crop(sat, tile_size_m, sat_m_per_px, self.bev_height)
        return self.net(crop)


def satellite_bev_crop(
    sat: torch.Tensor, tile_size_m: float, sat_m_per_px: float, size: int,
) -> torch.Tensor:
    """Canonical satellite preprocessing shared by encoder input and nadir target.

    Center-crop the north-up source to the tile extent, flip rows into the
    south-up BEV convention, and resample to ``size x size``.  Sharing this
    single function keeps the nadir supervision target and the encoder input
    bit-identical by construction.
    """
    _, _, H, W = sat.shape
    tile_px = int(round(tile_size_m / sat_m_per_px))
    tile_px = min(tile_px, H, W)
    y0, x0 = (H - tile_px) // 2, (W - tile_px) // 2
    crop = sat[..., y0 : y0 + tile_px, x0 : x0 + tile_px]
    # The source image is north-up (row 0 = max world y); the canonical
    # BEV raster is south-up, so flip rows before resampling.
    crop = torch.flip(crop, dims=[-2])
    return F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)


def nadir_distance(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked grayscale+gradient L1 between a nadir render and the satellite crop.

    ``pred``/``ref`` are ``(B,3,H,W)`` RGB in [0,1]; ``mask`` is ``(B,1,H,W)``
    with 1 marking the cells to compare (typically unobserved cells, where the
    satellite is the only external reference).  Grayscale+gradient avoids
    demanding street-view-trained colors to reproduce satellite season and
    illumination.
    """
    m = mask.bool()
    gp = pred.mean(dim=1, keepdim=True)
    gr = ref.mean(dim=1, keepdim=True)
    pdx, pdy = gp[..., 1:] - gp[..., :-1], gp[..., 1:, :] - gp[..., :-1, :]
    rdx, rdy = gr[..., 1:] - gr[..., :-1], gr[..., 1:, :] - gr[..., :-1, :]
    num = ((gp - gr).abs() * m).sum() \
        + ((pdx - rdx).abs() * m[..., 1:]).sum() \
        + ((pdy - rdy).abs() * m[..., 1:, :]).sum()
    den = m.sum() + m[..., 1:].sum() + m[..., 1:, :].sum()
    return num / den.clamp_min(1.0)


def _patch_xy_pos_embed(dim: int, grid: int, patch: int, tile_size_m: float) -> torch.Tensor:
    """Fixed sin-cos positional embedding for satellite patch tokens.

    Row 0 is south and column 0 is west on the south-up BEV raster, the same
    convention as ``fixed_relative_xy_encoding`` (the coordinate-only control's
    prior), so treatment and control share identical position semantics.  The
    values are never learned: the transformer learns how to use position and
    content, not position itself.
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4, got {dim}")
    cells = grid * patch
    rows = ((torch.arange(grid, dtype=torch.float32) * patch + patch / 2.0) / cells - 0.5) * 2.0
    cols = rows.clone()
    py, px = torch.meshgrid(rows, cols, indexing="ij")  # py varies with row (south-up)
    freqs = (2.0 ** torch.arange(dim // 4, dtype=torch.float32)) * torch.pi
    emb = torch.cat([
        torch.sin(px[..., None] * freqs).flatten(-2),
        torch.cos(px[..., None] * freqs).flatten(-2),
        torch.sin(py[..., None] * freqs).flatten(-2),
        torch.cos(py[..., None] * freqs).flatten(-2),
    ], dim=-1)
    return emb.reshape(grid * grid, dim)


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class SatelliteViTEncoder(nn.Module):
    """Patch-token satellite encoder with a global receptive field.

    The 3-layer CNN predecessor (~5 m receptive field, 58k params) did not
    survive geographic isolation: its registration-sharp features were an
    in-sample asset (grad_20260822: B7 sensitivity gone, render conversion
    negative on unseen drives).  Patch self-attention lets every output cell
    aggregate the whole 64 m tile, and the fixed metric-XY sin-cos embedding
    anchors tokens to the BEV grid.  The output contract is identical to
    ``SatelliteBEVEncoder``: ``(B, latent_channels, H, W)`` on the south-up
    raster, so completions, decoders, gates, and controls are unchanged.
    """

    def __init__(self, latent_channels: int = 64, bev_height: int = 128, bev_width: int = 128,
                 dim: int = 256, depth: int = 4, heads: int = 4, patch: int = 8,
                 tile_size_m: float = 64.0):
        super().__init__()
        if bev_height != bev_width or bev_height % patch != 0:
            raise ValueError(f"square BEV grid divisible by patch required, got {bev_height}x{bev_width} patch={patch}")
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.patch = patch
        self.grid = bev_height // patch
        self.pos_tile_size_m = float(tile_size_m)
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.blocks = nn.ModuleList(_TransformerBlock(dim, heads) for _ in range(depth))
        self.to_latent = nn.Linear(dim, latent_channels)
        self.head = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            nn.GroupNorm(8, latent_channels),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
        )
        self.register_buffer("pos", _patch_xy_pos_embed(dim, self.grid, patch, tile_size_m))

    def forward(self, sat: torch.Tensor, tile_size_m: float, sat_m_per_px: float) -> torch.Tensor:
        if abs(float(tile_size_m) - self.pos_tile_size_m) > 1e-6:
            raise ValueError(
                f"tile_size_m {tile_size_m} does not match the positional encoding "
                f"built for {self.pos_tile_size_m}"
            )
        crop = satellite_bev_crop(sat, tile_size_m, sat_m_per_px, self.bev_height)
        tokens = self.patch_embed(crop).flatten(2).transpose(1, 2) + self.pos
        for block in self.blocks:
            tokens = block(tokens)
        grid_latent = self.to_latent(tokens).transpose(1, 2).reshape(
            tokens.shape[0], -1, self.grid, self.grid,
        )
        up = F.interpolate(grid_latent, size=(self.bev_height, self.bev_width),
                           mode="bilinear", align_corners=False)
        return self.head(up)


class _CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h = self.norm_q(q)
        kvn = self.norm_kv(kv)
        q = q + self.attn(h, kvn, kvn, need_weights=False)[0]
        return q + self.mlp(self.norm2(q))


class HeightMapSatellitePrior(nn.Module):
    """CVS-pattern satellite prior: height-map regression on the BEV grid.

    Perspective/depth formulations of orthorectified imagery are a measured
    dead end (scripts/diag_vggt_gate.py: nadir depth vs LiDAR heights
    r=+0.01, depth_conf 1.0), matching Cross-View Splatter's motivation.
    Here the satellite branch instead regresses a metric height map: its
    tokens cross-attend to the street latent (both live on the same south-up
    BEV grid, so the correspondence is exact by construction), the height
    head is supervised by dense LiDAR h_mean acting as a per-tile "DEM", and
    placement needs no projection at all -- ``satellite_bev_crop`` plus the
    known meters-per-pixel already anchor cells to world XY.  The output
    ``prior`` drops into the completion slot that z_sat occupied, interface
    unchanged (identity gates unaffected: at Ns=N_dense alpha=0 discards it).

    Method borrowing from Cross-View Splatter (height branch, cross-view
    attention, top-down consistency) must be cited explicitly.
    """

    def __init__(self, latent_channels: int = 64, bev_height: int = 128, bev_width: int = 128,
                 dim: int = 256, depth: int = 4, heads: int = 4, patch: int = 8,
                 tile_size_m: float = 64.0):
        super().__init__()
        if bev_height != bev_width or bev_height % patch != 0:
            raise ValueError(f"square BEV grid divisible by patch required, got {bev_height}x{bev_width} patch={patch}")
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.patch = patch
        self.grid = bev_height // patch
        self.pos_tile_size_m = float(tile_size_m)
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.street_proj = nn.Conv2d(latent_channels, dim, kernel_size=patch, stride=patch)
        self.blocks = nn.ModuleList(_CrossAttentionBlock(dim, heads) for _ in range(depth))
        self.height_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        self.prior_head = nn.Sequential(
            nn.Conv2d(dim, latent_channels, 3, padding=1),
            nn.GroupNorm(8, latent_channels), nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
        )
        self.register_buffer("pos", _patch_xy_pos_embed(dim, self.grid, patch, tile_size_m))

    def forward(self, sat: torch.Tensor, z_gnd: torch.Tensor, tile_size_m: float,
                sat_m_per_px: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if abs(float(tile_size_m) - self.pos_tile_size_m) > 1e-6:
            raise ValueError(
                f"tile_size_m {tile_size_m} does not match the positional encoding "
                f"built for {self.pos_tile_size_m}"
            )
        crop = satellite_bev_crop(sat, tile_size_m, sat_m_per_px, self.bev_height)
        sat_tokens = self.patch_embed(crop).flatten(2).transpose(1, 2) + self.pos
        street_tokens = self.street_proj(z_gnd).flatten(2).transpose(1, 2) + self.pos
        for block in self.blocks:
            sat_tokens = block(sat_tokens, street_tokens)
        grid = sat_tokens.transpose(1, 2).reshape(
            sat_tokens.shape[0], -1, self.grid, self.grid,
        )
        full = F.interpolate(grid, size=(self.bev_height, self.bev_width),
                             mode="bilinear", align_corners=False)
        # Heights are non-negative (meters above ground zero); clamp guards
        # the far LiDAR-noise tail (single-return cells up to ~139 m).
        h_pred = F.softplus(self.height_head(full)).clamp(max=60.0)
        prior = self.prior_head(full)
        return prior, h_pred, h_pred.new_zeros(h_pred.shape)


def fixed_relative_xy_encoding(
    channels: int,
    height: int,
    width: int,
    tile_size_m: float,
) -> torch.Tensor:
    """Parameter-free metric-relative encoding on canonical BEV cells.

    The first two channels are normalized east/north cell-center offsets;
    row 0 is south and column 0 is west, matching the splat, decoder, and the
    vertically flipped satellite BEV.  Remaining channels are deterministic
    Fourier features.  The tensor is shared by every tile and contains no
    absolute GPS or scene identity.
    """
    if channels < 4 or (channels - 4) % 4 != 0:
        raise ValueError(f"channels must be 4 + 4k, got {channels}")
    if height <= 0 or width <= 0 or tile_size_m <= 0:
        raise ValueError("height, width, and tile_size_m must be positive")

    x_m = ((torch.arange(width, dtype=torch.float32) + 0.5) / width - 0.5) * tile_size_m
    y_m = ((torch.arange(height, dtype=torch.float32) + 0.5) / height - 0.5) * tile_size_m
    y_m, x_m = torch.meshgrid(y_m, x_m, indexing="ij")
    half = tile_size_m / 2.0
    x = x_m / half
    y = y_m / half
    radius = torch.sqrt(x.square() + y.square()) / (2.0 ** 0.5)
    features = [x, y, radius, torch.ones_like(x)]
    for frequency in range(1, (channels - 4) // 4 + 1):
        phase = torch.pi * frequency
        features.extend([
            torch.sin(phase * x), torch.cos(phase * x),
            torch.sin(phase * y), torch.cos(phase * y),
        ])
    return torch.stack(features, dim=0).unsqueeze(0)


class LatentCompletion(nn.Module):
    """Ground-anchored completion: a prior corrects the sparse-ground latent
    only where the ground pathway is uncertain.

    ``z_hat = z_gnd + alpha(Ns) * (1 - conf_gnd) * delta(prior, z_gnd, conf_gnd)``

    ``alpha(Ns) = 1 - Ns / N_dense`` is exactly zero at the dense source
    count, so the completion degenerates to ``z_gnd`` bitwise (regression
    tested).  The previous identity ``z_sat + mask * gate * (z_gnd - z_sat)``
    forced ``z_hat = z_sat`` on every cell outside the raw splat mask (~63%
    even at Ns=8), discarding the ground encoder's context-propagated
    information there; the dense-convergence gate was unpassable by
    construction.  The confidence must stay a learned soft map: the raw
    binary coverage is an input to it, never the fusion weight itself.
    """

    def __init__(
        self,
        channels: int = 64,
        mode: str = "residual",
        bev_height: int = 128,
        bev_width: int = 128,
        tile_size_m: float = 64.0,
    ):
        super().__init__()
        self.mode = mode
        # B3 must test whether relative XY alone is sufficient.  A trainable
        # CxHxW table is a learned spatial template, not positional encoding,
        # and adds 1M effective parameters at C=64,H=W=128.  Keep the legacy
        # ``coord_embed`` state-dict key as a buffer so old learned-template
        # checkpoints still load exactly, while new runs use this fixed grid.
        self.register_buffer(
            "coord_embed",
            fixed_relative_xy_encoding(channels, bev_height, bev_width, tile_size_m),
        )
        self.conf = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # Zero-init the correction branch: training starts from the ground
        # latent and only learns where the prior reduces the loss.
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, z_sat: torch.Tensor, z_gnd: torch.Tensor, coverage: torch.Tensor,
                n_sparse: int, dense_sources: int) -> torch.Tensor:
        if not 1 <= int(n_sparse) <= int(dense_sources):
            raise ValueError(f"n_sparse must be in [1,{dense_sources}], got {n_sparse}")
        alpha = 1.0 - float(n_sparse) / float(dense_sources)
        if self.mode == "satellite_only":
            return z_sat
        if self.mode == "ground_only":
            return z_gnd
        if self.mode == "coordinate_only":
            prior = self.coord_embed.expand(z_gnd.shape[0], -1, -1, -1)
        elif self.mode == "residual":
            prior = z_sat
        else:
            raise ValueError(f"unknown fusion mode: {self.mode}")
        conf = self.conf(torch.cat([z_gnd, coverage], dim=1))
        correction = self.delta(torch.cat([prior, z_gnd, conf], dim=1))
        return z_gnd + alpha * (1.0 - conf) * correction


class ColumnFieldDecoder(nn.Module):
    """Implicit RGB/density decoder queried by target camera rays."""

    def __init__(self, latent_channels: int = 64, hidden: int = 128, samples: int = 24):
        super().__init__()
        self.samples = samples
        self.field = nn.Sequential(
            nn.Linear(latent_channels + 4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4),
        )

    def _query(self, z: torch.Tensor, points_world: torch.Tensor, dirs_world: torch.Tensor,
               origin_xy: torch.Tensor, tile_size_m: float) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, S, _ = points_world.shape
        xy = points_world[..., :2]
        grid = bev_grid_from_world_xy(xy, origin_xy[:, None, None, None, :], tile_size_m)
        grid = grid.view(B, H * W * S, 1, 2)
        # BEV raster convention: row 0 = min world y (south), col 0 = min x.
        sampled = F.grid_sample(z, grid, mode="bilinear", align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2).view(B, H, W, S, -1)
        height = points_world[..., 2:3] / 10.0
        direction = dirs_world[:, :, :, None, :3].expand(-1, -1, -1, S, -1)
        out = self.field(torch.cat([sampled, height, direction], dim=-1))
        return out[..., :1], torch.sigmoid(out[..., 1:])

    def render(
        self,
        z: torch.Tensor,
        K: torch.Tensor,
        T_world_cam: torch.Tensor,
        origin_xy: torch.Tensor,
        *,
        tile_size_m: float,
        image_size: Tuple[int, int],
        near_m: float = 1.0,
        far_m: float = 60.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W, H = image_size
        device, dtype = z.device, z.dtype
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype) + 0.5,
            torch.arange(W, device=device, dtype=dtype) + 0.5,
            indexing="ij",
        )
        pix = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(1, H * W, 3)
        pix = pix.expand(K.shape[0], -1, -1)
        dirs_cam = pix @ torch.inverse(K).transpose(-1, -2)
        dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dirs_world = dirs_cam @ T_world_cam[:, :3, :3].transpose(-1, -2)
        origins = T_world_cam[:, None, :3, 3].expand(-1, H * W, -1)
        depths = torch.linspace(near_m, far_m, self.samples, device=device, dtype=dtype)
        pts = origins[:, :, None, :] + dirs_world[:, :, None, :] * depths[None, None, :, None]
        pts = pts.view(-1, H, W, self.samples, 3)
        dw = dirs_world.view(-1, H, W, 3)
        sigma, rgb = self._query(z, pts, dw, origin_xy, tile_size_m)
        rgb_out, ray_depth, opacity = render_volume(sigma.squeeze(-1), rgb, depths)
        # LiDAR target depth is camera-axis z.  Volume sampling uses metric
        # distance along unit rays, so convert the rendered expectation before
        # training/evaluation instead of comparing two different depth types.
        depth_z = ray_distance_to_camera_z(ray_depth, dirs_cam.view(-1, H, W, 3))
        return rgb_out.permute(0, 3, 1, 2), depth_z, opacity

    def render_nadir(
        self,
        z: torch.Tensor,
        origin_xy: torch.Tensor,
        *,
        tile_size_m: float,
        bev_size: int,
        samples: int | None = None,
        z_top_m: float = 48.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render the latent straight down, one vertical ray per BEV cell.

        The output raster follows the same south-up convention as the latent
        grid (row 0 = min world y), so ``satellite_bev_crop`` of the same tile
        is a directly comparable reference.  Rays start at ``z_top_m`` and
        integrate downward; near-to-far ordering equals top-to-bottom, which
        is the correct compositing order for a downward-looking camera.
        """
        B = z.shape[0]
        device, dtype = z.device, z.dtype
        if samples is None:
            samples = self.samples
        res = tile_size_m / bev_size
        off = (torch.arange(bev_size, device=device, dtype=dtype) + 0.5) * res
        # World XY of each cell center: row i = south (min y) + i*res, col j = west + j*res.
        xg = (origin_xy[:, 0, None, None] + off[None, None, :]).expand(B, bev_size, bev_size)
        yg = (origin_xy[:, 1, None, None] + off[None, :, None]).expand(B, bev_size, bev_size)
        dist = torch.linspace(0.0, z_top_m, samples, device=device, dtype=dtype)
        zg = (z_top_m - dist)[None, None, None, :].expand(B, bev_size, bev_size, samples)
        pts = torch.stack([
            xg.unsqueeze(-1).expand_as(zg),
            yg.unsqueeze(-1).expand_as(zg),
            zg,
        ], dim=-1)
        dirs = torch.zeros(B, bev_size, bev_size, 3, device=device, dtype=dtype)
        dirs[..., 2] = -1.0
        sigma, rgb = self._query(z, pts, dirs, origin_xy, tile_size_m)
        rgb_out, _, opacity = render_volume(sigma.squeeze(-1), rgb, dist)
        return rgb_out.permute(0, 3, 1, 2), opacity
