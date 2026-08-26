"""World-defined encoder, frozen readers, source writers, and updater.

Stage A fits ``WorldGeometryEncoder`` plus height/density/depth readers to
accumulated LiDAR.  Stage B freezes those modules.  Satellite and ground
writers never see future-route labels; the updater never sees satellite or
historical images.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import bilinear_splat, height_statistics
from .models import (
    ColumnFieldDecoder,
    GroundDenseBEVEncoder,
    fixed_relative_xy_encoding,
    unproject_dense,
)
from .readouts import BEVHeightDecoder, freeze_module
from .world_state import (
    C_INIT,
    FORBIDDEN_MODEL_INPUT_KEYS,
    GroundMeasurement,
    ModelInputs,
    SceneTileSpec,
    WorldState,
    WorldStateUpdate,
    apply_satellite_metadata,
    apply_vehicle_metadata,
    assert_no_supervision_leak,
    assert_preserved_outside_support,
    empty_state,
)


class WorldGeometryEncoder(nn.Module):
    """Encode world-defined height/density/valid maps.  No RGB, VGGT, or satellite."""

    def __init__(self, latent_channels: int = 64, context_blocks: int = 4):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(3, latent_channels, 3, padding=1),
            nn.GroupNorm(8, latent_channels), nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
                nn.GroupNorm(8, latent_channels), nn.GELU(),
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
                nn.GroupNorm(8, latent_channels),
            )
            for _ in range(context_blocks)
        )
        self.out = nn.Sequential(
            nn.GELU(), nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
        )

    def forward(self, height: torch.Tensor, density: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if height.shape != density.shape or height.shape != valid.shape:
            raise ValueError("height/density/valid must share shape [B,1,H,W]")
        x = torch.cat([height, density, valid.to(height.dtype)], dim=1)
        z = self.stem(x)
        for block in self.blocks:
            z = z + block(z)
        return self.out(z)


class BEVWorldHeightDecoder(BEVHeightDecoder):
    """Same topology as ``BEVHeightDecoder``; target is world-Z p90 minus datum."""


class BEVSurfaceDensityDecoder(BEVHeightDecoder):
    """Same topology; output is log-normalized surface density in [0, 1]."""

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(latent))


def _drop_dino_keys(state):
    return {k: v for k, v in state.items() if ".dino." not in k and not k.startswith("dino.")}


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINO_INPUT = 224
_XY_CHANNELS = 16


class FrozenDINOv2(nn.Module):
    """DINOv2 ViT-B/14 encoder.  Weights are never trained or checkpointed."""

    def __init__(self, name: str = "dinov2_vitb14"):
        super().__init__()
        # Prefer the local hub cache (repo zip + pretrain weights) so nothing
        # phones home; fall back to GitHub on a machine without a cache.
        repo_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if repo_dir.is_dir():
            self.model = torch.hub.load(str(repo_dir), name, source="local", pretrained=True)
        else:
            self.model = torch.hub.load(
                "facebookresearch/dinov2", name, pretrained=True, skip_validation=True,
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.patch = 14
        self.embed_dim = int(self.model.embed_dim)
        self.register_buffer(
            "mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected [B,3,H,W], got {tuple(image.shape)}")
        x = F.interpolate(image, size=(_DINO_INPUT, _DINO_INPUT), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        b, n, c = tokens.shape
        side = int(n ** 0.5)
        if side * side != n:
            raise RuntimeError(f"DINOv2 patch tokens are not square: N={n}")
        return tokens.transpose(1, 2).reshape(b, c, side, side)


class SatelliteInitializer(nn.Module):
    """Z0 from a georeferenced south-up satellite raster.

    Default backbone is frozen DINOv2-ViT-B/14.  Only ``write_head`` is trained.
    ``backbone='tiny'`` is a frozen conv stub for unit tests (no hub download).
    """

    def __init__(
        self,
        latent_channels: int = 64,
        bev_height: int = 200,
        bev_width: int = 200,
        tile_size_m: float = 100.0,
        c_init: float = C_INIT,
        backbone: str = "dinov2_vitb14",
        **_ignored,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.bev_height = int(bev_height)
        self.bev_width = int(bev_width)
        self.c_init = float(c_init)
        self.pos_tile_size_m = float(tile_size_m)
        self.backbone = str(backbone)
        if backbone == "tiny":
            stub = nn.Conv2d(3, 32, 3, stride=2, padding=1)
            for parameter in stub.parameters():
                parameter.requires_grad_(False)
            self.dino = stub
            self.feat_dim = 32
        elif backbone.startswith("dinov2"):
            self.dino = FrozenDINOv2(backbone)
            self.feat_dim = self.dino.embed_dim
        else:
            raise ValueError(f"unknown satellite backbone {backbone!r}")
        self.register_buffer(
            "xy",
            fixed_relative_xy_encoding(_XY_CHANNELS, bev_height, bev_width, tile_size_m),
        )
        self.write_head = nn.Sequential(
            nn.Conv2d(self.feat_dim + _XY_CHANNELS, latent_channels, 1),
            nn.GroupNorm(8, latent_channels), nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    def empty_features(self, satellite_bev: torch.Tensor) -> torch.Tensor:
        """Zeros with the same map size DINO would produce (XY control)."""
        if self.backbone == "tiny":
            with torch.no_grad():
                ref = self.dino(satellite_bev)
            return torch.zeros_like(ref)
        side = _DINO_INPUT // 14
        return satellite_bev.new_zeros(
            satellite_bev.shape[0], self.feat_dim, side, side,
        )

    def encode(self, satellite_bev: torch.Tensor) -> torch.Tensor:
        if self.backbone == "tiny":
            with torch.no_grad():
                return self.dino(satellite_bev)
        return self.dino(satellite_bev)

    def write(self, features: torch.Tensor, spec: SceneTileSpec) -> WorldState:
        up = F.interpolate(
            features, size=(self.bev_height, self.bev_width),
            mode="bilinear", align_corners=False,
        )
        xy = self.xy.to(dtype=up.dtype, device=up.device).expand(up.shape[0], -1, -1, -1)
        latent = self.write_head(torch.cat([up, xy], dim=1))
        valid = torch.ones(
            up.shape[0], 1, self.bev_height, self.bev_width,
            dtype=torch.bool, device=up.device,
        )
        state = empty_state(spec, self.latent_channels, device=latent.device, dtype=latent.dtype)
        state.latent = latent
        return apply_satellite_metadata(state, valid, self.c_init)

    def forward(self, satellite_bev: torch.Tensor, spec: SceneTileSpec, **kwargs) -> WorldState:
        assert_no_supervision_leak(kwargs, context="SatelliteInitializer")
        if kwargs:
            raise RuntimeError(f"SatelliteInitializer got unexpected kwargs {sorted(kwargs)}")
        if satellite_bev.ndim != 4 or satellite_bev.shape[1] != 3:
            raise ValueError(f"satellite_bev must be [B,3,H,W], got {tuple(satellite_bev.shape)}")
        if abs(float(spec.tile_size_m) - self.pos_tile_size_m) > 1e-6:
            raise ValueError("tile_size_m does not match the XY encoding")
        return self.write(self.encode(satellite_bev), spec)

    def state_dict(self, *args, **kwargs):
        return _drop_dino_keys(super().state_dict(*args, **kwargs))

    def load_state_dict(self, state_dict, strict: bool = True):
        filtered = _drop_dino_keys(state_dict)
        incompatible = super().load_state_dict(filtered, strict=False)
        if strict:
            missing = [k for k in incompatible.missing_keys if ".dino." not in k and not k.startswith("dino.")]
            unexpected = list(incompatible.unexpected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    f"satellite write-head mismatch missing={missing[:8]} unexpected={unexpected[:8]}"
                )
        return incompatible


class FixedXYInitializer(nn.Module):
    """Same write head as satellite init; DINO features replaced by zeros."""

    def __init__(self, **kwargs):
        super().__init__()
        self.inner = SatelliteInitializer(**kwargs)

    def forward(self, satellite_bev: torch.Tensor, spec: SceneTileSpec, **kwargs) -> WorldState:
        assert_no_supervision_leak(kwargs, context="FixedXYInitializer")
        if kwargs:
            raise RuntimeError(f"FixedXYInitializer got unexpected kwargs {sorted(kwargs)}")
        return self.inner.write(self.inner.empty_features(satellite_bev), spec)

    def state_dict(self, *args, **kwargs):
        return _drop_dino_keys(super().state_dict(*args, **kwargs))

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.inner.load_state_dict(
            {k[len("inner."):] if k.startswith("inner.") else k: v for k, v in state_dict.items()},
            strict=strict,
        )


class GroundMeasurementEncoder(nn.Module):
    """Lift one chunk's calibrated views into a world-grid measurement.

    Height features are ``world_z - z_datum``, never a per-chunk quantile.
    Support/confidence come from geometry evidence, not free attention.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        bev_height: int = 200,
        bev_width: int = 200,
        context_blocks: int = 4,
        conf_threshold: float = 0.3,
        min_depth_m: float = 0.5,
        max_depth_m: float = 60.0,
    ):
        super().__init__()
        self.lift = GroundDenseBEVEncoder(
            latent_channels=latent_channels, bev_height=bev_height, bev_width=bev_width,
            context_blocks=context_blocks, conf_threshold=conf_threshold,
            min_depth_m=min_depth_m, max_depth_m=max_depth_m,
        )
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.conf_threshold = conf_threshold
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def forward(
        self,
        images: torch.Tensor,
        K: torch.Tensor,
        dense_depth: torch.Tensor,
        dense_conf: torch.Tensor,
        T_world_cam: torch.Tensor,
        origin_xy: torch.Tensor,
        resolution_m: float,
        z_datum_m: torch.Tensor,
        chunk_index: int,
        **kwargs,
    ) -> GroundMeasurement:
        assert_no_supervision_leak(kwargs, context="GroundMeasurementEncoder")
        B, N, _, H, W = images.shape
        dense_depth = torch.nan_to_num(dense_depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
        dense_conf = torch.nan_to_num(dense_conf.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        feat = self.lift.image_encoder(images.reshape(B * N, 3, H, W))
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        C = feat.shape[1]
        feat = feat.view(B, N, C, H * W).transpose(2, 3)
        pts = unproject_dense(dense_depth, K, T_world_cam).view(B, N, H * W, 3)
        gate = (
            (dense_conf > self.conf_threshold)
            & (dense_depth > self.min_depth_m)
            & (dense_depth < self.max_depth_m)
        ).view(B, N, H * W)
        bev, count = bilinear_splat(
            feat, pts[..., :2], gate,
            origin_xy=origin_xy, resolution_m=resolution_m,
            height=self.bev_height, width=self.bev_width,
        )
        z_abs, _ = bilinear_splat(
            pts[..., 2:3], pts[..., :2], gate,
            origin_xy=origin_xy, resolution_m=resolution_m,
            height=self.bev_height, width=self.bev_width,
        )
        datum = z_datum_m.view(B, 1, 1, 1).to(z_abs.dtype)
        h_rel = (z_abs - datum) * (count > 0).to(z_abs.dtype)
        _, h_var = height_statistics(
            pts, gate, origin_xy, resolution_m, self.bev_height, self.bev_width,
        )
        coverage = (count > 0).to(bev.dtype)
        x = torch.cat([
            bev, coverage, h_rel, h_var * coverage, (count + 1).log() / 5.0,
        ], dim=1)
        z = self.lift.stem(x)
        for block in self.lift.blocks:
            z = z + block(z)
        z = self.lift.out(z)
        conf_splat, _ = bilinear_splat(
            dense_conf.view(B, N, H * W, 1), pts[..., :2], gate,
            origin_xy=origin_xy, resolution_m=resolution_m,
            height=self.bev_height, width=self.bev_width,
        )
        support = coverage.bool()
        confidence = torch.where(support, conf_splat.clamp(0.0, 1.0), torch.zeros_like(conf_splat))
        meas = GroundMeasurement(z, support, confidence, int(chunk_index))
        meas.validate(SceneTileSpec(
            scene_id="measurement", origin_xy=origin_xy,
            tile_size_m=float(self.bev_height) * float(resolution_m),
            resolution_m=float(resolution_m), z_datum_m=z_datum_m.view(B, 1),
        ))
        return meas


class EvidenceAwareUpdater(nn.Module):
    """Recurrent write: support-gated residual; unsupported cells stay bitwise equal."""

    def __init__(self, channels: int = 64):
        super().__init__()
        in_ch = channels * 2 + 2
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, 1, 1), nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, previous_state: WorldState, measurement: GroundMeasurement, **kwargs) -> WorldStateUpdate:
        assert_no_supervision_leak(kwargs, context="EvidenceAwareUpdater")
        if kwargs:
            raise RuntimeError(f"EvidenceAwareUpdater got unexpected kwargs {sorted(kwargs)}")
        previous_state.validate()
        measurement.validate(previous_state.spec)
        support = measurement.support
        features = torch.cat([
            previous_state.latent, measurement.latent,
            previous_state.confidence, measurement.confidence,
        ], dim=1)
        learned = self.gate(features)
        alpha = support.to(learned.dtype) * learned
        proposal = previous_state.latent + self.delta(features)
        latent = (1.0 - alpha) * previous_state.latent + alpha * proposal
        new_conf = torch.max(previous_state.confidence, measurement.confidence)
        state = apply_vehicle_metadata(
            previous_state, latent, new_conf, support, measurement.chunk_index,
        )
        assert_preserved_outside_support(previous_state, state, support)
        conflict = (previous_state.latent - measurement.latent).abs().mean(dim=1, keepdim=True)
        conflict = torch.where(support, conflict, torch.zeros_like(conflict))
        return WorldStateUpdate(state=state, gate=alpha, correction=alpha * (proposal - previous_state.latent),
                                conflict=conflict)


def aggregate_measurements(measurements: Sequence[GroundMeasurement]) -> GroundMeasurement:
    """Parameter-free union used by the one-shot control."""
    if not measurements:
        raise ValueError("need at least one measurement")
    support = measurements[0].support.clone()
    conf = measurements[0].confidence.clone()
    acc = measurements[0].latent * support.to(measurements[0].latent.dtype)
    count = support.to(measurements[0].latent.dtype)
    for meas in measurements[1:]:
        s = meas.support
        support = support | s
        conf = torch.max(conf, meas.confidence)
        w = s.to(meas.latent.dtype)
        acc = acc + meas.latent * w
        count = count + w
    latent = acc / count.clamp_min(1.0)
    return GroundMeasurement(latent, support, conf, chunk_index=max(m.chunk_index for m in measurements))


class OneShotAssimilator(nn.Module):
    """Same updater weights; one write from the union of all measurements."""

    def __init__(self, updater: EvidenceAwareUpdater):
        super().__init__()
        self.updater = updater

    def forward(self, initial: WorldState, measurements: Sequence[GroundMeasurement]) -> WorldStateUpdate:
        return self.updater(initial, aggregate_measurements(list(measurements)))


def freeze_interface(encoder: nn.Module, readers: Sequence[nn.Module]) -> None:
    freeze_module(encoder)
    for reader in readers:
        freeze_module(reader)
