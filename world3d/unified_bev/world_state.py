"""Persistent georeferenced world-state contract.

This module is the v1 source of truth for shapes, provenance bits, and the
input/supervision split.  Writers and the updater may only consume
``ModelInputs`` fields; accumulated LiDAR and future-route masks live in
``SupervisionBundle`` and are rejected if they leak into a forward.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, FrozenSet, Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor


PROVENANCE_SATELLITE = 1
PROVENANCE_UAV = 2
PROVENANCE_VEHICLE = 4
PROVENANCE_INFERRED = 8
PROVENANCE_ENUM = {
    "SATELLITE": PROVENANCE_SATELLITE,
    "UAV": PROVENANCE_UAV,
    "VEHICLE": PROVENANCE_VEHICLE,
    "INFERRED": PROVENANCE_INFERRED,
}

WORLD_STATE_SCHEMA_VERSION = "world_state_v1"
# v3: targets come from the official static-semantics accumulation
# (dynamic split off, per-point labels, confidence) with label-policy-driven
# surface selection; raw scans keep only the per-chunk support masks.
WORLD_TARGET_VERSION = "official_semantics_surface_v3"
Z_DATUM_POLICY = "first_chunk_lidar_optical_center_world_z_median_v1"
C_INIT = 0.25
CHUNKING_VERSION = "route_chunk_v1"
TASK_FAMILY = "persistent_world_state"

FORBIDDEN_MODEL_INPUT_KEYS: FrozenSet[str] = frozenset({
    "world_target",
    "future_route_support",
    "final_support",
    "full_accumulated_lidar",
    "height",
    "density",
    "world_valid",
    "chunk_lidar_support",
    "query_depth",
    "query_depth_mask",
})


@dataclass(frozen=True)
class SceneTileSpec:
    scene_id: str
    origin_xy: Tensor
    tile_size_m: float
    resolution_m: float
    z_datum_m: Tensor
    chunking_version: str = CHUNKING_VERSION

    def bev_hw(self) -> Tuple[int, int]:
        size = int(round(float(self.tile_size_m) / float(self.resolution_m)))
        return size, size

    def validate(self) -> None:
        if self.origin_xy.ndim != 2 or self.origin_xy.shape[-1] != 2:
            raise ValueError(f"origin_xy must be [B,2], got {tuple(self.origin_xy.shape)}")
        if self.z_datum_m.ndim not in (1, 2) or self.z_datum_m.shape[0] != self.origin_xy.shape[0]:
            raise ValueError(
                f"z_datum_m batch {tuple(self.z_datum_m.shape)} does not match "
                f"origin_xy {tuple(self.origin_xy.shape)}"
            )
        if float(self.tile_size_m) <= 0 or float(self.resolution_m) <= 0:
            raise ValueError("tile_size_m and resolution_m must be positive")
        h, w = self.bev_hw()
        if abs(h * float(self.resolution_m) - float(self.tile_size_m)) > 1e-6:
            raise ValueError("tile_size_m must be an integer number of cells")


@dataclass
class WorldState:
    latent: Tensor
    confidence: Tensor
    provenance: Tensor
    last_update: Tensor
    spec: SceneTileSpec
    version: int

    def validate(self) -> None:
        self.spec.validate()
        b = self.spec.origin_xy.shape[0]
        h, w = self.spec.bev_hw()
        if self.latent.ndim != 4 or self.latent.shape[0] != b or self.latent.shape[-2:] != (h, w):
            raise ValueError(f"latent shape {tuple(self.latent.shape)} != [B,C,{h},{w}]")
        for name, tensor, dtype in (
            ("confidence", self.confidence, torch.float32),
            ("provenance", self.provenance, torch.uint8),
            ("last_update", self.last_update, torch.int32),
        ):
            if tensor.shape != (b, 1, h, w):
                raise ValueError(f"{name} shape {tuple(tensor.shape)} != [{b},1,{h},{w}]")
            if tensor.dtype != dtype:
                raise ValueError(f"{name} dtype {tensor.dtype} != {dtype}")
        if not bool((self.confidence >= 0).all() and (self.confidence <= 1).all()):
            raise ValueError("confidence must lie in [0, 1]")
        if int(self.version) < 0:
            raise ValueError("version must be >= 0")


@dataclass
class GroundMeasurement:
    latent: Tensor
    support: Tensor
    confidence: Tensor
    chunk_index: int

    def validate(self, spec: SceneTileSpec) -> None:
        spec.validate()
        b = spec.origin_xy.shape[0]
        h, w = spec.bev_hw()
        if self.latent.shape[0] != b or self.latent.shape[-2:] != (h, w):
            raise ValueError(f"measurement latent {tuple(self.latent.shape)} != [B,C,{h},{w}]")
        if self.support.shape != (b, 1, h, w) or self.support.dtype != torch.bool:
            raise ValueError(f"support must be bool [{b},1,{h},{w}]")
        if self.confidence.shape != (b, 1, h, w):
            raise ValueError(f"confidence shape {tuple(self.confidence.shape)}")
        if not bool((self.confidence >= 0).all() and (self.confidence <= 1).all()):
            raise ValueError("measurement confidence must lie in [0, 1]")
        if int(self.chunk_index) < 1:
            raise ValueError("vehicle chunk_index is 1-based")


@dataclass
class WorldStateUpdate:
    state: WorldState
    gate: Tensor
    correction: Tensor
    conflict: Tensor


@dataclass
class ModelInputs:
    """Fields a writer/updater is allowed to see."""

    satellite_bev: Tensor
    origin_xy: Tensor
    z_datum_m: Tensor
    scene_id: Tuple[str, ...]
    tile_size_m: float
    resolution_m: float
    chunking_version: str = CHUNKING_VERSION

    def as_spec(self) -> SceneTileSpec:
        return SceneTileSpec(
            scene_id="|".join(self.scene_id),
            origin_xy=self.origin_xy,
            tile_size_m=float(self.tile_size_m),
            resolution_m=float(self.resolution_m),
            z_datum_m=self.z_datum_m,
            chunking_version=self.chunking_version,
        )


@dataclass
class SupervisionBundle:
    """Labels and future-route masks.  Never passed to writer/updater forward."""

    height: Tensor
    density: Tensor
    world_valid: Tensor
    chunk_lidar_support: Tensor
    future_route_support: Tensor
    final_support: Tensor
    query_rgb: Tensor
    query_K: Tensor
    query_T_world_cam: Tensor
    query_depth: Tensor
    query_depth_mask: Tensor
    traversed_m: Tensor


def empty_state(spec: SceneTileSpec, channels: int, device=None, dtype=torch.float32) -> WorldState:
    spec.validate()
    b = spec.origin_xy.shape[0]
    h, w = spec.bev_hw()
    device = spec.origin_xy.device if device is None else device
    zeros = torch.zeros(b, channels, h, w, device=device, dtype=dtype)
    conf = torch.zeros(b, 1, h, w, device=device, dtype=torch.float32)
    prov = torch.zeros(b, 1, h, w, device=device, dtype=torch.uint8)
    last = torch.full((b, 1, h, w), -1, device=device, dtype=torch.int32)
    state = WorldState(zeros, conf, prov, last, spec, version=0)
    state.validate()
    return state


def clone_state(state: WorldState) -> WorldState:
    return WorldState(
        latent=state.latent.clone(),
        confidence=state.confidence.clone(),
        provenance=state.provenance.clone(),
        last_update=state.last_update.clone(),
        spec=state.spec,
        version=int(state.version),
    )


def assert_no_supervision_leak(payload: Any, *, context: str = "forward") -> None:
    """Fail-fast if a writer/updater is handed evaluation-only fields."""
    keys: Iterable[str]
    if isinstance(payload, Mapping):
        keys = payload.keys()
    elif hasattr(payload, "__dataclass_fields__"):
        keys = (f.name for f in fields(payload))
    else:
        return
    leaked = FORBIDDEN_MODEL_INPUT_KEYS.intersection(keys)
    if leaked:
        raise RuntimeError(
            f"{context} received supervision-only fields {sorted(leaked)}; "
            "writers/updater may only consume ModelInputs / GroundMeasurement"
        )


def apply_satellite_metadata(state: WorldState, valid: Tensor, c_init: float = C_INIT) -> WorldState:
    """Deterministic satellite init metadata: SATELLITE|INFERRED, last_update=0."""
    if valid.dtype != torch.bool:
        valid = valid.bool()
    conf = torch.where(valid, torch.full_like(state.confidence, float(c_init)), state.confidence)
    bits = torch.tensor(PROVENANCE_SATELLITE | PROVENANCE_INFERRED, dtype=torch.uint8, device=valid.device)
    prov = torch.where(valid, bits.view(1, 1, 1, 1).expand_as(state.provenance), state.provenance)
    last = torch.where(valid, torch.zeros_like(state.last_update), state.last_update)
    out = WorldState(state.latent, conf, prov, last, state.spec, version=0)
    out.validate()
    return out


def apply_vehicle_metadata(
    previous: WorldState,
    updated_latent: Tensor,
    updated_confidence: Tensor,
    support: Tensor,
    chunk_index: int,
) -> WorldState:
    """OR VEHICLE into support cells; last_update=t; leave unsupported cells identical."""
    if int(chunk_index) < 1:
        raise ValueError("vehicle chunk_index is 1-based")
    support = support.bool()
    latent = torch.where(support, updated_latent, previous.latent)
    conf = torch.where(support, updated_confidence.clamp(0.0, 1.0), previous.confidence)
    vehicle = torch.tensor(PROVENANCE_VEHICLE, dtype=torch.uint8, device=support.device)
    prov = previous.provenance.clone()
    prov = torch.where(support, prov | vehicle, prov)
    last = torch.where(
        support,
        torch.full_like(previous.last_update, int(chunk_index)),
        previous.last_update,
    )
    out = WorldState(latent, conf, prov, last, previous.spec, version=int(chunk_index))
    out.validate()
    return out


def assert_preserved_outside_support(before: WorldState, after: WorldState, support: Tensor) -> None:
    outside = ~support.bool()
    for name in ("latent", "confidence", "provenance", "last_update"):
        prev = getattr(before, name)
        now = getattr(after, name)
        if name == "latent":
            mask = outside.expand_as(prev)
        else:
            mask = outside
        if not torch.equal(prev[mask], now[mask]):
            raise RuntimeError(f"{name} changed outside measurement.support")


def visited_mask(chunk_support: Tensor, t: int) -> Tensor:
    """``chunk_support`` is [B,T,1,H,W]; ``t`` is 1-based inclusive."""
    if t < 0:
        raise ValueError(t)
    if t == 0:
        return torch.zeros_like(chunk_support[:, 0], dtype=torch.bool)
    return chunk_support[:, :t].any(dim=1)


def ahead_mask(future_route_support: Tensor, visited: Tensor) -> Tensor:
    return future_route_support.bool() & ~visited.bool()


def offroute_mask(world_valid: Tensor, all_ground_supported: Tensor) -> Tensor:
    """Coverage diagnostic only; never a v1 headline region."""
    return world_valid.bool() & ~all_ground_supported.bool()


def supervised_region(measurement_support: Tensor, world_valid: Tensor) -> Tensor:
    """Where a measurement may be supervised against the static world target.

    VGGT support extends well beyond LiDAR coverage, and the target maps are
    zero (not "unknown") outside ``world_valid``; supervising the raw
    measurement support would push unlabelled cells toward height 0 / density
    0 — pseudo-negative labels, not masked supervision.
    """
    return measurement_support.bool() & world_valid.bool()
