"""Per-chunk frozen-VGGT ground measurements for the world-state chain.

One cache file per scene.  Each chunk entry stores the chunk's calibrated
views plus VGGT's frozen, vehicle-motion-scaled depth/confidence at view
resolution, so training and evaluation can build ``GroundMeasurement``
objects without re-running VGGT.  The cache is bound to the scene blob's
``world_target_version``/``world_target_hash``: rebuilding targets
invalidates the cache instead of silently mixing contracts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from .state_models import GroundMeasurementEncoder
from .world_state import GroundMeasurement

WORLD_VGGT_CACHE_VERSION = "world_vggt_chunk_measurement_v2"

_ENTRY_RANKS = {"rgb": 5, "K": 4, "depth": 4, "conf": 4, "T_world_cam": 4}


def assert_query_isolation(entry: Dict, query_fid: int) -> None:
    """The depth-query frame must never be among the VGGT measurement frames.

    Without this, ``depth_absrel`` on the query pose is not held-out: the
    measurement literally saw the frame it is evaluated against.
    """
    for key in ("query_fid", "measurement_fids"):
        if key not in entry:
            raise RuntimeError(
                f"cache entry lacks {key}; it predates query isolation — "
                "rebuild with the v2 cache builder"
            )
    if int(entry["query_fid"]) != int(query_fid):
        raise RuntimeError(
            f"cache entry query_fid {entry['query_fid']} != blob core fid "
            f"{query_fid}; chunk identity mismatch"
        )
    if int(query_fid) in {int(f) for f in entry["measurement_fids"]}:
        raise RuntimeError(
            "query frame leaked into the VGGT measurement frames; "
            "depth_absrel would not be held-out"
        )


def load_world_vggt_cache(
    cache_root: str,
    scene_id: str,
    world_target_version: str,
    world_target_hash: str,
) -> Dict:
    """Load and identity-check one scene's VGGT measurement cache."""
    path = Path(cache_root) / f"{scene_id}.pt"
    if not path.exists():
        raise RuntimeError(
            f"missing VGGT measurement cache {path}; run scripts/build_world_vggt_cache.py"
        )
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("schema") != WORLD_VGGT_CACHE_VERSION:
        raise RuntimeError(
            f"{path} has schema {cache.get('schema')!r}; expected {WORLD_VGGT_CACHE_VERSION!r}"
        )
    for key, expected in (
        ("scene_id", scene_id),
        ("world_target_version", world_target_version),
        ("world_target_hash", world_target_hash),
    ):
        got = cache.get(key)
        if got != expected:
            raise RuntimeError(
                f"{path} {key} mismatch: cache={got!r} targets={expected!r}; "
                "rebuild the cache for the current targets"
            )
    if not cache.get("chunks"):
        raise RuntimeError(f"{path} contains no chunk measurements")
    return cache


def chunk_measurement_from_cache(
    encoder: GroundMeasurementEncoder,
    entry: Dict,
    *,
    origin_xy: torch.Tensor,
    resolution_m: float,
    z_datum_m: torch.Tensor,
    chunk_index: int,
    query_fid: int,
    detach: bool = False,
    dgm_abs_z: Optional[torch.Tensor] = None,
    dgm_valid: Optional[torch.Tensor] = None,
) -> GroundMeasurement:
    """Turn one cached chunk entry into a GroundMeasurement.

    ``entry`` tensors are stored half precision; they are cast to float here.
    ``detach`` stops gradients at the measurement latent (used for prefix
    replay); the frozen VGGT side never carries gradients either way.
    ``query_fid`` is the chunk's depth-query core frame; the entry must be
    isolated from it (``assert_query_isolation``).
    ``dgm_abs_z``/``dgm_valid`` are the scene's pre-sampled DGM tile
    (absolute DHHN2016 heights on the BEV grid); without them the encoder
    uses the legacy camera-rig median anchor.
    """
    assert_query_isolation(entry, query_fid)

    def _view(key: str) -> torch.Tensor:
        tensor = entry[key].float().to(device)
        while tensor.ndim < _ENTRY_RANKS[key]:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != _ENTRY_RANKS[key]:
            raise ValueError(f"cache entry {key} has rank {tensor.ndim}; expected {_ENTRY_RANKS[key]}")
        return tensor

    device = next(encoder.parameters()).device

    measurement = encoder(
        images=_view("rgb"),
        K=_view("K"),
        dense_depth=_view("depth"),
        dense_conf=_view("conf"),
        T_world_cam=_view("T_world_cam"),
        origin_xy=origin_xy,
        resolution_m=float(resolution_m),
        z_datum_m=z_datum_m,
        chunk_index=int(chunk_index),
        dgm_abs_z=dgm_abs_z.to(device) if dgm_abs_z is not None else None,
        dgm_valid=dgm_valid.to(device) if dgm_valid is not None else None,
    )
    if detach:
        measurement.latent = measurement.latent.detach()
    return measurement


def teacher_measurement(
    latent: torch.Tensor,
    support: torch.Tensor,
    chunk_index: int,
) -> GroundMeasurement:
    """LiDAR-supervision fallback: encode the world teacher restricted to the
    current chunk's support.  Diagnostic only — this reads accumulated LiDAR
    supervision and never validates the VGGT measurement path."""
    return GroundMeasurement(
        latent=latent, support=support,
        confidence=support.float() * 0.8, chunk_index=int(chunk_index),
    )
