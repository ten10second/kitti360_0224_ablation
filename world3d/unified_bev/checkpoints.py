"""Versioned checkpoint contracts for the unified-BEV claim pipeline."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import torch


STAGE_A_SCHEMA_VERSION = 3
STAGE_B_SCHEMA_VERSION = 3
GEOMETRY_TARGET_VERSION = "relative_height_q10_clamp_-2_30_v1"


def _update_state_dict_hash(digest, name: str, state: Mapping[str, torch.Tensor]) -> None:
    digest.update(name.encode("utf-8"))
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        if tensor.numel():
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())


def compute_stage_a_fingerprint(checkpoint: Mapping[str, object]) -> str:
    """SHA-256 identity of Stage-A weights and semantic architecture metadata."""
    digest = hashlib.sha256()
    metadata = {
        key: checkpoint[key]
        for key in (
            "schema_version", "ground_config", "renderer_config",
            "geometry_decoder_config", "geometry_target_version", "grid_config",
        )
        if key in checkpoint
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for component in ("ground", "decoder", "geometry_decoder"):
        state = checkpoint.get(component)
        if not isinstance(state, Mapping):
            raise RuntimeError(f"Stage-A checkpoint is missing state dict {component!r}")
        _update_state_dict_hash(digest, component, state)
    return digest.hexdigest()


def validate_stage_a_checkpoint(checkpoint: Mapping[str, object]) -> str:
    """Validate the new shared-interface schema and return its fingerprint."""
    version = checkpoint.get("schema_version")
    if version != STAGE_A_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported Stage-A schema {version!r}; expected {STAGE_A_SCHEMA_VERSION}. "
            "Legacy checkpoints must be retrained for the shared geometry interface."
        )
    if checkpoint.get("geometry_target_version") != GEOMETRY_TARGET_VERSION:
        raise RuntimeError("Stage-A relative-height target version does not match this code")
    ground_config = checkpoint.get("ground_config")
    if not isinstance(ground_config, Mapping) or ground_config.get("family") not in {"dense", "sparse"}:
        raise RuntimeError("Stage-A checkpoint has no valid ground_config.family")
    for key in ("renderer_config", "geometry_decoder_config", "grid_config"):
        if not isinstance(checkpoint.get(key), Mapping):
            raise RuntimeError(f"Stage-A checkpoint has no {key}")
    computed = compute_stage_a_fingerprint(checkpoint)
    stored = checkpoint.get("fingerprint")
    if stored != computed:
        raise RuntimeError(
            "Stage-A fingerprint mismatch: weights or semantic metadata changed after saving"
        )
    return computed


def validate_stage_a_dataset(
    checkpoint: Mapping[str, object],
    dataset,
    *,
    dense_geometry_attached: bool,
) -> None:
    """Reject silent ground-family/grid mismatches before loading weights."""
    ground_family = checkpoint["ground_config"]["family"]
    expected_dense = ground_family == "dense"
    if expected_dense != bool(dense_geometry_attached):
        raise RuntimeError(
            f"Stage A uses ground family {ground_family!r}, but this run "
            f"{'has' if dense_geometry_attached else 'does not have'} a dense geometry cache"
        )
    grid = checkpoint["grid_config"]
    actual = {
        "bev_size": int(dataset.bev_size),
        "bev_resolution_m": float(dataset.bev_resolution_m),
        "tile_size_m": float(dataset.tile_size_m),
        "views_per_frame": int(dataset.views_per_frame),
        "target_views": int(dataset.target_views),
    }
    for key, value in actual.items():
        expected = grid.get(key)
        if expected is None or abs(float(expected) - float(value)) > 1e-6:
            raise RuntimeError(
                f"Stage-A grid mismatch for {key}: checkpoint={expected!r}, dataset={value!r}"
            )
    expected_target_layout = grid.get("target_view_layout_version")
    actual_target_layout = getattr(dataset, "target_view_layout_version", None)
    if expected_target_layout != actual_target_layout:
        raise RuntimeError(
            "Stage-A target view layout mismatch: "
            f"checkpoint={expected_target_layout!r}, dataset={actual_target_layout!r}"
        )


def validate_stage_b_checkpoint(
    checkpoint: Mapping[str, object],
    stage_a_fingerprint: str,
) -> None:
    if checkpoint.get("schema_version") != STAGE_B_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported Stage-B schema {checkpoint.get('schema_version')!r}; "
            f"expected {STAGE_B_SCHEMA_VERSION}"
        )
    if checkpoint.get("stage_a_fingerprint") != stage_a_fingerprint:
        raise RuntimeError(
            "Stage-B checkpoint was trained against a different Stage-A representation"
        )
