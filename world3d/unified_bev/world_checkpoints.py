"""world_state_v1 checkpoint fingerprints.  Independent of Stage-A/B schema 3."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import torch

from .checkpoints import _update_state_dict_hash
from .world_state import WORLD_STATE_SCHEMA_VERSION, WORLD_TARGET_VERSION, Z_DATUM_POLICY


def compute_world_interface_fingerprint(checkpoint: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    metadata = {
        key: checkpoint[key]
        for key in (
            "schema_version", "world_target_version", "z_datum_policy",
            "scenes_manifest_hash",
            "encoder_config", "height_reader_config", "density_reader_config",
            "depth_reader_config", "grid_config", "chunk_config",
            "provenance_enum",
        )
        if key in checkpoint
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in ("encoder", "height_reader", "density_reader", "depth_reader"):
        state = checkpoint.get(name)
        if not isinstance(state, Mapping):
            raise RuntimeError(f"world-interface checkpoint missing {name}")
        _update_state_dict_hash(digest, name, state)
    return digest.hexdigest()


def validate_world_interface_checkpoint(checkpoint: Mapping[str, object]) -> str:
    if checkpoint.get("schema_version") != WORLD_STATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported world-state schema {checkpoint.get('schema_version')!r}; "
            f"expected {WORLD_STATE_SCHEMA_VERSION}"
        )
    if checkpoint.get("world_target_version") != WORLD_TARGET_VERSION:
        raise RuntimeError("world-target version mismatch")
    if checkpoint.get("z_datum_policy") != Z_DATUM_POLICY:
        raise RuntimeError("z_datum policy mismatch")
    computed = compute_world_interface_fingerprint(checkpoint)
    if checkpoint.get("fingerprint") != computed:
        raise RuntimeError("world-interface fingerprint mismatch")
    return computed


def validate_scenes_manifest(checkpoint: Mapping[str, object], manifest_hash: str) -> None:
    """Bind a checkpoint to the exact scene set it was fitted on.

    Only for code paths that reload the SAME scenes the interface was trained
    on (e.g. assimilation training).  Held-out evaluation scenes deliberately
    use a different manifest and must not call this.
    """
    bound = checkpoint.get("scenes_manifest_hash")
    if bound is not None and bound != manifest_hash:
        raise RuntimeError(
            "scene manifest mismatch: interface was fitted on a different "
            "target set — rebuild targets or retrain the interface"
        )


def validate_assimilation_checkpoint(checkpoint: Mapping[str, object], interface_fp: str) -> None:
    if checkpoint.get("schema_version") != WORLD_STATE_SCHEMA_VERSION:
        raise RuntimeError("assimilation checkpoint has the wrong schema")
    if checkpoint.get("interface_fingerprint") != interface_fp:
        raise RuntimeError("assimilation checkpoint was trained against a different frozen interface")
