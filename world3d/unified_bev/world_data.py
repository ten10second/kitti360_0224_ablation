"""Scene-centered world-state dataset: one 100 m tile, ordered ground chunks."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .chunks import RouteChunk, build_route_chunks, core_member_index, select_chunk_frames
from .data import (
    SAT_M_PER_PX,
    UnifiedBEVDataset,
    _open_image,
    load_frame_records,
)
from .world_state import (
    CHUNKING_VERSION,
    FORBIDDEN_MODEL_INPUT_KEYS,
    ModelInputs,
    SceneTileSpec,
    SupervisionBundle,
    WORLD_TARGET_VERSION,
    Z_DATUM_POLICY,
    assert_no_supervision_leak,
)
from .world_targets import (
    accumulate_lidar_surface,
    georeferenced_satellite_resample,
    height_minus_datum,
    lidar_optical_center_world,
    log_normalize_density,
    z_datum_from_centers,
)


TILE_SIZE_M = 100.0
RESOLUTION_M = 0.5
MAX_CHUNKS = 8
FRAMES_PER_CHUNK = 4
MAX_GEOMETRY_FRAMES = 8
GUARD_M = 0.0
CHUNK_ARC_M = 12.0
MIN_FRAMES_PER_CHUNK = 6


def _points_world(rec, max_points: Optional[int] = None) -> np.ndarray:
    pts = np.fromfile(rec.lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    if max_points is not None and len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        pts = pts[idx]
    T = rec.T_world_cam.astype(np.float64) @ rec._T_cam_velo.astype(np.float64)
    return pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]


def _scene_id(drive: str, anchor_fid: int, origin_xy: np.ndarray) -> str:
    return f"{drive}__fid{anchor_fid:010d}__ox{origin_xy[0]:.1f}_oy{origin_xy[1]:.1f}"


def _hash_targets(height, density, valid) -> str:
    h = hashlib.sha256()
    for arr in (height, density, valid.astype(np.uint8)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def propose_scenes(
    records: List,
    *,
    tile_size_m: float = TILE_SIZE_M,
    max_chunks: int = MAX_CHUNKS,
    min_chunks: int = 4,
    chunk_arc_m: float = CHUNK_ARC_M,
    min_frames_per_chunk: int = MIN_FRAMES_PER_CHUNK,
    min_scene_gap_m: float = 80.0,
    max_scenes: Optional[int] = None,
) -> List[dict]:
    """Non-overlapping scene tiles along one drive's trajectory."""
    if not records:
        return []
    xy = np.asarray([[r.T_world_imu[0, 3], r.T_world_imu[1, 3]] for r in records], dtype=np.float64)
    chunks = build_route_chunks(xy, [r.fid for r in records], chunk_arc_m=chunk_arc_m)
    usable = [c for c in chunks if len(c.fids) >= min_frames_per_chunk]
    by_fid = {r.fid: r for r in records}
    placed: List[np.ndarray] = []
    scenes: List[dict] = []
    i = 0
    while i < len(usable):
        first = usable[i]
        window: List[RouteChunk] = []
        for c in usable[i:]:
            if c.segment != first.segment:
                break
            window.append(c)
            if len(window) >= max_chunks:
                break
        if len(window) < min_chunks:
            i += 1
            continue
        centers = np.stack([c.center_xy for c in window])
        mean_xy = centers.mean(axis=0)
        if any(np.linalg.norm(mean_xy - p) < min_scene_gap_m for p in placed):
            i += max(1, len(window) // 2)
            continue
        window_fids = [f for c in window for f in c.fids]
        anchor = min(
            (by_fid[f] for f in window_fids if f in by_fid),
            key=lambda r: (
                np.hypot(r.T_world_imu[0, 3] - mean_xy[0], r.T_world_imu[1, 3] - mean_xy[1]),
                r.fid,
            ),
        )
        sat_center = np.asarray([anchor.T_world_imu[0, 3], anchor.T_world_imu[1, 3]], dtype=np.float64)
        origin = sat_center - tile_size_m / 2.0
        if np.any((centers < origin).any(axis=1) | (centers > origin + tile_size_m).any(axis=1)):
            i += 1
            continue
        try:
            selections = {
                c.index: select_chunk_frames(
                    c, FRAMES_PER_CHUNK, GUARD_M, MAX_GEOMETRY_FRAMES,
                    guard_left=False,
                )
                for c in window
            }
            cores = {c.index: core_member_index(c, 0.0) for c in window}
        except ValueError:
            i += 1
            continue
        scenes.append({
            "drive": anchor.drive,
            "anchor_fid": int(anchor.fid),
            "sat_center_xy": sat_center,
            "origin_xy": origin,
            "window": window,
            "selections": selections,
            "cores": cores,
            "by_fid": by_fid,
            "anchor": anchor,
        })
        placed.append(mean_xy)
        if max_scenes is not None and len(scenes) >= max_scenes:
            return scenes
        i += len(window)
    return scenes


def build_scene_blob(
    proposal: dict,
    view_helper: UnifiedBEVDataset,
    *,
    tile_size_m: float = TILE_SIZE_M,
    resolution_m: float = RESOLUTION_M,
    split: str = "train",
) -> dict:
    window: List[RouteChunk] = proposal["window"]
    by_fid = proposal["by_fid"]
    origin = proposal["origin_xy"]
    sat_center = proposal["sat_center_xy"]
    anchor = proposal["anchor"]
    centers_z = []
    all_points = []
    chunk_points: List[np.ndarray] = []
    for c in window:
        pts_c = []
        for fid in c.fids:
            rec = by_fid[fid]
            centers_z.append(float(lidar_optical_center_world(rec.T_world_cam, rec._T_cam_velo)[2]))
            pts = _points_world(rec)
            pts_c.append(pts)
            all_points.append(pts)
        chunk_points.append(np.concatenate(pts_c, axis=0) if pts_c else np.zeros((0, 3)))
    z_datum = z_datum_from_centers(centers_z)
    packed = accumulate_lidar_surface(
        np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3)),
        origin, tile_size_m=tile_size_m, resolution_m=resolution_m,
    )
    height = height_minus_datum(packed["height_world_z"], z_datum)
    density = log_normalize_density(packed["count"])
    valid = packed["valid"]
    size = height.shape[0]
    chunk_support = []
    for pts in chunk_points:
        sub = accumulate_lidar_surface(pts, origin, tile_size_m=tile_size_m, resolution_m=resolution_m)
        chunk_support.append(sub["valid"])
    chunk_support_np = np.stack(chunk_support, axis=0)
    # drop scenes that do not expand coverage
    visited = np.zeros((size, size), dtype=bool)
    new_counts = []
    keep_idx = []
    for i, mask in enumerate(chunk_support_np):
        added = int((mask & ~visited).sum())
        new_counts.append(added)
        if added >= 1:
            keep_idx.append(i)
        visited |= mask
    if not keep_idx:
        raise RuntimeError("scene has no chunk that adds support")
    frac_nonempty = len(keep_idx) / len(window)
    final_cells = int(visited.sum())
    median_new = float(np.median([new_counts[i] for i in keep_idx]))
    if frac_nonempty < 0.8 or (final_cells > 0 and median_new < 0.05 * final_cells):
        raise RuntimeError(
            f"scene coverage gate failed: nonempty_frac={frac_nonempty:.2f} "
            f"median_new={median_new:.1f} final={final_cells}"
        )

    with _open_image(anchor.sat_path) as im:
        sat = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32)).permute(2, 0, 1) / 255.0
    sat_b = sat.unsqueeze(0)
    origin_t = torch.tensor(origin, dtype=torch.float32).view(1, 2)
    center_t = torch.tensor(sat_center, dtype=torch.float32).view(1, 2)
    satellite_bev = georeferenced_satellite_resample(
        sat_b, center_t, origin_t, tile_size_m=tile_size_m, resolution_m=resolution_m,
    )[0].cpu()

    from .geometry import sparse_depth_zbuffer as _zbuf
    queries = []
    for c in window:
        rec = by_fid[c.fids[proposal["cores"][c.index]]]
        views = view_helper._front_views(rec)
        depths = [_zbuf(v[3][:, :3], v[2], v[1], view_helper.image_size) for v in views]
        queries.append({
            "fid": int(rec.fid),
            "rgb": torch.stack([v[0] for v in views]),
            "K": torch.stack([v[1] for v in views]),
            "T": torch.stack([v[2] for v in views]),
            "depth": torch.stack([d for d, _ in depths]),
            "depth_mask": torch.stack([m for _, m in depths]),
        })

    scene_id = _scene_id(anchor.drive, anchor.fid, origin)
    table = []
    traversed = []
    acc = 0.0
    for t, c in enumerate(window):
        lift, geom = proposal["selections"][c.index]
        acc += float(c.arc_length)
        traversed.append(acc)
        table.append({
            "chunk_index": t + 1,
            "route_index": int(c.index),
            "fids": list(c.fids),
            "lift_fids": [c.fids[m] for m in lift],
            "geometry_fids": [c.fids[m] for m in geom],
            "core_fid": int(c.fids[proposal["cores"][c.index]]),
            "arc_start": float(c.arc_start),
            "arc_end": float(c.arc_end),
            "new_support_cells": int(new_counts[t]),
        })
    blob = {
        "scene_id": scene_id,
        "split": split,
        "drive": anchor.drive,
        "anchor_fid": int(anchor.fid),
        "origin_xy": torch.tensor(origin, dtype=torch.float32),
        "sat_center_xy": torch.tensor(sat_center, dtype=torch.float32),
        "z_datum_m": torch.tensor([z_datum], dtype=torch.float32),
        "z_datum_policy": Z_DATUM_POLICY,
        "tile_size_m": float(tile_size_m),
        "resolution_m": float(resolution_m),
        "chunking_version": CHUNKING_VERSION,
        "world_target_version": WORLD_TARGET_VERSION,
        "dynamic_filter": "none",
        "satellite_bev": satellite_bev,
        "height": torch.from_numpy(height).unsqueeze(0),
        "density": torch.from_numpy(density).unsqueeze(0),
        "world_valid": torch.from_numpy(valid).unsqueeze(0),
        "count": torch.from_numpy(packed["count"].astype(np.int32)).unsqueeze(0),
        "chunk_lidar_support": torch.from_numpy(chunk_support_np).unsqueeze(1),
        "chunk_table": table,
        "traversed_m": torch.tensor(traversed, dtype=torch.float32),
        "queries": queries,
        "world_target_hash": _hash_targets(height, density, valid),
        "satellite_identity": {
            "path": str(anchor.sat_path),
            "anchor_fid": int(anchor.fid),
            "sat_m_per_px": SAT_M_PER_PX,
        },
    }
    return blob


class WorldStateSceneDataset(Dataset):
    """Loads prebuilt scene blobs.  ``__getitem__`` returns (ModelInputs, SupervisionBundle)."""

    def __init__(self, root: str, split: Optional[str] = None):
        self.root = Path(root)
        manifest_path = self.root / "scenes.jsonl"
        rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
        if split is not None:
            rows = [r for r in rows if r.get("split") == split]
        self.rows = rows
        if not self.rows:
            raise RuntimeError(f"no world-state scenes in {manifest_path} split={split!r}")
        first = torch.load(self.root / self.rows[0]["file"], map_location="cpu", weights_only=False)
        self.tile_size_m = float(first["tile_size_m"])
        self.resolution_m = float(first["resolution_m"])
        self.bev_size = int(round(self.tile_size_m / self.resolution_m))
        self.chunking_version = str(first["chunking_version"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[ModelInputs, SupervisionBundle, dict]:
        blob = torch.load(self.root / self.rows[idx]["file"], map_location="cpu", weights_only=False)
        sat = blob["satellite_bev"]
        if sat.ndim == 3:
            sat = sat.unsqueeze(0)
        origin = blob["origin_xy"].view(1, 2)
        datum = blob["z_datum_m"].view(1, 1)
        inputs = ModelInputs(
            satellite_bev=sat,
            origin_xy=origin,
            z_datum_m=datum,
            scene_id=(blob["scene_id"],),
            tile_size_m=float(blob["tile_size_m"]),
            resolution_m=float(blob["resolution_m"]),
            chunking_version=str(blob["chunking_version"]),
        )
        assert_no_supervision_leak(inputs.__dict__, context="ModelInputs")
        queries = blob["queries"]
        support = blob["chunk_lidar_support"].bool()
        if support.ndim == 3:
            support = support.unsqueeze(1)
        if support.ndim == 4:
            support = support.unsqueeze(0)
        final = support.any(dim=1)
        height = blob["height"].float()
        if height.ndim == 3:
            height = height.unsqueeze(0)
        density = blob["density"].float()
        if density.ndim == 3:
            density = density.unsqueeze(0)
        valid = blob["world_valid"].bool()
        if valid.ndim == 3:
            valid = valid.unsqueeze(0)
        supervision = SupervisionBundle(
            height=height,
            density=density,
            world_valid=valid,
            chunk_lidar_support=support,
            future_route_support=final,
            final_support=final,
            query_rgb=torch.stack([q["rgb"] for q in queries]).unsqueeze(0),
            query_K=torch.stack([q["K"] for q in queries]).unsqueeze(0),
            query_T_world_cam=torch.stack([q["T"] for q in queries]).unsqueeze(0),
            query_depth=torch.stack([q["depth"] for q in queries]).unsqueeze(0),
            query_depth_mask=torch.stack([q["depth_mask"] for q in queries]).unsqueeze(0),
            traversed_m=blob["traversed_m"].view(1, -1),
        )
        return inputs, supervision, blob


def collate_world_state(batch):
    """Batch size 1 in v1; keep the triple intact."""
    if len(batch) != 1:
        raise ValueError("world-state v1 uses batch_size=1")
    return batch[0]


def spec_from_inputs(inputs: ModelInputs, device=None) -> SceneTileSpec:
    origin = inputs.origin_xy if inputs.origin_xy.ndim == 2 else inputs.origin_xy.view(1, 2)
    datum = inputs.z_datum_m if inputs.z_datum_m.ndim >= 1 else inputs.z_datum_m.view(1)
    if origin.ndim == 1:
        origin = origin.view(1, 2)
    if datum.ndim == 1:
        datum = datum.view(-1, 1) if datum.numel() > 1 else datum.view(1, 1)
    if device is not None:
        origin = origin.to(device)
        datum = datum.to(device)
    return SceneTileSpec(
        scene_id=inputs.scene_id[0] if inputs.scene_id else "unknown",
        origin_xy=origin,
        tile_size_m=float(inputs.tile_size_m),
        resolution_m=float(inputs.resolution_m),
        z_datum_m=datum,
        chunking_version=inputs.chunking_version,
    )
