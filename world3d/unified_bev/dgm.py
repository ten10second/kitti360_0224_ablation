"""DGM1 (BW official 1 m bare-earth terrain) anchoring support.

The world->UTM affine, bilinear tile sampling, and the agreement magnitudes
were validated by scripts/qa_dgm_alignment.py (world->UTM fit RMSE 0.03-0.08 m)
and scripts/qa_dgm_scene_check.py (road-cell MAD vs our static targets:
median 3.5 cm, worst 15 cm, after one constant per-scene offset).  Vertical
datum: DGM1 is DHHN2016 absolute; KITTI-360 world z is per-drive anchored, so
every consumer must estimate one constant offset from its own trusted
(near-field) data — never assume 0.
"""
from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from pyproj import Transformer

UTM32 = "EPSG:25832"
_XYZ_PATTERN = re.compile(r"_32_(\d+)_(\d+)_1_bw_2022\.xyz$")


class DgmTileSet:
    """1 km x 1 km cell-center z rasters keyed by (east_km, north_km)."""

    def __init__(self, rasters: Dict[Tuple[int, int], np.ndarray]):
        self.rasters = rasters

    @classmethod
    def from_dir(cls, root: Path) -> "DgmTileSet":
        rasters: Dict[Tuple[int, int], np.ndarray] = {}
        for archive in sorted(Path(root).glob("dgm*.zip")):
            with zipfile.ZipFile(archive) as handle:
                for name in handle.namelist():
                    match = _XYZ_PATTERN.search(name)
                    if not match:
                        continue
                    tokens = np.array(handle.read(name).split(), dtype=np.float32)
                    if tokens.size % 3:
                        raise ValueError(f"{name}: token count not divisible by 3")
                    z = tokens.reshape(-1, 3)[:, 2]
                    if z.size != 1_000_000:
                        raise ValueError(f"{name}: expected 1M samples, got {z.size}")
                    rasters[(int(match.group(1)), int(match.group(2)))] = z.reshape(1000, 1000)
        if not rasters:
            raise FileNotFoundError(f"no DGM rasters under {root}")
        return cls(rasters)

    def sample(self, utm_xy: np.ndarray) -> np.ndarray:
        """Bilinear cell-center sampling; NaN outside covered tiles."""
        out = np.full(utm_xy.shape[0], np.nan, dtype=np.float64)
        east = np.floor(utm_xy[:, 0] / 1000).astype(np.int64)
        north = np.floor(utm_xy[:, 1] / 1000).astype(np.int64)
        for key, raster in self.rasters.items():
            mask = (east == key[0]) & (north == key[1])
            if not np.any(mask):
                continue
            x = np.clip(utm_xy[mask, 0] - key[0] * 1000 - 0.5, 0, 999)
            y = np.clip(key[1] * 1000 + 999.5 - utm_xy[mask, 1], 0, 999)
            c0, r0 = np.floor(x).astype(int), np.floor(y).astype(int)
            c1, r1 = np.minimum(c0 + 1, 999), np.minimum(r0 + 1, 999)
            fx, fy = x - c0, y - r0
            top = raster[r0, c0] * (1 - fx) + raster[r0, c1] * fx
            bottom = raster[r1, c0] * (1 - fx) + raster[r1, c1] * fx
            out[mask] = top * (1 - fy) + bottom * fy
        return out


def fit_world_to_utm(poses_world_xy: np.ndarray, utm_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares 2-D affine from KITTI-360 world xy to UTM32.

    Returns (transform [2,2], translation [2], fit rmse).  Validated at
    RMSE 0.03-0.08 m on scene-length frame windows.
    """
    if len(poses_world_xy) < 10:
        raise RuntimeError(f"only {len(poses_world_xy)} pose/UTM correspondences")
    sm = poses_world_xy.mean(0)
    tm = utm_xy.mean(0)
    transform, _, _, _ = np.linalg.lstsq(poses_world_xy - sm, utm_xy - tm, rcond=None)
    resid = np.linalg.norm((poses_world_xy - sm) @ transform - (utm_xy - tm), axis=1)
    return transform, tm - sm @ transform, float(np.sqrt(np.mean(resid**2)))


def load_drive_pose_utm(sequence_root: Path, drive: str):
    """cam0_to_world xy poses + per-frame oxts UTM for one drive.

    ``drive`` is the sequence directory name (e.g. ``2013_05_28_drive_0003_sync``).
    """
    sequence = Path(sequence_root) / drive
    poses: Dict[int, np.ndarray] = {}
    for line in (sequence / "cam0_to_world.txt").read_text().splitlines():
        v = np.fromstring(line, sep=" ")
        if v.size == 17:
            poses[int(v[0])] = v[1:].reshape(4, 4)
    transformer = Transformer.from_crs("EPSG:4326", UTM32, always_xy=True)
    utm: Dict[int, np.ndarray] = {}
    for p in (sequence / "oxts" / "data").glob("*.txt"):
        v = p.read_text().split()
        if len(v) >= 2:
            utm[int(p.stem)] = np.asarray(transformer.transform(float(v[1]), float(v[0])))
    return poses, utm


class DgmAnchor:
    """Per-scene world->DGM bridge: affine fit on the scene's own frames."""

    def __init__(self, tile_set: DgmTileSet, transform: np.ndarray, translation: np.ndarray,
                 fit_rmse_m: float, scene_id: str):
        self.tile_set = tile_set
        self.transform = transform
        self.translation = translation
        self.fit_rmse_m = fit_rmse_m
        self.scene_id = scene_id

    @classmethod
    def from_blob(cls, blob: dict, tile_set: DgmTileSet, sequence_root: Path) -> "DgmAnchor":
        scene_id = str(blob["scene_id"])
        drive = scene_id.split("__")[0]
        poses, utm = load_drive_pose_utm(sequence_root, drive)
        frames = [int(f) for c in blob["chunk_table"] for f in c["geometry_fids"]]
        common = [f for f in frames if f in poses and f in utm]
        src = np.asarray([poses[f][:2, 3] for f in common], dtype=np.float64)
        dst = np.asarray([utm[f] for f in common], dtype=np.float64)
        transform, translation, rmse = fit_world_to_utm(src, dst)
        return cls(tile_set, transform, translation, rmse, scene_id)

    def to_utm(self, world_xy: np.ndarray) -> np.ndarray:
        return world_xy @ self.transform + self.translation

    def sample_tile(self, origin_xy, bev_hw: int, resolution_m: float) -> Tuple[np.ndarray, np.ndarray]:
        """Absolute DGM heights on the BEV cell-center grid: (z [H,W] float32, valid)."""
        h = w = int(bev_hw)
        xs = float(origin_xy[0]) + (np.arange(w) + 0.5) * resolution_m
        ys = float(origin_xy[1]) + (np.arange(h) + 0.5) * resolution_m
        gx, gy = np.meshgrid(xs, ys)
        utm_xy = self.to_utm(np.stack([gx.ravel(), gy.ravel()], axis=1))
        z = self.tile_set.sample(utm_xy).reshape(h, w).astype(np.float32)
        return z, np.isfinite(z)


def anchor_tile_tensor(anchor: DgmAnchor, origin_xy, bev_hw: int, resolution_m: float,
                       device) -> Tuple[torch.Tensor, torch.Tensor]:
    """(z_abs [1,1,H,W] float32 with NaN outside coverage, valid [1,1,H,W] bool)."""
    z, valid = anchor.sample_tile(origin_xy, bev_hw, resolution_m)
    z_t = torch.from_numpy(z).view(1, 1, *z.shape).to(device)
    v_t = torch.from_numpy(valid).view(1, 1, *valid.shape).to(device)
    return z_t, v_t
