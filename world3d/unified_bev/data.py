"""Small tile sampler for the unified BEV latent experiments.

The sampler intentionally uses one drive and one target-centered 64m tile in
the first implementation.  Satellite crops are already centered at the
vehicle position, so the target frame's crop is the canonical tile prior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .fisheye import FisheyeVirtualRig
from .geometry import project_points_to_image, se3_inverse, sparse_depth_zbuffer, transform_points


SAT_M_PER_PX = 0.196


def _parse_matrix_file(path: Path, matrix_size: int) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) != matrix_size + 1:
            continue
        try:
            fid = int(p[0])
            vals = np.asarray([float(x) for x in p[1:]], dtype=np.float64)
        except ValueError:
            continue
        if matrix_size == 12:
            T = np.eye(4, dtype=np.float64)
            T[:3] = vals.reshape(3, 4)
        elif matrix_size == 16:
            T = vals.reshape(4, 4)
        else:
            raise ValueError(matrix_size)
        out[fid] = T
    return out


def _read_p_rect_00(path: Path) -> np.ndarray:
    values: Dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        try:
            values[key.strip()] = np.asarray([float(x) for x in rest.split()], dtype=np.float64)
        except ValueError:
            continue
    for key in ("P_rect_00", "K_00"):
        if key in values:
            return values[key].reshape(3, 4 if key.startswith("P_") else 3)[:, :3]
    raise KeyError(f"P_rect_00/K_00 not found in {path}")


def _read_cam_to_velo(path: Path) -> np.ndarray:
    vals = [float(x) for x in path.read_text().split()]
    T_velo_cam = np.eye(4, dtype=np.float64)
    T_velo_cam[:3] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
    return np.linalg.inv(T_velo_cam)


def _open_image(path: Path, retries: int = 3) -> Image.Image:
    """Open-and-load an image, retrying transient external-drive IO errors."""
    last: Optional[Exception] = None
    for _ in range(retries):
        try:
            im = Image.open(path)
            im.load()
            return im
        except Exception as e:  # noqa: BLE001 - transient IO errors are retried
            last = e
    raise last  # type: ignore[misc]


def _load_rgb(path: Path, image_size: Tuple[int, int]) -> torch.Tensor:
    W, H = image_size
    with _open_image(path) as im:
        im = im.convert("RGB").resize((W, H), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) / 255.0


def _scaled_K(K0: np.ndarray, raw_size: Tuple[int, int], image_size: Tuple[int, int]) -> torch.Tensor:
    W0, H0 = raw_size
    W, H = image_size
    K = K0.copy()
    K[0] *= W / float(W0)
    K[1] *= H / float(H0)
    return torch.from_numpy(K.astype(np.float32))


@dataclass
class FrameRecord:
    drive: str
    fid: int
    rgb_path: Path
    sat_path: Path
    T_world_imu: np.ndarray
    T_world_cam: np.ndarray
    lidar_path: Path
    drive_dir: Path = Path(".")


class UnifiedBEVDataset(Dataset):
    """Target-centered tiles with dense/sparse ground view sets.

    A sample is valid only if its target and all selected source frames have
    RGB, satellite, exact camera pose, and LiDAR.  This makes the first probe
    honest: no hidden fallback to a missing geometry source.
    """

    def __init__(
        self,
        manifest: str,
        *,
        lidar_root: str,
        dense_source_count: int = 8,
        sparse_source_count: int = 2,
        tile_size_m: float = 64.0,
        bev_resolution_m: float = 0.5,
        image_size: Tuple[int, int] = (160, 96),
        max_points_per_view: int = 2048,
        max_samples: Optional[int] = None,
        drive: Optional[str] = None,
        seed: int = 0,
        min_target_spacing_m: float = 0.0,
        use_fisheye: bool = True,
        fisheye_yaws_deg: Tuple[float, ...] = (-45.0, 0.0, 45.0),
        fisheye_hfov_deg: float = 90.0,
        min_source_distance_m: float = 2.0,
    ):
        self.manifest = Path(manifest)
        self.lidar_root = Path(lidar_root)
        self.dense_source_count = int(dense_source_count)
        self.sparse_source_count = min(int(sparse_source_count), self.dense_source_count)
        self.tile_size_m = float(tile_size_m)
        self.bev_resolution_m = float(bev_resolution_m)
        self.image_size = tuple(int(x) for x in image_size)
        self.bev_size = int(round(self.tile_size_m / self.bev_resolution_m))
        self.max_points_per_view = int(max_points_per_view)
        self.seed = int(seed)
        self.min_target_spacing_m = float(min_target_spacing_m)
        self.min_source_distance_m = float(min_source_distance_m)
        self.use_fisheye = bool(use_fisheye)
        self.fisheye_yaws_deg = tuple(float(y) for y in fisheye_yaws_deg)
        self._rigs: Dict[str, FisheyeVirtualRig] = {}
        self._records_by_drive = self._load_records(drive)
        self.views_per_frame = 1 + (
            len(self.fisheye_yaws_deg) * 2 if self.use_fisheye else 0
        )
        self.samples = self._build_samples(max_samples)
        if not self.samples:
            raise RuntimeError("No valid unified BEV samples found")

    def _load_records(self, drive_filter: Optional[str]) -> Dict[str, List[FrameRecord]]:
        raw: Dict[str, List[dict]] = {}
        for line in self.manifest.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                if drive_filter is None or item["drive"] == drive_filter:
                    raw.setdefault(item["drive"], []).append(item)
        output: Dict[str, List[FrameRecord]] = {}
        for drive, rows in raw.items():
            d = Path(rows[0]["image_00_path"]).parents[2]
            poses = _parse_matrix_file(d / "poses.txt", 12)
            cam_poses = _parse_matrix_file(d / "cam0_to_world.txt", 16)
            K0 = _read_p_rect_00(d / "calibration" / "perspective.txt")
            with _open_image(Path(rows[0]["image_00_path"])) as im:
                raw_size = im.size
            T_cam_velo = _read_cam_to_velo(d / "calibration" / "calib_cam_to_velo.txt")
            if self.use_fisheye:
                self._rigs[drive] = FisheyeVirtualRig(
                    d / "calibration", self.image_size,
                    yaws_deg=self.fisheye_yaws_deg,
                )
            records: List[FrameRecord] = []
            for row in rows:
                fid = int(row["frame_index"])
                lidar = self.lidar_root / drive / "velodyne_points" / "data" / f"{fid:010d}.bin"
                T_imu = poses.get(fid)
                T_cam = cam_poses.get(fid)
                if T_imu is None or T_cam is None or not lidar.exists():
                    continue
                rgb = Path(row["image_00_path"])
                sat = Path(row["satellite_path"])
                if not rgb.exists() or not sat.exists():
                    continue
                # Store the camera pose and attach T_world_velo through the
                # camera calibration when points are loaded.
                rec = FrameRecord(drive, fid, rgb, sat, T_imu, T_cam, lidar, drive_dir=d)
                rec._K0 = K0  # type: ignore[attr-defined]
                rec._raw_size = raw_size  # type: ignore[attr-defined]
                rec._T_cam_velo = T_cam_velo  # type: ignore[attr-defined]
                records.append(rec)
            records.sort(key=lambda r: r.fid)
            output[drive] = records
        return output

    def _build_samples(self, max_samples: Optional[int]) -> List[Tuple[FrameRecord, List[FrameRecord]]]:
        samples: List[Tuple[FrameRecord, List[FrameRecord]]] = []
        for records in self._records_by_drive.values():
            xy = np.asarray([[r.T_world_imu[0, 3], r.T_world_imu[1, 3]] for r in records])
            accepted: List[np.ndarray] = []
            for i, target in enumerate(records):
                # Spec 2.4: consecutive near-duplicate targets inflate trivial
                # copy baselines; enforce a minimum spacing between targets.
                if self.min_target_spacing_m > 0 and any(
                    np.linalg.norm(xy[i] - a) < self.min_target_spacing_m for a in accepted
                ):
                    continue
                accepted.append(xy[i])
                dist = np.linalg.norm(xy - xy[i : i + 1], axis=1)
                candidates = [j for j, d in enumerate(dist)
                              if j != i and self.min_source_distance_m <= d <= self.tile_size_m / 2]
                if len(candidates) < self.dense_source_count:
                    continue
                # Greedy farthest-point selection yields spatial coverage and
                # is deterministic, while still excluding the target frame.
                chosen: List[int] = [min(candidates, key=lambda j: (dist[j], j))]
                while len(chosen) < self.dense_source_count:
                    j = max(
                        (j for j in candidates if j not in chosen),
                        key=lambda j: (min(np.linalg.norm(xy[j] - xy[k]) for k in chosen), -j),
                    )
                    chosen.append(j)
                sources = [records[j] for j in chosen]
                samples.append((target, sources))
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_points_world(self, rec: FrameRecord) -> torch.Tensor:
        pts = np.fromfile(rec.lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        if len(pts) > self.max_points_per_view:
            # Deterministic spatial subsampling; retain near and far structure.
            idx = np.linspace(0, len(pts) - 1, self.max_points_per_view).astype(np.int64)
            pts = pts[idx]
        p = torch.from_numpy(pts.astype(np.float32))
        T_world_velo = torch.from_numpy(rec.T_world_cam.astype(np.float32)) @ torch.from_numpy(
            rec._T_cam_velo.astype(np.float32)  # type: ignore[attr-defined]
        )
        return transform_points(T_world_velo, p)

    def _view(self, rec: FrameRecord) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = _load_rgb(rec.rgb_path, self.image_size)
        K = _scaled_K(rec._K0, rec._raw_size, self.image_size)  # type: ignore[attr-defined]
        points = self._load_points_world(rec)
        uv, _, valid = project_points_to_image(
            points, torch.from_numpy(rec.T_world_cam.astype(np.float32)), K, self.image_size
        )
        return image, K, torch.from_numpy(rec.T_world_cam.astype(np.float32)), torch.cat(
            [points, uv, valid[:, None].to(points.dtype)], dim=-1
        )

    def _virtual_views(self, rec: FrameRecord) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Warp both fisheyes into tangent perspective crops for one frame."""
        rig = self._rigs[rec.drive]
        points = self._load_points_world(rec)
        views: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        W, H = self.image_size
        for cam in rig.cameras:
            path = rig.fisheye_path(rec.drive_dir, cam, rec.fid)
            if not path.exists():
                continue
            with _open_image(path) as im:
                fish = np.asarray(im.convert("RGB"), dtype=np.uint8)
            fish = cv2.cvtColor(fish, cv2.COLOR_RGB2BGR)
            for yaw in rig.yaws:
                warped = rig.warp(cam, yaw, fish)
                rgb = torch.from_numpy(
                    cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32)
                ).permute(2, 0, 1) / 255.0
                valid_img = torch.from_numpy(rig.valid_mask(cam, yaw, fish))
                mask_flat = valid_img.reshape(-1)
                T_wv = torch.from_numpy(rig.T_world_virtual(cam, yaw, rec.T_world_cam).astype(np.float32))
                K = torch.from_numpy(rig.K.astype(np.float32))
                uv, _, valid = project_points_to_image(points, T_wv, K, self.image_size)
                inb = (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
                idx = (uv[..., 1].long().clamp(0, H - 1) * W + uv[..., 0].long().clamp(0, W - 1))
                look = torch.zeros_like(valid)
                look[inb] = mask_flat[idx[inb]]
                valid = valid & inb & look
                views.append((rgb, K, T_wv, torch.cat([points, uv, valid[:, None].to(points.dtype)], dim=-1)))
        return views

    def __getitem__(self, idx: int) -> dict:
        target, sources = self.samples[idx]
        target_image, target_K, target_T, target_pts = self._view(target)
        target_points = target_pts[:, :3]
        target_depth, target_depth_mask = sparse_depth_zbuffer(
            target_points, target_T, target_K, self.image_size
        )
        source_items: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for r in sources:
            source_items.append(self._view(r))
            if self.use_fisheye:
                source_items.extend(self._virtual_views(r))
        max_p = max(x[3].shape[0] for x in source_items)
        n = len(source_items)
        source_images = torch.stack([x[0] for x in source_items])
        source_K = torch.stack([x[1] for x in source_items])
        source_T = torch.stack([x[2] for x in source_items])
        points_world = torch.zeros((n, max_p, 3), dtype=torch.float32)
        points_uv = torch.zeros((n, max_p, 2), dtype=torch.float32)
        points_valid = torch.zeros((n, max_p), dtype=torch.bool)
        for j, item in enumerate(source_items):
            p = item[3]
            m = p.shape[0]
            points_world[j, :m] = p[:, :3]
            points_uv[j, :m] = p[:, 3:5]
            points_valid[j, :m] = p[:, 5] > 0.5
        with _open_image(target.sat_path) as im:
            sat = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32)).permute(2, 0, 1) / 255.0
        origin_xy = torch.tensor(
            [target.T_world_imu[0, 3] - self.tile_size_m / 2,
             target.T_world_imu[1, 3] - self.tile_size_m / 2], dtype=torch.float32
        )
        return {
            "target_rgb": target_image,
            "target_K": target_K,
            "target_T_world_cam": target_T,
            "target_depth": target_depth,
            "target_depth_mask": target_depth_mask,
            "source_rgb": source_images,
            "source_K": source_K,
            "source_T_world_cam": source_T,
            "source_points_world": points_world,
            "source_points_uv": points_uv,
            "source_points_valid": points_valid,
            "satellite": sat,
            "origin_xy": origin_xy,
            "meta": {"drive": target.drive, "target_fid": target.fid,
                     "source_fids": [r.fid for r in sources],
                     "views_per_frame": self.views_per_frame},
        }


_CACHE_IMAGE_KEYS = ("target_rgb", "source_rgb", "satellite")


def _sample_to_storage(sample: dict) -> dict:
    """Pack a dataset sample for disk caching; images go to uint8 losslessly
    (they originate from uint8 pixels divided by 255)."""
    out = dict(sample)
    for key in _CACHE_IMAGE_KEYS:
        t = sample[key]
        if t.dtype == torch.uint8:
            continue
        out[key] = (t * 255.0).round().clamp(0, 255).to(torch.uint8)
    return out


def _storage_to_sample(storage: dict) -> dict:
    out = dict(storage)
    for key in _CACHE_IMAGE_KEYS:
        out[key] = storage[key].to(torch.float32) / 255.0
    return out


def load_cached_unified_bev(path):
    """Load a prebuilt sample cache as a Dataset with the original attributes.

    ``path`` may be a single .pt file (fully materialised in RAM; watch the
    transient double-memory spike while unpickling) or a directory of
    per-sample ``NNNNNN.pt`` files plus ``attrs.pt`` (lazy per-index load,
    no RAM spike; the OS page cache absorbs reuse).
    """
    import os
    from torch.utils.data import Dataset as _Dataset

    class _CachedUnifiedBEV(_Dataset):
        def __init__(self, attrs):
            for key, value in attrs.items():
                setattr(self, key, value)

        def __len__(self):
            return self._n

        def __getitem__(self, idx):
            return _storage_to_sample(torch.load(self._file(idx), map_location="cpu", weights_only=False))

    ds = _CachedUnifiedBEV.__new__(_CachedUnifiedBEV)
    if os.path.isdir(path):
        n = len([f for f in os.listdir(path) if f.endswith(".pt") and f != "attrs.pt"])
        _CachedUnifiedBEV.__init__(ds, torch.load(os.path.join(path, "attrs.pt"), weights_only=False))
        ds._n = n
        ds._file = lambda i: os.path.join(path, f"{i:06d}.pt")
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        samples = blob["samples"]
        _CachedUnifiedBEV.__init__(ds, blob["attrs"])
        ds._n = len(samples)
        ds._file = None
        ds.__class__ = type("_RamCached", (_CachedUnifiedBEV,), {
            "__getitem__": lambda self, i: _storage_to_sample(samples[i]),
        })
    return ds
