"""Small tile sampler for the unified BEV latent experiments.

The sampler intentionally uses one drive and one target-centered 64m tile in
the first implementation.  Satellite crops are already centered at the
vehicle position, so the target frame's crop is the canonical tile prior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .fisheye import FisheyeVirtualRig
from .geometry import project_points_to_image, se3_inverse, sparse_depth_zbuffer, transform_points


SAT_M_PER_PX = 0.196
FRONT_CROP_WIDTH = 560
FRONT_CROP_OVERLAP = 72
VIEW_LAYOUT_VERSION = "front2_left3_right3_v1"
VIEW_CAMERA_IDS = (0, 0, 1, 1, 1, 2, 2, 2)
TARGET_VIEW_LAYOUT_VERSION = "front2_v1"


def centered_two_crop_starts(
    image_width: int,
    crop_width: int,
    overlap: int,
    center_x: float,
) -> List[int]:
    """Two overlapping front-camera crops centered on the calibrated axis."""
    if crop_width <= 0 or overlap < 0 or overlap >= crop_width:
        raise ValueError("require crop_width > 0 and 0 <= overlap < crop_width")
    step = crop_width - overlap
    span = crop_width + step
    if span > image_width:
        raise ValueError("two crops do not fit inside the image")
    left = int(round(float(center_x) - span / 2.0))
    left = max(0, min(left, image_width - span))
    return [left, left + step]


def scaled_crop_intrinsics(
    K0: np.ndarray,
    raw_size: Tuple[int, int],
    crop_start_x: int,
    crop_width: int,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Intrinsics for a horizontal image window resized to ``image_size``."""
    raw_width, raw_height = raw_size
    if crop_start_x < 0 or crop_start_x + crop_width > raw_width:
        raise ValueError("front crop lies outside the source image")
    out_width, out_height = image_size
    K = np.asarray(K0, dtype=np.float64).copy()
    K[0, 2] -= float(crop_start_x)
    K[0] *= out_width / float(crop_width)
    K[1] *= out_height / float(raw_height)
    return torch.from_numpy(K.astype(np.float32))


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
        self.front_crop_width = FRONT_CROP_WIDTH
        self.front_crop_overlap = FRONT_CROP_OVERLAP
        self.view_layout_version = VIEW_LAYOUT_VERSION
        self.target_view_layout_version = TARGET_VIEW_LAYOUT_VERSION
        self.target_views = 2
        self.view_camera_ids = (0, 0) + (
            tuple([1] * len(self.fisheye_yaws_deg) + [2] * len(self.fisheye_yaws_deg))
            if self.use_fisheye else ()
        )
        self._rigs: Dict[str, FisheyeVirtualRig] = {}
        self._records_by_drive = self._load_records(drive)
        self.views_per_frame = 2 + (
            len(self.fisheye_yaws_deg) * 2 if self.use_fisheye else 0
        )
        if self.views_per_frame != len(self.view_camera_ids):
            raise ValueError("the fixed front2/left3/right3 layout requires three fisheye yaws")
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
            front_crop_starts = centered_two_crop_starts(
                raw_size[0], self.front_crop_width, self.front_crop_overlap, float(K0[0, 2]),
            )
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
                rec._front_crop_starts = front_crop_starts  # type: ignore[attr-defined]
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

    def _front_views(self, rec: FrameRecord) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Two calibrated windows from image_00, sharing the physical cam0 center."""
        points = self._load_points_world(rec)
        T_world_cam = torch.from_numpy(rec.T_world_cam.astype(np.float32))
        views: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        with _open_image(rec.rgb_path) as im:
            image = im.convert("RGB")
            raw_height = rec._raw_size[1]  # type: ignore[attr-defined]
            for start_x in rec._front_crop_starts:  # type: ignore[attr-defined]
                crop = image.crop((start_x, 0, start_x + self.front_crop_width, raw_height))
                crop = crop.resize(self.image_size, Image.Resampling.BILINEAR)
                rgb = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
                K = scaled_crop_intrinsics(
                    rec._K0, rec._raw_size, start_x, self.front_crop_width, self.image_size,  # type: ignore[attr-defined]
                )
                uv, _, valid = project_points_to_image(points, T_world_cam, K, self.image_size)
                views.append((rgb, K, T_world_cam, torch.cat([
                    points, uv, valid[:, None].to(points.dtype),
                ], dim=-1)))
        return views

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
        target_items = self._front_views(target)
        target_image = torch.stack([item[0] for item in target_items])
        target_K = torch.stack([item[1] for item in target_items])
        target_T = torch.stack([item[2] for item in target_items])
        target_depth_items = [
            sparse_depth_zbuffer(item[3][:, :3], item[2], item[1], self.image_size)
            for item in target_items
        ]
        target_depth = torch.stack([item[0] for item in target_depth_items])
        target_depth_mask = torch.stack([item[1] for item in target_depth_items])
        source_items: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for r in sources:
            source_items.extend(self._front_views(r))
            if self.use_fisheye:
                source_items.extend(self._virtual_views(r))
        expected_views = len(sources) * self.views_per_frame
        if len(source_items) != expected_views:
            raise RuntimeError(
                f"incomplete source rig: expected {expected_views} views, got {len(source_items)}"
            )
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
            "source_T_world_imu": torch.stack([
                torch.from_numpy(r.T_world_imu.astype(np.float32)) for r in sources
            ]),
            "source_points_world": points_world,
            "source_points_uv": points_uv,
            "source_points_valid": points_valid,
            "satellite": sat,
            "origin_xy": origin_xy,
            "meta": {"drive": target.drive, "target_fid": target.fid,
                     "source_fids": [r.fid for r in sources],
                     "views_per_frame": self.views_per_frame,
                     "view_layout_version": self.view_layout_version,
                     "target_view_layout_version": self.target_view_layout_version},
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
    if getattr(ds, "view_layout_version", None) != VIEW_LAYOUT_VERSION:
        raise RuntimeError(
            f"sample cache uses view layout {getattr(ds, 'view_layout_version', None)!r}; "
            f"rebuild it for {VIEW_LAYOUT_VERSION!r}"
        )
    if getattr(ds, "target_view_layout_version", None) != TARGET_VIEW_LAYOUT_VERSION:
        raise RuntimeError(
            "sample cache uses a legacy target RGB layout; rebuild it for "
            f"{TARGET_VIEW_LAYOUT_VERSION!r}"
        )
    return ds


def dense_geometry_subset_key(start_frame: int, frame_count: int) -> str:
    return f"s{start_frame}_n{frame_count}"


def geometry_sample_identity(meta: Mapping[str, object]) -> Dict[str, object]:
    """Canonical cache identity for one target tile and its source frames."""
    source_fids = meta.get("source_fids")
    if torch.is_tensor(source_fids):
        source_fids = source_fids.detach().cpu().flatten().tolist()
    if not isinstance(source_fids, (list, tuple)):
        raise RuntimeError("sample metadata has no source_fids sequence")
    return {
        "drive": str(meta["drive"]),
        "target_fid": int(meta["target_fid"]),
        "source_fids": [int(value) for value in source_fids],
        "view_layout_version": str(meta.get("view_layout_version", "legacy_unknown")),
    }


def validate_geometry_blob_identity(
    blob: Mapping[str, object],
    expected_meta: Mapping[str, object],
    *,
    context: str = "geometry cache",
) -> None:
    """Fail before an index-aligned cache can be paired with the wrong tile."""
    stored = blob.get("sample_identity")
    if not isinstance(stored, Mapping):
        raise RuntimeError(
            f"{context} has no sample_identity; rebuild the legacy geometry cache"
        )
    expected = geometry_sample_identity(expected_meta)
    actual = geometry_sample_identity(stored)
    if actual != expected:
        raise RuntimeError(
            f"{context} sample mismatch: cache={actual!r}, dataset={expected!r}"
        )


def geometry_scale_reliability(scale_source: str, pair_count: int) -> str:
    """Return an auditable label for a subset's metric-scale evidence.

    A single source frame has no temporal vehicle motion, so its only metric
    cue is the calibrated multi-camera rig.  Two source frames provide one
    vehicle-motion baseline; three or more valid baselines allow a robust
    median and dispersion diagnostic.
    """
    if scale_source == "camera_rig":
        return "single_frame_camera_rig_fallback"
    if scale_source != "vehicle_motion":
        return "unknown"
    if pair_count <= 1:
        return "single_baseline_vehicle_motion"
    if pair_count < 3:
        return "limited_baseline_vehicle_motion"
    return "multi_baseline_vehicle_motion"


def dense_geometry_subset_qa(
    blob: dict,
    start_frame: int,
    frame_count: int,
) -> Dict[str, float | int | str]:
    """Read subset-level VGGT scale QA without touching geometry tensors."""
    subsets = blob.get("subsets")
    if subsets is None:
        return {}
    key = dense_geometry_subset_key(start_frame, frame_count)
    if key not in subsets:
        raise KeyError(f"geometry cache is missing independently inferred subset {key}")
    entry = subsets[key]
    source = str(entry.get("scale_source", "unknown"))
    pair_count = int(entry.get("scale_pair_count", 0))

    def scalar(name: str) -> float:
        value = entry.get(name, float("nan"))
        return float(value.item()) if torch.is_tensor(value) else float(value)

    return {
        "metric_scale": scalar("metric_scale"),
        "scale_source": source,
        "scale_reliability": str(entry.get(
            "scale_reliability",
            geometry_scale_reliability(source, pair_count),
        )),
        "scale_pair_count": pair_count,
        "scale_relative_mad": scalar("scale_relative_mad"),
        "pose_alignment_rmse_m": scalar("pose_alignment_rmse_m"),
    }


def dense_geometry_from_blob(
    blob: dict,
    start_frame: int,
    frame_count: int,
    views_per_frame: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read geometry inferred for exactly one source subset.

    Joint-view caches must contain an explicit subset entry; slicing a larger
    joint prediction would leak held-out views through the geometry backbone.
    Legacy per-view Metric3D caches remain sliceable because their images were
    inferred independently.
    """
    key = dense_geometry_subset_key(start_frame, frame_count)
    subsets = blob.get("subsets")
    if subsets is not None:
        if key not in subsets:
            raise KeyError(f"geometry cache is missing independently inferred subset {key}")
        entry = subsets[key]
        return entry["depth"].float(), entry["conf"].float()
    start = start_frame * views_per_frame
    stop = (start_frame + frame_count) * views_per_frame
    return blob["depth"][start:stop].float(), blob["conf"][start:stop].float()


def dense_geometry_from_batch(
    batch: dict,
    start_frame: int,
    frame_count: int,
    views_per_frame: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = dense_geometry_subset_key(start_frame, frame_count)
    depth_key, conf_key = f"dense_depth_{key}", f"dense_conf_{key}"
    if depth_key in batch:
        return batch[depth_key], batch[conf_key]
    joint = batch.get("dense_joint_geometry", False)
    if torch.is_tensor(joint):
        joint = bool(joint.any())
    if joint:
        raise KeyError(f"batch is missing independently inferred subset {key}")
    start = start_frame * views_per_frame
    stop = (start_frame + frame_count) * views_per_frame
    return batch["dense_depth"][:, start:stop], batch["dense_conf"][:, start:stop]


def attach_dense_geometry(base, geometry_cache: str):
    """Join any index-stable unified-BEV dataset with geometry files by index.

    Items additionally carry ``dense_depth``/``dense_conf`` for the full
    source set. Joint-view caches also expose explicit
    ``dense_*_s{start}_n{count}`` tensors so sparse subsets cannot be sliced
    from a larger VGGT inference. ``base`` may be a raw or cached dataset;
    its ordering must match the geometry-cache build exactly.
    """
    import os
    from torch.utils.data import Dataset as _Dataset

    class _DensePair(_Dataset):
        def __init__(self, base, geometry_dir):
            self.base = base
            self.geometry_dir = geometry_dir

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            s = self.base[idx]
            blob = torch.load(os.path.join(self.geometry_dir, f"{idx:06d}.pt"),
                              map_location="cpu", weights_only=False)
            validate_geometry_blob_identity(
                blob, s["meta"], context=f"geometry cache index {idx}",
            )
            source_view_count = int(s["source_rgb"].shape[0])
            if source_view_count % self.base.views_per_frame:
                raise RuntimeError(
                    f"sample has {source_view_count} source views, which is not divisible by "
                    f"views_per_frame={self.base.views_per_frame}"
                )
            full_count = source_view_count // self.base.views_per_frame
            depth, conf = dense_geometry_from_blob(blob, 0, full_count, self.base.views_per_frame)
            s["dense_depth"], s["dense_conf"] = depth, conf
            subsets = blob.get("subsets")
            s["dense_joint_geometry"] = subsets is not None
            if subsets is not None:
                for key, entry in subsets.items():
                    s[f"dense_depth_{key}"] = entry["depth"].float()
                    s[f"dense_conf_{key}"] = entry["conf"].float()
            return s

        def __getattr__(self, name):
            return getattr(self.base, name)

    return _DensePair(base, geometry_cache)


def load_dense_cached_unified_bev(sample_cache: str, geometry_cache: str):
    """Load a sample cache and join its dense geometry cache by index."""
    return attach_dense_geometry(load_cached_unified_bev(sample_cache), geometry_cache)
