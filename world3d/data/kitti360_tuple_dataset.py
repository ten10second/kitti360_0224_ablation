"""KITTI-360 tuple dataset for the ICASSP27 framework (front view only).

One sample = one (window, sources, target) tuple:
  - window: ~60 m route segment; ALL tuples of a window share ONE satellite
    crop taken at the window-center frame (anti crop-shift-leak, doc §1).
  - sources: K in {1..3} front-view frames, anchors spaced >= anchor_spacing m
    along the drive arc (doc §4 tuple sampler).
  - target: a frame 2..20 m ahead of the last source (metric, not frame index),
    dyaw(source->target) <= dyaw_max_deg, extrapolation by construction.
  - eval mode is deterministic: fixed anchor stride, K from cfg, targets at
    bin mid distances {[2,5):3.5, [5,10):7.5, [10,20]:15} m by default.
    An explicit ``eval_distances`` sequence can replace those midpoints for
    inference-only dense-distance diagnostics; it never affects training.

Poses: prefer per-frame cam0_to_world.txt (cam0 = image_00 rectified);
fallback imu pose @ calib_cam_to_pose['image_00']. World frame is the shared
KITTI-360 metric frame (x east, y north, z up; satellite north-up @0.196 m/px).
Window-local frame translates world coords so the window center is the origin.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

from world3d.data.deterministic import make_rng
from world3d.io.kitti360d_dataloader import (
    _load_perspective_calib,
    _load_cam_to_pose,
    _load_indexed_poses_txt,
)
from world3d.train.pose_ar import build_pose_vec, rotmat_to_6d

SAT_M_PER_PX = 0.196  # measured by phase correlation (v2b), isotropic for KITTI-360
DEFAULT_BINS = ((2.0, 5.0), (5.0, 10.0), (10.0, 20.0))


@dataclass
class TupleSpec:
    drive: str
    window_id: int
    anchor_fid: int
    source_fids: List[int]
    target_fid: int
    dist_m: float
    dyaw_deg: float
    window_center_fid: int
    window_origin_xyz: np.ndarray  # (3,) world coords of window center (translation-only window frame)


class Kitti360TupleDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        *,
        mode: str = "train",  # "train" (stochastic) | "eval" (deterministic)
        img_size: Tuple[int, int] = (640, 256),  # (W, H)
        window_m: float = 60.0,
        anchor_spacing_m: float = 2.0,
        anchor_stride_m: float = 4.0,
        k_min: int = 1,
        k_max: int = 3,
        eval_k: Tuple[int, ...] = (1, 3),
        eval_distances: Optional[Tuple[float, ...]] = None,
        dist_min_m: float = 2.0,
        dist_max_m: float = 20.0,
        dyaw_max_deg: float = 20.0,
        seed: int = 0,
        epoch: int = 0,
        sat_m_per_px: float = SAT_M_PER_PX,
    ):
        self.manifest_path = Path(manifest_path)
        self.mode = mode
        self.img_w, self.img_h = int(img_size[0]), int(img_size[1])
        self.window_m = float(window_m)
        self.anchor_spacing_m = float(anchor_spacing_m)
        self.anchor_stride_m = float(anchor_stride_m)
        self.k_min, self.k_max = int(k_min), int(k_max)
        self.eval_k = tuple(eval_k)
        self.dist_min_m, self.dist_max_m = float(dist_min_m), float(dist_max_m)
        self.eval_distances = None if eval_distances is None else tuple(
            float(distance) for distance in eval_distances
        )
        if self.eval_distances is not None:
            if not self.eval_distances:
                raise ValueError("eval_distances must be non-empty when provided")
            if len(set(self.eval_distances)) != len(self.eval_distances):
                raise ValueError("eval_distances must not contain duplicates")
            if any(not self.dist_min_m <= distance <= self.dist_max_m for distance in self.eval_distances):
                raise ValueError(
                    f"eval_distances must lie in [{self.dist_min_m}, {self.dist_max_m}] m"
                )
        self.dyaw_max_deg = float(dyaw_max_deg)
        self.seed = int(seed)
        self.epoch = 0
        self.sat_m_per_px = float(sat_m_per_px)
        self.bins = DEFAULT_BINS

        self._build_index()
        self._build_tuples()

    # ------------------------------------------------------------------ index
    def _build_index(self):
        by_drive: Dict[str, List[dict]] = {}
        for line in self.manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_drive.setdefault(r["drive"], []).append(r)

        self.frames: Dict[str, Dict[int, dict]] = {}
        self._imu_poses: Dict[str, Dict[int, np.ndarray]] = {}
        self._cam0_poses: Dict[str, Dict[int, np.ndarray]] = {}
        self._cam_to_pose: Dict[str, Dict[str, np.ndarray]] = {}
        self._K0: Dict[str, Optional[np.ndarray]] = {}

        for drive, recs in by_drive.items():
            drive_dir = Path(recs[0]["image_00_path"]).parent.parent.parent  # .../<drive>/image_00/data_rect/f.png
            calib_dir = drive_dir / "calibration"
            self._imu_poses[drive] = _load_indexed_poses_txt(drive_dir / "poses.txt", mat_size=12)
            self._cam0_poses[drive] = _load_indexed_poses_txt(drive_dir / "cam0_to_world.txt", mat_size=16)
            c2p_path = calib_dir / "calib_cam_to_pose.txt"
            self._cam_to_pose[drive] = _load_cam_to_pose(c2p_path) if c2p_path.exists() else {}
            K0 = None
            persp = calib_dir / "perspective.txt"
            if persp.exists():
                calib = _load_perspective_calib(persp)
                if "P_rect_00" in calib:
                    K0 = calib["P_rect_00"][:, :3].astype(np.float64)
                elif "K_00" in calib:
                    K0 = calib["K_00"].astype(np.float64)
            self._K0[drive] = K0

            fr: Dict[int, dict] = {}
            for r in recs:
                fr[int(r["frame_index"])] = r
            self.frames[drive] = fr

    def _T_cam_to_world(self, drive: str, fid: int) -> Optional[np.ndarray]:
        c0 = self._cam0_poses[drive].get(fid)
        if c0 is not None:
            return c0
        T_imu = self._imu_poses[drive].get(fid)
        T_pose_cam = self._cam_to_pose[drive].get("image_00")
        if T_imu is not None and T_pose_cam is not None:
            return T_imu @ T_pose_cam
        return None

    # ---------------------------------------------------------------- tuples
    def _build_tuples(self):
        self.tuples: List[TupleSpec] = []
        n_reject_yaw = 0
        n_reject_pose = 0
        for drive, fr in self.frames.items():
            fids = sorted(fr)
            if len(fids) < 4:
                continue
            # arc length from imu positions
            T_imu = self._imu_poses[drive]
            xs = np.array([[T_imu[f][0, 3], T_imu[f][1, 3]] if f in T_imu else [np.nan, np.nan] for f in fids])
            arc = np.zeros(len(fids))
            acc = 0.0
            for i in range(1, len(fids)):
                if not np.isnan(xs[i]).any() and not np.isnan(xs[i - 1]).any():
                    acc += float(np.hypot(*(xs[i] - xs[i - 1])))
                arc[i] = acc

            def yaw_at(fid: int) -> Optional[float]:
                T = T_imu.get(fid)
                if T is None:
                    return None
                return math.atan2(T[1, 0], T[0, 0])

            def frame_at_arc(a: float) -> int:
                idx = np.searchsorted(arc, a, side="left")
                return fids[idx] if idx < len(fids) else -1

            total = arc[-1]
            # windows on a fixed 60 m grid
            for w_start in np.arange(0, max(total - self.window_m, 1.0), self.window_m * 0.5):
                w_end = w_start + self.window_m
                center_fid = frame_at_arc(w_start + self.window_m / 2)
                if center_fid < 0:
                    continue
                Tc = T_imu.get(center_fid)
                if Tc is None:
                    continue
                origin_xyz = np.array([Tc[0, 3], Tc[1, 3], Tc[2, 3]])
                # deterministic anchors within [w_start, w_end - (K-1)*spacing - dist_max]
                a = w_start
                while a <= w_end - (self.k_max - 1) * self.anchor_spacing_m - self.dist_max_m - 1e-6:
                    anchor_fid = frame_at_arc(a)
                    if anchor_fid < 0:
                        break
                    yaw_a = yaw_at(anchor_fid)
                    if yaw_a is None:
                        a += self.anchor_stride_m
                        continue
                    if self.mode == "eval" and self.eval_distances is not None:
                        distances = self.eval_distances
                    else:
                        distances = tuple((lo + hi) / 2.0 for lo, hi in self.bins)
                    for dist in distances:
                        target_fid = frame_at_arc(a + (self.k_min - 1) * self.anchor_spacing_m + dist)
                        if target_fid < 0:
                            continue
                        yaw_t = yaw_at(target_fid)
                        if yaw_t is None:
                            n_reject_pose += 1
                            continue
                        dyaw = math.degrees(abs((yaw_t - yaw_a + math.pi) % (2 * math.pi) - math.pi))
                        if dyaw > self.dyaw_max_deg:
                            n_reject_yaw += 1
                            continue
                        self.tuples.append(TupleSpec(
                            drive=drive,
                            window_id=int(w_start / (self.window_m * 0.5)),
                            anchor_fid=anchor_fid,
                            source_fids=[],
                            target_fid=target_fid,
                            dist_m=dist,
                            dyaw_deg=dyaw,
                            window_center_fid=center_fid,
                            window_origin_xyz=origin_xyz,
                        ))
                    a += self.anchor_stride_m
        self.n_reject_yaw = n_reject_yaw
        self.n_reject_pose = n_reject_pose

    def __len__(self):
        return len(self.tuples) * (1 if self.mode == "train" else len(self.eval_k))

    # ------------------------------------------------------------------ io
    def _read_rgb(self, path: Path, drive: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        K0 = self._K0.get(drive)
        img_r = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        K = None
        if K0 is not None:
            h0, w0 = img.shape[:2]
            sx, sy = self.img_w / float(w0), self.img_h / float(h0)
            K = K0.copy()
            K[0, 0] *= sx; K[1, 1] *= sy
            K[0, 2] *= sx; K[1, 2] *= sy
        rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1), K  # (3,H,W)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == "train":
            rng = make_rng(self.seed, self.epoch, idx, salt=7)
            spec = self.tuples[idx]
            # stochastic choices: K and target distance (uniform in [2,20] m)
            K = rng.randint(self.k_min, self.k_max)
            dist = rng.uniform(self.dist_min_m, self.dist_max_m)
        else:
            k_slot = idx % len(self.eval_k)
            K = self.eval_k[k_slot]
            spec = self.tuples[idx // len(self.eval_k)]
            dist = spec.dist_m

        fr = self.frames[spec.drive]
        T_imu = self._imu_poses[spec.drive]
        fids = sorted(fr)
        ai = fids.index(spec.anchor_fid)
        # sources: anchor + (K-1) later frames, each >= anchor_spacing_m apart (chord approx)
        source_fids = [fids[ai]]
        for k in range(1, K):
            prev = source_fids[-1]
            j = fids.index(prev) + 1
            while j < len(fids):
                if prev in T_imu and fids[j] in T_imu:
                    d = float(np.hypot(*(T_imu[fids[j]][:2, 3] - T_imu[prev][:2, 3])))
                    if d >= self.anchor_spacing_m - 1e-3:
                        break
                j += 1
            if j >= len(fids):
                j = len(fids) - 1
            source_fids.append(fids[j])
        # target: first frame >= dist m beyond the LAST source (same rule for both modes)
        last = source_fids[-1]
        j = fids.index(last) + 1
        target_fid = fids[min(fids.index(last) + 1, len(fids) - 1)]
        while j < len(fids):
            if last in T_imu and fids[j] in T_imu:
                d = float(np.hypot(*(T_imu[fids[j]][:2, 3] - T_imu[last][:2, 3])))
                if d >= dist:
                    target_fid = fids[j]
                    break
            j += 1

        # images & poses
        tgt_rec = fr[target_fid]
        tgt_rgb, tgt_K = self._read_rgb(Path(tgt_rec["image_00_path"]), spec.drive)
        T_tgt = self._T_cam_to_world(spec.drive, target_fid)
        T_tgt_imu = T_imu.get(target_fid, np.eye(4))

        src_rgbs, src_Ks, src_Ts, rel_poses = [], [], [], []
        for sf in source_fids:
            rec = fr[sf]
            rgb, Ks = self._read_rgb(Path(rec["image_00_path"]), spec.drive)
            Ts = self._T_cam_to_world(spec.drive, sf)
            if Ts is None:
                Ts = np.eye(4)
            src_rgbs.append(torch.from_numpy(rgb))
            src_Ks.append(torch.from_numpy(Ks.astype(np.float32)))
            src_Ts.append(torch.from_numpy(Ts.astype(np.float32)))
            # Target -> source pose in the target-camera frame.  Translation,
            # rotation, target rays, and satellite patch coordinates now share
            # this reference; do not express dt in global/world axes.
            dt_world = Ts[:3, 3] - T_tgt[:3, 3]
            dt = (T_tgt[:3, :3].T @ dt_world).astype(np.float32)
            R_rel = (T_tgt[:3, :3].T @ Ts[:3, :3]).astype(np.float32)
            rel_poses.append(torch.from_numpy(np.concatenate([dt, rotmat_to_6d(torch.from_numpy(R_rel)).numpy()])))

        # Requested distance is sampled along the route; use this actual
        # camera-center ground-plane distance for reporting/evaluation bins.
        actual_source_target_dist_m = float(np.linalg.norm(src_Ts[-1][:2, 3] - T_tgt[:2, 3]))

        # window-shared satellite crop (from window center frame)
        sat_rec = fr[spec.window_center_fid]
        sat = cv2.imread(str(Path(sat_rec["satellite_path"])), cv2.IMREAD_COLOR)
        sat = cv2.cvtColor(sat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sat_t = torch.from_numpy(sat).permute(2, 0, 1)

        # window-local coords of satellite patch centers (for metric PE, computed here
        # as origin xy; the model derives per-patch world xy from origin + grid)
        origin_xyz = torch.from_numpy(spec.window_origin_xyz.astype(np.float32))

        pose_vec = build_pose_vec(
            torch.from_numpy(tgt_K.astype(np.float32)),
            torch.from_numpy(T_tgt.astype(np.float32)) if T_tgt is not None else torch.eye(4),
            torch.from_numpy(T_tgt_imu.astype(np.float32)),
            self.img_h, self.img_w, torch.device("cpu"),
        )

        bin_id = next((i for i, (lo, hi) in enumerate(self.bins) if lo <= dist < hi or (i == len(self.bins) - 1 and dist == hi)), -1)
        actual_bin_id = next(
            (i for i, (lo, hi) in enumerate(self.bins)
             if lo <= actual_source_target_dist_m < hi
             or (i == len(self.bins) - 1 and actual_source_target_dist_m == hi)),
            -1,
        )

        return {
            "tgt_rgb": torch.from_numpy(tgt_rgb),
            "tgt_K": torch.from_numpy(tgt_K.astype(np.float32)),
            "tgt_T_cam": torch.from_numpy(T_tgt.astype(np.float32)) if T_tgt is not None else torch.eye(4),
            "tgt_T_imu": torch.from_numpy(T_tgt_imu.astype(np.float32)),
            "pose_vec": pose_vec,
            "src_rgbs": torch.stack(src_rgbs),
            "src_Ks": torch.stack(src_Ks),
            "src_Ts": torch.stack(src_Ts),
            "rel_poses": torch.stack(rel_poses),
            "n_src": len(source_fids),
            "sat": sat_t,
            "window_origin_xyz": origin_xyz,
            "sat_m_per_px": self.sat_m_per_px,
            "actual_source_target_dist_m": actual_source_target_dist_m,
            "meta": {
                "drive": spec.drive,
                "window_id": spec.window_id,
                "source_fids": source_fids,
                "target_fid": target_fid,
                "dist_m": float(dist),
                "actual_source_target_dist_m": actual_source_target_dist_m,
                "bin": bin_id,
                "actual_bin": actual_bin_id,
                "dyaw_deg": float(spec.dyaw_deg),
                "split_mode": self.mode,
            },
        }


def collate_tuples(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    k_max = max(b["n_src"] for b in batch)
    def pad_stack(key, pad_val=0.0):
        out = []
        for b in batch:
            t = b[key]
            if t.shape[0] < k_max:
                pad = torch.zeros((k_max - t.shape[0], *t.shape[1:]), dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            out.append(t)
        return torch.stack(out)

    src_mask = torch.zeros(B, k_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        src_mask[i, : b["n_src"]] = True

    return {
        "tgt_rgb": torch.stack([b["tgt_rgb"] for b in batch]),
        "tgt_K": torch.stack([b["tgt_K"] for b in batch]),
        "tgt_T_cam": torch.stack([b["tgt_T_cam"] for b in batch]),
        "tgt_T_imu": torch.stack([b["tgt_T_imu"] for b in batch]),
        "pose_vec": torch.stack([b["pose_vec"] for b in batch]),
        "src_rgbs": pad_stack("src_rgbs"),
        "src_Ts": pad_stack("src_Ts"),
        "rel_poses": pad_stack("rel_poses"),
        "src_mask": src_mask,
        "n_src": torch.tensor([b["n_src"] for b in batch], dtype=torch.long),
        "sat": torch.stack([b["sat"] for b in batch]),
        "window_origin_xyz": torch.stack([b["window_origin_xyz"] for b in batch]),
        "actual_source_target_dist_m": torch.tensor(
            [b["actual_source_target_dist_m"] for b in batch], dtype=torch.float32
        ),
        "meta": [b["meta"] for b in batch],
    }
