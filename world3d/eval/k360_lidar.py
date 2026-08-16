"""KITTI-360 LiDAR evaluation tools (Gate A instrument).

Provides:
  - velodyne -> rectified cam0 (image_00) projection
    NOTE: calibration/calib_cam_to_velo.txt is cam->velo direction; invert it
    for projection (verified: coverage ~50-53%, depth-row monotonicity 3-4x).
  - sparse ground-truth depth maps at any frame / image scale
  - static/dynamic point splitting by LiDAR temporal consistency:
    a point visible at t is STATIC iff some scan point at t-1 or t+1 lands
    within `th` meters of it in the world frame (exact ego poses from
    cam0_to_world.txt). Moving vehicles fail this test; parked cars pass.
  - scale-aligned AbsRel for relative-depth predictions vs sparse LiDAR GT.

LiDAR stays evaluation-only: nothing here is imported by the training stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from world3d.io.kitti360d_dataloader import _load_indexed_poses_txt, _load_perspective_calib


class K360Lidar:
    def __init__(self, drive_dir: str, lidar_dir: Optional[str] = None):
        self.drive = Path(drive_dir)
        self.lidar = Path(lidar_dir) if lidar_dir else self.drive / "velodyne_points" / "data"
        calib = self.drive / "calibration"
        vals = [float(x) for x in (calib / "calib_cam_to_velo.txt").read_text().split()]
        T_velo_from_cam = np.eye(4)
        T_velo_from_cam[:3, :] = np.array(vals).reshape(3, 4)
        self.T_cam_from_velo = np.linalg.inv(T_velo_from_cam)
        persp = _load_perspective_calib(calib / "perspective.txt")
        self.P_rect00 = persp["P_rect_00"]
        self.raw_size = (int(persp["S_rect_00"][0]), int(persp["S_rect_00"][1]))  # (W, H)
        self.cam0_world = _load_indexed_poses_txt(self.drive / "cam0_to_world.txt", mat_size=16)

    # ------------------------------------------------------------- basics
    def scan(self, fid: int) -> np.ndarray:
        return np.fromfile(str(self.lidar / f"{fid:010d}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]

    def has_scan(self, fid: int) -> bool:
        return (self.lidar / f"{fid:010d}.bin").exists()

    def project(self, fid: int, image_size: Optional[Tuple[int, int]] = None):
        """-> (uv (N,2) float, z (N,), inb (N,) bool) at the given output size (W,H)."""
        pts = self.scan(fid)
        h = np.hstack([pts, np.ones((len(pts), 1))])
        cam = (self.P_rect00 @ self.T_cam_from_velo @ h.T).T
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cam[:, 0] / z
            v = cam[:, 1] / z
        W, H = image_size if image_size else self.raw_size
        sx, sy = W / self.raw_size[0], H / self.raw_size[1]
        u = u * sx
        v = v * sy
        inb = (z > 1.0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return np.stack([u, v], 1), z, inb

    def world_from_velo(self, fid: int) -> Optional[np.ndarray]:
        Tcw = self.cam0_world.get(fid)
        if Tcw is None:
            return None
        return Tcw @ self.T_cam_from_velo

    def scan_world(self, fid: int):
        T = self.world_from_velo(fid)
        if T is None:
            return None, None
        pts = self.scan(fid)
        h = np.hstack([pts, np.ones((len(pts), 1))])
        return (T @ h.T).T[:, :3], pts

    # --------------------------------------------------- static / dynamic
    def static_dynamic_labels(self, fid: int, th: float = 0.25, neighbors=(1, 1)):
        """Temporal-consistency labels for scan `fid`.

        A point is STATIC iff the nearest point of the previous or next scan
        (transformed into the world frame with exact ego poses) is within `th`
        meters. Occlusion at one neighbor is tolerated by taking the min over
        the available neighbors.

        Returns (labels (N,) bool static, min_dist (N,)) or (None, None) if
        no neighbor scan has a pose.
        """
        from scipy.spatial import cKDTree

        pts_world, _ = self.scan_world(fid)
        if pts_world is None:
            return None, None
        min_d = np.full(len(pts_world), np.inf)
        used = 0
        for k in range(1, max(neighbors) + 1):
            for sign in (-1, 1):
                nf = fid + sign * k
                if not self.has_scan(nf):
                    continue
                nb_world, _ = self.scan_world(nf)
                if nb_world is None:
                    continue
                tree = cKDTree(nb_world[::2])  # subsample x2 for speed
                d, _ = tree.query(pts_world, k=1, workers=-1)
                min_d = np.minimum(min_d, d)
                used += 1
        if used == 0:
            return None, None
        static = min_d < th
        return static, min_d

    def pixel_masks(self, fid: int, image_size: Tuple[int, int], th: float = 0.25,
                    dilate: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """-> (static_mask (H,W) bool, dynamic_mask (H,W) bool)."""
        import cv2

        static, _ = self.static_dynamic_labels(fid, th=th)
        uv, z, inb = self.project(fid, image_size)
        W, H = image_size
        sm = np.zeros((H, W), np.uint8)
        dm = np.zeros((H, W), np.uint8)
        if static is not None:
            sel = inb & static
            sm[uv[sel, 1].astype(int), uv[sel, 0].astype(int)] = 1
            sel = inb & ~static
            dm[uv[sel, 1].astype(int), uv[sel, 0].astype(int)] = 1
        k = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
        sm = cv2.dilate(sm, k) if k is not None else sm
        dm = cv2.dilate(dm, k) if k is not None else dm
        return sm.astype(bool), dm.astype(bool)

    def sparse_depth(self, fid: int, image_size: Tuple[int, int], static_only: bool = False):
        """-> (depth (H,W) float32 with 0=invalid, valid (H,W) bool)."""
        W, H = image_size
        uv, z, inb = self.project(fid, image_size)
        keep = inb.copy()
        if static_only:
            static, _ = self.static_dynamic_labels(fid)
            if static is not None:
                keep &= static
        depth = np.zeros((H, W), np.float32)
        ui = uv[keep, 0].astype(int)
        vi = uv[keep, 1].astype(int)
        zz = z[keep]
        # keep closest point per pixel (front-most surface)
        order = np.argsort(-zz)
        depth[vi[order], ui[order]] = zz[order]
        return depth, depth > 0


def absrel_scale_aligned(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """AbsRel after least-squares scale alignment of pred to sparse gt.

    pred: dense relative/absolute depth (H,W); gt/valid from sparse_depth.
    Same instrument for all model variants; comparisons are relative.
    """
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    if len(g) < 10:
        return float("nan")
    s = max((p @ g) / max(p @ p, 1e-9), 1e-9)
    return float(np.mean(np.abs(s * p - g) / g))
