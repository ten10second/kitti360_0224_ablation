"""KITTI-raw Day-0 verification: shared IO helpers (oxts->pose, calib, velodyne).

Follows the official KITTI-raw devkit conventions:
- oxts -> ENU world pose (Mercator scale at drive's first latitude)
- velodyne -> rectified cam2 (color left) projection via
  P_rect_02 @ [R_rect_00|0] @ [R_velo2cam|T_velo2cam]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ER = 6378137.0  # WGS84 earth radius (m)


def read_oxts(oxts_dir: Path) -> np.ndarray:
    files = sorted(Path(oxts_dir).glob("*.txt"))
    rows = []
    for f in files:
        vals = [float(x) for x in f.read_text().strip().split()]
        rows.append(vals[:6])  # lat lon alt roll pitch yaw
    return np.array(rows, dtype=np.float64)


def _rot_axis(axis: int, t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    if axis == 0:  # roll, x-axis
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == 1:  # pitch, y-axis
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)  # yaw, z


def oxts_to_poses_enu(oxts: np.ndarray) -> np.ndarray:
    """oxts packets -> absolute ENU pose matrices (T_world_imu), world = ENU at sea level.

    T[i] maps imu frame at time i into the absolute (Mercator-scaled) ENU frame:
    x east, y north, z up. yaw in oxts is heading CCW from east.
    """
    lat0 = oxts[0, 0]
    scale = np.cos(np.radians(lat0))
    poses = np.zeros((len(oxts), 4, 4), dtype=np.float64)
    for i, pkt in enumerate(oxts):
        lat, lon, alt, roll, pitch, yaw = pkt
        t = np.array([scale * np.radians(lon) * ER, scale * np.radians(lat) * ER, alt])
        R = _rot_axis(2, yaw) @ _rot_axis(1, pitch) @ _rot_axis(0, roll)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        poses[i] = T
    return poses


def poses_to_local(poses: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    """Convert absolute poses to a local frame anchored at ref_idx: T_local = inv(T[ref]) @ T[i]."""
    T_ref_inv = np.linalg.inv(poses[ref_idx])
    out = np.zeros_like(poses)
    for i in range(len(poses)):
        out[i] = T_ref_inv @ poses[i]
    return out


def read_calib(calib_dir: Path) -> Dict[str, np.ndarray]:
    """Read the three KITTI-raw calib files. Handles nested <date>/<file> layout."""
    calib_dir = Path(calib_dir)
    c2c = calib_dir / "calib_cam_to_cam.txt"
    if not c2c.exists():
        # nested layout: <date>_calib/<date>/calib_*.txt
        subs = [p for p in calib_dir.iterdir() if p.is_dir()]
        if subs:
            c2c = subs[0] / "calib_cam_to_cam.txt"
    v2c = c2c.parent / "calib_velo_to_cam.txt"
    i2v = c2c.parent / "calib_imu_to_velo.txt"

    def parse(path: Path) -> Dict[str, np.ndarray]:
        data = {}
        for line in path.read_text().splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            try:
                data[key.strip()] = np.array([float(x) for x in val.split()])
            except ValueError:
                continue  # calib_time etc.
        return data

    cam = parse(c2c)
    velo = parse(v2c)
    imu = parse(i2v)

    out: Dict[str, np.ndarray] = {}
    for cam_id in ["00", "02", "03"]:
        out[f"P_rect_{cam_id}"] = cam[f"P_rect_{cam_id}"].reshape(3, 4)
        out[f"R_rect_{cam_id}"] = cam[f"R_rect_{cam_id}"].reshape(3, 3)
        out[f"K_{cam_id}"] = cam[f"K_{cam_id}"].reshape(3, 3)
        out[f"S_rect_{cam_id}"] = cam[f"S_rect_{cam_id}"]
    R_v2c = velo["R"].reshape(3, 3)
    T_v2c = velo["T"].reshape(3, 1)
    out["velo_to_cam"] = np.vstack([np.hstack([R_v2c, T_v2c]), [0, 0, 0, 1]])  # 4x4
    R_i2v = imu["R"].reshape(3, 3)
    T_i2v = imu["T"].reshape(3, 1)
    out["imu_to_velo"] = np.vstack([np.hstack([R_i2v, T_i2v]), [0, 0, 0, 1]])  # 4x4
    return out


def project_velo_to_cam(pts_velo: np.ndarray, calib: Dict[str, np.ndarray], cam_id: str = "02") -> Tuple[np.ndarray, np.ndarray]:
    """Project (N,3) velodyne points to rectified cam image.

    Returns (uv (N,2), depth_z (N,)) where depth_z is z in the rectified cam frame.
    """
    P = calib[f"P_rect_{cam_id}"]
    R_rect = np.eye(4)
    R_rect[:3, :3] = calib[f"R_rect_00"]  # velodyne is registered to cam00 reference
    pts_h = np.hstack([pts_velo, np.ones((len(pts_velo), 1))])  # (N,4)
    pts_cam = (P @ R_rect @ calib["velo_to_cam"] @ pts_h.T).T  # (N,3)
    z = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = pts_cam[:, 0] / z
        v = pts_cam[:, 1] / z
    return np.stack([u, v], axis=1), z


def load_velo(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def effective_drives(root: str) -> List[str]:
    """Drives with image_02 + oxts + velodyne + matching satellite counts."""
    out = []
    for date in sorted(os.listdir(root)):
        ddir = os.path.join(root, date)
        if not os.path.isdir(ddir):
            continue
        for drive in sorted(os.listdir(ddir)):
            if not drive.endswith("_sync"):
                continue
            p = os.path.join(ddir, drive)
            n_img = len(os.listdir(os.path.join(p, "image_02", "data"))) if os.path.isdir(os.path.join(p, "image_02", "data")) else 0
            n_sat = len(os.listdir(os.path.join(p, "satellite"))) if os.path.isdir(os.path.join(p, "satellite")) else 0
            has_vel = os.path.isdir(os.path.join(p, "velodyne_points", "data"))
            if n_img > 0 and n_sat == n_img and has_vel:
                out.append(p)
    return out
