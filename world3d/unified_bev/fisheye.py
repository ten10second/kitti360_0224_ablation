"""Virtual perspective crops from the KITTI-360 MEI fisheye cameras.

Each fisheye (image_02 left, image_03 right) is sampled with several tangent
90-degree perspective crops at configurable yaw offsets (spec section 3.2).
The virtual cameras are geometric pinhole cameras: ``K_virtual`` comes from
the requested FOV and ``T_world_virtual = T_world_pose @ T_pose_fisheye @
rot_y(-yaw)``.  LiDAR projection into a virtual view therefore only needs the
pinhole model; the MEI model is used solely to warp RGB.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


def _load_mei_yaml(path: Path) -> Tuple[float, np.ndarray, np.ndarray]:
    raw = path.read_text()
    if raw.lstrip().startswith("%YAML"):
        raw = "\n".join(raw.splitlines()[1:])
    y = yaml.safe_load(raw)
    xi = float(y["mirror_parameters"]["xi"])
    d = y["distortion_parameters"]
    p = y["projection_parameters"]
    K = np.array([[p["gamma1"], 0, p["u0"]], [0, p["gamma2"], p["v0"]], [0, 0, 1]], dtype=np.float64)
    D = np.array([d["k1"], d["k2"], d["p1"], d["p2"]], dtype=np.float64)
    return xi, K, D


def _rot_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def virtual_K(out_w: int, out_h: int, hfov_deg: float) -> np.ndarray:
    hfov = math.radians(hfov_deg)
    fx = (out_w * 0.5) / math.tan(hfov * 0.5)
    vfov = 2.0 * math.atan((out_h / out_w) * math.tan(hfov * 0.5))
    fy = (out_h * 0.5) / math.tan(vfov * 0.5)
    return np.array([[fx, 0, out_w * 0.5], [0, fy, out_h * 0.5], [0, 0, 1]], dtype=np.float64)


def mei_project_rays(rays: np.ndarray, xi: float, K: np.ndarray, D: np.ndarray):
    """Project unit-sphere ray directions through the MEI model to pixels.

    Standard Mei & Rives omnidirectional model matching the KITTI-360
    calibration YAML: sphere -> xi-plane -> radial+tangential distortion -> K.
    """
    xs = rays / np.clip(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12, None)
    denom = xs[:, 2] + xi
    ok = denom > 1e-9
    xu = xs[:, :2] / np.clip(denom, 1e-9, None)[:, None]
    r2 = np.sum(xu**2, axis=1)
    k1, k2, p1, p2 = (float(d) for d in D)
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd_x = xu[:, 0] * radial + 2.0 * p1 * xu[:, 0] * xu[:, 1] + p2 * (r2 + 2.0 * xu[:, 0] ** 2)
    xd_y = xu[:, 1] * radial + p1 * (r2 + 2.0 * xu[:, 1] ** 2) + 2.0 * p2 * xu[:, 0] * xu[:, 1]
    u = K[0, 0] * xd_x + K[0, 1] * xd_y + K[0, 2]
    v = K[1, 0] * xd_x + K[1, 1] * xd_y + K[1, 2]
    return u, v, ok


def build_virtual_map(
    xi: float, K: np.ndarray, D: np.ndarray, R_virt_to_fish: np.ndarray,
    K_virtual: np.ndarray, image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Source-pixel maps for warping fisheye RGB into the virtual crop.

    For every virtual pixel, backproject to a virtual-frame ray, rotate to
    the fisheye frame, and project through MEI.  Returns ``(map_x, map_y)``;
    out-of-model pixels are set to -1 so remapping can mask them.
    """
    W, H = image_size
    src_w, src_h = float(K[0, 2] * 2), float(K[1, 2] * 2)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
    pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1).reshape(-1, 3)
    rays_v = pix @ np.linalg.inv(K_virtual).T
    rays_f = rays_v @ R_virt_to_fish.T
    u, v, ok = mei_project_rays(rays_f, xi, K, D)
    inb = ok & (u >= 0) & (u < src_w) & (v >= 0) & (v < src_h)
    map_x = np.where(inb, u, -1.0).reshape(H, W).astype(np.float32)
    map_y = np.where(inb, v, -1.0).reshape(H, W).astype(np.float32)
    return map_x, map_y


def load_cam_to_pose(path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        vals = np.asarray([float(x) for x in rest.split()], dtype=np.float64)
        if vals.size != 12:
            continue
        T = np.eye(4, dtype=np.float64)
        T[:3] = vals.reshape(3, 4)
        out[name.strip()] = T
    return out


class FisheyeVirtualRig:
    """Precomputed warp maps and extrinsics for the virtual crop set."""

    def __init__(
        self,
        calib_dir: Path,
        image_size: Tuple[int, int],
        yaws_deg: List[float] = (-45.0, 0.0, 45.0),
        hfov_deg: float = 90.0,
        cameras: Tuple[str, ...] = ("image_02", "image_03"),
    ):
        self.image_size = image_size
        self.cameras = cameras
        self.yaws = tuple(float(y) for y in yaws_deg)
        self.K = virtual_K(image_size[0], image_size[1], hfov_deg)
        self._mei = {c: _load_mei_yaml(calib_dir / f"{c}.yaml") for c in cameras}
        calib = load_cam_to_pose(calib_dir / "calib_cam_to_pose.txt")
        self.T_pose_fisheye = {c: calib[c] for c in cameras}
        T_pose_cam0 = calib["image_00"]
        R = T_pose_cam0[:3, :3]
        self.T_cam0_pose = np.eye(4, dtype=np.float64)
        self.T_cam0_pose[:3, :3] = R.T
        self.T_cam0_pose[:3, 3] = -R.T @ T_pose_cam0[:3, 3]
        # One (map, valid mask) per (camera, yaw); the fisheye image content
        # changes per frame but the warp does not.  Pixels the MEI model
        # cannot reach are mapped to -1 (black border in cv2.remap).
        self._maps: Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray]] = {}
        for cam in cameras:
            xi, K, D = self._mei[cam]
            for yaw in self.yaws:
                self._maps[(cam, yaw)] = build_virtual_map(
                    xi, K, D, _rot_y(-yaw), self.K, image_size
                )
        self._valid: Dict[Tuple[str, float], np.ndarray] = {}
        for key, (m1, m2) in self._maps.items():
            self._valid[key] = m1 >= 0

    @property
    def views_per_camera(self) -> int:
        return len(self.yaws)

    def warp(self, cam: str, yaw: float, fisheye_bgr: np.ndarray) -> np.ndarray:
        m1, m2 = self._maps[(cam, yaw)]
        return cv2.remap(fisheye_bgr, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def valid_mask(self, cam: str, yaw: float, fisheye_bgr: np.ndarray) -> np.ndarray:
        return self._valid[(cam, yaw)]

    def T_world_virtual(self, cam: str, yaw: float, T_world_cam0: np.ndarray) -> np.ndarray:
        T_world_pose = T_world_cam0 @ self.T_cam0_pose
        T_fish_virtual = np.eye(4, dtype=np.float64)
        T_fish_virtual[:3, :3] = _rot_y(-yaw)
        return T_world_pose @ self.T_pose_fisheye[cam] @ T_fish_virtual

    def fisheye_path(self, drive_dir: Path, cam: str, fid: int) -> Path:
        return drive_dir / cam / "data_rgb" / f"{fid:010d}.png"
