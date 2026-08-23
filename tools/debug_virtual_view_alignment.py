#!/usr/bin/env python3
"""Debug/verify that dataloader-provided K and T_pose_cam are consistent with
warp_bev_to_camera_with_coords used in training.

We compare:
- target view image (front or virtual)
- BEV->camera inverse projection (warped_front) computed from the same K and pose
- valid mask

Outputs a grid image for quick inspection.
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world3d.io.kitti360d_dataloader import Kitti360dDataset
from utils.geometry import warp_bev_to_camera_with_coords

# Keep consistent with training script world3d/train/train_ar_1215_rot.py
IMU_TO_GROUND_HEIGHT = 0.93


def extract_center_square(img: np.ndarray, size: int = 512) -> np.ndarray | None:
    """Extract center square crop from HxWxC image."""
    if img is None:
        return None
    h, w = img.shape[:2]
    if h < size or w < size:
        return None
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0 : y0 + size, x0 : x0 + size]


def compute_inverse_projection_view(
    sat_img: np.ndarray | None,
    K: torch.Tensor | None,
    T_cam_to_world: torch.Tensor | None,
    T_imu_to_world: torch.Tensor | None,
    target_h: int,
    target_w: int,
    device: torch.device,
    bev_size: int = 512,
    resolution: float = 0.196,
):
    """Same BEV->camera backward warping wrapper as train_ar_1215_rot.py."""
    if sat_img is None or K is None or T_cam_to_world is None or T_imu_to_world is None:
        return None, None, None

    sat_square = extract_center_square(sat_img, size=bev_size)
    if sat_square is None:
        return None, None, None

    sat_tensor = torch.from_numpy(sat_square).float().permute(2, 0, 1).unsqueeze(0)
    sat_tensor = sat_tensor.to(device).clamp(0.0, 1.0)

    ground_z = float(T_imu_to_world[2, 3].item()) - IMU_TO_GROUND_HEIGHT

    warped_front, valid_mask, coords_map = warp_bev_to_camera_with_coords(
        sat_image=sat_tensor,
        K=K,
        T_cam_to_world=T_cam_to_world,
        T_imu_to_world=T_imu_to_world,
        cam_height=target_h,
        cam_width=target_w,
        sat_size=bev_size,
        resolution=resolution,
        ground_height=ground_z,
    )
    return warped_front, valid_mask, coords_map


def to_uint8(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0).round().astype(np.uint8)


def hstack(imgs, pad=8, pad_color=20):
    hs = [im.shape[0] for im in imgs]
    ws = [im.shape[1] for im in imgs]
    H = max(hs)
    W = sum(ws) + pad * (len(imgs) - 1)
    out = np.full((H, W, 3), pad_color, dtype=np.uint8)
    x = 0
    for im in imgs:
        h, w = im.shape[:2]
        y = (H - h) // 2
        out[y : y + h, x : x + w] = im
        x += w + pad
    return out


def main():
    # Pick a drive and frame
    drive_dir = REPO_ROOT / "2013_05_28_drive_0003_sync"
    frame_id = 0

    # Views to test
    views = [
        ("front", None, None, "front"),
        ("fisheye_virtual", "image_02", -60.0, "v_-60"),
        ("fisheye_virtual", "image_02", -30.0, "v_-30"),
        ("fisheye_virtual", "image_03", +30.0, "v_+30"),
        ("fisheye_virtual", "image_03", +60.0, "v_+60"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for mode, cam, yaw, tag in views:
        if mode == "front":
            ds = Kitti360dDataset(drives=drive_dir, mode="front")
        else:
            ds = Kitti360dDataset(
                drives=drive_dir,
                mode=mode,
                fisheye_camera=cam,
                vehicle_relative_yaw_deg=yaw,
                virtual_hfov_deg=80.0,
                virtual_size=(640, 256),
            )
        s = ds[frame_id]

        img = s["image"].permute(1, 2, 0).numpy()  # (H,W,3) in [0,1]
        sat = s["sat"].permute(1, 2, 0).numpy()  # (512,512,3) in [0,1]
        mpp = float(s["sat_m_per_px"])

        K = s["K"].to(device)
        # Use the dataloader-composed poses (these match training / geometry.warp expectations)
        T_cam_to_world = s["T_cam_to_world"].to(device)

        # Match training call style: use IMU pose if available, otherwise fall back to camera pose.
        # (Using identity here breaks the ground-plane intersection logic.)
        T_imu_to_world = s.get("T_imu_to_world", None)
        if T_imu_to_world is None:
            T_imu_to_world = T_cam_to_world
        else:
            T_imu_to_world = T_imu_to_world.to(device)

        warped_front, valid_mask, _ = compute_inverse_projection_view(
            sat_img=sat,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            target_h=img.shape[0],
            target_w=img.shape[1],
            device=device,
            bev_size=int(sat.shape[0]),
            resolution=mpp,
        )

        if warped_front is None or valid_mask is None:
            wf = np.zeros_like(img)
            vm = np.zeros_like(img)
        else:
            wf = warped_front.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            vm = valid_mask.detach().cpu().squeeze(0).numpy()  # (1,H,W)
            vm = np.repeat(vm.transpose(1, 2, 0), 3, axis=2)

        # Visualize: target / warped / validmask
        row = hstack(
            [
                to_uint8(img),
                to_uint8(wf),
                to_uint8(vm),
            ],
            pad=10,
        )

        # Add a small text bar by increasing top padding (simple)
        bar = np.full((24, row.shape[1], 3), 0, dtype=np.uint8)
        out = np.vstack([bar, row])
        rows.append(out)

        print(f"[{tag}] image={img.shape} sat={sat.shape} mpp={mpp} K[0,0]={float(K[0,0]):.2f}")

    grid = np.vstack(rows)
    out_path = REPO_ROOT / "tools" / "_debug_virtual_view_alignment_frame0000000000.png"
    Image.fromarray(grid).save(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
