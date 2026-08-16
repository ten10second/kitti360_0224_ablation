#!/usr/bin/env python3
"""Visualize BEV->camera warp for front or virtual perspective views.

Modes:
  --mode front:   visualize front camera (image_00)
  --mode virtual: visualize multiple virtual fisheye yaws for one frame
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math
import numpy as np
import cv2
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from world3d.io.kitti360d_dataloader import Kitti360dDataset
from utils.geometry import warp_bev_to_camera_with_coords

IMU_TO_GROUND_HEIGHT = 0.93
def oxts_yaw_to_north0_cw_deg(oxts_yaw_rad: float) -> float:
    """Convert KITTI-360 OXTS yaw to north=0, CW+ degrees."""
    deg = math.degrees(math.pi / 2.0 - float(oxts_yaw_rad))
    deg = (deg + 180.0) % 360.0 - 180.0
    return deg


def north0_cw_deg_to_unitvec_xy_east_north(angle_deg: float) -> np.ndarray:
    """Convert north=0, CW+ angle to a unit vector in ENU (x=east, y=north)."""
    a = math.radians(float(angle_deg))
    north = math.cos(a)
    east = math.sin(a)
    return np.array([east, north], dtype=np.float64)


def to_uint8(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0).round().astype(np.uint8)


def hstack(imgs, pad=10, pad_color=20):
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


def vstack(imgs, pad=10, pad_color=20):
    hs = [im.shape[0] for im in imgs]
    ws = [im.shape[1] for im in imgs]
    H = sum(hs) + pad * (len(imgs) - 1)
    W = max(ws)
    out = np.full((H, W, 3), pad_color, dtype=np.uint8)
    y = 0
    for im in imgs:
        h, w = im.shape[:2]
        x = (W - w) // 2
        out[y : y + h, x : x + w] = im
        y += h + pad
    return out


def warp_one(s, device: torch.device):
    """Run warp on a single dataloader sample and return vis image row."""
    img = s["image"].permute(1, 2, 0).cpu().numpy()
    H, W = img.shape[:2]

    sat = s["sat"].unsqueeze(0).to(device)
    K = s["K"].to(device)
    T_cam_to_world = s.get("T_cam_to_world", None)
    T_imu_to_world = s.get("T_imu_to_world", None)

    if T_cam_to_world is None or T_imu_to_world is None:
        raise RuntimeError("Missing T_cam_to_world/T_imu_to_world in sample")

    T_cam_to_world = T_cam_to_world.to(device)
    T_imu_to_world = T_imu_to_world.to(device)

    ground_z = float(T_imu_to_world[2, 3].item()) - IMU_TO_GROUND_HEIGHT
    mpp = float(s["sat_m_per_px"])

    warped, _, _ = warp_bev_to_camera_with_coords(
        sat_image=sat,
        K=K,
        T_cam_to_world=T_cam_to_world,
        T_imu_to_world=T_imu_to_world,
        cam_height=H,
        cam_width=W,
        sat_size=512,
        resolution=mpp,
        ground_height=ground_z,
    )

    warped_np = warped.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # Get satellite image (BEV)
    sat_np = s["sat"].permute(1, 2, 0).cpu().numpy()
    return to_uint8(img), to_uint8(warped_np), sat_np


def draw_arrow_on_bev(
    bev_img: np.ndarray,
    heading_xy: np.ndarray,
    label: str,
    color: tuple,
    arrow_length: int = 80,
    draw_north_indicator: bool = False,
) -> np.ndarray:
    """Draw a heading vector on the BEV image and return the new image."""
    bev_img_out = bev_img.copy()
    h, w = bev_img_out.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Draw North indicator first, if requested
    if draw_north_indicator:
        north_start = (30, 60)
        north_end = (30, 20)
        cv2.arrowedLine(bev_img_out, north_start, north_end, (255, 255, 255), 2, tipLength=0.4)
        cv2.putText(bev_img_out, "N", (north_start[0] - 8, north_start[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # heading_xy is (east, north). Image coords are (x_right, y_down).
    # So east -> +x, north -> -y
    end_x = center_x + int(arrow_length * heading_xy[0])
    end_y = center_y - int(arrow_length * heading_xy[1])

    cv2.arrowedLine(bev_img_out, (center_x, center_y), (end_x, end_y), color, 2, tipLength=0.2)
    cv2.putText(bev_img_out, label, (end_x + 5, end_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return bev_img_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=str, default="2013_05_28_drive_0003_sync")
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--mode", type=str, default="virtual", choices=["front", "virtual"])
    args = parser.parse_args()

    drive_dir = REPO_ROOT / args.drive
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Get base vehicle heading from one of the samples ---
    ds_probe = Kitti360dDataset(drives=drive_dir, mode="front")
    s_probe = ds_probe[args.frame]
    oxts_yaw = float(s_probe["meta"].get("oxts_yaw") or 0.0)
    vehicle_yaw_deg = oxts_yaw_to_north0_cw_deg(oxts_yaw)

    if args.mode == "front":
        img, warped, bev = warp_one(s_probe, device)
        front_heading_xy = north0_cw_deg_to_unitvec_xy_east_north(vehicle_yaw_deg)
        bev_with_arrow = draw_arrow_on_bev(
            bev.copy(), front_heading_xy, "front", (0, 255, 0), draw_north_indicator=True
        )
        out = hstack([img, warped, bev_with_arrow])
        out_path = REPO_ROOT / "tools" / f"_warp_front_{args.frame:010d}.png"
        print(f"Visualizing front camera for frame {args.frame}")

    else:  # virtual
        yaws_to_test = [90.0, 50.0, -50.0, -90.0]
        colors = [
            (255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255), (0, 0, 255)
        ]

        rows = []
        print(f"Base vehicle heading for frame {args.frame}: {vehicle_yaw_deg:+.1f} deg (North=0, CW+)")
        for yaw, color in zip(yaws_to_test, colors):
            ds = Kitti360dDataset(
                drives=drive_dir,
                mode="fisheye_virtual",
                vehicle_relative_yaw_deg=yaw,
                virtual_hfov_deg=80.0,
                virtual_size=(640, 256),
            )
            s = ds[args.frame]
            img, warped, bev = warp_one(s, device)

            yaw_world_deg = vehicle_yaw_deg + yaw
            heading_xy = north0_cw_deg_to_unitvec_xy_east_north(yaw_world_deg)

            label = f"{yaw:+.0f}"
            bev_with_arrow = draw_arrow_on_bev(
                bev, heading_xy, label, color, draw_north_indicator=True
            )

            row = hstack([img, warped, bev_with_arrow])
            rows.append(row)

            cam_used = s["meta"].get("fisheye_camera")
            print(f"  - Processed virtual yaw={yaw:+.1f} (world: {yaw_world_deg:+.1f}) deg, used {cam_used}")

        out = vstack(rows, pad=8)
        out_path = REPO_ROOT / "tools" / f"_warp_virtual_{args.frame:010d}.png"
        print(f"Visualizing virtual cameras for frame {args.frame}")

    Image.fromarray(out).save(out_path)
    print(f"\nSaved visualization to: {out_path}")


if __name__ == "__main__":
    main()