#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated fisheye to perspective converter with satellite visualization
Combines fisheye2flatten.py and draw_vehicle_heading.py functionality
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import yaml

# Configuration
SAT_IMG_PATH = "squence_03_frame0.png"
METERS_PER_PIXEL = 0.196   # measured KITTI-360 resolution

# Camera mounting angle (relative to vehicle front)
# This needs to be adjusted based on which camera is being used.
# As per KITTI conventions: image_02 is left, image_03 is right.
# With "north=0, CW+" convention:
#  - vehicle left = vehicle_yaw - 90
#  - vehicle right = vehicle_yaw + 90
# The value here should correspond to the camera used in --calib
MOUNT_ANGLES = {
    "image_02": -90.0,
    "image_03": +90.0,
}


def load_mei_yaml(path: Path):
    """Return xi, K, D from a MEI YAML."""
    raw = path.read_text()
    if raw.lstrip().startswith("%YAML"):
        raw = "\n".join(raw.splitlines()[1:])
    y = yaml.safe_load(raw)

    xi = float(y["mirror_parameters"]["xi"])
    k1 = float(y["distortion_parameters"]["k1"])
    k2 = float(y["distortion_parameters"]["k2"])
    p1 = float(y["distortion_parameters"]["p1"])
    p2 = float(y["distortion_parameters"]["p2"])

    gamma1 = float(y["projection_parameters"]["gamma1"])
    gamma2 = float(y["projection_parameters"]["gamma2"])
    u0 = float(y["projection_parameters"]["u0"])
    v0 = float(y["projection_parameters"]["v0"])

    K = np.array([[gamma1, 0, u0], [0, gamma2, v0], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, p1, p2], dtype=np.float64)

    return xi, K, D


def rot_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def make_newK(out_w: int, out_h: int, hfov_deg: float):
    hfov = math.radians(hfov_deg)
    fx = (out_w * 0.5) / math.tan(hfov * 0.5)
    vfov = 2.0 * math.atan((out_h / out_w) * math.tan(hfov * 0.5))
    fy = (out_h * 0.5) / math.tan(vfov * 0.5)
    cx, cy = out_w * 0.5, out_h * 0.5
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def angle_to_unitvec_north0_cw(angle_rad: float):
    """
    Convert "north=0, CW+" angle to a world plane unit vector (X_north, Y_east).
    """
    dx = math.cos(angle_rad)   # North component
    dy = math.sin(angle_rad)   # East component
    return dx, dy


def world_to_pixel(X_north, Y_east, u0, v0, mpp):
    """
    Convert world meter coordinates to satellite pixel coordinates.
    """
    u = u0 + (Y_east / mpp)
    v = v0 - (X_north / mpp)
    return u, v

def get_yaw_from_oxts(oxts_path: Path) -> float:
    """Reads the yaw value from an oxts data file."""
    if not oxts_path.exists():
        raise FileNotFoundError(f"Oxts file not found at: {oxts_path}")
    
    with open(oxts_path, 'r') as f:
        line = f.readline()
        values = line.strip().split()
        if len(values) > 5:
            return float(values[5])
        else:
            raise ValueError("Oxts file does not contain enough data.")

def generate_and_visualize(drive_path: Path, frame_id: int, vehicle_relative_angle: float, hfov: float, out_w: int, out_h: int, out_path: Path | None, sat_img_path: Path):
    # Normalize the input angle to the range [-180, 180]
    vehicle_relative_angle = (vehicle_relative_angle + 180) % 360 - 180
    frame_str = f"{frame_id:010d}"

    # --- Automatic Camera Selection ---
    # Decide which camera to use based on the requested angle relative to the vehicle.
    # Generally, left camera for left-side views, right for right-side.
    use_camera = "image_02" if vehicle_relative_angle < 0 else "image_03"
    mount_deg = MOUNT_ANGLES[use_camera]

    # Construct paths based on selected camera and frame
    img_path = drive_path / use_camera / "data_rgb" / f"{frame_str}.png"
    calib_path = drive_path / "calibration" / f"{use_camera}.yaml"
    oxts_file = drive_path / "oxts" / "data" / f"{frame_str}.txt"

    if not calib_path.exists(): raise FileNotFoundError(calib_path)
    if not img_path.exists(): raise FileNotFoundError(img_path)

    # --- Fisheye Unwrapping ---
    # Calculate the internal yaw needed for the unwrapping function.
    # The formula is: internal_yaw = mount_angle - vehicle_relative_angle
    internal_yaw = mount_deg - vehicle_relative_angle

    xi, K, D = load_mei_yaml(calib_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(img_path)

    newK = make_newK(out_w, out_h, hfov)
    R = rot_y(internal_yaw)

    map1, map2 = cv2.omnidir.initUndistortRectifyMap(
        K, D, np.array([xi]), R, newK, (out_w, out_h),
        cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
    )
    view = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    if out_path is None:
        out_path = img_path.with_name(img_path.stem + f"_angle{int(vehicle_relative_angle)}_hfov{int(hfov)}.png")
    cv2.imwrite(str(out_path), view)
    print(f"Saved perspective view to {out_path}")

    # --- Satellite Visualization ---
    yaw_rad = get_yaw_from_oxts(oxts_file)
    vehicle_yaw_deg = math.degrees(math.pi/2 - yaw_rad)
    vehicle_yaw_deg = (vehicle_yaw_deg + 180) % 360 - 180

    # The final world angle is simply the vehicle's heading plus the requested relative angle.
    yaw_world_deg = vehicle_yaw_deg + vehicle_relative_angle

    sat_img = cv2.imread(str(sat_img_path))
    if sat_img is None:
        raise FileNotFoundError(sat_img_path)

    h, w = sat_img.shape[:2]
    u0, v0 = w // 2, h // 2

    yaw_rad_world = math.radians(yaw_world_deg)

    # Draw camera position
    cv2.circle(sat_img, (u0, v0), 8, (255, 255, 255), -1)
    cv2.circle(sat_img, (u0, v0), 6, (0, 0, 255), -1)

    # Draw direction arrow
    arrow_len_m = 10.0
    dxC, dyC = angle_to_unitvec_north0_cw(yaw_rad_world)
    
    # Arrow endpoint in world coordinates
    Xe = arrow_len_m * dxC
    Ye = arrow_len_m * dyC
    
    # Convert to pixel coordinates
    ue, ve = world_to_pixel(Xe, Ye, u0, v0, METERS_PER_PIXEL)

    # Draw arrow with outline for better visibility
    cv2.arrowedLine(sat_img, (u0, v0), (int(ue), int(ve)), (255, 255, 255), 5, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(sat_img, (u0, v0), (int(ue), int(ve)), (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.3)

    vis_out_path = out_path.with_name(out_path.stem + "_satellite_vis.png")
    cv2.imwrite(str(vis_out_path), sat_img)
    print(f"Saved satellite visualization to {vis_out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Flatten fisheye to perspective and create satellite visualization.")
    p.add_argument("--drive", type=Path, required=True, help="Path to the drive sync folder (e.g., 2013_05_28_drive_0003_sync)")
    p.add_argument("--frame", type=int, required=True, help="Frame number (e.g., 0)")
    p.add_argument("--calib", type=Path, help="(Optional) Calibration YAML path. If not provided, it's inferred from the --img path.")
    p.add_argument("--angle", type=float, default=0.0, help="Desired viewing angle in degrees relative to the vehicle's front. CW is positive (e.g., +30 is 30 deg to the right).")
    p.add_argument("--hfov", type=float, default=100.0, help="Horizontal FOV of virtual camera (deg)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--out", type=Path, help="Output image path for the perspective view")
    p.add_argument("--sat", type=Path, default=Path(SAT_IMG_PATH), help="Path to satellite image")
    return p.parse_args()


def main():
    args = parse_args()
    generate_and_visualize(args.drive, args.frame, args.angle, args.hfov, args.width, args.height, args.out, args.sat)


if __name__ == "__main__":
    main()
