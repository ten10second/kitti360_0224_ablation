#!/usr/bin/env python3
"""Calculates the resolution of the BEV satellite images.

It works by comparing the physical distance moved by the vehicle between two
consecutive frames (from IMU poses) with the pixel shift between their
corresponding BEV satellite images.

Resolution (m/px) = Physical Distance (m) / Pixel Distance (px)
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch

# Make repo root importable
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CUR_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from world3d.io.kitti360d_dataloader import Kitti360dDataset

def _tensor_to_u8_bgr(img_t: torch.Tensor) -> np.ndarray:
    """Convert (3,H,W) float tensor in [0,1] to uint8 BGR for OpenCV."""
    if img_t.dim() != 3 or img_t.shape[0] != 3:
        raise ValueError(f"Expected (3,H,W), got {tuple(img_t.shape)}")
    t = img_t.detach().cpu().float().clamp(0, 1)
    img_rgb = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(img_bgr)

def find_pixel_shift(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
    """Find the pixel shift of img2 relative to img1 using template matching."""
    # Use a central patch from img2 as the template
    h, w = img2.shape[:2]
    top = h // 4
    left = w // 4
    template = img2[top:top + h//2, left:left + w//2]

    # Search for the template in img1
    res = cv2.matchTemplate(img1, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    # The shift is the difference between the found location and the original
    # location of the template's top-left corner.
    shift_x = left - max_loc[0]
    shift_y = top - max_loc[1]

    return shift_x, shift_y

def main():
    parser = argparse.ArgumentParser(
        description='Calculate BEV image resolution.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--drive', type=str, default='2013_05_28_drive_0003_sync',
                        help='KITTI-360 drive name.')
    parser.add_argument('--frame1', type=int, default=200, help='First frame index.')
    parser.add_argument('--frame2', type=int, default=201, help='Second frame index.')
    parser.add_argument('--data_dir', type=str, default=REPO_ROOT, help='Path to dataset root.')

    args = parser.parse_args()

    print(f"Comparing frame {args.frame1} and {args.frame2} from drive {args.drive}")

    # 1. Load dataset samples
    ds = Kitti360dDataset(
        drives=[os.path.join(args.data_dir, args.drive)],
        mode='fisheye_virtual',  # Use fisheye_virtual mode to get BEV images
        fisheye_camera='image_03',  # Use the right fisheye camera
        virtual_hfov_deg=80,  # Match the HFOV used in debug_ipm_alignment.py
        virtual_size=(640, 256),  # Match the size used in debug_ipm_alignment.py
    )

    # Get the list of frame IDs and find the indices of the requested frames
    frame_ids = [s.frame_id for s in ds.samples]
    try:
        idx1 = frame_ids.index(args.frame1)
        idx2 = frame_ids.index(args.frame2)
    except ValueError as e:
        print(f"Error: {e}. Available frame IDs: {frame_ids[:10]}...")
        return

    # Get the samples using the indices
    sample1 = ds[idx1]
    sample2 = ds[idx2]

    # 2. Calculate physical distance
    T1 = sample1['T_imu_to_world'].numpy()
    T2 = sample2['T_imu_to_world'].numpy()
    pos1 = T1[:3, 3]
    pos2 = T2[:3, 3]
    dist_world = np.linalg.norm(pos2[:2] - pos1[:2]) # Only consider XY plane

    print(f"World position 1 (X,Y): ({pos1[0]:.3f}, {pos1[1]:.3f}) m")
    print(f"World position 2 (X,Y): ({pos2[0]:.3f}, {pos2[1]:.3f}) m")
    print(f"Physical distance moved: {dist_world:.4f} m")

    if dist_world < 1e-3:
        print("Warning: Vehicle has barely moved. Resolution calculation might be inaccurate.")

    # 3. Calculate pixel distance
    sat1_img = _tensor_to_u8_bgr(sample1['sat'])
    sat2_img = _tensor_to_u8_bgr(sample2['sat'])

    shift_x, shift_y = find_pixel_shift(sat1_img, sat2_img)
    dist_pixel = np.sqrt(shift_x**2 + shift_y**2)

    print(f"Pixel shift (x,y): ({shift_x:.3f}, {shift_y:.3f}) px")
    print(f"Pixel distance: {dist_pixel:.4f} px")

    # 4. Compute resolution
    if dist_pixel < 1e-3:
        print("Error: No pixel shift detected. Cannot calculate resolution.")
        sys.exit(1)

    resolution = dist_world / dist_pixel
    print(f"\nCalculated BEV resolution: {resolution:.6f} m/pixel")

    # For reference, check against the default value in the dataloader
    print(f"Default resolution in dataloader: {ds.sat_m_per_px} m/pixel")

if __name__ == '__main__':
    main()

