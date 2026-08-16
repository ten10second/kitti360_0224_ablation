"""Day-0 V1: oxts->world pose pipeline + LiDAR->front camera depth projection.

Checks (against plan requirements):
 [P1] oxts->ENU poses give metric scale: consecutive-frame displacement ~ typical KITTI speeds
 [P2] trajectory continuity / no jumps (GPS glitches would break window construction)
 [P3] LiDAR projects onto image_02 with high in-bounds coverage (Gate A prerequisite)
 [P4] visualization: RGB + projected LiDAR depth overlay (human check)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from kitti_raw_io import (
    read_oxts, oxts_to_poses_enu, poses_to_local, read_calib,
    load_velo, project_velo_to_cam, effective_drives,
)

ROOT = "/media/shizhm/Lenovo/KITTI_RAW"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

DRIVES = [
    "2011_09_26/2011_09_26_drive_0002_sync",
    "2011_09_26/2011_09_26_drive_0009_sync",   # velodyne has 4 frames fewer
    "2011_09_26/2011_09_26_drive_0022_sync",   # 800 frames, city+residential
    "2011_10_03/2011_10_03_drive_0034_sync",   # highway-ish, another date/calib
]


def depth_to_color(depth: np.ndarray, dmin=2.0, dmax=60.0) -> np.ndarray:
    x = np.clip((depth - dmin) / (dmax - dmin), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def main():
    print("=" * 72)
    print("V1: oxts->pose + LiDAR->cam projection on KITTI-raw")
    print("=" * 72)
    all_step = []
    all_jump = []
    for dspec in DRIVES:
        drive_dir = Path(ROOT) / dspec
        date = dspec.split("/")[0]
        calib_dir = Path(ROOT) / date / f"{date}_calib"

        oxts = read_oxts(drive_dir / "oxts" / "data")
        poses_abs = oxts_to_poses_enu(oxts)
        poses_loc = poses_to_local(poses_abs, 0)
        xy = poses_loc[:, :2, 3]
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        all_step.append(step)
        # GPS glitch check: per-step speed > 30 m/s at 10Hz is impossible
        n_jump = int((step > 30.0).sum())
        all_jump.append(n_jump)
        # yaw consistency: relative heading from pose vs oxts yaw delta (both start at 0)
        yaw_pose = np.unwrap(np.arctan2(poses_loc[:, 1, 0], poses_loc[:, 0, 0]))
        yaw_oxts = np.unwrap(oxts[:, 5] - oxts[0, 5])
        yaw_err = np.degrees(np.abs((yaw_pose - yaw_oxts + np.pi) % (2 * np.pi) - np.pi))
        print(f"\n[{dspec}]")
        print(f"  frames={len(oxts)}  path_len={float(np.linalg.norm(xy[-1])):.1f}m(start->end)  cum={step.sum():.1f}m")
        print(f"  step(m): mean={step.mean():.3f} p50={np.percentile(step,50):.3f} p99={np.percentile(step,99):.3f} max={step.max():.3f}")
        print(f"  speed(km/h): mean={step.mean()*36:.1f} max={step.max()*36:.1f}  jumps>30m/s: {n_jump}")
        print(f"  pose-yaw vs oxts-yaw: max_err={yaw_err.max():.3f} deg mean_err={yaw_err.mean():.3f} deg")

        # ---- LiDAR projection ----
        calib = read_calib(calib_dir)
        n_vel_files = len(list((drive_dir / "velodyne_points" / "data").glob("*.bin")))
        probe_ids = [0, len(oxts) // 2, len(oxts) - 2]
        cov_list, zmed_list = [], []
        for fid in probe_ids:
            bin_path = drive_dir / "velodyne_points" / "data" / f"{fid:010d}.bin"
            img = cv2.imread(str(drive_dir / "image_02" / "data" / f"{fid:010d}.png"))
            if not bin_path.exists() or img is None:
                print(f"  [frame {fid}] velodyne missing (known gap)")
                continue
            H, W = img.shape[:2]
            pts = load_velo(bin_path)
            uv, z = project_velo_to_cam(pts, calib, "02")
            inb = (z > 1.0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            # pixel coverage: dilate point mask so sparse far points count
            pm = np.zeros((H, W), dtype=np.uint8)
            pm[uv[inb, 1].astype(int), uv[inb, 0].astype(int)] = 1
            pm_cov = float(cv2.dilate(pm, np.ones((5, 5), np.uint8)).mean())
            cov_list.append(pm_cov)
            zmed_list.append(float(np.median(z[inb])) if inb.any() else float("nan"))
            canvas = img.copy()
            order = np.argsort(-z)  # far first, near overwrites
            sel = order[np.isin(order, np.where(inb)[0])] if False else order[inb[order]]
            ui, vi = uv[sel, 0].astype(int), uv[sel, 1].astype(int)
            colors = depth_to_color(z[sel]).reshape(-1, 3)
            canvas[vi, ui] = colors
            # side-by-side
            side = np.concatenate([img, canvas], axis=1)
            tag = dspec.replace("/", "_")
            cv2.imwrite(str(OUT / f"v1_lidar_{tag}_{fid:010d}.jpg"), side,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  velodyne files={n_vel_files} vs frames={len(oxts)}")
        if cov_list:
            print(f"  LiDAR pixel coverage(5px dilate): {['%.1f%%' % (c*100) for c in cov_list]}  median_z(m): {['%.1f' % z for z in zmed_list]}")

    step = np.concatenate(all_step)
    print("\n---- V1 summary over %d drives ----" % len(DRIVES))
    print(f"  total frames={sum(len(s)+1 for s in all_step)}  step mean={step.mean():.3f}m  p99={np.percentile(step,99):.3f}m  jumps={sum(all_jump)}")
    print("  PASS criteria: step mean in [0.5,2] m @10Hz city; jumps==0; LiDAR coverage>50%; vis overlay locks onto image structure")


if __name__ == "__main__":
    main()
