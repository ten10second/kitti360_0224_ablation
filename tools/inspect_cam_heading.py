#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def load_poses(seq_dir: Path):
    pose_path = seq_dir / "poses.txt"
    d = {}
    with open(pose_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 13:
                continue
            fid = int(parts[0])
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = np.array([float(x) for x in parts[1:]], dtype=np.float64).reshape(3, 4)
            d[fid] = T
    return d

def load_cam_to_pose(seq_dir: Path):
    path = seq_dir / "calibration" / "calib_cam_to_pose.txt"
    d = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            name, rest = line.split(":", 1)
            vals = [float(x) for x in rest.strip().split()]
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)
            d[name.strip()] = T
    return d

def inv_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv

def yaw_heading_deg_of_cam_z(T_cam_to_world: np.ndarray) -> float:
    # optical axis +Z_cam in world
    z_w = T_cam_to_world[:3, :3] @ np.array([0.0, 0.0, 1.0])
    yaw = np.degrees(np.arctan2(z_w[1], z_w[0]))
    return float(yaw)

def yaw_heading_deg_of_imu_x(T_imu_to_world: np.ndarray) -> float:
    # IMU +X axis (vehicle-forward) heading
    x_w = T_imu_to_world[:3, :3] @ np.array([1.0, 0.0, 0.0])
    yaw = np.degrees(np.arctan2(x_w[1], x_w[0]))
    return float(yaw)

def wrap180(a):
    a = (a + 180.0) % 360.0 - 180.0
    return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--frame", type=int, required=True)
    args = ap.parse_args()

    seq_dir = Path(args.data_root) / args.drive
    poses = load_poses(seq_dir)
    cams = load_cam_to_pose(seq_dir)

    if args.frame not in poses:
        raise SystemExit(f"frame {args.frame} not found in poses.txt")

    T_imu_to_world = poses[args.frame]
    yaw_imu = yaw_heading_deg_of_imu_x(T_imu_to_world)

    print(f"Frame {args.frame}")
    print(f"IMU/vehicle heading (world XY, +X_imu): {yaw_imu:.3f} deg")

    for cam in ["image_02", "image_03"]:
        if cam not in cams:
            print(f"- {cam}: not found in calib_cam_to_pose.txt")
            continue
        T_cam_to_pose = cams[cam]
        # Assumption A: file stores T_cam_to_pose
        T_cam_to_world_A = T_imu_to_world @ T_cam_to_pose
        yaw_cam_A = yaw_heading_deg_of_cam_z(T_cam_to_world_A)
        dA = wrap180(yaw_cam_A - yaw_imu)
        # Assumption B: file stores T_pose_to_cam
        T_pose_to_cam = T_cam_to_pose
        T_cam_to_world_B = T_imu_to_world @ inv_se3(T_pose_to_cam)
        yaw_cam_B = yaw_heading_deg_of_cam_z(T_cam_to_world_B)
        dB = wrap180(yaw_cam_B - yaw_imu)

        print(f"- {cam}:")
        print(f"  as T_cam_to_pose: yaw_world(z_cam)={yaw_cam_A:.3f}°, delta_vs_IMU={dA:.3f}°")
        print(f"  as T_pose_to_cam: yaw_world(z_cam)={yaw_cam_B:.3f}°, delta_vs_IMU={dB:.3f}°")

    # Also, show left-right delta under both assumptions
    if all(k in cams for k in ("image_02","image_03")):
        T02 = cams["image_02"]; T03 = cams["image_03"]
        yaw02A = yaw_heading_deg_of_cam_z(poses[args.frame] @ T02)
        yaw03A = yaw_heading_deg_of_cam_z(poses[args.frame] @ T03)
        yaw02B = yaw_heading_deg_of_cam_z(poses[args.frame] @ inv_se3(T02))
        yaw03B = yaw_heading_deg_of_cam_z(poses[args.frame] @ inv_se3(T03))
        print(f"\nLeft-Right delta (yaw02 - yaw03):")
        print(f"  Assumption A (cam_to_pose): {wrap180(yaw02A - yaw03A):.3f}°")
        print(f"  Assumption B (pose_to_cam): {wrap180(yaw02B - yaw03B):.3f}°")

if __name__ == "__main__":
    main()
