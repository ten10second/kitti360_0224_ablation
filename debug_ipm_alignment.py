#!/usr/bin/env python3
"""
KITTI-360 IPM Alignment - Simple Local Rotation Mode
Logic:
- No vehicle-front alignment.
- Virtual camera starts aligned exactly with the Physical Fisheye Optical Axis.
- Rotate ONLY around the Physical Fisheye's Y-axis.

python debug_ipm_alignment.py   --data_root /home/zhimiao/code_space/kitti360_camera_pose   --drive 2013_05_28_drive_0003_sync   --frame 240   --camera image_02   --out_dir debug_output   --yaw -30   --calib_yaw_fix 4

For front view (image_00):
python debug_ipm_alignment.py   --data_root /home/zhimiao/code_space/kitti360_camera_pose   --drive 2013_05_28_drive_0003_sync   --frame 240   --camera image_00   --out_dir debug_output

ps: 不自动切换相机，根据指定的 yaw 角度手动选择相机。
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import yaml
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.geometry.bev_to_camera_warp import (
    warp_bev_to_camera_with_coords,
)

# Constants
IMU_TO_GROUND_HEIGHT = 0.93

# ----------------------- Geometry Helpers -----------------------

def rot_y(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

def rot_x(deg: float) -> np.ndarray:
    """Create a 3x3 rotation matrix around the X axis."""
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ], dtype=np.float64)

def rot_z(deg: float) -> np.ndarray:
    """Create a 3x3 rotation matrix around the Z axis."""
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

def load_mei_yaml(path: Path):
    raw = path.read_text()
    if raw.lstrip().startswith("%YAML"):
        raw = "\n".join(raw.splitlines()[1:])
    y = yaml.safe_load(raw)
    xi = float(y["mirror_parameters"]["xi"])
    K = np.array([
        [float(y["projection_parameters"]["gamma1"]), 0.0, float(y["projection_parameters"]["u0"])],
        [0.0, float(y["projection_parameters"]["gamma2"]), float(y["projection_parameters"]["v0"])],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    D = np.array([
        float(y["distortion_parameters"]["k1"]),
        float(y["distortion_parameters"]["k2"]),
        float(y["distortion_parameters"]["p1"]),
        float(y["distortion_parameters"]["p2"])
    ], dtype=np.float64)
    return xi, K, D

def load_perspective_calib(path: Path):
    """Load calibration from perspective.txt file for front camera (image_00)."""
    calib_data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('calib_time:'):
                continue
            parts = line.split(':')
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            values_str = parts[1].strip()
            if key.startswith('K_') or key.startswith('D_') or key.startswith('R_') or key.startswith('T_'):
                values = [float(v) for v in values_str.split()]
                if key.startswith('K_'):
                    calib_data[key] = np.array(values).reshape(3, 3)
                elif key.startswith('D_'):
                    calib_data[key] = np.array(values)
                elif key.startswith('R_') or key.startswith('T_'):
                    calib_data[key] = np.array(values)
            elif key.startswith('P_rect_'):
                values = [float(v) for v in values_str.split()]
                calib_data[key] = np.array(values).reshape(3, 4)
            elif key.startswith('S_') or key.startswith('S_rect_'):
                values = [float(v) for v in values_str.split()]
                calib_data[key] = np.array(values)
    return calib_data

def scale_intrinsics(K: np.ndarray, orig_h: int, orig_w: int, target_h: int, target_w: int) -> np.ndarray:
    """Scale intrinsics matrix K to match resized image."""
    sx = float(target_w) / float(orig_w)
    sy = float(target_h) / float(orig_h)
    K_out = K.copy()
    K_out[0, 0] *= sx  # fx
    K_out[1, 1] *= sy  # fy
    K_out[0, 2] *= sx  # cx
    K_out[1, 2] *= sy  # cy
    return K_out

def visualize_camera_frustum_on_bev(
    sat_img: np.ndarray,
    K_virtual: torch.Tensor,
    K_ipm: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    ground_z: float,
    resolution: float,
    sat_size: int,
) -> np.ndarray:
    """Draw camera frustum on BEV image."""
    bev_vis = sat_img.copy()
    H_sat, W_sat = bev_vis.shape[:2]

    # Camera position in world
    cam_pos_world = T_cam_to_world[:3, 3]

    # Convert to BEV coordinates (pixels)
    u_cam = cam_pos_world[0] / resolution + W_sat / 2.0
    v_cam = -cam_pos_world[1] / resolution + H_sat / 2.0

    cv2.circle(bev_vis, (int(u_cam), int(v_cam)), 5, (0, 255, 0), -1) # Green circle for camera

    # Frustum corners in virtual camera NDC space
    corners_ndc = torch.tensor([
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ], dtype=torch.float32, device=K_virtual.device)

    # Unproject to virtual camera coordinates
    inv_K_virtual = torch.inverse(K_virtual)
    corners_cam = (inv_K_virtual @ corners_ndc.T).T

    # Project to ground plane
    t = (ground_z - T_cam_to_world[2, 3]) / (T_cam_to_world[:3, :3] @ corners_cam.T)[2, :]
    points_world = T_cam_to_world[:3, 3].unsqueeze(1) + (T_cam_to_world[:3, :3] @ corners_cam.T) * t

    # Convert to BEV coordinates
    u_bev = points_world[0, :] / resolution + W_sat / 2.0
    v_bev = -points_world[1, :] / resolution + H_sat / 2.0

    pts_bev = torch.stack([u_bev, v_bev], dim=1).cpu().numpy().astype(np.int32)

    cv2.polylines(bev_vis, [pts_bev], isClosed=True, color=(255, 0, 255), thickness=2) # Magenta frustum

    return bev_vis

def make_newK(out_w: int, out_h: int, hfov_deg: float) -> torch.Tensor:
    hfov = float(np.deg2rad(hfov_deg))
    fx = (out_w * 0.5) / float(np.tan(hfov * 0.5))
    fy = fx
    cx, cy = out_w * 0.5, out_h * 0.5
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)

def fisheye_to_virtual_perspective(
    img_bgr: np.ndarray,
    calib_yaml: Path,
    R_rectify: np.ndarray,
    out_w: int,
    out_h: int,
    K_virtual: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Generate a virtual perspective view from a fisheye image."""
    xi, K_raw, D_raw = load_mei_yaml(calib_yaml)
    K_virtual_np = K_virtual.cpu().numpy()

    # Ensure D has 4 elements (k1, k2, p1, p2)
    D_raw = np.array(D_raw, dtype=np.float64)  # Ensure it's a numpy array
    if len(D_raw) == 2:  # If only k1, k2 are provided
        D_raw = np.array([D_raw[0], D_raw[1], 0, 0], dtype=np.float64)  # Pad with zeros for p1, p2

    map1, map2 = cv2.omnidir.initUndistortRectifyMap(
        K_raw, D_raw, np.array([xi], dtype=np.float64), R_rectify.astype(np.float64),
        K_virtual_np.astype(np.float64), (out_w, out_h), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    virtual_view = cv2.remap(img_bgr, map1, map2, cv2.INTER_LINEAR)

    return virtual_view, map1, map2, (xi, K_raw, D_raw)

# ----------------------- Core Logic -----------------------

def compute_camera_to_world_simple(T_imu_to_world, T_pose_cam, yaw_deg, device):
    """
    极简逻辑：直接在鱼眼物理系下绕 Y 轴旋转。
    """
    # 物理鱼眼相机的位姿 (World Frame)
    T_cam_phys_to_world = T_imu_to_world @ T_pose_cam
    R_phys_to_world = T_cam_phys_to_world[:3, :3]

    # 直接定义旋转：在鱼眼坐标系下绕其 Y 轴转动
    # 对于 KITTI-360 鱼眼，Y 是向下，所以 -yaw 表示向右转
    R_rel = torch.from_numpy(rot_y(-yaw_deg).astype(np.float32)).to(device)

    # 虚拟相机旋转：保持光轴语义，仅在鱼眼坐标系下做相对旋转
    R_virt = R_phys_to_world @ R_rel

    # 写回位姿
    T_cam_virt_to_world = T_cam_phys_to_world.clone()
    T_cam_virt_to_world[:3, :3] = R_virt

    # 计算 OpenCV 需要的 Rectify 矩阵 (Phys -> Virt)
    # R_rectify satisfies: R_virt = R_phys * R_rectify^T  =>  R_rectify = R_virt^T * R_phys = R_rel^T
    # R_rectify = R_rel.t()
    R_rectify = R_rel.t()

    return T_cam_virt_to_world, R_rectify

def compute_camera_to_world_front(T_imu_to_world, T_pose_cam, device):
    """
    前视图相机：直接使用相机的物理位姿，不需要额外旋转。
    """
    # 前视图相机的位姿 (World Frame)
    T_cam_to_world = T_imu_to_world @ T_pose_cam
    return T_cam_to_world

# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--drive", type=str, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera", type=str, default="image_02", choices=["image_00", "image_01", "image_02", "image_03"])
    parser.add_argument("--yaw", type=float, default=0.0, help="Rotation angle relative to fisheye optical axis (only for image_02/image_03)")
    parser.add_argument("--hfov", type=float, default=120.0, help="HFov for virtual perspective (only for fisheye)")
    parser.add_argument("--out_dir", type=str, default="debug_output")
    parser.add_argument("--res", type=float, default=0.2)
    parser.add_argument('--roll_ipm_deg', type=float, default=0.0, help='Manual roll correction for IPM only around camera Z axis (deg). roll>0 is CCW.')
    parser.add_argument('--pitch_ipm_deg', type=float, default=0.0, help='Manual pitch correction for IPM only around camera X axis (deg).')
    parser.add_argument('--calib_yaw_fix', type=float, default=0.0, help='Yaw correction (deg) to apply to the camera-to-IMU extrinsics. Positive is CCW.')
    parser.add_argument('--cam_height_m', type=float, default=1.95, help='Height of the camera above ground (in meters). Default: 1.95m for fisheye camera.')
    # Front view specific options
    parser.add_argument("--front_width", type=int, default=640, help="Output width for front view")
    parser.add_argument("--front_height", type=int, default=256, help="Output height for front view")
    # Intrinsics debugging helpers
    parser.add_argument('--calib_yaml_override', type=str, default=None,
                        help='Override path to MEI yaml for this camera. Useful to swap image_02 calib onto image_03, etc.')
    parser.add_argument('--override_xi', type=float, default=None, help='If set, override mirror parameter xi for rectification.')
    parser.add_argument('--scale_k1', type=float, default=1.0, help='Multiply k1 by this factor before rectification.')
    parser.add_argument('--scale_k2', type=float, default=1.0, help='Multiply k2 by this factor before rectification (e.g. 0.37 to map 4.5->1.65).')
    parser.add_argument('--zero_tangential', action='store_true', help='Force p1=p2=0 for rectification.')
    parser.add_argument('--save_overlay', action='store_true', help='Also save an overlay of IPM edges (G) on virtual view (R/B).')
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_dir = Path(args.data_root) / args.drive
    frame_str = f"{args.frame:010d}"

    is_front_view = args.camera in ["image_00", "image_01"]

    # --- 1. 数据加载 ---
    pose_path = seq_dir / "poses.txt"
    pose_dict = {}
    with open(pose_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 13: continue
            fid = int(parts[0])
            T = np.eye(4)
            T[:3, :] = np.array([float(x) for x in parts[1:]]).reshape(3, 4)
            pose_dict[fid] = T

    T_imu_to_world = torch.from_numpy(pose_dict[args.frame]).float().to(device)

    cam_to_pose_path = seq_dir / "calibration" / "calib_cam_to_pose.txt"
    cam_to_pose = {}
    with open(cam_to_pose_path, "r") as f:
        for line in f:
            if ":" not in line: continue
            name, rest = line.split(":", 1)
            vals = [float(x) for x in rest.strip().split()]
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            cam_to_pose[name.strip()] = T

    T_pose_cam = torch.from_numpy(cam_to_pose[args.camera]).float().to(device)

    # 应用 calib_yaw_fix 修正（如果非零）
    if abs(args.calib_yaw_fix) > 1e-6:
        print(f"[Debug] Applying calib_yaw_fix = {args.calib_yaw_fix:.4f} deg")
        # 创建一个绕 Z 轴旋转的修正矩阵
        yaw_rad = np.radians(args.calib_yaw_fix)
        R_fix_cam = torch.tensor([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [np.sin(yaw_rad),  np.cos(yaw_rad), 0.0],
            [0.0,              0.0,             1.0],
        ], dtype=torch.float32, device=device)
        T_pose_cam[:3, :3] = T_pose_cam[:3, :3] @ R_fix_cam  # 右乘，在相机坐标系内修正

        print(f"[Debug] Modified T_pose_cam rotation (first 3x3):")
        print(T_pose_cam[:3, :3].cpu().numpy())

    # --- 2. 核心变换 ---
    if is_front_view:
        # 前视图：直接使用相机位姿
        print(f"[Debug] Front view mode: {args.camera}")
        T_cam_virt_to_world = compute_camera_to_world_front(T_imu_to_world, T_pose_cam, device)
        R_rectify = None
    else:
        # 鱼眼视图：使用旋转逻辑
        T_cam_virt_to_world, R_rectify = compute_camera_to_world_simple(
            T_imu_to_world, T_pose_cam, args.yaw, device
        )

    # --- 3. 加载图像和内参 ---
    if is_front_view:
        # 前视图：读取 perspective.txt 标定
        out_w, out_h = args.front_width, args.front_height

        # 尝试从不同路径读取前视图图像
        img_path = seq_dir / args.camera / "data_rect" / f"{frame_str}.png"
        if not img_path.exists():
            img_path = seq_dir / args.camera / "data_rgb" / f"{frame_str}.png"
        if not img_path.exists():
            img_path = seq_dir / args.camera / "data" / f"{frame_str}.png"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Cannot load front image from: {img_path}")

        orig_h, orig_w = img_bgr.shape[:2]
        print(f"[Debug] Front view original size: {orig_w}x{orig_h}, resize to: {out_w}x{out_h}")

        # 读取 perspective.txt 标定
        calib_path = seq_dir / "calibration" / "perspective.txt"
        if not calib_path.exists():
            raise RuntimeError(f"perspective.txt not found: {calib_path}")

        calib = load_perspective_calib(calib_path)

        # 获取内参：优先使用 P_rect_xx，其次使用 K_xx
        cam_idx = args.camera[-2:]  # "00", "01", etc.
        P_key = f"P_rect_{cam_idx}"
        K_key = f"K_{cam_idx}"

        if P_key in calib:
            P = calib[P_key].astype(np.float64)
            K_orig = P[:, :3].copy()
            print(f"[Debug] Using {P_key} for intrinsics")
        elif K_key in calib:
            K_orig = calib[K_key].astype(np.float64)
            print(f"[Debug] Using {K_key} for intrinsics")
        else:
            raise RuntimeError(f"Cannot find intrinsics for {args.camera} in perspective.txt")

        # 调整图像大小和内参
        view_virtual = cv2.resize(img_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        K = scale_intrinsics(K_orig, orig_h, orig_w, out_h, out_w)
        K_v = torch.from_numpy(K).float().to(device)

        print(f"[Debug] Front view K (original):\n{K_orig}")
        print(f"[Debug] Front view K (scaled):\n{K}")

    else:
        # 鱼眼视图：原有逻辑
        out_w, out_h = 640, 256
        calib_yaml_path = Path(args.calib_yaml_override) if args.calib_yaml_override else (seq_dir / "calibration" / f"{args.camera}.yaml")
        xi, K_f, D_f = load_mei_yaml(calib_yaml_path)
        K_v = make_newK(out_w, out_h, args.hfov).to(device)

        img_path = seq_dir / args.camera / "data_rgb" / f"{frame_str}.png"
        if not img_path.exists(): img_path = seq_dir / args.camera / "data" / f"{frame_str}.png"

        img_bgr = cv2.imread(str(img_path))

        map1, map2 = cv2.omnidir.initUndistortRectifyMap(
            K_f, D_f, np.array([xi]), R_rectify.cpu().numpy().astype(np.float64),
            K_v.cpu().numpy().astype(np.float64), (out_w, out_h), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
        )
        view_virtual = cv2.remap(img_bgr, map1, map2, cv2.INTER_LINEAR)

    # --- Debug info for fisheye ---
    if not is_front_view and R_rectify is not None:
        def _yaw_y_deg_from_Ry(R: torch.Tensor) -> float:
            # For a pure rot_y(theta):
            #   R[0,2] = sin(theta), R[2,2] = cos(theta)
            # Recover theta in degrees.
            theta = torch.atan2(R[0, 2], R[2, 2])
            return float(theta.item() * 180.0 / np.pi)

        try:
            # 1) R_rectify should be a (phys->virt) rotation; report its equivalent y-rotation angle.
            yaw_rectify_y = _yaw_y_deg_from_Ry(R_rectify)

            # 2) Report where the virtual camera's forward (+Z) points in the *physical fisheye* frame:
            #    z_phys = R_rectify^T * z_virt
            z_virt = torch.tensor([0.0, 0.0, 1.0], device=R_rectify.device, dtype=R_rectify.dtype)
            z_in_phys = R_rectify.t() @ z_virt
            yaw_zphys_y = float(torch.atan2(z_in_phys[0], z_in_phys[2]).item() * 180.0 / np.pi)

            print(f"[Debug] y-rot(R_rectify)={yaw_rectify_y:.2f} deg")
            print(f"[Debug] yaw_y(z_virt in phys)={yaw_zphys_y:.2f} deg")
        except Exception as e:
            print(f"[Debug] yaw diagnostics failed: {e}")

    # --- 4. 生成 IPM ---
    # 尝试加载卫星图，支持多种扩展名
    sat_available = False
    sat_img = np.zeros((512, 512, 3), dtype=np.uint8)
    sat_dir = seq_dir / "satellite"
    for ext in ['.jpg', '.jpeg', '.png']:
        sat_path = sat_dir / f"{frame_str}{ext}"
        if sat_path.exists():
            sat_img = cv2.imread(str(sat_path))
            if sat_img is not None:
                sat_img = cv2.resize(sat_img, (512, 512))
                sat_available = True
                print(f"[Debug] Loaded satellite image from: {sat_path}")
                break

    if not sat_available:
        print(f"[Warning] Satellite image not found for frame {frame_str}, using empty image")

    sat_tensor = torch.from_numpy(sat_img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # Debug: 打印 IMU 高度和地面高度
    imu_height = float(T_imu_to_world[2, 3].item())
    ground_z = imu_height - IMU_TO_GROUND_HEIGHT
    print(f"[Debug] T_imu_to_world z={imu_height:.3f} m")
    print(f"[Debug] ground_z={ground_z:.3f} m (IMU_TO_GROUND_HEIGHT={IMU_TO_GROUND_HEIGHT:.2f} m)")

    # Apply roll and pitch corrections for IPM
    if args.roll_ipm_deg != 0.0 or args.pitch_ipm_deg != 0.0:
        print(f"[IPM] Applying roll={args.roll_ipm_deg:.2f}°, pitch={args.pitch_ipm_deg:.2f}° correction")

        # Create rotation matrices for roll and pitch and ensure they match the device and dtype of T_cam_virt_to_world
        R_roll = torch.from_numpy(rot_z(args.roll_ipm_deg)).to(
            device=T_cam_virt_to_world.device,
            dtype=T_cam_virt_to_world.dtype
        )
        R_pitch = torch.from_numpy(rot_x(args.pitch_ipm_deg)).to(
            device=T_cam_virt_to_world.device,
            dtype=T_cam_virt_to_world.dtype
        )

        # Combine the rotations (order matters: roll first, then pitch)
        R_correction = R_pitch @ R_roll

        # Apply the correction to the virtual camera's rotation
        R_virt = T_cam_virt_to_world[:3, :3]
        R_virt_corrected = R_virt @ R_correction

        # Create a new transform with the corrected rotation
        T_cam_virt_corrected = T_cam_virt_to_world.clone()
        T_cam_virt_corrected[:3, :3] = R_virt_corrected
    else:
        # No correction needed, use the original transform
        T_cam_virt_corrected = T_cam_virt_to_world

    # Generate IPM with the corrected transform
    warped, _, _ = warp_bev_to_camera_with_coords(
        sat_image=sat_tensor,
        K=K_v,
        T_cam_to_world=T_cam_virt_corrected,
        T_imu_to_world=T_imu_to_world,
        cam_height=out_h,
        cam_width=out_w,
        sat_size=512,
        resolution=args.res,
        ground_height=ground_z,
    )
    ipm_img = (warped[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # --- 5. 存储结果 ---
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Generate BEV visualization with camera frustum
    bev_with_rays = visualize_camera_frustum_on_bev(
        sat_img=sat_img,
        K_virtual=K_v,
        K_ipm=K_v,  # Using the same K for simplicity
        T_cam_to_world=T_cam_virt_to_world,
        ground_z=ground_z,
        resolution=args.res,
        sat_size=512
    )

    # Create three-panel visualization
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    if is_front_view:
        title = f"IPM Alignment Debug (Front View) | drive={args.drive} frame={frame_str} cam={args.camera} res={args.res}"
    else:
        title = f"IPM Alignment Debug | drive={args.drive} frame={frame_str} cam={args.camera} yaw={args.yaw:.1f} res={args.res}"
    fig.suptitle(title, fontsize=12)

    # (1) BEV with rays
    axes[0].imshow(cv2.cvtColor(bev_with_rays, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(1) BEV with Frustum")
    axes[0].axis("off")

    # (2) Virtual perspective image (or front view image)
    axes[1].imshow(cv2.cvtColor(view_virtual, cv2.COLOR_BGR2RGB))
    if is_front_view:
        axes[1].set_title(f"(2) Front view {out_w}x{out_h}")
    else:
        axes[1].set_title(f"(2) Virtual view {out_w}x{out_h} hfov={args.hfov:.1f}")
    axes[1].axis("off")

    # (3) IPM result
    ipm_uint8 = (np.clip(ipm_img / 255.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    axes[2].imshow(cv2.cvtColor(ipm_uint8, cv2.COLOR_BGR2RGB), aspect="equal")
    axes[2].set_title("(3) IPM Result")
    axes[2].axis("off")

    plt.tight_layout()

    output_path = os.path.join(args.out_dir, f"debug_triplet_{args.camera}_{frame_str}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved visualization to: {output_path}")

    # Also save individual images
    cv2.imwrite(os.path.join(args.out_dir, f"debug1_bev_rays_{args.camera}_{frame_str}.png"), bev_with_rays)
    cv2.imwrite(os.path.join(args.out_dir, f"debug2_virtual_{args.camera}_{frame_str}.png"), view_virtual)
    cv2.imwrite(os.path.join(args.out_dir, f"debug3_ipm_{args.camera}_{frame_str}.png"), ipm_img)

if __name__ == "__main__":
    main()
