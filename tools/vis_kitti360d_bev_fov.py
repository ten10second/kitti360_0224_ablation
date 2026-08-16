import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure repo root is on PYTHONPATH
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from world3d.io.kitti360d_dataloader import Kitti360dDataset


def rotz(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def cam_forward_in_pose(T_pose_cam: np.ndarray) -> np.ndarray:
    """Return camera optical axis (z_cam) expressed in pose/world coords.

    In standard pinhole: x right, y down, z forward.
    So forward direction in cam is (0,0,1).
    """
    R = T_pose_cam[:3, :3]
    f = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normalize(f)


def fov_half_angle_from_K(K: np.ndarray, w: int) -> float:
    fx = float(K[0, 0])
    return math.atan((w / 2.0) / fx)


def oxts_yaw_to_north0_cw_deg(oxts_yaw_rad: float) -> float:
    """Match tools/integrated_fisheye_visualizer.py convention.

    integrated_fisheye_visualizer.py does:
        vehicle_yaw_deg = degrees(pi/2 - yaw_rad)
    and then uses a "north=0, CW+" angle convention.

    Returns angle in degrees in range [-180, 180].
    """
    deg = math.degrees(math.pi / 2.0 - float(oxts_yaw_rad))
    deg = (deg + 180.0) % 360.0 - 180.0
    return deg


def north0_cw_deg_to_unitvec_xy_east_north(angle_deg: float) -> np.ndarray:
    """Convert "north=0, CW+" angle to a unit vector in ENU (x=east, y=north)."""
    a = math.radians(float(angle_deg))
    # integrated script uses (North, East) = (cos, sin)
    north = math.cos(a)
    east = math.sin(a)
    return normalize(np.array([east, north, 0.0], dtype=np.float64))


def draw_fov_on_sat(
    sat_rgb: np.ndarray,
    m_per_px: float,
    cam_pos_xy_m: np.ndarray,
    heading_xy_enu: np.ndarray,
    K: np.ndarray,
    out_w: int,
    label: str,
    color: tuple,
):
    """Draw a camera FOV wedge on a north-up satellite patch.

    Inputs:
    - cam_pos_xy_m: camera position in meters in ENU (x=east, y=north)
    - heading_xy_enu: unit heading direction on ground in ENU (x=east, y=north)

    This follows the same yaw convention as tools/integrated_fisheye_visualizer.py.
    """
    H, W = sat_rgb.shape[:2]
    cx_px, cy_px = W // 2, H // 2

    f_xy = normalize(np.array([heading_xy_enu[0], heading_xy_enu[1], 0.0], dtype=np.float64))
    if np.linalg.norm(f_xy[:2]) < 1e-6:
        return sat_rgb

    half = fov_half_angle_from_K(K, out_w)

    R_left = rotz(+half)
    R_right = rotz(-half)
    d_left = normalize(R_left @ f_xy)
    d_right = normalize(R_right @ f_xy)

    L = 30.0

    def to_px(p_xy_m: np.ndarray):
        x_m, y_m = float(p_xy_m[0]), float(p_xy_m[1])
        x_px = cx_px + int(round(x_m / m_per_px))
        y_px = cy_px - int(round(y_m / m_per_px))
        return x_px, y_px

    p0 = np.array([float(cam_pos_xy_m[0]), float(cam_pos_xy_m[1])], dtype=np.float64)
    pL = p0 + L * d_left[:2]
    pR = p0 + L * d_right[:2]

    x0, y0 = to_px(p0)
    xL, yL = to_px(pL)
    xR, yR = to_px(pR)

    im = Image.fromarray(sat_rgb)
    draw = ImageDraw.Draw(im, "RGBA")

    draw.polygon([(x0, y0), (xL, yL), (xR, yR)], fill=(*color, 60))
    draw.line([(x0, y0), (xL, yL)], fill=(*color, 200), width=3)
    draw.line([(x0, y0), (xR, yR)], fill=(*color, 200), width=3)

    xc, yc = to_px(p0 + L * f_xy[:2])
    draw.line([(x0, y0), (xc, yc)], fill=(*color, 255), width=2)

    draw.text((x0 + 6, y0 + 6), label, fill=(*color, 255))

    return np.array(im)


def main():
    drive_dir = repo_root / "2013_05_28_drive_0003_sync"
    frame_id = 0

    # We will visualize FOVs for:
    # - front image_00
    # - virtual views A: [-60,-30,+30,+60] using image_02 for left and image_03 for right

    ds_front = Kitti360dDataset(drives=drive_dir, mode="front")
    s_front = ds_front[frame_id]

    sat = (s_front["sat"].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    m_per_px = float(s_front["sat_m_per_px"])

    overlays = sat.copy()

    oxts_yaw = float(s_front["meta"].get("oxts_yaw") or 0.0)

    # Convert oxts yaw (rad) to north=0, CW+ vehicle heading (deg)
    vehicle_yaw_deg = oxts_yaw_to_north0_cw_deg(oxts_yaw)

    # camera position (we assume BEV center is vehicle/IMU position)
    cam_pos_xy = np.array([0.0, 0.0], dtype=np.float64)

    # front heading in ENU (east,north)
    front_heading_xy = north0_cw_deg_to_unitvec_xy_east_north(vehicle_yaw_deg)

    # front
    if s_front["K"] is not None:
        overlays = draw_fov_on_sat(
            overlays,
            m_per_px,
            cam_pos_xy,
            front_heading_xy,
            s_front["K"].numpy(),
            out_w=int(s_front["image"].shape[-1]),
            label=f"front image_00 yaw={vehicle_yaw_deg:+.0f}",
            color=(0, 255, 0),
        )

    # Views A: [-60, -30, +30, +60] relative to vehicle front
    views = [
        ("image_02", -60.0, (255, 0, 0)),
        ("image_02", -40.0, (255, 80, 0)),
        ("image_03", +40.0, (0, 128, 255)),
        ("image_03", +60.0, (0, 0, 255)),
    ]

    for cam, rel_yaw_deg, color in views:
        # world heading = vehicle heading + relative yaw (CW+)
        yaw_world_deg = vehicle_yaw_deg + rel_yaw_deg
        heading_xy = north0_cw_deg_to_unitvec_xy_east_north(yaw_world_deg)

        ds_v = Kitti360dDataset(
            drives=drive_dir,
            mode="fisheye_virtual",
            fisheye_camera=cam,
            vehicle_relative_yaw_deg=rel_yaw_deg,
            virtual_hfov_deg=80.0,
            virtual_size=(640, 256),
        )
        s = ds_v[frame_id]
        if s["K"] is None:
            continue

        overlays = draw_fov_on_sat(
            overlays,
            m_per_px,
            cam_pos_xy,
            heading_xy,
            s["K"].numpy(),
            out_w=int(s["image"].shape[-1]),
            label=f"{cam} rel={rel_yaw_deg:+.0f}",
            color=color,
        )

    out_path = repo_root / "tools" / "_vis_kitti360d_sat_fov_frame0000000000.png"
    Image.fromarray(overlays).save(out_path)
    print("Saved BEV FOV overlay to:", out_path)


if __name__ == "__main__":
    main()

