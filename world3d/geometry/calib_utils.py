import os
import numpy as np
import torch
from typing import Tuple, Optional


def _read_calib_file(path: str) -> dict:
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            k, v = line.strip().split(':', 1)
            v = v.strip()
            try:
                nums = [float(x) for x in v.split()] if v else []
            except Exception:
                nums = []
            data[k] = np.array(nums, dtype=np.float64)
    return data


def load_kitti_calib(calib_dir: str, cam: str = 'P2'):
    """
    Load KITTI calibration: returns K (3x3) from P rectified of given camera (P2 by default),
    and T_cam_lidar (4x4) as transform from LiDAR to camera coordinates.
    """
    cam2cam = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
    velo2cam = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
    if not os.path.isfile(cam2cam) or not os.path.isfile(velo2cam):
        raise FileNotFoundError(f"Expected KITTI calib files at {calib_dir}")
    c = _read_calib_file(cam2cam)
    v = _read_calib_file(velo2cam)

    key = cam
    if key not in c:
        # try P_rect_x
        if key == 'P2' and 'P_rect_02' in c:
            key = 'P_rect_02'
        elif key == 'P0' and 'P_rect_00' in c:
            key = 'P_rect_00'
        elif key == 'P1' and 'P_rect_01' in c:
            key = 'P_rect_01'
        elif key == 'P3' and 'P_rect_03' in c:
            key = 'P_rect_03'
    P = c[key].reshape(3, 4)
    K = P[:3, :3]

    # Tr_velo_to_cam (3x4)
    if 'Tr_velo_to_cam' in v:
        Tr = v['Tr_velo_to_cam'].reshape(3, 4)
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = Tr
    else:
        # Some variants store R and T separately
        R = v.get('R', np.eye(9, dtype=np.float64)).reshape(3, 3)
        t = v.get('T', np.zeros(3, dtype=np.float64)).reshape(3, 1)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3:4] = t

    K_t = torch.from_numpy(K.astype(np.float32))
    T_t = torch.from_numpy(T.astype(np.float32))
    return K_t, T_t


def adjust_intrinsics_after_crop_resize(K: torch.Tensor,
                                         orig_w: int, orig_h: int,
                                         crop_top: int, crop_bottom: int,
                                         out_w: int, out_h: int) -> torch.Tensor:
    """
    Given original intrinsics K for an image of size (orig_w, orig_h), we first crop rows
    [crop_top : orig_h - crop_bottom] and keep full width, then resize to (out_w, out_h).
    Returns adjusted K' (3x3).
    """
    assert K.shape == (3, 3)
    crop_h = orig_h - crop_top - crop_bottom
    Sx = out_w / float(orig_w)
    Sy = out_h / float(crop_h)
    Kp = K.clone()
    Kp[0, 0] *= Sx
    Kp[1, 1] *= Sy
    Kp[0, 2] = (K[0, 2] - 0.0) * Sx  # no horizontal crop
    Kp[1, 2] = (K[1, 2] - float(crop_top)) * Sy
    return Kp


def load_imu_to_velo_calib(calib_dir: str):
    """
    Load KITTI IMU to Velodyne calibration.

    Args:
        calib_dir: Path to calibration directory (e.g., '2011_09_26_calib')

    Returns:
        T_imu_to_velo: 4x4 torch.Tensor transform from IMU to Velodyne (Lidar) coordinates
    """
    imu2velo = os.path.join(calib_dir, 'calib_imu_to_velo.txt')
    if not os.path.isfile(imu2velo):
        raise FileNotFoundError(f"Expected calib_imu_to_velo.txt at {calib_dir}")

    data = _read_calib_file(imu2velo)

    # Read R (3x3) and T (3x1)
    R = data.get('R', np.eye(9, dtype=np.float64)).reshape(3, 3)
    t = data.get('T', np.zeros(3, dtype=np.float64)).reshape(3, 1)

    # Build 4x4 transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t

    T_t = torch.from_numpy(T.astype(np.float32))
    return T_t


def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float, int, str]:
    """
    Convert latitude/longitude to UTM coordinates.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        easting: UTM easting in meters
        northing: UTM northing in meters
        zone_number: UTM zone number
        zone_letter: UTM zone letter
    """
    try:
        import utm
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        return easting, northing, zone_number, zone_letter
    except ImportError:
        # Fallback: simple approximation (not accurate for large distances)
        # This is a simplified Mercator projection
        # For production, install utm: pip install utm
        import math

        # WGS84 parameters
        a = 6378137.0  # semi-major axis
        e = 0.0818191908426  # eccentricity

        # UTM zone
        zone_number = int((lon + 180) / 6) + 1

        # Central meridian
        lon0 = (zone_number - 1) * 6 - 180 + 3

        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        lon0_rad = math.radians(lon0)

        # UTM projection (simplified)
        N = a / math.sqrt(1 - e**2 * math.sin(lat_rad)**2)
        T = math.tan(lat_rad)**2
        C = (e**2 / (1 - e**2)) * math.cos(lat_rad)**2
        A = (lon_rad - lon0_rad) * math.cos(lat_rad)

        # Scale factor
        k0 = 0.9996

        # Easting
        easting = k0 * N * (A + (1 - T + C) * A**3 / 6) + 500000.0

        # Northing
        M = a * ((1 - e**2/4 - 3*e**4/64) * lat_rad)
        northing = k0 * M

        if lat < 0:
            northing += 10000000.0

        zone_letter = 'N' if lat >= 0 else 'S'

        return easting, northing, zone_number, zone_letter


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix.

    KITTI convention:
    - Roll (α): rotation around x-axis (forward)
    - Pitch (β): rotation around y-axis (left)
    - Yaw (γ): rotation around z-axis (up)

    Rotation order: Rz(yaw) * Ry(pitch) * Rx(roll)

    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians

    Returns:
        R: 3x3 rotation matrix (IMU → World/NED frame)
    """
    # Roll (rotation around x-axis)
    cr = np.cos(roll)
    sr = np.sin(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ], dtype=np.float64)

    # Pitch (rotation around y-axis)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ], dtype=np.float64)

    # Yaw (rotation around z-axis)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # Combined rotation: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    return R


def parse_oxts_line(line: str) -> dict:
    """
    Parse a single line from KITTI OXTS file.

    OXTS format (30 values):
    0-2: lat, lon, alt (GPS position)
    3-5: roll, pitch, yaw (IMU orientation in radians)
    6-8: vn, ve, vf (velocity north/east/forward in m/s)
    9-11: vl, vu, vw (velocity left/up/angular in m/s and rad/s)
    12-14: ax, ay, az (acceleration in m/s^2)
    15-17: af, al, au (acceleration forward/left/up in m/s^2)
    18-20: wx, wy, wz (angular rate in rad/s)
    21-23: wf, wl, wu (angular rate forward/left/up in rad/s)
    24-26: pos_accuracy, vel_accuracy, navstat
    27-29: numsats, posmode, velmode, orimode

    Returns:
        dict with parsed values
    """
    values = [float(x) for x in line.strip().split()]

    return {
        'lat': values[0],
        'lon': values[1],
        'alt': values[2],
        'roll': values[3],
        'pitch': values[4],
        'yaw': values[5],
    }


def load_oxts_pose(oxts_file: str, origin: Optional[Tuple[float, float, float]] = None) -> Tuple[torch.Tensor, dict]:
    """
    Load OXTS pose and compute T_imu_to_world transformation.

    Args:
        oxts_file: Path to OXTS .txt file
        origin: Optional (easting_0, northing_0, alt_0) origin for world frame.
                If None, uses the current frame as origin.

    Returns:
        T_imu_to_world: 4x4 transformation matrix (IMU → World)
        oxts_data: dict with parsed OXTS data
    """
    with open(oxts_file, 'r') as f:
        line = f.readline()

    oxts_data = parse_oxts_line(line)

    # Convert GPS to UTM
    easting, northing, zone_num, zone_letter = latlon_to_utm(oxts_data['lat'], oxts_data['lon'])

    # Set origin if not provided
    if origin is None:
        origin = (easting, northing, oxts_data['alt'])

    # Compute translation in world frame
    tx = easting - origin[0]
    ty = northing - origin[1]
    tz = oxts_data['alt'] - origin[2]

    # Compute rotation matrix from Euler angles
    R = euler_to_rotation_matrix(oxts_data['roll'], oxts_data['pitch'], oxts_data['yaw'])

    # Build 4x4 transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]

    T_t = torch.from_numpy(T.astype(np.float32))

    # Add UTM info to oxts_data
    oxts_data['easting'] = easting
    oxts_data['northing'] = northing
    oxts_data['zone_num'] = zone_num
    oxts_data['zone_letter'] = zone_letter
    oxts_data['origin'] = origin

    return T_t, oxts_data


def get_world_to_satellite_transform(
    sat_size: int = 512,
    resolution_m_per_px: float = 0.2,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Get transformation from World coordinates (UTM) to Satellite image coordinates.

    卫星图坐标系定义 (北向上):
    - 原点在图像中心 (sat_size/2, sat_size/2)
    - 图像行0 = 北边, 图像行511 = 南边
    - 图像列0 = 西边, 图像列511 = 东边
    - 卫星图是北向上的 (North-up)

    World坐标系 (UTM):
    - X轴向东 (Easting)
    - Y轴向北 (Northing)
    - 原点在第一帧的GPS位置

    坐标映射:
    - World +X (东) → Satellite +列 (右)
    - World +Y (北) → Satellite -行 (上)

    Args:
        sat_size: Satellite image size in pixels (default: 512)
        resolution_m_per_px: Satellite resolution in meters per pixel (default: 0.2)
        device: torch device

    Returns:
        T_world_to_sat: 4x4 transformation matrix (World → Satellite image coords)
    """
    # Scale factor: meters to pixels
    scale = 1.0 / resolution_m_per_px  # 5 pixels/meter

    # Translation: move origin to image center
    tx = sat_size / 2.0  # 256 pixels
    ty = sat_size / 2.0  # 256 pixels

    # Build transformation matrix
    # World (X=East, Y=North) → Satellite (col, row)
    #
    # 关键：卫星图是北向上的，所以：
    # - World +X (东) → Satellite +col (向右) ✓
    # - World +Y (北) → Satellite -row (向上，因为图像row从上到下递增)
    #
    # 因此：
    # sat_col = scale * world_x + tx
    # sat_row = -scale * world_y + ty  (注意负号！)
    #
    T = torch.eye(4, dtype=torch.float32, device=device)
    T[0, 0] = scale      # X: East (meters) → columns (pixels), 向右为正
    T[1, 1] = -scale     # Y: North (meters) → rows (pixels), 向上为负（因为row向下递增）
    T[0, 3] = tx         # Translate to image center X
    T[1, 3] = ty         # Translate to image center Y

    return T


def compose_camera_to_satellite_transform(
    T_cam_to_velo: torch.Tensor,
    T_velo_to_imu: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    sat_size: int = 512,
    resolution_m_per_px: float = 0.2
) -> torch.Tensor:
    """
    Compose full transformation from Camera to Satellite image coordinates.

    T_cam_to_sat = T_world_to_sat @ T_imu_to_world @ T_velo_to_imu @ T_cam_to_velo

    Args:
        T_cam_to_velo: 4x4 Camera → Lidar transform
        T_velo_to_imu: 4x4 Lidar → IMU transform
        T_imu_to_world: 4x4 IMU → World transform
        sat_size: Satellite image size in pixels
        resolution_m_per_px: Satellite resolution in m/pixel

    Returns:
        T_cam_to_sat: 4x4 Camera → Satellite image transform
        T_world_to_sat: 4x4 World → Satellite image transform (for reference)
    """
    device = T_cam_to_velo.device

    # Get World → Satellite transform
    T_world_to_sat = get_world_to_satellite_transform(sat_size, resolution_m_per_px, device)

    # Compose full chain: Camera → Lidar → IMU → World → Satellite
    T_cam_to_sat = T_world_to_sat @ T_imu_to_world @ T_velo_to_imu @ T_cam_to_velo

    return T_cam_to_sat, T_world_to_sat


def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE3 matrix."""
    assert T.shape[-2:] == (4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.t()
    t_inv = -R_inv @ t
    Tout = torch.eye(4, dtype=T.dtype, device=T.device)
    Tout[:3, :3] = R_inv
    Tout[:3, 3] = t_inv
    return Tout

