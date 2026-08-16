"""
Geometry utilities for coordinate transformations and calibration loading.

(Removed in the ICASSP27 refactor: bev_to_camera_warp, camera_to_sat_projection,
camera_to_camera_ground, differentiable_projection, pose_loss, homography —
all ground-plane BEV warp machinery of the deleted MM26 path.)
"""

from .kitti_transforms import (
    load_kitti_calib,
    load_imu_to_velo_calib,
    load_oxts_pose,
    get_world_to_satellite_transform,
    compose_camera_to_satellite_transform,
    invert_se3,
    latlon_to_utm,
    euler_to_rotation_matrix,
    parse_oxts_line,
)

from .pose_encoding import (
    PoseEncoder,
    rotation_matrix_to_6d,
    rotation_matrix_to_quaternion,
)

__all__ = [
    # KITTI transforms
    'load_kitti_calib',
    'load_imu_to_velo_calib',
    'load_oxts_pose',
    'get_world_to_satellite_transform',
    'compose_camera_to_satellite_transform',
    'invert_se3',
    'latlon_to_utm',
    'euler_to_rotation_matrix',
    'parse_oxts_line',
    # Pose encoding
    'PoseEncoder',
    'rotation_matrix_to_6d',
    'rotation_matrix_to_quaternion',
]
