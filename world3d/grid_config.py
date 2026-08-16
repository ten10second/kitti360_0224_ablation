#!/usr/bin/env python3
from dataclasses import dataclass

@dataclass
class GridConfig:
    # Vehicle-centric XY in meters (KITTI velodyne frame: x forward, y left, z up)
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    # Ground-relative height z' = z + lidar_height
    z_min: float = 0.0
    z_max: float = 6.0
    # Grid resolution
    nx: int = 64  # X bins (forward/back)
    ny: int = 64  # Y bins (left/right)
    nz: int = 16  # Z' bins (height)
    # Sensor mounting height in meters
    lidar_height: float = 1.73

    def voxel_sizes(self):
        sx = (self.x_max - self.x_min) / self.nx
        sy = (self.y_max - self.y_min) / self.ny
        sz = (self.z_max - self.z_min) / self.nz
        return sx, sy, sz

