#!/usr/bin/env python3
"""
Utilities to align the 3D BEV grid (vehicle-centric) with the KITTI-360
satellite image (512x512, 0.196 m/pixel, centered at vehicle GPS).

Assumptions:
- Satellite is north-up (y_north increases toward image top), pixel (cx,cy) at image center.
- Vehicle-centric frame (KITTI velodyne): x forward, y left (west), z up.
- We need vehicle heading (yaw) in radians in ENU convention: 0 along +east,
  increasing counter-clockwise toward +north.
"""
from dataclasses import dataclass
import numpy as np
from typing import Tuple

@dataclass
class SatSpec:
    width: int = 512
    height: int = 512
    meters_per_pixel: float = 0.196
    cx: float = 256.0
    cy: float = 256.0


def veh_to_enu(x_fwd: float, y_left: float, yaw_rad: float) -> Tuple[float, float]:
    # Rotate vehicle frame (x_fwd,y_left) to ENU (east,north)
    c = np.cos(yaw_rad); s = np.sin(yaw_rad)
    east =  c * x_fwd - s * y_left
    north = s * x_fwd + c * y_left
    return float(east), float(north)


def enu_to_sat_px(east: float, north: float, spec: SatSpec) -> Tuple[float, float]:
    # Satellite x right = +east; image y down = +, so invert north
    u = spec.cx + east / spec.meters_per_pixel
    v = spec.cy - north / spec.meters_per_pixel
    return float(u), float(v)


def grid_cell_center_xy(i_y: int, i_x: int, x_min: float, x_max: float, y_min: float, y_max: float,
                        ny: int, nx: int) -> Tuple[float, float]:
    # i_y in [0,ny), i_x in [0,nx). Return vehicle-frame (x,y) at cell center
    sx = (x_max - x_min) / nx
    sy = (y_max - y_min) / ny
    x = x_min + (i_x + 0.5) * sx
    y = y_min + (i_y + 0.5) * sy
    return float(x), float(y)


def bev_grid_to_sat_indices(nx: int, ny: int, x_min: float, x_max: float, y_min: float, y_max: float,
                            yaw_rad: float, spec: SatSpec) -> np.ndarray:
    """Compute satellite pixel centers (u,v) for each BEV cell center.
    Returns array of shape (ny, nx, 2) with float pixel coords.
    """
    out = np.zeros((ny, nx, 2), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            x, y = grid_cell_center_xy(iy, ix, x_min, x_max, y_min, y_max, ny, nx)
            east, north = veh_to_enu(x, y, yaw_rad)
            u, v = enu_to_sat_px(east, north, spec)
            out[iy, ix, 0] = u
            out[iy, ix, 1] = v
    return out
