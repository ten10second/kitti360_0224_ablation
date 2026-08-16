from __future__ import annotations

import torch


def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def build_pose_vec(
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    img_h: int,
    img_w: int,
    device: torch.device,
) -> torch.Tensor:
    if K.dim() == 3:
        K = K[0]
    if T_cam_to_world.dim() == 3:
        T_cam_to_world = T_cam_to_world[0]
    if T_imu_to_world.dim() == 3:
        T_imu_to_world = T_imu_to_world[0]

    R = T_cam_to_world[:3, :3]
    t_cam = T_cam_to_world[:3, 3]
    t_imu = T_imu_to_world[:3, 3]
    t_rel = t_cam - t_imu

    rot6d = rotmat_to_6d(R)

    fx = K[0, 0] / float(img_w)
    fy = K[1, 1] / float(img_h)
    cx = K[0, 2] / float(img_w)
    cy = K[1, 2] / float(img_h)
    intr = torch.tensor([fx, fy, cx, cy], device=device, dtype=torch.float32)

    pose_vec = torch.cat([rot6d.to(device).float(), t_rel.to(device).float(), intr], dim=0)
    return pose_vec

