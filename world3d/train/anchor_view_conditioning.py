"""Modules for anchor-view conditioned generation.

实现将 anchor 视角的图像/特征 warp 到 target 视角的功能。
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


IMU_TO_GROUND_HEIGHT = 0.93


def warp_camera_to_camera(
    anchor_image: torch.Tensor,
    anchor_K: torch.Tensor,
    anchor_T_cam_to_world: torch.Tensor,
    anchor_T_imu_to_world: torch.Tensor,
    target_K: torch.Tensor,
    target_T_cam_to_world: torch.Tensor,
    target_T_imu_to_world: torch.Tensor,
    cam_h: int = 256,
    cam_w: int = 640,
    bev_size: int = 512,
    bev_resolution: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 anchor 视角的图像 warp 到 target 视角的相机平面。

    通过 BEV 作为中间桥梁：
    1. anchor 相机像素 → BEV 世界坐标
    2. BEV 世界坐标 → target 相机像素

    Args:
        anchor_image: (B, C, H, W) - anchor 视角图像
        anchor_K: (B, 3, 3) or (3, 3) - anchor 相机内参
        anchor_T_cam_to_world: (B, 4, 4) or (4, 4) - anchor 到世界变换
        anchor_T_imu_to_world: (B, 4, 4) or (4, 4) - anchor IMU 到世界变换
        target_K: (B, 3, 3) or (3, 3) - target 相机内参
        target_T_cam_to_world: (B, 4, 4) or (4, 4) - target 到世界变换
        target_T_imu_to_world: (B, 4, 4) or (4, 4) - target IMU 到世界变换
        cam_h: int - 输出图像高度
        cam_w: int - 输出图像宽度
        bev_size: int - BEV 尺寸
        bev_resolution: float - BEV 分辨率 (m/pixel)

    Returns:
        warped_image: (B, C, cam_h, cam_w) - warp 后的图像
        valid_mask: (B, 1, cam_h, cam_w) - 有效像素掩码
    """
    B, C, H_anc, W_anc = anchor_image.shape
    device = anchor_image.device

    # 确保输入批次形状一致
    if anchor_K.dim() == 2:
        anchor_K = anchor_K.unsqueeze(0).expand(B, 3, 3)
    if target_K.dim() == 2:
        target_K = target_K.unsqueeze(0).expand(B, 3, 3)
    if anchor_T_cam_to_world.dim() == 2:
        anchor_T_cam_to_world = anchor_T_cam_to_world.unsqueeze(0).expand(B, 4, 4)
    if target_T_cam_to_world.dim() == 2:
        target_T_cam_to_world = target_T_cam_to_world.unsqueeze(0).expand(B, 4, 4)
    if anchor_T_imu_to_world.dim() == 2:
        anchor_T_imu_to_world = anchor_T_imu_to_world.unsqueeze(0).expand(B, 4, 4)
    if target_T_imu_to_world.dim() == 2:
        target_T_imu_to_world = target_T_imu_to_world.unsqueeze(0).expand(B, 4, 4)

    # 提取 anchor 变换
    R_anchor = anchor_T_cam_to_world[:, :3, :3]
    t_anchor = anchor_T_cam_to_world[:, :3, 3:4]

    # 提取 target 变换
    R_target = target_T_cam_to_world[:, :3, :3]
    t_target = target_T_cam_to_world[:, :3, 3:4]

    # 目标图像的像素网格
    v_target = torch.arange(cam_h, dtype=torch.float32, device=device)
    u_target = torch.arange(cam_w, dtype=torch.float32, device=device)
    vv_target, uu_target = torch.meshgrid(v_target, u_target, indexing='ij')  # (cam_h, cam_w)

    # 归一化像素坐标
    target_pixels_homo = torch.stack([
        uu_target.reshape(-1),
        vv_target.reshape(-1),
        torch.ones_like(uu_target.reshape(-1)),
    ], dim=0).unsqueeze(0).expand(B, 3, -1)  # (B, 3, cam_h*cam_w)

    # target 相机射线 → world 坐标
    K_target_inv = torch.inverse(target_K)
    rays_target = torch.bmm(K_target_inv, target_pixels_homo)  # (B, 3, N)
    rays_world = torch.bmm(R_target, rays_target)  # (B, 3, N)

    # 与地面相交 (Z = ground_height)
    # 使用 target IMU 高度估计地面高度
    t_imu_target = target_T_imu_to_world[:, :3, 3]  # (B, 3)
    ground_height = (t_imu_target[:, 2:3] - IMU_TO_GROUND_HEIGHT).unsqueeze(-1)  # (B, 1, 1)

    rays_z = rays_world[:, 2:3, :]
    t = torch.where(
        rays_z.abs() > 1e-6,
        (ground_height - t_target[:, 2:3, :]) / rays_z,
        torch.full_like(rays_z, -1.0),
    )  # (B, 1, N)

    # 世界坐标交点
    points_world = t_target + t * rays_world  # (B, 3, N)

    # 世界坐标 → anchor 相机坐标
    T_world_to_anchor = torch.inverse(anchor_T_cam_to_world)  # (B, 4, 4)
    points_world_homo = torch.cat([points_world, torch.ones_like(points_world[:, :1, :])], dim=1)  # (B, 4, N)
    points_anchor = torch.bmm(T_world_to_anchor, points_world_homo)[:, :3, :]  # (B, 3, N)

    # 投影到 anchor 相机图像
    x_anchor = points_anchor[:, 0, :]
    y_anchor = points_anchor[:, 1, :]
    z_anchor = points_anchor[:, 2, :]

    # 有效性：必须在 anchor 相机前方
    valid_mask = (z_anchor > 0.1)

    # 投影到像素
    u_anchor_proj = anchor_K[:, 0, 0:1] * x_anchor / z_anchor + anchor_K[:, 0, 2:3]
    v_anchor_proj = anchor_K[:, 1, 1:2] * y_anchor / z_anchor + anchor_K[:, 1, 2:3]

    # 检查是否在 anchor 图像范围内
    valid_mask = valid_mask & (u_anchor_proj >= 0) & (u_anchor_proj < W_anc)
    valid_mask = valid_mask & (v_anchor_proj >= 0) & (v_anchor_proj < H_anc)

    # 归一化到 [-1, 1] 用于 grid_sample
    u_norm = 2.0 * u_anchor_proj / (W_anc - 1) - 1.0
    v_norm = 2.0 * v_anchor_proj / (H_anc - 1) - 1.0

    # 对于无效点，设置为超出范围
    u_norm = torch.where(valid_mask, u_norm, torch.full_like(u_norm, -2.0))
    v_norm = torch.where(valid_mask, v_norm, torch.full_like(v_norm, -2.0))

    # 重塑为 (B, cam_h, cam_w, 2)
    u_norm = u_norm.reshape(B, cam_h, cam_w)
    v_norm = v_norm.reshape(B, cam_h, cam_w)
    grid = torch.stack([u_norm, v_norm], dim=-1)

    # 从 anchor 图像采样
    warped_image = F.grid_sample(
        anchor_image,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )

    # 返回有效掩码
    valid_mask = valid_mask.reshape(B, 1, cam_h, cam_w).float()

    return warped_image, valid_mask


class AnchorViewConditioner(nn.Module):
    """
    将 anchor 视角的特征转换为 target 视角的条件输入。

    核心功能：
    1. 通过相机到相机 warp 将 anchor 特征/图像对齐到 target 视角
    2. 融合几何差分（pose delta）和 warped 特征
    3. 输出条件张量供 target 生成使用
    """

    def __init__(
        self,
        feature_channels: int = 512,
        image_channels: int = 4,  # RGB + mask
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.image_channels = image_channels

        # Pose delta 编码器：编码 anchor 到 target 的相对位姿
        self.pose_delta_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 6D rotation + translation
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Anchor 图像投影网络（处理 warped 图像）
        self.anchor_image_proj = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, feature_channels, kernel_size=1),
        )

        # Anchor 特征投影网络（如果直接使用特征）
        self.anchor_feat_proj = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, feature_channels, kernel_size=1),
        )

        # 融合网络：融合 warped 特征和 pose delta
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_channels + hidden_dim, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, feature_channels),  # GroupNorm(1, C) 等效于 LayerNorm over spatial dims for NCHW
            nn.ReLU(inplace=False),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
        )

    def compute_pose_delta(
        self,
        anchor_T_cam_to_world: torch.Tensor,
        target_T_cam_to_world: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 anchor 到 target 的 6D 相对位姿。

        Args:
            anchor_T_cam_to_world: (B, 4, 4) or (4, 4)
            target_T_cam_to_world: (B, 4, 4) or (4, 4)

        Returns:
            pose_delta: (B, 6) - [rx, ry, rz, tx, ty, tz]
        """
        # 确保批次形状
        if anchor_T_cam_to_world.dim() == 2:
            anchor_T_cam_to_world = anchor_T_cam_to_world.unsqueeze(0)
        if target_T_cam_to_world.dim() == 2:
            target_T_cam_to_world = target_T_cam_to_world.unsqueeze(0)

        B = anchor_T_cam_to_world.shape[0]

        # T_anchor_to_target = T_world_to_target @ T_anchor_to_world
        T_world_to_target = torch.inverse(target_T_cam_to_world)
        T_anchor_to_target = torch.bmm(T_world_to_target, anchor_T_cam_to_world)

        # 提取旋转和平移
        R = T_anchor_to_target[:, :3, :3]  # (B, 3, 3)
        t = T_anchor_to_target[:, :3, 3]    # (B, 3)

        # 将旋转矩阵转换为轴角表示
        trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)  # (B,)
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

        # 防止 theta=0 的数值问题
        safe_theta = torch.where(theta < 1e-6, torch.ones_like(theta), theta)
        factor = 1 / (2 * torch.sin(safe_theta))

        rx = factor * (R[:, 2, 1] - R[:, 1, 2]) * safe_theta
        ry = factor * (R[:, 0, 2] - R[:, 2, 0]) * safe_theta
        rz = factor * (R[:, 1, 0] - R[:, 0, 1]) * safe_theta

        # 拼接旋转和平移
        pose_delta = torch.stack([rx, ry, rz, t[:, 0], t[:, 1], t[:, 2]], dim=1)  # (B, 6)

        return pose_delta

    def forward(
        self,
        anchor_image: torch.Tensor,
        anchor_T_cam_to_world: torch.Tensor,
        anchor_T_imu_to_world: torch.Tensor,
        anchor_K: torch.Tensor,
        target_T_cam_to_world: torch.Tensor,
        target_T_imu_to_world: torch.Tensor,
        target_K: torch.Tensor,
        anchor_feat: Optional[torch.Tensor] = None,
        target_h: int = 256,
        target_w: int = 640,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 anchor 视角转换为 target 视角的条件输入。

        Args:
            anchor_image: (B, C, H, W) - anchor 视角图像（RGB+mask）
            anchor_T_cam_to_world: (B, 4, 4) - anchor 变换
            anchor_T_imu_to_world: (B, 4, 4) - anchor IMU 变换
            anchor_K: (B, 3, 3) - anchor 内参
            target_T_cam_to_world: (B, 4, 4) - target 变换
            target_T_imu_to_world: (B, 4, 4) - target IMU 变换
            target_K: (B, 3, 3) - target 内参
            anchor_feat: (B, D, H, W) - anchor 中间特征（可选）
            target_h: int - target 图像高度
            target_w: int - target 图像宽度

        Returns:
            condition: (B, D, target_h, target_w) - 条件特征
            valid_mask: (B, 1, target_h, target_w) - 有效掩码
        """
        B = anchor_image.shape[0]
        device = anchor_image.device

        # 1. Warp anchor 图像到 target 视角
        warped_image, valid_mask = warp_camera_to_camera(
            anchor_image=anchor_image,
            anchor_K=anchor_K,
            anchor_T_cam_to_world=anchor_T_cam_to_world,
            anchor_T_imu_to_world=anchor_T_imu_to_world,
            target_K=target_K,
            target_T_cam_to_world=target_T_cam_to_world,
            target_T_imu_to_world=target_T_imu_to_world,
            cam_h=target_h,
            cam_w=target_w,
        )

        # 2. 编码 pose delta
        pose_delta = self.compute_pose_delta(
            anchor_T_cam_to_world, target_T_cam_to_world
        )  # (B, 6)

        # 将 pose delta 扩展为空间图
        pose_delta_feat = self.pose_delta_encoder(pose_delta).contiguous().clone()  # (B, hidden_dim)
        pose_delta_map = pose_delta_feat.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, target_h, target_w
        ).contiguous()  # (B, hidden_dim, H, W)

        # 3. 处理 warped 图像
        warped_condition = self.anchor_image_proj(warped_image)  # (B, feature_channels, H, W)

        # 4. 融合 warped 特征和 pose delta
        combined = torch.cat([warped_condition, pose_delta_map], dim=1).contiguous()  # (B, feat_dim + hidden_dim, H, W)
        condition = self.fusion(combined)  # (B, feature_channels, H, W)

        return condition, valid_mask
