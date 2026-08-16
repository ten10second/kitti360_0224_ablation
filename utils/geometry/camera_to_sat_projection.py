#!/usr/bin/env python3
"""
相机图到卫星图的逆向投影

使用射线与地面平面相交的方法，将相机图像素投影到卫星图。

投影链：
1. 相机像素 (u_cam, v_cam) → 3D 射线（相机坐标系）
2. 射线 → 世界坐标系
3. 射线与地面平面 (Z=0) 相交 → 3D 点（世界坐标系）
4. 3D 点 → 卫星图像素 (u_sat, v_sat)
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def project_camera_to_satellite(
    cam_image: torch.Tensor,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    sat_size: int = 512,
    resolution: float = 0.2,
    ground_height: float = 0.0,
    use_splatting: bool = True,
    splat_radius: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将相机图投影到卫星图（逆向投影）

    投影链：
    1. 相机像素 → 3D 射线（相机坐标系）
    2. 射线 → 世界坐标系
    3. 射线与地面平面相交 → 3D 点
    4. 3D 点 → 卫星图像素

    Args:
        cam_image: (B, C, H_cam, W_cam) - 相机图像
        K: (3, 3) 或 (B, 3, 3) - 相机内参矩阵
        T_cam_to_world: (4, 4) 或 (B, 4, 4) - 相机到世界的变换矩阵
        sat_size: int - 卫星图尺寸（默认 512）
        resolution: float - 卫星图分辨率 m/pixel（默认 0.2）
        ground_height: float - 地面高度（默认 0.0）
        use_splatting: bool - 是否使用 splatting 来填充空洞（默认 True）
        splat_radius: float - Splatting 半径（默认 2.0）

    Returns:
        sat_image: (B, C, sat_size, sat_size) - 投影后的卫星图
        valid_mask: (B, 1, sat_size, sat_size) - 有效像素掩码

    注意：
    - 卫星图坐标系：中心在图像中心，X 向东（右），Y 向北（上）
    - 卫星图像素坐标：(0, 0) 在左上角
    - 世界坐标系：X 向东，Y 向北，Z 向上
    """
    B, C, H_cam, W_cam = cam_image.shape
    device = cam_image.device

    # 处理批次维度
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, 3, 3)
    if T_cam_to_world.dim() == 2:
        T_cam_to_world = T_cam_to_world.unsqueeze(0).expand(B, 4, 4)

    # 提取旋转和平移
    R_cam_to_world = T_cam_to_world[:, :3, :3]  # (B, 3, 3)
    t_cam_to_world = T_cam_to_world[:, :3, 3:4]  # (B, 3, 1)

    # 相机内参的逆
    K_inv = torch.inverse(K)  # (B, 3, 3)

    # 创建相机图的像素网格
    v_cam, u_cam = torch.meshgrid(
        torch.arange(H_cam, device=device, dtype=torch.float32),
        torch.arange(W_cam, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # 齐次坐标 (3, H_cam*W_cam)
    pixels_homo = torch.stack([
        u_cam.reshape(-1),
        v_cam.reshape(-1),
        torch.ones_like(u_cam.reshape(-1)),
    ], dim=0).unsqueeze(0).expand(B, 3, -1)  # (B, 3, H_cam*W_cam)

    # 相机坐标系中的射线方向
    rays_cam = torch.bmm(K_inv, pixels_homo)  # (B, 3, H_cam*W_cam)

    # 转换到世界坐标系
    rays_world = torch.bmm(R_cam_to_world, rays_cam)  # (B, 3, H_cam*W_cam)

    # 与地面平面相交 (Z = ground_height)
    # P_world = t_cam_to_world + t * rays_world
    # Z = t_cam_to_world[2] + t * rays_world[2] = ground_height
    # t = (ground_height - t_cam_to_world[2]) / rays_world[2]
    rays_z = rays_world[:, 2:3, :]  # (B, 1, H_cam*W_cam)
    t = torch.where(
        rays_z.abs() > 1e-6,
        (ground_height - t_cam_to_world[:, 2:3, :]) / rays_z,
        torch.full_like(rays_z, -1.0),  # 平行于地面的射线，设置为无效
    )  # (B, 1, H_cam*W_cam)

    # 计算交点（世界坐标系）
    points_world = t_cam_to_world + t * rays_world  # (B, 3, H_cam*W_cam)

    # 世界坐标 → 卫星图像素
    X_world = points_world[:, 0, :]  # (B, H_cam*W_cam) - 东向
    Y_world = points_world[:, 1, :]  # (B, H_cam*W_cam) - 北向

    # 卫星图中心对齐到相机在世界坐标的平面投影位置（与 warp_bev_to_camera 一致）
    sat_center = t_cam_to_world[:, :2, 0]  # (B, 2)
    X_off = X_world - sat_center[:, 0:1]
    Y_off = Y_world - sat_center[:, 1:2]

    # 卫星图像素坐标：u = sat_size/2 + X_off/resolution
    #               v = sat_size/2 - Y_off/resolution (注意：v 向下增加，Y 向北增加)
    u_sat = sat_size / 2 + X_off / resolution  # (B, H_cam*W_cam)
    v_sat = sat_size / 2 - Y_off / resolution  # (B, H_cam*W_cam)

    # 创建有效性掩码
    valid_mask = (t > 0) & \
                 (u_sat >= 0) & (u_sat < sat_size) & \
                 (v_sat >= 0) & (v_sat < sat_size)  # (B, H_cam*W_cam)

    # 归一化到 [-1, 1] 用于 grid_sample
    u_sat_norm = 2.0 * u_sat / (sat_size - 1) - 1.0
    v_sat_norm = 2.0 * v_sat / (sat_size - 1) - 1.0

    # 创建采样网格 (B, H_cam, W_cam, 2)
    grid = torch.stack([u_sat_norm, v_sat_norm], dim=-1).reshape(B, H_cam, W_cam, 2)

    # 使用 grid_sample 从相机图采样到卫星图
    sat_image = torch.zeros(B, C, sat_size, sat_size, device=device)
    valid_mask_sat = torch.zeros(B, 1, sat_size, sat_size, device=device)

    # 将相机图像素投影到卫星图
    valid_mask_reshaped = valid_mask.reshape(B, H_cam * W_cam)
    u_sat_reshaped = u_sat.reshape(B, H_cam * W_cam)
    v_sat_reshaped = v_sat.reshape(B, H_cam * W_cam)

    if use_splatting:
        # 快速矢量化的双线性 splatting，将每个有效相机像素按双线性权重分配到其在卫星图上的 4 个邻居
        for b in range(B):
            valid_b = valid_mask_reshaped[b]  # (H_cam*W_cam,)
            if valid_b.sum() == 0:
                continue
            u_b = u_sat_reshaped[b, valid_b]  # (N_valid,)
            v_b = v_sat_reshaped[b, valid_b]  # (N_valid,)
            colors_b = cam_image[b].reshape(C, -1)[:, valid_b]  # (C, N_valid)

            # 4 邻居坐标
            x0 = torch.floor(u_b)
            y0 = torch.floor(v_b)
            x1 = x0 + 1
            y1 = y0 + 1

            # 权重（双线性）
            wx1 = u_b - x0
            wy1 = v_b - y0
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1
            w00 = (wx0 * wy0).clamp(min=0.0)
            w01 = (wx0 * wy1).clamp(min=0.0)
            w10 = (wx1 * wy0).clamp(min=0.0)
            w11 = (wx1 * wy1).clamp(min=0.0)

            # 邻居索引并裁剪到边界
            def clamp_xy(x, y):
                x = x.long().clamp_(0, sat_size - 1)
                y = y.long().clamp_(0, sat_size - 1)
                return x, y

            x0i, y0i = clamp_xy(x0, y0)
            x0i_y1i, y1i = clamp_xy(x0, y1)
            x1i, y0i2 = clamp_xy(x1, y0)
            x1i2, y1i2 = clamp_xy(x1, y1)

            # 展平后的索引
            idx00 = (y0i * sat_size + x0i)
            idx01 = (y1i * sat_size + x0i_y1i)
            idx10 = (y0i2 * sat_size + x1i)
            idx11 = (y1i2 * sat_size + x1i2)

            # 展平后的目标缓冲区
            sat_flat = sat_image[b].reshape(C, -1)  # (C, H*W)
            wflat = valid_mask_sat[b, 0].reshape(-1)  # (H*W)




            # 将颜色按权重累加到 4 个邻居
            def add_neighbor(idx, w):
                if idx.numel() == 0:
                    return
                vals = colors_b * w.unsqueeze(0)  # (C, N_valid)
                sat_flat.index_add_(1, idx, vals)
                wflat.index_add_(0, idx, w)

            add_neighbor(idx00, w00)
            add_neighbor(idx01, w01)
            add_neighbor(idx10, w10)
            add_neighbor(idx11, w11)

            # 归一化
            mask = valid_mask_sat[b, 0] > 0
            sat_image[b, :, mask] /= valid_mask_sat[b, 0, mask]
            valid_mask_sat[b, 0, mask] = 1.0
    else:
        # 使用最近邻（原始方法）
        for b in range(B):
            valid_b = valid_mask_reshaped[b]  # (H_cam*W_cam,)
            if valid_b.sum() == 0:
                continue
            u_sat_b = u_sat_reshaped[b, valid_b].long()  # (N_valid,)
            v_sat_b = v_sat_reshaped[b, valid_b].long()  # (N_valid,)

            # 获取有效的相机像素坐标
            cam_pixels_flat = cam_image[b].reshape(C, -1)[:, valid_b]  # (C, N_valid)

            # 填充到卫星图（最近邻）
            sat_image[b, :, v_sat_b, u_sat_b] = cam_pixels_flat
            valid_mask_sat[b, 0, v_sat_b, u_sat_b] = 1.0

    return sat_image, valid_mask_sat




def project_camera_to_satellite_pull(
    cam_image: torch.Tensor,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    sat_size: int = 512,
    resolution: float = 0.2,
    ground_height: float = 0.0,
    padding_mode: str = 'zeros',
    center_world: torch.Tensor = None,
):
    """
    Pull-based dense projection (no striping): for each satellite pixel, compute its
    corresponding camera pixel and sample with grid_sample.

    Args:
        cam_image: (B, C, Hc, Wc)
        K: (3,3) or (B,3,3)
        T_cam_to_world: (4,4) or (B,4,4)
        sat_size: int, output satellite size
        resolution: float, meters per pixel on satellite
        ground_height: float, world Z of the ground plane
        padding_mode: str, passed to grid_sample
        center_world: (3,) or None, optional world XYZ position to center BEV at.
                      If None, defaults to camera XY position (original behavior).

    Returns:
        sat_image: (B, C, sat_size, sat_size)
        valid_mask: (B, 1, sat_size, sat_size)
    """
    B, C, Hc, Wc = cam_image.shape
    device = cam_image.device

    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, 3, 3)
    if T_cam_to_world.dim() == 2:
        T_cam_to_world = T_cam_to_world.unsqueeze(0).expand(B, 4, 4)

    Rcw = T_cam_to_world[:, :3, :3]          # (B,3,3)
    tcw = T_cam_to_world[:, :3, 3:4]         # (B,3,1)

    # Satellite center: use provided center_world if given, else camera XY position
    if center_world is not None:
        # center_world is (3,) tensor with XYZ world coordinates
        sat_center = center_world[:2].unsqueeze(0).expand(B, 2)  # (B,2)
    else:
        # Default: align to camera XY position (consistent with warp_bev_to_camera)
        sat_center = tcw[:, :2, 0]               # (B,2)

    # Build satellite grid (B, S, S)
    v_sat = torch.arange(sat_size, device=device, dtype=torch.float32)
    u_sat = torch.arange(sat_size, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v_sat, u_sat, indexing='ij')  # (S,S)
    uu = uu.unsqueeze(0).expand(B, -1, -1)
    vv = vv.unsqueeze(0).expand(B, -1, -1)

    # Satellite pixels -> world XY at Z=ground_height
    Xw = sat_center[:, 0:1, None] + (uu - (sat_size / 2.0)) * resolution
    Yw = sat_center[:, 1:2, None] - (vv - (sat_size / 2.0)) * resolution
    Zw = torch.full_like(Xw, ground_height)

    # (B,3,S,S) -> (B,3,S*S)
    Xw3 = torch.stack([Xw, Yw, Zw], dim=1).reshape(B, 3, -1)

    # World -> camera: Xc = R^T (Xw - t)
    Xc = torch.bmm(Rcw.transpose(1, 2), (Xw3 - tcw))  # (B,3,S*S)
    z = Xc[:, 2:3, :]                                  # (B,1,S*S)

    # Project to camera pixels
    Xc_norm = Xc / torch.clamp(z, min=1e-6)
    uv = torch.bmm(K, Xc_norm)                         # (B,3,S*S)
    u = uv[:, 0, :]                                    # (B,S*S)
    v = uv[:, 1, :]

    # Valid mask: in front and inside camera image
    valid = (z[:, 0, :] > 0) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)

    # Normalize to [-1,1] for grid_sample
    u_n = 2.0 * u / (Wc - 1) - 1.0
    v_n = 2.0 * v / (Hc - 1) - 1.0

    # Push invalid samples out of range
    u_n = torch.where(valid, u_n, torch.full_like(u_n, -2.0))
    v_n = torch.where(valid, v_n, torch.full_like(v_n, -2.0))

    grid = torch.stack([u_n, v_n], dim=-1).reshape(B, sat_size, sat_size, 2)

    sat_image = F.grid_sample(
        cam_image, grid, mode='bilinear', padding_mode=padding_mode, align_corners=True
    )
    valid_mask = valid.reshape(B, 1, sat_size, sat_size).float()
    return sat_image, valid_mask



def compute_camera_to_sat_grid_norm(
    H_cam: int,
    W_cam: int,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    T_imu_to_world: torch.Tensor, # ADDED: Must be provided for correct centering
    sat_size: int = 512,
    resolution: float = 0.2,
    ground_height: float = 0.0,
    invalid_value: float = -2.0,
):
    """
    Compute per-pixel normalized coordinates on the satellite/BEV image given camera intrinsics and pose.

    This extracts only the geometric coordinates from project_camera_to_satellite:
      - For each camera pixel (u,v), cast a ray, intersect Z=ground_height to get (Xw,Yw)
      - Convert to satellite pixel coords (u_sat, v_sat)
      - Normalize to [-1,1] w.r.t. sat_size

    Args:
        H_cam, W_cam: target camera image size (after any resize)
        K: (3,3) or (B,3,3) camera intrinsics (corresponding to H_cam,W_cam)
        T_cam_to_world: (4,4) or (B,4,4)
        sat_size: satellite image size (square)
        resolution: meters per pixel on satellite
        ground_height: Z of ground plane in world coords
        invalid_value: value to use for invalid (u,v)

    Returns:
        grid_norm: (B, H_cam, W_cam, 2) in [-1,1], invalid set to invalid_value
        valid_mask: (B, H_cam, W_cam) bool
    """
    device = K.device
    T_cam_to_world = T_cam_to_world.to(device)
    T_imu_to_world = T_imu_to_world.to(device)

    # Batch-ify K and T
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if T_cam_to_world.dim() == 2:
        T_cam_to_world = T_cam_to_world.unsqueeze(0)
    if T_imu_to_world.dim() == 2:
        T_imu_to_world = T_imu_to_world.unsqueeze(0)
    B = K.shape[0]

    # Inverse intrinsics
    K_inv = torch.inverse(K)  # (B,3,3)

    # Pixel grid (v,u)
    v_cam, u_cam = torch.meshgrid(
        torch.arange(H_cam, device=device, dtype=torch.float32),
        torch.arange(W_cam, device=device, dtype=torch.float32),
        indexing='ij'
    )
    pixels_homo = torch.stack([
        u_cam.reshape(-1),
        v_cam.reshape(-1),
        torch.ones_like(u_cam.reshape(-1)),
    ], dim=0).unsqueeze(0).expand(B, 3, -1)  # (B,3,H*W)

    # Rays in camera/world
    rays_cam = torch.bmm(K_inv, pixels_homo)                 # (B,3,H*W)
    R = T_cam_to_world[:, :3, :3]
    t = T_cam_to_world[:, :3, 3:4]
    rays_world = torch.bmm(R, rays_cam)                      # (B,3,H*W)

    # Intersect with Z = ground_height
    rays_z = rays_world[:, 2:3, :]                           # (B,1,H*W)
    tt = torch.where(
        rays_z.abs() > 1e-6,
        (ground_height - t[:, 2:3, :]) / rays_z,
        torch.full_like(rays_z, -1.0),
    )
    points_world = t + tt * rays_world                       # (B,3,H*W)

    Xw = points_world[:, 0, :]                               # (B,H*W)
    Yw = points_world[:, 1, :]

    # Align satellite center to IMU/vehicle XY position for correct centering
    if T_imu_to_world.dim() == 2:
        T_imu_to_world = T_imu_to_world.unsqueeze(0)
    t_imu = T_imu_to_world[:, :3, 3]
    sat_center = t_imu[:, :2] # (B, 2)
    X_off = Xw - sat_center[:, 0:1]
    Y_off = Yw - sat_center[:, 1:2]

    # Satellite pixel coords
    u_sat = sat_size / 2.0 + X_off / resolution              # (B,H*W)
    v_sat = sat_size / 2.0 - Y_off / resolution              # (B,H*W)

    # Valid mask
    valid = (tt > 0) & (u_sat >= 0) & (u_sat < sat_size) & (v_sat >= 0) & (v_sat < sat_size)
    valid = valid.reshape(B, H_cam, W_cam)

    # Normalize to [-1,1]
    u_sat_norm = 2.0 * u_sat / (sat_size - 1) - 1.0
    v_sat_norm = 2.0 * v_sat / (sat_size - 1) - 1.0
    grid_norm = torch.stack([u_sat_norm, v_sat_norm], dim=-1).reshape(B, H_cam, W_cam, 2)

    # Set invalid to invalid_value
    grid_norm = torch.where(valid.unsqueeze(-1), grid_norm, torch.full_like(grid_norm, float(invalid_value)))

    return grid_norm, valid
