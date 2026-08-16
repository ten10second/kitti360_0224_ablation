"""
Latent-space BEV Consistency Loss

核心改进：
1. 在latent特征空间做IPM，省掉VQ decoder
2. 使用STE保持magnitude一致
3. 收紧FOV边界，避免边缘插值伪影
4. 添加重叠面积阈值，避免异常梯度
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F


def extract_continuous_latent_from_logits(
    logits: torch.Tensor,
    vq_model,
    rows: int,
    cols: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    从logits提取连续的latent特征（强制使用STE）

    ⚠️ 关键：必须使用STE！
    原因：目标视角用的是硬量化 codebook[tokens]，L2 norm是标准的。
    如果源视角用软量化 probs @ codebook，L2 norm会小很多，
    导致两边"刻度"不一致，产生错误的梯度偏移。

    Args:
        logits: (B, L, vocab_size) 模型输出
        vq_model: VQ-VAE模型（用于访问codebook）
        rows, cols: token grid尺寸 (16, 40)
        temperature: softmax温度

    Returns:
        latent_img: (B, 256, rows, cols) latent特征图
    """
    B = logits.shape[0]
    codebook = vq_model.quantize.embedding.weight  # (vocab_size, 256)

    # 1. Soft quantization (可微，用于梯度)
    probs = F.softmax(logits / temperature, dim=-1)  # (B, L, vocab_size)
    soft_quant = probs @ codebook  # (B, L, 256)

    # 2. Hard quantization (保持magnitude)
    hard_indices = logits.argmax(dim=-1)
    hard_quant = vq_model.quantize.embedding(hard_indices)

    # 3. STE: forward用hard（保持magnitude），backward用soft（保持梯度）
    quant_feature = soft_quant + (hard_quant - soft_quant).detach()

    # 4. Reshape为特征图
    latent_img = quant_feature.view(B, rows, cols, 256).permute(0, 3, 1, 2)
    # 输出: (B, 256, rows, cols)

    return latent_img


def project_latent_to_bev_pull(
    latent_img: torch.Tensor,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    sat_size: int = 512,
    resolution: float = 0.2,
    ground_height: float = 0.0,
    center_world: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pull-based: 从BEV网格采样latent特征

    Args:
        latent_img: (B, 256, rows, cols) latent特征图
        K: (B, 3, 3) 相机内参
        T_cam_to_world: (B, 4, 4) 相机外参
        sat_size: BEV网格尺寸
        resolution: BEV分辨率 (m/pixel)
        ground_height: 地面高度
        center_world: (3,) BEV中心的世界坐标（通常是IMU位置）

    Returns:
        bev_latent: (B, 256, sat_size, sat_size) BEV latent特征
        valid_mask: (B, sat_size, sat_size) 有效区域mask
    """
    B, C, rows, cols = latent_img.shape
    device = latent_img.device

    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, 3, 3)
    if T_cam_to_world.dim() == 2:
        T_cam_to_world = T_cam_to_world.unsqueeze(0).expand(B, 4, 4)

    Rcw = T_cam_to_world[:, :3, :3]  # (B, 3, 3)
    tcw = T_cam_to_world[:, :3, 3:4]  # (B, 3, 1)

    # BEV中心：使用IMU位置对齐
    if center_world is not None:
        sat_center = center_world[:2].unsqueeze(0).expand(B, 2)  # (B, 2)
    else:
        sat_center = tcw[:, :2, 0]

    # Step 1: 构建BEV网格的世界坐标
    v_sat = torch.arange(sat_size, device=device, dtype=torch.float32)
    u_sat = torch.arange(sat_size, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v_sat, u_sat, indexing='ij')  # (S, S)
    uu = uu.unsqueeze(0).expand(B, -1, -1)
    vv = vv.unsqueeze(0).expand(B, -1, -1)

    # BEV像素 → 世界坐标 (X向东，Y向北，Z=ground_height)
    Xw = sat_center[:, 0:1, None] + (uu - (sat_size / 2.0)) * resolution
    Yw = sat_center[:, 1:2, None] - (vv - (sat_size / 2.0)) * resolution
    Zw = torch.full_like(Xw, ground_height)

    # (B, 3, S, S) → (B, 3, S*S)
    Xw3 = torch.stack([Xw, Yw, Zw], dim=1).reshape(B, 3, -1)

    # Step 2: 世界坐标 → 相机坐标
    Xc = torch.bmm(Rcw.transpose(1, 2), (Xw3 - tcw))  # (B, 3, S*S)
    z = Xc[:, 2:3, :]  # (B, 1, S*S)

    # Step 3: 相机坐标 → 像素坐标
    Xc_norm = Xc / torch.clamp(z, min=1e-6)
    uv = torch.bmm(K, Xc_norm)  # (B, 3, S*S)
    u = uv[:, 0, :]  # (B, S*S)
    v = uv[:, 1, :]

    # Step 4: 计算有效mask（在相机FOV内且深度>0）
    # ✅ 修正：收紧边界，避免边缘插值伪影
    # align_corners=True时：u_n=-1对应索引0，u_n=1对应索引cols-1
    # 收紧0.5个单位，确保插值点远离边缘
    valid = (z[:, 0, :] > 0) & \
            (u >= 0.5) & (u <= cols - 1.5) & \
            (v >= 0.5) & (v <= rows - 1.5)

    # Step 5: 归一化到[-1, 1]用于grid_sample
    u_n = 2.0 * u / (cols - 1) - 1.0
    v_n = 2.0 * v / (rows - 1) - 1.0

    # ✅ 修正：不强推无效点到-2.0
    # 原因：后续用overlap mask会自动过滤，强推反而引入异常梯度
    # 直接让grid_sample处理，padding_mode='zeros'会自动填0

    grid = torch.stack([u_n, v_n], dim=-1).reshape(B, sat_size, sat_size, 2)

    # Step 6: grid_sample拉取latent特征
    bev_latent = F.grid_sample(
        latent_img,
        grid,
        mode='bilinear',
        padding_mode='zeros',  # FOV外自动填0
        align_corners=True  # 与valid mask的边界定义一致
    )  # (B, 256, sat_size, sat_size)

    valid_mask = valid.reshape(B, sat_size, sat_size)

    return bev_latent, valid_mask


def compute_latent_bev_consistency_loss(
    logits_s: torch.Tensor,
    target_tokens_t: torch.Tensor,
    vq_model,
    rows: int,
    cols: int,
    K_s: torch.Tensor,
    K_t: torch.Tensor,
    T_cam_to_world_s: torch.Tensor,
    T_cam_to_world_t: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    sat_size: int = 512,
    resolution: float = 0.2,
    temperature: float = 1.0,
    min_overlap_pixels: float = 100.0,
    return_intermediates: bool = False,
):
    """
    在latent空间计算BEV一致性损失

    Args:
        logits_s: (1, L, vocab_size) 源视角的模型输出
        target_tokens_t: (1, L) 目标视角的GT tokens
        vq_model: VQ-VAE模型
        rows, cols: token grid尺寸 (16, 40)
        K_s, K_t: (1, 3, 3) 两个视角的内参
        T_cam_to_world_s, T_cam_to_world_t: (1, 4, 4) 两个视角的外参
        T_imu_to_world: (1, 4, 4) IMU位置（用于BEV对齐）
        sat_size: BEV网格尺寸
        resolution: BEV分辨率
        temperature: softmax温度
        min_overlap_pixels: 最小重叠像素数，低于此值返回0 loss
                           (避免边缘极小重叠区域的异常梯度)
        return_intermediates: 是否返回中间结果（用于可视化）

    Returns:
        loss: scalar，在重叠区域的Smooth L1 loss
        intermediates (optional): 中间结果字典
    """
    # 1. 提取源视角的latent特征（强制STE）
    latent_s = extract_continuous_latent_from_logits(
        logits_s, vq_model, rows, cols, temperature
    )  # (1, 256, 16, 40)

    # 2. 提取目标视角的latent特征（GT，硬量化）
    codebook = vq_model.quantize.embedding.weight
    latent_t = codebook[target_tokens_t]  # (1, L, 256)
    latent_t = latent_t.view(1, rows, cols, 256).permute(0, 3, 1, 2)
    latent_t = latent_t.detach()  # 阻断梯度

    # 3. 计算地面高度和IMU位置
    ground_h = float(T_imu_to_world[0, 2, 3].item()) - 0.93
    imu_pos = T_imu_to_world[0, :3, 3]  # (3,)

    # 4. 投影到BEV空间
    bev_latent_s, valid_s = project_latent_to_bev_pull(
        latent_s, K_s, T_cam_to_world_s,
        sat_size=sat_size, resolution=resolution,
        ground_height=ground_h, center_world=imu_pos
    )

    bev_latent_t, valid_t = project_latent_to_bev_pull(
        latent_t, K_t, T_cam_to_world_t,
        sat_size=sat_size, resolution=resolution,
        ground_height=ground_h, center_world=imu_pos
    )

    # 5. 计算重叠区域
    overlap = (valid_s & valid_t).float()  # (1, sat_size, sat_size)
    valid_area = overlap.sum()

    # ✅ 修正：重叠面积阈值截断
    # 如果重叠区域太小（<100像素），说明只是边缘擦边，
    # 这种区域的插值极不可靠，强行算loss会产生异常梯度
    if valid_area < min_overlap_pixels:
        # 返回0 loss，但保持requires_grad=True以维持计算图
        zero_loss = torch.tensor(0.0, device=logits_s.device, requires_grad=True)
        if return_intermediates:
            return zero_loss, {
                "latent_s": latent_s,
                "latent_t": latent_t,
                "bev_latent_s": bev_latent_s,
                "bev_latent_t": bev_latent_t,
                "valid_s": valid_s,
                "valid_t": valid_t,
                "overlap": overlap,
                "valid_area": valid_area,
                "skipped": True,
            }
        return zero_loss

    # 6. 在重叠区域计算Smooth L1 loss
    # 使用Smooth L1而不是MSE，对outlier更鲁棒
    diff_map = F.smooth_l1_loss(
        bev_latent_s, bev_latent_t, reduction='none'
    )  # (1, 256, sat_size, sat_size)

    # 压缩通道维度
    diff_spatial = diff_map.mean(dim=1)  # (1, sat_size, sat_size)

    # 只在重叠区域求均值
    loss = (diff_spatial * overlap).sum() / valid_area

    if return_intermediates:
        return loss, {
            "latent_s": latent_s,
            "latent_t": latent_t,
            "bev_latent_s": bev_latent_s,
            "bev_latent_t": bev_latent_t,
            "valid_s": valid_s,
            "valid_t": valid_t,
            "overlap": overlap,
            "valid_area": valid_area,
            "diff_spatial": diff_spatial,
            "skipped": False,
        }

    return loss
