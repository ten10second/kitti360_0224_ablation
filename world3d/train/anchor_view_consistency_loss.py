"""Anchor-view overlap-only consistency loss utilities.

当前 stage2 的一致性目标收敛为：
- 只在 anchor_view 与 target_view 的有效重叠区域上施加约束；
- 优先约束静态结构的 feature identity consistency；
- 不再把 BEV / RGB 投影一致性作为默认主损失。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def compute_feature_consistency_loss(
    anchor_feat: torch.Tensor,
    target_feat: torch.Tensor,
    anchor_coords: torch.Tensor,
    target_coords: torch.Tensor,
    overlap_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.5,
    distance_threshold: float = 0.05,
) -> dict:
    """
    overlap-only 特征一致性损失：
    仅在 anchor / target 的有效重叠区域内，通过 BEV 坐标匹配计算 InfoNCE。

    Args:
        anchor_feat: (B, D, R, C) - anchor 视角特征
        target_feat: (B, D, R, C) - target 视角特征
        anchor_coords: (B, 2, R, C) - anchor BEV 坐标
        target_coords: (B, 2, R, C) - target BEV 坐标
        overlap_mask: (B, 1, R, C) or (B, R, C) - target 端有效重叠区域掩码
        temperature: float - 对比学习温度
        distance_threshold: float - 坐标距离阈值

    Returns:
        dict:
            - loss: overlap-only 特征一致性损失
            - overlap_ratio: 匹配到的重叠 token 比例
            - matched_pairs: 匹配到的 token 对数量
    """
    B, D, R, C = anchor_feat.shape
    device = anchor_feat.device

    anchor_flat = anchor_feat.permute(0, 2, 3, 1).reshape(B, R * C, D)
    target_flat = target_feat.permute(0, 2, 3, 1).reshape(B, R * C, D)
    coords_anchor = anchor_coords.permute(0, 2, 3, 1).reshape(B, R * C, 2)
    coords_target = target_coords.permute(0, 2, 3, 1).reshape(B, R * C, 2)

    if overlap_mask is not None:
        if overlap_mask.dim() == 4:
            overlap_mask = overlap_mask[:, 0]
        overlap_mask = overlap_mask.reshape(B, R * C).bool()

    total_loss = torch.tensor(0.0, device=device)
    pair_count = 0
    total_matched_pairs = 0
    total_target_overlap = 0

    for b in range(B):
        # 过滤无效坐标（哨兵值通常<=-1.2，正常BEV坐标范围在[-1,1]之间）
        valid_anchor = (coords_anchor[b, :, 0] > -1.2) & (coords_anchor[b, :, 1] > -1.2)
        valid_target = (coords_target[b, :, 0] > -1.2) & (coords_target[b, :, 1] > -1.2)
        if overlap_mask is not None:
            valid_target = valid_target & overlap_mask[b]

        # 获取有效坐标的原始全局索引
        anchor_idx_all = torch.where(valid_anchor)[0]  # (M,)
        target_idx_all = torch.where(valid_target)[0]  # (N,)
        total_target_overlap += int(target_idx_all.numel())

        if anchor_idx_all.numel() == 0 or target_idx_all.numel() == 0:
            continue

        # 只在有效坐标之间计算距离
        dist_matrix = torch.cdist(coords_anchor[b, anchor_idx_all], coords_target[b, target_idx_all], p=2)  # (M, N)
        min_dist, nn_idx = dist_matrix.min(dim=1)  # (M,)
        valid = min_dist < distance_threshold

        if not valid.any():
            continue

        matched_pairs = int(valid.sum().item())
        total_matched_pairs += matched_pairs

        # 用全局索引获取对应的特征
        valid_idx_local = torch.where(valid)[0]
        anchor_idx_global = anchor_idx_all[valid_idx_local]
        target_idx_global = target_idx_all[nn_idx[valid_idx_local]]

        feat_anchor = anchor_flat[b, anchor_idx_global]
        feat_target = target_flat[b, target_idx_global]

        feat_anchor = F.normalize(feat_anchor, dim=-1)
        feat_target = F.normalize(feat_target, dim=-1)

        sim_matrix = torch.matmul(feat_anchor, feat_target.t()) / temperature  # (M, M)
        labels = torch.arange(sim_matrix.size(0), device=device)

        loss_b = F.cross_entropy(sim_matrix, labels)
        loss_t = F.cross_entropy(sim_matrix.t(), labels)
        loss_bilateral = 0.5 * (loss_b + loss_t)

        total_loss = total_loss + loss_bilateral
        pair_count += 1

    if pair_count > 0:
        total_loss = total_loss / pair_count

    overlap_ratio = 0.0
    if total_target_overlap > 0:
        overlap_ratio = float(total_matched_pairs) / float(total_target_overlap)

    return {
        "loss": total_loss,
        "overlap_ratio": torch.tensor(overlap_ratio, device=device),
        "matched_pairs": torch.tensor(float(total_matched_pairs), device=device),
    }


class AnchorViewConsistencyLoss(nn.Module):
    """
    overlap-only 的 Anchor 视角一致性损失模块。

    当前默认只保留：
    1. overlap-masked feature consistency

    可选的 BEV / RGB 正则不再作为默认主路径。
    """

    def __init__(
        self,
        feature_loss_weight: float = 1.0,
        temperature: float = 0.5,
        distance_threshold: float = 0.05,
    ):
        super().__init__()
        self.feature_loss_weight = feature_loss_weight
        self.temperature = temperature
        self.distance_threshold = distance_threshold

    def forward(
        self,
        anchor_feat: torch.Tensor,
        target_feat: torch.Tensor,
        anchor_coords: torch.Tensor,
        target_coords: torch.Tensor,
        overlap_mask: Optional[torch.Tensor] = None,
        use_feature_loss: bool = True,
    ) -> dict:
        """
        计算 overlap-only 一致性损失。

        Args:
            anchor_feat: (B, D, R, C) - anchor 特征
            target_feat: (B, D, R, C) - target 特征
            anchor_coords: (B, 2, R, C) - anchor BEV 坐标
            target_coords: (B, 2, R, C) - target BEV 坐标
            overlap_mask: (B, 1, R, C) or (B, R, C) - overlap 有效区域
            use_feature_loss: bool - 是否使用特征对齐损失

        Returns:
            dict: 包含各项损失和总损失
                - overlap_feature_loss: overlap-only 特征损失
                - overlap_ratio: 有效重叠比例
                - matched_pairs: 匹配到的 token 对数量
                - total_loss: 加权总损失
        """
        device = anchor_feat.device
        results = {
            'overlap_feature_loss': torch.tensor(0.0, device=device),
            'overlap_ratio': torch.tensor(0.0, device=device),
            'matched_pairs': torch.tensor(0.0, device=device),
            'total_loss': torch.tensor(0.0, device=device),
        }

        # overlap-only 特征对齐一致性损失
        if use_feature_loss:
            feature_results = compute_feature_consistency_loss(
                anchor_feat=anchor_feat,
                target_feat=target_feat,
                anchor_coords=anchor_coords,
                target_coords=target_coords,
                overlap_mask=overlap_mask,
                temperature=self.temperature,
                distance_threshold=self.distance_threshold,
            )
            results['overlap_feature_loss'] = feature_results['loss']
            results['overlap_ratio'] = feature_results['overlap_ratio']
            results['matched_pairs'] = feature_results['matched_pairs']

        total = self.feature_loss_weight * results['overlap_feature_loss']
        results['total_loss'] = total

        return results
