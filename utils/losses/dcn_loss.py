"""
DCN (Deformable Cross-Attention) Loss Functions
Includes geometry-aware loss computations to improve semantic generation
"""

import torch
import torch.nn.functional as F

def compute_offset_regularization(sampling_offsets, scale=0.1):
    """
    正则化采样偏移量，鼓励offset保持小值（接近几何投影）
    """
    offset_reg = (sampling_offsets ** 2).mean()
    return scale * offset_reg

def compute_geometry_alignment_loss(
    sampling_locations,
    dcn_mask,
    bev_height,
    bev_width,
    scale=0.1
):
    """
    几何对齐loss：采样点应该在有效区域内
    """
    # 将mask下采样到BEV特征图大小
    dcn_mask_resized = F.interpolate(
        dcn_mask.float().unsqueeze(1),
        size=(bev_height, bev_width),
        mode='bilinear',
        align_corners=False
    )
    
    # 使用grid_sample检查采样点是否在有效区域内
    # sampling_locations is (B, L, n_heads, n_points, 2) with values in [-1, 1]
    # grid_sample expects (N, C, H_in, W_in) and grid (N, H_out, W_out, 2)
    B, L, n_heads, n_points, _ = sampling_locations.shape
    grid = sampling_locations.view(B, L * n_heads, n_points, 2)

    mask_values = F.grid_sample(
        dcn_mask_resized, # (B, 1, H_bev, W_bev)
        grid,             # (B, L*n_heads, n_points, 2)
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    ) # (B, 1, L*n_heads, n_points)
    
    # Loss：最大化采样点在有效区域内的概率（值接近1）
    geometry_loss = (1.0 - mask_values).mean()
    
    return scale * geometry_loss

def compute_attention_entropy_loss(att_weights, scale=0.05):
    """
    Attention熵loss：鼓励attention权重分布更集中
    """
    eps = 1e-8
    entropy = -(att_weights * (torch.log(att_weights + eps))).sum(dim=-1)
    entropy_loss = entropy.mean()
    return scale * entropy_loss

def compute_dcn_total_loss(
    logits,
    target_tokens,
    sampling_offsets,
    sampling_locations,
    dcn_mask,
    bev_height,
    bev_width,
    att_weights=None,
    vocab_size=16384,
    ce_weight=1.0,
    offset_reg_weight=0.1,
    geometry_weight=0.2,
    entropy_weight=0.05
):
    """
    综合DCN loss计算
    """
    # 1. Cross-Entropy Loss
    ce_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        target_tokens.view(-1)
    )
    
    # 2. Offset Regularization
    offset_reg = compute_offset_regularization(sampling_offsets, scale=1.0)
    
    # 3. Geometry Alignment Loss
    geometry_loss = compute_geometry_alignment_loss(
        sampling_locations, dcn_mask, bev_height, bev_width, scale=1.0
    )
    
    # 4. Attention Entropy Loss (可选)
    entropy_loss = torch.tensor(0.0, device=logits.device)
    if att_weights is not None:
        entropy_loss = compute_attention_entropy_loss(att_weights, scale=1.0)
    
    # 加权求和
    total_loss = (
        ce_weight * ce_loss +
        offset_reg_weight * offset_reg +
        geometry_weight * geometry_loss +
        entropy_weight * entropy_loss
    )
    
    loss_dict = {
        'ce_loss': ce_loss.item(),
        'offset_reg': offset_reg.item(),
        'geometry_loss': geometry_loss.item(),
        'entropy_loss': entropy_loss.item() if att_weights is not None else 0.0,
        'total_loss': total_loss.item()
    }
    
    return total_loss, loss_dict

def get_curriculum_weights(step, total_steps, strategy='balanced'):
    """
    Curriculum Learning权重调度
    """
    progress = step / max(total_steps, 1)
    
    if strategy == 'geometry_first':
        if progress < 0.3:
            return {'ce_weight': 0.5, 'offset_reg_weight': 0.2, 'geometry_weight': 0.5, 'entropy_weight': 0.05}
        elif progress < 0.7:
            return {'ce_weight': 0.7, 'offset_reg_weight': 0.15, 'geometry_weight': 0.2, 'entropy_weight': 0.05}
        else:
            return {'ce_weight': 0.8, 'offset_reg_weight': 0.1, 'geometry_weight': 0.1, 'entropy_weight': 0.05}
    
    elif strategy == 'semantic_first':
        if progress < 0.3:
            return {'ce_weight': 0.8, 'offset_reg_weight': 0.1, 'geometry_weight': 0.1, 'entropy_weight': 0.05}
        elif progress < 0.7:
            return {'ce_weight': 0.7, 'offset_reg_weight': 0.15, 'geometry_weight': 0.2, 'entropy_weight': 0.05}
        else:
            return {'ce_weight': 0.6, 'offset_reg_weight': 0.2, 'geometry_weight': 0.3, 'entropy_weight': 0.05}
    
    else:  # balanced
        return {'ce_weight': 0.7, 'offset_reg_weight': 0.1, 'geometry_weight': 0.2, 'entropy_weight': 0.05}

