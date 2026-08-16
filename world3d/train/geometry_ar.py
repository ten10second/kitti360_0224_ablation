from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from utils.geometry import warp_bev_to_camera_with_coords


def compute_inverse_projection_view(
    sat_tensor: Optional[torch.Tensor],
    K: torch.Tensor,
    T_cam_to_world: Optional[torch.Tensor],
    T_imu_to_world: Optional[torch.Tensor],
    target_h: int,
    target_w: int,
    device: torch.device,
    return_ipm_image: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if T_cam_to_world is not None and T_cam_to_world.dim() == 2:
        T_cam_to_world = T_cam_to_world.unsqueeze(0)
    if T_imu_to_world is not None and T_imu_to_world.dim() == 2:
        T_imu_to_world = T_imu_to_world.unsqueeze(0)
    if K.dim() == 2:
        K = K.unsqueeze(0)

    if sat_tensor is None or T_cam_to_world is None or T_imu_to_world is None:
        # Return dummy values instead of None, shape (1, C, H, W)
        warped_front = torch.zeros((1, 3, target_h, target_w), device=device)
        valid_mask = torch.zeros((1, 1, target_h, target_w), device=device)
        coords_map = torch.zeros((1, 2, target_h, target_w), device=device)
        return warped_front, valid_mask, coords_map

    add_batch_dim = False
    if sat_tensor.dim() == 3:
        add_batch_dim = True
        sat_tensor = sat_tensor.unsqueeze(0)

    sat_tensor = sat_tensor.to(device).float()
    if sat_tensor.numel() > 0 and float(sat_tensor.max()) > 1.5:
        sat_tensor = sat_tensor / 255.0

    warped_front, valid_mask, coords_map = warp_bev_to_camera_with_coords(
        sat_image=sat_tensor,
        K=K,
        T_cam_to_world=T_cam_to_world,
        T_imu_to_world=T_imu_to_world,
        cam_height=target_h,
        cam_width=target_w,
        return_warped_sat=bool(return_ipm_image),
    )


    if warped_front is None or warped_front.numel() == 0:
        warped_front = torch.zeros((1, 3, target_h, target_w), device=device)
        valid_mask = torch.zeros((1, 1, target_h, target_w), device=device)
        coords_map = torch.zeros((1, 2, target_h, target_w), device=device)

    # 不要在返回时挤压批次维度，保留 (1, C, H, W) 形状，以便 torch.stack 能正确堆叠
    return warped_front, valid_mask, coords_map


def scale_intrinsics(K: torch.Tensor, orig_hw: tuple, target_hw: tuple) -> torch.Tensor:
    if K.dim() == 2:
        K = K.unsqueeze(0)
    H0, W0 = orig_hw
    Ht, Wt = target_hw
    sx = float(Wt) / float(W0)
    sy = float(Ht) / float(H0)
    K_out = K.clone()
    K_out[:, 0, 0] *= sx
    K_out[:, 1, 1] *= sy
    K_out[:, 0, 2] = (K_out[:, 0, 2] + 0.5) * sx - 0.5
    K_out[:, 1, 2] = (K_out[:, 1, 2] + 0.5) * sy - 0.5
    return K_out if K_out.shape[0] > 1 else K_out[0]


def soft_decode_logits(
    logits: torch.Tensor,
    vq,
    rows: int,
    cols: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft decode logits to image using Straight-Through Estimator.

    Args:
        logits: (B, L, vocab_size) model output logits
        vq: PretrainedTokenizer with frozen decoder
        rows: grid rows
        cols: grid cols
        temperature: softmax temperature for soft quantization

    Returns:
        img: (B, 3, H, W) decoded image in [0, 1]

    Note:
        Uses STE trick: forward pass uses hard quantization (preserves image quality),
        backward pass uses soft gradients (enables differentiability).
        This avoids the magnitude collapse issue where soft_quant = probs @ codebook
        produces vectors with much smaller L2 norm than actual codebook vectors,
        causing severe artifacts when passed through the frozen decoder.
    """
    B = logits.shape[0]
    codebook = vq.quantize.embedding.weight  # (vocab_size, 256)

    # 1. Soft quantization (for gradient flow)
    probs = F.softmax(logits / temperature, dim=-1)  # (B, L, vocab_size)
    soft_quant = probs @ codebook  # (B, L, 256)

    # 2. Hard quantization (for decoder input quality)
    hard_indices = logits.argmax(dim=-1)  # (B, L)
    hard_quant = vq.quantize.embedding(hard_indices)  # (B, L, 256)

    # 3. STE: forward uses hard, backward uses soft gradient
    quant_feature = soft_quant + (hard_quant - soft_quant).detach()

    # 4. Reshape and decode
    quant_feature = quant_feature.view(B, rows, cols, 256).permute(0, 3, 1, 2)
    img = vq.decoder(quant_feature)  # (B, 3, H, W) in [0, 1]
    return img.clamp(0, 1)


def compute_bev_consistency_loss(
    img_s: torch.Tensor,
    vq,
    target_tokens_t: torch.Tensor,
    rows: int,
    cols: int,
    K_s: torch.Tensor,
    K_t: torch.Tensor,
    T_cam_to_world_s: torch.Tensor,
    T_cam_to_world_t: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    sat_size: int = 512,
    resolution: float = 0.2,
    return_intermediates: bool = False,
):
    """Compute BEV-space consistency loss between two views of the same frame.

    Args:
        img_s: (1, 3, H, W) soft-decoded source view in [0, 1]
        vq: PretrainedTokenizer with frozen decoder
        target_tokens_t: (1, L) target view GT tokens
        rows: grid rows
        cols: grid cols
        K_s: (1, 3, 3) source camera intrinsics
        K_t: (1, 3, 3) target camera intrinsics
        T_cam_to_world_s: (1, 4, 4) source camera extrinsics
        T_cam_to_world_t: (1, 4, 4) target camera extrinsics
        T_imu_to_world: (1, 4, 4) shared IMU pose (same frame)
        sat_size: BEV grid size
        resolution: BEV resolution (meters per pixel)
        return_intermediates: if True, also return (img_t_recon, bev_s, bev_t, valid_s, valid_t)

    Returns:
        loss: scalar MSE loss in BEV overlap region
        intermediates (optional): dict with keys img_t, bev_s, bev_t, valid_s, valid_t
    """
    from utils.geometry.camera_to_sat_projection import project_camera_to_satellite_pull

    # 1. Reconstruct target GT via VQ decoder (efficient, no encoder needed)
    codebook = vq.quantize.embedding.weight  # (vocab_size, 256)
    t_quant = codebook[target_tokens_t]  # (1, L, 256)
    t_quant = t_quant.view(1, rows, cols, 256).permute(0, 3, 1, 2)
    with torch.no_grad():  # GT branch doesn't need gradients
        img_t_recon = vq.decoder(t_quant).clamp(0, 1)  # (1, 3, H, W) [0, 1]

    # 2. Project both views to BEV (anchored at IMU position for alignment)
    ground_h = float(T_imu_to_world[0, 2, 3].item()) - 0.93
    imu_pos = T_imu_to_world[0, :3, 3]  # (3,) IMU world position

    bev_s, valid_s = project_camera_to_satellite_pull(
        img_s[0], K_s[0], T_cam_to_world_s[0],
        sat_size=sat_size, resolution=resolution, ground_height=ground_h,
        center_world=imu_pos,
    )
    bev_t, valid_t = project_camera_to_satellite_pull(
        img_t_recon[0], K_t[0], T_cam_to_world_t[0],
        sat_size=sat_size, resolution=resolution, ground_height=ground_h,
        center_world=imu_pos,
    )

    # 3. Compute MSE in overlap region
    overlap = (valid_s & valid_t).float().unsqueeze(0)  # (1, H, W)
    denom = overlap.sum().clamp(min=1.0)
    loss = ((bev_s - bev_t) ** 2 * overlap).sum() / denom

    if return_intermediates:
        return loss, {
            "img_t": img_t_recon,
            "bev_s": bev_s,
            "bev_t": bev_t,
            "valid_s": valid_s,
            "valid_t": valid_t,
        }
    return loss
