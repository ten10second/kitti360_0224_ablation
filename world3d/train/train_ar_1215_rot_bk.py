#!/usr/bin/env python3
"""
BEV-Only Stage2 Training - 纯BEV条件驱动的RGB token预测 (Autoregressive)

Notes:
- Tokenizer/codebook is fixed (ckpts/maskgit-vqgan-imagenet-f16-256.bin)
- 使用自回归模型，训练和推理一致
"""
import os
import sys


import argparse
import random
from contextlib import nullcontext
from typing import Tuple, Optional

import numpy as np

import cv2
import matplotlib
matplotlib.use('Agg')

import torch

import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP

# Make repo root importable
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CUR_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Distributed training utilities
from utils.distributed import (
    init_distributed_mode, is_dist_avail_and_initialized,
    get_world_size, get_rank, is_main_process, wait_for_everyone
)

from metrics.lpips import LPIPS

# Stage1/2
from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from models.stage2.simplified_token_predictor import SimplifiedTokenPredictor, BottomUpSimplifiedTokenPredictor
from pathlib import Path
from world3d.io.kitti360d_dataloader import Kitti360dDataset



# Geometry utilities (keep K/T reading)
from utils.geometry import (
    warp_bev_to_camera_with_coords,
)

IMU_TO_GROUND_HEIGHT = 0.93


from world3d.train.vis_utils import save_coords_step_debug, plot_loss_curve


def get_model_attr(model, attr: str):
    if hasattr(model, 'module'):
        return getattr(model.module, attr)
    return getattr(model, attr)


def decode_logits_to_rgb(logits, vocab_size, target_rows, target_cols, codebook_weight, decoder):
    """
    Soft-decode logits into RGB by taking expectation over VQ codebook embeddings.
    Accepts either sequence logits or grid logits:
      - logits: (B, L, Vtot) in row-major sequence order (top-left) -> will be reshaped to (B,R,C,V)
      - logits: (B, R, C, Vtot) already in NORMAL top-down grid order -> used as is
    Args:
        logits: (B, L, model_vocab_size) OR (B, R, C, model_vocab_size)
        vocab_size: number of VQ entries (e.g., 1024)
        target_rows/cols: token grid shape
        codebook_weight: (vocab_size, embedding_dim)
        decoder: VQ decoder module mapping (B, C, rows, cols) -> RGB
    Returns:
        recon_rgb: (B, 3, rows*16, cols*16) in [-1, 1]
    """
    if logits.dim() == 3:
        B, _, _ = logits.shape
        logits_vq = logits[:, :, :vocab_size]
        probs = F.softmax(logits_vq, dim=-1)
        probs = probs.view(B, target_rows, target_cols, vocab_size)
    elif logits.dim() == 4:
        # Already (B, R, C, V)
        logits_vq = logits[:, :, :, :vocab_size]
        probs = F.softmax(logits_vq, dim=-1)
    else:
        raise ValueError(f"decode_logits_to_rgb expects logits of dim 3 or 4, got {tuple(logits.shape)}")

    # expected embedding per token (B, rows, cols, embed_dim)
    emb = torch.matmul(probs, codebook_weight)
    emb = emb.permute(0, 3, 1, 2).contiguous()
    recon = decoder(emb)
    recon = recon * 2.0 - 1.0
    recon = torch.clamp(recon, -1.0, 1.0)
    return recon

def compute_inverse_projection_view(
    sat_tensor: Optional[torch.Tensor],
    K: torch.Tensor,
    T_cam_to_world: Optional[torch.Tensor],
    T_imu_to_world: Optional[torch.Tensor],
    target_h: int,
    target_w: int,
    device: torch.device,
    sat_m_per_px: float = 0.2,
    sat_size_px: int = 512
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:


    if T_cam_to_world is not None and T_cam_to_world.dim() == 2:
        T_cam_to_world = T_cam_to_world.unsqueeze(0)
    if T_imu_to_world is not None and T_imu_to_world.dim() == 2:
        T_imu_to_world = T_imu_to_world.unsqueeze(0)
    if K.dim() == 2:
        K = K.unsqueeze(0)

    """
    Compute backward warping from BEV to camera view using ground plane assumption.

    Process:
    1. For each camera pixel, compute corresponding BEV coordinate via ground plane intersection
    2. Sample BEV image at those coordinates (backward warping, not forward projection)

    Returns:
        warped_front: (1, 3, H, W) - BEV RGB sampled at camera pixel positions
        valid_mask: (1, 1, H, W) - validity mask for sampling
        coords_map: (1, 2, H, W) - normalized BEV coordinates [-1,1] for each camera pixel
    """
    if sat_tensor is None or T_cam_to_world is None or T_imu_to_world is None:
        return None, None, None


    if sat_tensor.dim() == 3:
        sat_tensor = sat_tensor.unsqueeze(0)

    # Normalize satellite image to [0,1].
    sat_tensor = sat_tensor.to(device).float()
    if sat_tensor.numel() > 0 and float(sat_tensor.max()) > 1.5:
        sat_tensor = sat_tensor / 255.0

    warped_front, valid_mask, coords_map = warp_bev_to_camera_with_coords(
        sat_image=sat_tensor,
        K=K,
        T_cam_to_world=T_cam_to_world,
        T_imu_to_world=T_imu_to_world,
        cam_height=target_h,
        cam_width=target_w
    )

    if warped_front is None or warped_front.numel() == 0:
        warped_front = torch.zeros((1, 3, target_h, target_w), device=device)
        valid_mask = torch.zeros((1, 1, target_h, target_w), device=device)
        coords_map = torch.zeros((1, 2, target_h, target_w), device=device)


    return warped_front, valid_mask, coords_map
def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D representation (first two columns).

    Args:
        R: (..., 3, 3)
    Returns:
        (..., 6)
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def build_pose_vec(
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    img_h: int,
    img_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Build pose vector [rot6d, t_rel, intr] for PoseMLP.

    rot6d: 6
    t_rel: 3  (t_cam_world - t_imu_world)
    intr:  4  (fx/W, fy/H, cx/W, cy/H)
    Returns:
        (13,) tensor
    """
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

def get_condition_scale_sizes(rows: int, cols: int):
    """For single-scale model, just return the target resolution."""
    return [("fine", rows, cols)]


def build_condition_tokens(
    warped_front: Optional[torch.Tensor],
    warped_valid: Optional[torch.Tensor],
    target_rows: int,
    target_cols: int,
    device: torch.device,
    rgb_gt: Optional[torch.Tensor] = None,
    scale_sizes=None,
):
    if scale_sizes is None:
        scale_sizes = get_condition_scale_sizes(target_rows, target_cols)
    token_h = max(1, int(target_rows))
    token_w = max(1, int(target_cols))
    full_h = token_h * 16
    full_w = token_w * 16

    if warped_front is None:
        cond_img = torch.zeros((1, 3, full_h, full_w), device=device)
    else:
        cond_img = warped_front.to(device)
    cond_img = torch.clamp(cond_img, 0.0, 1.0)
    cond_img = cond_img * 2.0 - 1.0

    if warped_valid is None:
        cond_mask = torch.zeros((1, 1, full_h, full_w), device=device)
    else:
        cond_mask = warped_valid.to(device).float()

    if cond_mask.shape[-2:] != (full_h, full_w):
        cond_mask = F.interpolate(cond_mask, size=(full_h, full_w), mode='nearest')

    pooled_rgb = F.avg_pool2d(cond_img, kernel_size=16, stride=16)
    pooled_mask = F.avg_pool2d(cond_mask, kernel_size=16, stride=16)
    if rgb_gt is None:
        confidence_map = pooled_mask.clone()
    else:
        gt_resized = F.interpolate(rgb_gt.to(device), size=(full_h, full_w), mode='bilinear', align_corners=False)
        diff = torch.mean(torch.abs(cond_img - gt_resized), dim=1, keepdim=True)
        norm_diff = torch.clamp(diff / 2.0, 0.0, 1.0)
        confidence_map = 1.0 - norm_diff

    confidence_patch = F.avg_pool2d(confidence_map, kernel_size=16, stride=16)
    outputs = {}
    for name, rows_s, cols_s in scale_sizes:
        rows_s = max(1, rows_s)
        cols_s = max(1, cols_s)
        condition_s = torch.cat([pooled_rgb, pooled_mask], dim=1)
        condition_s = F.adaptive_avg_pool2d(condition_s, (rows_s, cols_s))
        confidence_s = F.adaptive_avg_pool2d(confidence_patch, (rows_s, cols_s))
        outputs[name] = (
            condition_s.squeeze(0),
            confidence_s.view(rows_s * cols_s)
        )
    return outputs


def build_condition_tokens_with_coords(
    warped_front: Optional[torch.Tensor],
    warped_coords: Optional[torch.Tensor],  # (1,2,H,W) in [-1,1], invalid may be -2
    warped_valid: Optional[torch.Tensor],
    target_rows: int,
    target_cols: int,
    device: torch.device,
    scale_sizes=None,
):
    """Build multi-scale condition tokens using RGB + valid mask and BEV coords.

    Returns:
      semantic_outputs[name]: (4, rows, cols)
      coord_outputs[name]: (2, rows, cols)
    """
    if scale_sizes is None:
        scale_sizes = get_condition_scale_sizes(target_rows, target_cols)
    token_h = max(1, int(target_rows))
    token_w = max(1, int(target_cols))
    full_h = token_h * 16
    full_w = token_w * 16

    if warped_front is None:
        cond_img = torch.zeros((1, 3, full_h, full_w), device=device)
    else:
        cond_img = warped_front.to(device)
    cond_img = torch.clamp(cond_img, 0.0, 1.0)
    cond_img = cond_img * 2.0 - 1.0

    if warped_coords is None:
        coords = torch.zeros((1, 2, full_h, full_w), device=device)
    else:
        coords = warped_coords.to(device)
        coords = torch.where(coords < -1.1, torch.full_like(coords, -1.5), coords)

    if warped_valid is None:
        cond_mask = torch.zeros((1, 1, full_h, full_w), device=device)
    else:
        cond_mask = warped_valid.to(device).float()
        if cond_mask.shape[-2:] != (full_h, full_w):
            cond_mask = F.interpolate(cond_mask, size=(full_h, full_w), mode='nearest')

    pooled_rgb = F.avg_pool2d(cond_img, kernel_size=16, stride=16)
    pooled_coords = F.interpolate(coords, size=(token_h, token_w), mode='nearest')
    pooled_mask = F.avg_pool2d(cond_mask, kernel_size=16, stride=16)

    semantic_outputs = {}
    coord_outputs = {}
    for name, rows_s, cols_s in scale_sizes:
        rows_s = max(1, rows_s)
        cols_s = max(1, cols_s)
        semantic_s = torch.cat([pooled_rgb, pooled_mask], dim=1)
        semantic_s = F.adaptive_avg_pool2d(semantic_s, (rows_s, cols_s))
        coords_s = F.interpolate(pooled_coords, size=(rows_s, cols_s), mode='nearest')
        semantic_outputs[name] = semantic_s.squeeze(0)
        coord_outputs[name] = coords_s.squeeze(0)
    return semantic_outputs, coord_outputs


def scale_intrinsics(K: torch.Tensor, orig_hw: tuple, target_hw: tuple) -> torch.Tensor:
    """Scale pinhole intrinsics K when image is resized from (H0,W0) -> (Ht,Wt).

    Args:
        K: (3,3) or (B,3,3)
        orig_hw: (H0, W0)
        target_hw: (Ht, Wt)
    Returns:
        Scaled K with principal point 0.5px correction.
    """
    if K.dim() == 2:
        K = K.unsqueeze(0)
    H0, W0 = orig_hw
    Ht, Wt = target_hw
    sx = float(Wt) / float(W0)
    sy = float(Ht) / float(H0)
    K_out = K.clone()
    K_out[:, 0, 0] *= sx  # fx'
    K_out[:, 1, 1] *= sy  # fy'
    K_out[:, 0, 2] = (K_out[:, 0, 2] + 0.5) * sx - 0.5  # cx'
    K_out[:, 1, 2] = (K_out[:, 1, 2] + 0.5) * sy - 0.5  # cy'
    return K_out if K_out.shape[0] > 1 else K_out[0]






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--data_seed',
        type=int,
        default=0,
        help='Seed for deterministic data sampling (indices/yaw). Use the same value across DDP runs for reproducibility.',
    )
    ap.add_argument(
        '--deterministic_data',
        action='store_true',
        help='If set, make per-step data index deterministic across runs/DP/DDP (recommended for reproducibility).',
    )
    ap.add_argument('--shuffle', action='store_true', help='Shuffle the sample list')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--drive', type=str, default='2013_05_28_drive_0003_sync', help='Drive folder name under repo root')
    ap.add_argument('--p_front', type=float, default=0, help='Probability of sampling front view (else virtual)')
    ap.add_argument('--yaw_min_abs', type=float, default=45.0, help='Min abs yaw for virtual view sampling (deg)')
    ap.add_argument('--yaw_max_abs', type=float, default=100.0, help='Max abs yaw for virtual view sampling (deg)')
    ap.add_argument('--subset', type=int, default=None, help='Use only first N samples')
    ap.add_argument('--virtual_hfov', type=float, default=80.0, help='HFOV for virtual perspective (deg)')
    ap.add_argument('--virtual_w', type=int, default=640, help='Width of virtual perspective')
    ap.add_argument('--virtual_h', type=int, default=256, help='Height of virtual perspective')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--vq_ckpt', type=str, default='ckpts/maskgit-vqgan-imagenet-f16-256.bin')
    ap.add_argument('--steps', type=int, default=120000)
    ap.add_argument('--print_every', type=int, default=100)
    ap.add_argument('--save_every', type=int, default=4000)
    ap.add_argument('--vis_every', type=int, default=100, help='Visualization frequency')
    ap.add_argument('--coords_vis_every', type=int, default=0,
                    help='Save coords_step_*.png/csv debug visualization every N steps (0=disabled)')
    ap.add_argument('--vis_temperature', type=float, default=1.0, help='Sampling temperature for visualization')
    ap.add_argument('--vis_top_k', type=int, default=100, help='Top-k sampling for visualization (0=disabled, default=100)')
    ap.add_argument('--vis_top_p', type=float, default=0.95, help='Top-p sampling for visualization (0.0=disabled, default=0.95)')
    ap.add_argument('--plot_every', type=int, default=100, help='Plot loss curve frequency (0=disabled)')
    ap.add_argument('--out_dir', type=str, default='runs/ar_simplified')

    # Model architecture (simplified single-scale)
    ap.add_argument('--fourier_freqs', type=int, default=10, help='Number of Fourier frequency bands for coordinate encoding')

    ap.add_argument('--ntp_order', type=str, default='topleft', choices=['topleft', 'bottomup'], help='Next-token prediction order')

    # BEV Encoder settings
    ap.add_argument('--train_bev_encoder', action='store_true', help='If set, the BEV encoder (ResNet50) will be trained')
    ap.add_argument('--no_bev_pretrain', action='store_true', help='If set, the BEV encoder will be initialized from scratch')

    ap.add_argument('--d_model', type=int, default=512)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--num_layers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--weight_decay', type=float, default=0.01)

    # LR scheduler (warmup + cosine)
    ap.add_argument('--use_warmup_cosine', action='store_true',
                    help='Use warmup + cosine LR schedule (default: off)')
    ap.add_argument('--warmup_updates', type=int, default=4000,
                    help='Number of optimizer updates for linear warmup (default: 2000)')
    ap.add_argument('--min_lr', type=float, default=1e-6,
                    help='Minimum LR at the end of cosine annealing (default: 1e-6)')
    ap.add_argument('--grid_cols', type=int, default=40)
    ap.add_argument('--grid_rows', type=int, default=16)
    ap.add_argument('--sat_dir', type=str, default=None)
    # Evaluation settings
    ap.add_argument('--eval_every', type=int, default=100, help='Run evaluation every N steps (0=disabled)')
    ap.add_argument('--eval_samples', type=int, default=4, help='How many samples from current batch to evaluate (generation)')
    ap.add_argument('--compute_fid', action='store_true', help='Compute FID on evaluated samples (requires torch-fidelity)')
    ap.add_argument('--lpips_net', type=str, default='alex', choices=['alex','vgg','squeeze'], help='Backbone for LPIPS metric')

    ap.add_argument('--label_smoothing', type=float, default=0.0)
    ap.add_argument('--bos_token', type=int, default=1024, help='Beginning-of-sequence token ID')
    ap.add_argument('--accum_steps', type=int, default=4, help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
    ap.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    ap.add_argument('--resume_ckpt', type=str, default=None,
                    help='Path to checkpoint to resume training from')

    ap.add_argument('--ce_weight', type=float, default=1.0, help='Weight for Cross-Entropy loss')



    args = ap.parse_args()

    # Initialize distributed training
    device = init_distributed_mode()
    world_size = get_world_size()
    rank = get_rank()
    is_main = is_main_process()

    if not is_dist_avail_and_initialized() and args.device:
        device = torch.device(args.device)

    if is_main:
        print(f"[Distributed] World size: {world_size}, Rank: {rank}, Device: {device}")

    # Enable anomaly detection for debugging (set env TORCH_ANOMALY_DETECT=1)
    if os.environ.get('TORCH_ANOMALY_DETECT', '0') == '1':
        torch.autograd.set_detect_anomaly(True)
        if is_main:
            print("[Debug] Anomaly detection enabled")

    # Tokenizer
    if is_main:
        print(f"[Tokenizer] Using VQGAN (ImageNet pretrained) from {args.vq_ckpt}")
    vq = PretrainedTokenizer(args.vq_ckpt).to(device)
    vocab_size = 1024

    vq.eval()
    vq.requires_grad_(False)


    model_vocab_size = max(vocab_size, args.bos_token + 1)
    target_cols = max(1, int(args.grid_cols))
    target_rows = max(1, int(args.grid_rows))
    target_w = target_cols * 16
    target_h = target_rows * 16
    seq_len = target_rows * target_cols

    # Single-scale configuration
    condition_scale_specs = get_condition_scale_sizes(target_rows, target_cols)
    condition_scale_names = ["fine"]
    coarse_rows, coarse_cols = target_rows, target_cols

    # Stage2 - Simplified single-scale autoregressive generation
    if is_main:
        print("[Model] Creating SimplifiedTokenPredictor (single-scale)...")
        print(f"[Model] VQGAN vocab_size: {vocab_size}, Model vocab_size: {model_vocab_size}, BOS token: {args.bos_token}")
        print(f"[Model] Fourier frequencies: {args.fourier_freqs}")

    if args.ntp_order == 'bottomup':
        ModelClass = BottomUpSimplifiedTokenPredictor
    else:
        ModelClass = SimplifiedTokenPredictor
    if is_main:
        print(f"[Model] Using predictor class: {ModelClass.__name__} (ntp_order={args.ntp_order})")

    predictor = ModelClass(
        d_model=args.d_model,
        vocab_size=model_vocab_size,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=0.1,
        max_seq_len=seq_len,
        target_rows=target_rows,
        target_cols=target_cols,
        semantic_dim=4,
        coord_dim=2,
        fourier_freqs=args.fourier_freqs,

        train_bev_encoder=args.train_bev_encoder,
        no_bev_pretrain=args.no_bev_pretrain,
        pose_dim=13,
        use_pose_token=True,
    ).to(device)
    predictor.train()

    if is_main:
        num_params = sum(p.numel() for p in predictor.parameters())
        print(f"[SimplifiedTokenPredictor] Parameters: {num_params:,}")

    # Wrap with DDP
    if world_size > 1:
        # Use LOCAL_RANK env var set by torchrun/launch launchers to get the correct device index
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        predictor = DDP(predictor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    bev_encoder_module = get_model_attr(predictor, 'bev_encoder')
    bev_feature_dim = get_model_attr(predictor, 'bev_feature_dim')

    # Optimizer
    optim = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ================= [Warmup + Cosine Scheduler START] =================
    scheduler = None
    if args.use_warmup_cosine:
        import math as _math
        total_updates = _math.ceil(args.steps / max(1, int(args.accum_steps)))
        warmup = max(0, int(args.warmup_updates))
        min_lr_ratio = max(0.0, float(args.min_lr) / max(1e-12, float(args.lr)))

        def lr_lambda(current_update: int):
            # current_update starts at 0 after the first optimizer.step()
            if current_update < warmup:
                # linear warmup from 0 -> 1
                return float(current_update + 1) / float(max(1, warmup))
            # cosine decay from 1 -> min_lr_ratio
            progress = float(current_update - warmup) / float(max(1, total_updates - warmup))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + _math.cos(_math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optim, lr_lambda=lr_lambda)
        if is_main:
            print(f"[Scheduler] Warmup+Cosine enabled. Updates: {total_updates}, Warmup: {warmup}, min_lr: {args.min_lr}")
    # ================= [Warmup + Cosine Scheduler END] =================

    # Resume from checkpoint if provided
    start_step = 1
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        if is_main:
            print(f"[Resume] Loading checkpoint from {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)

        # Load model state (fallback to non-strict for newly added modules)
        try:
            if world_size > 1:
                predictor.module.load_state_dict(ckpt['model'], strict=True)
            else:
                predictor.load_state_dict(ckpt['model'], strict=True)
        except RuntimeError as e:
            if is_main:
                print(f"[Resume] Strict load failed, falling back to non-strict: {e}")
            if world_size > 1:
                incompatible = predictor.module.load_state_dict(ckpt['model'], strict=False)
            else:
                incompatible = predictor.load_state_dict(ckpt['model'], strict=False)
            if is_main:
                missing = getattr(incompatible, 'missing_keys', [])
                unexpected = getattr(incompatible, 'unexpected_keys', [])
                print(f"[Resume] Non-strict load applied. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                # Optionally print lists when small
                if len(missing) <= 20 and len(unexpected) <= 20:
                    print(f"[Resume] Missing: {missing}")
                    print(f"[Resume] Unexpected: {unexpected}")


        # Load optimizer state if available; fall back to fresh optimizer when groups mismatch
        if 'optimizer' in ckpt:
            try:
                optim.load_state_dict(ckpt['optimizer'])
            except Exception as opt_err:
                if is_main:
                    print(f"[Resume] Skipping optimizer state due to mismatch: {opt_err}")
                    print("[Resume] Continuing with freshly initialized optimizer.")

        # Resume from the saved step
        start_step = ckpt.get('step', 1) + 1
        if is_main:
            print(f"[Resume] Resuming from step {start_step}")
            print(f"[Resume] Loaded model_vocab_size: {ckpt.get('model_vocab_size', model_vocab_size)}")
    elif args.resume_ckpt:
        if is_main:
            print(f"[Warning] Checkpoint not found: {args.resume_ckpt}")

    # Dataset (single drive, mixed views)
    # Use pose-centered supervision: mix front (image_00) and fisheye_virtual views.
    drive_dir = Path(REPO_ROOT) / args.drive

    # Build list of frames that have an exact IMU pose (no nearest-pose fallback).
    poses_path = drive_dir / "poses.txt"
    if not poses_path.exists():
        raise SystemExit(f"poses.txt not found: {poses_path}")

    pose_frame_ids = []
    with open(poses_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: frame_id + 12 floats
            parts = line.split()
            try:
                pose_frame_ids.append(int(parts[0]))
            except Exception:
                continue

    pose_frame_ids = sorted(list(set(pose_frame_ids)))

    # Optional: shuffle frame list once (will remove strict temporal order)
    if args.shuffle:
        random.shuffle(pose_frame_ids)
        if is_main:
            print("[Dataset] Shuffled pose_frame_ids")
    if args.subset is not None:
        pose_frame_ids = pose_frame_ids[:max(0, int(args.subset))]
        if is_main:
            print(f"[Dataset] Subset: limited to {len(pose_frame_ids)} pose frames")

    if len(pose_frame_ids) == 0:
        raise SystemExit("No pose frames found in poses.txt")

    if is_main:
        print(f"[Dataset] Drive: {drive_dir}")
        print(f"[Dataset] Pose frames: {len(pose_frame_ids)}")
        print(f"[Dataset] Virtual view HFOV: {args.virtual_hfov} degrees")
        print(f"[Dataset] Virtual view size: {args.virtual_w}x{args.virtual_h}")
        print(f"[Dataset] Yaw range: [-{args.yaw_max_abs}, -{args.yaw_min_abs}] U [{args.yaw_min_abs}, {args.yaw_max_abs}] degrees")

    # Dataset instances
    # For front view (fixed)
    ds_front = Kitti360dDataset(
        drives=drive_dir,
        frames=pose_frame_ids,
        require_exact_pose=True,
        mode="front",
        front_resize=(args.virtual_w, args.virtual_h),
        seed=args.data_seed if args.deterministic_data else None,
    )

    # For virtual view: random yaw per-sample inside the Dataset (keeps caches effective)
    ds_virtual = Kitti360dDataset(
        drives=drive_dir,
        frames=pose_frame_ids,
        require_exact_pose=True,
        mode="fisheye_virtual",
        virtual_hfov_deg=args.virtual_hfov,
        virtual_size=(args.virtual_w, args.virtual_h),
        random_virtual_yaw=True,
        yaw_min_abs=args.yaw_min_abs,
        yaw_max_abs=args.yaw_max_abs,
        seed=args.data_seed if args.deterministic_data else None,
    )

    # Init metrics
    lpips_metric = LPIPS(net=args.lpips_net).to(device).eval()

    dataset_size = len(pose_frame_ids)
    if dataset_size == 0:
        raise SystemExit("No samples found")

    os.makedirs(args.out_dir, exist_ok=True)

    loss_history = []  # List of (step, loss) tuples

    if is_main:
        if start_step > 1:
            print(f"[Training] Resuming autoregressive training from step {start_step} to {args.steps}")
        else:
            print(f"[Training] Starting autoregressive training for {args.steps} steps")
        print(f"[Training] Dataset size: {dataset_size}")
        print(f"[Training] Batch size per GPU: {args.batch_size}")
        print(f"[Training] Grad accumulation steps: {max(1, int(args.accum_steps))}")
        print(f"[Training] Effective global batch: {args.batch_size * world_size * max(1, int(args.accum_steps))}")
        print(f"[Training] BOS token: {args.bos_token}")
        print(f"[Training] Target size: {target_h}x{target_w} ({target_rows}x{target_cols} tokens)")

    # Training loop
    # Holders for visualization coordinate PE (use first sample in batch)

    for step in range(start_step, args.steps + 1):
        inverse_proj_vis = None
        inverse_valid_mask_vis = None
        # 收集一个batch的数据（可选坐标PE）
        batch_input_tokens = []
        batch_target_tokens = []
        batch_condition_semantic = {name: [] for name in condition_scale_names}
        batch_condition_coords = {name: [] for name in condition_scale_names}
        sat_img_vis = None
        gt_img_vis = None
        K_vis, T_cam_to_world_vis, T_imu_to_world_vis = None, None, None


        # Supervision valid mask per sample (fine scale, length = rows*cols)
        batch_supervision_valid = []
        batch_aligned_bev = []
        batch_bev_available = []  # Track per-sample BEV availability
        batch_ipm_rgb = []
        batch_ipm_valid = []
        batch_pose_vecs = []  # (B,13)
        batch_K = []  # (B,3,3) intrinsics for ray-direction PE

        # Prepare holder for visualization (use first sample)


        for b in range(args.batch_size):

            current_sample_data = None

            # Deterministic index sampling (reproducible across runs and DDP) if enabled.
            if args.deterministic_data:
                # Use a step/rank/b dependent seed so each rank gets different (but reproducible) samples.
                per_item_seed = int(args.data_seed) + int(step) * 1000003 + int(rank) * 10007 + int(b)
                rng = random.Random(per_item_seed)
                sample_idx = rng.randrange(dataset_size)
            else:
                # Randomize sample index to avoid strict temporal cycling (reduces periodic loss waves)
                sample_idx = random.randint(0, dataset_size - 1)

            try:
                # --- [核心修改：混合视角采样逻辑] ---
                # 决定是用前视还是虚拟视角
                use_front = random.random() < args.p_front  # args.p_front 建议 0.5

                if use_front:
                    sample = ds_front[sample_idx]
                    view_name = "front"
                else:
                    # Virtual view: yaw is randomized inside ds_virtual per-sample
                    sample = ds_virtual[sample_idx]
                    meta_yaw = None
                    try:
                        meta_yaw = float(sample.get("meta", {}).get("vehicle_relative_yaw_deg"))
                    except Exception:
                        meta_yaw = None
                    view_name = "virtual" if meta_yaw is None else f"virtual_{meta_yaw:.1f}"

                # --- [解包数据] ---
                # image: (3, H, W) Tensor [0, 1]
                rgb = sample['image'].to(device)
                rgb = rgb * 2.0 - 1.0 # 转为 [-1, 1] 给 VQGAN 用

                # K, R, T 都是 Tensor
                K = sample['K'].to(device)
                assert K.shape == (3, 3), f"K shape wrong: {K.shape}, mode={sample['meta']['mode']}, frame={sample['frame_id']}"

                T_cam_to_world = sample['T_cam_to_world'].to(device)
                T_imu_to_world = sample['T_imu_to_world'].to(device)
                sat_available = sample.get('sat_available', False)

                # 卫星图: (3, 512, 512) Tensor [0, 1] -> 转 Numpy 给 compute_inverse_projection_view 用
                sat_tensor = sample['sat']

                with torch.no_grad():
                    idx_grid = vq.encode(rgb.unsqueeze(0))
                    # vq.encode may return either (B,R,C) or (B,L). Normalize to (R,C).
                    if idx_grid.dim() == 3:
                        idx_grid = idx_grid.squeeze(0)
                    elif idx_grid.dim() == 2 and idx_grid.shape[0] == 1 and idx_grid.shape[1] == target_rows * target_cols:
                        idx_grid = idx_grid.view(target_rows, target_cols)
                    else:
                        # keep as-is; make_teacher_forcing has additional safety reshapes
                        pass

                # teacher-forcing
                predictor_module = predictor.module if hasattr(predictor, 'module') else predictor
                input_tokens, target_tokens = predictor_module.make_teacher_forcing(idx_grid, bos_token=args.bos_token)
                input_tokens = input_tokens.squeeze(0) # (L,)
                target_tokens = target_tokens.squeeze(0) # (L,)


                # calculate warping
                warped_front, warped_valid, warped_coords = compute_inverse_projection_view(
                    sat_tensor=sat_tensor,
                    K=K,
                    T_cam_to_world=T_cam_to_world,
                    T_imu_to_world=T_imu_to_world,
                    target_h=target_h,
                    target_w=target_w,
                    device=device,
                )
                if warped_front is None or warped_front.numel() == 0 or warped_front.shape[-1] == 0:
                    raise ValueError("Projection returned empty tensor")

                # Save first sample for visualization and debug comparison
                if b == 0:
                    gt_img_vis = rgb
                    inverse_proj_vis = warped_front.detach().squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                    inverse_valid_mask_vis = warped_valid.detach().squeeze(0).float().cpu().numpy() if warped_valid is not None else None
                    warped_coords_vis = warped_coords

                    if sat_tensor is not None:
                        sat_img_vis = (sat_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    K_vis = K
                    T_cam_to_world_vis = T_cam_to_world
                    T_imu_to_world_vis = T_imu_to_world

                # Build Condition (semantic + coordinates)
                semantic_tokens, coord_tokens = build_condition_tokens_with_coords(
                    warped_front, warped_coords, warped_valid,
                    target_rows, target_cols, device,
                    scale_sizes=condition_scale_specs,
                )

                # Build Pose Vector (for PoseMLP)
                pv = build_pose_vec(
                    K=K,
                    T_cam_to_world=T_cam_to_world,
                    T_imu_to_world=T_imu_to_world,
                    img_h=target_h,
                    img_w=target_w,
                    device=device
                )


                # BEV Feature extraction logic
                if sat_available:
                    with torch.no_grad():
                        bev_feats = bev_encoder_module(sat_tensor.unsqueeze(0).to(device))

                    # Process BEV features to match token resolution
                    pooled_coords_token = coord_tokens['fine'].unsqueeze(0)  # (1,2,rows,cols)
                    grid_token = pooled_coords_token.permute(0, 2, 3, 1)  # (1,rows,cols,2)
                    aligned_token = F.grid_sample(
                        bev_feats,
                        grid_token,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=True,
                    )  # (1, 256, rows, cols)

                    # Mask out invalid tokens using pooled valid mask from semantic branch
                    pooled_mask_token = semantic_tokens['fine'][3:4, :, :].unsqueeze(0)  # (1,1,rows,cols)
                    aligned_token = aligned_token * pooled_mask_token
                    aligned_bev_token = aligned_token.squeeze(0)
                    bev_available = True
                else:
                    predictor_module = predictor.module if hasattr(predictor, 'module') else predictor
                    bev_feature_dim = get_model_attr(predictor_module, 'bev_feature_dim')
                    aligned_bev_token = torch.zeros(bev_feature_dim, target_rows, target_cols, device=device, dtype=torch.float32)
                    bev_available = False

                # Prepare IPM for visualization
                ipm_rgb = torch.clamp(warped_front.to(device), 0.0, 1.0) * 2.0 - 1.0
                ipm_rgb = ipm_rgb.squeeze(0) if ipm_rgb is not None else torch.zeros(3, target_h, target_w, device=device)
                ipm_valid = warped_valid.to(device).float().squeeze(0) if warped_valid is not None else torch.zeros(1, target_h, target_w, device=device)

                current_sample_data = {
                    'input_tokens': input_tokens.squeeze(0),
                    'target_tokens': target_tokens.squeeze(0),
                    'semantic': semantic_tokens,
                    'coords': coord_tokens,
                    'pose': pv,
                    'K': K,
                    'bev': aligned_bev_token,
                    'bev_available': bev_available,
                    'ipm_rgb': warped_front.squeeze(0),
                    'ipm_valid': ipm_valid.squeeze(0),
                    'sup_valid': torch.ones(seq_len, device=device)
                }

                # Save first sample for visualization
                if b == 0:
                    gt_img_vis = rgb
                    inverse_proj_vis = warped_front.detach().squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                    inverse_valid_mask_vis = warped_valid.detach().squeeze(0).float().cpu().numpy() if warped_valid is not None else None
                    warped_coords_vis = warped_coords
                    # sat_img_vis needs to be a numpy array for visualization later
                    if sat_tensor is not None:
                        sat_img_vis = (sat_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    K_vis = K
                    T_cam_to_world_vis = T_cam_to_world
                    T_imu_to_world_vis = T_imu_to_world

            except Exception as e:

                # import traceback
                # traceback.print_exc()
                if is_main:
                    print(
                        f"[Warning] Failed to process sample {sample_idx} "
                        f"({view_name if 'view_name' in locals() else 'unknown'}): {e}"
                    )

                # Fallback to a zeroed sample so that each `b` contributes exactly one element
                # to the batch lists (length consistency / atomicity).
                current_sample_data = {
                    'input_tokens': torch.zeros(seq_len, dtype=torch.long, device=device),
                    'target_tokens': torch.zeros(seq_len, dtype=torch.long, device=device),
                    'semantic': {n: torch.zeros(4, rs, cs, device=device) for n, rs, cs in condition_scale_specs},
                    'coords': {n: torch.zeros(2, rs, cs, device=device) for n, rs, cs in condition_scale_specs},
                    'pose': torch.zeros(13, device=device),
                    'K': torch.eye(3, device=device),
                    'bev': torch.zeros(bev_feature_dim, target_rows, target_cols, device=device),
                    'bev_available': False,
                    'ipm_rgb': torch.zeros(3, target_h, target_w, device=device),
                    'ipm_valid': torch.zeros(1, target_h, target_w, device=device),
                    'sup_valid': torch.zeros(seq_len, device=device),
                }

            # Append exactly once per `b` (atomic sample object)
            batch_input_tokens.append(current_sample_data['input_tokens'])
            batch_target_tokens.append(current_sample_data['target_tokens'])
            batch_pose_vecs.append(current_sample_data['pose'])
            batch_K.append(current_sample_data['K'])

            for scale_name in condition_scale_names:
                batch_condition_semantic[scale_name].append(current_sample_data['semantic'][scale_name])
                batch_condition_coords[scale_name].append(current_sample_data['coords'][scale_name])

            batch_aligned_bev.append(current_sample_data['bev'])
            batch_bev_available.append(current_sample_data['bev_available'])
            batch_ipm_rgb.append(current_sample_data['ipm_rgb'])
            batch_ipm_valid.append(current_sample_data['ipm_valid'])
            batch_supervision_valid.append(current_sample_data['sup_valid'])


        # Check if all samples failed to load (empty batch)
        if len(batch_input_tokens) == 0:
            if is_main:
                print(f"[Warning] All samples failed to load at step {step}, skipping...")
            continue



        input_tokens_batch = torch.stack(batch_input_tokens, dim=0)  # (B, L)
        target_tokens_batch = torch.stack(batch_target_tokens, dim=0)  # (B, L)

        batch_size = input_tokens_batch.size(0)
        input_tokens_batch = input_tokens_batch.view(batch_size, 16, 40)  # (B, 16, 40)
        target_tokens_batch = target_tokens_batch.view(batch_size, 16, 40)  # (B, 16, 40)

        condition_tokens_batch = {
            'semantic': {
                name: torch.stack(tensors, dim=0)
                for name, tensors in batch_condition_semantic.items()
            },
            'coords': {
                name: torch.stack(tensors, dim=0)
                for name, tensors in batch_condition_coords.items()
            }
        }
        aligned_bev_batch = torch.stack(batch_aligned_bev, dim=0)
        # Ensure BEV features batch is on the correct device (GPU)
        aligned_bev_input = aligned_bev_batch.to(device)

        # Stack K matrices for ray-direction PE
        K_batch = torch.stack(batch_K, dim=0)  # (B,3,3)

        # Supervision valid mask (fine scale, length=L)
        supervision_valid_batch = torch.stack(batch_supervision_valid, dim=0)  # (B, L)
        pose_batch = torch.stack(batch_pose_vecs, dim=0)  # (B,13)

        # -------------------- Sanity check (batch pose/K alignment) --------------------
        # If batch_size=1 works but batch_size>1 has wrong coords/pose, it is often caused by
        # silently inserting fallback samples (zeros) into the batch. That makes per-sample
        # condition (pose/K/coords) inconsistent.
        # We explicitly mark invalid samples by checking the supervision mask.
        valid_sample_mask = (supervision_valid_batch.sum(dim=1) > 0)  # (B,)
        if valid_sample_mask.numel() > 0 and (not bool(valid_sample_mask.all())):
            # If any invalid samples exist, drop them for this step to keep pose/coords consistent.
            if is_main:
                bad = (~valid_sample_mask).nonzero(as_tuple=False).flatten().tolist()
                print(f"[Warn] Found {len(bad)} invalid samples in batch at step {step}: idx={bad}. Dropping them.")

            input_tokens_batch = input_tokens_batch[valid_sample_mask]
            target_tokens_batch = target_tokens_batch[valid_sample_mask]
            supervision_valid_batch = supervision_valid_batch[valid_sample_mask]
            pose_batch = pose_batch[valid_sample_mask]
            K_batch = K_batch[valid_sample_mask]
            aligned_bev_input = aligned_bev_input[valid_sample_mask]
            condition_tokens_batch = {
                'semantic': {'fine': condition_tokens_batch['semantic']['fine'][valid_sample_mask]},
                'coords': {'fine': condition_tokens_batch['coords']['fine'][valid_sample_mask]},
            }

        # If nothing left after dropping invalid samples, skip this training step
        if input_tokens_batch.size(0) == 0:
            if is_main:
                print(f"[Warn] All samples invalid after filtering at step {step}, skipping...")
            continue
        # -----------------------------------------------------------------------------

        # Forward pass (simplified model - only fine scale)
        condition_tokens_input = {
            'semantic': condition_tokens_batch['semantic']['fine'],
            'coords': condition_tokens_batch['coords']['fine'],
            'pose': pose_batch,
            # For ray-direction positional encoding inside SimplifiedTokenPredictor
            'K': K_batch,  # Use the collected K matrices
        }
        logits = predictor(
            generated_tokens=input_tokens_batch.view(input_tokens_batch.size(0), -1),
            condition_tokens=condition_tokens_input,
            aligned_bev_feature_map=aligned_bev_input,
        )  # (B, L, vocab_size)

        ce_per_token = F.cross_entropy(
            logits.view(-1, model_vocab_size),
            target_tokens_batch.view(-1),
            reduction='none',
            label_smoothing=args.label_smoothing
        ).view_as(target_tokens_batch)

        sup_valid = supervision_valid_batch.clamp(0.0, 1.0).view_as(ce_per_token)
        denom = sup_valid.sum().clamp(min=1e-6)
        ce_loss = (ce_per_token * sup_valid).sum() / denom

        # Train with CE only
        total_loss = args.ce_weight * ce_loss

        # Gradient accumulation setup
        accum_steps = max(1, int(args.accum_steps))
        # zero gradients at the start of each accumulation window
        if (step - start_step) % accum_steps == 0:
            optim.zero_grad(set_to_none=True)

        # Use DDP no_sync on non-final micro-steps to save communication
        is_last_micro = ((step - start_step) % accum_steps == accum_steps - 1) or (step == args.steps)
        ddp_no_sync_ctx_pred = predictor.no_sync() if (world_size > 1 and hasattr(predictor, 'no_sync') and not is_last_micro) else nullcontext()
        with ddp_no_sync_ctx_pred:
            (total_loss / accum_steps).backward()

        # Step optimizer only on last micro-step
        if is_last_micro:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
            optim.step()
            if scheduler is not None:
                scheduler.step()

        # Record loss
        if is_main:
            loss_history.append((step, total_loss.item()))

        if is_main and step % args.print_every == 0:
            # Print losses
            loss_str = f"[Step {step}] CE Loss: {ce_loss.item():.4f} | Total: {total_loss.item():.4f}"
            if 'scheduler' in locals() and scheduler is not None:
                try:
                    cur_lr = scheduler.get_last_lr()[0]
                    loss_str += f" | LR={cur_lr:.2e}"
                except Exception:
                    pass
            # Quick modality diagnostics
            try:
                predictor_module = predictor.module if hasattr(predictor, 'module') else predictor
                if 'batch_bev_available' in locals() and isinstance(batch_bev_available, list) and len(batch_bev_available) > 0:
                    bev_avail_cnt = sum(1 for v in batch_bev_available if v)
                else:
                    bev_avail_cnt = input_tokens_batch.size(0)
                loss_str += f" | BEV used=True (avail {bev_avail_cnt}/{input_tokens_batch.size(0)})"
                if getattr(predictor_module, 'last_bev_feat_mean', None) is not None:
                    m = float(predictor_module.last_bev_feat_mean)
                    s = float(predictor_module.last_bev_feat_std)
                    loss_str += f" | BEV μ={m:.3f}, σ={s:.3f}"
                gates = getattr(predictor_module, 'last_fusion_gates', None)
                if gates is not None:
                    gm = gates.mean(dim=0).detach().cpu().tolist()  # [sem, coord, bev]
                    loss_str += " | Gate(mean)=[{:.2f},{:.2f},{:.2f}]".format(*gm)
            except Exception:
                pass
            print(loss_str)

        # Quick evaluation (LPIPS, optional FID) every eval_every steps
        if is_main and int(getattr(args, 'eval_every', 0)) > 0 and step % int(getattr(args, 'eval_every', 0)) == 0:
            with torch.no_grad():
                predictor.eval()
                os.makedirs(os.path.join(args.out_dir, 'eval'), exist_ok=True)
                eval_count = min(int(getattr(args, 'eval_samples', 1)), input_tokens_batch.size(0))
                lpips_vals = []
                save_for_fid_real = []
                save_for_fid_fake = []

                # 确保 K_batch 是张量并打印调试信息
                try:
                    K_batch_tensor = torch.stack(batch_K, dim=0)
                    if K_batch_tensor.dim() != 3 or K_batch_tensor.shape[1:] != (3, 3):
                        raise ValueError(f"Invalid K_batch shape: {K_batch_tensor.shape}, expected (B,3,3)")
                except Exception as e:
                    print(f"[Warn] Failed to stack K_batch: {e}. Using identity matrices.")
                    K_batch_tensor = torch.eye(3, device=device).unsqueeze(0).repeat(eval_count, 1, 1)

                for eval_idx in range(eval_count):
                    # Prepare per-sample condition and optional aligned BEV
                    K = K_batch_tensor[eval_idx:eval_idx+1]  # (1,3,3)
                    condition_tokens_vis = {
                        'semantic': condition_tokens_batch['semantic']['fine'][eval_idx:eval_idx+1],
                        'coords': condition_tokens_batch['coords']['fine'][eval_idx:eval_idx+1],
                        'pose': pose_batch[eval_idx:eval_idx+1],
                        'K': K,  # 确保形状为 (1,3,3)
                    }
                    aligned_bev_vis = aligned_bev_batch[eval_idx:eval_idx+1]

                    # Ground-truth image
                    predictor_module = predictor.module if hasattr(predictor, 'module') else predictor
                    # target_tokens_batch is already a grid of (B,R,C). Slicing keeps the dims.
                    gt_tokens_grid = target_tokens_batch[eval_idx:eval_idx+1] # (1,R,C)
                    gt_decoded = vq.decode(gt_tokens_grid)  # (1,3,H*16,W*16) in [-1,1]
                    gt_01 = torch.clamp(gt_decoded * 0.5 + 0.5, 0.0, 1.0)

                    # AR sampling
                    current_input = torch.full((1, 1), args.bos_token, dtype=torch.long, device=device)
                    gen_list = []
                    for _ in range(seq_len):
                        logits = predictor(
                            generated_tokens=current_input,
                            condition_tokens=condition_tokens_vis,
                            aligned_bev_feature_map=aligned_bev_vis,
                        )
                        next_token_logits = logits[:, -1, :vocab_size]
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.argmax(probs, dim=-1)
                        gen_list.append(next_token.item())
                        current_input = torch.cat([current_input, next_token.unsqueeze(1)], dim=1)
                    gen_tokens_seq = torch.tensor(gen_list, dtype=torch.long, device=device).view(1, -1)
                    gen_tokens_grid = predictor_module.seq_to_grid(gen_tokens_seq)
                    gen_decoded = vq.decode(gen_tokens_grid)  # [-1,1]
                    gen_01 = torch.clamp(gen_decoded * 0.5 + 0.5, 0.0, 1.0)

                    # Masked LPIPS using supervision mask (for non-aug it's all-ones)
                    sup_valid_vec = supervision_valid_batch[eval_idx].view(1, 1, target_rows, target_cols)
                    sup_valid_full = F.interpolate(sup_valid_vec.float(), size=(target_h, target_w), mode='nearest')
                    sup_valid_full = sup_valid_full.repeat(1, 3, 1, 1)  # (1,3,H,W)
                    # set outside mask to GT to nullify LPIPS contribution
                    const = gt_01
                    gt_masked = gt_01 * sup_valid_full + const * (1.0 - sup_valid_full)
                    gen_masked = gen_01 * sup_valid_full + const * (1.0 - sup_valid_full)
                    lp = lpips_metric(gt_masked, gen_masked)
                    if torch.is_tensor(lp):
                        lp_val = float(lp.item())
                    else:
                        lp_val = float(lp)
                    lpips_vals.append(lp_val)

                    # Optionally accumulate images for FID (save later as PNG)
                    if bool(getattr(args, 'compute_fid', False)):
                        # Ensure tensors are in (C,H,W) format before permuting to (H,W,C) for numpy
                        gt_np = (gt_01.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                        gen_np = (gen_01.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                        save_for_fid_real.append(gt_np)
                        save_for_fid_fake.append(gen_np)

                # Print LPIPS summary
                if len(lpips_vals) > 0:
                    mean_lpips = sum(lpips_vals) / len(lpips_vals)
                    print(f"[Eval@step {step}] Masked-LPIPS ({eval_count}samp): {mean_lpips:.4f}")

                # Coords debug visualization (export coords_step_*.png/csv)
                if is_main and getattr(args, 'coords_vis_every', 0) and (step % int(args.coords_vis_every) == 0):
                    try:
                        vis_dir = os.path.join(args.out_dir, 'samples')
                        coords_fine = condition_tokens_batch['coords']['fine'][0]  # (2, rows, cols)
                        save_coords_step_debug(
                            step=step,
                            out_dir=vis_dir,
                            sat_img_vis=sat_img_vis,
                            front_ipm_vis=inverse_proj_vis,
                            front_h=target_h,
                            front_w=target_w,
                            coords_fine=coords_fine,
                            K_vis=K_vis,
                            T_cam_to_world_vis=T_cam_to_world_vis,
                            T_imu_to_world_vis=T_imu_to_world_vis,
                            gt_img_vis=gt_img_vis,
                            is_main=is_main,
                        )
                    except Exception as e:
                        if is_main:
                            print(f"[Vis] Coord visualization failed: {e}")

                # Optional FID with torch-fidelity (quick estimate on small subset)
                if bool(getattr(args, 'compute_fid', False)) and len(save_for_fid_real) > 0:
                    try:
                        from torch_fidelity import calculate_metrics
                        import imageio.v2 as imageio
                        tmp_dir_real = os.path.join(args.out_dir, 'eval', f'step_{step:06d}', 'real')
                        tmp_dir_fake = os.path.join(args.out_dir, 'eval', f'step_{step:06d}', 'fake')
                        os.makedirs(tmp_dir_real, exist_ok=True)
                        os.makedirs(tmp_dir_fake, exist_ok=True)
                        for i, (r, f_) in enumerate(zip(save_for_fid_real, save_for_fid_fake)):
                            imageio.imwrite(os.path.join(tmp_dir_real, f'{i:03d}.png'), r)
                            imageio.imwrite(os.path.join(tmp_dir_fake, f'{i:03d}.png'), f_)
                        metrics = calculate_metrics(input1=tmp_dir_real, input2=tmp_dir_fake, fid=True, cuda=torch.cuda.is_available())
                        fid_val = metrics.get('frechet_inception_distance', None)
                        if fid_val is not None:
                            print(f"[Eval@step {step}] FID ({eval_count}samp): {float(fid_val):.2f}")
                    except Exception as e:
                        print(f"[Eval] Skipping FID: {e}. Install torch-fidelity for FID.")
                predictor.train()

        # Plot loss curve
        if is_main and args.plot_every > 0 and step % args.plot_every == 0 and len(loss_history) > 0:
            plot_loss_curve(loss_history, args.out_dir, step)

        if is_main and step % args.vis_every == 0 and gt_img_vis is not None:
            os.makedirs(os.path.join(args.out_dir, 'samples'), exist_ok=True)
            with torch.no_grad():
                predictor.eval()

                # Use the ground truth image saved specifically for visualization
                gt_img = gt_img_vis
                predictor_module = predictor.module if hasattr(predictor, 'module') else predictor
                # target_tokens_batch is already a grid of (B,R,C)
                gt_tokens = target_tokens_batch[0:1]  # (1,R,C)

                vq_recon = vq.decode(gt_tokens)
                vq_img = vq_recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
                vq_img = np.clip(vq_img * 0.5 + 0.5, 0, 1)

                condition_tokens_vis = {
                    'semantic': condition_tokens_batch['semantic']['fine'][0:1],
                    'coords': condition_tokens_batch['coords']['fine'][0:1],
                    'pose': pose_batch[0:1],
                    'K': K_batch[0:1],
                }
                aligned_bev_vis = aligned_bev_batch[0:1]

                # Autoregressive generation with top-k + top-p sampling
                gen_tokens = []
                current_input = torch.full((1, 1), args.bos_token, dtype=torch.long, device=device)  # (1, 1) [BOS]

                for _ in range(seq_len):
                    logits = predictor(
                        generated_tokens=current_input,
                        condition_tokens=condition_tokens_vis,
                        aligned_bev_feature_map=aligned_bev_vis,
                    )
                    next_token_logits = logits[:, -1, :vocab_size]

                    # Apply temperature, top-k, top-p as before
                    if args.vis_temperature != 1.0:
                        next_token_logits = next_token_logits / args.vis_temperature
                    if args.vis_top_k > 0:
                        top_k = min(args.vis_top_k, next_token_logits.size(-1))
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    if args.vis_top_p > 0.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > args.vis_top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')

                    probs = F.softmax(next_token_logits, dim=-1)
                    if args.vis_temperature == 1.0 and args.vis_top_k == 0 and args.vis_top_p == 0.0:
                        next_token = torch.argmax(probs, dim=-1)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    gen_tokens.append(next_token.item())
                    current_input = torch.cat([current_input, next_token.unsqueeze(1)], dim=1)

                # Decode generated tokens
                gen_tokens_seq = torch.tensor(gen_tokens, dtype=torch.long, device=device).view(1, -1)
                predictor_module = predictor.module if hasattr(predictor, 'module') else predictor
                gen_tokens_grid = predictor_module.seq_to_grid(gen_tokens_seq)
                gen_decoded = vq.decode(gen_tokens_grid)
                gen_img = gen_decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
                gen_img = np.clip(gen_img * 0.5 + 0.5, 0, 1)

                # Convert gt_img tensor to numpy array for visualization (C,H,W) -> (H,W,C), [-1,1] -> [0,1] -> [0,255]
                gt_img_denorm = np.clip(gt_img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, 0, 1)
                gt_np = (gt_img_denorm * 255).astype(np.uint8)
                recon_np = (vq_img * 255).astype(np.uint8)
                generated_np = (gen_img * 255).astype(np.uint8)

                if inverse_proj_vis is not None:
                    inv_src = inverse_proj_vis
                    if torch.is_tensor(inv_src):
                        inv_src = inv_src.detach().squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                    inv_np = (np.clip(inv_src, 0, 1) * 255).astype(np.uint8)
                    # Ensure contiguous memory layout for OpenCV
                    inv_np = np.ascontiguousarray(inv_np)
                    if inverse_valid_mask_vis is not None:
                        mask_vis = inverse_valid_mask_vis
                        if mask_vis.ndim > 2:
                            mask_vis = np.squeeze(mask_vis)
                        if mask_vis.ndim == 2 and mask_vis.shape[0] > 0 and mask_vis.shape[1] > 0:
                            mask_uint8 = (np.clip(mask_vis, 0, 1) * 255).astype(np.uint8)
                            try:
                                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if len(contours) > 0:
                                    cv2.drawContours(inv_np, contours, -1, (0, 255, 0), 2)
                            except cv2.error as _err:
                                if is_main:
                                    print(f"[Vis] Skipping contour drawing: {_err}")
                else:
                    inv_np = np.zeros_like(gt_np)

                top_row = np.concatenate([gt_np, inv_np], axis=1)
                bottom_row = np.concatenate([recon_np, generated_np], axis=1)
                grid = np.concatenate([top_row, bottom_row], axis=0)

                from PIL import Image as PILImage, ImageDraw, ImageFont
                grid_pil = PILImage.fromarray(grid)
                draw = ImageDraw.Draw(grid_pil)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                except Exception:
                    font = ImageFont.load_default()

                img_h, img_w = gt_np.shape[:2]
                labels = {
                    (10, 10): "Ground Truth",
                    (img_w + 10, 10): "Inverse Projection",
                    (10, img_h + 10): "VQ Recon",
                    (img_w + 10, img_h + 10): "Generated",
                }
                for (x, y), label in labels.items():
                    draw.text((x, y), label, fill=(255, 255, 0), font=font)

                grid_pil.save(os.path.join(args.out_dir, 'samples', f'step_{step:06d}.png'))

                # Build sampling info string
                sampling_parts = []
                if args.vis_temperature != 1.0:
                    sampling_parts.append(f"temp={args.vis_temperature:.2f}")
                if args.vis_top_k > 0:
                    sampling_parts.append(f"top_k={args.vis_top_k}")
                if args.vis_top_p > 0.0:
                    sampling_parts.append(f"top_p={args.vis_top_p:.2f}")

                if sampling_parts:
                    sampling_info = ", ".join(sampling_parts)
                else:
                    sampling_info = "greedy"

                print(f"[Step {step}] Saved visualization to samples/step_{step:06d}.png (sampling: {sampling_info})")

                predictor.train()

        if is_main and step % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f'ckpt_step_{step}.pt')
            model_state = predictor.module.state_dict() if world_size > 1 else predictor.state_dict()


            ckpt_dict = {
                'model': model_state,
                'optimizer': optim.state_dict(),
                'step': step,
                'bos_token': args.bos_token,
                'vocab_size': vocab_size,
                'model_vocab_size': model_vocab_size,
                'd_model': args.d_model,
                'num_layers': args.num_layers,
                'nhead': args.nhead,
            }


            torch.save(ckpt_dict, ckpt_path)
            print(f"[Step {step}] Saved checkpoint to {ckpt_path}")

    wait_for_everyone()
    if is_main:
        print('Training complete')


if __name__ == '__main__':
    main()
