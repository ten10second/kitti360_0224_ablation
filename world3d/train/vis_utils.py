"""Visualization utilities for training.

This module intentionally contains side-effect heavy debug visualization code
(e.g. writing PNG/CSV) to keep the main training loop cleaner.

All functions here should be safe to call under torch.no_grad().
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _as_contiguous_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is a contiguous uint8 HxWx3 RGB array for OpenCV."""
    if img is None:
        return img
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img)}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got {img.shape}")
    return np.ascontiguousarray(img)


def _to_uint8_rgb(img) -> Optional[np.ndarray]:
    if img is None:
        return None
    if torch.is_tensor(img):
        t = img.detach()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] == 3:
            t = t.permute(1, 2, 0)
        arr = t.float().cpu().numpy()
        # assume either [0,1] or [-1,1]
        if arr.min() < 0:
            arr = arr * 0.5 + 0.5
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = img
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy array or torch tensor, got {type(img)}")
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            # assume [0,1]
            arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {arr.shape}")
    return np.ascontiguousarray(arr)


def render_bev_attn_heatmap(
    attn_weights: np.ndarray,
    sat_img: Optional[np.ndarray] = None,
    bev_size: int = 64,
    anchor_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render BEV attention heatmap, optionally overlaid on satellite image.

    Args:
        attn_weights: (nhead, N_q, N_kv) or (64, 64) attention weights.
        sat_img: (H, W, 3) uint8 RGB satellite image as background.
        bev_size: spatial size of BEV grid (default 64).
        anchor_points: (N_q, 2) anchor coordinates in [-1, 1] BEV-normalized.
            If provided, draws anchor points as cyan dots on the heatmap.

    Returns:
        result: (H, W, 3) uint8 RGB image with heatmap overlay.
    """
    if attn_weights.ndim == 3:
        heatmap = attn_weights.mean(axis=(0, 1))  # (N_kv,)
    elif attn_weights.ndim == 2:
        heatmap = attn_weights.ravel()
    else:
        heatmap = attn_weights.ravel()

    if heatmap.size == bev_size * bev_size:
        heatmap = heatmap.reshape(bev_size, bev_size)
    else:
        side = int(np.sqrt(heatmap.size))
        heatmap = heatmap[:side * side].reshape(side, side)

    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 1e-8:
        heatmap = heatmap / heatmap.max()
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if sat_img is not None:
        h_sat, w_sat = sat_img.shape[:2]
        heatmap_color = cv2.resize(heatmap_color, (w_sat, h_sat), interpolation=cv2.INTER_LINEAR)
        result = cv2.addWeighted(sat_img, 0.4, heatmap_color, 0.6, 0)
    else:
        h_sat, w_sat = 512, 512
        result = cv2.resize(heatmap_color, (w_sat, h_sat), interpolation=cv2.INTER_LINEAR)

    # Draw anchor points if provided
    if anchor_points is not None:
        for i in range(anchor_points.shape[0]):
            ax, ay = anchor_points[i]  # in [-1, 1]
            # Map [-1, 1] → pixel coords: center of image ± half
            px = int((ax + 1.0) / 2.0 * w_sat)
            py = int((ay + 1.0) / 2.0 * h_sat)
            px = np.clip(px, 0, w_sat - 1)
            py = np.clip(py, 0, h_sat - 1)
            cv2.circle(result, (px, py), 3, (0, 255, 255), -1)   # cyan filled
            cv2.circle(result, (px, py), 4, (0, 0, 0), 1)        # black outline

    return np.ascontiguousarray(result)


IMU_TO_GROUND_HEIGHT_VIS = 0.93


def render_sat_with_frustum(
    sat_img: np.ndarray,
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    T_imu_to_world: np.ndarray,
    sat_resolution: float = 0.196,
    cam_h: int = 256,
    cam_w: int = 640,
    fill_alpha: float = 0.35,
    fill_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw camera FOV frustum on satellite image.

    Args:
        sat_img: (H, W, 3) uint8 RGB satellite image.
        K: (3, 3) camera intrinsics.
        T_cam_to_world: (4, 4) camera-to-world transform.
        T_imu_to_world: (4, 4) IMU-to-world transform (satellite center).
        sat_resolution: meters per pixel in satellite image.
        cam_h, cam_w: camera image dimensions for frustum corner rays.

    Returns:
        canvas: (H, W, 3) uint8 RGB with frustum drawn.
    """
    canvas = sat_img.copy()
    H_sat, W_sat = canvas.shape[:2]

    imu_pos = T_imu_to_world[:3, 3]
    t_cam = T_cam_to_world[:3, 3]
    R_cam = T_cam_to_world[:3, :3]
    ground_z = float(imu_pos[2]) - IMU_TO_GROUND_HEIGHT_VIS

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    corners_uv = np.array([
        [0, 0], [cam_w - 1, 0], [cam_w - 1, cam_h - 1], [0, cam_h - 1],
    ], dtype=np.float64)

    frustum_pts = []
    for u, v in corners_uv:
        d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        d_world = R_cam @ d_cam
        if abs(d_world[2]) < 1e-6:
            t = 50.0 / max(np.linalg.norm(d_world[:2]), 1e-6)
        else:
            t = (ground_z - t_cam[2]) / d_world[2]
        if t < 0:
            t = 50.0 / max(np.linalg.norm(d_world[:2]), 1e-6)
        pt_world = t_cam + t * d_world
        u_bev = (pt_world[0] - imu_pos[0]) / sat_resolution + W_sat / 2.0
        v_bev = -(pt_world[1] - imu_pos[1]) / sat_resolution + H_sat / 2.0
        frustum_pts.append([int(np.clip(u_bev, -5000, 5000)), int(np.clip(v_bev, -5000, 5000))])

    if len(frustum_pts) >= 3:
        pts = np.array(frustum_pts, dtype=np.int32)
        if fill_alpha > 0:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], fill_color)
            canvas = cv2.addWeighted(overlay, fill_alpha, canvas, 1 - fill_alpha, 0)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)

    u_cam_bev = int((t_cam[0] - imu_pos[0]) / sat_resolution + W_sat / 2.0)
    v_cam_bev = int(-(t_cam[1] - imu_pos[1]) / sat_resolution + H_sat / 2.0)
    cv2.circle(canvas, (u_cam_bev, v_cam_bev), 5, (255, 255, 0), -1)
    cv2.circle(canvas, (u_cam_bev, v_cam_bev), 8, (0, 0, 0), 2)

    return np.ascontiguousarray(canvas)


def _resize_panel(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize an image to target height, preserving aspect ratio (width scales proportionally)."""
    h, w = img.shape[:2]
    scale = target_h / h
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_samples_grid_step_debug(
    *,
    step: int,
    out_dir: str,
    gt_img_vis,
    inverse_proj_vis=None,
    inverse_valid_mask_vis: Optional[np.ndarray] = None,
    vq_recon_vis=None,
    generated_vis=None,
    bev_attn_heatmap_vis: Optional[np.ndarray] = None,
    sat_frustum_vis: Optional[np.ndarray] = None,
    is_main: bool = True,
    subdir: str = "samples",
) -> None:
    """Save debug sample grid.

    Layout when IPM is enabled:
        Row 1: Ground Truth | IPM (+ mask contour) | BEV Attn Heatmap
        Row 2: VQ Recon     | Generated            | Satellite + Frustum

    Layout when IPM is disabled:
        Row 1: Ground Truth | Generated
        Row 2: BEV Attn Heatmap | Satellite + Frustum
    """
    os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

    gt_np = _to_uint8_rgb(gt_img_vis)
    if gt_np is None:
        if is_main:
            print("[Vis] save_samples_grid_step_debug: gt_img_vis is None, skipping")
        return

    img_h, img_w = gt_np.shape[:2]

    show_ipm = inverse_proj_vis is not None
    inv_np = _to_uint8_rgb(inverse_proj_vis) if show_ipm else None
    recon_np = _to_uint8_rgb(vq_recon_vis) if vq_recon_vis is not None else np.zeros_like(gt_np)
    gen_np = _to_uint8_rgb(generated_vis) if generated_vis is not None else np.zeros_like(gt_np)

    if show_ipm and inverse_valid_mask_vis is not None:
        mask_vis = inverse_valid_mask_vis
        if torch.is_tensor(mask_vis):
            mask_vis = mask_vis.detach().cpu().numpy()
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

    # Prepare right-column panels (BEV heatmap and satellite+frustum)
    has_right_col = bev_attn_heatmap_vis is not None or sat_frustum_vis is not None
    if has_right_col:
        # Resize square panels to match camera image dimensions
        if bev_attn_heatmap_vis is not None:
            heatmap_np = _resize_panel(
                _as_contiguous_uint8_rgb(bev_attn_heatmap_vis), img_h, img_w
            )
        else:
            heatmap_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        if sat_frustum_vis is not None:
            sat_np = _resize_panel(
                _as_contiguous_uint8_rgb(sat_frustum_vis), img_h, img_w
            )
        else:
            sat_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        if show_ipm:
            # 2x3 grid (with IPM)
            top_row = np.concatenate([gt_np, inv_np, heatmap_np], axis=1)
            bottom_row = np.concatenate([recon_np, gen_np, sat_np], axis=1)
        else:
            # 2x2 grid (IPM removed)
            top_row = np.concatenate([gt_np, gen_np], axis=1)
            bottom_row = np.concatenate([heatmap_np, sat_np], axis=1)
    else:
        if show_ipm:
            # Fallback 2x2 grid (with IPM)
            top_row = np.concatenate([gt_np, inv_np], axis=1)
            bottom_row = np.concatenate([recon_np, gen_np], axis=1)
        else:
            # Fallback 2x2 grid (IPM removed)
            top_row = np.concatenate([gt_np, gen_np], axis=1)
            bottom_row = np.concatenate([recon_np, np.zeros_like(gt_np)], axis=1)

    grid = np.concatenate([top_row, bottom_row], axis=0)

    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont

        grid_pil = PILImage.fromarray(grid)
        draw = ImageDraw.Draw(grid_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        if show_ipm:
            labels: Dict[tuple, str] = {
                (10, 10): "Ground Truth",
                (img_w + 10, 10): "IPM",
                (10, img_h + 10): "VQ Recon",
                (img_w + 10, img_h + 10): "Generated",
            }
            if has_right_col:
                labels[(2 * img_w + 10, 10)] = "BEV Attn Heatmap"
                labels[(2 * img_w + 10, img_h + 10)] = "Satellite + Frustum"
        else:
            labels = {
                (10, 10): "Ground Truth",
                (img_w + 10, 10): "Generated",
            }
            if has_right_col:
                labels[(10, img_h + 10)] = "BEV Attn Heatmap"
                labels[(img_w + 10, img_h + 10)] = "Satellite + Frustum"
            else:
                labels[(10, img_h + 10)] = "VQ Recon"

        for (x, y), label in labels.items():
            # 调整第三列标签的字体大小
            if show_ipm and x >= 2 * img_w:
                draw.text((x, y), label, fill=(255, 255, 0), font=font_small)
            else:
                draw.text((x, y), label, fill=(255, 255, 0), font=font)

        save_path = os.path.join(out_dir, subdir, f"step_{step:06d}.png")
        grid_pil.save(save_path)
        if is_main:
            print(f"[Vis] Saved samples grid to {save_path}")
    except Exception as e:
        save_path = os.path.join(out_dir, subdir, f"step_{step:06d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        if is_main:
            print(f"[Vis] Saved samples grid (no labels) to {save_path} (PIL failed: {e})")


def plot_loss_curve(loss_history, out_dir: str, current_step: int) -> None:
    """Plot training loss curve and save to file."""
    if len(loss_history) == 0:
        return

    # Support extended tuples:
    # (step, total, ce, consistency)
    # (step, total, ce, consistency, geometric)
    # (step, total, ce, consistency, consistency_cos, consistency_nce,
    #  consistency_nce_weight, matched_pairs, valid_view_pairs, overlap_ratio)
    sample_entry = loss_history[0]
    tuple_len = len(sample_entry) if isinstance(sample_entry, (tuple, list)) else 0
    has_ce = tuple_len >= 3
    has_consistency = tuple_len >= 4
    has_geometric = tuple_len == 5
    has_consistency_breakdown = tuple_len >= 10

    steps = [entry[0] for entry in loss_history]
    losses = [entry[1] for entry in loss_history]

    # Extract separate loss components if available
    ce_losses = []
    consistency_losses = []
    geometric_losses = []

    for entry in loss_history:
        if has_ce and len(entry) > 2:
            ce_losses.append(entry[2])
        if has_consistency and len(entry) > 3:
            consistency_losses.append(entry[3])
        if has_geometric and len(entry) > 4:
            geometric_losses.append(entry[4])

    plt.figure(figsize=(12, 6))

    # Plot 1: Full loss curve
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses, linewidth=1.5, alpha=0.7, label='Training Loss')

    # Add smoothed curve (moving average)
    if len(losses) > 10:
        window_size = min(50, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
        smoothed_steps = steps[window_size - 1:]
        plt.plot(smoothed_steps, smoothed, linewidth=2, color='red', label=f'Smoothed (window={window_size})')

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss Curve (Step {current_step})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Plot 2: Recent loss (last 20% of training)
    plt.subplot(1, 2, 2)
    recent_start_idx = max(0, len(loss_history) - len(loss_history) // 5)
    recent_steps = steps[recent_start_idx:]
    recent_losses = losses[recent_start_idx:]

    plt.plot(recent_steps, recent_losses, linewidth=1.5, alpha=0.7, label='Recent Loss')

    if len(recent_losses) > 10:
        window_size = min(20, len(recent_losses) // 5)
        smoothed = np.convolve(recent_losses, np.ones(window_size) / window_size, mode='valid')
        smoothed_steps = recent_steps[window_size - 1:]
        plt.plot(smoothed_steps, smoothed, linewidth=2, color='red', label=f'Smoothed (window={window_size})')

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Recent Loss (Last 20%)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    loss_txt_path = os.path.join(out_dir, 'loss_history.txt')
    with open(loss_txt_path, 'w') as f:
        if has_consistency_breakdown:
            f.write('# step\ttotal_loss\tce_loss\tconsistency_loss\tconsistency_cos\tconsistency_nce\tconsistency_nce_weight\tmatched_pairs\tvalid_view_pairs\toverlap_ratio\n')
        elif has_ce and has_consistency and has_geometric:
            f.write('# step\ttotal_loss\tce_loss\tconsistency_loss\tgeometric_loss\n')
        elif has_ce and has_consistency:
            f.write('# step\ttotal_loss\tce_loss\tconsistency_loss\n')
        elif has_ce:
            f.write('# step\ttotal_loss\tce_loss\n')
        else:
            f.write('# step\tloss\n')

        for entry in loss_history:
            step = entry[0]
            total_loss = entry[1]
            if has_ce:
                ce_loss = entry[2]
            if has_consistency:
                consistency_loss = entry[3]
            if has_geometric:
                geometric_loss = entry[4]
            if has_consistency_breakdown:
                consistency_cos = entry[4]
                consistency_nce = entry[5]
                consistency_nce_weight = entry[6]
                matched_pairs = entry[7]
                valid_view_pairs = entry[8]
                overlap_ratio = entry[9]

            if has_consistency_breakdown:
                f.write(
                    f'{step}\t{total_loss:.6f}\t{ce_loss:.6f}\t{consistency_loss:.6f}\t'
                    f'{consistency_cos:.6f}\t{consistency_nce:.6f}\t{consistency_nce_weight:.6f}\t'
                    f'{matched_pairs:.6f}\t{valid_view_pairs:.6f}\t{overlap_ratio:.6f}\n'
                )
            elif has_ce and has_consistency and has_geometric:
                f.write(f'{step}\t{total_loss:.6f}\t{ce_loss:.6f}\t{consistency_loss:.6f}\t{geometric_loss:.6f}\n')
            elif has_ce and has_consistency:
                f.write(f'{step}\t{total_loss:.6f}\t{ce_loss:.6f}\t{consistency_loss:.6f}\n')
            elif has_ce:
                f.write(f'{step}\t{total_loss:.6f}\t{ce_loss:.6f}\n')
            else:
                f.write(f'{step}\t{total_loss:.6f}\n')

    print(f"[Plot] Saved loss curve to {save_path}")


def plot_anchor_stage2_loss_curve(loss_history, out_dir: str, current_step: int) -> None:
    """Plot baseline/target conditional CE curves for anchor-view stage2."""
    if len(loss_history) == 0:
        return

    steps = [entry["step"] for entry in loss_history]
    baseline_losses = [entry["ce_loss"] for entry in loss_history]
    target_losses = [entry["anchor_ce_loss"] for entry in loss_history]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, baseline_losses, linewidth=1.5, alpha=0.85, label='Baseline CE')
    plt.plot(steps, target_losses, linewidth=1.5, alpha=0.85, label='Target Cond CE')
    if len(steps) > 10:
        window_size = min(50, max(5, len(steps) // 10))
        baseline_smoothed = np.convolve(baseline_losses, np.ones(window_size) / window_size, mode='valid')
        target_smoothed = np.convolve(target_losses, np.ones(window_size) / window_size, mode='valid')
        smoothed_steps = steps[window_size - 1:]
        plt.plot(smoothed_steps, baseline_smoothed, linewidth=2, linestyle='--', label=f'Baseline Smooth ({window_size})')
        plt.plot(smoothed_steps, target_smoothed, linewidth=2, linestyle='--', label=f'Target Smooth ({window_size})')

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Cross Entropy', fontsize=12)
    plt.title(f'Anchor Stage2 CE Curves (Step {current_step})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    plt.subplot(1, 2, 2)
    recent_start_idx = max(0, len(loss_history) - max(20, len(loss_history) // 5))
    recent_steps = steps[recent_start_idx:]
    recent_baseline = baseline_losses[recent_start_idx:]
    recent_target = target_losses[recent_start_idx:]
    plt.plot(recent_steps, recent_baseline, linewidth=1.5, alpha=0.85, label='Baseline CE')
    plt.plot(recent_steps, recent_target, linewidth=1.5, alpha=0.85, label='Target Cond CE')
    if len(recent_steps) > 10:
        window_size = min(20, max(5, len(recent_steps) // 5))
        baseline_smoothed = np.convolve(recent_baseline, np.ones(window_size) / window_size, mode='valid')
        target_smoothed = np.convolve(recent_target, np.ones(window_size) / window_size, mode='valid')
        smoothed_steps = recent_steps[window_size - 1:]
        plt.plot(smoothed_steps, baseline_smoothed, linewidth=2, linestyle='--', label=f'Baseline Smooth ({window_size})')
        plt.plot(smoothed_steps, target_smoothed, linewidth=2, linestyle='--', label=f'Target Smooth ({window_size})')

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Cross Entropy', fontsize=12)
    plt.title('Recent CE Curves', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'anchor_ce_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Plot] Saved anchor CE curve to {save_path}")


def visualize_bev_pair_consistency(
    img_s: torch.Tensor,
    img_t: torch.Tensor,
    bev_s: torch.Tensor,
    bev_t: torch.Tensor,
    valid_s: torch.Tensor,
    valid_t: torch.Tensor,
    step: int,
    out_dir: str,
    pair_name: str = "pair",
) -> None:
    """Visualize BEV pair consistency for debugging.

    Creates a 2x3 grid showing:
    Row 1: Source view | Target view | Overlap mask
    Row 2: Source BEV  | Target BEV  | Difference map

    Args:
        img_s: (1, 3, H, W) source view image in [0, 1]
        img_t: (1, 3, H, W) target view image in [0, 1]
        bev_s: (1, 3, S, S) source BEV projection in [0, 1]
        bev_t: (1, 3, S, S) target BEV projection in [0, 1]
        valid_s: (1, 1, S, S) source valid mask
        valid_t: (1, 1, S, S) target valid mask
        step: training step
        out_dir: output directory
        pair_name: name of the pair (e.g., "front_left30")
    """
    os.makedirs(os.path.join(out_dir, "bev_pairs"), exist_ok=True)

    # Convert to numpy
    img_s_np = _to_uint8_rgb(img_s)
    img_t_np = _to_uint8_rgb(img_t)
    bev_s_np = _to_uint8_rgb(bev_s)
    bev_t_np = _to_uint8_rgb(bev_t)

    if img_s_np is None or img_t_np is None or bev_s_np is None or bev_t_np is None:
        return

    # Compute overlap mask
    overlap = (valid_s[0, 0] & valid_t[0, 0]).cpu().numpy()  # (S, S)
    overlap_vis = (overlap * 255).astype(np.uint8)
    overlap_rgb = cv2.applyColorMap(overlap_vis, cv2.COLORMAP_JET)

    # Compute difference map (only in overlap region)
    diff = torch.abs(bev_s - bev_t).mean(dim=1, keepdim=True)  # (1, 1, S, S)
    diff_masked = diff[0, 0].cpu().numpy() * overlap  # (S, S)
    diff_vis = (np.clip(diff_masked * 5.0, 0, 1) * 255).astype(np.uint8)  # amplify for visibility
    diff_rgb = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)

    # Resize images to match BEV size for grid
    bev_h, bev_w = bev_s_np.shape[:2]
    img_s_resized = cv2.resize(img_s_np, (bev_w, bev_h))
    img_t_resized = cv2.resize(img_t_np, (bev_w, bev_h))

    # Create 2x3 grid
    row1 = np.hstack([img_s_resized, img_t_resized, overlap_rgb])
    row2 = np.hstack([bev_s_np, bev_t_np, diff_rgb])
    grid = np.vstack([row1, row2])

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)

    labels = [
        ("Source View", (10, 30)),
        ("Target View", (bev_w + 10, 30)),
        ("Overlap Mask", (2 * bev_w + 10, 30)),
        ("Source BEV", (10, bev_h + 30)),
        ("Target BEV", (bev_w + 10, bev_h + 30)),
        ("Difference", (2 * bev_w + 10, bev_h + 30)),
    ]

    for text, (x, y) in labels:
        cv2.putText(grid, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Save
    save_path = os.path.join(out_dir, "bev_pairs", f"step_{step:07d}_{pair_name}.png")
    cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[BEV Pair] Saved {pair_name} visualization to {save_path}")


def visualize_diffusion_results(
    generated_images: torch.Tensor,
    gt_images: torch.Tensor,
    sat_images: torch.Tensor,
    save_path: str,
) -> None:
    """
    Visualize diffusion model results by creating a comparison grid.

    Args:
        generated_images: (B, 3, H, W) generated images in [-1, 1]
        gt_images: (B, 3, H, W) ground truth images in [-1, 1]
        sat_images: (B, 3, H, W) satellite images in [0, 1]
        save_path: path to save visualization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Take first batch element for visualization
    gen_img = generated_images[0]
    gt_img = gt_images[0]
    sat_img = sat_images[0]

    # Convert to RGB uint8
    gen_np = _to_uint8_rgb(gen_img)
    gt_np = _to_uint8_rgb(gt_img)
    sat_np = _to_uint8_rgb(sat_img)

    # Resize all images to the same height for grid
    img_h, img_w = gt_np.shape[:2]
    gen_np = cv2.resize(gen_np, (img_w, img_h))
    sat_np = cv2.resize(sat_np, (img_w, img_h))

    # Create comparison grid (Gen | GT | SAT)
    grid = np.concatenate([gen_np, gt_np, sat_np], axis=1)

    # Add labels using PIL
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont

        grid_pil = PILImage.fromarray(grid)
        draw = ImageDraw.Draw(grid_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        labels: Dict[tuple, str] = {
            (10, 10): "Generated",
            (img_w + 10, 10): "Ground Truth",
            (2 * img_w + 10, 10): "Satellite",
        }

        for (x, y), label in labels.items():
            draw.text((x, y), label, fill=(255, 255, 0), font=font)

        grid_pil.save(save_path)
    except Exception as e:
        # Fallback to OpenCV if PIL fails
        for i, label in enumerate(["Generated", "Ground Truth", "Satellite"]):
            x = i * img_w + 10
            y = 10
            cv2.putText(
                grid, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA
            )
        cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    print(f"[Vis] Saved diffusion results to {save_path}")


def save_pair_consistency_metrics(
    pair_losses: Dict[str, float],
    step: int,
    out_dir: str,
) -> None:
    """Save per-pair consistency metrics to CSV for tracking.

    Args:
        pair_losses: dict mapping pair names to loss values
        step: training step
        out_dir: output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pair_consistency_metrics.csv")

    # Write header if file doesn't exist
    write_header = not os.path.exists(csv_path)

    with open(csv_path, 'a') as f:
        if write_header:
            pair_names = sorted(pair_losses.keys())
            f.write(f"step,{','.join(pair_names)}\n")

        values = [str(pair_losses[k]) for k in sorted(pair_losses.keys())]
        f.write(f"{step},{','.join(values)}\n")
