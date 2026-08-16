#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vanilla AR Yaw Sweep Inference - for ar_vanilla.yaml trained models.

This script is specifically designed for the "vanilla" mode models trained
with ar_vanilla.yaml. It supports both fixed-5 views and 360° sweep modes.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from models.stage2.simplified_token_predictor import SimplifiedTokenPredictor
from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.train.geometry_ar import compute_inverse_projection_view
from world3d.train.conditioning_ar import build_condition_tokens_with_coords
from world3d.train.pose_ar import build_pose_vec
from world3d.data.ar_pipeline import compute_bev_visibility_mask
from world3d.train.vis_utils import render_bev_attn_heatmap
def get_decoder_cross_attn(model):
    """Get cross-attention weights from decoder layers.

    Returns:
        mean_attn: (N_q, N_kv) mean attention over layers
    """
    all_attns = []

    for i, layer in enumerate(model.decoder_layers):
        if hasattr(layer.multihead_attn, "_last_attn_weights"):
            attn = layer.multihead_attn._last_attn_weights
            # attn shape: (B, T, S) - already averaged over heads by nn.MultiheadAttention
            all_attns.append(attn)

    if all_attns:
        # Stack and average over layers
        stacked = torch.stack(all_attns)  # (L, B, T, S)
        mean_attn = stacked.mean(dim=(0, 1))  # (T, S)
        return mean_attn.cpu().numpy()
    return None


# Model config (from ar_vanilla.yaml)
D_MODEL = 512
NUM_LAYERS = 8
NHEAD = 8
GRID_ROWS = 16
GRID_COLS = 40
SEQ_LEN = GRID_ROWS * GRID_COLS  # 640
VOCAB_SIZE = 1025  # 1024 codebook + 1 BOS
BOS_TOKEN = 1024
TARGET_H = GRID_ROWS * 16  # 256
TARGET_W = GRID_COLS * 16  # 640


def parse_args():
    p = argparse.ArgumentParser(description="Vanilla AR Yaw Sweep Inference")
    p.add_argument("--ckpt", default=str(REPO_ROOT / "runs/vanilla_ar/ckpt_step_0080000.pt"))
    p.add_argument("--vq-ckpt", default=str(REPO_ROOT / "ckpts/maskgit-vqgan-imagenet-f16-256.bin"))
    p.add_argument("--data-root", default="/media/zhimiao/Lenovo/KITTI-360")
    p.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs/vanilla_ar/yaw_sweep"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--mode", default="default", choices=["default", "fixed5", "zero_shot", "360sweep"],
                    help="default: run fixed5 + zero_shot test views | fixed5: run only fixed5 views | zero_shot: run only held-out zero-shot views | 360sweep: 360° sweep on frame 113")
    p.add_argument("--test-drives", type=str, nargs="+",
                    default=[
                        "2013_05_28_drive_0003_sync:978,979",
                        "2013_05_28_drive_0000_sync:10926,10927",
                        "2013_05_28_drive_0002_sync:18277,18278",
                        "2013_05_28_drive_0005_sync:6386,6387",
                        "2013_05_28_drive_0006_sync:9213,9214",
                        "2013_05_28_drive_0007_sync:3003,3004",
                        "2013_05_28_drive_0009_sync:13257,13258",
                        "2013_05_28_drive_0010_sync:3555,3557",
                    ],
                    help="Test drive and frame IDs in format 'drive:frame1,frame2' (for default/fixed5/zero_shot mode)")
    p.add_argument("--sweep-frame", type=int, default=113,
                    help="Frame ID for 360° sweep (for 360sweep mode)")
    p.add_argument("--sweep-interval", type=int, default=20,
                    help="Angle interval in degrees for 360° sweep")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────

def load_model(ckpt_path, device):
    """Load SimplifiedTokenPredictor in VANILLA mode."""
    model = SimplifiedTokenPredictor(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS,
        nhead=NHEAD, dropout=0.0, max_seq_len=SEQ_LEN,
        target_rows=GRID_ROWS, target_cols=GRID_COLS, semantic_dim=4,
        fourier_freqs=10, train_bev_encoder=False, no_bev_pretrain=True,
        pose_dim=13, use_pose_token=True, n_pose_queries=64,
        mode="vanilla"  # CRITICAL: Load in vanilla mode
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[Model] Loaded VANILLA AR from {ckpt_path} (step {ckpt.get('step', '?')})")
    return model


# ── AR generation ─────────────────────────────────────────────────

@torch.no_grad()
def top_k_sample(logits, k=50, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    k = min(k, logits.size(-1))
    top_vals, top_idx = torch.topk(logits, k, dim=-1)
    probs = F.softmax(top_vals, dim=-1)
    sampled = torch.multinomial(probs, 1)
    return torch.gather(top_idx, -1, sampled).squeeze(-1)


@torch.no_grad()
def generate_ar(model, condition, bev, bev_vis_mask, device, top_k=50, temperature=1.0):
    generated = torch.full((1, 1), BOS_TOKEN, dtype=torch.long, device=device)
    past_kv = None
    for step in range(SEQ_LEN):
        inp = generated if past_kv is None else generated[:, -1:]
        logits, past_kv = model(
            generated_tokens=inp,
            condition_tokens=condition,
            aligned_bev_feature_map=bev,
            bev_vis_mask=bev_vis_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        next_tok = top_k_sample(logits[:, -1, :1024], k=top_k, temperature=temperature)
        generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
    token_seq = generated[:, 1:]
    return model.seq_to_grid(token_seq)


# ── Prepare sample using training pipeline ────────────────────────

@torch.no_grad()
def prepare_sample(sample, vq, bev_encoder, device):
    """Process a Kitti360dDataset sample into model inputs, same as ArTransformDataset."""
    rgb = sample["image"].to(device)
    rgb_norm = rgb * 2.0 - 1.0  # [0,1] -> [-1,1]
    K = sample["K"].to(device)
    T_cam_to_world = sample["T_cam_to_world"].to(device)
    T_imu_to_world = sample["T_imu_to_world"].to(device)
    sat = sample["sat"].to(device)

    # IPM warp (used for visualization only in vanilla mode)
    warped_front, warped_valid, warped_coords = compute_inverse_projection_view(
        sat, K, T_cam_to_world, T_imu_to_world, TARGET_H, TARGET_W, device,
    )

    # Semantic conditioning (built for model compatibility - vanilla mode doesn't use them)
    sem_dict, coord_dict = build_condition_tokens_with_coords(
        warped_front, warped_coords, warped_valid, GRID_ROWS, GRID_COLS, device,
    )

    # Pose vector
    pose = build_pose_vec(K, T_cam_to_world, T_imu_to_world, TARGET_H, TARGET_W, device)

    # BEV features
    bev_feats = bev_encoder(sat.unsqueeze(0))

    # BEV visibility mask
    bev_vis = compute_bev_visibility_mask(
        K=K, T_cam_to_world=T_cam_to_world, T_imu_to_world=T_imu_to_world,
        cam_h=TARGET_H, cam_w=TARGET_W,
    )

    # IPM image for visualization
    ipm_img = warped_front[0] if warped_front is not None else torch.zeros(3, TARGET_H, TARGET_W, device=device)

    return {
        "condition": {
            "semantic": sem_dict["fine"].unsqueeze(0),
            "coords": coord_dict["fine"].unsqueeze(0),
            "pose": pose.unsqueeze(0),
            "K": K.unsqueeze(0),
        },
        "bev": bev_feats,
        "bev_vis_mask": bev_vis.unsqueeze(0),
        "ipm_img": ipm_img,
        "rgb": rgb,
        "sat": sat,
        "K_raw": K,
        "T_cam_to_world": T_cam_to_world,
        "T_imu_to_world": T_imu_to_world,
    }


def get_fixed_five_views():
    """Return the fixed 5 training-view specifications."""
    return [
        ("front", "front", None, None),  # (name, source, fisheye_camera, yaw_deg)
        ("left_to_front_30", "image_02", "image_02", 30.0),
        ("right_to_front_30", "image_03", "image_03", -30.0),
        ("left_axis", "image_02", "image_02", 0.0),
        ("right_axis", "image_03", "image_03", 0.0),
    ]


def get_zero_shot_views():
    """Return held-out zero-shot view specifications."""
    return [
        ("left_back_30", "image_02", "image_02", -30.0),
        ("right_back_30", "image_03", "image_03", 30.0),
    ]


def get_requested_view_groups(mode):
    if mode == "default":
        return [
            ("fixed5", get_fixed_five_views(), "fixed5"),
            ("zero_shot", get_zero_shot_views(), "zero_shot"),
        ]
    if mode == "fixed5":
        return [("fixed5", get_fixed_five_views(), "fixed5")]
    if mode == "zero_shot":
        return [("zero_shot", get_zero_shot_views(), "zero_shot")]
    raise ValueError(f"Unsupported grouped view mode: {mode}")


# ── Visualization ─────────────────────────────────────────────────

def tensor_to_pil(img_tensor):
    """(3, H, W) in [-1, 1] -> PIL"""
    arr = (img_tensor.cpu().float() * 0.5 + 0.5).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def tensor01_to_pil(img_tensor):
    """(3, H, W) in [0, 1] -> PIL"""
    arr = img_tensor.cpu().float().clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_yaw_grid(rows, frame_id, out_dir, drive=None, is_vanilla=True):
    """Save visualization grid.

    rows: list of (label, bev_vis_pil, ipm_pil, gen_pil, gt_pil)
    Layout: one row per yaw angle, columns = BEV Atten / Satellite | IPM | Gen | GT
    """
    W, H = rows[0][2].size  # 640 x 256 (from IPM)
    sat_W = H  # satellite is square, scale to row height
    total_W = sat_W + 3 * W
    nrows = len(rows)
    pad = 30

    grid = Image.new("RGB", (total_W, nrows * H + pad), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # Header
    mode_label = "Vanilla AR" if is_vanilla else "Ours"
    draw.text((total_W // 2 - 50, 5), mode_label, fill=(0, 255, 255), font=font)

    headers = [("BEV/Sat", sat_W // 2 - 40),
               ("IPM", sat_W + W // 2 - 20),
               ("Generated", sat_W + W + W // 2 - 40),
               ("GT", sat_W + 2 * W + W // 2 - 10)]
    for hdr, x in headers:
        draw.text((x, pad - 25), hdr, fill=(255, 255, 255), font=font)

    for r, (label, bev_vis_img, ipm_img, gen_img, gt_img) in enumerate(rows):
        y = r * H + pad
        bev_vis_resized = bev_vis_img.resize((sat_W, H), Image.LANCZOS)
        grid.paste(bev_vis_resized, (0, y))
        grid.paste(ipm_img, (sat_W, y))
        grid.paste(gen_img, (sat_W + W, y))
        grid.paste(gt_img, (sat_W + 2 * W, y))
        draw.text((5, y + 5), label, fill=(0, 255, 255), font=font)

    if drive:
        path = os.path.join(out_dir, f"vanilla_{drive}_frame_{frame_id:010d}_yaw_sweep.png")
    else:
        path = os.path.join(out_dir, f"vanilla_frame_{frame_id:010d}_yaw_sweep.png")
    grid.save(path)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────

def run_fixed_five_views(args, model, vq, bev_encoder, device, data_root):
    """Test grouped views on test frames from multiple drives."""
    print("\n" + "=" * 60)
    if args.mode == "default":
        print("VANILLA AR: Fixed 5 + Zero-shot Views on Test Frames from Multiple Drives")
    elif args.mode == "zero_shot":
        print("VANILLA AR: Zero-shot Views on Test Frames from Multiple Drives")
    else:
        print("VANILLA AR: Fixed 5 Views on Test Frames from Multiple Drives")
    print("=" * 60)

    view_groups = get_requested_view_groups(args.mode)
    inference_times = []

    test_drive_frames = []
    for drive_frame_str in args.test_drives:
        drive, frame_str = drive_frame_str.split(':')
        frames = list(map(int, frame_str.split(',')))
        test_drive_frames.append((drive, frames))

    for drive, frames in test_drive_frames:
        drive_path = Path(data_root) / drive
        print(f"\n\nDrive: {drive}")
        drive_safe = drive.replace('/', '_').replace('\\', '_')

        for frame_id in frames:
            print(f"\n[Frame {frame_id}]")

            for group_name, group_views, subdir in view_groups:
                rows = []
                print(f"  [{group_name}]")
                group_out_dir = os.path.join(args.out_dir, subdir)

                for view_name, source, fisheye_cam, yaw_deg in group_views:
                    print(f"    {view_name} ...")

                    if source == "front":
                        ds = Kitti360dDataset(
                            drives=str(drive_path),
                            frames=[frame_id],
                            mode="front",
                            virtual_hfov_deg=80.0,
                            virtual_size=(640, 256),
                        )
                    else:
                        ds = Kitti360dDataset(
                            drives=str(drive_path),
                            frames=[frame_id],
                            mode="fisheye_virtual",
                            fisheye_camera=fisheye_cam,
                            fisheye_relative_yaw_deg=float(yaw_deg),
                            virtual_hfov_deg=80.0,
                            virtual_size=(640, 256),
                            random_fisheye_relative_yaw=False,
                            calib_yaw_fix_deg=4.0,
                        )

                    sample = ds[0]
                    if sample.get("meta", {}).get("dummy", False):
                        print("      Skipped (dummy sample)")
                        continue

                    data = prepare_sample(sample, vq, bev_encoder, device)
                    gt_pil = tensor01_to_pil(sample["image"])
                    sat_np = (data["sat"].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    sat_pil = tensor01_to_pil(data["sat"])

                    model.eval()
                    bos = torch.full((1, 1), BOS_TOKEN, dtype=torch.long, device=device)
                    _ = model(
                        generated_tokens=bos,
                        condition_tokens=data["condition"],
                        aligned_bev_feature_map=data["bev"],
                        bev_vis_mask=data["bev_vis_mask"],
                        use_cache=False,
                    )

                    attn_w = None
                    anchor_pts = None
                    if hasattr(model, "pose_route"):
                        attn_w = getattr(model.pose_route, "_last_attn_weights", None)
                        anchor_pts = getattr(model.pose_route, "_last_anchors", None)
                    else:
                        attn_w = get_decoder_cross_attn(model)

                    if attn_w is not None:
                        if hasattr(model, "pose_route"):
                            aw_np = attn_w[0].cpu().numpy()
                            anchor_np = anchor_pts[0].cpu().numpy() if anchor_pts is not None else None
                            heatmap_np = render_bev_attn_heatmap(aw_np, sat_img=sat_np.copy(), anchor_points=anchor_np)
                        else:
                            if attn_w.shape[0] > 0:
                                bos_attn = attn_w[0]
                                if bos_attn.size > 1:
                                    bev_attn = bos_attn[1:]
                                    heatmap_np = render_bev_attn_heatmap(bev_attn, sat_img=sat_np.copy())
                                else:
                                    heatmap_np = sat_np
                            else:
                                heatmap_np = sat_np
                        heatmap_pil = Image.fromarray(heatmap_np)
                    else:
                        heatmap_pil = sat_pil

                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    token_grid = generate_ar(
                        model, data["condition"], data["bev"], data["bev_vis_mask"], device,
                        top_k=args.top_k, temperature=args.temperature,
                    )
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    elapsed = time.time() - start_time
                    inference_times.append(elapsed)
                    print(f"      Generation time: {elapsed:.3f}s")
                    gen_img = vq.decode(token_grid)[0]
                    gen_pil = tensor_to_pil(gen_img)

                    rows.append((
                        view_name,
                        heatmap_pil,
                        tensor01_to_pil(data["ipm_img"]),
                        gen_pil,
                        gt_pil,
                    ))

                    scene_dir = os.path.join(group_out_dir, drive_safe)
                    view_name_normalized = view_name.lower().replace(' ', '_').replace('°', '').replace('.', '')
                    view_dir = os.path.join(scene_dir, view_name_normalized)
                    os.makedirs(view_dir, exist_ok=True)
                    gen_pil.save(os.path.join(view_dir, f"frame_{frame_id:010d}_generated.png"))
                    gt_pil.save(os.path.join(view_dir, f"frame_{frame_id:010d}_gt.png"))

                if rows:
                    save_yaw_grid(rows, frame_id, group_out_dir, drive)

    return inference_times


def run_360_sweep(args, model, vq, bev_encoder, device, drive_path):
    """360° sweep on a single frame for vanilla AR."""
    print("\n" + "="*60)
    print(f"VANILLA AR: 360° Sweep on Frame {args.sweep_frame}")
    print(f"Interval: {args.sweep_interval}°")
    print("="*60)

    frame_id = args.sweep_frame
    angles = list(range(0, 360, args.sweep_interval))
    inference_times = []
    rows = []

    for angle in angles:
        print(f"  Angle {angle}° ...")

        # Determine camera and mode based on angle
        if 330 <= angle or angle < 30:
            # Front view: use front camera directly
            mode = "front"
            fisheye_cam = None
            relative_yaw = None
            camera_label = "front"
        elif 30 <= angle < 150:
            # Left side: use left fisheye (image_02)
            mode = "fisheye_virtual"
            fisheye_cam = "image_02"
            relative_yaw = 90.0 - float(angle)
            camera_label = "left_fisheye"
        elif 150 <= angle < 210:
            # Back view: use left fisheye with large yaw
            mode = "fisheye_virtual"
            fisheye_cam = "image_02"
            relative_yaw = 90.0 - float(angle)
            camera_label = "left_fisheye"
        else:  # 210 <= angle < 330
            # Right side: use right fisheye (image_03)
            mode = "fisheye_virtual"
            fisheye_cam = "image_03"
            relative_yaw = 270.0 - float(angle)
            camera_label = "right_fisheye"

        # Create dataset with appropriate camera
        if mode == "front":
            ds = Kitti360dDataset(
                drives=str(drive_path),
                frames=[frame_id],
                mode="front",
                virtual_hfov_deg=80.0,
                virtual_size=(640, 256),
            )
        else:
            ds = Kitti360dDataset(
                drives=str(drive_path),
                frames=[frame_id],
                mode="fisheye_virtual",
                fisheye_camera=fisheye_cam,
                fisheye_relative_yaw_deg=relative_yaw,
                virtual_hfov_deg=80.0,
                virtual_size=(640, 256),
                random_fisheye_relative_yaw=False,
                calib_yaw_fix_deg=4.0,
            )

        sample = ds[0]
        if sample.get("meta", {}).get("dummy", False):
            print(f"    Skipped (dummy sample)")
            continue

        data = prepare_sample(sample, vq, bev_encoder, device)
        gt_pil = tensor01_to_pil(sample["image"])
        sat_np = (data["sat"].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        sat_pil = tensor01_to_pil(data["sat"])

        print(f"    Camera: {camera_label}" + (f", yaw={relative_yaw:.1f}°" if relative_yaw is not None else ""))

        # BEV attention visualization (or fallback to satellite for vanilla mode)
        model.eval()
        bos = torch.full((1, 1), BOS_TOKEN, dtype=torch.long, device=device)
        _ = model(
            generated_tokens=bos,
            condition_tokens=data["condition"],
            aligned_bev_feature_map=data["bev"],
            bev_vis_mask=data["bev_vis_mask"],
            use_cache=False,
        )

        # Try to get attention weights: Ours mode (pose_route) or Vanilla mode (decoder cross-attention)
        attn_w = None
        anchor_pts = None

        if hasattr(model, "pose_route"):
            # Ours mode
            attn_w = getattr(model.pose_route, "_last_attn_weights", None)
            anchor_pts = getattr(model.pose_route, "_last_anchors", None)
        else:
            # Vanilla mode: try to get decoder cross-attention
            attn_w = get_decoder_cross_attn(model)

        if attn_w is not None:
            if hasattr(model, "pose_route"):
                # Ours mode: average over heads and queries
                aw_np = attn_w[0].cpu().numpy()
                anchor_np = anchor_pts[0].cpu().numpy() if anchor_pts is not None else None
                heatmap_np = render_bev_attn_heatmap(aw_np, sat_img=sat_np.copy(), anchor_points=anchor_np)
            else:
                # Vanilla mode: focus on BOS token's attention to BEV grid
                # attn_w shape: (T, S), where T=1 (BOS), S=1 + 64x64 (pose token + bev grid)
                if attn_w.shape[0] > 0:
                    bos_attn = attn_w[0]
                    if bos_attn.size > 1:
                        # Skip pose token
                        bev_attn = bos_attn[1:]
                        heatmap_np = render_bev_attn_heatmap(bev_attn, sat_img=sat_np.copy())
                    else:
                        heatmap_np = sat_np
                else:
                    heatmap_np = sat_np

            heatmap_pil = Image.fromarray(heatmap_np)
        else:
            # Fallback: use satellite image
            heatmap_pil = sat_pil

        # Generate
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        token_grid = generate_ar(
            model, data["condition"], data["bev"], data["bev_vis_mask"], device,
            top_k=args.top_k, temperature=args.temperature,
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time
        inference_times.append(elapsed)
        print(f"    Generation time: {elapsed:.3f}s")
        gen_img = vq.decode(token_grid)[0]

        rows.append((
            f"{angle}°",
            heatmap_pil,
            tensor01_to_pil(data["ipm_img"]),
            tensor_to_pil(gen_img),
            gt_pil,
        ))

    if rows:
        save_yaw_grid(rows, frame_id, args.out_dir)

    return inference_times


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    drive_path = Path(args.data_root) / args.drive

    # Load model and tokenizer
    model = load_model(args.ckpt, device)
    vq = PretrainedTokenizer(args.vq_ckpt).to(device)

    # Get BEV encoder from model
    bev_encoder = model.bev_encoder
    bev_encoder.eval()

    # Run inference based on mode
    if args.mode in {"default", "fixed5", "zero_shot"}:
        inference_times = run_fixed_five_views(args, model, vq, bev_encoder, device, args.data_root)
    elif args.mode == "360sweep":
        drive_path = Path(args.data_root) / args.drive
        inference_times = run_360_sweep(args, model, vq, bev_encoder, device, drive_path)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Print timing statistics
    if inference_times:
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        total_frames = len(inference_times)

        print("\n" + "="*60)
        print("VANILLA AR Inference Timing Statistics")
        print("="*60)
        print(f"Total frames: {total_frames}")
        print(f"Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"Min time: {min_time:.3f}s")
        print(f"Max time: {max_time:.3f}s")
        print(f"FPS: {1.0/avg_time:.2f}")
        print("="*60)

    print("\nDone.")


if __name__ == "__main__":
    main()
