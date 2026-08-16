#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vanilla AR Inference - for ar_vanilla.yaml trained models.

This script is specifically designed for the "vanilla" mode models trained
with ar_vanilla.yaml, which use simplified 2D positional encoding and standard
cross-attention instead of pose-based routing.
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
    p = argparse.ArgumentParser(description="Vanilla AR Inference (for ar_vanilla.yaml models)")
    p.add_argument("--ckpt", default=str(REPO_ROOT / "runs/ar_vanilla/ckpt_step_0080000.pt"))
    p.add_argument("--vq-ckpt", default=str(REPO_ROOT / "ckpts/maskgit-vqgan-imagenet-f16-256.bin"))
    p.add_argument("--data-root", default="/media/zhimiao/Lenovo/KITTI-360")
    p.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    p.add_argument("--frame", type=int, default=113, help="Frame ID to test")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs/ar_vanilla/inference"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.9)
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
    print("[Model] Loaded VANILLA AR from %s (step %s)" % (ckpt_path, ckpt.get("step", "?")))
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
    """Process a Kitti360dDataset sample into model inputs."""
    rgb = sample["image"].to(device)
    K = sample["K"].to(device)
    T_cam_to_world = sample["T_cam_to_world"].to(device)
    T_imu_to_world = sample["T_imu_to_world"].to(device)
    sat = sample["sat"].to(device)

    # IPM warp (used for building semantic tokens for model compatibility)
    warped_front, warped_valid, warped_coords = compute_inverse_projection_view(
        sat, K, T_cam_to_world, T_imu_to_world, TARGET_H, TARGET_W, device,
    )

    # Semantic conditioning (built for model compatibility - vanilla mode doesn't use them)
    sem_dict, coord_dict = build_condition_tokens_with_coords(
        warped_front, warped_coords, warped_valid, GRID_ROWS, GRID_COLS, device,
    )

    # Pose vector
    pose = build_pose_vec(K, T_cam_to_world, T_imu_to_world, TARGET_H, TARGET_W, device)

    # BEV features (vanilla mode primarily uses this)
    bev_feats = bev_encoder(sat.unsqueeze(0))

    # BEV visibility mask
    bev_vis = compute_bev_visibility_mask(
        K=K, T_cam_to_world=T_cam_to_world, T_imu_to_world=T_imu_to_world,
        cam_h=TARGET_H, cam_w=TARGET_W,
    )

    return {
        "condition": {
            "semantic": sem_dict["fine"].unsqueeze(0),
            "coords": coord_dict["fine"].unsqueeze(0),
            "pose": pose.unsqueeze(0),
            "K": K.unsqueeze(0),
        },
        "bev": bev_feats,
        "bev_vis_mask": bev_vis.unsqueeze(0),
        "rgb": rgb,
        "sat": sat,
        "K_raw": K,
        "T_cam_to_world": T_cam_to_world,
        "T_imu_to_world": T_imu_to_world,
    }


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


def save_vanilla_result(gen_img, gt_img, sat_img, pose, out_path):
    """Save comparison result for vanilla AR."""
    W, H = gen_img.size

    total_W = W * 2 + 10
    total_H = H + 100  # Extra height for BEV visualization
    pad = 10

    grid = Image.new("RGB", (total_W, total_H), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # Draw titles
    draw.text((pad + W//2 - 60, pad), "Generated", fill=(255, 255, 255), font=font)
    draw.text((pad + W + 10 + W//2 - 40, pad), "GT", fill=(255, 255, 255), font=font)

    # Paste images
    grid.paste(gen_img, (pad, 50))
    grid.paste(gt_img, (pad + W + 10, 50))

    # Save
    grid.save(out_path)
    print("  Saved: %s" % out_path)


def save_bev_vis_result(sat_img, pose, bev_feats, out_path):
    """Save BEV visualization."""
    # Visualization code would go here
    sat_img.save(out_path)
    print("  Saved BEV: %s" % out_path)


# ── Main ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model and tokenizer
    model = load_model(args.ckpt, device)
    vq = PretrainedTokenizer(args.vq_ckpt).to(device)
    bev_encoder = model.bev_encoder
    bev_encoder.eval()

    # Create dataset for the specific drive and frame
    drive_path = Path(args.data_root) / args.drive
    ds = Kitti360dDataset(
        drives=str(drive_path),
        frames=[args.frame],
        mode="front",
        virtual_hfov_deg=80.0,
        virtual_size=(640, 256),
    )

    if len(ds) == 0:
        print("[Error] No data found for drive %s, frame %d" % (args.drive, args.frame))
        return

    sample = ds[0]
    if sample.get("meta", {}).get("dummy", False):
        print("[Error] Dummy sample returned for drive %s, frame %d" % (args.drive, args.frame))
        return

    # Prepare model inputs
    print("\n[Data] Processing sample...")
    data = prepare_sample(sample, vq, bev_encoder, device)

    # Generate
    print("\n[Inference] Generating...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    token_grid = generate_ar(
        model, data["condition"], data["bev"], data["bev_vis_mask"], device,
        top_k=args.top_k, temperature=args.temperature,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time
    print("[Inference] Generation time: %.3fs" % elapsed)

    # Decode and save
    print("\n[Post-processing] Decoding...")
    gen_img = vq.decode(token_grid)[0]
    gen_pil = tensor_to_pil(gen_img)
    gt_pil = tensor01_to_pil(sample["image"])
    sat_pil = tensor01_to_pil(sample["sat"])

    # Save comparison image
    out_path = os.path.join(args.out_dir, "vanilla_ar_drive_%s_frame_%010d.png" % (args.drive, args.frame))
    save_vanilla_result(gen_pil, gt_pil, sat_pil, data["condition"]["pose"], out_path)

    # Save BEV features for debugging
    sat_out_path = os.path.join(args.out_dir, "satellite_drive_%s_frame_%010d.png" % (args.drive, args.frame))
    save_bev_vis_result(sat_pil, data["condition"]["pose"], data["bev"], sat_out_path)

    # Save raw token grid for analysis
    token_out_path = os.path.join(args.out_dir, "tokens_drive_%s_frame_%010d.npy" % (args.drive, args.frame))
    np.save(token_out_path, token_grid[0].cpu().numpy())

    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)
    print("Output directory: %s" % args.out_dir)
    print("Files created:")
    print("  - %s: Generated vs GT comparison" % os.path.basename(out_path))
    print("  - %s: BEV input" % os.path.basename(sat_out_path))
    print("  - %s: Raw token grid" % os.path.basename(token_out_path))
    print("\nGeneration time: %.3f seconds" % elapsed)


if __name__ == "__main__":
    main()
