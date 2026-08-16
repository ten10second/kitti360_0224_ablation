#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for diffusion-based view synthesis model.

This script is specifically designed for the diffusion model trained with
diffusion_1d_cond.yaml, which uses satellite image features and camera pose
embeddings to generate novel views.

Usage:
    python scripts/infer_diffusion.py --ckpt runs/diffusion_1d_cond/checkpoints/ckpt_step_0080000.pt
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

from models.stage2.diffusion_model import DiffusionPoseModel
from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.data.diffusion_pipeline import collate_diffusion_samples
from world3d.train.pose_ar import build_pose_vec


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Diffusion Model Inference (for diffusion_1d_cond.yaml trained models)"
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to diffusion model checkpoint",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="/media/zhimiao/Lenovo/KITTI-360",
        help="KITTI-360 data root directory",
    )
    p.add_argument(
        "--drive",
        type=str,
        default="2013_05_28_drive_0003_sync",
        help="KITTI-360 drive to use for inference",
    )
    p.add_argument(
        "--frame",
        type=int,
        default=113,
        help="Frame ID to test",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "runs/diffusion_1d_cond/inference"),
        help="Output directory for generated images",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size",
    )

    return p.parse_args()


def load_model(ckpt_path: str, device: str) -> DiffusionPoseModel:
    """Load diffusion model from checkpoint."""
    model = DiffusionPoseModel(
        sd_model_id="runwayml/stable-diffusion-v1-5",
        bev_encoder_ckpt=str(REPO_ROOT / "ckpts/fmow_pretrain.pth"),
        freeze_sd=True,
        freeze_bev_encoder=True,
        device=device,
    )

    print(f"[Model] Loading from {ckpt_path}")
    if os.path.isdir(ckpt_path):
        model.load_pretrained(ckpt_path)
    else:
        # Load from checkpoint dict
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
        except Exception as e:
            print(f"[Model] Failed to load from {ckpt_path}, trying ControlNet directory")
            model.load_pretrained(os.path.dirname(ckpt_path))

    model.eval()
    print(f"[Model] Loaded successfully")

    return model


@torch.no_grad()
def prepare_sample(sample, device: str):
    """Process a Kitti360dDataset sample into model inputs."""
    rgb = sample["image"].to(device)
    sat = sample["sat"].to(device)
    K = sample["K"].to(device)
    T_cam_to_world = sample["T_cam_to_world"].to(device)
    T_imu_to_world = sample["T_imu_to_world"].to(device)

    # Convert RGB to [-1, 1] range
    rgb = rgb * 2.0 - 1.0

    # Build pose vector
    pose = build_pose_vec(
        K,
        T_cam_to_world,
        T_imu_to_world,
        img_h=256,
        img_w=640,
        device=device,
    )

    return {
        "rgb": rgb.unsqueeze(0),
        "sat": sat.unsqueeze(0),
        "pose": pose.unsqueeze(0),
        "K": K.unsqueeze(0),
        "T_cam_to_world": T_cam_to_world.unsqueeze(0),
        "T_imu_to_world": T_imu_to_world.unsqueeze(0),
    }


@torch.no_grad()
def generate_sample(
    model: DiffusionPoseModel,
    batch: dict,
    num_inference_steps: int,
    guidance_scale: float,
):
    """Generate an image from a single sample."""
    start_time = time.time()

    generated_images = model.generate(
        sat_images=batch["sat"],
        pose_vec=batch["pose"],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    elapsed = time.time() - start_time
    print(f"[Inference] Generation time: {elapsed:.3f}s")

    return generated_images


def tensor_to_pil(img_tensor):
    """Convert PyTorch tensor to PIL Image."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    arr = (img_tensor.cpu().float() * 0.5 + 0.5).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def tensor01_to_pil(img_tensor):
    """Convert [0, 1] tensor to PIL Image."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    arr = img_tensor.cpu().float().clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_result(
    gen_img: Image.Image,
    gt_img: Image.Image,
    sat_img: Image.Image,
    drive: str,
    frame: int,
    out_path: str,
):
    """Save comparison result for diffusion model."""
    W, H = gen_img.size

    total_W = W * 3 + 20  # Gen, GT, SAT
    total_H = H + 100  # Extra height for title and metadata
    pad = 10

    grid = Image.new("RGB", (total_W, total_H), (0, 0, 0))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # Draw titles
    draw.text((pad + W // 2 - 70, pad), "Generated", fill=(255, 255, 255), font=font)
    draw.text((pad + W + 10 + W // 2 - 30, pad), "GT", fill=(255, 255, 255), font=font)
    draw.text((pad + 2 * W + 20 + W // 2 - 40, pad), "Satellite", fill=(255, 255, 255), font=font)

    # Paste images
    grid.paste(gen_img, (pad, 50))
    grid.paste(gt_img, (pad + W + 10, 50))
    grid.paste(sat_img, (pad + 2 * W + 20, 50))

    # Save to file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    print(f"  Saved: {out_path}")

    return grid


def save_individual_images(
    gen_img: Image.Image,
    gt_img: Image.Image,
    sat_img: Image.Image,
    drive: str,
    frame: int,
    out_dir: str,
):
    """Save individual images (generated, GT, satellite) to separate files."""
    os.makedirs(out_dir, exist_ok=True)

    # Generated
    gen_out = os.path.join(out_dir, f"generated_{drive}_frame_{frame:010d}.png")
    gen_img.save(gen_out)

    # Ground Truth
    gt_out = os.path.join(out_dir, f"gt_{drive}_frame_{frame:010d}.png")
    gt_img.save(gt_out)

    # Satellite
    sat_out = os.path.join(out_dir, f"satellite_{drive}_frame_{frame:010d}.png")
    sat_img.save(sat_out)

    return gen_out, gt_out, sat_out


def main():
    """Main inference pipeline."""
    args = parse_args()
    device = torch.device(args.device)

    # Load model
    print("\n[Loading model]")
    model = load_model(args.ckpt, device)

    # Create dataset for the specific drive and frame
    print("\n[Loading data]")
    drive_path = Path(args.data_root) / args.drive
    ds = Kitti360dDataset(
        drives=str(drive_path),
        frames=[args.frame],
        mode="front",
        virtual_hfov_deg=80.0,
        virtual_size=(640, 256),
    )

    if len(ds) == 0:
        print(f"[Error] No data found for drive {args.drive}, frame {args.frame}")
        return

    sample = ds[0]
    if sample.get("meta", {}).get("dummy", False):
        print(f"[Error] Dummy sample returned for drive {args.drive}, frame {args.frame}")
        return

    # Prepare inputs
    print("\n[Processing sample]")
    data = prepare_sample(sample, device)

    # Generate
    print("\n[Generating]")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    generated_images = model.generate(
        sat_images=data["sat"],
        pose_vec=data["pose"],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time
    print(f"[Inference] Generation time: {elapsed:.3f}s")

    # Decode and save
    print("\n[Post-processing]")

    # Convert to PIL images
    gen_img = tensor_to_pil(generated_images[0])
    gt_img = tensor01_to_pil(sample["image"])
    sat_img = tensor01_to_pil(sample["sat"])

    # Save comparison image
    out_path = os.path.join(
        args.out_dir,
        f"diffusion_result_drive_{args.drive}_frame_{args.frame:010d}.png",
    )
    save_result(gen_img, gt_img, sat_img, args.drive, args.frame, out_path)

    # Save individual images
    save_individual_images(gen_img, gt_img, sat_img, args.drive, args.frame, args.out_dir)

    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)
    print("Output directory:", args.out_dir)


if __name__ == "__main__":
    main()
