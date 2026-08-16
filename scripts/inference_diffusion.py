#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple inference script for diffusion-based view synthesis model."""

import argparse
from pathlib import Path
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Use huggingface mirror site for faster downloads in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.train.pose_ar import build_pose_vec
from models.stage2.diffusion_model import DiffusionPoseModel

# Constants
TARGET_H = 256
TARGET_W = 640


def parse_args():
    p = argparse.ArgumentParser(description="Simple inference for diffusion-based view synthesis")
    p.add_argument("--ckpt", required=True, help="Path to diffusion model checkpoint")
    p.add_argument("--data-root", default="/media/zhimiao/Lenovo/KITTI-360")
    p.add_argument("--drive", default="2013_05_28_drive_0000_sync")
    p.add_argument("--frame-id", type=int, default=0)
    p.add_argument("--view", default="front", choices=["front", "left_to_front_30", "right_to_front_30", "left_axis", "right_axis"])
    p.add_argument("--out-dir", default=str(Path(__file__).parent.parent / "runs/diffusion_1d_cond/inference"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    return p.parse_args()


def load_model(ckpt_path, device):
    """Load the diffusion model from checkpoint."""
    model = DiffusionPoseModel(
        sd_model_id="runwayml/stable-diffusion-v1-5",
        bev_encoder_ckpt="ckpts/fmow_pretrain.pth",
        freeze_sd=True,
        freeze_bev_encoder=True,
        device=device,
    )

    # Load trainable components
    load_dir = Path(ckpt_path)
    if load_dir.is_dir():
        model.load_pretrained(load_dir)
    else:
        # If it's a checkpoint file, try to load state dict
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    model.eval()
    print(f"[Model] Loaded diffusion model from {ckpt_path}")
    return model


@torch.no_grad()
def generate(model, sat, pose, device, num_inference_steps=50, guidance_scale=5.0):
    """Generate using diffusion model with given conditions."""
    sat_images = sat.unsqueeze(0).to(device)
    pose_vec = pose.unsqueeze(0).to(device)

    generated = model.generate(
        sat_images=sat_images,
        pose_vec=pose_vec,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    # Resize to target size
    generated = F.interpolate(
        generated,
        size=(TARGET_H, TARGET_W),
        mode="bilinear",
        align_corners=False,
    )

    return generated[0]


@torch.no_grad()
def prepare_sample(sample, device):
    """Process a Kitti360dDataset sample into model inputs."""
    rgb = sample["image"].to(device)
    sat = sample["sat"].to(device)
    K = sample["K"].to(device)
    T_cam_to_world = sample["T_cam_to_world"].to(device)
    T_imu_to_world = sample["T_imu_to_world"].to(device)

    # Pose vector
    pose = build_pose_vec(K, T_cam_to_world, T_imu_to_world, TARGET_H, TARGET_W, device)

    return {
        "sat": sat,
        "pose": pose,
        "rgb": rgb,
    }


def tensor_to_pil(img_tensor):
    """Convert tensor in [-1, 1] range to PIL Image."""
    arr = img_tensor.cpu().float().clamp(-1, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
    return Image.fromarray(arr)


def tensor01_to_pil(img_tensor):
    """Convert tensor in [0, 1] range to PIL Image."""
    arr = img_tensor.cpu().float().clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_result(sat_img, gen_img, gt_img, out_dir, drive, frame_id, view_name):
    """Save generated result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Convert to PIL images
    sat_pil = tensor01_to_pil(sat_img)
    gen_pil = tensor_to_pil(gen_img)
    gt_pil = tensor01_to_pil(gt_img)

    # Create a grid
    width, height = gt_pil.size
    total_width = width * 3
    total_height = height
    grid = Image.new("RGB", (total_width, total_height))

    grid.paste(sat_pil, (0, 0))
    grid.paste(gen_pil, (width, 0))
    grid.paste(gt_pil, (width * 2, 0))

    # Add labels
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 10), "Satellite", (255, 0, 0), font=font)
    draw.text((width + 10, 10), "Generated", (0, 255, 0), font=font)
    draw.text((width * 2 + 10, 10), "GT", (0, 0, 255), font=font)

    # Save the grid
    filename = f"{drive}_frame_{frame_id:010d}_{view_name}_result.png"
    grid.save(out_dir / filename)
    print(f"Result saved to {out_dir / filename}")

    # Save individual images
    sat_pil.save(out_dir / f"{drive}_frame_{frame_id:010d}_{view_name}_sat.png")
    gen_pil.save(out_dir / f"{drive}_frame_{frame_id:010d}_{view_name}_gen.png")
    gt_pil.save(out_dir / f"{drive}_frame_{frame_id:010d}_{view_name}_gt.png")


def get_view_spec(view_name):
    """Get view specification based on view name."""
    view_map = {
        "front": ("front", None, None),
        "left_to_front_30": ("fisheye_virtual", "image_02", 30.0),
        "right_to_front_30": ("fisheye_virtual", "image_03", -30.0),
        "left_axis": ("fisheye_virtual", "image_02", 0.0),
        "right_axis": ("fisheye_virtual", "image_03", 0.0),
    }

    return view_map.get(view_name, ("front", None, None))


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

    # Load model
    model = load_model(args.ckpt, device)

    # Load data
    drive_path = Path(args.data_root) / args.drive
    mode, fisheye_cam, yaw_deg = get_view_spec(args.view)

    if mode == "front":
        ds = Kitti360dDataset(
            drives=str(drive_path),
            frames=[args.frame_id],
            mode="front",
            virtual_hfov_deg=80.0,
            virtual_size=(TARGET_W, TARGET_H),
        )
    else:
        ds = Kitti360dDataset(
            drives=str(drive_path),
            frames=[args.frame_id],
            mode="fisheye_virtual",
            fisheye_camera=fisheye_cam,
            fisheye_relative_yaw_deg=float(yaw_deg) if yaw_deg is not None else 0.0,
            virtual_hfov_deg=80.0,
            virtual_size=(TARGET_W, TARGET_H),
            random_fisheye_relative_yaw=False,
            calib_yaw_fix_deg=4.0,
        )

    sample = ds[0]
    if sample.get("meta", {}).get("dummy", False):
        print("Error: Got dummy sample")
        return

    # Prepare sample
    data = prepare_sample(sample, device)

    # Generate
    print(f"Generating view '{args.view}'...")
    gen_img = model.generate(
        sat_images=data["sat"].unsqueeze(0).to(device),
        pose_vec=data["pose"].unsqueeze(0).to(device),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    # Resize to target size
    gen_img = F.interpolate(
        gen_img,
        size=(TARGET_H, TARGET_W),
        mode="bilinear",
        align_corners=False,
    )[0]

    # Save result
    save_result(data["sat"], gen_img, data["rgb"], args.out_dir, args.drive, args.frame_id, args.view)

    # Show result (optional)
    try:
        from PIL import ImageShow
        img = Image.open(Path(args.out_dir) / f"{args.drive}_frame_{args.frame_id:010d}_{args.view}_result.png")
        ImageShow.show(img)
    except Exception as e:
        print(f"Could not show image: {e}")

    print("Inference completed!")


if __name__ == "__main__":
    main()
