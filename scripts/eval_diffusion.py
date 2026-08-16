#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate metrics (FID, SSIM, PSNR) for diffusion-based view synthesis model."""

import argparse
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Use huggingface mirror site for faster downloads in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.train.pose_ar import build_pose_vec
from metrics.psnr import PSNR
from metrics.ssim import SSIM
from metrics.lpips import LPIPS
from metrics.fid import FID

# Model imports
from models.stage2.diffusion_model import DiffusionPoseModel

# Constants
TARGET_H = 256
TARGET_W = 640
DIFFUSION_TARGET_SIZE = (512, 512)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate metrics for diffusion-based view synthesis model")
    p.add_argument("--ckpt", required=True, help="Path to diffusion model checkpoint")
    p.add_argument("--data-root", default="/media/zhimiao/Lenovo/KITTI-360")
    p.add_argument("--out-dir", default=str(Path(__file__).parent.parent / "runs/diffusion_1d_cond/metrics"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--rank", default=0, type=int, help="Rank of the process")
    p.add_argument("--world-size", default=1, type=int, help="Total number of processes")
    p.add_argument("--dist-url", default="tcp://127.0.0.1:29501", type=str, help="URL for distributed training")
    p.add_argument("--dist-backend", default="nccl", type=str, help="Backend for distributed training")
    p.add_argument("--multiprocessing-distributed", action="store_true", help="Use multi-processing distributed training")
    p.add_argument("--visualize", action="store_true", help="Visualize generated and GT images")
    return p.parse_args()


def setup_distributed(rank, world_size, args):
    """Setup distributed training/evaluation environment."""
    os.environ['MASTER_ADDR'] = args.dist_url.split('://')[1].split(':')[0]
    os.environ['MASTER_PORT'] = args.dist_url.split(':')[-1]
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training/evaluation environment."""
    dist.destroy_process_group()


def load_model(ckpt_path, device):
    """Load the diffusion model from checkpoint."""
    model = DiffusionPoseModel(
        sd_model_id="runwayml/stable-diffusion-v1-5",
        bev_encoder_ckpt="ckpts/fmow_pretrain.pth",
        freeze_sd=True,
        freeze_bev_encoder=True,
        device=device,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[Model] Loaded diffusion model from {ckpt_path} (step {ckpt.get('step', '?')})")
    return model


def load_diffusion_components(ckpt_path, device):
    """Load diffusion model components from pretrained directory."""
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
def generate_diffusion(model, sat, pose, device, num_inference_steps=50, guidance_scale=5.0):
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


def get_eval_views():
    """Return evaluation views (training and untrained angles)."""
    # Training views (fixed 5 views matching training config)
    training_views = [
        ("front", "front", None, None),  # (name, source, fisheye_camera, yaw_deg)
        ("left_to_front_30", "image_02", "image_02", 30.0),
        ("right_to_front_30", "image_03", "image_03", -30.0),
        ("left_axis", "image_02", "image_02", 0.0),
        ("right_axis", "image_03", "image_03", 0.0),
    ]

    # Untrained views (left rear 30° and right rear 30°)
    untrained_views = [
        ("left_rear_30", "image_02", "image_02", -30.0),
        ("right_rear_30", "image_03", "image_03", 30.0),
    ]

    return training_views, untrained_views


def load_test_frames(data_root):
    """Load test frames from train_test_split_config.yaml."""
    split_config = Path(data_root) / "train_test_split_config.yaml"
    if not split_config.exists():
        raise FileNotFoundError(f"Train/test split config not found: {split_config}")

    import yaml
    with open(split_config, "r") as f:
        config = yaml.safe_load(f)

    test_frames = []
    for drive_info in config.get("test", []):
        drive = drive_info["drive"]
        frames_file = Path(data_root) / drive / drive_info["frames_file"]
        if frames_file.exists():
            with open(frames_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Handle format like "     1→978"
                        if "→" in line:
                            idx, frame_id = line.split("→")
                            frame_id = int(frame_id.strip())
                        else:
                            frame_id = int(line.strip())
                        test_frames.append((drive, frame_id))

    return test_frames


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


def save_visualization(sat_img, gen_img, gt_img, out_dir, drive, frame_id, view_name):
    """Save visualization of satellite, generated, and GT images."""
    out_dir = Path(out_dir) / "visualizations"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Convert to PIL images
    sat_pil = tensor01_to_pil(sat_img)
    gen_pil = tensor_to_pil(gen_img)  # gen_img is in [-1, 1] range
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
    filename = f"{drive}_frame_{frame_id:010d}_{view_name}_vis.png"
    grid.save(out_dir / filename)
    print(f"Visualization saved to {out_dir / filename}")


def evaluate_metrics(model, test_frames, data_root, device, args, views_to_eval):
    """Evaluate metrics on test frames for specified views."""
    # Initialize metric calculators
    psnr_calc = PSNR(reduction="none")
    ssim_calc = SSIM(reduction="none")
    lpips_calc = LPIPS(net="alex", reduction="none").to(device)
    fid_calc = FID(device=device)

    # Metrics storage
    psnr_values = []
    ssim_values = []
    lpips_values = []
    inference_times = []
    all_gt_imgs = []
    all_gen_imgs = []

    # Eval views
    eval_views = views_to_eval

    # Process each frame
    for drive, frame_id in tqdm(test_frames, desc="Processing frames"):
        drive_path = Path(data_root) / drive

        # Process each view
        for view_name, source, fisheye_cam, yaw_deg in eval_views:
            try:
                # Create dataset for this view
                if source == "front":
                    ds = Kitti360dDataset(
                        drives=str(drive_path),
                        frames=[frame_id],
                        mode="front",
                        virtual_hfov_deg=80.0,
                        virtual_size=(TARGET_W, TARGET_H),
                    )
                else:
                    ds = Kitti360dDataset(
                        drives=str(drive_path),
                        frames=[frame_id],
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
                    continue

                # Prepare sample
                start_time = time.time()
                data = prepare_sample(sample, device)
                gt_img = data["rgb"]

                # Generate
                gen_img = generate_diffusion(
                    model, data["sat"], data["pose"], device,
                    args.num_inference_steps, args.guidance_scale
                )

                # Post-process images to [0, 1] range
                gen_img = (gen_img + 1.0) / 2.0  # [-1, 1] to [0, 1]
                gt_img = gt_img.clamp(0, 1)

                # Compute metrics
                psnr = psnr_calc(gen_img.unsqueeze(0), gt_img.unsqueeze(0))
                ssim = ssim_calc(gen_img.unsqueeze(0), gt_img.unsqueeze(0))
                lpips = lpips_calc(gen_img.unsqueeze(0), gt_img.unsqueeze(0))

                # Store results
                psnr_values.append(psnr.item())
                ssim_values.append(ssim.item())
                lpips_values.append(lpips.item())
                inference_times.append(time.time() - start_time)
                # Collect all images for FID calculation (even in distributed mode)
                all_gt_imgs.append(gt_img.unsqueeze(0))
                all_gen_imgs.append(gen_img.unsqueeze(0))

                # Visualize if required
                if args.visualize and args.rank == 0:
                    save_visualization(data["sat"], gen_img, gt_img, args.out_dir, drive, frame_id, view_name)

            except Exception as e:
                print(f"Error processing {drive} frame {frame_id} view {view_name}: {e}")
                continue

    # Calculate statistics
    metrics = {
        "psnr": {
            "mean": np.mean(psnr_values),
            "std": np.std(psnr_values),
            "min": np.min(psnr_values),
            "max": np.max(psnr_values),
        },
        "ssim": {
            "mean": np.mean(ssim_values),
            "std": np.std(ssim_values),
            "min": np.min(ssim_values),
            "max": np.max(ssim_values),
        },
        "lpips": {
            "mean": np.mean(lpips_values),
            "std": np.std(lpips_values),
            "min": np.min(lpips_values),
            "max": np.max(lpips_values),
        },
        "inference": {
            "mean": np.mean(inference_times),
            "std": np.std(inference_times),
            "min": np.min(inference_times),
            "max": np.max(inference_times),
        },
        "count": len(psnr_values),
    }

    # Calculate FID
    if len(all_gt_imgs) > 0 and len(all_gen_imgs) > 0:
        gt_stack = torch.cat(all_gt_imgs, dim=0)
        gen_stack = torch.cat(all_gen_imgs, dim=0)
        try:
            fid_score = fid_calc(gt_stack, gen_stack)
            metrics["fid"] = fid_score
        except Exception as e:
            print(f"Error calculating FID: {e}")
            metrics["fid"] = None

    return metrics, psnr_values, ssim_values, lpips_values, inference_times, all_gt_imgs, all_gen_imgs


def save_results(metrics, psnr_values, ssim_values, lpips_values, out_dir):
    """Save results to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Save detailed values as CSV for easy analysis
    import csv
    with open(out_dir / "metrics.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index", "PSNR", "SSIM", "LPIPS"])
        for i, (psnr, ssim, lpips) in enumerate(zip(psnr_values, ssim_values, lpips_values)):
            writer.writerow([i, f"{psnr:.4f}", f"{ssim:.4f}", f"{lpips:.4f}"])

    # Save detailed values as text files for compatibility
    np.savetxt(out_dir / "psnr.txt", np.array(psnr_values), fmt="%.4f")
    np.savetxt(out_dir / "ssim.txt", np.array(ssim_values), fmt="%.4f")
    np.savetxt(out_dir / "lpips.txt", np.array(lpips_values), fmt="%.4f")

    # Save summary
    with open(out_dir / "summary.txt", "w") as f:
        f.write("Metrics Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Frames processed: {metrics['count']}\n")
        f.write("\n")
        f.write("PSNR:\n")
        f.write(f"  Mean: {metrics['psnr']['mean']:.4f} ± {metrics['psnr']['std']:.4f}\n")
        f.write(f"  Min: {metrics['psnr']['min']:.4f}, Max: {metrics['psnr']['max']:.4f}\n")
        f.write("\n")
        f.write("SSIM:\n")
        f.write(f"  Mean: {metrics['ssim']['mean']:.4f} ± {metrics['ssim']['std']:.4f}\n")
        f.write(f"  Min: {metrics['ssim']['min']:.4f}, Max: {metrics['ssim']['max']:.4f}\n")
        f.write("\n")
        f.write("LPIPS:\n")
        f.write(f"  Mean: {metrics['lpips']['mean']:.4f} ± {metrics['lpips']['std']:.4f}\n")
        f.write(f"  Min: {metrics['lpips']['min']:.4f}, Max: {metrics['lpips']['max']:.4f}\n")
        f.write("\n")
        if "fid" in metrics and metrics["fid"] is not None:
            f.write("FID:\n")
            f.write(f"  Score: {metrics['fid']:.4f}\n")
        f.write("\n")
        f.write("Inference Time:\n")
        f.write(f"  Mean: {metrics['inference']['mean']:.3f} ± {metrics['inference']['std']:.3f} s\n")
        f.write(f"  Min: {metrics['inference']['min']:.3f} s, Max: {metrics['inference']['max']:.3f} s\n")

    # Save results as JSON for easy parsing
    import json
    metrics_json = {
        "count": metrics["count"],
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "lpips": metrics["lpips"],
        "inference": metrics["inference"],
    }
    if "fid" in metrics:
        metrics_json["fid"] = metrics["fid"]

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"Results saved to {out_dir}")


def main_worker(rank, world_size, args):
    """Main worker function for each process."""
    print(f"Rank {rank}: Starting...")

    # Setup distributed
    if world_size > 1:
        setup_distributed(rank, world_size, args)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        # Use LOCAL_RANK if available (set by torchrun)
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    print(f"Rank {rank}: Using device {device}")

    # Load model
    model = load_diffusion_components(args.ckpt, device)
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        model = DDP(model, device_ids=[local_rank])

    # Load test frames
    test_frames = load_test_frames(args.data_root)

    # Distribute test frames across ranks
    if world_size > 1:
        per_rank_frames = len(test_frames) // world_size
        start_idx = rank * per_rank_frames
        if rank == world_size - 1:
            end_idx = len(test_frames)
        else:
            end_idx = start_idx + per_rank_frames
        test_frames = test_frames[start_idx:end_idx]

    print(f"Rank {rank}: Loaded {len(test_frames)} test frames")

    # Get training and untrained views
    training_views, untrained_views = get_eval_views()

    # Evaluate metrics for training views
    print(f"Rank {rank}: Evaluating training views...")
    training_metrics, training_psnr, training_ssim, training_lpips, training_times, training_gt_list, training_gen_list = evaluate_metrics(
        model, test_frames, args.data_root, device, args, training_views
    )

    # Evaluate metrics for untrained views
    print(f"Rank {rank}: Evaluating untrained views...")
    untrained_metrics, untrained_psnr, untrained_ssim, untrained_lpips, untrained_times, untrained_gt_list, untrained_gen_list = evaluate_metrics(
        model, test_frames, args.data_root, device, args, untrained_views
    )

    # Collect metrics from all ranks if distributed
    if world_size > 1:
        # Collect training metrics
        all_training_psnr = [None for _ in range(world_size)]
        dist.all_gather_object(all_training_psnr, training_psnr)
        all_training_ssim = [None for _ in range(world_size)]
        dist.all_gather_object(all_training_ssim, training_ssim)
        all_training_lpips = [None for _ in range(world_size)]
        dist.all_gather_object(all_training_lpips, training_lpips)
        all_training_times = [None for _ in range(world_size)]
        dist.all_gather_object(all_training_times, training_times)

        # Collect untrained metrics
        all_untrained_psnr = [None for _ in range(world_size)]
        dist.all_gather_object(all_untrained_psnr, untrained_psnr)
        all_untrained_ssim = [None for _ in range(world_size)]
        dist.all_gather_object(all_untrained_ssim, untrained_ssim)
        all_untrained_lpips = [None for _ in range(world_size)]
        dist.all_gather_object(all_untrained_lpips, untrained_lpips)
        all_untrained_times = [None for _ in range(world_size)]
        dist.all_gather_object(all_untrained_times, untrained_times)

        # Flatten training metrics
        training_psnr = [item for sublist in all_training_psnr for item in sublist]
        training_ssim = [item for sublist in all_training_ssim for item in sublist]
        training_lpips = [item for sublist in all_training_lpips for item in sublist]
        training_times = [item for sublist in all_training_times for item in sublist]

        # Flatten untrained metrics
        untrained_psnr = [item for sublist in all_untrained_psnr for item in sublist]
        untrained_ssim = [item for sublist in all_untrained_ssim for item in sublist]
        untrained_lpips = [item for sublist in all_untrained_lpips for item in sublist]
        untrained_times = [item for sublist in all_untrained_times for item in sublist]

        # Recalculate training metrics
        training_metrics = {
            "psnr": {
                "mean": np.mean(training_psnr),
                "std": np.std(training_psnr),
                "min": np.min(training_psnr),
                "max": np.max(training_psnr),
            },
            "ssim": {
                "mean": np.mean(training_ssim),
                "std": np.std(training_ssim),
                "min": np.min(training_ssim),
                "max": np.max(training_ssim),
            },
            "lpips": {
                "mean": np.mean(training_lpips),
                "std": np.std(training_lpips),
                "min": np.min(training_lpips),
                "max": np.max(training_lpips),
            },
            "inference": {
                "mean": np.mean(training_times),
                "std": np.std(training_times),
                "min": np.min(training_times),
                "max": np.max(training_times),
            },
            "count": len(training_psnr),
        }

        # Recalculate untrained metrics
        untrained_metrics = {
            "psnr": {
                "mean": np.mean(untrained_psnr),
                "std": np.std(untrained_psnr),
                "min": np.min(untrained_psnr),
                "max": np.max(untrained_psnr),
            },
            "ssim": {
                "mean": np.mean(untrained_ssim),
                "std": np.std(untrained_ssim),
                "min": np.min(untrained_ssim),
                "max": np.max(untrained_ssim),
            },
            "lpips": {
                "mean": np.mean(untrained_lpips),
                "std": np.std(untrained_lpips),
                "min": np.min(untrained_lpips),
                "max": np.max(untrained_lpips),
            },
            "inference": {
                "mean": np.mean(untrained_times),
                "std": np.std(untrained_times),
                "min": np.min(untrained_times),
                "max": np.max(untrained_times),
            },
            "count": len(untrained_psnr),
        }

        # Calculate FID on rank 0
        if rank == 0:
            print("Calculating FID (this may take a while)...")

            # Collect all GT and generated images from all ranks
            all_gt_imgs = []
            all_gen_imgs = []

            # For training views
            training_gt_imgs = [None for _ in range(world_size)]
            training_gen_imgs = [None for _ in range(world_size)]
            dist.all_gather_object(training_gt_imgs, training_gt_list)
            dist.all_gather_object(training_gen_imgs, training_gen_list)

            # Flatten and collect
            for rank_gt in training_gt_imgs:
                all_gt_imgs.extend(rank_gt)
            for rank_gen in training_gen_imgs:
                all_gen_imgs.extend(rank_gen)

            # Calculate FID for training views
            if len(all_gt_imgs) > 0 and len(all_gen_imgs) > 0:
                gt_stack = torch.cat(all_gt_imgs, dim=0)
                gen_stack = torch.cat(all_gen_imgs, dim=0)
                try:
                    fid_score = FID(device=device)(gt_stack, gen_stack)
                    training_metrics["fid"] = fid_score
                except Exception as e:
                    print(f"Error calculating FID for training views: {e}")
                    training_metrics["fid"] = None
            else:
                training_metrics["fid"] = None

            # For untrained views
            untrained_gt_imgs = [None for _ in range(world_size)]
            untrained_gen_imgs = [None for _ in range(world_size)]
            dist.all_gather_object(untrained_gt_imgs, untrained_gt_list)
            dist.all_gather_object(untrained_gen_imgs, untrained_gen_list)

            # Flatten and collect
            all_gt_imgs = []
            all_gen_imgs = []
            for rank_gt in untrained_gt_imgs:
                all_gt_imgs.extend(rank_gt)
            for rank_gen in untrained_gen_imgs:
                all_gen_imgs.extend(rank_gen)

            # Calculate FID for untrained views
            if len(all_gt_imgs) > 0 and len(all_gen_imgs) > 0:
                gt_stack = torch.cat(all_gt_imgs, dim=0)
                gen_stack = torch.cat(all_gen_imgs, dim=0)
                try:
                    fid_score = FID(device=device)(gt_stack, gen_stack)
                    untrained_metrics["fid"] = fid_score
                except Exception as e:
                    print(f"Error calculating FID for untrained views: {e}")
                    untrained_metrics["fid"] = None
            else:
                untrained_metrics["fid"] = None

    # Save results only on rank 0
    if rank == 0:
        # Save training results
        training_out_dir = Path(args.out_dir) / "training"
        training_out_dir.mkdir(exist_ok=True, parents=True)
        save_results(training_metrics, training_psnr, training_ssim, training_lpips, training_out_dir)

        # Save untrained results
        untrained_out_dir = Path(args.out_dir) / "untrained"
        untrained_out_dir.mkdir(exist_ok=True, parents=True)
        save_results(untrained_metrics, untrained_psnr, untrained_ssim, untrained_lpips, untrained_out_dir)

        # Print summary
        print("\nTraining Views Metrics Evaluation Summary")
        print("=" * 50)
        print(f"Frames processed: {training_metrics['count']}")
        print("\nPSNR:")
        print(f"  Mean: {training_metrics['psnr']['mean']:.4f} ± {training_metrics['psnr']['std']:.4f}")
        print(f"  Min: {training_metrics['psnr']['min']:.4f}, Max: {training_metrics['psnr']['max']:.4f}")
        print("\nSSIM:")
        print(f"  Mean: {training_metrics['ssim']['mean']:.4f} ± {training_metrics['ssim']['std']:.4f}")
        print(f"  Min: {training_metrics['ssim']['min']:.4f}, Max: {training_metrics['ssim']['max']:.4f}")
        print("\nLPIPS:")
        print(f"  Mean: {training_metrics['lpips']['mean']:.4f} ± {training_metrics['lpips']['std']:.4f}")
        print(f"  Min: {training_metrics['lpips']['min']:.4f}, Max: {training_metrics['lpips']['max']:.4f}")
        if "fid" in training_metrics and training_metrics["fid"] is not None:
            print("\nFID:")
            print(f"  Score: {training_metrics['fid']:.4f}")
        print("\nInference Time:")
        print(f"  Mean: {training_metrics['inference']['mean']:.3f} ± {training_metrics['inference']['std']:.3f} s")
        print(f"  Min: {training_metrics['inference']['min']:.3f} s, Max: {training_metrics['inference']['max']:.3f} s")

        print("\n\nUntrained Views Metrics Evaluation Summary")
        print("=" * 50)
        print(f"Frames processed: {untrained_metrics['count']}")
        print("\nPSNR:")
        print(f"  Mean: {untrained_metrics['psnr']['mean']:.4f} ± {untrained_metrics['psnr']['std']:.4f}")
        print(f"  Min: {untrained_metrics['psnr']['min']:.4f}, Max: {untrained_metrics['psnr']['max']:.4f}")
        print("\nSSIM:")
        print(f"  Mean: {untrained_metrics['ssim']['mean']:.4f} ± {untrained_metrics['ssim']['std']:.4f}")
        print(f"  Min: {untrained_metrics['ssim']['min']:.4f}, Max: {untrained_metrics['ssim']['max']:.4f}")
        print("\nLPIPS:")
        print(f"  Mean: {untrained_metrics['lpips']['mean']:.4f} ± {untrained_metrics['lpips']['std']:.4f}")
        print(f"  Min: {untrained_metrics['lpips']['min']:.4f}, Max: {untrained_metrics['lpips']['max']:.4f}")
        if "fid" in untrained_metrics and untrained_metrics["fid"] is not None:
            print("\nFID:")
            print(f"  Score: {untrained_metrics['fid']:.4f}")
        print("\nInference Time:")
        print(f"  Mean: {untrained_metrics['inference']['mean']:.3f} ± {untrained_metrics['inference']['std']:.3f} s")
        print(f"  Min: {untrained_metrics['inference']['min']:.3f} s, Max: {untrained_metrics['inference']['max']:.3f} s")

    # Cleanup
    if world_size > 1:
        cleanup_distributed()

    print(f"Rank {rank}: Done")


def main():
    # Parse args
    args = parse_args()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = "cpu"

    # Determine world size and rank
    if args.multiprocessing_distributed:
        # If using torchrun, rank and world_size are already set by environment variables
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            # If not using torchrun, use mp.spawn
            if torch.cuda.is_available():
                args.world_size = torch.cuda.device_count()
            else:
                args.world_size = 1
            mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
            return

    # Run main worker
    main_worker(args.rank, args.world_size, args)


if __name__ == "__main__":
    main()
