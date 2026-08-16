#!/usr/bin/env python3
"""Inference on test frames with multiple yaw angles — uses training data pipeline."""

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
from models.stage2.simplified_token_predictor import SimplifiedTokenPredictor, MaskGITTokenPredictor
from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.train.geometry_ar import compute_inverse_projection_view
from world3d.train.conditioning_ar import build_condition_tokens_with_coords
from world3d.train.pose_ar import build_pose_vec
from world3d.data.ar_pipeline import compute_bev_visibility_mask
from world3d.train.vis_utils import render_bev_attn_heatmap

# Model config
D_MODEL = 512
VOCAB_SIZE = 1025
NUM_LAYERS = 8
NHEAD = 8
GRID_ROWS = 16
GRID_COLS = 40
SEQ_LEN = GRID_ROWS * GRID_COLS
BOS_TOKEN = 1024
VIRTUAL_HFOV = 80.0
VIRTUAL_W = 640
VIRTUAL_H = 256


def parse_args():
    p = argparse.ArgumentParser(description="Inference on test frames with multiple yaw angles")
    p.add_argument("--test-frames-dir", type=str, required=True,
                   help="Directory containing test frame .pt files")
    p.add_argument("--data-root", type=str, required=True, help="KITTI-360 data root")
    p.add_argument("--drive", type=str, default="2013_05_28_drive_0003_sync")
    p.add_argument("--ckpt", type=str, required=True, help="Model checkpoint")
    p.add_argument("--vq-ckpt", type=str, default="ckpts/maskgit-vqgan-imagenet-f16-256.bin")
    p.add_argument("--out-dir", type=str, default="test_results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--yaw-angles", type=float, nargs="+", default=[-40, -20, 0, 20, 40])
    p.add_argument("--fisheye-camera", default="image_02", choices=["image_02", "image_03"])
    p.add_argument("--frame-ids", type=int, nargs="+", default=None,
                   help="Frame IDs to test (if not provided, will try to extract from test frames)")
    p.add_argument("--maskgit", action="store_true", help="Use MaskGIT model")
    p.add_argument("--maskgit-steps", type=int, default=24)
    return p.parse_args()


def load_ar_model(ckpt_path, device):
    model = SimplifiedTokenPredictor(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS,
        nhead=NHEAD, dropout=0.0, max_seq_len=SEQ_LEN,
        target_rows=GRID_ROWS, target_cols=GRID_COLS, semantic_dim=4,
        fourier_freqs=10, train_bev_encoder=False, no_bev_pretrain=True,
        pose_dim=13, use_pose_token=True, n_pose_queries=64,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[Model] Loaded AR from {ckpt_path} (step {ckpt.get('step', '?')})")
    return model


def load_maskgit_model(ckpt_path, device):
    model = MaskGITTokenPredictor(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS,
        nhead=NHEAD, dropout=0.0, max_seq_len=SEQ_LEN,
        target_rows=GRID_ROWS, target_cols=GRID_COLS, semantic_dim=4,
        fourier_freqs=10, train_bev_encoder=False, no_bev_pretrain=True,
        pose_dim=13, use_pose_token=True, n_pose_queries=64,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[Model] Loaded MaskGIT from {ckpt_path} (step {ckpt.get('step', '?')})")
    return model


@torch.no_grad()
def top_k_sample(logits, k=50, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    k = min(k, logits.size(-1))
    top_vals, top_idx = torch.topk(logits, k, dim=-1)
    probs = F.softmax(top_vals, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return torch.gather(top_idx, -1, sampled_idx).squeeze(-1)


@torch.no_grad()
def generate_ar(model, condition, bev, bev_vis_mask, device, top_k=50, temperature=1.0):
    bos = torch.full((1, 1), BOS_TOKEN, dtype=torch.long, device=device)
    generated = bos
    for _ in range(SEQ_LEN):
        logits, _ = model(
            generated_tokens=generated,
            condition_tokens=condition,
            aligned_bev_feature_map=bev,
            bev_vis_mask=bev_vis_mask,
            use_cache=True,
        )
        next_tok = top_k_sample(logits[:, -1, :1024], k=top_k, temperature=temperature)
        generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
    token_seq = generated[:, 1:]
    return model.seq_to_grid(token_seq)


@torch.no_grad()
def generate_maskgit(model, condition, bev, bev_vis_mask, num_steps=12, top_k=50, temperature=1.0):
    return model.generate(
        condition_tokens=condition,
        aligned_bev_feature_map=bev,
        bev_vis_mask=bev_vis_mask,
        num_steps=num_steps,
        temperature=temperature,
        top_k=top_k,
    )


@torch.no_grad()
def prepare_sample(sample, vq, bev_encoder, device):
    """Process a Kitti360dDataset sample into model inputs."""
    rgb = sample["image"].to(device)
    rgb_norm = rgb * 2.0 - 1.0
    K = sample["K"].to(device)
    T_cam_to_world = sample["T_cam_to_world"].to(device)
    T_imu_to_world = sample["T_imu_to_world"].to(device)
    sat_img = sample.get("sat")
    if sat_img is not None:
        sat_img = sat_img.to(device)

    # Tokenize
    with torch.no_grad():
        tokens = vq.encode(rgb_norm.unsqueeze(0))[0]
    target_tokens = tokens.view(GRID_ROWS, GRID_COLS)

    # IPM (warp satellite image to camera perspective)
    ipm_img, ipm_valid, ipm_coords = compute_inverse_projection_view(
        sat_img, K, T_cam_to_world, T_imu_to_world, target_h=GRID_ROWS * 16, target_w=GRID_COLS * 16, device=device,
    )

    # Condition tokens
    pose_vec = build_pose_vec(K, T_cam_to_world, T_imu_to_world, VIRTUAL_H, VIRTUAL_W, device)
    semantic_dict, coord_dict = build_condition_tokens_with_coords(
        ipm_img, ipm_coords, ipm_valid,
        target_rows=GRID_ROWS, target_cols=GRID_COLS, device=device,
    )

    # BEV features
    if sat_img is not None:
        with torch.no_grad():
            bev_feats = bev_encoder(sat_img.unsqueeze(0))[0]
        bev_vis_mask = compute_bev_visibility_mask(
            K, T_cam_to_world, T_imu_to_world,
            bev_size=bev_feats.shape[-1],
            sat_pixels=sat_img.shape[-1],
            sat_resolution=sample.get("sat_m_per_px", 0.2),
            cam_h=VIRTUAL_H,
            cam_w=VIRTUAL_W,
        )
    else:
        bev_feats = None
        bev_vis_mask = None

    # Process IPM image for visualization (remove batch dimension if present)
    if ipm_img is not None and ipm_img.dim() == 4:
        ipm_img = ipm_img[0]

    return {
        "condition": {
            "semantic": semantic_dict["fine"].unsqueeze(0),
            "pose": pose_vec.unsqueeze(0),
            "K": K.unsqueeze(0),
        },
        "bev": bev_feats.unsqueeze(0) if bev_feats is not None else None,
        "bev_vis_mask": bev_vis_mask.unsqueeze(0) if bev_vis_mask is not None else None,
        "target_tokens": target_tokens,
        "ipm_img": ipm_img,
        "sat": sat_img,
    }


def tensor_to_pil(tensor):
    """Convert tensor in [-1, 1] to PIL Image."""
    img = (tensor.clamp(-1, 1) + 1) / 2
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def tensor01_to_pil(tensor):
    """Convert tensor in [0, 1] to PIL Image."""
    img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def save_yaw_grid(rows, frame_id, out_dir):
    """Save a grid with multiple yaw angles: BEV Heatmap | IPM | Generated | GT."""
    W, H = 640, 256
    hm_W = H
    total_W = hm_W + 3 * W
    nrows = len(rows)
    pad = 30

    grid = Image.new("RGB", (total_W, nrows * H + pad), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # Headers
    headers = [("BEV Attn", hm_W // 2 - 30),
               ("IPM", hm_W + W // 2 - 20),
               ("Generated", hm_W + W + W // 2 - 40),
               ("GT", hm_W + 2 * W + W // 2 - 10)]
    for hdr, x in headers:
        draw.text((x, 5), hdr, fill=(255, 255, 255), font=font)

    # Paste rows
    for i, (label, heatmap, ipm, gen, gt) in enumerate(rows):
        y = pad + i * H
        # Yaw label
        draw.text((5, y + H // 2 - 10), label, fill=(255, 255, 255), font=font)
        # Images
        if heatmap is not None:
            hm_resized = heatmap.resize((hm_W, H), Image.LANCZOS)
            grid.paste(hm_resized, (0, y))
        if ipm is not None:
            ipm_resized = ipm.resize((W, H), Image.LANCZOS)
            grid.paste(ipm_resized, (hm_W, y))
        if gen is not None:
            gen_resized = gen.resize((W, H), Image.LANCZOS)
            grid.paste(gen_resized, (hm_W + W, y))
        if gt is not None:
            gt_resized = gt.resize((W, H), Image.LANCZOS)
            grid.paste(gt_resized, (hm_W + 2 * W, y))

    out_path = Path(out_dir) / f"frame_{frame_id:04d}.png"
    grid.save(str(out_path))
    print(f"  Saved: {out_path}")


def extract_frame_ids_from_test_frames(test_frames_dir):
    """Extract frame IDs from saved test frame files."""
    test_frames_dir = Path(test_frames_dir)
    frame_files = sorted(test_frames_dir.glob("frame_*.pt"))

    if not frame_files:
        return []

    frame_ids = []
    for frame_file in frame_files:
        try:
            frame_data = torch.load(frame_file, map_location="cpu", weights_only=False)

            # Try to get frame_id directly
            if "frame_id" in frame_data:
                frame_id = int(frame_data["frame_id"])
                frame_ids.append(frame_id)
                print(f"  Loaded {frame_file.name} -> frame_id {frame_id}")
            else:
                print(f"  Warning: No frame_id in {frame_file.name}")
        except Exception as e:
            print(f"  Warning: Failed to load {frame_file.name}: {e}")
            continue

    return sorted(list(set(frame_ids)))


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    drive_path = Path(args.data_root) / args.drive

    # Determine frame IDs to test
    if args.frame_ids is not None:
        # Use command line provided frame IDs
        frame_ids = args.frame_ids
        print(f"Using command line frame IDs: {frame_ids}")
    else:
        # Extract frame IDs from saved test frames
        print(f"Extracting frame IDs from {args.test_frames_dir}...")
        frame_ids = extract_frame_ids_from_test_frames(args.test_frames_dir)
        if not frame_ids:
            print(f"Failed to extract frame IDs from test frames")
            # Fallback to default frame IDs
            frame_ids = [100, 300, 500, 700]
            print(f"Using default frame IDs: {frame_ids}")
        else:
            print(f"Extracted {len(frame_ids)} frame IDs: {frame_ids}")

    # Load model and tokenizer
    if args.maskgit:
        model = load_maskgit_model(args.ckpt, device)
    else:
        model = load_ar_model(args.ckpt, device)
    vq = PretrainedTokenizer(args.vq_ckpt).to(device)
    vq.eval()

    # Get BEV encoder
    bev_encoder = model.bev_encoder
    bev_encoder.eval()

    inference_times = []

    for frame_id in frame_ids:
        print(f"\n[Frame {frame_id}] camera={args.fisheye_camera}")
        rows = []

        for yaw_deg in args.yaw_angles:
            print(f"  yaw={yaw_deg:+.0f}° ...")

            # Create dataset with this specific yaw
            ds = Kitti360dDataset(
                drives=drive_path, frames=[frame_id], require_exact_pose=True,
                mode="fisheye_virtual", virtual_hfov_deg=VIRTUAL_HFOV,
                virtual_size=(VIRTUAL_W, VIRTUAL_H),
                fisheye_camera=args.fisheye_camera,
                fisheye_relative_yaw_deg=yaw_deg,
            )
            sample = ds[0]

            # Prepare inputs
            data = prepare_sample(sample, vq, bev_encoder, device)

            # GT image
            gt_pil = tensor01_to_pil(sample["image"])

            # Satellite for heatmap
            sat_np = (data["sat"].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

            # BEV attention heatmap
            model.eval()
            if args.maskgit:
                dummy_tokens = torch.zeros((1, SEQ_LEN), dtype=torch.long, device=device)
                dummy_mask = torch.zeros((1, SEQ_LEN), dtype=torch.bool, device=device)
                _ = model(
                    tokens=dummy_tokens,
                    mask=dummy_mask,
                    condition_tokens=data["condition"],
                    aligned_bev_feature_map=data["bev"],
                    bev_vis_mask=data["bev_vis_mask"],
                )
            else:
                bos = torch.full((1, 1), BOS_TOKEN, dtype=torch.long, device=device)
                _ = model(
                    generated_tokens=bos,
                    condition_tokens=data["condition"],
                    aligned_bev_feature_map=data["bev"],
                    bev_vis_mask=data["bev_vis_mask"],
                    use_cache=False,
                )

            attn_w = getattr(model.pose_route, "_last_attn_weights", None)
            if attn_w is not None:
                aw_np = attn_w[0].cpu().numpy()
                anchor_pts = getattr(model.pose_route, "_last_anchors", None)
                anchor_np = anchor_pts[0].cpu().numpy() if anchor_pts is not None else None
                heatmap_np = render_bev_attn_heatmap(aw_np, sat_img=sat_np.copy(), anchor_points=anchor_np)
            else:
                heatmap_np = sat_np

            heatmap_pil = Image.fromarray(heatmap_np)

            # Generate with timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            if args.maskgit:
                token_grid = generate_maskgit(
                    model, data["condition"], data["bev"], data["bev_vis_mask"],
                    num_steps=args.maskgit_steps, top_k=args.top_k, temperature=args.temperature,
                )
            else:
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
                f"yaw={yaw_deg:+.0f}",
                heatmap_pil,
                tensor01_to_pil(data["ipm_img"]),
                tensor_to_pil(gen_img),
                gt_pil,
            ))

        if rows:
            save_yaw_grid(rows, frame_id, args.out_dir)

    # Print timing statistics
    if inference_times:
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        total_frames = len(inference_times)

        print("\n" + "="*60)
        print("Inference Timing Statistics")
        print("="*60)
        print(f"Model: {'MaskGIT' if args.maskgit else 'AR'}")
        if args.maskgit:
            print(f"MaskGIT steps: {args.maskgit_steps}")
        print(f"Total frames: {total_frames}")
        print(f"Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"Min time: {min_time:.3f}s")
        print(f"Max time: {max_time:.3f}s")
        print(f"FPS: {1.0/avg_time:.2f}")
        print("="*60)

    print("\nDone.")


if __name__ == "__main__":
    main()
