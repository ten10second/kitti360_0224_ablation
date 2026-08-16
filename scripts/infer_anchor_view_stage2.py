#!/usr/bin/env python3
"""Anchor-view stage2 inference for fixed-five-view pairs."""

import argparse
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
from world3d.data.ar_pipeline import FixedFiveViewDataset, compute_bev_visibility_mask
from world3d.train.anchor_view_conditioning import AnchorViewConditioner
from world3d.train.conditioning_ar import build_condition_tokens_with_coords
from world3d.train.geometry_ar import compute_inverse_projection_view
from world3d.train.pose_ar import build_pose_vec
from world3d.train.view_pairing import ANCHOR_VIEW_MAP


D_MODEL = 512
NUM_LAYERS = 8
NHEAD = 8
GRID_ROWS = 16
GRID_COLS = 40
SEQ_LEN = GRID_ROWS * GRID_COLS
VOCAB_SIZE = 1025
BOS_TOKEN = 1024
TARGET_H = GRID_ROWS * 16
TARGET_W = GRID_COLS * 16


def parse_args():
    parser = argparse.ArgumentParser(description="Anchor-view stage2 inference")
    parser.add_argument("--ckpt", default=str(REPO_ROOT / "runs/ar_anchor_view/ckpt_step_0080000.pt"))
    parser.add_argument("--vq-ckpt", default=str(REPO_ROOT / "ckpts/maskgit-vqgan-imagenet-f16-256.bin"))
    parser.add_argument("--data-root", default="/media/zhimiao/Lenovo/KITTI-360")
    parser.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    parser.add_argument("--frame", type=int, default=113)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "runs/ar_anchor_view/inference"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--model-mode", default="hybrid", choices=["direct", "hybrid"])
    parser.add_argument("--anchor-source", default="gt", choices=["gt", "none"])
    parser.add_argument("--virtual-hfov", type=float, default=80.0)
    parser.add_argument("--virtual-w", type=int, default=640)
    parser.add_argument("--virtual-h", type=int, default=256)
    parser.add_argument("--fixed-view-turn-deg", type=float, default=30.0)
    parser.add_argument("--fourier-freqs", type=int, default=10)
    parser.add_argument("--n-pose-queries", type=int, default=64)
    parser.add_argument("--use-ipm-semantic", action="store_true")
    return parser.parse_args()


def load_model_and_conditioner(args, device):
    model = SimplifiedTokenPredictor(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        nhead=NHEAD,
        dropout=0.0,
        max_seq_len=SEQ_LEN,
        target_rows=GRID_ROWS,
        target_cols=GRID_COLS,
        semantic_dim=4,
        fourier_freqs=args.fourier_freqs,
        train_bev_encoder=False,
        no_bev_pretrain=True,
        pose_dim=13,
        use_pose_token=True,
        n_pose_queries=args.n_pose_queries,
        mode=args.model_mode,
        use_ipm_semantic=bool(args.use_ipm_semantic),
    ).to(device)

    anchor_conditioner = AnchorViewConditioner(
        feature_channels=D_MODEL,
        image_channels=3,
        hidden_dim=128,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    load_result = model.load_state_dict(ckpt["model"], strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if missing:
        print(f"[Model] Missing model keys during load: {missing}")
    if unexpected:
        print(f"[Model] Unexpected model keys during load: {unexpected}")
    if "anchor_conditioner" not in ckpt:
        raise KeyError(f"Checkpoint does not contain anchor_conditioner: {args.ckpt}")
    anchor_conditioner.load_state_dict(ckpt["anchor_conditioner"], strict=True)

    model.eval()
    anchor_conditioner.eval()
    print(f"[Model] Loaded {args.model_mode} anchor-view stage2 from {args.ckpt} (step {ckpt.get('step', '?')})")
    return model, anchor_conditioner


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
def generate_ar(model, condition, bev, bev_vis_mask, device, top_k=50, temperature=1.0, anchor_memory=None):
    generated = torch.full((1, 1), BOS_TOKEN, dtype=torch.long, device=device)
    past_kv = None
    for _ in range(SEQ_LEN):
        inp = generated if past_kv is None else generated[:, -1:]
        logits, past_kv = model(
            generated_tokens=inp,
            condition_tokens=condition,
            aligned_bev_feature_map=bev,
            bev_vis_mask=bev_vis_mask,
            past_key_values=past_kv,
            use_cache=True,
            anchor_memory=anchor_memory,
        )
        next_tok = top_k_sample(logits[:, -1, :1024], k=top_k, temperature=temperature)
        generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
    token_seq = generated[:, 1:]
    return model.seq_to_grid(token_seq)


@torch.no_grad()
def prepare_sample(sample, bev_encoder, device, use_ipm_semantic=False):
    rgb = sample["image"].to(device)
    rgb_norm = rgb * 2.0 - 1.0
    K = sample["K"].to(device)
    T_cam_to_world = sample["T_cam_to_world"].to(device)
    T_imu_to_world = sample["T_imu_to_world"].to(device)
    sat = sample["sat"].to(device)

    warped_front, warped_valid, warped_coords = compute_inverse_projection_view(
        sat,
        K,
        T_cam_to_world,
        T_imu_to_world,
        TARGET_H,
        TARGET_W,
        device,
        return_ipm_image=bool(use_ipm_semantic),
    )

    sem_dict, coord_dict = build_condition_tokens_with_coords(
        warped_front,
        warped_coords,
        warped_valid,
        GRID_ROWS,
        GRID_COLS,
        device,
    )

    pose = build_pose_vec(K, T_cam_to_world, T_imu_to_world, TARGET_H, TARGET_W, device)
    bev_feats = bev_encoder(sat.unsqueeze(0))
    bev_vis = compute_bev_visibility_mask(
        K=K,
        T_cam_to_world=T_cam_to_world,
        T_imu_to_world=T_imu_to_world,
        bev_size=64,
        cam_h=TARGET_H,
        cam_w=TARGET_W,
    )

    condition = {
        "coords": coord_dict["fine"].unsqueeze(0),
        "pose": pose.unsqueeze(0),
        "K": K.unsqueeze(0),
        "T_cam_to_world": T_cam_to_world.unsqueeze(0),
    }
    if bool(use_ipm_semantic):
        condition["semantic"] = sem_dict["fine"].unsqueeze(0)

    return {
        "rgb": rgb_norm.unsqueeze(0),
        "rgb_raw": rgb,
        "sat": sat,
        "condition": condition,
        "bev": bev_feats,
        "bev_vis_mask": bev_vis.unsqueeze(0),
        "K": K.unsqueeze(0),
        "T_cam_to_world": T_cam_to_world.unsqueeze(0),
        "T_imu_to_world": T_imu_to_world.unsqueeze(0),
    }


def load_fixed_five_samples(args):
    drive_path = Path(args.data_root) / args.drive
    ds_front = Kitti360dDataset(
        drives=[drive_path],
        frames=[[args.frame]],
        mode="front",
        front_resize=(args.virtual_w, args.virtual_h),
    )
    ds_virtual = Kitti360dDataset(
        drives=[drive_path],
        frames=[[args.frame]],
        mode="fisheye_virtual",
        virtual_hfov_deg=args.virtual_hfov,
        virtual_size=(args.virtual_w, args.virtual_h),
        random_fisheye_relative_yaw=False,
    )
    fixed_ds = FixedFiveViewDataset(
        ds_front,
        ds_virtual,
        turn_to_front_deg=args.fixed_view_turn_deg,
    )

    samples = {}
    for index in range(len(fixed_ds)):
        sample = fixed_ds[index]
        name = sample.get("meta", {}).get("fixed_view_name", f"view_{index}")
        samples[name] = sample
    return samples


def tensor_to_pil(img_tensor):
    arr = (img_tensor.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def tensor01_to_pil(img_tensor):
    arr = img_tensor.detach().cpu().float().clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_result_grid(anchor_pil, gt_pil, baseline_pil, anchor_gen_pil, target_name, anchor_name, overlap_ratio, elapsed_baseline, elapsed_anchor):
    tiles = [anchor_pil, gt_pil, baseline_pil, anchor_gen_pil]
    labels = [f"Anchor: {anchor_name}", f"GT Target: {target_name}", f"Baseline ({elapsed_baseline:.2f}s)", f"Anchor-Cond ({elapsed_anchor:.2f}s)"]
    tile_w, tile_h = tiles[0].size
    pad = 12
    title_h = 44
    canvas = Image.new("RGB", (tile_w * 2 + pad * 3, tile_h * 2 + pad * 3 + title_h * 2), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    positions = [
        (pad, pad + title_h),
        (pad * 2 + tile_w, pad + title_h),
        (pad, pad * 2 + tile_h + title_h * 2),
        (pad * 2 + tile_w, pad * 2 + tile_h + title_h * 2),
    ]
    text_positions = [
        (pad, pad),
        (pad * 2 + tile_w, pad),
        (pad, pad * 2 + tile_h + title_h),
        (pad * 2 + tile_w, pad * 2 + tile_h + title_h),
    ]

    for image, image_pos, text, text_pos in zip(tiles, positions, labels, text_positions):
        canvas.paste(image, image_pos)
        draw.text(text_pos, text, fill=(255, 255, 255), font=font)

    footer = f"Overlap={overlap_ratio:.3f}"
    draw.text((pad, canvas.size[1] - title_h), footer, fill=(220, 220, 220), font=font)
    return canvas


def make_token_diff_heatmap(baseline_grid: torch.Tensor, anchor_grid: torch.Tensor, cell_size: int = 24):
    baseline_np = baseline_grid.detach().cpu().numpy().astype(np.int32)
    anchor_np = anchor_grid.detach().cpu().numpy().astype(np.int32)
    diff = np.abs(anchor_np - baseline_np)
    changed = (diff > 0).astype(np.uint8)

    if diff.max() > 0:
        diff_norm = diff.astype(np.float32) / float(diff.max())
    else:
        diff_norm = np.zeros_like(diff, dtype=np.float32)

    heat = np.zeros(diff.shape + (3,), dtype=np.uint8)
    heat[..., 0] = np.clip(40 + diff_norm * 215, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip(diff_norm * 180, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip((1.0 - diff_norm) * 60, 0, 255).astype(np.uint8)
    heat[changed == 0] = np.array([20, 20, 20], dtype=np.uint8)

    heat_img = Image.fromarray(heat, mode="RGB").resize(
        (diff.shape[1] * cell_size, diff.shape[0] * cell_size),
        resample=Image.Resampling.NEAREST,
    )
    draw = ImageDraw.Draw(heat_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    changed_count = int(changed.sum())
    total_count = int(changed.size)
    draw.rectangle((0, 0, heat_img.size[0], 30), fill=(0, 0, 0))
    draw.text(
        (8, 6),
        f"Changed tokens: {changed_count}/{total_count} ({changed_count / max(1, total_count):.1%})",
        fill=(255, 255, 255),
        font=font,
    )
    return heat_img, changed_count, total_count


def build_anchor_memory(anchor_conditioner, anchor_data, target_data, device):
    anchor_cond_feat, valid_mask = anchor_conditioner(
        anchor_image=anchor_data["rgb"],
        anchor_T_cam_to_world=anchor_data["T_cam_to_world"],
        anchor_T_imu_to_world=anchor_data["T_imu_to_world"],
        anchor_K=anchor_data["K"],
        target_T_cam_to_world=target_data["T_cam_to_world"],
        target_T_imu_to_world=target_data["T_imu_to_world"],
        target_K=target_data["K"],
    )
    anchor_cond_feat = F.adaptive_avg_pool2d(anchor_cond_feat, (GRID_ROWS, GRID_COLS))
    valid_mask_down = F.adaptive_avg_pool2d(valid_mask, (GRID_ROWS, GRID_COLS))
    anchor_cond_feat = anchor_cond_feat * valid_mask_down
    overlap_ratio = float(valid_mask.mean().item())
    return anchor_cond_feat.to(device), overlap_ratio


def get_inference_pairs(samples):
    pairs = []
    for target_name, anchor_name in ANCHOR_VIEW_MAP.items():
        if anchor_name is None:
            continue
        if target_name not in samples or anchor_name not in samples:
            continue
        pairs.append((anchor_name, target_name))
    return pairs


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    model, anchor_conditioner = load_model_and_conditioner(args, device)
    vq = PretrainedTokenizer(args.vq_ckpt).to(device)
    vq.eval()
    bev_encoder = model.bev_encoder
    bev_encoder.eval()

    samples = load_fixed_five_samples(args)
    pairs = get_inference_pairs(samples)
    if not pairs:
        raise RuntimeError("No valid anchor-target pairs found for fixed-five-view inference")

    print(f"[Data] Loaded fixed-five-view samples for frame {args.frame}: {sorted(samples.keys())}")
    print(f"[Pairs] Running {len(pairs)} anchor-target pairs: {pairs}")

    for anchor_name, target_name in pairs:
        anchor_sample = samples[anchor_name]
        target_sample = samples[target_name]

        target_data = prepare_sample(target_sample, bev_encoder, device, use_ipm_semantic=args.use_ipm_semantic)
        anchor_data = prepare_sample(anchor_sample, bev_encoder, device, use_ipm_semantic=args.use_ipm_semantic)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        baseline_grid = generate_ar(
            model,
            target_data["condition"],
            target_data["bev"],
            target_data["bev_vis_mask"],
            device,
            top_k=args.top_k,
            temperature=args.temperature,
            anchor_memory=None,
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        baseline_elapsed = time.time() - t0

        overlap_ratio = 0.0
        anchor_memory = None
        if args.anchor_source == "gt":
            anchor_memory, overlap_ratio = build_anchor_memory(anchor_conditioner, anchor_data, target_data, device)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        anchor_grid = generate_ar(
            model,
            target_data["condition"],
            target_data["bev"],
            target_data["bev_vis_mask"],
            device,
            top_k=args.top_k,
            temperature=args.temperature,
            anchor_memory=anchor_memory,
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        anchor_elapsed = time.time() - t1

        baseline_img = vq.decode(baseline_grid)[0]
        anchor_img = vq.decode(anchor_grid)[0]
        anchor_pil = tensor_to_pil(anchor_data["rgb"][0])
        gt_pil = tensor01_to_pil(target_sample["image"])
        baseline_pil = tensor_to_pil(baseline_img)
        anchor_gen_pil = tensor_to_pil(anchor_img)

        grid = make_result_grid(
            anchor_pil=anchor_pil,
            gt_pil=gt_pil,
            baseline_pil=baseline_pil,
            anchor_gen_pil=anchor_gen_pil,
            target_name=target_name,
            anchor_name=anchor_name,
            overlap_ratio=overlap_ratio,
            elapsed_baseline=baseline_elapsed,
            elapsed_anchor=anchor_elapsed,
        )

        pair_slug = f"{anchor_name}_to_{target_name}"
        pair_dir = Path(args.out_dir) / args.drive / f"frame_{args.frame:010d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        out_path = pair_dir / f"{pair_slug}.png"
        grid.save(out_path)

        np.save(pair_dir / f"{pair_slug}_baseline_tokens.npy", baseline_grid[0].cpu().numpy())
        np.save(pair_dir / f"{pair_slug}_anchor_tokens.npy", anchor_grid[0].cpu().numpy())
        diff_heatmap, changed_count, total_count = make_token_diff_heatmap(baseline_grid[0], anchor_grid[0])
        diff_path = pair_dir / f"{pair_slug}_token_diff.png"
        diff_heatmap.save(diff_path)
        print(
            f"[Saved] {out_path} | overlap={overlap_ratio:.3f} | "
            f"baseline={baseline_elapsed:.2f}s | anchor={anchor_elapsed:.2f}s | "
            f"token_diff={changed_count}/{total_count}"
        )


if __name__ == "__main__":
    main()
