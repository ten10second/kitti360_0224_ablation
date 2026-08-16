#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Yaw angle sweep inference for Direct mode — uses training data pipeline directly."""

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
from models.stage2.simplified_token_predictor import MaskGITTokenPredictor
from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.train.geometry_ar import compute_inverse_projection_view
from world3d.train.conditioning_ar import build_condition_tokens_with_coords
from world3d.train.pose_ar import build_pose_vec
from world3d.data.ar_pipeline import compute_bev_visibility_mask
from world3d.train.vis_utils import render_sat_with_frustum, render_bev_attn_heatmap

# Model config (from ar_direct.yaml / ArTrainConfig)
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
    p = argparse.ArgumentParser(description="Yaw sweep inference for direct/hybrid models (training pipeline)")
    p.add_argument("--ckpt", default=str(REPO_ROOT / "runs/ar_direct/ckpt_step_0060000.pt"))
    p.add_argument("--vq-ckpt", default=str(REPO_ROOT / "ckpts/maskgit-vqgan-imagenet-f16-256.bin"))
    p.add_argument("--data-root", default="/media/zhimiao/Lenovo/KITTI-360")
    p.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs/ar_direct/yaw_sweep"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--mode", default="default", choices=["default", "fixed5", "interpolated", "zero_shot", "360sweep"],
                    help="default: run fixed5 + zero_shot test views | fixed5: run only fixed5 views | interpolated: run only left_to_front_30/right_to_front_30 views | zero_shot: run only held-out zero-shot views | 360sweep: 360° sweep on frame 113")
    p.add_argument("--model-mode", default="direct", choices=["direct", "hybrid"],
                    help="Model architecture to load (direct or hybrid).")
    p.add_argument("--test-drives", type=str, nargs="+",
                    default=None,
                    help="Test drive and frame IDs in format 'drive:frame1,frame2' (for default/fixed5/interpolated/zero_shot mode). If not specified, will auto-read test_frames.txt from each sync directory in data_root.")
    p.add_argument("--sweep-frame", type=int, default=113,
                    help="Frame ID for 360° sweep (for 360sweep mode)")
    p.add_argument("--sweep-interval", type=int, default=20,
                    help="Angle interval in degrees for 360° sweep")
    p.add_argument("--maskgit", action="store_true", help="Use MaskGIT model instead of AR")
    p.add_argument("--maskgit-steps", type=int, default=12, help="MaskGIT iterative decoding steps")
    p.add_argument("--use-ipm-semantic", action="store_true",
                    help="Use semantic condition tokens in model input (coords are always computed)")
    p.add_argument("--n-pose-queries", type=int, default=64,
                    help="Number of pose queries/anchors (must match training config, default: 64)")
    p.add_argument("--hybrid-memory-source", default="enhanced",
                    choices=["enhanced", "anchor", "anchor_tokens"],
                    help="Hybrid memory source to use when --model-mode hybrid (must match training)")

    explicit_pos_group = p.add_mutually_exclusive_group()
    explicit_pos_group.add_argument("--use-explicit-token-pos", dest="use_explicit_token_pos", action="store_true",
                                    help="Enable explicit token positional embedding (must match training)")
    explicit_pos_group.add_argument("--no-use-explicit-token-pos", dest="use_explicit_token_pos", action="store_false",
                                    help="Disable explicit token positional embedding (must match training)")
    p.set_defaults(use_explicit_token_pos=False)

    pose_token_group = p.add_mutually_exclusive_group()
    pose_token_group.add_argument("--use-pose-token", dest="use_pose_token", action="store_true",
                                  help="Enable pose token in memory (must match training)")
    pose_token_group.add_argument("--no-use-pose-token", dest="use_pose_token", action="store_false",
                                  help="Disable pose token in memory (must match training)")
    p.set_defaults(use_pose_token=True)

    strict_group = p.add_mutually_exclusive_group()
    strict_group.add_argument("--strict-load", dest="strict_load", action="store_true",
                              help="Strict checkpoint loading (recommended for consistency)")
    strict_group.add_argument("--non-strict-load", dest="strict_load", action="store_false",
                              help="Allow partial checkpoint loading")
    p.set_defaults(strict_load=True)
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────

def _load_predictor_state_dict_with_compat(model, state_dict, strict_load: bool, checkpoint_label: str):
    if not strict_load:
        model.load_state_dict(state_dict, strict=False)
        return

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as err:
        missing_anchor_gate = {
            "anchor_condition_gate.weight",
            "anchor_condition_gate.bias",
        }
        load_result = model.load_state_dict(state_dict, strict=False)
        missing_keys = set(load_result.missing_keys)
        unexpected_keys = set(load_result.unexpected_keys)
        if unexpected_keys or (missing_keys - missing_anchor_gate):
            raise err
        print(
            f"[Model] {checkpoint_label} predates anchor_condition_gate; "
            "using freshly initialized gate parameters."
        )

def load_model(
    ckpt_path,
    device,
    model_mode: str = "direct",
    use_ipm_semantic: bool = False,
    use_pose_token: bool = True,
    strict_load: bool = True,
    n_pose_queries: int = 64,
    hybrid_memory_source: str = "enhanced",
    use_explicit_token_pos: bool = False,
):
    model = SimplifiedTokenPredictor(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS,
        nhead=NHEAD, dropout=0.0, max_seq_len=SEQ_LEN,
        target_rows=GRID_ROWS, target_cols=GRID_COLS, semantic_dim=4,
        fourier_freqs=10, train_bev_encoder=False, no_bev_pretrain=True,
        pose_dim=13, use_pose_token=use_pose_token, n_pose_queries=n_pose_queries,
        mode=model_mode,
        use_ipm_semantic=use_ipm_semantic,
        hybrid_memory_source=hybrid_memory_source,
        use_explicit_token_pos=use_explicit_token_pos,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get("model_type", "ar") == "maskgit":
        raise ValueError("Checkpoint is MaskGIT, but AR inference path is selected. Remove --maskgit only for AR checkpoints.")
    _load_predictor_state_dict_with_compat(model, ckpt["model"], strict_load, "AR checkpoint")
    # Manually set use_ipm_semantic for pose_aware_anchor_query (in case it wasn't loaded properly)
    if hasattr(model, 'pose_aware_anchor_query') and model.pose_aware_anchor_query is not None:
        model.pose_aware_anchor_query.use_ipm_semantic = use_ipm_semantic
    model.eval()
    print(f"[Model] Loaded {model_mode.capitalize()} AR from {ckpt_path} (step {ckpt.get('step', '?')}), hybrid_memory_source={hybrid_memory_source}, use_explicit_token_pos={use_explicit_token_pos}")
    return model


def load_maskgit_model(
    ckpt_path,
    device,
    model_mode: str = "direct",
    use_ipm_semantic: bool = False,
    use_pose_token: bool = True,
    strict_load: bool = True,
    n_pose_queries: int = 64,
    hybrid_memory_source: str = "enhanced",
    use_explicit_token_pos: bool = False,
): 
    model = MaskGITTokenPredictor(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS,
        nhead=NHEAD, dropout=0.0, max_seq_len=SEQ_LEN,
        target_rows=GRID_ROWS, target_cols=GRID_COLS, semantic_dim=4,
        fourier_freqs=10, train_bev_encoder=False, no_bev_pretrain=True,
        pose_dim=13, use_pose_token=use_pose_token, n_pose_queries=n_pose_queries,
        mode=model_mode,
        use_ipm_semantic=use_ipm_semantic,
        hybrid_memory_source=hybrid_memory_source,
        use_explicit_token_pos=use_explicit_token_pos,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model_type = ckpt.get("model_type", None)
    if model_type not in {"maskgit", "oneslot"}:
        raise ValueError(
            "Checkpoint is not marked as MaskGIT/OneSlot. "
            "Use AR inference path (without --maskgit) for AR checkpoints."
        )
    _load_predictor_state_dict_with_compat(
        model, ckpt["model"], strict_load, f"{str(model_type).capitalize()} checkpoint"
    )
    # Manually set use_ipm_semantic for pose_aware_anchor_query (in case it wasn't loaded properly)
    if hasattr(model, 'pose_aware_anchor_query') and model.pose_aware_anchor_query is not None:
        model.pose_aware_anchor_query.use_ipm_semantic = use_ipm_semantic
    model.eval()
    print(
        f"[Model] Loaded {model_mode.capitalize()} {str(model_type).capitalize()} from {ckpt_path} "
        f"(step {ckpt.get('step', '?')}), hybrid_memory_source={hybrid_memory_source}, "
        f"use_explicit_token_pos={use_explicit_token_pos}"
    )
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


@torch.no_grad()
def generate_maskgit(model, condition, bev, bev_vis_mask, device,
                     top_k=50, temperature=1.0, num_steps=12):
    return model.generate(
        condition_tokens=condition,
        aligned_bev_feature_map=bev,
        bev_vis_mask=bev_vis_mask,
        num_steps=num_steps,
        temperature=temperature,
        top_k=top_k,
    )


# ── Prepare sample using training pipeline ────────────────────────

@torch.no_grad()
def fast_compute_coords_map(K, T_cam_to_world, T_imu_to_world, target_h, target_w, device):
    """只计算 coords_map，完全跳过卫星图 warping，提升推理速度"""
    from utils.geometry.bev_to_camera_warp import IMU_TO_GROUND_HEIGHT

    B = 1
    sat_size = 512
    resolution = 0.2
    add_batch_dim = False

    if K.dim() == 2:
        K = K.unsqueeze(0)
        add_batch_dim = True
    if T_cam_to_world.dim() == 2:
        T_cam_to_world = T_cam_to_world.unsqueeze(0)
        add_batch_dim = True
    if T_imu_to_world is not None and T_imu_to_world.dim() == 2:
        T_imu_to_world = T_imu_to_world.unsqueeze(0)
        add_batch_dim = True

    R_cam_to_world = T_cam_to_world[:, :3, :3]
    t_cam_to_world = T_cam_to_world[:, :3, 3:4]

    # 计算 ground height
    if T_imu_to_world is not None:
        ground_height = float(T_imu_to_world[0, 2, 3].detach().cpu().item() - IMU_TO_GROUND_HEIGHT)
    else:
        ground_height = 0.0

    # 卫星中心
    if T_imu_to_world is not None:
        sat_center = T_imu_to_world[:, :2, 3]
    else:
        sat_center = t_cam_to_world[:, :2, 0]

    # 相机像素网格
    v_cam = torch.arange(target_h, dtype=torch.float32, device=device)
    u_cam = torch.arange(target_w, dtype=torch.float32, device=device)
    vv_cam, uu_cam = torch.meshgrid(v_cam, u_cam, indexing='ij')

    # 像素 → 射线
    K_inv = torch.inverse(K)
    pixels_homo = torch.stack([
        uu_cam.reshape(-1),
        vv_cam.reshape(-1),
        torch.ones_like(uu_cam.reshape(-1)),
    ], dim=0).unsqueeze(0).expand(B, 3, -1)
    rays_cam = torch.bmm(K_inv, pixels_homo)
    rays_world = torch.bmm(R_cam_to_world, rays_cam)

    # 与地平面相交
    rays_z = rays_world[:, 2:3, :]
    t = torch.where(
        rays_z.abs() > 1e-6,
        (ground_height - t_cam_to_world[:, 2:3, :]) / rays_z,
        torch.full_like(rays_z, -1.0),
    )

    points_world = t_cam_to_world + t * rays_world

    # 转为卫星坐标
    X_world = points_world[:, 0, :]
    Y_world = points_world[:, 1, :]

    X_off = X_world - sat_center[:, 0:1]
    Y_off = Y_world - sat_center[:, 1:2]

    u_sat = sat_size / 2 + X_off / resolution
    v_sat = sat_size / 2 - Y_off / resolution

    u_sat = u_sat.reshape(B, target_h, target_w)
    v_sat = v_sat.reshape(B, target_h, target_w)
    t_img = t.reshape(B, target_h, target_w)

    # 有效性判断
    from utils.geometry.bev_to_camera_warp import ipm_valid_mask
    valid_mask = ipm_valid_mask(
        u_sat=u_sat,
        v_sat=v_sat,
        t_img=t_img,
        rays_z=rays_world[:, 2:3, :].reshape(B, target_h, target_w),
        vv_cam=vv_cam,
        K=K,
        W_sat=sat_size,
        H_sat=sat_size,
        t_max=120.0,
        cy_margin_px=0.0,
    )

    # 归一化到 [-1,1]
    u_norm = 2.0 * u_sat / (sat_size - 1) - 1.0
    v_norm = 2.0 * v_sat / (sat_size - 1) - 1.0

    coords_u = torch.where(valid_mask, u_norm, torch.full_like(u_norm, -2.0))
    coords_v = torch.where(valid_mask, v_norm, torch.full_like(v_norm, -2.0))
    coords_map = torch.stack([coords_u, coords_v], dim=1)

    return valid_mask.unsqueeze(1).float(), coords_map


@torch.no_grad()
def prepare_sample(sample, vq, bev_encoder, device, use_ipm_semantic=False):
    """Process a Kitti360dDataset sample into model inputs, same as ArTransformDataset."""
    rgb = sample["image"].to(device)
    rgb_norm = rgb * 2.0 - 1.0  # [0,1] -> [-1,1]
    K = sample["K"].to(device)
    T_cam_to_world = sample["T_cam_to_world"].to(device)
    T_imu_to_world = sample["T_imu_to_world"].to(device)
    sat = sample["sat"].to(device)

    if use_ipm_semantic:
        warped_front, warped_valid, warped_coords = compute_inverse_projection_view(
            sat, K, T_cam_to_world, T_imu_to_world, TARGET_H, TARGET_W, device,
        )
    else:
        # 快速计算，只算 coords，跳过 IPM 图像 warping
        warped_front = None
        warped_valid, warped_coords = fast_compute_coords_map(
            K, T_cam_to_world, T_imu_to_world, TARGET_H, TARGET_W, device
        )

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

    # Build condition tokens
    condition = {
        "coords": coord_dict["fine"].unsqueeze(0),
        "pose": pose.unsqueeze(0),
        "K": K.unsqueeze(0),
        "T_cam_to_world": T_cam_to_world.unsqueeze(0),  # 新增：RayRoPE需要的世界坐标系变换
    }
    if use_ipm_semantic and 'sem_dict' in locals():
        condition["semantic"] = sem_dict["fine"].unsqueeze(0)

    return {
        "condition": condition,
        "bev": bev_feats,
        "bev_vis_mask": bev_vis.unsqueeze(0),
        "rgb": rgb,
        "sat": sat,
        "K_raw": K,
        "T_cam_to_world": T_cam_to_world,
        "T_imu_to_world": T_imu_to_world,
    }


def get_fixed_five_views():
    """Return the fixed 5 training-view specifications."""
    return [
        ("left", "image_02", "image_02", 0.0),
        ("left_to_front_30", "image_02", "image_02", 30.0),
        ("front", "front", None, None),  # (name, source, fisheye_camera, yaw_deg)
        ("right_to_front_30", "image_03", "image_03", -30.0),
        ("right", "image_03", "image_03", 0.0),
    ]


def get_interpolated_views():
    """Return only the interpolated front-side views."""
    return [
        ("left_to_front_30", "image_02", "image_02", 30.0),
        ("right_to_front_30", "image_03", "image_03", -30.0),
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
    if mode == "interpolated":
        return [("interpolated", get_interpolated_views(), "interpolated")]
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


def save_yaw_grid(rows, scene_dir, frame_id):
    """Save visualization grid.

    rows: list of (label, heatmap_pil, gen_pil, gt_pil)
    Layout: one row per yaw angle, columns = BEV Heatmap | Gen | GT
    """
    W, H = rows[0][2].size
    hm_W = H  # heatmap is square, scale to row height
    total_W = hm_W + 2 * W
    nrows = len(rows)
    pad = 30

    grid = Image.new("RGB", (total_W, nrows * H + pad), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    headers = [("BEV Attn", hm_W // 2 - 30),
               ("Generated", hm_W + W // 2 - 40),
               ("GT", hm_W + W + W // 2 - 10)]
    for hdr, x in headers:
        draw.text((x, 5), hdr, fill=(255, 255, 255), font=font)

    for r, (label, hm_img, gen_img, gt_img) in enumerate(rows):
        y = r * H + pad
        hm_resized = hm_img.resize((hm_W, H), Image.LANCZOS)
        grid.paste(hm_resized, (0, y))
        grid.paste(gen_img, (hm_W, y))
        grid.paste(gt_img, (hm_W + W, y))
        draw.text((5, y + 5), label, fill=(0, 255, 255), font=font)

    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    path = scene_dir / f"frame_{frame_id:010d}_yaw_grid.png"
    grid.save(str(path))
    print(f"  Saved grid: {path}")


def save_view_outputs(scene_dir, view_name, frame_id, gen_pil, gt_pil, K_tensor, T_tensor):
    scene_dir = Path(scene_dir)
    # Directly use view name as directory: left, left_to_front_30, front, right_to_front_30, right
    view_dir = scene_dir / view_name
    # Create frame-specific directory (10-digit frame id as folder name)
    frame_dir = view_dir / f"{frame_id:010d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    # Simplified filenames inside frame directory
    gen_pil.save(str(frame_dir / "generated.png"))
    gt_pil.save(str(frame_dir / "gt.png"))
    np.save(frame_dir / "K.npy", K_tensor.detach().cpu().numpy())
    np.save(frame_dir / "T_cam_to_world.npy", T_tensor.detach().cpu().numpy())


def save_view_strip(rows, frame_id, out_dir, drive=None, kind="generated"):
    """Save a 5-view strip image for either generated or GT views."""
    if not rows:
        return
    W, H = rows[0][2].size  # IPM size but generated/gt have same dims
    label_h = 32
    total_w = W * len(rows)
    strip = Image.new("RGB", (total_w, H + label_h), (0, 0, 0))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for idx, (view_name, _, _, gen_img, gt_img) in enumerate(rows):
        img = gen_img if kind == "generated" else gt_img
        strip.paste(img, (idx * W, 0))
        text = view_name
        text_x = idx * W + 10
        text_y = H + 5
        draw.text((text_x, text_y), text, fill=(0, 255, 255), font=font)

    if drive:
        base_dir = Path(out_dir) / drive
    else:
        base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{drive + '_' if drive else ''}frame_{frame_id:010d}_{kind}.png"
    path = base_dir / filename
    strip.save(str(path))
    print(f"  Saved strip: {path}")


# ── Main ──────────────────────────────────────────────────────────

def run_fixed_five_views(args, model, vq, bev_encoder, device, data_root):
    """Test grouped views on test frames from multiple drives."""
    print("\n" + "=" * 60)
    if args.mode == "default":
        print("Mode: Fixed 5 + Zero-shot Views on Test Frames from Multiple Drives")
    elif args.mode == "interpolated":
        print("Mode: Interpolated Views on Test Frames from Multiple Drives")
    elif args.mode == "zero_shot":
        print("Mode: Zero-shot Views on Test Frames from Multiple Drives")
    else:
        print("Mode: Fixed 5 Views on Test Frames from Multiple Drives")
    print("=" * 60)

    view_groups = get_requested_view_groups(args.mode)
    inference_times = []

    # Parse test drives and frames
    test_drive_frames = []
    data_root = Path(data_root)

    if args.test_drives is None:
        # Auto-detect all sync directories with test_frames.txt
        print("[Auto-scan] Reading test_frames.txt from all sync directories...")
        for sync_dir in sorted(data_root.iterdir()):
            if sync_dir.is_dir() and "sync" in sync_dir.name:
                test_frames_path = sync_dir / "test_frames.txt"
                if test_frames_path.exists():
                    with open(test_frames_path, 'r') as f:
                        frames = [int(line.strip()) for line in f if line.strip().isdigit()]
                    if frames:
                        test_drive_frames.append((sync_dir.name, frames))
                        print(f"  Found {sync_dir.name}: {len(frames)} frames")
        if not test_drive_frames:
            raise ValueError("No test_frames.txt found in any sync directory!")
    else:
        # Use user-specified test drives
        for drive_frame_str in args.test_drives:
            if ':' in drive_frame_str:
                # Format: drive:frame1,frame2
                drive, frame_str = drive_frame_str.split(':')
                frames = list(map(int, frame_str.split(',')))
                test_drive_frames.append((drive, frames))
                print(f"  User specified {drive}: {len(frames)} frames")
            else:
                # Format: only drive name, auto-read test_frames.txt
                drive = drive_frame_str
                test_frames_path = data_root / drive / "test_frames.txt"
                if test_frames_path.exists():
                    with open(test_frames_path, 'r') as f:
                        frames = [int(line.strip()) for line in f if line.strip().isdigit()]
                    if frames:
                        frames = frames[:2]  # Only take first 2 frames for quick testing
                        test_drive_frames.append((drive, frames))
                        print(f"  User specified {drive}: loaded {len(frames)} frames (first 2 of test_frames.txt)")
                    else:
                        print(f"[Warning] No valid frames found in {test_frames_path}")
                else:
                    raise ValueError(f"test_frames.txt not found for drive {drive} at {test_frames_path}")

    for drive, frames in test_drive_frames:
        drive_path = Path(data_root) / drive
        print(f"\n\nDrive: {drive}")

        # Scene directory: out_dir/model_mode/drive/ (each sync is a scene)
        base_scene_dir = Path(args.out_dir) / args.model_mode / drive
        base_scene_dir.mkdir(parents=True, exist_ok=True)

        for frame_id in frames:
            print(f"\n[Frame {frame_id}]")

            for group_name, group_views, subdir in view_groups:
                scene_dir = base_scene_dir / subdir
                scene_dir.mkdir(parents=True, exist_ok=True)
                rows = []
                print(f"  [{group_name}]")

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

                    data = prepare_sample(sample, vq, bev_encoder, device, use_ipm_semantic=args.use_ipm_semantic)
                    gt_pil = tensor01_to_pil(sample["image"])
                    sat_np = (data["sat"].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

                    sat_with_frustum = render_sat_with_frustum(
                        sat_img=sat_np,
                        K=data["K_raw"].detach().cpu().numpy(),
                        T_cam_to_world=data["T_cam_to_world"].detach().cpu().numpy(),
                        T_imu_to_world=data["T_imu_to_world"].detach().cpu().numpy(),
                        cam_h=TARGET_H,
                        cam_w=TARGET_W,
                    )

                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()

                    if args.maskgit:
                        token_grid = generate_maskgit(
                            model, data["condition"], data["bev"], data["bev_vis_mask"], device,
                            top_k=args.top_k, temperature=args.temperature,
                            num_steps=args.maskgit_steps,
                        )
                    else:
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
                        Image.fromarray(sat_with_frustum),
                        gen_pil,
                        gt_pil,
                    ))
                    save_view_outputs(
                        scene_dir,
                        view_name,
                        frame_id,
                        gen_pil,
                        gt_pil,
                        data["K_raw"],
                        data["T_cam_to_world"],
                    )

                if rows:
                    save_yaw_grid(rows, scene_dir, frame_id)

    return inference_times


def run_360_sweep(args, model, vq, bev_encoder, device, drive_path):
    """360° sweep on a single frame.

    Uses appropriate camera for GT based on angle:
    - Front camera (image_00): for front-facing angles
    - Left fisheye (image_02): for left side angles
    - Right fisheye (image_03): for right side angles
    """
    print("\n" + "="*60)
    print(f"Mode: 360° Sweep on Frame {args.sweep_frame}")
    print(f"Interval: {args.sweep_interval}°")
    print("="*60)

    frame_id = args.sweep_frame
    angles = list(range(0, 360, args.sweep_interval))
    inference_times = []
    rows = []

    for angle in angles:
        print(f"  Angle {angle}° ...")

        # Determine camera and mode based on angle.
        # Angle convention for sweep visualization:
        #   0° = front, 90° = left, 180° = back, 270° = right.
        #
        # Fisheye-relative yaw convention used by the dataloader is defined around
        # the physical fisheye optical axis:
        #   - image_02 (left fisheye):  relative_yaw = 0°  -> left
        #       +30° -> left_to_front_30,  -30° -> left_back_30
        #   - image_03 (right fisheye): relative_yaw = 0°  -> right
        #       -30° -> right_to_front_30, +30° -> right_back_30
        #
        # Therefore the correct mapping from target sweep angle θ is:
        #   left fisheye:  relative_yaw = 90°  - θ
        #   right fisheye: relative_yaw = 270° - θ
        # rather than using signed installation angles (-90/+90).

        if 330 <= angle or angle < 30:
            # Front view: use front camera directly
            mode = "front"
            fisheye_cam = None
            relative_yaw = None
            camera_label = "front"
        elif 30 <= angle < 210:
            # Left side (30° to 210°): use left fisheye (image_02)
            mode = "fisheye_virtual"
            fisheye_cam = "image_02"
            relative_yaw = 90.0 - float(angle)
            camera_label = "left_fisheye"
        else:  # 210 <= angle < 330
            # Right side (210° to 330°): use right fisheye (image_03)
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

        data = prepare_sample(sample, vq, bev_encoder, device, use_ipm_semantic=args.use_ipm_semantic)
        gt_pil = tensor01_to_pil(sample["image"])
        sat_np = (data["sat"].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        sat_with_frustum = render_sat_with_frustum(
            sat_img=sat_np,
            K=data["K_raw"].detach().cpu().numpy(),
            T_cam_to_world=data["T_cam_to_world"].detach().cpu().numpy(),
            T_imu_to_world=data["T_imu_to_world"].detach().cpu().numpy(),
            cam_h=TARGET_H,
            cam_w=TARGET_W,
        )

        print(f"    Camera: {camera_label}" + (f", yaw={relative_yaw:.1f}°" if relative_yaw is not None else ""))

        # BEV attention heatmap (Direct mode doesn't have pose_route; use frustum overlay)
        heatmap_np = sat_with_frustum

        # Generate
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        if args.maskgit:
            token_grid = generate_maskgit(
                model, data["condition"], data["bev"], data["bev_vis_mask"], device,
                top_k=args.top_k, temperature=args.temperature,
                num_steps=args.maskgit_steps,
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
            f"{angle}°",
            Image.fromarray(heatmap_np),
            tensor_to_pil(gen_img),
            gt_pil,
        ))

    if rows:
        scene_dir = Path(args.out_dir) / args.model_mode / args.drive
        scene_dir.mkdir(parents=True, exist_ok=True)
        save_yaw_grid(rows, scene_dir, frame_id)

    return inference_times


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    drive_path = Path(args.data_root) / args.drive

    # Load model and tokenizer
    if args.maskgit:
        model = load_maskgit_model(
            args.ckpt, device,
            model_mode=args.model_mode,
            use_ipm_semantic=args.use_ipm_semantic,
            use_pose_token=args.use_pose_token,
            strict_load=args.strict_load,
            n_pose_queries=args.n_pose_queries,
            hybrid_memory_source=args.hybrid_memory_source,
            use_explicit_token_pos=args.use_explicit_token_pos,
        )
    else:
        model = load_model(
            args.ckpt, device,
            model_mode=args.model_mode,
            use_ipm_semantic=args.use_ipm_semantic,
            use_pose_token=args.use_pose_token,
            strict_load=args.strict_load,
            n_pose_queries=args.n_pose_queries,
            hybrid_memory_source=args.hybrid_memory_source,
            use_explicit_token_pos=args.use_explicit_token_pos,
        )
    vq = PretrainedTokenizer(args.vq_ckpt).to(device)

    # Get BEV encoder from model
    bev_encoder = model.bev_encoder
    bev_encoder.eval()

    # Run inference based on mode
    if args.mode in {"default", "fixed5", "interpolated", "zero_shot"}:
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
        print("Inference Timing Statistics")
        print("="*60)
        # print(f"Model: {'Direct MaskGIT' if args.maskgit else 'Direct AR'}")
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
