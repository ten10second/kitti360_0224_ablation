#!/usr/bin/env python3
"""Visualize one frame with fixed five views and corresponding IPM results.

This script mirrors the current training-time data path:
1) Kitti360dDataset (front + fisheye_virtual)
2) FixedFiveViewDataset expansion
3) compute_inverse_projection_view for IPM

Output:
- A grid image with 5 rows (one per fixed view) and 3 columns:
  [Perspective | IPM | IPM valid mask]
- A JSON metadata file with per-view intrinsics/extrinsics summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world3d.data.ar_pipeline import FixedFiveViewDataset
from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.train.geometry_ar import compute_inverse_projection_view


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root folder that contains the drive directory.")
    ap.add_argument("--drive", type=str, default="2013_05_28_drive_0003_sync")
    ap.add_argument("--frame", type=int, required=True, help="Frame id to visualize.")
    ap.add_argument("--turn_deg", type=float, default=30.0, help="Turn-to-front angle for fixed side views.")
    ap.add_argument("--virtual_hfov", type=float, default=80.0)
    ap.add_argument("--virtual_w", type=int, default=640)
    ap.add_argument("--virtual_h", type=int, default=256)
    ap.add_argument("--roll_deg", type=float, default=0.0, help="Roll correction for IPM (degrees)")
    ap.add_argument("--pitch_deg", type=float, default=0.0, help="Pitch correction for IPM (degrees)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--out_dir", type=str, default="tools")
    ap.add_argument("--out_name", type=str, default=None, help="Optional output image filename.")
    return ap.parse_args()


def tensor01_to_u8_rgb(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x[0]
    arr = x.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def mask_to_u8_rgb(mask: torch.Tensor) -> np.ndarray:
    if mask.dim() == 4:
        mask = mask[0]
    if mask.dim() == 3:
        mask = mask[0]
    arr = mask.detach().cpu().float().clamp(0.0, 1.0).numpy()
    arr_u8 = (arr * 255.0).round().astype(np.uint8)
    return np.stack([arr_u8, arr_u8, arr_u8], axis=-1)


def draw_text_row(
    canvas: Image.Image,
    y0: int,
    text: str,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text((12, y0 + 6), text, fill=color, font=font)


def as_float_list(x: torch.Tensor) -> List[List[float]]:
    arr = x.detach().cpu().numpy().astype(float)
    return arr.tolist()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    drive_dir = Path(args.data_root) / args.drive
    if not drive_dir.exists():
        raise FileNotFoundError(f"Drive folder not found: {drive_dir}")

    ds_front = Kitti360dDataset(
        drives=drive_dir,
        frames=[int(args.frame)],
        require_exact_pose=True,
        mode="front",
        front_resize=(int(args.virtual_w), int(args.virtual_h)),
        roll_deg=float(args.roll_deg),
        pitch_deg=float(args.pitch_deg),
    )
    ds_virtual = Kitti360dDataset(
        drives=drive_dir,
        frames=[int(args.frame)],
        require_exact_pose=True,
        mode="fisheye_virtual",
        virtual_hfov_deg=float(args.virtual_hfov),
        virtual_size=(int(args.virtual_w), int(args.virtual_h)),
        random_fisheye_relative_yaw=False,
        roll_deg=float(args.roll_deg),
        pitch_deg=float(args.pitch_deg),
    )

    fixed_ds = FixedFiveViewDataset(
        ds_front,
        ds_virtual,
        turn_to_front_deg=float(args.turn_deg),
    )

    n_views = len(fixed_ds.view_specs)
    col_w = int(args.virtual_w)
    col_h = int(args.virtual_h)
    row_header_h = 28
    title_h = 34
    pad = 8
    total_w = col_w * 3 + pad * 4
    total_h = title_h + n_views * (row_header_h + col_h + pad) + pad

    canvas = Image.new("RGB", (total_w, total_h), (18, 18, 18))
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"Fixed-5 Views | frame={args.frame} | turn_deg={args.turn_deg} | roll={args.roll_deg} | pitch={args.pitch_deg}", fill=(255, 255, 255), font=font)
    draw.text((pad + col_w // 2 - 50, 8), "Perspective", fill=(255, 255, 255), font=font)
    draw.text((pad * 2 + col_w + col_w // 2 - 20, 8), "IPM", fill=(255, 255, 255), font=font)
    draw.text((pad * 3 + col_w * 2 + col_w // 2 - 45, 8), "IPM Valid", fill=(255, 255, 255), font=font)

    metadata: Dict[str, Any] = {
        "frame": int(args.frame),
        "data_root": str(Path(args.data_root)),
        "drive": str(args.drive),
        "turn_deg": float(args.turn_deg),
        "roll_deg": float(args.roll_deg),
        "pitch_deg": float(args.pitch_deg),
        "views": [],
    }

    y = title_h
    for i in range(n_views):
        sample = fixed_ds[i]
        meta = sample.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}

        rgb = sample["image"].to(device)
        sat = sample["sat"].to(device)
        K = sample["K"].to(device)
        T_cam_to_world = sample["T_cam_to_world"].to(device)
        T_imu_to_world = sample["T_imu_to_world"].to(device)

        warped_front, warped_valid, _ = compute_inverse_projection_view(
            sat_tensor=sat,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            target_h=col_h,
            target_w=col_w,
            device=device,
        )
        if warped_front is None:
            warped_front = torch.zeros((1, 3, col_h, col_w), device=device)
        if warped_valid is None:
            warped_valid = torch.zeros((1, 1, col_h, col_w), device=device)

        rgb_u8 = tensor01_to_u8_rgb(rgb)
        ipm_u8 = tensor01_to_u8_rgb(warped_front)
        valid_u8 = mask_to_u8_rgb(warped_valid)

        draw_text_row(
            canvas,
            y0=y,
            text=(
                f"[{i}] {meta.get('fixed_view_name', 'unknown')} | "
                f"cam={meta.get('aug', {}).get('fisheye_camera_used', meta.get('fisheye_camera_used', meta.get('fisheye_camera_override', 'image_00')))} | "
                f"yaw={meta.get('aug', {}).get('fisheye_relative_yaw_deg', meta.get('fisheye_relative_yaw_deg', meta.get('fisheye_relative_yaw_deg_override', 'n/a')))}"
            ),
            font=font,
        )

        y_img = y + row_header_h
        canvas.paste(Image.fromarray(rgb_u8), (pad, y_img))
        canvas.paste(Image.fromarray(ipm_u8), (pad * 2 + col_w, y_img))
        canvas.paste(Image.fromarray(valid_u8), (pad * 3 + col_w * 2, y_img))

        metadata["views"].append(
            {
                "index": int(i),
                "name": str(meta.get("fixed_view_name", "unknown")),
                "frame_id": int(sample.get("frame_id", args.frame)),
                "fisheye_camera_used": meta.get("aug", {}).get("fisheye_camera_used", meta.get("fisheye_camera_used", meta.get("fisheye_camera_override"))),
                "fisheye_relative_yaw_deg": meta.get("aug", {}).get("fisheye_relative_yaw_deg", meta.get(
                    "fisheye_relative_yaw_deg",
                    meta.get("fisheye_relative_yaw_deg_override"),
                )),
                "K": as_float_list(K),
                "T_cam_to_world": as_float_list(T_cam_to_world),
                "T_imu_to_world": as_float_list(T_imu_to_world),
            }
        )

        y += row_header_h + col_h + pad

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_name:
        out_png = out_dir / args.out_name
    else:
        out_png = out_dir / f"fixed5_views_ipm_frame_{args.frame:010d}.png"
    out_json = out_png.with_suffix(".json")

    canvas.save(out_png)
    out_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK] Saved visualization: {out_png}")
    print(f"[OK] Saved metadata:      {out_json}")

    for item in metadata["views"]:
        K_np = np.array(item["K"], dtype=np.float64)
        print(
            f"  - [{item['index']}] {item['name']}: "
            f"fx={K_np[0,0]:.3f}, fy={K_np[1,1]:.3f}, cx={K_np[0,2]:.3f}, cy={K_np[1,2]:.3f}"
        )


if __name__ == "__main__":
    main()
