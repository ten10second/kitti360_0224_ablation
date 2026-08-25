#!/usr/bin/env python3
"""Render one dense-ground Stage-A tile as an auditable image montage."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.checkpoints import validate_stage_a_checkpoint, validate_stage_a_dataset
from world3d.unified_bev.data import load_dense_cached_unified_bev
from world3d.unified_bev.geometry import relative_height_map
from world3d.unified_bev.models import ColumnFieldDecoder, GroundDenseBEVEncoder
from world3d.unified_bev.readouts import BEVHeightDecoder, freeze_module


def rgb_image(value: torch.Tensor) -> Image.Image:
    array = (value.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def heatmap_image(value: torch.Tensor, valid: torch.Tensor | None = None) -> Image.Image:
    value = value.detach().cpu().float().squeeze()
    if valid is None:
        valid = torch.isfinite(value)
    else:
        valid = valid.detach().cpu().bool().squeeze() & torch.isfinite(value)
    normalized = torch.zeros_like(value)
    if valid.any():
        samples = value[valid]
        lo = torch.quantile(samples, 0.02)
        hi = torch.quantile(samples, 0.98)
        normalized[valid] = ((value[valid] - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    red = normalized
    green = 1.0 - (2.0 * normalized - 1.0).abs()
    blue = 1.0 - normalized
    color = torch.stack((red, green, blue), dim=-1)
    color[~valid] = 0
    return Image.fromarray((color.numpy() * 255).round().astype("uint8"), mode="RGB")


def labeled(image: Image.Image, title: str, size: tuple[int, int]) -> Image.Image:
    image = image.resize(size, Image.Resampling.NEAREST)
    panel = Image.new("RGB", (size[0], size[1] + 22), "white")
    panel.paste(image, (0, 22))
    ImageDraw.Draw(panel).text((5, 4), title, fill="black")
    return panel


def row(panels: list[Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_a", required=True)
    parser.add_argument("--sample_cache", required=True)
    parser.add_argument("--geometry_cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = load_dense_cached_unified_bev(args.sample_cache, args.geometry_cache)
    checkpoint = torch.load(args.stage_a, map_location=device, weights_only=False)
    validate_stage_a_checkpoint(checkpoint)
    validate_stage_a_dataset(checkpoint, dataset, dense_geometry_attached=True)

    ground = GroundDenseBEVEncoder(**{
        key: checkpoint["ground_config"][key]
        for key in (
            "latent_channels", "bev_height", "bev_width", "context_blocks",
            "conf_threshold", "min_depth_m", "max_depth_m",
        )
    }).to(device)
    decoder = ColumnFieldDecoder(**checkpoint["renderer_config"]).to(device)
    geometry_decoder = BEVHeightDecoder(**checkpoint["geometry_decoder_config"]).to(device)
    ground.load_state_dict(checkpoint["ground"])
    decoder.load_state_dict(checkpoint["decoder"])
    geometry_decoder.load_state_dict(checkpoint["geometry_decoder"])
    for module in (ground, decoder, geometry_decoder):
        freeze_module(module)

    sample = dataset[args.index]
    batch = {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
    with torch.no_grad():
        latent, coverage = ground(
            batch["source_rgb"], batch["source_K"], batch["dense_depth"],
            batch["dense_conf"], batch["source_T_world_cam"],
            batch["origin_xy"], dataset.bev_resolution_m,
        )
        pred_rgb, pred_depth, _ = decoder.render(
            latent, batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
            tile_size_m=dataset.tile_size_m, image_size=dataset.image_size,
        )
        pred_height = geometry_decoder(latent)
        ref_height, height_valid, _ = relative_height_map(
            batch["source_points_world"], batch["source_points_valid"],
            batch["origin_xy"], dataset.bev_resolution_m, dataset.bev_size, dataset.bev_size,
        )

    target_rgb = batch["target_rgb"]
    target_depth = batch["target_depth"]
    depth_valid = batch["target_depth_mask"]
    mse = F.mse_loss(pred_rgb, target_rgb)
    psnr = -10.0 * math.log10(max(float(mse), 1e-12))
    if depth_valid.any():
        depth_absrel = float(
            ((pred_depth[depth_valid] - target_depth[depth_valid]).abs()
             / target_depth[depth_valid].clamp_min(1e-3)).mean()
        )
        depth_rmse = float(
            ((pred_depth[depth_valid] - target_depth[depth_valid]).square().mean()).sqrt()
        )
    else:
        depth_absrel = depth_rmse = float("nan")
    height_mae = float((pred_height[height_valid] - ref_height[height_valid]).abs().mean())

    view_names = ("front L", "front R", "left -45", "left 0", "left +45",
                  "right -45", "right 0", "right +45")
    source_panels = [
        labeled(rgb_image(sample["source_rgb"][i]), name, (240, 144))
        for i, name in enumerate(view_names)
    ]
    rows = [row(source_panels[:4]), row(source_panels[4:])]

    rgb_panels = []
    depth_panels = []
    for view, name in enumerate(("left crop", "right crop")):
        rgb_error = (pred_rgb[0, view] - target_rgb[0, view]).abs().mean(dim=0)
        rgb_panels.extend([
            labeled(rgb_image(target_rgb[0, view]), f"target {name}", (240, 144)),
            labeled(rgb_image(pred_rgb[0, view]), f"reconstruction {name}", (240, 144)),
            labeled(heatmap_image(rgb_error), f"RGB error {name}", (240, 144)),
        ])
        depth_error = (pred_depth[0, view] - target_depth[0, view]).abs()
        depth_panels.extend([
            labeled(heatmap_image(target_depth[0, view], depth_valid[0, view]),
                    f"LiDAR depth {name}", (240, 144)),
            labeled(heatmap_image(pred_depth[0, view]), f"pred depth {name}", (240, 144)),
            labeled(heatmap_image(depth_error, depth_valid[0, view]),
                    f"depth error {name}", (240, 144)),
        ])
    rows.extend([row(rgb_panels[:3]), row(rgb_panels[3:]),
                 row(depth_panels[:3]), row(depth_panels[3:])])
    rows.append(row([
        labeled(rgb_image(sample["satellite"]), "satellite QA (not Stage-A input)", (240, 240)),
        labeled(heatmap_image(coverage[0, 0]), f"VGGT lift coverage {float(coverage.mean()):.3f}", (240, 240)),
        labeled(heatmap_image(ref_height[0, 0], height_valid[0, 0]), "LiDAR relative height", (240, 240)),
        labeled(heatmap_image(pred_height[0, 0]), f"predicted height | MAE {height_mae:.3f} m", (240, 240)),
    ]))

    canvas = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for image_row in rows:
        canvas.paste(image_row, (0, y))
        y += image_row.height
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    target_pair = row([
        labeled(rgb_image(target_rgb[0, 0]), "target left crop", (480, 288)),
        labeled(rgb_image(target_rgb[0, 1]), "target right crop", (480, 288)),
    ])
    reconstruction_pair = row([
        labeled(rgb_image(pred_rgb[0, 0]), f"reconstruction left | overall PSNR {psnr:.2f} dB", (480, 288)),
        labeled(rgb_image(pred_rgb[0, 1]), "reconstruction right", (480, 288)),
    ])
    comparison = Image.new(
        "RGB", (target_pair.width, target_pair.height + reconstruction_pair.height), "white",
    )
    comparison.paste(target_pair, (0, 0))
    comparison.paste(reconstruction_pair, (0, target_pair.height))
    comparison.save(output.with_name(f"{output.stem}_rgb.png"))
    print({
        "output": str(output), "drive": sample["meta"]["drive"],
        "target_fid": int(sample["meta"]["target_fid"]), "psnr": round(psnr, 4),
        "depth_absrel": round(depth_absrel, 4), "depth_rmse_m": round(depth_rmse, 4),
        "height_mae_m": round(height_mae, 4), "coverage": round(float(coverage.mean()), 4),
    })


if __name__ == "__main__":
    main()
