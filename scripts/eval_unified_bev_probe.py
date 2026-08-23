#!/usr/bin/env python3
"""Paired frozen-decoder evaluation for unified-BEV Stage A/B probes.

Reports RGB, latent, and LiDAR-anchored camera-z depth metrics. Optional
satellite perturbations implement the B7/B8 inference-only controls.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world3d.unified_bev.data import SAT_M_PER_PX, UnifiedBEVDataset, _open_image
from world3d.unified_bev.geometry import bilinear_splat, height_statistics
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    HeightMapSatellitePrior,
    LatentCompletion,
    SatelliteBEVEncoder,
    SatelliteViTEncoder,
    nadir_distance,
    satellite_bev_crop,
)


def perturb_satellite(
    sat: torch.Tensor,
    *,
    meters_per_pixel: float,
    shift_x_m: float = 0.0,
    shift_y_m: float = 0.0,
    rotate_deg: float = 0.0,
) -> torch.Tensor:
    """Misregister north-up satellite content around the image center.

    Positive x shifts content east/right, positive y shifts content north/up,
    and positive rotation is counter-clockwise in the north-up display.
    """
    if shift_x_m == 0.0 and shift_y_m == 0.0 and rotate_deg == 0.0:
        return sat
    B, _, H, W = sat.shape
    angle = math.radians(rotate_deg)
    c, s = math.cos(angle), math.sin(angle)
    # Desired content translation in normalized image coordinates. affine_grid
    # needs the inverse map from output locations to source locations.
    tx = 2.0 * shift_x_m / (meters_per_pixel * W)
    ty = -2.0 * shift_y_m / (meters_per_pixel * H)
    theta = sat.new_tensor([
        [c, -s, -(c * tx - s * ty)],
        [s,  c, -(s * tx + c * ty)],
    ]).unsqueeze(0).expand(B, -1, -1)
    grid = F.affine_grid(theta, sat.shape, align_corners=False)
    return F.grid_sample(sat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def road_headings(ds: UnifiedBEVDataset) -> np.ndarray:
    """Unit travel-direction vectors (east, north) of the target trajectory.

    Eval samples follow drive order, so central differences of neighboring
    target IMU positions give the local road direction used by the B7
    along-road / cross-road misregistration controls.
    """
    pos = np.asarray([
        [target.T_world_imu[0, 3], target.T_world_imu[1, 3]]
        for target, _ in ds.samples
    ])
    headings = np.zeros((len(pos), 2), dtype=np.float64)
    for i in range(len(pos)):
        a, b = max(0, i - 1), min(len(pos) - 1, i + 1)
        d = pos[b] - pos[a]
        norm = float(np.hypot(d[0], d[1]))
        headings[i] = d / norm if norm > 1e-6 else (0.0, 1.0)
    return headings


def road_frame_shift(heading: np.ndarray, road_m: float, cross_m: float) -> tuple[float, float]:
    """Convert along-road / left-of-road meters into (east, north) meters.

    ``cross`` follows the left-hand normal of the unit travel direction
    (rotate the heading +90 degrees counter-clockwise in ENU).
    """
    h = np.asarray(heading, dtype=np.float64)
    normal = np.array([-h[1], h[0]])
    return float(road_m * h[0] + cross_m * normal[0]), float(road_m * h[1] + cross_m * normal[1])


def depth_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> List[Dict[str, float]]:
    """Per-item AbsRel/RMSE/delta1 on sparse LiDAR-valid camera-z pixels."""
    output: List[Dict[str, float]] = []
    for p, t, m in zip(pred, target, mask):
        valid = m & torch.isfinite(p) & torch.isfinite(t) & (t > 1e-3)
        if not valid.any():
            output.append({"absrel": float("nan"), "rmse": float("nan"),
                           "delta1": float("nan"), "valid_px": 0.0})
            continue
        pv = p[valid].clamp_min(1e-3)
        tv = t[valid]
        ratio = torch.maximum(pv / tv, tv / pv)
        output.append({
            "absrel": float(((pv - tv).abs() / tv).mean()),
            "rmse": float(torch.sqrt(((pv - tv) ** 2).mean())),
            "delta1": float((ratio < 1.25).float().mean()),
            "valid_px": float(valid.sum()),
        })
    return output


def per_item_l1(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    return [float(x) for x in (pred - target).abs().flatten(1).mean(dim=1)]


def per_item_psnr(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    mse = (pred - target).square().flatten(1).mean(dim=1).clamp_min(1e-10)
    return [float(x) for x in 10.0 * torch.log10(1.0 / mse)]


def per_item_masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> List[float]:
    weights = mask.expand(-1, pred.shape[1], -1, -1).to(pred.dtype)
    numer = ((pred - target).abs() * weights).flatten(1).sum(dim=1)
    denom = weights.flatten(1).sum(dim=1).clamp_min(1.0)
    return [float(x) for x in numer / denom]


def _gaussian_window(device, dtype, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    return (g @ g.t()).unsqueeze(0).unsqueeze(0)


def per_item_ssim(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    """Standard SSIM (11x11 Gaussian window, L=1), matching the definition
    used by Sat2Scene / S-NeRF / CrossView-Splatter-style NVS tables."""
    window = _gaussian_window(pred.device, pred.dtype)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    out: List[float] = []
    for p, t in zip(pred, target):
        w = window.expand(p.shape[0], 1, -1, -1)
        conv = lambda x: F.conv2d(x.unsqueeze(0), w, groups=p.shape[0])
        mu1, mu2 = conv(p), conv(t)
        sigma1 = conv(p * p) - mu1 ** 2
        sigma2 = conv(t * t) - mu2 ** 2
        sigma12 = conv(p * t) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
        )
        out.append(float(ssim_map.mean()))
    return out


def per_item_lpips(pred: torch.Tensor, target: torch.Tensor, net) -> List[float]:
    # lpips expects inputs in [-1, 1].
    return [float(v) for v in net(pred * 2 - 1, target * 2 - 1).flatten()]


def load_donor_satellites(ds: UnifiedBEVDataset, indexes: range, device: torch.device) -> torch.Tensor:
    """Load different target-centered tiles without decoding donor ground views."""
    offset = max(1, len(ds) // 2)
    images = []
    for index in indexes:
        donor_target, _ = ds.samples[(index + offset) % len(ds)]
        with _open_image(donor_target.sat_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32).copy()
        images.append(torch.from_numpy(arr).permute(2, 0, 1) / 255.0)
    return torch.stack(images).to(device)


def mean_finite(values: List[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return sum(finite) / max(1, len(finite))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--stage_b", default=None,
                    help="optional; when omitted only dense/sparse ground metrics are computed")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--sparse_sources", type=int, default=2)
    ap.add_argument("--max_points", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min_target_spacing_m", type=float, default=5.0)
    ap.add_argument("--sat_shift_x_m", type=float, default=0.0)
    ap.add_argument("--sat_shift_y_m", type=float, default=0.0)
    ap.add_argument("--sat_shift_road_m", type=float, default=0.0,
                    help="B7: shift satellite content along the local road direction")
    ap.add_argument("--sat_shift_cross_m", type=float, default=0.0,
                    help="B7: shift satellite content left of the local road direction")
    ap.add_argument("--sat_rotate_deg", type=float, default=0.0)
    ap.add_argument("--sat_random_tile", action="store_true")
    ap.add_argument("--eval_nadir", action="store_true",
                    help="report nadir round-trip distance of each latent branch vs the satellite crop")
    ap.add_argument("--eval_ssim_lpips", action="store_true",
                    help="report SSIM and LPIPS(alex) for every render branch")
    ap.add_argument("--records_out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ds = UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, drive=args.drive,
        min_target_spacing_m=args.min_target_spacing_m, max_samples=args.max_samples,
        dense_source_count=args.dense_sources, sparse_source_count=args.sparse_sources,
        image_size=(160, 96), max_points_per_view=args.max_points,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    headings = road_headings(ds)
    lpips_net = None
    if args.eval_ssim_lpips:
        import lpips
        lpips_net = lpips.LPIPS(net="alex").to(device)
        lpips_net.eval()
    a = torch.load(args.stage_a, map_location=device, weights_only=False)
    ground = GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    decoder = ColumnFieldDecoder(
        hidden=a.get("config", {}).get("hidden", 128),
        samples=a.get("config", {}).get("ray_samples", 24),
    ).to(device)
    ground.load_state_dict(a["ground"])
    decoder.load_state_dict(a["decoder"])
    have_b = args.stage_b is not None
    b = None
    fusion_mode = None
    coordinate_prior = None
    uses_satellite = False
    sat = None
    if have_b:
        b = torch.load(args.stage_b, map_location=device, weights_only=False)
        fusion_mode = b.get("config", {}).get("fusion", "residual")
        if fusion_mode == "coordinate_only":
            coordinate_prior = b.get("config", {}).get(
                "coordinate_prior", "legacy_learned_spatial_template",
            )
        uses_satellite = fusion_mode in ("residual", "satellite_only")
        completion = LatentCompletion(
            mode=fusion_mode,
            bev_height=ds.bev_size, bev_width=ds.bev_size,
            tile_size_m=ds.tile_size_m,
        ).to(device)
        completion.load_state_dict(b["completion"])
        completion.eval()
        if uses_satellite:
            if b.get("config", {}).get("sat_encoder", "cnn") == "heightmap":
                sat = HeightMapSatellitePrior(
                    bev_height=ds.bev_size, bev_width=ds.bev_size,
                    **b["config"].get("sat_encoder_kwargs", {}),
                ).to(device)
            elif b.get("config", {}).get("sat_encoder", "cnn") == "vit":
                sat = SatelliteViTEncoder(
                    bev_height=ds.bev_size, bev_width=ds.bev_size,
                    **b["config"].get("sat_encoder_kwargs", {}),
                ).to(device)
            else:
                sat = SatelliteBEVEncoder(
                    bev_height=ds.bev_size, bev_width=ds.bev_size,
                ).to(device)
            sat.load_state_dict(b["satellite_encoder"])
            sat.eval()
    modules = (ground, decoder) if not have_b else (
        (ground, decoder, sat, completion) if sat is not None
        else (ground, decoder, completion)
    )
    for module in modules:
        module.eval()

    records: List[Dict[str, float | int | str]] = []
    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch_size = batch["target_rgb"].shape[0]
            z_star, ref_mask = ground(
                batch["source_rgb"], batch["source_points_world"], batch["source_points_uv"],
                batch["source_points_valid"], batch["origin_xy"], ds.bev_resolution_m,
            )
            sp = slice(0, args.sparse_sources * ds.views_per_frame)
            z_sparse, sparse_mask = ground(
                batch["source_rgb"][:, sp], batch["source_points_world"][:, sp],
                batch["source_points_uv"][:, sp], batch["source_points_valid"][:, sp],
                batch["origin_xy"], ds.bev_resolution_m,
            )
            height_metrics = None
            if have_b:
                if uses_satellite:
                    sat_input = batch["satellite"]
                    if args.sat_random_tile:
                        sat_input = load_donor_satellites(
                            ds, range(sample_offset, sample_offset + batch_size), device,
                        )
                    shift_x_m = args.sat_shift_x_m
                    shift_y_m = args.sat_shift_y_m
                    if args.sat_shift_road_m != 0.0 or args.sat_shift_cross_m != 0.0:
                        if batch_size != 1:
                            raise SystemExit("road-frame satellite shifts require --batch_size 1")
                        road_dx, road_dy = road_frame_shift(
                            headings[sample_offset], args.sat_shift_road_m, args.sat_shift_cross_m,
                        )
                        shift_x_m += road_dx
                        shift_y_m += road_dy
                    sat_input = perturb_satellite(
                        sat_input, meters_per_pixel=SAT_M_PER_PX,
                        shift_x_m=shift_x_m, shift_y_m=shift_y_m,
                        rotate_deg=args.sat_rotate_deg,
                    )
                    if b.get("config", {}).get("sat_encoder") == "heightmap":
                        z_sat, h_pred, _ = sat(sat_input, z_sparse, ds.tile_size_m, SAT_M_PER_PX)
                        # DEM-quality metrics of the height branch vs dense LiDAR.
                        dense_pts = batch["source_points_world"]
                        h_ref, _ = height_statistics(
                            dense_pts, batch["source_points_valid"], batch["origin_xy"],
                            ds.bev_resolution_m, ds.bev_size, ds.bev_size,
                        )
                        h_ref = h_ref.clamp(max=30.0)
                        ones = batch["source_points_valid"].float().unsqueeze(-1)
                        _, cnt = bilinear_splat(
                            ones, dense_pts[..., :2], batch["source_points_valid"],
                            origin_xy=batch["origin_xy"], resolution_m=ds.bev_resolution_m,
                            height=ds.bev_size, width=ds.bev_size,
                        )
                        m = (cnt > 0)
                        if m.any():
                            err = (h_pred - h_ref).abs()
                            mae = float((err * m).sum() / m.sum())
                            a = (h_pred - h_pred.mean())[m]
                            bb = (h_ref - h_ref.mean())[m]
                            r = float((a * bb).sum() / max(float(a.norm() * bb.norm()), 1e-8))
                        else:
                            mae, r = float("nan"), float("nan")
                        height_metrics = (mae, r)
                    else:
                        z_sat = sat(sat_input, ds.tile_size_m, SAT_M_PER_PX)
                else:
                    z_sat = torch.zeros_like(z_sparse)
                z_full = completion(
                    z_sat, z_sparse, sparse_mask,
                    n_sparse=args.sparse_sources, dense_sources=args.dense_sources,
                )

            def render(z):
                return decoder.render(
                    z, batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
                    tile_size_m=ds.tile_size_m, image_size=ds.image_size,
                )[:2]

            predictions = {"dense": render(z_star), "sparse": render(z_sparse)}
            if have_b:
                predictions["full"] = render(z_full)
                if uses_satellite:
                    predictions["sat"] = render(z_sat)

            target_rgb = batch["target_rgb"]
            target_depth = batch["target_depth"]
            target_depth_mask = batch["target_depth_mask"]
            tile_rows: List[Dict[str, float | int | str]] = []
            for local in range(batch_size):
                meta = batch["meta"]
                tile_rows.append({
                    "drive": meta["drive"][local],
                    "target_fid": int(meta["target_fid"][local]),
                    "sparse_sources": args.sparse_sources,
                    "coordinate_prior": coordinate_prior,
                })
            if height_metrics is not None:
                for row in tile_rows:
                    row["height_mae"], row["height_pearson"] = height_metrics

            for name, (rgb, depth) in predictions.items():
                rgb_l1 = per_item_l1(rgb, target_rgb)
                rgb_psnr = per_item_psnr(rgb, target_rgb)
                depth_rows = depth_metrics(depth, target_depth, target_depth_mask)
                rgb_ssim = per_item_ssim(rgb, target_rgb) if lpips_net is not None else None
                rgb_lpips = per_item_lpips(rgb, target_rgb, lpips_net) if lpips_net is not None else None
                for local, row in enumerate(tile_rows):
                    row[f"{name}_rgb_l1"] = rgb_l1[local]
                    row[f"{name}_psnr"] = rgb_psnr[local]
                    if rgb_ssim is not None:
                        row[f"{name}_ssim"] = rgb_ssim[local]
                        row[f"{name}_lpips"] = rgb_lpips[local]
                    row[f"{name}_absrel"] = depth_rows[local]["absrel"]
                    row[f"{name}_rmse"] = depth_rows[local]["rmse"]
                    row[f"{name}_delta1"] = depth_rows[local]["delta1"]
                    row[f"{name}_depth_valid_px"] = int(depth_rows[local]["valid_px"])

            if args.eval_nadir:
                # Nadir round-trip: how close is the frozen decoder's top-down
                # render of each latent to the aligned satellite crop?  The
                # unobserved-cell mask matches the training-side supervision.
                nadir_ref = satellite_bev_crop(
                    batch["satellite"], ds.tile_size_m, SAT_M_PER_PX, ds.bev_size,
                )
                nadir_latents = {"dense": z_star, "sparse": z_sparse}
                if have_b:
                    nadir_latents["full"] = z_full
                    if uses_satellite:
                        nadir_latents["sat"] = z_sat
                for name, z_lat in nadir_latents.items():
                    nadir_rgb, _ = decoder.render_nadir(
                        z_lat, batch["origin_xy"], tile_size_m=ds.tile_size_m,
                        bev_size=ds.bev_size,
                    )
                    row_value = float(nadir_distance(nadir_rgb, nadir_ref, 1.0 - ref_mask))
                    for row in tile_rows:
                        row[f"nadir_l1_{name}"] = row_value

            latent_values = {
                "sparse_latent_l1": per_item_l1(z_sparse, z_star),
                "coverage_dense": [float(x) for x in ref_mask.flatten(1).mean(dim=1)],
                "coverage_sparse": [float(x) for x in sparse_mask.flatten(1).mean(dim=1)],
            }
            if have_b:
                latent_values.update({
                    "full_latent_l1": per_item_l1(z_full, z_star),
                    "full_observed_l1": per_item_masked_l1(z_full, z_star, sparse_mask),
                    "full_unobserved_l1": per_item_masked_l1(z_full, z_star, 1.0 - sparse_mask),
                })
                if uses_satellite:
                    latent_values["sat_latent_l1"] = per_item_l1(z_sat, z_star)
            for key, values in latent_values.items():
                for local, row in enumerate(tile_rows):
                    row[key] = values[local]
            records.extend(tile_rows)
            sample_offset += batch_size

    numeric_keys = sorted({key for row in records for key, value in row.items()
                           if isinstance(value, (int, float)) and key != "target_fid"})
    summary = {
        key: round(mean_finite([float(row[key]) for row in records if key in row]), 6)
        for key in numeric_keys
    }
    print(summary)
    if have_b:
        paired = {
            "delta_psnr_full_minus_sparse": mean_finite([
                float(r["full_psnr"]) - float(r["sparse_psnr"]) for r in records
            ]),
            "delta_absrel_sparse_minus_full": mean_finite([
                float(r["sparse_absrel"]) - float(r["full_absrel"]) for r in records
            ]),
            "delta_rmse_sparse_minus_full": mean_finite([
                float(r["sparse_rmse"]) - float(r["full_rmse"]) for r in records
            ]),
            "delta_delta1_full_minus_sparse": mean_finite([
                float(r["full_delta1"]) - float(r["sparse_delta1"]) for r in records
            ]),
            "psnr_wins": f"{sum(float(r['full_psnr']) > float(r['sparse_psnr']) for r in records)}/{len(records)}",
        }
        print({key: round(value, 4) if isinstance(value, float) else value
               for key, value in paired.items()})
    control = {
        "shift_x_m": args.sat_shift_x_m,
        "shift_y_m": args.sat_shift_y_m,
        "shift_road_m": args.sat_shift_road_m,
        "shift_cross_m": args.sat_shift_cross_m,
        "rotate_deg": args.sat_rotate_deg,
        "random_tile": args.sat_random_tile,
    }
    print({
        "paired_tiles": len(records), "decoder": "frozen", "mpp": SAT_M_PER_PX,
        "fusion": fusion_mode, "coordinate_prior": coordinate_prior,
        "satellite_control": control,
    })
    if args.records_out:
        out = Path(args.records_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in records))


if __name__ == "__main__":
    main()
