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
from world3d.unified_bev.data import (
    SAT_M_PER_PX,
    UnifiedBEVDataset,
    _open_image,
    dense_geometry_from_blob,
    dense_geometry_subset_qa,
    validate_geometry_blob_identity,
)
from world3d.unified_bev.checkpoints import (
    validate_stage_a_checkpoint,
    validate_stage_a_dataset,
    validate_stage_b_checkpoint,
)
from world3d.unified_bev.geometry import (
    geometry_supervision_support,
    observation_partition,
    relative_height_map,
    target_pixels_supported_by_bev,
)
from world3d.unified_bev.losses import low_frequency
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundDenseBEVEncoder,
    GroundBEVEncoder,
    HeightMapSatellitePrior,
    LatentCompletion,
    SatelliteBEVEncoder,
    SatelliteViTEncoder,
    nadir_distance,
    satellite_bev_crop,
)
from world3d.unified_bev.readouts import BEVHeightDecoder, freeze_module


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
    weights = mask.expand_as(pred).to(pred.dtype)
    numer = ((pred - target).abs() * weights).flatten(1).sum(dim=1)
    denom = weights.flatten(1).sum(dim=1)
    return [float(numer[i] / denom[i]) if denom[i] > 0 else float("nan")
            for i in range(pred.shape[0])]


def per_item_masked_rmse(pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor) -> List[float]:
    weights = mask.expand_as(pred).to(pred.dtype)
    numer = ((pred - target).square() * weights).flatten(1).sum(dim=1)
    denom = weights.flatten(1).sum(dim=1)
    return [float(torch.sqrt(numer[i] / denom[i])) if denom[i] > 0 else float("nan")
            for i in range(pred.shape[0])]


def per_item_masked_psnr(pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor) -> List[float]:
    weights = mask.expand_as(pred).to(pred.dtype)
    numer = ((pred - target).square() * weights).flatten(1).sum(dim=1)
    denom = weights.flatten(1).sum(dim=1)
    return [
        float(10.0 * torch.log10(1.0 / (numer[i] / denom[i]).clamp_min(1e-10)))
        if denom[i] > 0 else float("nan")
        for i in range(pred.shape[0])
    ]


def per_item_lowfreq_psnr(pred: torch.Tensor, target: torch.Tensor,
                          scale: int = 8) -> List[float]:
    return per_item_psnr(low_frequency(pred, scale), low_frequency(target, scale))


def per_item_masked_abs_mean(value: torch.Tensor, mask: torch.Tensor) -> List[float]:
    weights = mask.expand(-1, value.shape[1], -1, -1).to(value.dtype)
    numer = (value.abs() * weights).flatten(1).sum(dim=1)
    denom = weights.flatten(1).sum(dim=1)
    return [float(numer[i] / denom[i]) if denom[i] > 0 else float("nan")
            for i in range(value.shape[0])]


def per_item_masked_mean(value: torch.Tensor, mask: torch.Tensor) -> List[float]:
    weights = mask.expand(-1, value.shape[1], -1, -1).to(value.dtype)
    numer = (value * weights).flatten(1).sum(dim=1)
    denom = weights.flatten(1).sum(dim=1)
    return [float(numer[i] / denom[i]) if denom[i] > 0 else float("nan")
            for i in range(value.shape[0])]


def _gaussian_window(device, dtype, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    return (g @ g.t()).unsqueeze(0).unsqueeze(0)


def per_item_ssim(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    """Standard SSIM (11x11 Gaussian window, L=1), matching the definition
    used by Sat2Scene / S-NeRF / CrossView-Splatter-style NVS tables."""
    if pred.ndim == 5:
        batch_size, target_views = pred.shape[:2]
        flat = per_item_ssim(
            pred.reshape(-1, *pred.shape[-3:]), target.reshape(-1, *target.shape[-3:]),
        )
        return [sum(flat[i * target_views:(i + 1) * target_views]) / target_views
                for i in range(batch_size)]
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
    if pred.ndim == 5:
        batch_size, target_views = pred.shape[:2]
        values = net(
            pred.reshape(-1, *pred.shape[-3:]) * 2 - 1,
            target.reshape(-1, *target.shape[-3:]) * 2 - 1,
        ).reshape(batch_size, target_views, -1).mean(dim=(1, 2))
        return [float(value) for value in values]
    return [float(v) for v in net(pred * 2 - 1, target * 2 - 1).flatten()]


def load_donor_satellites(
    ds: UnifiedBEVDataset,
    indexes: range,
    device: torch.device,
) -> tuple[torch.Tensor, List[Dict[str, str | int]]]:
    """Load different target-centered tiles without decoding donor ground views."""
    offset = max(1, len(ds) // 2)
    images = []
    metadata: List[Dict[str, str | int]] = []
    for index in indexes:
        donor_target, _ = ds.samples[(index + offset) % len(ds)]
        with _open_image(donor_target.sat_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32).copy()
        images.append(torch.from_numpy(arr).permute(2, 0, 1) / 255.0)
        metadata.append({"drive": donor_target.drive, "target_fid": donor_target.fid})
    return torch.stack(images).to(device), metadata


def mean_finite(values: List[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return sum(finite) / len(finite) if finite else float("nan")


def json_safe_record(row: Dict[str, object]) -> Dict[str, object]:
    """Replace non-finite floats with JSON ``null`` instead of NaN tokens."""
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in row.items()
    }


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
    ap.add_argument("--geometry_cache", "--m3d_cache", dest="geometry_cache", default=None,
                    help="dense geometry cache aligned with this eval split")
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
    geometry_blobs = None
    if args.geometry_cache:
        geometry_blobs = [torch.load(Path(args.geometry_cache) / f"{i:06d}.pt", map_location="cpu", weights_only=False)
                          for i in range(len(ds))]
        for index, blob in enumerate(geometry_blobs):
            target, sources = ds.samples[index]
            validate_geometry_blob_identity(
                blob,
                {
                    "drive": target.drive,
                    "target_fid": target.fid,
                    "source_fids": [source.fid for source in sources],
                    "view_layout_version": ds.view_layout_version,
                },
                context=f"geometry cache index {index}",
            )
    lpips_net = None
    if args.eval_ssim_lpips:
        import lpips
        lpips_net = lpips.LPIPS(net="alex").to(device)
        lpips_net.eval()
    a = torch.load(args.stage_a, map_location=device, weights_only=False)
    stage_a_fingerprint = validate_stage_a_checkpoint(a)
    validate_stage_a_dataset(
        a, ds, dense_geometry_attached=bool(args.geometry_cache),
    )
    ground_config = dict(a["ground_config"])
    ground_family = ground_config.pop("family")
    ground_cls = GroundDenseBEVEncoder if ground_family == "dense" else GroundBEVEncoder
    ground = ground_cls(**ground_config).to(device)
    decoder = ColumnFieldDecoder(**a["renderer_config"]).to(device)
    geometry_decoder = BEVHeightDecoder(**a["geometry_decoder_config"]).to(device)
    ground.load_state_dict(a["ground"])
    decoder.load_state_dict(a["decoder"])
    geometry_decoder.load_state_dict(a["geometry_decoder"])
    have_b = args.stage_b is not None
    b = None
    fusion_mode = None
    coordinate_prior = None
    uses_satellite = False
    sat = None
    if have_b:
        b = torch.load(args.stage_b, map_location=device, weights_only=False)
        validate_stage_b_checkpoint(b, stage_a_fingerprint)
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
    modules = (ground, decoder, geometry_decoder) if not have_b else (
        (ground, decoder, geometry_decoder, sat, completion) if sat is not None
        else (ground, decoder, geometry_decoder, completion)
    )
    for module in modules:
        freeze_module(module)

    records: List[Dict[str, float | int | str]] = []
    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch_size = batch["target_rgb"].shape[0]
            batch_blobs = (geometry_blobs[sample_offset:sample_offset + batch_size]
                           if geometry_blobs is not None else None)

            def _lift(start_frame, frame_count):
                sl = slice(start_frame * ds.views_per_frame,
                           (start_frame + frame_count) * ds.views_per_frame)
                if batch_blobs is not None:
                    pairs = [dense_geometry_from_blob(
                        blob, start_frame, frame_count, ds.views_per_frame,
                    ) for blob in batch_blobs]
                    dense_depth = torch.stack([p[0] for p in pairs]).to(device)
                    dense_conf = torch.stack([p[1] for p in pairs]).to(device)
                    return ground(
                        batch["source_rgb"][:, sl], batch["source_K"][:, sl],
                        dense_depth, dense_conf,
                        batch["source_T_world_cam"][:, sl],
                        batch["origin_xy"], ds.bev_resolution_m,
                    )
                return ground(
                    batch["source_rgb"][:, sl], batch["source_points_world"][:, sl],
                    batch["source_points_uv"][:, sl], batch["source_points_valid"][:, sl],
                    batch["origin_xy"], ds.bev_resolution_m,
                )
            z_star, ref_mask = _lift(0, args.dense_sources)
            z_sparse, sparse_mask = _lift(0, args.sparse_sources)
            h_ref, h_valid, ground_z = relative_height_map(
                batch["source_points_world"], batch["source_points_valid"],
                batch["origin_xy"], ds.bev_resolution_m, ds.bev_size, ds.bev_size,
            )
            dense_height_support = geometry_supervision_support(ref_mask, h_valid)
            observed_cells, fill_cells = observation_partition(
                sparse_mask, dense_height_support,
            )
            height_aux_pred = None
            donor_metadata: List[Dict[str, str | int]] | None = None
            completion_output = None
            if have_b:
                if uses_satellite:
                    sat_input = batch["satellite"]
                    if args.sat_random_tile:
                        sat_input, donor_metadata = load_donor_satellites(
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
                        z_sat, height_aux_pred = sat(
                            sat_input, z_sparse, ds.tile_size_m, SAT_M_PER_PX,
                        )
                    else:
                        z_sat = sat(sat_input, ds.tile_size_m, SAT_M_PER_PX)
                else:
                    z_sat = torch.zeros_like(z_sparse)
                completion_output = completion(
                    z_sat, z_sparse, sparse_mask,
                    n_sparse=args.sparse_sources, dense_sources=args.dense_sources,
                )
                z_full = completion_output.latent

            def render(z):
                return decoder.render(
                    z, batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
                    tile_size_m=ds.tile_size_m, image_size=ds.image_size,
                )[:2]

            latent_branches = {
                "dense": z_star,
                "sparse": z_sparse,
                # A schema-stable ground-only control: without Stage B the
                # recovered state is exactly the sparse-ground latent.
                "full": z_full if have_b else z_sparse,
            }
            if have_b:
                if uses_satellite:
                    latent_branches["sat_prior"] = z_sat
            predictions = {name: render(latent) for name, latent in latent_branches.items()}
            height_predictions = {
                name: geometry_decoder(latent) for name, latent in latent_branches.items()
            }
            teacher_depth = torch.where(
                batch["target_depth_mask"], batch["target_depth"], predictions["dense"][1],
            )
            teacher_depth_valid = torch.isfinite(teacher_depth) & (teacher_depth > 1e-3)
            target_supported_pixels = target_pixels_supported_by_bev(
                teacher_depth, teacher_depth_valid,
                batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
                ds.tile_size_m, sparse_mask,
            )

            if not have_b:
                condition = "ground_only"
            elif fusion_mode == "coordinate_only":
                condition = "fixed_xy"
            elif args.sat_random_tile:
                condition = "random_tile"
            elif any(value != 0.0 for value in (
                args.sat_shift_x_m, args.sat_shift_y_m,
                args.sat_shift_road_m, args.sat_shift_cross_m,
                args.sat_rotate_deg,
            )):
                condition = "misaligned"
            else:
                condition = "aligned"

            target_rgb = batch["target_rgb"]
            target_depth = batch["target_depth"]
            target_depth_mask = batch["target_depth_mask"]
            tile_rows: List[Dict[str, float | int | str]] = []
            for local in range(batch_size):
                meta = batch["meta"]
                row: Dict[str, float | int | str] = {
                    "drive": meta["drive"][local],
                    "target_fid": int(meta["target_fid"][local]),
                    "condition": condition,
                    "dense_sources": args.dense_sources,
                    "sparse_sources": args.sparse_sources,
                    "fusion": fusion_mode or "none",
                    "satellite_encoder": (
                        b.get("config", {}).get("sat_encoder", "none") if b is not None else "none"
                    ),
                    "coordinate_prior": coordinate_prior or "none",
                    "geometry_model": "lidar_sparse",
                    "stage_a_fingerprint": stage_a_fingerprint,
                    "sat_shift_x_m": args.sat_shift_x_m,
                    "sat_shift_y_m": args.sat_shift_y_m,
                    "sat_shift_road_m": args.sat_shift_road_m,
                    "sat_shift_cross_m": args.sat_shift_cross_m,
                    "sat_rotate_deg": args.sat_rotate_deg,
                    "sat_random_tile": int(args.sat_random_tile),
                    "ground_z_m": float(ground_z[local, 0, 0, 0]),
                    "observed_cell_fraction": float(observed_cells[local].float().mean()),
                    "fill_cell_fraction": float(fill_cells[local].float().mean()),
                    "height_valid_fraction": float(h_valid[local].float().mean()),
                    "dense_height_support_fraction": float(
                        dense_height_support[local].float().mean()
                    ),
                    "target_supported_fraction": float(
                        target_supported_pixels[local].float().mean()
                    ),
                    "target_supported_px": int(target_supported_pixels[local].sum()),
                }
                if donor_metadata is not None:
                    row["donor_drive"] = str(donor_metadata[local]["drive"])
                    row["donor_target_fid"] = int(donor_metadata[local]["target_fid"])
                if batch_blobs is not None:
                    blob = batch_blobs[local]
                    row["geometry_model"] = str(blob.get("geometry_model", "legacy"))
                    for prefix, start, count in (
                        ("dense_geometry", 0, args.dense_sources),
                        ("sparse_geometry", 0, args.sparse_sources),
                    ):
                        for key, value in dense_geometry_subset_qa(
                            blob, start, count,
                        ).items():
                            row[f"{prefix}_{key}"] = value
                tile_rows.append(row)

            for name, (rgb, depth) in predictions.items():
                rgb_l1 = per_item_l1(rgb, target_rgb)
                rgb_psnr = per_item_psnr(rgb, target_rgb)
                rgb_lowfreq_psnr = per_item_lowfreq_psnr(rgb, target_rgb)
                rgb_supported_psnr = per_item_masked_psnr(
                    rgb, target_rgb, target_supported_pixels,
                )
                depth_rows = depth_metrics(depth, target_depth, target_depth_mask)
                rgb_ssim = per_item_ssim(rgb, target_rgb) if lpips_net is not None else None
                rgb_lpips = per_item_lpips(rgb, target_rgb, lpips_net) if lpips_net is not None else None
                rgb_supported_lpips = (
                    per_item_lpips(
                        rgb * target_supported_pixels,
                        target_rgb * target_supported_pixels,
                        lpips_net,
                    ) if lpips_net is not None else None
                )
                for local, row in enumerate(tile_rows):
                    row[f"{name}_rgb_l1"] = rgb_l1[local]
                    row[f"{name}_psnr"] = rgb_psnr[local]
                    row[f"{name}_rgb_lowfreq_psnr"] = rgb_lowfreq_psnr[local]
                    row[f"{name}_rgb_supported_psnr"] = rgb_supported_psnr[local]
                    if rgb_ssim is not None:
                        row[f"{name}_ssim"] = rgb_ssim[local]
                        row[f"{name}_lpips"] = rgb_lpips[local]
                        row[f"{name}_rgb_supported_lpips"] = rgb_supported_lpips[local]
                    row[f"{name}_absrel"] = depth_rows[local]["absrel"]
                    row[f"{name}_rmse"] = depth_rows[local]["rmse"]
                    row[f"{name}_delta1"] = depth_rows[local]["delta1"]
                    row[f"{name}_depth_valid_px"] = int(depth_rows[local]["valid_px"])

            for name, height_pred in height_predictions.items():
                region_masks = {
                    "all": h_valid,
                    "observed": observed_cells & h_valid,
                    "fill": fill_cells,
                }
                for region, region_mask in region_masks.items():
                    maes = per_item_masked_l1(height_pred, h_ref, region_mask)
                    rmses = per_item_masked_rmse(height_pred, h_ref, region_mask)
                    counts = region_mask.flatten(1).sum(dim=1)
                    for local, row in enumerate(tile_rows):
                        row[f"{name}_height_{region}_mae"] = maes[local]
                        row[f"{name}_height_{region}_rmse"] = rmses[local]
                        row[f"{name}_height_{region}_cells"] = int(counts[local])

            if height_aux_pred is not None:
                aux_mae = per_item_masked_l1(height_aux_pred, h_ref, h_valid)
                aux_rmse = per_item_masked_rmse(height_aux_pred, h_ref, h_valid)
                for local, row in enumerate(tile_rows):
                    row["satellite_height_aux_mae"] = aux_mae[local]
                    row["satellite_height_aux_rmse"] = aux_rmse[local]

            if completion_output is not None:
                correction_observed = per_item_masked_abs_mean(
                    completion_output.correction, observed_cells,
                )
                correction_fill = per_item_masked_abs_mean(
                    completion_output.correction, fill_cells,
                )
                gate_observed = per_item_masked_mean(
                    completion_output.write_gate, observed_cells,
                )
                gate_fill = per_item_masked_mean(
                    completion_output.write_gate, fill_cells,
                )
                for local, row in enumerate(tile_rows):
                    row["correction_observed_norm"] = correction_observed[local]
                    row["correction_fill_norm"] = correction_fill[local]
                    row["write_gate_observed_mean"] = gate_observed[local]
                    row["write_gate_fill_mean"] = gate_fill[local]
            else:
                for row in tile_rows:
                    row["correction_observed_norm"] = 0.0
                    row["correction_fill_norm"] = 0.0

            for row in tile_rows:
                row["geometry_observed_mae"] = row["full_height_observed_mae"]
                row["geometry_observed_rmse"] = row["full_height_observed_rmse"]
                row["geometry_fill_mae"] = row["full_height_fill_mae"]
                row["geometry_fill_rmse"] = row["full_height_fill_rmse"]
                row["rgb_lowfreq_psnr"] = row["full_rgb_lowfreq_psnr"]
                row["rgb_supported_psnr"] = row["full_rgb_supported_psnr"]

            if args.eval_nadir:
                nadir_ref = satellite_bev_crop(
                    batch["satellite"], ds.tile_size_m, SAT_M_PER_PX, ds.bev_size,
                )
                for name, latent in latent_branches.items():
                    nadir_rgb, _ = decoder.render_nadir(
                        latent, batch["origin_xy"], tile_size_m=ds.tile_size_m,
                        bev_size=ds.bev_size,
                    )
                    for local, row in enumerate(tile_rows):
                        row[f"nadir_l1_{name}"] = float(nadir_distance(
                            nadir_rgb[local:local + 1], nadir_ref[local:local + 1],
                            (~h_valid[local:local + 1]).to(nadir_rgb.dtype),
                        ))

            latent_values = {
                "sparse_latent_l1_diag": per_item_l1(z_sparse, z_star),
                "coverage_dense": [float(x) for x in ref_mask.flatten(1).mean(dim=1)],
                "coverage_sparse": [float(x) for x in sparse_mask.flatten(1).mean(dim=1)],
            }
            recovered = z_full if have_b else z_sparse
            latent_values.update({
                "full_latent_l1_diag": per_item_l1(recovered, z_star),
                "full_observed_l1_diag": per_item_masked_l1(
                    recovered, z_star, observed_cells,
                ),
                "full_fill_l1_diag": per_item_masked_l1(recovered, z_star, fill_cells),
            })
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
    paired: Dict[str, float | str] = {}
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
            "delta_height_fill_mae_sparse_minus_full": mean_finite([
                float(r["sparse_height_fill_mae"]) - float(r["full_height_fill_mae"])
                for r in records
            ]),
            "delta_height_fill_rmse_sparse_minus_full": mean_finite([
                float(r["sparse_height_fill_rmse"]) - float(r["full_height_fill_rmse"])
                for r in records
            ]),
            "delta_lowfreq_psnr_full_minus_sparse": mean_finite([
                float(r["full_rgb_lowfreq_psnr"]) - float(r["sparse_rgb_lowfreq_psnr"])
                for r in records
            ]),
            "delta_supported_psnr_full_minus_sparse": mean_finite([
                float(r["full_rgb_supported_psnr"]) - float(r["sparse_rgb_supported_psnr"])
                for r in records
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
        out.write_text("".join(
            json.dumps(json_safe_record(row), allow_nan=False) + "\n" for row in records
        ))
        summary_path = out.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(json_safe_record({
            "paired_tiles": len(records),
            "decoder": "frozen",
            "geometry_decoder": "stage_a_shared_bev_height",
            "summary": json_safe_record(summary),
            "paired": json_safe_record(paired),
            "control": control,
        }), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
