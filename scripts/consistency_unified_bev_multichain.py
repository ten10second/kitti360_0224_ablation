#!/usr/bin/env python3
"""C2 frozen-query consistency for disjoint ground observation subsets.

Two non-overlapping source subsets independently run the complete geometry
backend, ground encoder, and completion path.  The primary question is not
whether every latent channel is numerically identical; it is whether one
Stage-A-trained frozen geometry/render interface reads compatible world
states from both outputs.  Raw latent distances are emitted only as
``*_diag`` fields.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.data import (
    SAT_M_PER_PX,
    UnifiedBEVDataset,
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
    relative_height_map,
)
from world3d.unified_bev.losses import low_frequency
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    GroundDenseBEVEncoder,
    HeightMapSatellitePrior,
    LatentCompletion,
    SatelliteBEVEncoder,
    SatelliteViTEncoder,
)
from world3d.unified_bev.readouts import BEVHeightDecoder, freeze_module


def _masked_values(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] > 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    elif value.ndim == 3 and mask.ndim == 4:
        mask = mask[:, 0]
    return value[mask]


def masked_mae(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    values = _masked_values((a - b).abs(), mask)
    return float(values.mean()) if values.numel() else float("nan")


def masked_rmse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    values = _masked_values((a - b).square(), mask)
    return float(values.mean().sqrt()) if values.numel() else float("nan")


def masked_l1_diag(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    return masked_mae(a, b, mask)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).clamp_min(1e-10)
    return float(10.0 * torch.log10(1.0 / mse))


def finite_mean(rows: Iterable[dict], key: str) -> float:
    values = [float(row[key]) for row in rows
              if key in row and row[key] is not None and math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else float("nan")


def json_safe(row: Dict[str, object]) -> Dict[str, object]:
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in row.items()
    }


def build_satellite_encoder(checkpoint: dict, bev_size: int):
    config = checkpoint.get("config", {})
    family = config.get("sat_encoder", "cnn")
    kwargs = config.get("sat_encoder_kwargs", {})
    if family == "heightmap":
        return HeightMapSatellitePrior(
            bev_height=bev_size, bev_width=bev_size, **kwargs,
        )
    if family == "vit":
        return SatelliteViTEncoder(
            bev_height=bev_size, bev_width=bev_size, **kwargs,
        )
    return SatelliteBEVEncoder(bev_height=bev_size, bev_width=bev_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_a", required=True)
    parser.add_argument("--stage_b_sat", required=True)
    parser.add_argument("--stage_b_xy", required=True)
    parser.add_argument(
        "--manifest",
        default="dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl",
    )
    parser.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    parser.add_argument("--eval_samples", type=int, default=32)
    parser.add_argument("--dense_sources", type=int, default=8)
    parser.add_argument("--ns_list", default="1,2")
    parser.add_argument("--chain_a_start", type=int, default=0)
    parser.add_argument("--chain_b_start", type=int, default=4)
    parser.add_argument(
        "--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw",
    )
    parser.add_argument(
        "--records_out", default="runs/consistency_frozen_query_0003.jsonl",
    )
    parser.add_argument(
        "--geometry_cache", "--m3d_cache", dest="geometry_cache", default=None,
        help="exact-subset geometry cache; joint VGGT must contain both chain starts",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ns_values = tuple(int(value) for value in args.ns_list.split(",") if value)
    if not ns_values:
        raise ValueError("ns_list cannot be empty")
    for ns in ns_values:
        a_range = set(range(args.chain_a_start, args.chain_a_start + ns))
        b_range = set(range(args.chain_b_start, args.chain_b_start + ns))
        if a_range & b_range:
            raise ValueError(f"C2 source subsets overlap for Ns={ns}")
        if max(a_range | b_range) >= args.dense_sources:
            raise ValueError(f"C2 source subset exceeds dense_sources for Ns={ns}")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    dataset = UnifiedBEVDataset(
        args.manifest,
        lidar_root=args.lidar_root,
        drive=None if args.drive == "none" else args.drive,
        min_target_spacing_m=5.0,
        max_samples=args.eval_samples,
        dense_source_count=args.dense_sources,
        sparse_source_count=args.dense_sources,
        image_size=(160, 96),
        max_points_per_view=4096,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    views_per_frame = dataset.views_per_frame

    geometry_blobs = None
    if args.geometry_cache:
        geometry_blobs = [
            torch.load(
                Path(args.geometry_cache) / f"{index:06d}.pt",
                map_location="cpu", weights_only=False,
            )
            for index in range(len(dataset))
        ]
        for index, blob in enumerate(geometry_blobs):
            target, sources = dataset.samples[index]
            validate_geometry_blob_identity(
                blob,
                {
                    "drive": target.drive,
                    "target_fid": target.fid,
                    "source_fids": [source.fid for source in sources],
                    "view_layout_version": dataset.view_layout_version,
                },
                context=f"geometry cache index {index}",
            )

    stage_a = torch.load(args.stage_a, map_location=device, weights_only=False)
    stage_a_fingerprint = validate_stage_a_checkpoint(stage_a)
    validate_stage_a_dataset(
        stage_a, dataset, dense_geometry_attached=geometry_blobs is not None,
    )
    ground_config = dict(stage_a["ground_config"])
    ground_family = ground_config.pop("family")
    ground_cls = GroundDenseBEVEncoder if ground_family == "dense" else GroundBEVEncoder
    ground = ground_cls(**ground_config).to(device)
    renderer = ColumnFieldDecoder(**stage_a["renderer_config"]).to(device)
    geometry_decoder = BEVHeightDecoder(**stage_a["geometry_decoder_config"]).to(device)
    ground.load_state_dict(stage_a["ground"])
    renderer.load_state_dict(stage_a["decoder"])
    geometry_decoder.load_state_dict(stage_a["geometry_decoder"])

    stage_b_sat = torch.load(args.stage_b_sat, map_location=device, weights_only=False)
    stage_b_xy = torch.load(args.stage_b_xy, map_location=device, weights_only=False)
    validate_stage_b_checkpoint(stage_b_sat, stage_a_fingerprint)
    validate_stage_b_checkpoint(stage_b_xy, stage_a_fingerprint)
    if stage_b_xy.get("config", {}).get("coordinate_prior") != "fixed_metric_relative_xy_fourier_v1":
        raise RuntimeError("C2 requires the fixed metric relative-XY control checkpoint")
    satellite_encoder = build_satellite_encoder(stage_b_sat, dataset.bev_size).to(device)
    satellite_encoder.load_state_dict(stage_b_sat["satellite_encoder"])
    completion_sat = LatentCompletion(
        mode="residual", bev_height=dataset.bev_size, bev_width=dataset.bev_size,
        tile_size_m=dataset.tile_size_m,
    ).to(device)
    completion_xy = LatentCompletion(
        mode="coordinate_only", bev_height=dataset.bev_size, bev_width=dataset.bev_size,
        tile_size_m=dataset.tile_size_m,
    ).to(device)
    completion_sat.load_state_dict(stage_b_sat["completion"])
    completion_xy.load_state_dict(stage_b_xy["completion"])
    for module in (
        ground, renderer, geometry_decoder, satellite_encoder,
        completion_sat, completion_xy,
    ):
        freeze_module(module)

    def lift(batch: dict, sample_index: int, start: int, count: int):
        view_slice = slice(
            start * views_per_frame, (start + count) * views_per_frame,
        )
        if geometry_blobs is not None:
            depth, confidence = dense_geometry_from_blob(
                geometry_blobs[sample_index], start, count, views_per_frame,
            )
            return ground(
                batch["source_rgb"][:, view_slice], batch["source_K"][:, view_slice],
                depth.unsqueeze(0).to(device), confidence.unsqueeze(0).to(device),
                batch["source_T_world_cam"][:, view_slice], batch["origin_xy"],
                dataset.bev_resolution_m,
            )
        return ground(
            batch["source_rgb"][:, view_slice],
            batch["source_points_world"][:, view_slice],
            batch["source_points_uv"][:, view_slice],
            batch["source_points_valid"][:, view_slice],
            batch["origin_xy"], dataset.bev_resolution_m,
        )

    def complete_chain(batch: dict, sample_index: int, start: int, ns: int):
        z_ground, support = lift(batch, sample_index, start, ns)
        if stage_b_sat.get("config", {}).get("sat_encoder") == "heightmap":
            prior, _ = satellite_encoder(
                batch["satellite"], z_ground, dataset.tile_size_m, SAT_M_PER_PX,
            )
        else:
            prior = satellite_encoder(
                batch["satellite"], dataset.tile_size_m, SAT_M_PER_PX,
            )
        return {
            "sat": completion_sat(
                prior, z_ground, support, ns, args.dense_sources,
            ).latent,
            "xy": completion_xy(
                torch.zeros_like(prior), z_ground, support, ns, args.dense_sources,
            ).latent,
            "support": support.bool(),
        }

    records: list[dict] = []
    different_location_cache: Dict[tuple[int, str], list[torch.Tensor]] = {
        (ns, family): [] for ns in ns_values for family in ("sat", "xy")
    }
    reference_masks: list[torch.Tensor] = []

    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            z_star, dense_encoder_support = lift(
                batch, sample_index, 0, args.dense_sources,
            )
            h_ref, h_valid, _ = relative_height_map(
                batch["source_points_world"], batch["source_points_valid"],
                batch["origin_xy"], dataset.bev_resolution_m,
                dataset.bev_size, dataset.bev_size,
            )
            reference = geometry_supervision_support(
                dense_encoder_support, h_valid,
            )
            reference_masks.append(reference.cpu())
            row: dict = {
                "drive": batch["meta"]["drive"][0],
                "target_fid": int(batch["meta"]["target_fid"][0]),
                "chain_a_start": args.chain_a_start,
                "chain_b_start": args.chain_b_start,
                "geometry_model": (
                    str(geometry_blobs[sample_index].get("geometry_model", "legacy"))
                    if geometry_blobs is not None else "lidar_sparse"
                ),
            }
            if geometry_blobs is not None:
                blob = geometry_blobs[sample_index]
                for key, value in dense_geometry_subset_qa(
                    blob, 0, args.dense_sources,
                ).items():
                    row[f"dense_geometry_{key}"] = value
            h_star = geometry_decoder(z_star)

            for ns in ns_values:
                if geometry_blobs is not None:
                    for chain_name, start in (
                        ("a", args.chain_a_start),
                        ("b", args.chain_b_start),
                    ):
                        for key, value in dense_geometry_subset_qa(
                            geometry_blobs[sample_index], start, ns,
                        ).items():
                            row[f"ns{ns}_chain_{chain_name}_geometry_{key}"] = value
                chain_a = complete_chain(batch, sample_index, args.chain_a_start, ns)
                chain_b = complete_chain(batch, sample_index, args.chain_b_start, ns)
                support_a = chain_a["support"]
                support_b = chain_b["support"]
                regions = {
                    "shared": reference & support_a & support_b,
                    "aonly": reference & support_a & ~support_b,
                    "bonly": reference & support_b & ~support_a,
                    "neither": reference & ~support_a & ~support_b,
                }
                for region_name, region_mask in regions.items():
                    row[f"ns{ns}_{region_name}_cells"] = int(region_mask.sum())

                for family in ("sat", "xy"):
                    latent_a = chain_a[family]
                    latent_b = chain_b[family]
                    height_a = geometry_decoder(latent_a)
                    height_b = geometry_decoder(latent_b)
                    different_location_cache[(ns, family)].append(height_a.cpu())
                    row[f"ns{ns}_{family}_height_ab_mae"] = masked_mae(
                        height_a, height_b, reference,
                    )
                    row[f"ns{ns}_{family}_height_ab_rmse"] = masked_rmse(
                        height_a, height_b, reference,
                    )

                    rgb_a, depth_a, _ = renderer.render(
                        latent_a, batch["target_K"], batch["target_T_world_cam"],
                        batch["origin_xy"], tile_size_m=dataset.tile_size_m,
                        image_size=dataset.image_size,
                    )
                    rgb_b, depth_b, _ = renderer.render(
                        latent_b, batch["target_K"], batch["target_T_world_cam"],
                        batch["origin_xy"], tile_size_m=dataset.tile_size_m,
                        image_size=dataset.image_size,
                    )
                    for region_name, region_mask in regions.items():
                        prefix = f"ns{ns}_{family}_{region_name}"
                        row[f"{prefix}_height_ab_mae"] = masked_mae(
                            height_a, height_b, region_mask,
                        )
                        row[f"{prefix}_height_ab_rmse"] = masked_rmse(
                            height_a, height_b, region_mask,
                        )
                        row[f"{prefix}_height_a_ref_mae"] = masked_mae(
                            height_a, h_ref, region_mask,
                        )
                        row[f"{prefix}_height_b_ref_mae"] = masked_mae(
                            height_b, h_ref, region_mask,
                        )
                        row[f"{prefix}_latent_ab_l1_diag"] = masked_l1_diag(
                            latent_a, latent_b, region_mask,
                        )

                    depth_mask = batch["target_depth_mask"]
                    row[f"ns{ns}_{family}_depth_ab_l1"] = masked_mae(
                        depth_a, depth_b, depth_mask,
                    )
                    row[f"ns{ns}_{family}_depth_a_target_l1"] = masked_mae(
                        depth_a, batch["target_depth"], depth_mask,
                    )
                    row[f"ns{ns}_{family}_depth_b_target_l1"] = masked_mae(
                        depth_b, batch["target_depth"], depth_mask,
                    )
                    row[f"ns{ns}_{family}_render_lowfreq_psnr"] = psnr(
                        low_frequency(rgb_a), low_frequency(rgb_b),
                    )
                    row[f"ns{ns}_{family}_height_a_star_mae"] = masked_mae(
                        height_a, h_star, reference,
                    )
                    row[f"ns{ns}_{family}_height_b_star_mae"] = masked_mae(
                        height_b, h_star, reference,
                    )
                    row[f"ns{ns}_{family}_latent_full_l1_diag"] = float(
                        (latent_a - latent_b).abs().mean()
                    )
            records.append(row)

    # Anti-collapse diagnostic in the same frozen geometry interface: compare
    # chain-A decoded heights at different geographic tiles.
    for index, row in enumerate(records):
        partner = (index + 1) % len(records)
        row["different_location_partner_fid"] = records[partner]["target_fid"]
        for ns in ns_values:
            shared_reference = reference_masks[index] & reference_masks[partner]
            for family in ("sat", "xy"):
                height_here = different_location_cache[(ns, family)][index]
                height_other = different_location_cache[(ns, family)][partner]
                different = masked_mae(
                    height_here, height_other, shared_reference,
                )
                row[f"ns{ns}_{family}_height_diffloc_mae_diag"] = different
                same = row[f"ns{ns}_{family}_height_ab_mae"]
                row[f"ns{ns}_{family}_height_same_over_diffloc_diag"] = (
                    float(same) / float(different)
                    if math.isfinite(float(same)) and math.isfinite(float(different))
                    and float(different) > 1e-8 else float("nan")
                )

    output = Path(args.records_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(
        json.dumps(json_safe(row), allow_nan=False) + "\n" for row in records
    ))
    summary = {
        "paired_tiles": len(records),
        "stage_a_fingerprint": stage_a_fingerprint,
        "chain_a_start": args.chain_a_start,
        "chain_b_start": args.chain_b_start,
        "metrics": {
            f"ns{ns}_{family}_{metric}": finite_mean(
                records, f"ns{ns}_{family}_{metric}",
            )
            for ns in ns_values
            for family in ("sat", "xy")
            for metric in (
                "height_ab_mae", "height_ab_rmse", "depth_ab_l1",
                "render_lowfreq_psnr", "height_same_over_diffloc_diag",
            )
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(json_safe({
            **summary,
            "metrics": json_safe(summary["metrics"]),
        }), indent=2, allow_nan=False)
    )

    print(
        f"=== frozen-query C2 on {args.drive} "
        f"(A={args.chain_a_start}, B={args.chain_b_start}) ==="
    )
    for ns in ns_values:
        print(f"\n-- Ns={ns}")
        for family in ("sat", "xy"):
            shared = finite_mean(records, f"ns{ns}_{family}_shared_height_ab_mae")
            aonly = finite_mean(records, f"ns{ns}_{family}_aonly_height_ab_mae")
            bonly = finite_mean(records, f"ns{ns}_{family}_bonly_height_ab_mae")
            neither = finite_mean(records, f"ns{ns}_{family}_neither_height_ab_mae")
            depth = finite_mean(records, f"ns{ns}_{family}_depth_ab_l1")
            render_psnr = finite_mean(records, f"ns{ns}_{family}_render_lowfreq_psnr")
            print(
                f"{family:>4s} height_MAE shared={shared:.3f} aonly={aonly:.3f} "
                f"bonly={bonly:.3f} neither={neither:.3f} "
                f"depth_AB={depth:.3f} lowfreq_PSNR={render_psnr:.2f}"
            )
        sat_bonly = finite_mean(records, f"ns{ns}_sat_bonly_height_ab_mae")
        xy_bonly = finite_mean(records, f"ns{ns}_xy_bonly_height_ab_mae")
        print(f"  B-only geometry delta (XY - satellite): {xy_bonly - sat_bonly:+.4f} m")
    print(f"\nrecords: {output}")


if __name__ == "__main__":
    main()
