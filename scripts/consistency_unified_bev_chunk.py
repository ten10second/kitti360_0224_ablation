#!/usr/bin/env python3
"""C2 frozen-query consistency at chunk granularity.

Two disjoint chunk subsets — chain A = {c0, c2} vs chain B = {c1, c3} —
each run the full geometry backend and completion path independently.  The
frozen Stage-A height decoder and target-pose renderer then read both world
states; consistency is measured on shared / a-only / b-only / neither
partitions of the dense reference support.  Raw latent distances remain
``*_diag`` fields only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from consistency_unified_bev_multichain import (  # noqa: E402
    finite_mean,
    json_safe,
    masked_l1_diag,
    masked_mae,
    masked_rmse,
    psnr,
)
from eval_unified_bev_chunk_probe import kept_positions  # noqa: E402
from world3d.unified_bev.checkpoints import (  # noqa: E402
    validate_stage_a_checkpoint,
    validate_stage_a_dataset,
    validate_stage_b_checkpoint,
)
from world3d.unified_bev.data import (  # noqa: E402
    SAT_M_PER_PX,
    ChunkedUnifiedBEVDataset,
    attach_chunk_geometry,
)
from world3d.unified_bev.geometry import (  # noqa: E402
    geometry_supervision_support,
    relative_height_map,
)
from world3d.unified_bev.losses import low_frequency  # noqa: E402
from world3d.unified_bev.models import (  # noqa: E402
    ColumnFieldDecoder,
    GroundDenseBEVEncoder,
    HeightMapSatellitePrior,
    LatentCompletion,
    SatelliteBEVEncoder,
    SatelliteViTEncoder,
)
from world3d.unified_bev.readouts import BEVHeightDecoder, freeze_module  # noqa: E402


def build_satellite_encoder(checkpoint: dict, bev_size: int):
    from consistency_unified_bev_multichain import build_satellite_encoder as _build
    return _build(checkpoint, bev_size)


def _recursive_json_safe(value):
    import math
    if isinstance(value, dict):
        return {k: _recursive_json_safe(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_a", required=True)
    parser.add_argument("--stage_b_sat", required=True)
    parser.add_argument("--stage_b_xy", required=True)
    parser.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl")
    parser.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    parser.add_argument("--eval_samples", type=int, default=32)
    parser.add_argument("--geometry_cache", required=True)
    parser.add_argument("--chain_a_chunks", default="0,2")
    parser.add_argument("--chain_b_chunks", default="1,3")
    parser.add_argument("--records_out", default="runs/consistency_chunk_0003.jsonl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ds = attach_chunk_geometry(ChunkedUnifiedBEVDataset(
        args.manifest, lidar_root="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw",
        drive=args.drive, max_samples=args.eval_samples,
    ), args.geometry_cache)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    n_chunks = ds.chunks_per_window
    chain_a = [int(x) for x in args.chain_a_chunks.split(",")]
    chain_b = [int(x) for x in args.chain_b_chunks.split(",")]
    if set(chain_a) & set(chain_b):
        raise ValueError(f"chains overlap: {chain_a} vs {chain_b}")
    if max(chain_a + chain_b) >= n_chunks:
        raise ValueError("chain chunk position exceeds window")

    stage_a = torch.load(args.stage_a, map_location=device, weights_only=False)
    stage_a_fingerprint = validate_stage_a_checkpoint(stage_a)
    validate_stage_a_dataset(stage_a, ds, dense_geometry_attached=True)
    ground = GroundDenseBEVEncoder(**{
        k: v for k, v in stage_a["ground_config"].items() if k != "family"
    }).to(device)
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
    satellite_encoder = build_satellite_encoder(stage_b_sat, ds.bev_size).to(device)
    satellite_encoder.load_state_dict(stage_b_sat["satellite_encoder"])
    completion_sat = LatentCompletion(
        mode="residual", bev_height=ds.bev_size, bev_width=ds.bev_size,
        tile_size_m=ds.tile_size_m,
    ).to(device)
    completion_xy = LatentCompletion(
        mode="coordinate_only", bev_height=ds.bev_size, bev_width=ds.bev_size,
        tile_size_m=ds.tile_size_m,
    ).to(device)
    completion_sat.load_state_dict(stage_b_sat["completion"])
    completion_xy.load_state_dict(stage_b_xy["completion"])
    for module in (ground, renderer, geometry_decoder, satellite_encoder,
                   completion_sat, completion_xy):
        freeze_module(module)

    def chunk_lift(batch, positions):
        fpc, vpf = ds.frames_per_chunk, ds.views_per_frame
        rows = torch.tensor(
            [r for p in positions for r in range(p * fpc * vpf, (p + 1) * fpc * vpf)],
            device=device,
        )
        depth = torch.cat([batch[f"dense_depth_c{p}"] for p in positions], dim=1)
        conf = torch.cat([batch[f"dense_conf_c{p}"] for p in positions], dim=1)
        return ground(
            batch["source_rgb"][:, rows], batch["source_K"][:, rows], depth, conf,
            batch["source_T_world_cam"][:, rows], batch["origin_xy"],
            ds.bev_resolution_m,
        )

    records = []
    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            z_star, dense_support = chunk_lift(batch, list(range(n_chunks)))
            h_ref, h_valid, _ = relative_height_map(
                batch["source_points_world"], batch["source_points_valid"],
                batch["origin_xy"], ds.bev_resolution_m, ds.bev_size, ds.bev_size,
            )
            reference = geometry_supervision_support(dense_support, h_valid)
            h_star = geometry_decoder(z_star)

            chains = {}
            for name, positions in (("a", chain_a), ("b", chain_b)):
                z_ground, support = chunk_lift(batch, positions)
                if isinstance(satellite_encoder, HeightMapSatellitePrior):
                    prior, _ = satellite_encoder(
                        batch["satellite"], z_ground, ds.tile_size_m, SAT_M_PER_PX)
                else:
                    prior = satellite_encoder(batch["satellite"], ds.tile_size_m, SAT_M_PER_PX)
                k = len(positions)
                chains[name] = {
                    "support": support.bool(),
                    "sat": completion_sat(prior, z_ground, support, k, n_chunks).latent,
                    "xy": completion_xy(
                        torch.zeros_like(prior), z_ground, support, k, n_chunks).latent,
                }
            regions = {
                "shared": reference & chains["a"]["support"] & chains["b"]["support"],
                "aonly": reference & chains["a"]["support"] & ~chains["b"]["support"],
                "bonly": reference & chains["b"]["support"] & ~chains["a"]["support"],
                "neither": reference & ~chains["a"]["support"] & ~chains["b"]["support"],
            }
            row = {
                "drive": batch["meta"]["drive"][0],
                "anchor_fid": int(batch["meta"]["target_fid"][0]),
                "chain_a_chunks": chain_a,
                "chain_b_chunks": chain_b,
                "stage_a_fingerprint": stage_a_fingerprint,
            }
            for region_name, mask in regions.items():
                row[f"{region_name}_cells"] = int(mask.sum())

            q_rgb = batch["target_rgb"][0]
            q_K = batch["target_K"][0]
            q_T = batch["target_T_world_cam"][0]
            for family in ("sat", "xy"):
                latent_a = chains["a"][family]
                latent_b = chains["b"][family]
                height_a = geometry_decoder(latent_a)
                height_b = geometry_decoder(latent_b)
                row[f"{family}_height_ab_mae"] = masked_mae(height_a, height_b, reference)
                row[f"{family}_height_ab_rmse"] = masked_rmse(height_a, height_b, reference)
                for region_name, mask in regions.items():
                    row[f"{family}_{region_name}_height_ab_mae"] = masked_mae(
                        height_a, height_b, mask)
                    row[f"{family}_{region_name}_latent_ab_l1_diag"] = masked_l1_diag(
                        latent_a, latent_b, mask)
                rgb_a, depth_a, _ = renderer.render(
                    latent_a.expand(q_rgb.shape[0], -1, -1, -1), q_K, q_T,
                    batch["origin_xy"], tile_size_m=ds.tile_size_m,
                    image_size=ds.image_size,
                )
                rgb_b, depth_b, _ = renderer.render(
                    latent_b.expand(q_rgb.shape[0], -1, -1, -1), q_K, q_T,
                    batch["origin_xy"], tile_size_m=ds.tile_size_m,
                    image_size=ds.image_size,
                )
                row[f"{family}_depth_ab_l1"] = masked_mae(
                    depth_a, depth_b, batch["target_depth_mask"][0])
                row[f"{family}_render_lowfreq_psnr"] = psnr(
                    low_frequency(rgb_a), low_frequency(rgb_b))
                row[f"{family}_height_a_star_mae"] = masked_mae(height_a, h_star, reference)
                row[f"{family}_height_b_star_mae"] = masked_mae(height_b, h_star, reference)
            records.append(row)

    output = Path(args.records_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(
        json.dumps(json_safe(r), allow_nan=False) + "\n" for r in records))
    summary = {
        "paired_tiles": len(records),
        "chains": {"a": chain_a, "b": chain_b},
        "metrics": {
            f"{family}_{metric}": finite_mean(records, f"{family}_{metric}")
            for family in ("sat", "xy")
            for metric in (
                "height_ab_mae", "height_ab_rmse", "depth_ab_l1",
                "render_lowfreq_psnr",
                "shared_height_ab_mae", "aonly_height_ab_mae",
                "bonly_height_ab_mae", "neither_height_ab_mae",
            )
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(_recursive_json_safe(summary), indent=2, allow_nan=False))
    print(json.dumps(_recursive_json_safe(summary), indent=2))
    print(f"records: {output}")


if __name__ == "__main__":
    main()
