#!/usr/bin/env python3
"""Chunk-mode claim evaluation: spatial hole completion.

For each kept-chunk count K (the inter-chunk sparsity axis) this evaluates
the same frozen Stage-A interface on three branches — dense reference,
sparse (kept chunks only), and completed (satellite prior + completion) —
and reports headline metrics on the hole partition: BEV cells supported by
the dense reference but not by any kept chunk, and further than ``guard_m``
(euclidean, BEV dilation) from the kept support.  Frozen queries are the
per-chunk core frames; per-query metrics are grouped by the query chunk so a
missing chunk's queries are never diluted by kept-chunk queries.

The intra-chunk axis (fewer frames inside a fixed chunk) is a separate
diagnostic and intentionally not evaluated here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_unified_bev_probe import (  # noqa: E402
    _open_image,
    depth_metrics,
    mean_finite,
    per_item_lowfreq_psnr,
    per_item_masked_l1,
    per_item_masked_rmse,
    per_item_psnr,
    perturb_satellite,
    road_frame_shift,
    json_safe_record,
)
from world3d.unified_bev.checkpoints import (  # noqa: E402
    validate_stage_a_checkpoint,
    validate_stage_a_dataset,
    validate_stage_b_checkpoint,
)
from world3d.unified_bev.data import (  # noqa: E402
    SAT_M_PER_PX,
    ChunkedUnifiedBEVDataset,
    attach_chunk_geometry,
    chunk_subset_qa,
)
from world3d.unified_bev.geometry import (  # noqa: E402
    geometry_supervision_support,
    observation_partition,
    relative_height_map,
)
from world3d.unified_bev.models import (  # noqa: E402
    ColumnFieldDecoder,
    GroundDenseBEVEncoder,
    HeightMapSatellitePrior,
    LatentCompletion,
    SatelliteBEVEncoder,
    SatelliteViTEncoder,
)
from world3d.unified_bev.readouts import BEVHeightDecoder, freeze_module  # noqa: E402


def kept_positions(n_chunks: int, kept_count: int) -> List[int]:
    """Kept chunks: head chunk plus the trailing kept block (hole = middle)."""
    return [0] + list(range(n_chunks - kept_count + 1, n_chunks))


def dilate(mask: torch.Tensor, radius_cells: int) -> torch.Tensor:
    if radius_cells <= 0:
        return mask.clone()
    return F.max_pool2d(
        mask.float(), kernel_size=2 * radius_cells + 1, stride=1, padding=radius_cells,
    ) > 0.5


def hole_partitions(ref_mask, sparse_mask, h_valid, guard_m, resolution_m):
    """Hole and guard-eroded hole-core masks (all (B,1,H,W) bool)."""
    dense_height = geometry_supervision_support(ref_mask, h_valid)
    observed, hole = observation_partition(sparse_mask, dense_height)
    radius = max(1, int(round(guard_m / resolution_m)))
    core = hole & ~dilate(sparse_mask, radius)
    return observed, hole, core


def anchor_road_headings(ds) -> list:
    """Local road-heading variant for chunk windows (anchor records)."""
    import numpy as np
    pos = np.asarray([
        [rec.T_world_imu[0, 3], rec.T_world_imu[1, 3]]
        for rec in (s[0] for s in ds.samples)
    ])
    headings = np.zeros((len(pos), 2), dtype=np.float64)
    for i in range(len(pos)):
        a, b = max(0, i - 1), min(len(pos) - 1, i + 1)
        d = pos[b] - pos[a]
        norm = float(np.hypot(d[0], d[1]))
        headings[i] = d / norm if norm > 1e-6 else (0.0, 1.0)
    return headings


def load_donor_anchor_satellites(ds, indexes, device):
    """Random-tile control donors: anchor satellites of other windows."""
    import numpy as np
    offset = max(1, len(ds) // 2)
    images, meta = [], []
    for index in indexes:
        donor = ds.samples[(index + offset) % len(ds)][0]
        with _open_image(donor.sat_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32).copy()
        images.append(torch.from_numpy(arr).permute(2, 0, 1) / 255.0)
        meta.append({"drive": donor.drive, "anchor_fid": donor.fid})
    return torch.stack(images).to(device), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--stage_b", default=None,
                    help="optional completion checkpoint; without it only dense/sparse are compared")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl")
    ap.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--geometry_cache", required=True, help="chunk cache v7 for this eval split")
    ap.add_argument("--kept_choices", default="1,2,3",
                    help="kept-chunk counts K to evaluate (dense identity K=Nc is always included)")
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--eval_ssim_lpips", action="store_true")
    ap.add_argument("--sat_random_tile", action="store_true")
    ap.add_argument("--sat_shift_cross_m", type=float, default=0.0)
    ap.add_argument("--sat_shift_road_m", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--records_out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    ds = attach_chunk_geometry(ChunkedUnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, drive=args.drive,
        max_samples=args.max_samples,
    ), args.geometry_cache)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    headings = anchor_road_headings(ds)
    n_chunks = ds.chunks_per_window
    kept_set = tuple(int(k) for k in args.kept_choices.split(",") if k.strip())
    if min(kept_set) < 1 or max(kept_set) >= n_chunks:
        raise ValueError(f"kept choices must be within [1,{n_chunks - 1}]")
    lpips_net = None
    if args.eval_ssim_lpips:
        import lpips
        lpips_net = lpips.LPIPS(net="alex").to(device)
        lpips_net.eval()

    a = torch.load(args.stage_a, map_location=device, weights_only=False)
    stage_a_fingerprint = validate_stage_a_checkpoint(a)
    validate_stage_a_dataset(a, ds, dense_geometry_attached=True)
    ground = GroundDenseBEVEncoder(**{k: v for k, v in a["ground_config"].items() if k != "family"}).to(device)
    decoder = ColumnFieldDecoder(**a["renderer_config"]).to(device)
    geometry_decoder = BEVHeightDecoder(**a["geometry_decoder_config"]).to(device)
    ground.load_state_dict(a["ground"]); decoder.load_state_dict(a["decoder"])
    geometry_decoder.load_state_dict(a["geometry_decoder"])

    b = None
    sat_encoder = None
    completion = None
    if args.stage_b is not None:
        b = torch.load(args.stage_b, map_location=device, weights_only=False)
        validate_stage_b_checkpoint(b, stage_a_fingerprint)
        cfg = b.get("config", {})
        if cfg.get("fusion") == "coordinate_only":
            raise ValueError("pass the satellite-fusion Stage-B checkpoint here")
        completion = LatentCompletion(
            mode=cfg.get("fusion", "residual"),
            bev_height=ds.bev_size, bev_width=ds.bev_size, tile_size_m=ds.tile_size_m,
        ).to(device)
        completion.load_state_dict(b["completion"])
        family = cfg.get("sat_encoder", "vit")
        kwargs = cfg.get("sat_encoder_kwargs", {})
        if family == "heightmap":
            sat_encoder = HeightMapSatellitePrior(
                bev_height=ds.bev_size, bev_width=ds.bev_size, **kwargs).to(device)
        elif family == "vit":
            sat_encoder = SatelliteViTEncoder(
                bev_height=ds.bev_size, bev_width=ds.bev_size, **kwargs).to(device)
        else:
            sat_encoder = SatelliteBEVEncoder(
                bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
        sat_encoder.load_state_dict(b["satellite_encoder"])
    modules = [ground, decoder, geometry_decoder]
    if sat_encoder is not None:
        modules.append(sat_encoder)
    if completion is not None:
        modules.append(completion)
    for m in modules:
        freeze_module(m)

    records: List[Dict] = []
    with torch.no_grad():
        for tile_idx, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            # default_collate transposes the chunk table; read the original
            table = ds.chunk_table(tile_idx)
            geometry_blob = torch.load(
                Path(args.geometry_cache) / f"{tile_idx:06d}.pt",
                map_location="cpu", weights_only=False,
            )
            chunk_qa_flat = {
                f"chunk{p}_{key}": value
                for p in range(n_chunks)
                for key, value in chunk_subset_qa(geometry_blob, p).items()
            }
            nq_per_chunk = ds.target_views // n_chunks  # front views per query frame
            queries_per_chunk = list(range(n_chunks))  # query chunk positions

            def chunk_lift(positions: List[int]):
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

            z_star, ref_mask = chunk_lift(list(range(n_chunks)))
            h_ref, h_valid, _ = relative_height_map(
                batch["source_points_world"], batch["source_points_valid"],
                batch["origin_xy"], ds.bev_resolution_m, ds.bev_size, ds.bev_size,
            )
            heights = {"dense": geometry_decoder(z_star)}

            # flattened query tensors (B=1)
            q_rgb = batch["target_rgb"][0]            # (Nq*views? no: (Nq_total,3,H,W))
            q_K = batch["target_K"][0]
            q_T = batch["target_T_world_cam"][0]
            q_depth = batch["target_depth"][0]
            q_mask = batch["target_depth_mask"][0]
            origin = batch["origin_xy"]
            n_views_total = q_rgb.shape[0]
            assert n_views_total == n_chunks * nq_per_chunk

            def render_branch(z):
                z_rep = z.expand(n_views_total, -1, -1, -1)
                rgb, depth, _ = decoder.render(
                    z_rep, q_K, q_T, origin,
                    tile_size_m=ds.tile_size_m, image_size=ds.image_size,
                )
                return rgb, depth

            dense_rgb, dense_depth_pred = render_branch(z_star)

            sat_input = batch["satellite"]
            donor_meta = None
            if sat_encoder is not None:
                if args.sat_random_tile:
                    sat_input, donor_meta = load_donor_anchor_satellites(
                        ds, range(tile_idx, tile_idx + 1), device)
                shift_x = shift_y = 0.0
                if args.sat_shift_road_m or args.sat_shift_cross_m:
                    dx, dy = road_frame_shift(
                        headings[tile_idx], args.sat_shift_road_m, args.sat_shift_cross_m)
                    shift_x, shift_y = shift_x + dx, shift_y + dy
                sat_input = perturb_satellite(
                    sat_input, meters_per_pixel=SAT_M_PER_PX,
                    shift_x_m=shift_x, shift_y_m=shift_y,
                )

            for K in kept_set:
                positions = kept_positions(n_chunks, K)
                missing = [p for p in range(n_chunks) if p not in positions]
                z_sparse, sparse_mask = chunk_lift(positions)
                z_full = None
                if completion is not None:
                    if sat_encoder is None:
                        z_sat = torch.zeros_like(z_sparse)
                    elif isinstance(sat_encoder, HeightMapSatellitePrior):
                        z_sat, _ = sat_encoder(sat_input, z_sparse, ds.tile_size_m, SAT_M_PER_PX)
                    else:
                        z_sat = sat_encoder(sat_input, ds.tile_size_m, SAT_M_PER_PX)
                    z_full = completion(
                        z_sat, z_sparse, sparse_mask, K, n_chunks,
                    ).latent
                observed, hole, core = hole_partitions(
                    ref_mask, sparse_mask, h_valid, ds.guard_m, ds.bev_resolution_m,
                )
                branches = {"sparse": z_sparse}
                if z_full is not None:
                    branches["full"] = z_full
                for name, z in branches.items():
                    heights[name] = geometry_decoder(z)

                base = {
                    "drive": batch["meta"]["drive"][0],
                    "anchor_fid": int(batch["meta"]["target_fid"][0]),
                    "kept_chunks": K,
                    "kept_positions": positions,
                    "missing_positions": missing,
                    "hole_pattern": "middle_block",
                    "stage_a_fingerprint": stage_a_fingerprint,
                    "satellite_control": {
                        "random_tile": args.sat_random_tile,
                        "shift_road_m": args.sat_shift_road_m,
                        "shift_cross_m": args.sat_shift_cross_m,
                    },
                    "observed_fraction": float(observed.float().mean()),
                    "hole_fraction": float(hole.float().mean()),
                    "hole_core_fraction": float(core.float().mean()),
                    "chunk_arc_m": [float(c["arc_end"] - c["arc_start"]) for c in table],
                    **chunk_qa_flat,
                }

                # ---- hole-partitioned frozen-height metrics ----------------
                for name, h in heights.items():
                    for region, mask in (
                        ("all", h_valid), ("observed", observed & h_valid),
                        ("hole", hole), ("hole_core", core),
                    ):
                        mae = per_item_masked_l1(h, h_ref, mask)
                        rmse = per_item_masked_rmse(h, h_ref, mask)
                        base[f"{name}_height_{region}_mae"] = mae[0]
                        base[f"{name}_height_{region}_rmse"] = rmse[0]
                        base[f"{name}_height_{region}_cells"] = int(mask.sum())

                # ---- per-query-chunk render metrics ------------------------
                rgb_branches = {"dense": (dense_rgb, dense_depth_pred)}
                rgb_branches["sparse"] = render_branch(z_sparse)
                if z_full is not None:
                    rgb_branches["full"] = render_branch(z_full)
                for qpos in queries_per_chunk:
                    row = dict(base)
                    row["query_chunk"] = qpos
                    row["query_fid"] = int(table[qpos]["core_fid"])
                    row["query_missing"] = qpos in missing
                    vs = slice(qpos * nq_per_chunk, (qpos + 1) * nq_per_chunk)
                    for name, (rgb, depth) in rgb_branches.items():
                        row[f"{name}_psnr"] = mean_finite(per_item_psnr(rgb[vs], q_rgb[vs]))
                        row[f"{name}_rgb_lowfreq_psnr"] = mean_finite(
                            per_item_lowfreq_psnr(rgb[vs], q_rgb[vs]))
                        depth_rows = depth_metrics(depth[vs], q_depth[vs], q_mask[vs])
                        row[f"{name}_absrel"] = mean_finite([d["absrel"] for d in depth_rows])
                        row[f"{name}_rmse"] = mean_finite([d["rmse"] for d in depth_rows])
                        row[f"{name}_delta1"] = mean_finite([d["delta1"] for d in depth_rows])
                    records.append(row)

    numeric = sorted({k for r in records for k, v in r.items()
                      if isinstance(v, (int, float)) and k != "anchor_fid"})
    summary = {k: round(mean_finite([float(r[k]) for r in records if k in r]), 6)
               for k in numeric}
    by_k: Dict[int, Dict[str, float]] = {}
    for K in kept_set:
        rows = [r for r in records if r["kept_chunks"] == K and r["query_missing"]]
        by_k[K] = {
            key: round(mean_finite([float(r[key]) for r in rows if key in r]), 6)
            for key in (
                "hole_core_fraction", "sparse_height_hole_core_mae",
                "full_height_hole_core_mae", "dense_height_hole_core_mae",
                "sparse_psnr", "full_psnr", "dense_psnr",
            )
            if any(key in r for r in rows)
        }
    print(json.dumps(json_safe_record({"by_kept_chunks": by_k}), indent=2))
    print(json.dumps(json_safe_record({"overall": summary}), indent=2))
    if args.records_out:
        out = Path(args.records_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(
            json.dumps(json_safe_record(r), allow_nan=False) + "\n" for r in records))
        print(f"records: {len(records)} rows -> {out}")


if __name__ == "__main__":
    main()
