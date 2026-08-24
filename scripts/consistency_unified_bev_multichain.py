#!/usr/bin/env python3
"""C2 multi-chain convergence: do different observations recover the SAME
world latent?

For one tile, two disjoint observation chains are formed by splitting the
8-frame dense source pool into halves: chain A reads frames[0:Ns], chain B
reads frames[4:4+Ns].  Each chain independently runs the completion operator.

Question: does Ẑ_A converge to Ẑ_B?  Measured at three layers, with an
anti-collapse baseline (same computation across different locations -- if
the model output a constant prior everywhere, same-location pairs would look
"consistent" for the wrong reason; the ratio must be << 1).

Families: v2sat (satellite prior) and coordinate_only (XY prior, no satellite
content).  The satellite's consistency role is isolated on cells observed by
chain B only: there, Ẑ_A is completion (prior + context) while Ẑ_B carries an
actual measurement.  If D_sat < D_xy on those cells, the satellite -- not a
positional prior -- pulls independent chains toward one world state.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
from world3d.unified_bev.data import UnifiedBEVDataset
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundDenseBEVEncoder,
    GroundBEVEncoder,
    HeightMapSatellitePrior,
    LatentCompletion,
)


def latent_l1(a, b):
    return float((a - b).abs().mean())


def masked_l1(a, b, m):
    return float((a - b).abs()[:, :, m].mean()) if m.any() else float("nan")


def psnr(a, b):
    mse = F.mse_loss(a, b)
    return float(10 * torch.log10(1.0 / mse.clamp_min(1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--stage_b_sat", required=True)
    ap.add_argument("--stage_b_xy", required=True)
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl")
    ap.add_argument("--drive", default="2013_05_28_drive_0003_sync", help="pass 'none' for all drives (in-sample)")
    ap.add_argument("--eval_samples", type=int, default=32)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--ns_list", default="1,2")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--records_out", default="runs/consistency_multichain_0003.jsonl")
    ap.add_argument("--m3d_cache", default=None, help="Metric3D cache for this eval split (dense lift)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    ds = UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root,
        drive=None if args.drive == 'none' else args.drive,
        min_target_spacing_m=5.0, max_samples=args.eval_samples,
        dense_source_count=args.dense_sources, sparse_source_count=args.dense_sources,
        image_size=(160, 96), max_points_per_view=4096,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    vpf = ds.views_per_frame

    a_ckpt = torch.load(args.stage_a, map_location=device, weights_only=False)
    m3d_blobs = None
    if args.m3d_cache:
        m3d_blobs = [torch.load(Path(args.m3d_cache) / f"{i:06d}.pt", map_location='cpu', weights_only=False)
                     for i in range(len(ds))]
    ground = (GroundDenseBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size) if args.m3d_cache
              else GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size)).to(device)
    ground.load_state_dict(a_ckpt["ground"]); ground.eval()
    decoder = ColumnFieldDecoder(
        hidden=a_ckpt.get("config", {}).get("hidden", 256),
        samples=a_ckpt.get("config", {}).get("ray_samples", 48),
    ).to(device)
    decoder.load_state_dict(a_ckpt["decoder"]); decoder.eval()
    bs = torch.load(args.stage_b_sat, map_location=device, weights_only=False)
    bx = torch.load(args.stage_b_xy, map_location=device, weights_only=False)
    sat = HeightMapSatellitePrior(bev_height=ds.bev_size, bev_width=ds.bev_size,
                                  **bs.get("config", {}).get("sat_encoder_kwargs", {})).to(device)
    sat.load_state_dict(bs["satellite_encoder"]); sat.eval()
    comp_sat = LatentCompletion(mode="residual", bev_height=ds.bev_size,
                                bev_width=ds.bev_size, tile_size_m=ds.tile_size_m).to(device)
    comp_sat.load_state_dict(bs["completion"]); comp_sat.eval()
    comp_xy = LatentCompletion(mode="coordinate_only", bev_height=ds.bev_size,
                               bev_width=ds.bev_size, tile_size_m=ds.tile_size_m).to(device)
    comp_xy.load_state_dict(bx["completion"]); comp_xy.eval()

    def chain_latents(batch, frames, bi=0):
        """Completion output for one observation chain (given source frames)."""
        sl = slice(frames.start * vpf, frames.stop * vpf)
        with torch.no_grad():
            if m3d_blobs is not None:
                blob = m3d_blobs[bi]
                dd = blob["depth"].unsqueeze(0).to(device).float()
                dc = blob["conf"].unsqueeze(0).to(device).float()
                z_gnd, cov = ground(
                    batch["source_rgb"][:, sl], batch["source_K"][:, sl],
                    dd[:, sl], dc[:, sl], batch["source_T_world_cam"][:, sl],
                    batch["origin_xy"], ds.bev_resolution_m,
                )
            else:
                z_gnd, cov = ground(
                    batch["source_rgb"][:, sl], batch["source_points_world"][:, sl],
                    batch["source_points_uv"][:, sl], batch["source_points_valid"][:, sl],
                    batch["origin_xy"], ds.bev_resolution_m,
                )
            prior, _, _ = sat(batch["satellite"], z_gnd, ds.tile_size_m, 0.196)
            n = frames.stop - frames.start
            z_sat = comp_sat(prior, z_gnd, cov, n, args.dense_sources)
            z_xy = comp_xy(torch.zeros_like(prior), z_gnd, cov, n, args.dense_sources)
        return z_sat, z_xy, cov[0, 0] > 0.5

    records = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if m3d_blobs is not None:
                blob = m3d_blobs[bi]
                z_star, _ = ground(
                    batch["source_rgb"], batch["source_K"],
                    blob["depth"].unsqueeze(0).to(device).float(), blob["conf"].unsqueeze(0).to(device).float(),
                    batch["source_T_world_cam"], batch["origin_xy"], ds.bev_resolution_m,
                )
            else:
                z_star, _ = ground(
                    batch["source_rgb"], batch["source_points_world"], batch["source_points_uv"],
                    batch["source_points_valid"], batch["origin_xy"], ds.bev_resolution_m,
                )
            row = {"target_fid": int(batch["meta"]["target_fid"][0])}
            for ns in [int(x) for x in args.ns_list.split(",")]:
                zA_sat, zA_xy, covA = chain_latents(batch, slice(0, ns), bi)
                zB_sat, zB_xy, covB = chain_latents(batch, slice(4, 4 + ns), bi)
                neither = ~covA & ~covB
                b_only = covB & ~covA
                a_only = covA & ~covB

                def render(z):
                    return decoder.render(
                        z, batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
                        tile_size_m=ds.tile_size_m, image_size=ds.image_size,
                    )[0]

                for fam, (zA, zB) in (("sat", (zA_sat, zB_sat)), ("xy", (zA_xy, zB_xy))):
                    row[f"ns{ns}_{fam}_l1"] = latent_l1(zA, zB)
                    row[f"ns{ns}_{fam}_l1_bonly"] = masked_l1(zA, zB, b_only)
                    row[f"ns{ns}_{fam}_l1_aonly"] = masked_l1(zA, zB, a_only)
                    row[f"ns{ns}_{fam}_l1_neither"] = masked_l1(zA, zB, neither)
                    row[f"ns{ns}_{fam}_acc"] = 0.5 * (latent_l1(zA, z_star) + latent_l1(zB, z_star))
                    row[f"ns{ns}_{fam}_render_psnr"] = psnr(render(zA), render(zB))
                # accuracy anchor for star context
                row[f"ns{ns}_star_selfdist"] = latent_l1(zA_sat, z_star)
            records.append(row)

    # anti-collapse baseline: chain-A of tile i vs chain-B of a different tile j
    rng = random.Random(0)
    n = len(records)
    partners = rng.sample(range(n), n)
    partners = [(i + 1 + (i == 0)) % n for i in range(n)]  # deterministic shift pairing
    # recompute cross-location latents cheaply: reuse stored per-tile chain-A latents is not
    # possible after the loop, so approximate the baseline with render-free latent L1 between
    # the stored per-chain latents -- we saved only scalars, so do a second pass.
    # (kept simple: second pass over the loader storing chain latents per tile)
    cache = []
    with torch.no_grad():
        for bi2, batch in enumerate(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)):
            bi = bi2
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            for ns in [int(x) for x in args.ns_list.split(",")]:
                zA_sat, zA_xy, _ = chain_latents(batch, slice(0, ns), bi)
                cache.append((bi, ns, zA_sat.cpu(), zA_xy.cpu()))
    lat_by_key = {(bi, ns): (zs, zx) for bi, ns, zs, zx in cache}
    for i, row in enumerate(records):
        j = (i + 1) % len(records)
        for ns in [int(x) for x in args.ns_list.split(",")]:
            zsA, zxA = lat_by_key[(i, ns)]
            zsB, zxB = lat_by_key[(j, ns)]
            row[f"ns{ns}_sat_l1_diffloc"] = latent_l1(zsA, zsB)
            row[f"ns{ns}_xy_l1_diffloc"] = latent_l1(zxA, zxB)

    out = Path(args.records_out)
    out.write_text("".join(json.dumps(r, allow_nan=True) + "\n" for r in records))

    import math
    def mean(rows, k):
        v = [r[k] for r in rows if k in r and math.isfinite(r[k])]
        return sum(v) / max(1, len(v))

    print(f"=== multi-chain convergence on {args.drive} (chains: frames[0:Ns] vs frames[4:4+Ns]) ===")
    for ns in [int(x) for x in args.ns_list.split(",")]:
        print(f"\n-- Ns={ns}")
        print(f"{'family':>6s} {'D_same':>8s} {'D_diffloc':>9s} {'ratio':>6s} "
              f"{'D_bonly':>8s} {'D_neither':>9s} {'acc(Z*)':>8s} {'rendPSNR':>8s}")
        for fam in ("sat", "xy"):
            d_same = mean(records, f"ns{ns}_{fam}_l1")
            d_diff = mean(records, f"ns{ns}_{fam}_l1_diffloc")
            print(f"{fam:>6s} {d_same:>8.4f} {d_diff:>9.4f} {d_same/d_diff:>6.3f} "
                  f"{mean(records, f'ns{ns}_{fam}_l1_bonly'):>8.4f} "
                  f"{mean(records, f'ns{ns}_{fam}_l1_neither'):>9.4f} "
                  f"{mean(records, f'ns{ns}_{fam}_acc'):>8.4f} "
                  f"{mean(records, f'ns{ns}_{fam}_render_psnr'):>8.2f}")
        # satellite consistency-recovery on the B-only region
        d_b_sat = mean(records, f"ns{ns}_sat_l1_bonly")
        d_b_xy = mean(records, f"ns{ns}_xy_l1_bonly")
        print(f"  Δconsistency on B-only cells (xy − sat): {d_b_xy - d_b_sat:+.4f} "
              f"({'sat pulls chains closer' if d_b_xy > d_b_sat else 'no sat benefit'})")
    print(f"\nrecords: {out}")


if __name__ == "__main__":
    main()
