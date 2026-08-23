#!/usr/bin/env python3
"""Train Stage B latent recovery with the Stage-A decoder frozen."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.data import UnifiedBEVDataset
from world3d.unified_bev.data import load_cached_unified_bev
from world3d.unified_bev.geometry import bilinear_splat, height_statistics
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    LatentCompletion,
    HeightMapSatellitePrior,
    SatelliteBEVEncoder,
    SatelliteViTEncoder,
    nadir_distance,
    satellite_bev_crop,
)


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def parse_source_choices(value: str | None, fixed: int, dense: int) -> tuple[int, ...]:
    choices = (fixed,) if value is None else tuple(int(x) for x in value.split(",") if x.strip())
    if not choices or any(x < 1 or x > dense for x in choices):
        raise ValueError(f"sparse source choices must be within [1,{dense}], got {choices}")
    return tuple(dict.fromkeys(choices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--out", default="runs/unified_bev_stage_b_probe")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max_samples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--sparse_sources", type=int, default=2)
    ap.add_argument("--sparse_source_choices", default=None,
                    help="comma-separated Stage-B training choices, e.g. 1,2,4,8; default uses --sparse_sources")
    ap.add_argument("--fusion", choices=["residual", "ground_only", "satellite_only", "coordinate_only"], default="residual")
    ap.add_argument("--sat_encoder", choices=["vit", "cnn", "heightmap"], default="vit",
                    help="satellite encoder family; cnn is the legacy 3-layer convolution")
    ap.add_argument("--sat_dim", type=int, default=256)
    ap.add_argument("--sat_depth", type=int, default=4)
    ap.add_argument("--sat_heads", type=int, default=4)
    ap.add_argument("--sat_patch", type=int, default=8)
    ap.add_argument("--height_weight", type=float, default=1.0,
                    help="heightmap mode: weight of the dense-LiDAR h_mean (DEM) supervision")
    ap.add_argument("--image_width", type=int, default=160)
    ap.add_argument("--image_height", type=int, default=96)
    ap.add_argument("--max_points", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_target_spacing_m", type=float, default=5.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--ray_samples", type=int, default=24)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--latent_weight", type=float, default=1.0)
    ap.add_argument("--render_weight", type=float, default=0.1)
    ap.add_argument("--depth_weight", type=float, default=0.05)
    ap.add_argument("--nadir_weight", type=float, default=0.0,
                    help="supervision-level satellite fusion: masked nadir-view consistency")
    ap.add_argument("--nadir_top_m", type=float, default=48.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--cache", default=None, help="prebuilt sample cache; serving from RAM, workers forced to 0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_choices = parse_source_choices(args.sparse_source_choices, args.sparse_sources, args.dense_sources)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.stage_a, map_location=device, weights_only=False)
    ds = (load_cached_unified_bev(args.cache) if args.cache else UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, dense_source_count=args.dense_sources,
        sparse_source_count=max(source_choices), image_size=(args.image_width, args.image_height),
        max_points_per_view=args.max_points, max_samples=args.max_samples, drive=args.drive,
        min_target_spacing_m=args.min_target_spacing_m,
    ))
    if args.cache:
        args.num_workers = 0  # RAM serving; forked workers would copy the cache
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        drop_last=True, persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    ground = GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    decoder = ColumnFieldDecoder(hidden=args.hidden, samples=args.ray_samples).to(device)
    ground.load_state_dict(ckpt["ground"])
    decoder.load_state_dict(ckpt["decoder"])
    ground.eval(); decoder.eval()
    for module in (ground, decoder):
        for p in module.parameters():
            p.requires_grad_(False)
    sat_encoder_kwargs = {"dim": args.sat_dim, "depth": args.sat_depth,
                          "heads": args.sat_heads, "patch": args.sat_patch,
                          "tile_size_m": ds.tile_size_m}
    if args.sat_encoder == "heightmap":
        sat_encoder = HeightMapSatellitePrior(
            bev_height=ds.bev_size, bev_width=ds.bev_size, **sat_encoder_kwargs,
        ).to(device)
    elif args.sat_encoder == "vit":
        sat_encoder = SatelliteViTEncoder(
            bev_height=ds.bev_size, bev_width=ds.bev_size, **sat_encoder_kwargs,
        ).to(device)
    else:
        sat_encoder = SatelliteBEVEncoder(
            bev_height=ds.bev_size, bev_width=ds.bev_size,
        ).to(device)
    completion = LatentCompletion(
        mode=args.fusion, bev_height=ds.bev_size, bev_width=ds.bev_size,
        tile_size_m=ds.tile_size_m,
    ).to(device)
    uses_satellite = args.fusion in ("residual", "satellite_only")
    trainable = list(completion.parameters())
    if uses_satellite:
        trainable += list(sat_encoder.parameters())
    else:
        for p in sat_encoder.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    print(f"[stage-b] device={device} samples={len(ds)} fusion={args.fusion} "
          f"Ns={source_choices} decoder=frozen mpp=0.196 "
          f"trainable_params={sum(p.numel() for p in trainable)}")

    iterator = iter(loader)
    running = 0.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        batch = move_batch(batch, device)
        n_sparse = rng.choice(source_choices)
        with torch.no_grad():
            z_star, ref_mask = ground(
                batch["source_rgb"], batch["source_points_world"], batch["source_points_uv"],
                batch["source_points_valid"], batch["origin_xy"], ds.bev_resolution_m,
            )
            sparse = slice(0, n_sparse * ds.views_per_frame)
            z_sparse, sparse_mask = ground(
                batch["source_rgb"][:, sparse], batch["source_points_world"][:, sparse],
                batch["source_points_uv"][:, sparse], batch["source_points_valid"][:, sparse],
                batch["origin_xy"], ds.bev_resolution_m,
            )
        # The coordinate-only B3 control must not consume satellite pixels at
        # all.  Its prior is a fixed relative-XY buffer inside completion.
        height_loss = z_sparse.new_tensor(0.0)
        if args.sat_encoder == "heightmap" and uses_satellite:
            # CVS-pattern DEM supervision: dense-LiDAR h_mean is the per-tile
            # ground truth for the height head on covered cells only.
            with torch.no_grad():
                dense_pts = batch["source_points_world"]
                dense_valid = batch["source_points_valid"]
                h_ref, _ = height_statistics(
                    dense_pts, dense_valid, batch["origin_xy"],
                    ds.bev_resolution_m, ds.bev_size, ds.bev_size,
                )
                h_ref = h_ref.clamp(max=30.0)
                ones = dense_valid.to(h_ref.dtype).unsqueeze(-1)
                _, count = bilinear_splat(
                    ones, dense_pts[..., :2], dense_valid, origin_xy=batch["origin_xy"],
                    resolution_m=ds.bev_resolution_m, height=ds.bev_size, width=ds.bev_size,
                )
                dem_mask = (count > 0).to(h_ref.dtype)
            prior_sat, h_pred, _ = sat_encoder(
                batch["satellite"], z_sparse, ds.tile_size_m, 0.196,
            )
            diff = (h_pred - h_ref).abs()
            height_loss = (diff * dem_mask).sum() / dem_mask.sum().clamp_min(1.0)
            z_sat = prior_sat
        else:
            z_sat = (
                sat_encoder(batch["satellite"], ds.tile_size_m, 0.196)
                if uses_satellite else torch.zeros_like(z_sparse)
            )
        z_hat = completion(z_sat, z_sparse, sparse_mask, n_sparse, args.dense_sources)
        pred_rgb, pred_depth, _ = decoder.render(
            z_hat, batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
            tile_size_m=ds.tile_size_m, image_size=ds.image_size,
        )
        # Z_star is the dense-ground scene state defined on the complete tile.
        # ref_mask remains an observation-coverage diagnostic; using it as the
        # only training mask would leave the satellite prior unsupervised on
        # most cells when the available LiDAR is sparse.
        latent_loss = F.smooth_l1_loss(z_hat, z_star)
        observed_latent_loss = F.smooth_l1_loss(z_hat * ref_mask, z_star * ref_mask)
        render_loss = F.smooth_l1_loss(pred_rgb, batch["target_rgb"])
        depth_mask = batch["target_depth_mask"]
        depth_loss = F.smooth_l1_loss(pred_depth[depth_mask], batch["target_depth"][depth_mask]) if depth_mask.any() else pred_depth.mean() * 0
        loss = (args.latent_weight * latent_loss + args.render_weight * render_loss
                + args.depth_weight * depth_loss + args.height_weight * height_loss)
        nadir_loss = pred_rgb.new_tensor(0.0)
        if args.nadir_weight > 0.0:
            # Family-5 variant B: on unobserved cells z* is the ground
            # encoder's own inpainting guess, not truth; the satellite crop is
            # the only external reference there.  The loss runs through the
            # frozen decoder, so delta must write satellite content in a
            # decoder-readable form (round-trip consistency).  At Ns=N_dense
            # alpha=0 makes z_hat=z_gnd (no_grad), so this term contributes no
            # gradient and the identity gate stays exact.
            nadir_ref = satellite_bev_crop(batch["satellite"], ds.tile_size_m, 0.196, ds.bev_size)
            nadir_rgb, _ = decoder.render_nadir(
                z_hat, batch["origin_xy"], tile_size_m=ds.tile_size_m,
                bev_size=ds.bev_size, z_top_m=args.nadir_top_m,
            )
            nadir_loss = nadir_distance(nadir_rgb, nadir_ref, 1.0 - ref_mask)
            loss = loss + args.nadir_weight * nadir_loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        running += float(loss.detach())
        if step == 1 or step % 50 == 0:
            print(f"step={step}/{args.steps} loss={running / (50 if step >= 50 else step):.5f} "
                  f"latent={float(latent_loss):.5f} observed={float(observed_latent_loss):.5f} render={float(render_loss):.5f} "
                  f"nadir={float(nadir_loss):.5f} height={float(height_loss):.5f} Ns={n_sparse} elapsed={time.time()-t0:.1f}s", flush=True)
            running = 0.0
        if step % 500 == 0 or step == args.steps:
            config = dict(vars(args))
            config["coordinate_prior"] = "fixed_metric_relative_xy_fourier_v1"
            config["sat_encoder"] = args.sat_encoder
            config["sat_encoder_kwargs"] = sat_encoder_kwargs
            torch.save({
                "satellite_encoder": sat_encoder.state_dict(), "completion": completion.state_dict(),
                "stage_a": args.stage_a, "config": config, "step": step,
            }, out / "stage_b.pt")
    print(f"[stage-b] checkpoint={out / 'stage_b.pt'}")


if __name__ == "__main__":
    main()
