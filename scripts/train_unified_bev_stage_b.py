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

from world3d.unified_bev.data import (
    UnifiedBEVDataset,
    attach_dense_geometry,
    dense_geometry_from_batch,
    load_dense_cached_unified_bev,
)
from world3d.unified_bev.data import load_cached_unified_bev
from world3d.unified_bev.checkpoints import (
    STAGE_B_SCHEMA_VERSION,
    validate_stage_a_checkpoint,
    validate_stage_a_dataset,
)
from world3d.unified_bev.geometry import (
    geometry_supervision_support,
    observation_partition,
    relative_height_map,
    target_pixels_supported_by_bev,
)
from world3d.unified_bev.losses import (
    high_frequency_masked_l1,
    low_frequency_l1,
    masked_smooth_l1,
)
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundDenseBEVEncoder,
    GroundBEVEncoder,
    LatentCompletion,
    HeightMapSatellitePrior,
    SatelliteBEVEncoder,
    SatelliteViTEncoder,
)
from world3d.unified_bev.readouts import BEVHeightDecoder, freeze_module


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def parse_source_choices(value: str | None, fixed: int, dense: int) -> tuple[int, ...]:
    choices = (fixed,) if value is None else tuple(int(x) for x in value.split(",") if x.strip())
    if not choices or any(x < 1 or x >= dense for x in choices):
        raise ValueError(
            f"Stage-B training source choices must be within [1,{dense - 1}], got {choices}; "
            f"Ns={dense} is the exact frozen-ground identity and belongs in evaluation only"
        )
    return tuple(dict.fromkeys(choices))


def masked_abs_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(value.dtype)
    if weights.ndim == value.ndim - 1:
        weights = weights.unsqueeze(1)
    if weights.shape[1] == 1 and value.shape[1] != 1:
        weights = weights.expand(-1, value.shape[1], -1, -1)
    denom = weights.sum()
    return (value.abs() * weights).sum() / denom.clamp_min(1.0)


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
                    help="comma-separated Stage-B training choices, e.g. 1,2,4; dense identity is eval-only")
    ap.add_argument("--fusion", choices=["residual", "ground_only", "satellite_only", "coordinate_only"], default="residual")
    ap.add_argument("--sat_encoder", choices=["vit", "cnn", "heightmap"], default="vit",
                    help="satellite encoder family; cnn is the legacy 3-layer convolution")
    ap.add_argument("--sat_dim", type=int, default=256)
    ap.add_argument("--sat_depth", type=int, default=4)
    ap.add_argument("--sat_heads", type=int, default=4)
    ap.add_argument("--sat_patch", type=int, default=8)
    ap.add_argument("--height_weight", type=float, default=0.0,
                    help="optional satellite direct-height auxiliary; not primary geometry evidence")
    ap.add_argument("--image_width", type=int, default=160)
    ap.add_argument("--image_height", type=int, default=96)
    ap.add_argument("--max_points", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_target_spacing_m", type=float, default=5.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--latent_weight", type=float, default=0.01,
                    help="small diagnostic regularizer to dense-ground Z*, never the main loss")
    ap.add_argument("--anchor_weight", type=float, default=1.0)
    ap.add_argument("--geometry_fill_weight", type=float, default=1.0)
    ap.add_argument("--rgb_lowfreq_weight", type=float, default=0.1)
    ap.add_argument("--rgb_observed_weight", type=float, default=0.1)
    ap.add_argument("--frequency_scale", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--cache", default=None, help="prebuilt sample cache; serving from RAM, workers forced to 0")
    ap.add_argument("--geometry_cache", "--m3d_cache", dest="geometry_cache", default=None,
                    help="dense geometry cache; Stage A must use the same geometry family")
    ap.add_argument("--chunked", action="store_true",
                    help="route-chunk mode (cache v7): sparse_source_choices become kept-chunk "
                         "counts K in [1, N_c-1]; the hole is the middle N_c-K chunks")
    ap.add_argument("--chunks_per_window", type=int, default=4)
    ap.add_argument("--chunk_arc_m", type=float, default=12.0)
    ap.add_argument("--guard_m", type=float, default=4.0)
    ap.add_argument("--frames_per_chunk", type=int, default=2)
    ap.add_argument("--max_geometry_frames", type=int, default=8)
    ap.add_argument("--window_stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_choices = parse_source_choices(args.sparse_source_choices, args.sparse_sources, args.dense_sources)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.stage_a, map_location=device, weights_only=False)
    stage_a_fingerprint = validate_stage_a_checkpoint(ckpt)
    if args.fusion == "ground_only":
        raise ValueError(
            "ground_only has no Stage-B parameters; evaluate sparse Stage A directly without --stage_b"
        )
    if args.chunked and not args.geometry_cache:
        raise ValueError("--chunked requires --geometry_cache (chunk cache v7)")
    if args.chunked and args.cache:
        raise ValueError("--chunked currently requires the raw dataset path")
    if args.cache and args.geometry_cache:
        ds = load_dense_cached_unified_bev(args.cache, args.geometry_cache)
    elif args.cache:
        ds = load_cached_unified_bev(args.cache)
    elif args.chunked:
        from world3d.unified_bev.data import ChunkedUnifiedBEVDataset, attach_chunk_geometry
        ds = attach_chunk_geometry(ChunkedUnifiedBEVDataset(
            args.manifest, lidar_root=args.lidar_root, drive=args.drive,
            chunks_per_window=args.chunks_per_window, chunk_arc_m=args.chunk_arc_m,
            guard_m=args.guard_m, frames_per_chunk=args.frames_per_chunk,
            max_geometry_frames=args.max_geometry_frames,
            window_stride=args.window_stride, max_samples=args.max_samples,
            image_size=(args.image_width, args.image_height),
            max_points_per_view=args.max_points,
        ), args.geometry_cache)
    else:
        ds = UnifiedBEVDataset(
            args.manifest, lidar_root=args.lidar_root, dense_source_count=args.dense_sources,
            sparse_source_count=max(source_choices), image_size=(args.image_width, args.image_height),
            max_points_per_view=args.max_points, max_samples=args.max_samples, drive=args.drive,
            min_target_spacing_m=args.min_target_spacing_m,
        )
        if args.geometry_cache:
            ds = attach_dense_geometry(ds, args.geometry_cache)
    if args.cache:
        args.num_workers = 0  # RAM serving; forked workers would copy the cache
    validate_stage_a_dataset(
        ckpt, ds, dense_geometry_attached=bool(args.geometry_cache),
    )
    if args.chunked:
        # choices are kept-chunk counts K; N_c replaces the frame count
        source_choices = parse_source_choices(
            args.sparse_source_choices, args.chunks_per_window - 1, ds.chunks_per_window,
        )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        drop_last=True, persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    ground_config = dict(ckpt["ground_config"])
    family = ground_config.pop("family")
    ground_cls = GroundDenseBEVEncoder if family == "dense" else GroundBEVEncoder
    ground = ground_cls(**ground_config).to(device)
    decoder = ColumnFieldDecoder(**ckpt["renderer_config"]).to(device)
    geometry_decoder = BEVHeightDecoder(**ckpt["geometry_decoder_config"]).to(device)
    ground.load_state_dict(ckpt["ground"])
    decoder.load_state_dict(ckpt["decoder"])
    geometry_decoder.load_state_dict(ckpt["geometry_decoder"])
    for module in (ground, decoder, geometry_decoder):
        freeze_module(module)
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
    frozen_ids = {
        id(parameter)
        for module in (ground, decoder, geometry_decoder)
        for parameter in module.parameters()
    }
    if any(id(parameter) in frozen_ids for group in opt.param_groups for parameter in group["params"]):
        raise RuntimeError("Stage-B optimizer contains a frozen Stage-A parameter")
    print(f"[stage-b] device={device} samples={len(ds)} fusion={args.fusion} "
          f"{'K(chunks)' if args.chunked else 'Ns'}={source_choices} decoder=frozen mpp=0.196 "
          f"trainable_params={sum(p.numel() for p in trainable)}")

    n_chunks = ds.chunks_per_window if args.chunked else None

    def _kept_positions(kept_count: int) -> List[int]:
        """Kept chunk positions: head chunk plus the trailing kept block."""
        return [0] + list(range(n_chunks - kept_count + 1, n_chunks))

    def _chunk_lift(rgb, positions: List[int]):
        fpc, vpf = ds.frames_per_chunk, ds.views_per_frame
        rows = torch.tensor(
            [r for p in positions for r in range(p * fpc * vpf, (p + 1) * fpc * vpf)],
            device=device,
        )
        depth = torch.cat([batch[f"dense_depth_c{p}"] for p in positions], dim=1)
        conf = torch.cat([batch[f"dense_conf_c{p}"] for p in positions], dim=1)
        return ground(
            rgb[:, rows], batch["source_K"][:, rows], depth, conf,
            batch["source_T_world_cam"][:, rows],
            batch["origin_xy"], ds.bev_resolution_m,
        )

    def _flatten_queries(batch):
        """(B, Nq, ...) query tensors -> (B*Nq, ...) so the loop below is
        identical to the single-query legacy path."""
        nq = batch["target_rgb"].shape[1]
        flat = {
            "target_rgb": batch["target_rgb"].flatten(0, 1),
            "target_K": batch["target_K"].flatten(0, 1),
            "target_T_world_cam": batch["target_T_world_cam"].flatten(0, 1),
            "target_depth": batch["target_depth"].flatten(0, 1),
            "target_depth_mask": batch["target_depth_mask"].flatten(0, 1),
            "origin_xy": batch["origin_xy"].repeat_interleave(nq, dim=0),
        }
        batch.update(flat)
        return batch

    iterator = iter(loader)
    running = 0.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        batch = move_batch(batch, device)
        if args.chunked:
            batch = _flatten_queries(batch)
        n_sparse = rng.choice(source_choices)
        with torch.no_grad():
            def _lift(rgb, start_frame, frame_count):
                sl = slice(start_frame * ds.views_per_frame,
                           (start_frame + frame_count) * ds.views_per_frame)
                if args.geometry_cache:
                    dense_depth, dense_conf = dense_geometry_from_batch(
                        batch, start_frame, frame_count, ds.views_per_frame,
                    )
                    return ground(
                        rgb[:, sl], batch["source_K"][:, sl], dense_depth,
                        dense_conf, batch["source_T_world_cam"][:, sl],
                        batch["origin_xy"], ds.bev_resolution_m,
                    )
                return ground(
                    rgb[:, sl], batch["source_points_world"][:, sl],
                    batch["source_points_uv"][:, sl], batch["source_points_valid"][:, sl],
                    batch["origin_xy"], ds.bev_resolution_m,
                )
            if args.chunked:
                z_star, ref_mask = _chunk_lift(batch["source_rgb"], list(range(n_chunks)))
                z_sparse, sparse_mask = _chunk_lift(
                    batch["source_rgb"], _kept_positions(int(n_sparse)),
                )
            else:
                z_star, ref_mask = _lift(batch["source_rgb"], 0, args.dense_sources)
                z_sparse, sparse_mask = _lift(batch["source_rgb"], 0, n_sparse)
        with torch.no_grad():
            h_ref, h_valid, _ = relative_height_map(
                batch["source_points_world"], batch["source_points_valid"],
                batch["origin_xy"], ds.bev_resolution_m, ds.bev_size, ds.bev_size,
            )
            dense_height_support = geometry_supervision_support(ref_mask, h_valid)
            observed_cells, fill_cells = observation_partition(
                sparse_mask, dense_height_support,
            )

        # The coordinate-only B3 control must not consume satellite pixels at
        # all. Its prior is a fixed relative-XY buffer inside completion.
        height_aux_loss = z_sparse.new_tensor(0.0)
        if args.sat_encoder == "heightmap" and uses_satellite:
            prior_sat, h_pred = sat_encoder(
                batch["satellite"], z_sparse, ds.tile_size_m, 0.196,
            )
            height_aux_loss = masked_smooth_l1(h_pred, h_ref, h_valid)
            z_sat = prior_sat
        else:
            z_sat = (
                sat_encoder(batch["satellite"], ds.tile_size_m, 0.196)
                if uses_satellite else torch.zeros_like(z_sparse)
            )
        completion_output = completion(
            z_sat, z_sparse, sparse_mask, n_sparse,
            n_chunks if args.chunked else args.dense_sources,
        )
        z_hat = completion_output.latent
        z_render = (
            z_hat.repeat_interleave(batch["target_K"].shape[0] // z_hat.shape[0], dim=0)
            if batch["target_K"].shape[0] != z_hat.shape[0] else z_hat
        )
        pred_rgb, pred_depth, _ = decoder.render(
            z_render, batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
            tile_size_m=ds.tile_size_m, image_size=ds.image_size,
        )
        geometry_pred = geometry_decoder(z_hat)
        with torch.no_grad():
            # Dense-ground Z* supplies a dense target-view geometry teacher.
            # Sparse target LiDAR overwrites it wherever metric truth exists.
            z_star_render = (
                z_star.repeat_interleave(batch["target_K"].shape[0] // z_star.shape[0], dim=0)
                if batch["target_K"].shape[0] != z_star.shape[0] else z_star
            )
            _, teacher_depth, _ = decoder.render(
                z_star_render, batch["target_K"], batch["target_T_world_cam"],
                batch["origin_xy"], tile_size_m=ds.tile_size_m,
                image_size=ds.image_size,
            )
            teacher_depth = torch.where(
                batch["target_depth_mask"], batch["target_depth"], teacher_depth,
            )
            teacher_depth_valid = torch.isfinite(teacher_depth) & (teacher_depth > 1e-3)
        sparse_support_queries = (
            sparse_mask.repeat_interleave(batch["target_K"].shape[0] // sparse_mask.shape[0], dim=0)
            if batch["target_K"].shape[0] != sparse_mask.shape[0] else sparse_mask
        )
        target_supported_pixels = target_pixels_supported_by_bev(
            teacher_depth, teacher_depth_valid,
            batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
            ds.tile_size_m, sparse_support_queries,
        )

        anchor_loss = masked_smooth_l1(z_hat, z_sparse, observed_cells)
        geometry_fill_loss = masked_smooth_l1(
            geometry_pred, h_ref, fill_cells,
        )
        rgb_lowfreq_loss = low_frequency_l1(
            pred_rgb, batch["target_rgb"], args.frequency_scale,
        )
        rgb_observed_loss = high_frequency_masked_l1(
            pred_rgb, batch["target_rgb"], target_supported_pixels,
            args.frequency_scale,
        )
        latent_regularizer = F.smooth_l1_loss(z_hat, z_star)
        geometry_all_diag = masked_smooth_l1(geometry_pred, h_ref, h_valid)
        depth_mask = batch["target_depth_mask"]
        depth_loss_diag = (
            F.smooth_l1_loss(pred_depth[depth_mask], batch["target_depth"][depth_mask])
            if depth_mask.any() else pred_depth.mean() * 0
        )
        loss = (
            args.anchor_weight * anchor_loss
            + args.geometry_fill_weight * geometry_fill_loss
            + args.rgb_lowfreq_weight * rgb_lowfreq_loss
            + args.rgb_observed_weight * rgb_observed_loss
            + args.latent_weight * latent_regularizer
            + args.height_weight * height_aux_loss
        )
        opt.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
            opt.step()
        running += float(loss.detach())
        if step == 1 or step % 50 == 0:
            correction_observed = masked_abs_mean(
                completion_output.correction, observed_cells,
            )
            correction_fill = masked_abs_mean(
                completion_output.correction, fill_cells,
            )
            print(f"step={step}/{args.steps} loss={running / (50 if step >= 50 else step):.5f} "
                  f"anchor={float(anchor_loss):.5f} geo_fill={float(geometry_fill_loss):.5f} "
                  f"rgb_low={float(rgb_lowfreq_loss):.5f} rgb_obs={float(rgb_observed_loss):.5f} "
                  f"latent_reg={float(latent_regularizer):.5f} geo_all={float(geometry_all_diag):.5f} "
                  f"depth_diag={float(depth_loss_diag):.5f} h_aux={float(height_aux_loss):.5f} "
                  f"corr_obs={float(correction_observed):.5f} corr_fill={float(correction_fill):.5f} "
                  f"Mobs={float(observed_cells.float().mean()):.3f} "
                  f"Mdense_height={float(dense_height_support.float().mean()):.3f} "
                  f"Mfill={float(fill_cells.float().mean()):.3f} "
                  f"target_support={float(target_supported_pixels.float().mean()):.5f} "
                  f"Ns={n_sparse} elapsed={time.time()-t0:.1f}s", flush=True)
            running = 0.0
        if step % 500 == 0 or step == args.steps:
            config = dict(vars(args))
            config["coordinate_prior"] = "fixed_metric_relative_xy_fourier_v1"
            config["sat_encoder"] = args.sat_encoder
            config["sat_encoder_kwargs"] = sat_encoder_kwargs
            torch.save({
                "schema_version": STAGE_B_SCHEMA_VERSION,
                "satellite_encoder": sat_encoder.state_dict(), "completion": completion.state_dict(),
                "stage_a": args.stage_a,
                "stage_a_fingerprint": stage_a_fingerprint,
                "config": config, "step": step,
            }, out / "stage_b.pt")
    print(f"[stage-b] checkpoint={out / 'stage_b.pt'}")


if __name__ == "__main__":
    main()
