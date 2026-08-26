#!/usr/bin/env python3
"""Evaluate Z0..ZT on visited / ahead regions with update gain and forgetting."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.models import ColumnFieldDecoder, render_multi_view  # noqa: E402
from world3d.unified_bev.readouts import freeze_module  # noqa: E402
from world3d.unified_bev.state_models import (  # noqa: E402
    BEVSurfaceDensityDecoder,
    BEVWorldHeightDecoder,
    EvidenceAwareUpdater,
    FixedXYInitializer,
    GroundMeasurementEncoder,
    OneShotAssimilator,
    SatelliteInitializer,
    WorldGeometryEncoder,
)
from world3d.unified_bev.world_checkpoints import (  # noqa: E402
    validate_assimilation_checkpoint,
    validate_world_interface_checkpoint,
)
from world3d.unified_bev.world_data import (  # noqa: E402
    WorldStateSceneDataset,
    collate_world_state,
    spec_from_inputs,
)
from world3d.unified_bev.world_state import (  # noqa: E402
    ahead_mask,
    empty_state,
    offroute_mask,
    visited_mask,
)
from world3d.unified_bev.world_vggt import (  # noqa: E402
    chunk_measurement_from_cache,
    load_world_vggt_cache,
    teacher_measurement,
)



def _mae(pred, target, mask) -> float:
    w = mask.to(pred.dtype)
    denom = w.sum()
    if float(denom) < 256:
        return float("nan")
    return float(((pred - target).abs() * w).sum() / denom)


def _finite(x):
    return x if math.isfinite(x) else None


def perturb_sat(sat, meters, axis="cross"):
    if meters == 0:
        return sat
    # south-up BEV: row=y, col=x. cross-road ~ x shift.
    pixels = int(round(meters / 0.5))
    return torch.roll(sat, shifts=pixels if axis == "cross" else pixels, dims=-1 if axis == "cross" else -2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--interface", required=True)
    ap.add_argument("--assimilation", required=True)
    ap.add_argument("--vggt_cache", default=None,
                    help="frozen-VGGT measurement cache matching the scenes; "
                         "without it the LiDAR teacher fallback is evaluated")
    ap.add_argument("--control", choices=["aligned", "xy", "random", "shift_cross", "shift_road",
                                          "sat_only", "ground_only", "one_shot", "world_upper"],
                    default="aligned")
    ap.add_argument("--shift_m", type=float, default=5.0)
    ap.add_argument("--records_out", default="runs/world_state_eval.jsonl")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    ds = WorldStateSceneDataset(args.scenes)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_world_state)
    iface = torch.load(args.interface, map_location=device, weights_only=False)
    fp = validate_world_interface_checkpoint(iface)
    b = torch.load(args.assimilation, map_location=device, weights_only=False)
    validate_assimilation_checkpoint(b, fp)

    world_enc = WorldGeometryEncoder(**iface["encoder_config"]).to(device)
    height_r = BEVWorldHeightDecoder(**iface["height_reader_config"]).to(device)
    density_r = BEVSurfaceDensityDecoder(**iface["density_reader_config"]).to(device)
    depth_r = ColumnFieldDecoder(**iface["depth_reader_config"]).to(device)
    world_enc.load_state_dict(iface["encoder"])
    height_r.load_state_dict(iface["height_reader"])
    density_r.load_state_dict(iface["density_reader"])
    depth_r.load_state_dict(iface["depth_reader"])
    bev = ds.bev_size
    sat_init = SatelliteInitializer(bev_height=bev, bev_width=bev, tile_size_m=ds.tile_size_m).to(device)
    xy_init = FixedXYInitializer(bev_height=bev, bev_width=bev, tile_size_m=ds.tile_size_m).to(device)
    updater = EvidenceAwareUpdater().to(device)
    sat_init.load_state_dict(b["satellite_initializer"])
    xy_init.load_state_dict(b["xy_initializer"])
    updater.load_state_dict(b["updater"])
    meas_enc = GroundMeasurementEncoder(bev_height=bev, bev_width=bev).to(device)
    meas_enc.load_state_dict(b["measurement_encoder"])
    for m in (world_enc, height_r, density_r, depth_r, sat_init, xy_init, updater, meas_enc):
        freeze_module(m)
    vggt_caches: dict = {}
    if args.vggt_cache is None:
        print("[world-eval] WARNING: no --vggt_cache; trajectories use the "
              "LiDAR teacher measurement fallback (diagnostic only)")

    rows = []
    with torch.no_grad():
        for inputs, sup, blob in loader:
            spec = spec_from_inputs(inputs, device=device)
            sat = inputs.satellite_bev.to(device)
            if args.control == "random":
                sat = torch.roll(sat, shifts=sat.shape[-1] // 2, dims=-1)
            elif args.control == "shift_cross":
                sat = perturb_sat(sat, args.shift_m, "cross")
            elif args.control == "shift_road":
                sat = perturb_sat(sat, args.shift_m, "road")
            height = sup.height.to(device)
            density = sup.density.to(device)
            valid = sup.world_valid.to(device)
            chunk_support = sup.chunk_lidar_support.to(device)
            z_world = world_enc(height, density, valid)
            if args.control == "ground_only":
                state = empty_state(spec, 64, device=device)
            elif args.control in ("xy",):
                state = xy_init(sat, spec)
            elif args.control == "world_upper":
                state = empty_state(spec, 64, device=device)
                state.latent = z_world
            else:
                state = sat_init(sat, spec)

            t_chunks = chunk_support.shape[1]
            if args.vggt_cache is not None:
                scene_id = blob["scene_id"]
                if scene_id not in vggt_caches:
                    vggt_caches[scene_id] = load_world_vggt_cache(
                        args.vggt_cache, scene_id,
                        str(blob["world_target_version"]), str(blob["world_target_hash"]),
                    )
            snapshots = {0: state}
            meas_list = []
            if args.control not in ("sat_only", "world_upper"):
                for t in range(t_chunks):
                    if args.vggt_cache is not None:
                        meas = chunk_measurement_from_cache(
                            meas_enc,
                            vggt_caches[scene_id]["chunks"][str(t + 1)],
                            origin_xy=spec.origin_xy,
                            resolution_m=float(spec.resolution_m),
                            z_datum_m=spec.z_datum_m,
                            chunk_index=t + 1,
                            detach=True,
                        )
                    else:
                        support = chunk_support[:, t]
                        meas = teacher_measurement(
                            world_enc(height * support, density * support, support),
                            support, t + 1,
                        )
                    meas_list.append(meas)
                if args.control == "one_shot":
                    snapshots[t_chunks] = OneShotAssimilator(updater)(state, meas_list).state
                    for t in range(1, t_chunks):
                        snapshots[t] = snapshots[0]
                else:
                    for t, meas in enumerate(meas_list, start=1):
                        state = updater(state, meas).state
                        snapshots[t] = state
            # E3/E4 regions are the measurement supports (what the updater
            # actually writes); with VGGT measurements these extend beyond the
            # LiDAR chunk mask, so preservation is checked against them
            meas_supports = [m.support for m in meas_list]

            all_ground = chunk_support.any(dim=1)
            for t, st in snapshots.items():
                vis = visited_mask(chunk_support, t)
                ahead = ahead_mask(sup.future_route_support.to(device), vis)
                off = offroute_mask(valid, all_ground)
                h_hat = height_r(st.latent)
                d_hat = density_r(st.latent)
                row = {
                    "scene_id": blob["scene_id"],
                    "control": args.control,
                    "measurement_source": "vggt_cache" if args.vggt_cache else "lidar_teacher_fallback",
                    "version": int(t),
                    "traversed_m": float(sup.traversed_m[0, t - 1]) if t > 0 else 0.0,
                    "visited_fraction": float(vis.float().mean()),
                    "ahead_fraction": float(ahead.float().mean()),
                    "offroute_cells": int(off.sum()),
                    "height_visited_mae": _finite(_mae(h_hat, height, vis & valid)),
                    "height_ahead_mae": _finite(_mae(h_hat, height, ahead & valid)),
                    "density_visited_mae": _finite(_mae(d_hat, density, vis & valid)),
                    "density_ahead_mae": _finite(_mae(d_hat, density, ahead & valid)),
                    "height_offroute_mae_diag": _finite(_mae(h_hat, height, off)),
                    "outside_latent_max": 0.0 if t == 0 else None,
                }
                if t > 0:
                    prev = snapshots[t - 1]
                    mt = meas_supports[t - 1] if meas_supports else chunk_support[:, t - 1]
                    mt_sup = mt & valid  # pseudo-negative guard: no labels outside valid
                    row["measurement_support_cells"] = int(mt.sum())
                    row["supervised_support_cells"] = int(mt_sup.sum())
                    row["measurement_target_overlap"] = (
                        float(mt_sup.sum()) / float(mt.sum()) if int(mt.sum()) else 0.0
                    )
                    row["g_update_height"] = _finite(
                        _mae(height_r(prev.latent), height, mt_sup) - _mae(h_hat, height, mt_sup)
                    )
                    outside = ~mt
                    row["outside_latent_max"] = float(
                        (st.latent - prev.latent).abs().masked_select(outside.expand_as(st.latent)).max().item()
                    ) if outside.any() else 0.0
                    if 1 in snapshots and t > 1:
                        m1 = meas_supports[0] if meas_supports else chunk_support[:, 0]
                        row["forget_1_to_t_height"] = _finite(
                            _mae(h_hat, height, m1 & valid) - _mae(height_r(snapshots[1].latent), height, m1 & valid)
                        )
                # held-out depth on the current chunk query
                q = min(max(t - 1, 0), sup.query_depth.shape[1] - 1)
                rgb = sup.query_rgb[:, q].to(device)
                k = sup.query_K[:, q].to(device)
                tw = sup.query_T_world_cam[:, q].to(device)
                gt = sup.query_depth[:, q].to(device)
                gm = sup.query_depth_mask[:, q].to(device)
                _, pred_d, _ = render_multi_view(
                    depth_r, st.latent, k, tw, spec.origin_xy,
                    tile_size_m=ds.tile_size_m, image_size=(rgb.shape[-1], rgb.shape[-2]),
                )
                if gm.any():
                    rel = ((pred_d - gt).abs() / gt.clamp_min(1e-3))[gm]
                    row["depth_absrel"] = _finite(float(rel.mean()))
                else:
                    row["depth_absrel"] = None
                rows.append(row)

    out = Path(args.records_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"records={len(rows)} -> {out}")


if __name__ == "__main__":
    main()
