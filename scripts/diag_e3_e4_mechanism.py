#!/usr/bin/env python3
"""Mechanism probe for the E3/E4 gates: WHY the trained updater barely writes.

Replays the aligned trajectory (and one-shot) with instrumentation and, per
step, reports on the current measurement's supervised cells:
  err_before/after   state height MAE before vs after the update (g_update)
  meas_readout_err   height MAE of the measurement latent itself = the
                     information a write COULD have landed
  copy_err           counterfactual: state with meas latent copied into
                     support (init kept outside) = naive-copy ceiling
  retention_overlap  fraction of the write region already inside the
                     visited mask the retention loss anchors
  latent_delta_in    mean |dlatent| inside support (is anything written?)
Usage: python scripts/diag_e3_e4_mechanism.py --scenes DIR --vggt_cache DIR
       --interface PT --assimilation PT [--out json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.models import ColumnFieldDecoder  # noqa: E402
from world3d.unified_bev.readouts import freeze_module  # noqa: E402
from world3d.unified_bev.state_models import (  # noqa: E402
    BEVSurfaceDensityDecoder,
    BEVWorldHeightDecoder,
    EvidenceAwareUpdater,
    GroundMeasurementEncoder,
    OneShotAssimilator,
    SatelliteInitializer,
    WorldGeometryEncoder,
    aggregate_measurements,
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
from world3d.unified_bev.world_state import visited_mask  # noqa: E402
from world3d.unified_bev.world_vggt import (  # noqa: E402
    chunk_measurement_from_cache,
    load_world_vggt_cache,
)


def _mae(pred, target, mask) -> float:
    w = mask.to(pred.dtype)
    denom = float(w.sum())
    if denom < 256:
        return float("nan")
    return float(((pred - target).abs() * w).sum() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--vggt_cache", required=True)
    ap.add_argument("--interface", required=True)
    ap.add_argument("--assimilation", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dgm_tiles", default=None,
                    help="must match the checkpoint's training-time anchor")
    ap.add_argument("--kitti360_root", default="/media/shizhm/sda1/KITTI-360")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = WorldStateSceneDataset(args.scenes)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_world_state)
    iface = torch.load(args.interface, map_location=device, weights_only=False)
    fp = validate_world_interface_checkpoint(iface)
    b = torch.load(args.assimilation, map_location=device, weights_only=False)
    validate_assimilation_checkpoint(b, fp)

    world_enc = WorldGeometryEncoder(**iface["encoder_config"]).to(device)
    height_r = BEVWorldHeightDecoder(**iface["height_reader_config"]).to(device)
    density_r = BEVSurfaceDensityDecoder(**iface["density_reader_config"]).to(device)
    world_enc.load_state_dict(iface["encoder"])
    height_r.load_state_dict(iface["height_reader"])
    density_r.load_state_dict(iface["density_reader"])
    sat_init = SatelliteInitializer(bev_height=ds.bev_size, bev_width=ds.bev_size,
                                    tile_size_m=ds.tile_size_m).to(device)
    updater = EvidenceAwareUpdater().to(device)
    sat_init.load_state_dict(b["satellite_initializer"])
    updater.load_state_dict(b["updater"])
    meas_enc = GroundMeasurementEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    meas_enc.load_state_dict(b["measurement_encoder"])
    for m in (world_enc, height_r, density_r, sat_init, updater, meas_enc):
        freeze_module(m)

    report = []
    with torch.no_grad():
        for inputs, sup, blob in loader:
            spec = spec_from_inputs(inputs, device=device)
            sat = inputs.satellite_bev.to(device)
            height = sup.height.to(device)
            valid = sup.world_valid.to(device)
            chunk_support = sup.chunk_lidar_support.to(device)
            z_world = world_enc(height, density := sup.density.to(device), valid)
            scene_id = blob["scene_id"]
            cache = load_world_vggt_cache(
                args.vggt_cache, scene_id,
                str(blob["world_target_version"]), str(blob["world_target_hash"]),
            )
            state = sat_init(sat, spec)
            snapshots = {0: state}
            meas_list = []
            dgm_tile = None
            if args.dgm_tiles is not None:
                from world3d.unified_bev.dgm import DgmAnchor, DgmTileSet, anchor_tile_tensor
                if not hasattr(main, "_dgm_tiles"):
                    main._dgm_tiles = DgmTileSet.from_dir(Path(args.dgm_tiles))
                anchor = DgmAnchor.from_blob(blob, main._dgm_tiles, Path(args.kitti360_root))
                dgm_tile = anchor_tile_tensor(anchor, inputs.origin_xy[0].numpy(),
                                              ds.bev_size, float(spec.resolution_m), device)
            t_chunks = chunk_support.shape[1]
            for t in range(t_chunks):
                meas_list.append(chunk_measurement_from_cache(
                    meas_enc, cache["chunks"][str(t + 1)],
                    origin_xy=spec.origin_xy,
                    resolution_m=float(spec.resolution_m),
                    z_datum_m=spec.z_datum_m,
                    chunk_index=t + 1,
                    query_fid=int(blob["chunk_table"][t]["core_fid"]),
                    detach=True,
                    dgm_abs_z=None if dgm_tile is None else dgm_tile[0],
                    dgm_valid=None if dgm_tile is None else dgm_tile[1],
                ))
            for t, meas in enumerate(meas_list, start=1):
                prev = state
                state = updater(state, meas).state
                snapshots[t] = state
                mt = meas.support
                mt_sup = mt & valid
                vis_prev = visited_mask(chunk_support, t - 1)
                overlap = float((vis_prev & mt_sup).sum()) / max(int(mt_sup.sum()), 1)
                h_prev = height_r(prev.latent)
                h_new = height_r(state.latent)
                h_meas = height_r(meas.latent)
                copied = torch.where(mt, meas.latent, prev.latent)
                h_copy = height_r(copied)
                d_in = float((state.latent - prev.latent).abs()
                             .masked_select(mt.expand_as(state.latent)).mean())
                row = {
                    "scene_id": scene_id, "t": t,
                    "mt_sup_cells": int(mt_sup.sum()),
                    "retention_overlap": round(overlap, 3),
                    "err_before": round(_mae(h_prev, height, mt_sup), 3),
                    "err_after": round(_mae(h_new, height, mt_sup), 3),
                    "g_update": round(_mae(h_prev, height, mt_sup) - _mae(h_new, height, mt_sup), 3),
                    "meas_readout_err": round(_mae(h_meas, height, mt_sup), 3),
                    "copy_err": round(_mae(h_copy, height, mt_sup), 3),
                    "latent_delta_in": round(d_in, 4),
                }
                if 1 in snapshots and t > 1:
                    m1 = meas_list[0].support & valid
                    row["forget_1_to_t"] = round(
                        _mae(h_new, height, m1) - _mae(height_r(snapshots[1].latent), height, m1), 3)
                # one-shot reference on the same cells
                if t == t_chunks:
                    agg = aggregate_measurements(meas_list)
                    os_state = OneShotAssimilator(updater)(snapshots[0], meas_list).state
                    row["oneshot_err_on_mt"] = round(_mae(height_r(os_state.latent), height, mt_sup), 3)
                    row["agg_readout_err_on_mt"] = round(_mae(height_r(agg.latent), height, mt_sup), 3)
                report.append(row)
            print(f"[scene {scene_id}] done")

    lines = [json.dumps(r) for r in report]
    print("\n".join(lines))
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
