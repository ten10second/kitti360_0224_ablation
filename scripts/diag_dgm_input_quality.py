#!/usr/bin/env python3
"""Input-level DGM validation: on tier2 (far-only) cells, compare the legacy
anchored envelope vs the DGM-anchored estimate against the static target
height.  Pure input comparison — no training involved."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.dgm import DgmAnchor, DgmTileSet, anchor_tile_tensor
from world3d.unified_bev.geometry import ground_height_quantile
from world3d.unified_bev.models import unproject_dense
from world3d.unified_bev.state_models import GroundMeasurementEncoder
from world3d.unified_bev.world_data import WorldStateSceneDataset, spec_from_inputs
from world3d.unified_bev.world_vggt import load_world_vggt_cache

DGM_TILES = "/media/shizhm/sda1/proposal/Cross-View Conditional Coding of Route-Specific Gaussian Scenes/outputs/bw_dgm_dom/tiles"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = WorldStateSceneDataset("runs/world_state_targets_smoke")
    dl = DataLoader(ds, batch_size=1, collate_fn=lambda b: b[0])
    inputs, sup, blob = next(iter(dl))
    spec = spec_from_inputs(inputs, device=device)
    tiles = DgmTileSet.from_dir(Path(DGM_TILES))
    enc = GroundMeasurementEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device).eval()
    cache = load_world_vggt_cache("runs/world_state_vggt_smoke", blob["scene_id"],
                                  str(blob["world_target_version"]), str(blob["world_target_hash"]))
    anchor = DgmAnchor.from_blob(blob, tiles, Path("/media/shizhm/sda1/KITTI-360"))
    z_t, v_t = anchor_tile_tensor(anchor, inputs.origin_xy[0].numpy(),
                                  ds.bev_size, float(ds.resolution_m), device)
    height = sup.height.to(device)
    datum = spec.z_datum_m.view(1, 1, 1, 1)
    n_chunks = sup.chunk_lidar_support.shape[1]

    print(f"scene {str(blob['scene_id'])[-40:]}  chunks={n_chunks}")
    print(f"{'t':>2s} {'n_t2':>6s} {'leg_med':>8s} {'dgm_med':>8s} {'leg_mae':>8s} {'dgm_mae':>8s}")
    agg_leg, agg_dgm = [], []
    for t in range(n_chunks):
        entry = cache["chunks"][str(t + 1)]
        dd = entry["depth"].float().to(device).unsqueeze(0)
        cc = entry["conf"].float().to(device).unsqueeze(0)
        K = entry["K"].float().to(device).unsqueeze(0)
        Tw = entry["T_world_cam"].float().to(device).unsqueeze(0)
        with torch.no_grad():
            gate, dep = enc._gate(dd, cc)
            tr = gate & (dep <= 15.0)
            pts = unproject_dense(dep, K, Tw).view(1, -1, dep.shape[-2] * dep.shape[-1], 3)
            _, c_tr = ground_height_quantile(
                pts, tr.view(1, -1, dep.shape[-2] * dep.shape[-1]),
                spec.origin_xy, float(spec.resolution_m), ds.bev_size, ds.bev_size,
                quantile=0.15)
            sup_tr = c_tr > 0
            t2 = (c_tr.new_zeros(1, 1, ds.bev_size, ds.bev_size) > 0)
            # legacy h_rel
            h0, s0, qa0 = enc.ground_field(dd, cc, K, Tw, spec.origin_xy,
                                           float(spec.resolution_m), spec.z_datum_m)
            # dgm h_rel
            h1, s1, qa1 = enc.ground_field(dd, cc, K, Tw, spec.origin_xy,
                                           float(spec.resolution_m), spec.z_datum_m,
                                           dgm_abs_z=z_t, dgm_valid=v_t)
            t2 = s0 & ~sup_tr & v_t.bool()
        if int(t2.sum()) < 50:
            print(f"{t+1:>2d} {int(t2.sum()):>6d}  (few tier2 cells)")
            continue
        gt = height[t2]
        leg_err = (h0[t2] - gt).abs()   # both datum-relative
        dgm_err = (h1[t2] - gt).abs()
        agg_leg.append(leg_err); agg_dgm.append(dgm_err)
        print(f"{t+1:>2d} {int(t2.sum()):>6d} {float(leg_err.median()):>7.3f}m {float(dgm_err.median()):>7.3f}m "
              f"{float(leg_err.mean()):>7.3f}m {float(dgm_err.mean()):>7.3f}m")
    if agg_leg:
        leg = torch.cat([x.flatten() for x in agg_leg])
        dgm = torch.cat([x.flatten() for x in agg_dgm])
        print(f"ALL tier2 cells: legacy med={float(leg.median()):.3f} mae={float(leg.mean()):.3f} | "
              f"dgm med={float(dgm.median()):.3f} mae={float(dgm.mean()):.3f}")


if __name__ == "__main__":
    main()
