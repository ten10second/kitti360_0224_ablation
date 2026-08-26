#!/usr/bin/env python3
"""Stage B: shared assimilation — one measurement stream and one updater serve
both the satellite-initialized and the XY-initialized chains in the same batch.

Z_t^sat = U(Z_{t-1}^sat, G_t) and Z_t^xy = U(Z_{t-1}^xy, G_t) share the exact
same G_t/M_t, updater, and frozen readers, so initialization is the only
manipulated variable.  ground_only and one_shot are evaluation-time variants
(eval_world_state_trajectory --control), never separately trained.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.losses import masked_smooth_l1  # noqa: E402
from world3d.unified_bev.readouts import freeze_module  # noqa: E402
from world3d.unified_bev.state_models import (  # noqa: E402
    BEVSurfaceDensityDecoder,
    BEVWorldHeightDecoder,
    EvidenceAwareUpdater,
    FixedXYInitializer,
    GroundMeasurementEncoder,
    SatelliteInitializer,
    WorldGeometryEncoder,
)
from world3d.unified_bev.world_checkpoints import (  # noqa: E402
    validate_scenes_manifest,
    validate_world_interface_checkpoint,
)
from world3d.unified_bev.world_data import (  # noqa: E402
    WorldStateSceneDataset,
    collate_world_state,
    spec_from_inputs,
)
from world3d.unified_bev.world_state import (  # noqa: E402
    WORLD_STATE_SCHEMA_VERSION,
    GroundMeasurement,
    supervised_region,
    visited_mask,
)
from world3d.unified_bev.world_vggt import (  # noqa: E402
    chunk_measurement_from_cache,
    load_world_vggt_cache,
    teacher_measurement,
)
from world3d.unified_bev.models import ColumnFieldDecoder  # noqa: E402


def _load_interface(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    fp = validate_world_interface_checkpoint(ck)
    enc = WorldGeometryEncoder(**{k: v for k, v in ck["encoder_config"].items()}).to(device)
    h = BEVWorldHeightDecoder(**ck["height_reader_config"]).to(device)
    d = BEVSurfaceDensityDecoder(**ck["density_reader_config"]).to(device)
    z = ColumnFieldDecoder(**ck["depth_reader_config"]).to(device)
    enc.load_state_dict(ck["encoder"]); h.load_state_dict(ck["height_reader"])
    d.load_state_dict(ck["density_reader"]); z.load_state_dict(ck["depth_reader"])
    for m in (enc, h, d, z):
        freeze_module(m)
    return ck, fp, enc, h, d, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--interface", required=True)
    ap.add_argument("--vggt_cache", default=None,
                    help="per-scene frozen-VGGT measurement cache (the real "
                         "measurement path); without it the LiDAR teacher "
                         "fallback is used and the run is diagnostic only")
    ap.add_argument("--out", default="runs/world_state_assimilation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--unroll", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    ds = WorldStateSceneDataset(args.scenes)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_world_state, num_workers=0)
    ck, fp, world_enc, height_r, density_r, depth_r = _load_interface(args.interface, device)
    validate_scenes_manifest(ck, ds.manifest_hash)
    vggt_caches: dict = {}
    if args.vggt_cache is None:
        print("[world-b] WARNING: no --vggt_cache; using the LiDAR teacher "
              "measurement fallback (diagnostic only, E3 does not hold)")
    else:
        print(f"[world-b] measurement source: frozen-VGGT cache {args.vggt_cache}")
    frozen_ids = {id(p) for m in (world_enc, height_r, density_r, depth_r) for p in m.parameters()}

    bev = ds.bev_size
    sat_init = SatelliteInitializer(bev_height=bev, bev_width=bev, tile_size_m=ds.tile_size_m).to(device)
    xy_init = FixedXYInitializer(bev_height=bev, bev_width=bev, tile_size_m=ds.tile_size_m).to(device)
    updater = EvidenceAwareUpdater().to(device)
    meas_enc = GroundMeasurementEncoder(bev_height=bev, bev_width=bev).to(device)
    # one shared assimilation: the SAME G_t/M_t/updater/readers serve both
    # chains, so init (satellite vs XY) is the only manipulated variable (E2)
    trainable = (
        list(updater.parameters()) + list(meas_enc.parameters())
        + [p for p in sat_init.parameters() if p.requires_grad]
        + [p for p in xy_init.parameters() if p.requires_grad]
    )
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    if any(id(p) in frozen_ids for g in opt.param_groups for p in g["params"]):
        raise RuntimeError("Stage B optimizer contains a frozen interface parameter")

    iterator = iter(loader)
    t0 = time.time()
    running = 0.0
    print(f"[world-b] shared_assimilation (sat-chain + xy-chain) scenes={len(ds)} device={device}")
    for step in range(1, args.steps + 1):
        try:
            inputs, sup, blob = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, sup, blob = next(iterator)
        spec = spec_from_inputs(inputs, device=device)
        sat = inputs.satellite_bev.to(device)
        height = sup.height.to(device)
        density = sup.density.to(device)
        valid = sup.world_valid.to(device)
        chunk_support = sup.chunk_lidar_support.to(device)
        t_chunks = chunk_support.shape[1]
        scene_id = blob["scene_id"]
        if args.vggt_cache is not None and scene_id not in vggt_caches:
            vggt_caches[scene_id] = load_world_vggt_cache(
                args.vggt_cache, scene_id,
                str(blob["world_target_version"]), str(blob["world_target_hash"]),
            )
        with torch.no_grad():
            z_world = world_enc(height, density, valid)

        def _chunk_measurement(t: int, detach_latent: bool) -> GroundMeasurement:
            if args.vggt_cache is not None:
                return chunk_measurement_from_cache(
                    meas_enc,
                    vggt_caches[scene_id]["chunks"][str(t + 1)],
                    origin_xy=inputs.origin_xy.to(device),
                    resolution_m=float(spec.resolution_m),
                    z_datum_m=spec.z_datum_m.to(device),
                    chunk_index=t + 1,
                    query_fid=int(blob["chunk_table"][t]["core_fid"]),
                    detach=detach_latent,
                )
            support = chunk_support[:, t]
            latent = world_enc(height * support, density * support, support)
            return teacher_measurement(latent, support, t + 1)

        # ONE measurement stream consumed by BOTH chains: identical G_t/M_t
        prefix = []
        recent = []
        n_back = min(args.unroll, t_chunks)
        start = max(0, t_chunks - n_back)
        for t in range(start):
            meas = _chunk_measurement(t, True)
            prefix.append(meas)
        for t in range(start, t_chunks):
            # recent steps keep measurement gradients so the shared
            # GroundMeasurementEncoder trains through both chains' losses
            recent.append(_chunk_measurement(t, False))

        def _run_chain(state):
            """Assimilate one chain through the shared measurement stream."""
            loss = masked_smooth_l1(height_r(state.latent), height, valid) + masked_smooth_l1(
                density_r(state.latent), density, valid,
            )
            # masked distill per experiment_plan §6: only where the static
            # cloud labels; unmasked would pull the satellite prior in ahead
            # regions toward the teacher's "unknown" extrapolation
            loss = loss + 0.1 * masked_smooth_l1(state.latent, z_world, valid)
            with torch.no_grad():
                for meas in prefix:
                    state = updater(state, meas).state
            for meas in recent:
                prev = state
                state = updater(state, meas).state
                # supervise only where BOTH the measurement writes and the
                # static target labels; VGGT support beyond LiDAR coverage
                # holds placeholder zeros, not ground truth
                supervised = supervised_region(meas.support, valid)
                loss = loss + masked_smooth_l1(height_r(state.latent), height, supervised)
                loss = loss + masked_smooth_l1(density_r(state.latent), density, supervised)
                if meas.chunk_index > 1:
                    loss = loss + 0.1 * masked_smooth_l1(
                        height_r(state.latent), height_r(prev.latent).detach(),
                        visited_mask(chunk_support, meas.chunk_index - 1),
                    )
            return state, loss

        _, loss_sat = _run_chain(sat_init(sat, spec))
        _, loss_xy = _run_chain(xy_init(sat, spec))
        loss = loss_sat + loss_xy

        opt.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
            opt.step()
        running += float(loss.detach())
        if step == 1 or step % 20 == 0:
            print(f"step={step}/{args.steps} loss={running / (20 if step >= 20 else step):.4f} "
                  f"sat={float(loss_sat):.4f} xy={float(loss_xy):.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
            running = 0.0
        if step == args.steps or step % 100 == 0:
            torch.save({
                "schema_version": WORLD_STATE_SCHEMA_VERSION,
                "interface_fingerprint": fp,
                "satellite_initializer": sat_init.state_dict(),
                "xy_initializer": xy_init.state_dict(),
                "updater": updater.state_dict(),
                "measurement_encoder": meas_enc.state_dict(),
                "branch": "shared_assimilation",
                "measurement_source": "vggt_cache" if args.vggt_cache else "lidar_teacher_fallback",
                "step": step,
                "config": vars(args),
            }, out / "assimilation.pt")
    print(f"[world-b] checkpoint={out / 'assimilation.pt'}")


if __name__ == "__main__":
    main()
