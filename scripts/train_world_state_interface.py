#!/usr/bin/env python3
"""Stage A: fit world-defined Z and freeze height/density/depth readers."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.losses import masked_smooth_l1  # noqa: E402
from world3d.unified_bev.models import ColumnFieldDecoder, render_multi_view  # noqa: E402
from world3d.unified_bev.state_models import (  # noqa: E402
    BEVSurfaceDensityDecoder,
    BEVWorldHeightDecoder,
    WorldGeometryEncoder,
    freeze_interface,
)
from world3d.unified_bev.world_checkpoints import (  # noqa: E402
    compute_world_interface_fingerprint,
)
from world3d.unified_bev.world_data import WorldStateSceneDataset, collate_world_state  # noqa: E402
from world3d.unified_bev.world_state import (  # noqa: E402
    PROVENANCE_ENUM,
    WORLD_STATE_SCHEMA_VERSION,
    WORLD_TARGET_VERSION,
    Z_DATUM_POLICY,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="runs/world_state_interface")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--ray_samples", type=int, default=24)
    ap.add_argument("--geometry_width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    ds = WorldStateSceneDataset(args.scenes, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_world_state, num_workers=0)
    encoder = WorldGeometryEncoder().to(device)
    height_reader = BEVWorldHeightDecoder(width=args.geometry_width).to(device)
    density_reader = BEVSurfaceDensityDecoder(width=args.geometry_width).to(device)
    depth_reader = ColumnFieldDecoder(hidden=args.hidden, samples=args.ray_samples).to(device)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(height_reader.parameters())
        + list(density_reader.parameters()) + list(depth_reader.parameters()),
        lr=args.lr,
    )
    print(f"[world-a] scenes={len(ds)} bev={ds.bev_size} device={device}")
    iterator = iter(loader)
    t0 = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        try:
            inputs, sup, blob = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, sup, blob = next(iterator)
        height = sup.height.to(device)
        density = sup.density.to(device)
        valid = sup.world_valid.to(device)
        z = encoder(height, density, valid)
        h_pred = height_reader(z)
        d_pred = density_reader(z)
        loss_h = masked_smooth_l1(h_pred, height, valid)
        loss_d = masked_smooth_l1(d_pred, density, valid)
        q_rgb = sup.query_rgb.to(device)
        b, t, nq = q_rgb.shape[:3]
        k = sup.query_K.to(device).flatten(1, 2)
        tw = sup.query_T_world_cam.to(device).flatten(1, 2)
        depth_gt = sup.query_depth.to(device).flatten(1, 2)
        depth_m = sup.query_depth_mask.to(device).flatten(1, 2)
        pred_rgb, pred_depth, _ = render_multi_view(
            depth_reader, z, k, tw, inputs.origin_xy.to(device),
            tile_size_m=ds.tile_size_m, image_size=(q_rgb.shape[-1], q_rgb.shape[-2]),
        )
        loss_z = F.smooth_l1_loss(pred_depth[depth_m], depth_gt[depth_m]) if depth_m.any() else pred_depth.mean() * 0
        loss = loss_h + loss_d + 0.1 * loss_z
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += float(loss.detach())
        if step == 1 or step % 20 == 0:
            print(
                f"step={step}/{args.steps} loss={running / (20 if step >= 20 else step):.4f} "
                f"h={float(loss_h):.4f} d={float(loss_d):.4f} z={float(loss_z):.4f} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
            running = 0.0
        if step == args.steps or step % 100 == 0:
            ckpt = {
                "schema_version": WORLD_STATE_SCHEMA_VERSION,
                "world_target_version": WORLD_TARGET_VERSION,
                "z_datum_policy": Z_DATUM_POLICY,
                "scenes_manifest_hash": ds.manifest_hash,
                "encoder": encoder.state_dict(),
                "height_reader": height_reader.state_dict(),
                "density_reader": density_reader.state_dict(),
                "depth_reader": depth_reader.state_dict(),
                "encoder_config": {"latent_channels": 64, "context_blocks": 4},
                "height_reader_config": {"latent_channels": 64, "width": args.geometry_width},
                "density_reader_config": {"latent_channels": 64, "width": args.geometry_width},
                "depth_reader_config": {"latent_channels": 64, "hidden": args.hidden, "samples": args.ray_samples},
                "grid_config": {
                    "tile_size_m": ds.tile_size_m, "resolution_m": ds.resolution_m, "bev_size": ds.bev_size,
                },
                "chunk_config": {"chunking_version": ds.chunking_version},
                "provenance_enum": PROVENANCE_ENUM,
                "step": step,
            }
            ckpt["fingerprint"] = compute_world_interface_fingerprint(ckpt)
            torch.save(ckpt, out / "world_interface.pt")
    freeze_interface(encoder, (height_reader, density_reader, depth_reader))
    print(f"[world-a] checkpoint={out / 'world_interface.pt'}")


if __name__ == "__main__":
    main()
