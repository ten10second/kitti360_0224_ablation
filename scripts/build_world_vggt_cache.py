#!/usr/bin/env python3
"""Build per-chunk frozen-VGGT measurement caches for world-state scenes.

For every scene in a targets directory, each chunk's ``geometry_fids``
(<= 8 frames) are expanded into the calibrated 8-view rig
(front2_left3_right3_v1) and sent through ONE independent joint VGGT forward.
Depth/confidence come back vehicle-motion-scaled at view resolution; the
calibrated intrinsics and world poses are stored alongside so training can
unproject and splat without ever re-running VGGT.

The cache is bound to the scene blob's world-target version and hash, and to
the chunk's exact frame list: any mismatch fails fast at load time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/media/shizhm/Lenovo/vggt")

from vggt.models.vggt import VGGT  # noqa: E402
from scripts.build_vggt_street_cache import run_joint_subset  # noqa: E402
from world3d.unified_bev.data import (  # noqa: E402
    FRONT_CROP_OVERLAP,
    FRONT_CROP_WIDTH,
    VIEW_LAYOUT_VERSION,
    UnifiedBEVDataset,
    load_frame_records,
)
from world3d.unified_bev.world_vggt import (  # noqa: E402
    WORLD_VGGT_CACHE_VERSION,
    assert_query_isolation,
)


def _view_helper(rigs):
    helper = UnifiedBEVDataset.__new__(UnifiedBEVDataset)
    helper.image_size = (160, 96)
    helper.front_crop_width = FRONT_CROP_WIDTH
    helper.front_crop_overlap = FRONT_CROP_OVERLAP
    helper.max_points_per_view = 4096
    helper.use_fisheye = True
    helper.fisheye_yaws_deg = (-45.0, 0.0, 45.0)
    helper.view_layout_version = VIEW_LAYOUT_VERSION
    helper.view_camera_ids = (0, 0) + (1, 1, 1, 2, 2, 2)
    helper.views_per_frame = 2 + len(helper.fisheye_yaws_deg) * 2
    helper._rigs = rigs
    return helper


def _chunk_views(helper: UnifiedBEVDataset, recs, fids):
    """Frame-major 8-view rig for one chunk: front2 then fisheye6 per frame."""
    rgb, K, T, imu = [], [], [], []
    for fid in fids:
        rec = recs[fid]
        items = helper._front_views(rec)
        if helper.use_fisheye:
            items = items + helper._virtual_views(rec)
        if len(items) != helper.views_per_frame:
            raise RuntimeError(
                f"chunk frame {fid} produced {len(items)} views; "
                f"expected {helper.views_per_frame}"
            )
        for rgb_v, K_v, T_v, _ in items:
            rgb.append(rgb_v)
            K.append(K_v)
            T.append(T_v)
        imu.append(torch.from_numpy(rec.T_world_imu.astype(np.float32)))
    return torch.stack(rgb), torch.stack(K), torch.stack(T), torch.stack(imu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="targets dir with scenes.jsonl")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/home/shizhm/Downloads/vggt.pt")
    ap.add_argument("--resolution", type=int, default=518)
    ap.add_argument("--min_baseline_m", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = VGGT(enable_point=False, enable_track=False).to(device)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state.get("model", state), strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"VGGT weights incomplete: {incompatible.missing_keys[:8]}")
    del state
    model.eval()

    scenes = [json.loads(l) for l in (Path(args.scenes) / "scenes.jsonl").read_text().splitlines() if l.strip()]
    rigs: dict = {}
    recs_by_drive: dict = {}
    t0 = time.time()
    built = 0
    for row in scenes:
        blob = torch.load(Path(args.scenes) / row["file"], map_location="cpu", weights_only=False)
        scene_id = row["scene_id"]
        dst = out_dir / f"{scene_id}.pt"
        header = {
            "schema": WORLD_VGGT_CACHE_VERSION,
            "scene_id": scene_id,
            "drive": row["drive"],
            "world_target_version": blob["world_target_version"],
            "world_target_hash": blob["world_target_hash"],
            "view_layout_version": VIEW_LAYOUT_VERSION,
            "vggt_resolution": int(args.resolution),
        }
        if dst.exists():
            existing = torch.load(dst, map_location="cpu", weights_only=False)
            stale = [k for k, v in header.items() if existing.get(k) != v]
            if stale:
                raise RuntimeError(f"{dst} is stale on {stale}; use a fresh output directory")
            have = {int(k) for k in existing.get("chunks", {})}
            need = {int(c["chunk_index"]) for c in blob["chunk_table"]}
            if need <= have:
                print(f"[world-vggt] {scene_id} already complete ({len(have)} chunks)", flush=True)
                continue
            chunks = existing["chunks"]
        else:
            chunks = {}
        if row["drive"] not in recs_by_drive:
            records = load_frame_records(args.manifest, args.lidar_root, row["drive"], rigs=rigs)
            recs_by_drive[row["drive"]] = {r.fid: r for r in records[row["drive"]]}
        recs = recs_by_drive[row["drive"]]
        helper = _view_helper(rigs)
        for chunk in blob["chunk_table"]:
            index = int(chunk["chunk_index"])
            if index in {int(k) for k in chunks}:
                continue
            # strict isolation: the depth-query core frame must not be among
            # the measurement frames; backfill from the same chunk's unused
            # frames (arc-nearest first) to keep the VGGT baseline width
            query_fid = int(chunk["core_fid"])
            base = [int(f) for f in chunk["geometry_fids"]]
            member_order = {int(f): i for i, f in enumerate(chunk["fids"])}
            measurement_fids = [f for f in base if f != query_fid]
            spare = [f for f in chunk["fids"] if f not in set(base) and f != query_fid]
            spare.sort(key=lambda f: abs(member_order[f] - member_order[query_fid]))
            take = min(len(spare), len(base) - len(measurement_fids))
            measurement_fids = sorted(measurement_fids + spare[:take])
            assert_query_isolation(
                {"query_fid": query_fid, "measurement_fids": measurement_fids}, query_fid,
            )
            fids = measurement_fids
            rgb, K, T_world_cam, T_world_imu = _chunk_views(helper, recs, fids)
            entry = run_joint_subset(
                model, rgb, T_world_cam, args.resolution, args.min_baseline_m,
                helper.views_per_frame, device,
                gt_world_vehicle=T_world_imu,
                view_camera_ids=torch.tensor(helper.view_camera_ids),
            )
            chunks[index] = {
                "fids": [int(f) for f in fids],
                "measurement_fids": [int(f) for f in measurement_fids],
                "query_fid": query_fid,
                "rgb": rgb.to(torch.float16),
                "K": K,
                "T_world_cam": T_world_cam,
                "T_world_imu": T_world_imu,
                "depth": entry["depth"],
                "conf": entry["conf"],
                "metric_scale": entry["metric_scale"],
                "scale_pair_count": entry["scale_pair_count"],
                "scale_relative_mad": entry["scale_relative_mad"],
                "pose_alignment_rmse_m": entry["pose_alignment_rmse_m"],
                "scale_source": entry["scale_source"],
                "scale_reliability": entry["scale_reliability"],
            }
            print(
                f"[world-vggt] {scene_id} chunk={index} frames={len(fids)} "
                f"(query {query_fid} excluded) views={rgb.shape[0]} "
                f"scale={float(entry['metric_scale']):.4f} source={entry['scale_source']} "
                f"mad={float(entry['scale_relative_mad']):.3f} "
                f"pose_rmse={float(entry['pose_alignment_rmse_m']):.3f}m "
                f"conf>0.3={float((entry['conf'] > 0.3).float().mean()):.3f}",
                flush=True,
            )
            payload = dict(header)
            payload["chunks"] = {str(k): v for k, v in sorted(chunks.items())}
            tmp = dst.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            tmp.replace(dst)
        built += 1
        print(f"[world-vggt] scene {scene_id} cached ({len(chunks)} chunks, "
              f"{(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[world-vggt] DONE scenes={built} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
