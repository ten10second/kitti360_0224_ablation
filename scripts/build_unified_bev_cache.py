#!/usr/bin/env python3
"""Build a RAM-friendly cache of UnifiedBEVDataset samples.

Each training sample is read from the external disk exactly once, converted to
a compact storage form (images as uint8, lossless), and saved to a single file.
Training then serves batches from RAM via load_cached_unified_bev.

Verifies bitwise equality against a freshly constructed dataset on a few
indices before declaring success.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world3d.unified_bev.data import UnifiedBEVDataset, _sample_to_storage

_DS = None


def _init(args):
    global _DS
    _DS = UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, drive=args.drive,
        min_target_spacing_m=args.min_target_spacing_m, max_samples=args.max_samples,
        dense_source_count=args.dense_sources, sparse_source_count=2,
        image_size=(args.image_width, args.image_height),
        max_points_per_view=args.max_points,
    )


def _fetch(i: int) -> dict:
    return _sample_to_storage(_DS[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--max_samples", type=int, default=2048)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--image_width", type=int, default=160)
    ap.add_argument("--image_height", type=int, default=96)
    ap.add_argument("--max_points", type=int, default=4096)
    ap.add_argument("--min_target_spacing_m", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    ds = UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, drive=args.drive,
        min_target_spacing_m=args.min_target_spacing_m, max_samples=args.max_samples,
        dense_source_count=args.dense_sources, sparse_source_count=2,
        image_size=(args.image_width, args.image_height),
        max_points_per_view=args.max_points,
    )
    n = len(ds)
    print(f"[cache] dataset samples={n}", flush=True)
    attrs = {
        "bev_size": ds.bev_size, "bev_resolution_m": ds.bev_resolution_m,
        "tile_size_m": ds.tile_size_m, "views_per_frame": ds.views_per_frame,
        "image_size": ds.image_size,
    }

    samples = [None] * n
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args,)) as pool:
        for i, s in enumerate(pool.map(_fetch, range(n), chunksize=4)):
            samples[i] = s
            if (i + 1) % 256 == 0:
                print(f"[cache] {i + 1}/{n} elapsed={time.time() - t0:.0f}s", flush=True)

    torch.save({"attrs": attrs, "samples": samples}, args.out)
    size_gb = Path(args.out).stat().st_size / 1e9
    print(f"[cache] saved {args.out} ({size_gb:.2f} GB) in {time.time() - t0:.0f}s", flush=True)

    # Verify bitwise against a fresh dataset pass.
    _init(args)
    for i in (0, n // 2, n - 1):
        fresh = _DS[i]
        cached = samples[i]
        for key, value in fresh.items():
            if key in ("target_rgb", "source_rgb", "satellite"):
                assert torch.equal(value, cached[key].to(torch.float32) / 255.0), key
            elif key == "meta":
                assert value == cached[key], key  # plain dict, not a tensor
            else:
                assert torch.equal(value, cached[key]), key
    print("[cache] verify PASS (bitwise)", flush=True)


if __name__ == "__main__":
    main()
