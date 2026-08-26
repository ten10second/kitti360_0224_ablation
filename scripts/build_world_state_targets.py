#!/usr/bin/env python3
"""Build scene-centered world-state tiles, LiDAR targets, and QA overlays."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.data import (  # noqa: E402
    FRONT_CROP_OVERLAP,
    FRONT_CROP_WIDTH,
    UnifiedBEVDataset,
    _open_image,
    load_frame_records,
)
from world3d.unified_bev.world_data import (  # noqa: E402
    RESOLUTION_M,
    TILE_SIZE_M,
    build_scene_blob,
    propose_scenes,
)
from world3d.unified_bev.world_targets import satellite_mapping_error_px  # noqa: E402


def _view_helper(image_size=(160, 96)):
    helper = UnifiedBEVDataset.__new__(UnifiedBEVDataset)
    helper.image_size = image_size
    helper.front_crop_width = FRONT_CROP_WIDTH
    helper.front_crop_overlap = FRONT_CROP_OVERLAP
    helper.max_points_per_view = 4096
    helper.use_fisheye = True
    helper._rigs = {}
    return helper


def _save_qa(blob: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height = blob["height"].squeeze().numpy()
    density = blob["density"].squeeze().numpy()
    valid = blob["world_valid"].squeeze().numpy()
    support = blob["chunk_lidar_support"].numpy()
    if support.ndim == 4:
        support = support[:, 0]
    sat = blob["satellite_bev"].permute(1, 2, 0).numpy()
    visited = support.any(axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(np.clip(sat, 0, 1))
    axes[0, 0].set_title("satellite BEV (south-up)")
    axes[0, 1].imshow(np.where(valid, height, np.nan), origin="lower")
    axes[0, 1].set_title("height p90 - datum")
    axes[0, 2].imshow(np.where(valid, density, np.nan), origin="lower")
    axes[0, 2].set_title("log density")
    axes[1, 0].imshow(visited, origin="lower")
    axes[1, 0].set_title("final ground support")
    colors = np.zeros((*visited.shape, 3), dtype=np.float32)
    cmap = plt.cm.tab10
    for t, mask in enumerate(support):
        colors[mask] = cmap(t % 10)[:3]
    axes[1, 1].imshow(colors, origin="lower")
    axes[1, 1].set_title("chunk colors")
    axes[1, 2].imshow(valid & ~visited, origin="lower")
    axes[1, 2].set_title("off-route diagnostic")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--out", default="runs/world_state_targets")
    ap.add_argument("--max_scenes", type=int, default=8)
    ap.add_argument("--max_chunks", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qa_dir = out / "qa"
    qa_dir.mkdir(exist_ok=True)

    rigs = {}
    drive = None if args.drive in (None, "none", "") else args.drive
    records_by_drive = load_frame_records(
        args.manifest, args.lidar_root, drive, rigs=rigs,
    )
    helper = _view_helper()
    helper._rigs = rigs
    manifest_rows = []
    built = 0
    skipped = 0
    for drive, records in records_by_drive.items():
        proposals = propose_scenes(records, max_chunks=args.max_chunks, max_scenes=args.max_scenes - built)
        for prop in proposals:
            try:
                blob = build_scene_blob(prop, helper, split=args.split)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"[skip] {drive} fid={prop['anchor_fid']}: {exc}")
                continue
            err = satellite_mapping_error_px(
                blob["sat_center_xy"].view(1, 2), blob["origin_xy"].view(1, 2),
            )
            if float(err.max()) > 1e-4:
                skipped += 1
                print(f"[skip] mapping error {float(err.max()):.3e} px")
                continue
            fname = f"{blob['scene_id']}.pt"
            torch.save(blob, out / fname)
            _save_qa(blob, qa_dir / f"{blob['scene_id']}.png")
            manifest_rows.append({
                "scene_id": blob["scene_id"],
                "split": args.split,
                "drive": blob["drive"],
                "file": fname,
                "origin_xy": blob["origin_xy"].tolist(),
                "sat_center_xy": blob["sat_center_xy"].tolist(),
                "z_datum_m": float(blob["z_datum_m"].reshape(-1)[0]),
                "world_target_hash": blob["world_target_hash"],
                "n_chunks": len(blob["chunk_table"]),
                "mapping_error_px": float(err.max()),
            })
            built += 1
            print(f"[scene] {blob['scene_id']} chunks={len(blob['chunk_table'])} "
                  f"valid={int(blob['world_valid'].sum())} datum={float(blob['z_datum_m']):.2f}")
            if built >= args.max_scenes:
                break
        if built >= args.max_scenes:
            break
    (out / "scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest_rows))
    print(f"[done] built={built} skipped={skipped} out={out}")
    if built == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
