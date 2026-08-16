
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

# Ensure repo root is on PYTHONPATH
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from world3d.io.kitti360d_dataloader import Kitti360dDataset


def main():
    # Adjust this path if needed
    repo_root = Path(__file__).resolve().parents[1]
    drive_dir = repo_root / "2013_05_28_drive_0003_sync"

    assert drive_dir.exists(), f"Drive dir not found: {drive_dir}"

    # Front view example (image_00), resized to 960x256
    ds_front = Kitti360dDataset(
        drives=drive_dir,
        mode="front",
        front_resize=(640, 256),
        front_center_crop=None,
        return_bgr=False,
    )

    # Fisheye virtual example (image_02 -> virtual perspective)
    ds_virtual = Kitti360dDataset(
        drives=drive_dir,
        mode="fisheye_virtual",
        fisheye_camera="image_02",
        vehicle_relative_yaw_deg=30.0,
        virtual_hfov_deg=100.0,
        virtual_size=(640, 256),
        return_bgr=False,
    )

    for name, ds in [("front", ds_front), ("virtual", ds_virtual)]:
        print("=" * 80)
        print(f"Dataset: {name}")
        print(f"Num samples: {len(ds)}")

        sample = ds[0]
        print("Keys:", list(sample.keys()))

        image = sample["image"]
        sat = sample["sat"]
        K = sample["K"]
        T_pose_cam = sample["T_pose_cam"]

        print(f"image: shape={tuple(image.shape)} dtype={image.dtype} range=[{image.min().item():.4f},{image.max().item():.4f}]")
        print(f"sat:   shape={tuple(sat.shape)} dtype={sat.dtype} range=[{sat.min().item():.4f},{sat.max().item():.4f}]")
        print(f"K:     shape={None if K is None else tuple(K.shape)}")
        if K is not None:
            print(K)
        print(f"T_pose_cam: shape={None if T_pose_cam is None else tuple(T_pose_cam.shape)}")
        if T_pose_cam is not None:
            print(T_pose_cam)

        print("sat_m_per_px:", sample["sat_m_per_px"])
        print("frame_id:", sample["frame_id"], "drive:", sample["drive"])
        print("meta:")
        for k, v in sample["meta"].items():
            print(f"  {k}: {v}")

        def minimal_collate(batch_list):
            out = {}
            out["image"] = torch.stack([b["image"] for b in batch_list], dim=0)
            out["sat"] = torch.stack([b["sat"] for b in batch_list], dim=0)
            out["K"] = torch.stack([b["K"] for b in batch_list], dim=0) if batch_list[0]["K"] is not None else None
            out["T_pose_cam"] = torch.stack([b["T_pose_cam"] for b in batch_list], dim=0) if batch_list[0]["T_pose_cam"] is not None else None
            out["frame_id"] = torch.tensor([b["frame_id"] for b in batch_list], dtype=torch.long)
            out["drive"] = [b["drive"] for b in batch_list]
            return out

        dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=minimal_collate)
        batch = next(iter(dl))
        print("\nBatched keys:", list(batch.keys()))
        print("batch['image'].shape:", tuple(batch["image"].shape))
        print("batch['sat'].shape:", tuple(batch["sat"].shape))
        print("batch['K'].shape:", None if batch["K"] is None else tuple(batch["K"].shape))
        print("batch['T_pose_cam'].shape:", None if batch["T_pose_cam"] is None else tuple(batch["T_pose_cam"].shape))
        print("batch['frame_id'].shape:", tuple(batch["frame_id"].shape))


if __name__ == "__main__":
    main()

