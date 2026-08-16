import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure repo root is on PYTHONPATH
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from world3d.io.kitti360d_dataloader import Kitti360dDataset


def to_uint8_rgb(img_t):
    """img_t: torch (3,H,W) in [0,1]"""
    import torch

    if isinstance(img_t, torch.Tensor):
        x = img_t.detach().cpu().clamp(0, 1).numpy()
    else:
        x = np.asarray(img_t)
    x = (x * 255.0).round().astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    return x


def hstack_with_padding(images, pad=10, pad_color=0):
    hs = [im.shape[0] for im in images]
    ws = [im.shape[1] for im in images]
    H = max(hs)
    out_w = sum(ws) + pad * (len(images) - 1)
    canvas = np.full((H, out_w, 3), pad_color, dtype=np.uint8)
    x0 = 0
    for im in images:
        h, w = im.shape[:2]
        y0 = (H - h) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = im
        x0 += w + pad
    return canvas


def main():
    drive_dir = repo_root / "2013_05_28_drive_0003_sync"
    frame_id = 0

    # Front sample
    ds_front = Kitti360dDataset(
        drives=drive_dir,
        mode="front",
        front_resize=(640, 256),
    )
    s_front = ds_front[frame_id]

    # Virtual views: use image_02 for left yaws, image_03 for right yaws
    # Option A: [-60, -30, +30, +60]
    views = [
        ("image_02", -45.0),
        ("image_02", -90.0),
        ("image_03", 45.0),
        ("image_03", 90.0),
    ]

    virtual_hfov = 80.0  # slightly narrower to reduce black invalid regions
    virtual_size = (640, 256)

    virtual_rgbs = []
    for cam, yaw in views:
        ds_v = Kitti360dDataset(
            drives=drive_dir,
            mode="fisheye_virtual",
            fisheye_camera=cam,
            vehicle_relative_yaw_deg=yaw,
            virtual_hfov_deg=virtual_hfov,
            virtual_size=virtual_size,
        )
        s_v = ds_v[frame_id]
        virtual_rgbs.append(to_uint8_rgb(s_v["image"]))

    front_rgb = to_uint8_rgb(s_front["image"])
    sat_rgb = to_uint8_rgb(s_front["sat"])  # same for all

    grid = hstack_with_padding([front_rgb, *virtual_rgbs, sat_rgb], pad=16, pad_color=20)

    out_path = repo_root / "tools" / "_vis_kitti360d_sample_frame0000000000_A.png"
    Image.fromarray(grid).save(out_path)

    print("Saved visualization to:", out_path)
    print("Front image shape:", front_rgb.shape)
    for (cam, yaw), rgb in zip(views, virtual_rgbs):
        print(f"Virtual shape: {rgb.shape} ({cam}, yaw={yaw:+.0f}, hfov={virtual_hfov})")
    print("Satellite shape:", sat_rgb.shape)


if __name__ == "__main__":
    main()

