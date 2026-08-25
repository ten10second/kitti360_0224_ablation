#!/usr/bin/env python3
"""Export a high-resolution, vehicle-motion-aligned VGGT RGB point map."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/media/shizhm/Lenovo/vggt")

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from world3d.unified_bev.data import UnifiedBEVDataset, VIEW_CAMERA_IDS
from scripts.build_vggt_street_cache import estimate_motion_metric_scale, vggt_confidence_score


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def voxel_first(points: np.ndarray, voxel_m: float) -> np.ndarray:
    if voxel_m <= 0 or len(points) == 0:
        return np.arange(len(points))
    quantized = np.floor((points - points.min(axis=0)) / voxel_m).astype(np.int64)
    extents = quantized.max(axis=0) + 1
    keys = (quantized[:, 0] * extents[1] + quantized[:, 1]) * extents[2] + quantized[:, 2]
    return np.sort(np.unique(keys, return_index=True)[1])


def quantile_sample(value: torch.Tensor, max_values: int = 4_000_000) -> torch.Tensor:
    flat = value.flatten()
    stride = max(1, (flat.numel() + max_values - 1) // max_values)
    return flat[::stride]


def save_preview(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    vehicle_centers: np.ndarray,
    max_points: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    if len(points) > max_points:
        keep = rng.choice(len(points), max_points, replace=False)
        points, colors = points[keep], colors[keep]
    center = vehicle_centers.mean(axis=0)
    local = points - center
    trajectory = vehicle_centers - center
    color = colors.astype(np.float32) / 255.0

    fig = plt.figure(figsize=(18, 6), dpi=160, facecolor="white")
    ax3d = fig.add_subplot(131, projection="3d")
    ax3d.scatter(local[:, 0], local[:, 1], local[:, 2], c=color, s=0.12, linewidths=0)
    ax3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="red", linewidth=2)
    ax3d.set_title("metric RGB point map (oblique)")
    ax3d.set_xlabel("east offset [m]")
    ax3d.set_ylabel("north offset [m]")
    ax3d.set_zlabel("height offset [m]")
    ax3d.view_init(elev=24, azim=-55)

    top = fig.add_subplot(132)
    top.scatter(local[:, 0], local[:, 1], c=color, s=0.12, linewidths=0)
    top.plot(trajectory[:, 0], trajectory[:, 1], color="red", linewidth=2)
    top.set_aspect("equal", adjustable="box")
    top.set_title("top-down RGB points + vehicle trajectory")
    top.set_xlabel("east offset [m]")
    top.set_ylabel("north offset [m]")

    side = fig.add_subplot(133)
    side.scatter(local[:, 1], local[:, 2], c=color, s=0.12, linewidths=0)
    side.plot(trajectory[:, 1], trajectory[:, 2], color="red", linewidth=2)
    side.set_title("north-height projection")
    side.set_xlabel("north offset [m]")
    side.set_ylabel("height offset [m]")
    for axis in (top, side):
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def contiguous_source_rig(
    dataset: UnifiedBEVDataset,
    sample_index: int,
    frame_count: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Load an ordered, overlapping trajectory window with the full 8-view rig."""
    target, _ = dataset.samples[sample_index]
    records = dataset._records_by_drive[target.drive]
    target_xy = target.T_world_imu[:2, 3]
    candidates = [
        record for record in records
        if record.fid > target.fid
        and np.linalg.norm(record.T_world_imu[:2, 3] - target_xy) >= dataset.min_source_distance_m
    ]
    selected = candidates[::stride][:frame_count]
    if len(selected) != frame_count:
        raise RuntimeError(
            f"trajectory window has only {len(selected)} frames for count={frame_count}, stride={stride}"
        )
    views = []
    poses = []
    vehicle_poses = []
    for record in selected:
        items = dataset._front_views(record) + dataset._virtual_views(record)
        if len(items) != dataset.views_per_frame:
            raise RuntimeError(f"frame {record.fid} has an incomplete 8-view rig")
        views.extend(item[0] for item in items)
        poses.extend(item[2] for item in items)
        vehicle_poses.append(torch.from_numpy(record.T_world_imu.astype(np.float32)))
    return (
        torch.stack(views), torch.stack(poses), torch.stack(vehicle_poses),
        [record.fid for record in selected],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="/home/shizhm/Downloads/vggt.pt")
    parser.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl")
    parser.add_argument("--drive", default="2013_05_28_drive_0000_sync")
    parser.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    parser.add_argument("--out", default="runs/vggt_metric_pointmap")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--dense_sources", type=int, default=8)
    parser.add_argument("--trajectory_frames", type=int, default=0,
                        help="ordered trajectory frames; 0 keeps the dataset's farthest-point source set")
    parser.add_argument("--trajectory_stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=350)
    parser.add_argument("--confidence_percentile", type=float, default=55.0)
    parser.add_argument("--radius_m", type=float, default=70.0)
    parser.add_argument("--voxel_m", type=float, default=0.08)
    parser.add_argument("--preview_points", type=int, default=250000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.width % 14 or args.height % 14:
        raise ValueError("VGGT width and height must be divisible by patch size 14")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, drive=args.drive,
        max_samples=args.sample_index + 1, dense_source_count=args.dense_sources,
        sparse_source_count=args.dense_sources, image_size=(args.width, args.height),
        max_points_per_view=64,
    )
    sample = dataset[args.sample_index]
    if args.trajectory_frames > 0:
        source_rgb, source_T_world_cam, source_T_world_imu, source_fids = contiguous_source_rig(
            dataset, args.sample_index, args.trajectory_frames, args.trajectory_stride,
        )
    else:
        source_rgb = sample["source_rgb"]
        source_T_world_cam = sample["source_T_world_cam"]
        source_T_world_imu = sample["source_T_world_imu"]
        source_fids = [int(value) for value in sample["meta"]["source_fids"]]
    images = source_rgb.to(device)

    model = VGGT(enable_point=True, enable_depth=True, enable_track=False).to(device)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state.get("model", state), strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"VGGT point-map weights are incomplete: {incompatible.missing_keys[:8]}")
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith("track_head.")
    ]
    if unexpected:
        raise RuntimeError(f"unexpected VGGT weights: {unexpected[:8]}")
    del state
    model.eval()

    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype):
        predictions = model(images[None])
    predicted_w2c, predicted_K = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:],
    )
    motion = estimate_motion_metric_scale(
        predicted_w2c[0].float().cpu(), source_T_world_cam,
        views_per_frame=dataset.views_per_frame,
        gt_world_vehicle=source_T_world_imu,
        view_camera_ids=torch.tensor(VIEW_CAMERA_IDS),
    )

    point_map = predictions["world_points"][0].float()
    point_confidence = predictions["world_points_conf"][0].float()
    depth_confidence = predictions["depth_conf"][0].float()
    confidence = vggt_confidence_score(depth_confidence)
    rotation = motion["alignment_rotation"].to(device)
    translation = motion["alignment_translation_m"].to(device)
    metric_points = point_map * motion["scale"].to(device)
    metric_points = metric_points @ rotation.T + translation
    colors = (images.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().to(torch.uint8)

    vehicle_centers = source_T_world_imu[:, :3, 3].to(device)
    scene_center = vehicle_centers.mean(dim=0)
    point_confidence_sample = quantile_sample(point_confidence)
    confidence_sample = quantile_sample(confidence)
    point_confidence_quantiles = torch.quantile(
        point_confidence_sample,
        point_confidence.new_tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]),
    )
    confidence_quantiles = torch.quantile(
        confidence_sample,
        confidence.new_tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]),
    )
    threshold = torch.quantile(confidence_sample, args.confidence_percentile / 100.0)
    delta = metric_points - scene_center
    valid = (
        torch.isfinite(metric_points).all(dim=-1)
        & (confidence >= threshold)
        & (delta[..., :2].norm(dim=-1) <= args.radius_m)
        & (delta[..., 2] >= -8.0)
        & (delta[..., 2] <= 25.0)
    )
    points_np = metric_points[valid].cpu().numpy().astype(np.float32)
    colors_np = colors[valid].cpu().numpy()
    confidence_np = confidence[valid].cpu().numpy().astype(np.float16)
    before_voxel = len(points_np)
    keep = voxel_first(points_np, args.voxel_m)
    points_np, colors_np, confidence_np = points_np[keep], colors_np[keep], confidence_np[keep]
    vehicle_np = vehicle_centers.cpu().numpy().astype(np.float32)
    predicted_w2c_cpu = predicted_w2c[0].float().cpu()
    predicted_rotation_c2w = predicted_w2c_cpu[:, :3, :3].transpose(-1, -2)
    predicted_centers = -(
        predicted_rotation_c2w @ predicted_w2c_cpu[:, :3, 3, None]
    ).squeeze(-1)
    camera_T_world = torch.eye(4).repeat(predicted_w2c_cpu.shape[0], 1, 1)
    camera_T_world[:, :3, :3] = motion["alignment_rotation"] @ predicted_rotation_c2w
    camera_T_world[:, :3, 3] = (
        predicted_centers * motion["scale"] @ motion["alignment_rotation"].T
        + motion["alignment_translation_m"]
    )

    ply_path = output / "metric_rgb_pointmap.ply"
    npz_path = output / "metric_rgb_pointmap.npz"
    preview_path = output / "metric_rgb_pointmap_preview.png"
    write_binary_ply(ply_path, points_np, colors_np)
    np.savez_compressed(
        npz_path, points=points_np, colors=colors_np, confidence=confidence_np,
        vehicle_centers=vehicle_np, camera_T_world=camera_T_world.numpy().astype(np.float32),
        camera_K=predicted_K[0].float().cpu().numpy().astype(np.float32),
        view_camera_ids=np.tile(np.asarray(VIEW_CAMERA_IDS, dtype=np.int16), len(source_fids)),
    )
    save_preview(preview_path, points_np, colors_np, vehicle_np, args.preview_points)

    metadata = {
        "drive": sample["meta"]["drive"],
        "target_fid_unused": int(sample["meta"]["target_fid"]),
        "source_fids": source_fids,
        "source_frames": len(source_fids),
        "source_views": int(images.shape[0]),
        "input_hw": [int(images.shape[-2]), int(images.shape[-1])],
        "raw_point_count": int(point_map.numel() // 3),
        "confidence_kept_before_voxel": int(before_voxel),
        "point_head_confidence_quantiles": [float(value) for value in point_confidence_quantiles],
        "filter_confidence_source": "depth_head_robust_score",
        "filter_confidence_quantiles": [float(value) for value in confidence_quantiles],
        "filter_confidence_threshold": float(threshold),
        "voxel_point_count": int(len(points_np)),
        "voxel_m": args.voxel_m,
        "metric_scale": float(motion["scale"]),
        "scale_source": motion["scale_source"],
        "scale_pair_count": int(motion["pair_count"]),
        "scale_relative_mad": float(motion["relative_mad"]),
        "pose_alignment_rmse_m": float(motion["alignment_rmse_m"]),
        "ply": str(ply_path),
        "npz": str(npz_path),
        "preview": str(preview_path),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata)


if __name__ == "__main__":
    main()
