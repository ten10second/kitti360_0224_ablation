#!/usr/bin/env python3
"""Build joint-view, motion-metric VGGT geometry caches for unified BEV.

Each requested source subset is sent through VGGT independently. VGGT jointly
predicts depth and camera poses in one shared (scale-ambiguous) gauge. Known
KITTI-360 camera geometry supplies one metric scale per subset: vehicle motion
for multi-frame subsets and the calibrated camera rig for a single frame.
Scaled depth is then consumed by the dense lift together with the calibrated
KITTI intrinsics and world poses.

The cache stores subset-specific predictions so sparse Stage-B/C2 evaluation
never slices depth that was inferred while VGGT could attend to held-out views.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/media/shizhm/Lenovo/vggt")

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from world3d.unified_bev.data import (
    ChunkedUnifiedBEVDataset,
    UnifiedBEVDataset,
    VIEW_LAYOUT_VERSION,
    geometry_sample_identity,
    geometry_scale_reliability,
    load_cached_unified_bev,
    validate_geometry_blob_identity,
)


def subset_key(start_frame: int, frame_count: int) -> str:
    return f"s{start_frame}_n{frame_count}"


def parse_subset_specs(text: str, total_frames: int) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    for item in text.split(","):
        start, count = (int(x) for x in item.split(":"))
        if start < 0 or count <= 0 or start + count > total_frames:
            raise ValueError(f"invalid subset {item!r} for {total_frames} frames")
        pair = (start, count)
        if pair not in specs:
            specs.append(pair)
    return specs


def camera_centers_from_w2c(extrinsics: torch.Tensor) -> torch.Tensor:
    """OpenCV world-to-camera matrices (...,3,4) -> camera centers (...,3)."""
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    return -(R.transpose(-1, -2) @ t[..., None]).squeeze(-1)


def _unique_center_indices(centers: torch.Tensor, tolerance_m: float = 1e-3) -> torch.Tensor:
    keep: list[int] = []
    for i in range(centers.shape[0]):
        if not keep or all(float((centers[i] - centers[j]).norm()) > tolerance_m for j in keep):
            keep.append(i)
    return torch.tensor(keep, dtype=torch.long, device=centers.device)


def estimate_motion_metric_scale(
    predicted_w2c: torch.Tensor,
    gt_world_cam: torch.Tensor,
    min_baseline_m: float = 0.25,
    views_per_frame: int | None = None,
    gt_world_vehicle: torch.Tensor | None = None,
    view_camera_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int | str]:
    """Estimate one metric scale from corresponding predicted/known motion.

    Pairwise camera-center distances remove the unknown rotation/translation
    gauge. The median GT/predicted baseline ratio is robust to individual pose
    errors. A fixed-scale Kabsch alignment supplies a residual QA metric.
    With multiple source frames, repeated virtual views are first averaged per
    physical camera and the resulting three-camera centroid is paired with the
    measured vehicle pose.  Thus the metric side of the scale fit is the actual
    vehicle displacement, independent of the number of virtual crops.  A
    one-frame subset has no vehicle motion and explicitly falls back to the
    calibrated multi-camera rig baseline.
    """
    pred = camera_centers_from_w2c(predicted_w2c.float())
    gt = gt_world_cam[..., :3, 3].float()
    scale_source = "camera_rig"
    if views_per_frame is not None:
        if views_per_frame <= 0 or pred.shape[0] % views_per_frame:
            raise ValueError(
                f"{pred.shape[0]} views cannot be grouped by views_per_frame={views_per_frame}"
            )
        frame_count = pred.shape[0] // views_per_frame
        if frame_count >= 2:
            if gt_world_vehicle is None:
                raise ValueError("multi-frame VGGT scale requires explicit vehicle poses")
            if view_camera_ids is None or int(view_camera_ids.numel()) != views_per_frame:
                raise ValueError("multi-frame VGGT scale requires one physical-camera id per view")
            grouped = pred.view(frame_count, views_per_frame, 3)
            camera_ids = view_camera_ids.to(device=pred.device).reshape(-1)
            physical_centers = [
                grouped[:, camera_ids == camera_id].mean(dim=1)
                for camera_id in torch.unique(camera_ids, sorted=True)
            ]
            pred = torch.stack(physical_centers, dim=1).mean(dim=1)
            vehicle = gt_world_vehicle.float().to(pred.device)
            if vehicle.ndim == 3 and vehicle.shape[-2:] == (4, 4):
                gt = vehicle[:, :3, 3]
            elif vehicle.ndim == 2 and vehicle.shape[-1] == 3:
                gt = vehicle
            else:
                raise ValueError("vehicle poses must have shape (F,4,4) or centers (F,3)")
            if gt.shape[0] != frame_count:
                raise ValueError(
                    f"vehicle pose count {gt.shape[0]} does not match frame count {frame_count}"
                )
            keep = torch.arange(frame_count, dtype=torch.long, device=pred.device)
            scale_source = "vehicle_motion"
        else:
            if view_camera_ids is not None:
                if int(view_camera_ids.numel()) != views_per_frame:
                    raise ValueError("single-frame VGGT scale received invalid physical-camera ids")
                camera_ids = view_camera_ids.to(device=pred.device).reshape(-1)
                pred = torch.stack([
                    pred[camera_ids == camera_id].mean(dim=0)
                    for camera_id in torch.unique(camera_ids, sorted=True)
                ])
                gt = torch.stack([
                    gt[camera_ids == camera_id].mean(dim=0)
                    for camera_id in torch.unique(camera_ids, sorted=True)
                ])
                keep = torch.arange(pred.shape[0], dtype=torch.long, device=pred.device)
            else:
                keep = _unique_center_indices(gt)
                pred, gt = pred[keep], gt[keep]
    else:
        keep = _unique_center_indices(gt)
        pred, gt = pred[keep], gt[keep]
    if pred.shape[0] < 2:
        raise ValueError("metric scale needs at least two distinct camera centers")

    pairs = torch.triu_indices(pred.shape[0], pred.shape[0], offset=1, device=pred.device)
    pred_dist = (pred[pairs[0]] - pred[pairs[1]]).norm(dim=-1)
    gt_dist = (gt[pairs[0]] - gt[pairs[1]]).norm(dim=-1)
    valid = (gt_dist >= float(min_baseline_m)) & (pred_dist > 1e-5)
    if not valid.any():
        raise ValueError(f"no camera baseline >= {min_baseline_m:.3f} m")
    ratios = gt_dist[valid] / pred_dist[valid]
    scale = ratios.median()
    rel_mad = (ratios - scale).abs().median() / scale.clamp_min(1e-8)

    src = pred * scale
    src_c, gt_c = src - src.mean(0), gt - gt.mean(0)
    u, _, vh = torch.linalg.svd(gt_c.T @ src_c)
    sign = torch.sign(torch.det(u @ vh))
    diag = torch.diag(torch.stack([scale.new_tensor(1.0), scale.new_tensor(1.0), sign]))
    rotation = u @ diag @ vh
    translation = gt.mean(0) - src.mean(0) @ rotation.T
    aligned = src @ rotation.T + translation
    rmse = ((aligned - gt).square().sum(-1).mean()).sqrt()
    return {
        "scale": scale,
        "pair_count": int(valid.sum()),
        "relative_mad": rel_mad,
        "alignment_rmse_m": rmse,
        "alignment_rotation": rotation,
        "alignment_translation_m": translation,
        "anchor_indices": keep,
        "scale_source": scale_source,
    }


def _resize_for_vggt(views: torch.Tensor, target_width: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Official-style [0,1] aspect-preserving resize, dimensions divisible by 14."""
    _, _, h, w = views.shape
    out_w = int(target_width)
    out_h = max(14, round(h * out_w / w / 14) * 14)
    resized = F.interpolate(views, size=(out_h, out_w), mode="bicubic", align_corners=False)
    return resized.clamp(0.0, 1.0), (h, w)


def vggt_confidence_score(
    raw_confidence: torch.Tensor,
    low_quantile: float = 0.1,
    high_quantile: float = 0.9,
) -> torch.Tensor:
    """Convert VGGT's relative ``1+exp(logit)`` confidence to [0,1].

    VGGT examples threshold confidence by percentile; its absolute magnitude
    is not calibrated across scenes. A fixed transform followed by a 0.3 gate
    collapsed real KITTI caches to zero coverage. Recover the monotonic logit
    and normalize robustly between per-view q10/q90 instead. Constant maps get
    a neutral 0.5 score rather than being silently discarded.
    """
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("confidence quantiles must satisfy 0 <= low < high <= 1")
    raw = raw_confidence.float()
    evidence = torch.log((raw - 1.0).clamp_min(1e-12))
    if evidence.ndim >= 2:
        flat = evidence.flatten(-2)
        low = torch.quantile(flat, low_quantile, dim=-1, keepdim=True)
        high = torch.quantile(flat, high_quantile, dim=-1, keepdim=True)
        while low.ndim < evidence.ndim:
            low = low.unsqueeze(-1)
            high = high.unsqueeze(-1)
    else:
        flat = evidence.reshape(1, -1)
        low = torch.quantile(flat, low_quantile).reshape(1)
        high = torch.quantile(flat, high_quantile).reshape(1)
    span = high - low
    normalized = (evidence - low) / span.clamp_min(1e-6)
    score = torch.where(span > 1e-6, normalized, torch.full_like(normalized, 0.5))
    return torch.where(torch.isfinite(raw), score.clamp(0.0, 1.0), torch.zeros_like(score))


def run_joint_subset(
    model: VGGT,
    views: torch.Tensor,
    gt_world_cam: torch.Tensor,
    resolution: int,
    min_baseline_m: float,
    views_per_frame: int,
    device: torch.device,
    *,
    gt_world_vehicle: torch.Tensor | None = None,
    view_camera_ids: torch.Tensor | None = None,
) -> dict:
    images, original_hw = _resize_for_vggt(views, resolution)
    images = images.to(device)
    amp_enabled = device.type == "cuda"
    amp_dtype = (torch.bfloat16 if amp_enabled and torch.cuda.get_device_capability(device)[0] >= 8
                 else torch.float16)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(images[None])
            pose_enc = predictions["pose_enc"]
            depth = predictions["depth"]
            confidence = predictions["depth_conf"]

    predicted_w2c, predicted_K = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    predicted_w2c = predicted_w2c[0].float().cpu()
    predicted_K = predicted_K[0].float().cpu()
    motion = estimate_motion_metric_scale(
        predicted_w2c,
        gt_world_cam.cpu(),
        min_baseline_m,
        views_per_frame=views_per_frame,
        gt_world_vehicle=gt_world_vehicle,
        view_camera_ids=view_camera_ids,
    )

    depth = depth[0, ..., 0].float().cpu() * motion["scale"]
    confidence = vggt_confidence_score(confidence[0].float().cpu())
    depth = F.interpolate(depth[:, None], size=original_hw, mode="bilinear", align_corners=False)[:, 0]
    confidence = F.interpolate(confidence[:, None], size=original_hw, mode="bilinear", align_corners=False)[:, 0]
    return {
        "depth": depth.clamp(0.0, 80.0).to(torch.float16),
        "conf": confidence.clamp(min=0.0).to(torch.float16),
        "metric_scale": motion["scale"].float(),
        "scale_pair_count": motion["pair_count"],
        "scale_relative_mad": motion["relative_mad"].float(),
        "pose_alignment_rmse_m": motion["alignment_rmse_m"].float(),
        "scale_anchor_indices": motion["anchor_indices"].cpu(),
        "scale_source": motion["scale_source"],
        "scale_reliability": geometry_scale_reliability(
            str(motion["scale_source"]), int(motion["pair_count"]),
        ),
        "predicted_w2c": predicted_w2c,
        "predicted_K": predicted_K,
        "vggt_input_hw": torch.tensor(images.shape[-2:], dtype=torch.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/shizhm/Downloads/vggt.pt")
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--out", default="runs/cache_vggt_motion_metric")
    ap.add_argument("--eval_split", "--raw_dataset", dest="raw_dataset", action="store_true",
                    help="build directly from a manifest instead of a prebuilt sample cache")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl")
    ap.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--subset_specs", default="0:1,0:2,0:4,0:8",
                    help="independent joint inferences as start_frame:frame_count")
    ap.add_argument("--chunked", action="store_true",
                    help="chunk mode: one independent joint forward per route chunk "
                         "(cache v7); subset_specs then names chunk positions, e.g. 0,1,2,3")
    ap.add_argument("--chunks_per_window", type=int, default=4)
    ap.add_argument("--chunk_arc_m", type=float, default=12.0)
    ap.add_argument("--guard_m", type=float, default=4.0)
    ap.add_argument("--frames_per_chunk", type=int, default=2)
    ap.add_argument("--max_geometry_frames", type=int, default=8)
    ap.add_argument("--window_stride", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=518, help="VGGT input width")
    ap.add_argument("--min_baseline_m", type=float, default=0.25)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = VGGT(enable_point=False, enable_track=False).to(device)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state.get("model", state), strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"VGGT weights are incomplete; missing keys: {incompatible.missing_keys[:8]}"
        )
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith(("point_head.", "track_head."))
    ]
    if unexpected:
        raise RuntimeError(f"unexpected VGGT checkpoint keys: {unexpected[:8]}")
    del state
    model.eval()

    if args.raw_dataset and args.chunked:
        ds = ChunkedUnifiedBEVDataset(
            args.manifest,
            lidar_root=args.lidar_root,
            drive=None if args.drive in (None, "none") else args.drive,
            chunks_per_window=args.chunks_per_window,
            chunk_arc_m=args.chunk_arc_m,
            guard_m=args.guard_m,
            frames_per_chunk=args.frames_per_chunk,
            max_geometry_frames=args.max_geometry_frames,
            window_stride=args.window_stride,
            max_samples=args.max_samples,
            image_size=(160, 96),
            max_points_per_view=4096,
        )
    elif args.raw_dataset:
        ds = UnifiedBEVDataset(
            args.manifest,
            lidar_root=args.lidar_root,
            drive=None if args.drive in (None, "none") else args.drive,
            min_target_spacing_m=5.0,
            max_samples=args.max_samples,
            dense_source_count=args.dense_sources,
            sparse_source_count=args.dense_sources,
            image_size=(160, 96),
            max_points_per_view=4096,
        )
    else:
        ds = load_cached_unified_bev(args.cache)
    if args.chunked:
        build_chunk_cache(model, ds, args, device, out_dir)
        return
    run_frame_cache(model, ds, args, device, out_dir)


def chunk_forward_views(ds, records):
    """Stack all rig views (front2 + fisheye virtuals) of geometry frames."""
    items = []
    for rec in records:
        items.extend(ds._front_views(rec))
        if ds.use_fisheye:
            items.extend(ds._virtual_views(rec))
    expected = len(records) * ds.views_per_frame
    if len(items) != expected:
        raise RuntimeError(f"incomplete chunk rig: expected {expected} views, got {len(items)}")
    rgb = torch.stack([x[0] for x in items])
    K = torch.stack([x[1] for x in items])
    T = torch.stack([x[2] for x in items])
    imu = torch.stack([
        torch.from_numpy(r.T_world_imu.astype(np.float32)) for r in records
    ])
    return rgb, K, T, imu


def build_chunk_cache(model, ds, args, device, out_dir):
    """Cache v7: one independent joint VGGT forward per route chunk.

    Chunk-level exactness: conditions of a window differ only in chunk
    membership; every chunk entry is inferred from exactly that chunk's
    geometry frames, and K-chunk conditions assemble lift rows from the
    entries without any additional inference.
    """
    import time as _time

    chunk_positions = [int(x) for x in args.subset_specs.split(",") if x.strip()]
    n_chunks = ds.chunks_per_window
    if chunk_positions and max(chunk_positions) >= n_chunks:
        raise ValueError(f"chunk positions {chunk_positions} exceed {n_chunks - 1}")
    stop = len(ds) if args.max_samples is None else min(len(ds), args.start + args.max_samples)
    t0 = _time.time()
    completed = 0
    for i in range(args.start, stop):
        dst = out_dir / f"{i:06d}.pt"
        sample = ds[i]
        sample_identity = geometry_sample_identity(sample["meta"])
        blob = None
        if dst.exists():
            blob = torch.load(dst, map_location="cpu", weights_only=False)
            if blob.get("geometry_model") != "vggt" or int(blob.get("geometry_version", -1)) != 7:
                raise RuntimeError(f"cannot augment non-chunk-v7 cache {dst}")
            if blob.get("view_layout_version") != VIEW_LAYOUT_VERSION:
                raise RuntimeError(f"view-layout mismatch in {dst}")
            validate_geometry_blob_identity(blob, sample["meta"], context=str(dst))
        subsets = dict(blob.get("subsets", {})) if blob is not None else {}
        missing = [p for p in chunk_positions if f"c{p}" not in subsets]
        if not missing:
            continue
        window_records = ds.window_records(i)
        for p in missing:
            rgb, K, T, imu = chunk_forward_views(ds, window_records[p])
            entry = run_joint_subset(
                model, rgb, T, args.resolution, args.min_baseline_m,
                ds.views_per_frame, device,
                gt_world_vehicle=imu,
                view_camera_ids=torch.tensor(ds.view_camera_ids),
            )
            subsets[f"c{p}"] = entry
            print(
                f"[vggt-chunk] tile={i} chunk=c{p} frames={len(window_records[p])} "
                f"scale={float(entry['metric_scale']):.4f} "
                f"source={entry['scale_source']} "
                f"reliability={entry['scale_reliability']} "
                f"mad={float(entry['scale_relative_mad']):.3f} "
                f"pose_rmse={float(entry['pose_alignment_rmse_m']):.3f}m",
                flush=True,
            )
        updated = {
            "geometry_model": "vggt",
            "geometry_version": 7,
            "sample_identity": sample_identity,
            "scale_policy": "per_chunk_vehicle_motion",
            "confidence_normalization": "per_view_log_expp1_q10_q90",
            "views_per_frame": ds.views_per_frame,
            "view_layout_version": VIEW_LAYOUT_VERSION,
            "chunking_version": sample["meta"]["chunking_version"],
            "subsets": subsets,
        }
        temporary = dst.with_suffix(".pt.tmp")
        torch.save(updated, temporary)
        temporary.replace(dst)
        completed += 1
        elapsed = _time.time() - t0
        print(f"[vggt-chunk] {i + 1}/{stop} rate={completed / elapsed * 3600:.0f}/h", flush=True)
    print(f"[vggt-chunk] DONE {completed} tiles in {(_time.time() - t0) / 60:.1f} min")


def run_frame_cache(model, ds, args, device, out_dir):
    """Legacy frame-interval cache mode (v6)."""
    total_frames = ds.dense_source_count
    specs = parse_subset_specs(args.subset_specs, total_frames)
    stop = len(ds) if args.max_samples is None else min(len(ds), args.start + args.max_samples)
    t0 = time.time()
    completed = 0
    for i in range(args.start, stop):
        dst = out_dir / f"{i:06d}.pt"
        blob = None
        if dst.exists():
            blob = torch.load(dst, map_location="cpu", weights_only=False)
            if blob.get("geometry_model") != "vggt":
                raise RuntimeError(f"cannot augment non-VGGT cache {dst}")
            if int(blob.get("geometry_version", -1)) != 6:
                raise RuntimeError(
                    f"cannot augment legacy VGGT cache {dst}; use a fresh output directory"
                )
            if blob.get("view_layout_version") != VIEW_LAYOUT_VERSION:
                raise RuntimeError(f"view-layout mismatch in {dst}")
            if int(blob.get("views_per_frame", -1)) != int(ds.views_per_frame):
                raise RuntimeError(f"views_per_frame mismatch in {dst}")
        sample = ds[i]
        sample_identity = geometry_sample_identity(sample["meta"])
        if blob is not None:
            validate_geometry_blob_identity(
                blob, sample_identity, context=str(dst),
            )
        subsets = dict(blob.get("subsets", {})) if blob is not None else {}
        missing = [spec for spec in specs if subset_key(*spec) not in subsets]
        if not missing:
            continue
        for start_frame, frame_count in missing:
            start_view = start_frame * ds.views_per_frame
            stop_view = (start_frame + frame_count) * ds.views_per_frame
            entry = run_joint_subset(
                model,
                sample["source_rgb"][start_view:stop_view],
                sample["source_T_world_cam"][start_view:stop_view],
                args.resolution,
                args.min_baseline_m,
                ds.views_per_frame,
                device,
                gt_world_vehicle=sample["source_T_world_imu"][
                    start_frame:start_frame + frame_count
                ],
                view_camera_ids=torch.tensor(ds.view_camera_ids),
            )
            subsets[subset_key(start_frame, frame_count)] = entry
            print(
                f"[vggt-cache] tile={i} subset={start_frame}:{frame_count} "
                f"scale={float(entry['metric_scale']):.4f} "
                f"source={entry['scale_source']} "
                f"reliability={entry['scale_reliability']} "
                f"mad={float(entry['scale_relative_mad']):.3f} "
                f"pose_rmse={float(entry['pose_alignment_rmse_m']):.3f}m",
                flush=True,
            )
        updated = {
            "geometry_model": "vggt",
            "geometry_version": 6,
            "sample_identity": sample_identity,
            "scale_policy": "vehicle_pose_motion_multiframe_camera_rig_singleframe",
            "confidence_normalization": "per_view_log_expp1_q10_q90",
            "views_per_frame": ds.views_per_frame,
            "view_layout_version": ds.view_layout_version,
            "subsets": subsets,
        }
        temporary = dst.with_suffix(".pt.tmp")
        torch.save(updated, temporary)
        temporary.replace(dst)
        completed += 1
        elapsed = time.time() - t0
        print(f"[vggt-cache] {i + 1}/{stop} rate={completed / elapsed * 3600:.0f}/h", flush=True)
    print(f"[vggt-cache] DONE {completed} tiles in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
