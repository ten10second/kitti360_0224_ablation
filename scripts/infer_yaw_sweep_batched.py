#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batched yaw sweep inference for direct/hybrid models.

This script keeps the same sample preparation behavior as
`scripts/infer_direct_yaw_sweep.py`, but batches the expensive model generation
and VQ decoding steps so larger GPUs can improve inference throughput.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.train.vis_utils import render_sat_with_frustum

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_direct_yaw_sweep as base


@dataclass(frozen=True)
class InferenceTask:
    dataset_key: Tuple[str, ...]
    sample_index: int
    drive: str
    frame_id: int
    label: str
    order_index: int
    scene_dir: Path
    group_key: Tuple[str, int]


def parse_args():
    p = argparse.ArgumentParser(description="Batched yaw sweep inference for direct/hybrid models")
    p.add_argument("--ckpt", default=str(REPO_ROOT / "runs/ar_direct/ckpt_step_0060000.pt"))
    p.add_argument("--vq-ckpt", default=str(REPO_ROOT / "ckpts/maskgit-vqgan-imagenet-f16-256.bin"))
    p.add_argument("--data-root", default="/media/zhimiao/Lenovo/KITTI-360")
    p.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs/ar_direct/yaw_sweep_batched"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Number of samples to generate in parallel")
    p.add_argument("--mode", default="default", choices=["default", "fixed5", "interpolated", "zero_shot", "360sweep"],
                    help="default: run fixed5 + zero_shot test views | fixed5: run only fixed5 views | interpolated: run only left_to_front_30/right_to_front_30 views | zero_shot: run only held-out zero-shot views | 360sweep: 360° sweep on frame 113")
    p.add_argument("--model-mode", default="direct", choices=["direct", "hybrid"],
                    help="Model architecture to load (direct or hybrid).")
    p.add_argument("--test-drives", type=str, nargs="+",
                    default=None,
                    help="Test drive and frame IDs in format 'drive:frame1,frame2' (for default/fixed5/interpolated/zero_shot mode). If not specified, will auto-read test_frames.txt from each sync directory in data_root.")
    p.add_argument("--sweep-frame", type=int, default=113,
                    help="Frame ID for 360° sweep (for 360sweep mode)")
    p.add_argument("--sweep-interval", type=int, default=20,
                    help="Angle interval in degrees for 360° sweep")
    p.add_argument("--maskgit", action="store_true", help="Use MaskGIT model instead of AR")
    p.add_argument("--maskgit-steps", type=int, default=12, help="MaskGIT iterative decoding steps")
    p.add_argument("--use-ipm-semantic", action="store_true",
                    help="Use semantic condition tokens in model input (coords are always computed)")
    p.add_argument("--n-pose-queries", type=int, default=64,
                    help="Number of pose queries/anchors (must match training config, default: 64)")
    p.add_argument("--hybrid-memory-source", default="enhanced",
                    choices=["enhanced", "anchor", "anchor_tokens"],
                    help="Hybrid memory source to use when --model-mode hybrid (must match training)")

    explicit_pos_group = p.add_mutually_exclusive_group()
    explicit_pos_group.add_argument("--use-explicit-token-pos", dest="use_explicit_token_pos", action="store_true",
                                    help="Enable explicit token positional embedding (must match training)")
    explicit_pos_group.add_argument("--no-use-explicit-token-pos", dest="use_explicit_token_pos", action="store_false",
                                    help="Disable explicit token positional embedding (must match training)")
    p.set_defaults(use_explicit_token_pos=False)

    pose_token_group = p.add_mutually_exclusive_group()
    pose_token_group.add_argument("--use-pose-token", dest="use_pose_token", action="store_true",
                                  help="Enable pose token in memory (must match training)")
    pose_token_group.add_argument("--no-use-pose-token", dest="use_pose_token", action="store_false",
                                  help="Disable pose token in memory (must match training)")
    p.set_defaults(use_pose_token=True)

    strict_group = p.add_mutually_exclusive_group()
    strict_group.add_argument("--strict-load", dest="strict_load", action="store_true",
                              help="Strict checkpoint loading (recommended for consistency)")
    strict_group.add_argument("--non-strict-load", dest="strict_load", action="store_false",
                              help="Allow partial checkpoint loading")
    p.set_defaults(strict_load=True)
    return p.parse_args()


def parse_test_drive_frames(data_root: Path, test_drives: Optional[List[str]]) -> List[Tuple[str, List[int]]]:
    test_drive_frames: List[Tuple[str, List[int]]] = []

    if test_drives is None:
        print("[Auto-scan] Reading test_frames.txt from all sync directories...")
        for sync_dir in sorted(data_root.iterdir()):
            if sync_dir.is_dir() and "sync" in sync_dir.name:
                test_frames_path = sync_dir / "test_frames.txt"
                if test_frames_path.exists():
                    with open(test_frames_path, "r") as handle:
                        frames = [int(line.strip()) for line in handle if line.strip().isdigit()]
                    if frames:
                        test_drive_frames.append((sync_dir.name, frames))
                        print(f"  Found {sync_dir.name}: {len(frames)} frames")
        if not test_drive_frames:
            raise ValueError("No test_frames.txt found in any sync directory!")
        return test_drive_frames

    for drive_frame_str in test_drives:
        if ":" in drive_frame_str:
            drive, frame_str = drive_frame_str.split(":")
            frames = list(map(int, frame_str.split(",")))
            test_drive_frames.append((drive, frames))
            print(f"  User specified {drive}: {len(frames)} frames")
        else:
            drive = drive_frame_str
            test_frames_path = data_root / drive / "test_frames.txt"
            if not test_frames_path.exists():
                raise ValueError(f"test_frames.txt not found for drive {drive} at {test_frames_path}")
            with open(test_frames_path, "r") as handle:
                frames = [int(line.strip()) for line in handle if line.strip().isdigit()]
            if frames:
                frames = frames[:2]
                test_drive_frames.append((drive, frames))
                print(f"  User specified {drive}: loaded {len(frames)} frames (first 2 of test_frames.txt)")
            else:
                print(f"[Warning] No valid frames found in {test_frames_path}")
    return test_drive_frames


def build_fixed5_tasks(args) -> Tuple[List[InferenceTask], Dict[Tuple[str, ...], Kitti360dDataset]]:
    print("\n" + "=" * 60)
    if args.mode == "default":
        print("Mode: Fixed 5 + Zero-shot Views on Test Frames from Multiple Drives (Batched)")
    elif args.mode == "interpolated":
        print("Mode: Interpolated Views on Test Frames from Multiple Drives (Batched)")
    elif args.mode == "zero_shot":
        print("Mode: Zero-shot Views on Test Frames from Multiple Drives (Batched)")
    else:
        print("Mode: Fixed 5 Views on Test Frames from Multiple Drives (Batched)")
    print("=" * 60)

    data_root = Path(args.data_root)
    view_groups = base.get_requested_view_groups(args.mode)
    test_drive_frames = parse_test_drive_frames(data_root, args.test_drives)

    dataset_cache: Dict[Tuple[str, ...], Kitti360dDataset] = {}
    tasks: List[InferenceTask] = []

    for drive, frames in test_drive_frames:
        drive_path = data_root / drive
        base_scene_dir = Path(args.out_dir) / args.model_mode / drive
        frame_to_index = {frame_id: index for index, frame_id in enumerate(frames)}

        for group_name, group_views, subdir in view_groups:
            scene_dir = base_scene_dir / subdir

            for view_order, (view_name, source, fisheye_cam, yaw_deg) in enumerate(group_views):
                if source == "front":
                    dataset_key = (drive, group_name, view_name, "front")
                    dataset_cache[dataset_key] = Kitti360dDataset(
                        drives=str(drive_path),
                        frames=frames,
                        mode="front",
                        virtual_hfov_deg=80.0,
                        virtual_size=(640, 256),
                    )
                else:
                    dataset_key = (drive, group_name, view_name, source, str(fisheye_cam), f"{float(yaw_deg):.1f}")
                    dataset_cache[dataset_key] = Kitti360dDataset(
                        drives=str(drive_path),
                        frames=frames,
                        mode="fisheye_virtual",
                        fisheye_camera=fisheye_cam,
                        fisheye_relative_yaw_deg=float(yaw_deg),
                        virtual_hfov_deg=80.0,
                        virtual_size=(640, 256),
                        random_fisheye_relative_yaw=False,
                        calib_yaw_fix_deg=4.0,
                    )

                for frame_id in frames:
                    tasks.append(
                        InferenceTask(
                            dataset_key=dataset_key,
                            sample_index=frame_to_index[frame_id],
                            drive=drive,
                            frame_id=frame_id,
                            label=view_name,
                            order_index=view_order,
                            scene_dir=scene_dir,
                            group_key=(str(scene_dir), frame_id),
                        )
                    )

    tasks.sort(key=lambda item: (str(item.scene_dir), item.drive, item.frame_id, item.order_index))
    print(f"[Tasks] Prepared {len(tasks)} view tasks across {len(test_drive_frames)} drives")
    return tasks, dataset_cache


def get_360_view_spec(angle: int) -> Tuple[str, Optional[str], Optional[float], str]:
    if 330 <= angle or angle < 30:
        return "front", None, None, "front"
    if 30 <= angle < 210:
        relative_yaw = -90.0 - float(angle)
        return "fisheye_virtual", "image_02", relative_yaw, "left_fisheye"
    relative_yaw = 90.0 - float(angle)
    return "fisheye_virtual", "image_03", relative_yaw, "right_fisheye"


def build_360_tasks(args) -> Tuple[List[InferenceTask], Dict[Tuple[str, ...], Kitti360dDataset]]:
    print("\n" + "=" * 60)
    print(f"Mode: 360° Sweep on Frame {args.sweep_frame} (Batched)")
    print(f"Interval: {args.sweep_interval}°")
    print("=" * 60)

    drive_path = Path(args.data_root) / args.drive
    scene_dir = Path(args.out_dir) / args.model_mode / args.drive
    frame_id = int(args.sweep_frame)
    angles = list(range(0, 360, args.sweep_interval))

    dataset_cache: Dict[Tuple[str, ...], Kitti360dDataset] = {}
    tasks: List[InferenceTask] = []

    for order_index, angle in enumerate(angles):
        mode, fisheye_cam, relative_yaw, camera_label = get_360_view_spec(angle)
        if mode == "front":
            dataset_key = (args.drive, f"angle_{angle:03d}", mode)
            dataset_cache[dataset_key] = Kitti360dDataset(
                drives=str(drive_path),
                frames=[frame_id],
                mode="front",
                virtual_hfov_deg=80.0,
                virtual_size=(640, 256),
            )
        else:
            dataset_key = (args.drive, f"angle_{angle:03d}", mode, str(fisheye_cam), f"{relative_yaw:.1f}")
            dataset_cache[dataset_key] = Kitti360dDataset(
                drives=str(drive_path),
                frames=[frame_id],
                mode="fisheye_virtual",
                fisheye_camera=fisheye_cam,
                fisheye_relative_yaw_deg=relative_yaw,
                virtual_hfov_deg=80.0,
                virtual_size=(640, 256),
                random_fisheye_relative_yaw=False,
                calib_yaw_fix_deg=4.0,
            )

        label = f"{angle}° ({camera_label})"
        tasks.append(
            InferenceTask(
                dataset_key=dataset_key,
                sample_index=0,
                drive=args.drive,
                frame_id=frame_id,
                label=label,
                order_index=order_index,
                scene_dir=scene_dir,
                group_key=(args.drive, frame_id),
            )
        )

    print(f"[Tasks] Prepared {len(tasks)} sweep views")
    return tasks, dataset_cache


@torch.no_grad()
def generate_ar_batched(model, condition, bev, bev_vis_mask, device, top_k=50, temperature=1.0):
    batch_size = condition["pose"].size(0)
    generated = torch.full((batch_size, 1), base.BOS_TOKEN, dtype=torch.long, device=device)
    past_kv = None

    for _ in range(base.SEQ_LEN):
        model_input = generated if past_kv is None else generated[:, -1:]
        logits, past_kv = model(
            generated_tokens=model_input,
            condition_tokens=condition,
            aligned_bev_feature_map=bev,
            bev_vis_mask=bev_vis_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        next_tok = base.top_k_sample(logits[:, -1, :1024], k=top_k, temperature=temperature)
        generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)

    token_seq = generated[:, 1:]
    return model.seq_to_grid(token_seq)


def stack_prepared_samples(prepared_samples: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    first_condition = prepared_samples[0]["condition"]
    condition = {
        key: torch.cat([item["condition"][key] for item in prepared_samples], dim=0)
        for key in first_condition.keys()
    }

    bev = None
    if prepared_samples[0]["bev"] is not None:
        bev = torch.cat([item["bev"] for item in prepared_samples], dim=0)

    bev_vis_mask = None
    if prepared_samples[0]["bev_vis_mask"] is not None:
        bev_vis_mask = torch.cat([item["bev_vis_mask"] for item in prepared_samples], dim=0)

    return {
        "condition": condition,
        "bev": bev,
        "bev_vis_mask": bev_vis_mask,
    }


def load_prepared_item(task: InferenceTask, dataset_cache, vq, bev_encoder, device, use_ipm_semantic: bool):
    dataset = dataset_cache[task.dataset_key]
    sample = dataset[task.sample_index]
    if sample.get("meta", {}).get("dummy", False):
        print(f"[Skip] drive={task.drive} frame={task.frame_id} view={task.label}: dummy sample")
        return None

    prepared = base.prepare_sample(sample, vq, bev_encoder, device, use_ipm_semantic=use_ipm_semantic)
    gt_pil = base.tensor01_to_pil(sample["image"])

    sat_np = (prepared["sat"].detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    sat_with_frustum = render_sat_with_frustum(
        sat_img=sat_np,
        K=prepared["K_raw"].detach().cpu().numpy(),
        T_cam_to_world=prepared["T_cam_to_world"].detach().cpu().numpy(),
        T_imu_to_world=prepared["T_imu_to_world"].detach().cpu().numpy(),
        cam_h=base.TARGET_H,
        cam_w=base.TARGET_W,
    )

    return {
        "task": task,
        "prepared": prepared,
        "gt_pil": gt_pil,
        "heatmap_pil": Image.fromarray(sat_with_frustum),
    }


def flush_batch(batch_items, args, model, vq, device, rows_by_group, batch_times, sample_times):
    if not batch_items:
        return

    batched_inputs = stack_prepared_samples([item["prepared"] for item in batch_items])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    if args.maskgit:
        token_grid = base.generate_maskgit(
            model,
            batched_inputs["condition"],
            batched_inputs["bev"],
            batched_inputs["bev_vis_mask"],
            device,
            top_k=args.top_k,
            temperature=args.temperature,
            num_steps=args.maskgit_steps,
        )
    else:
        token_grid = generate_ar_batched(
            model,
            batched_inputs["condition"],
            batched_inputs["bev"],
            batched_inputs["bev_vis_mask"],
            device,
            top_k=args.top_k,
            temperature=args.temperature,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    batch_size = len(batch_items)
    batch_times.append(elapsed)
    sample_times.extend([elapsed / batch_size] * batch_size)

    print(f"[Batch] size={batch_size} generation={elapsed:.3f}s throughput={batch_size / elapsed:.2f} samples/s")

    gen_imgs = vq.decode(token_grid)
    for index, item in enumerate(batch_items):
        task = item["task"]
        prepared = item["prepared"]
        gen_pil = base.tensor_to_pil(gen_imgs[index])
        base.save_view_outputs(
            task.scene_dir,
            task.label.split(" (")[0],
            task.frame_id,
            gen_pil,
            item["gt_pil"],
            prepared["K_raw"],
            prepared["T_cam_to_world"],
        )
        rows_by_group.setdefault(task.group_key, []).append(
            (task.order_index, (task.label, item["heatmap_pil"], gen_pil, item["gt_pil"], task.scene_dir, task.frame_id))
        )

        del item["prepared"]


def finalize_outputs(rows_by_group: Dict[Tuple[str, int], List[Tuple[int, tuple]]]):
    for _, rows_with_order in sorted(rows_by_group.items()):
        rows_with_order.sort(key=lambda item: item[0])
        rows = [item[1][:4] for item in rows_with_order]
        scene_dir = rows_with_order[0][1][4]
        frame_id = rows_with_order[0][1][5]
        base.save_yaw_grid(rows, scene_dir, frame_id)


def print_timing_summary(args, total_samples: int, batch_times: List[float], sample_times: List[float], wall_time: float):
    if total_samples == 0:
        print("\nNo valid samples were processed.")
        return

    batch_mean = float(np.mean(batch_times)) if batch_times else 0.0
    sample_mean = float(np.mean(sample_times)) if sample_times else 0.0
    throughput = total_samples / wall_time if wall_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("Batched Inference Timing Statistics")
    print("=" * 60)
    print(f"Model: {'Direct MaskGIT' if args.maskgit else 'Direct AR'}")
    print(f"Batch size: {args.batch_size}")
    if args.maskgit:
        print(f"MaskGIT steps: {args.maskgit_steps}")
    print(f"Total samples: {total_samples}")
    print(f"Total wall time: {wall_time:.3f}s")
    print(f"Average batch generation time: {batch_mean:.3f}s")
    print(f"Average per-sample generation time: {sample_mean:.3f}s")
    print(f"End-to-end throughput: {throughput:.2f} samples/s")
    if sample_mean > 0:
        print(f"Model-only equivalent FPS: {1.0 / sample_mean:.2f}")
    print("=" * 60)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    if args.maskgit:
        model = base.load_maskgit_model(
            args.ckpt, device,
            model_mode=args.model_mode,
            use_ipm_semantic=args.use_ipm_semantic,
            use_pose_token=args.use_pose_token,
            strict_load=args.strict_load,
            n_pose_queries=args.n_pose_queries,
            hybrid_memory_source=args.hybrid_memory_source,
            use_explicit_token_pos=args.use_explicit_token_pos,
        )
    else:
        model = base.load_model(
            args.ckpt, device,
            model_mode=args.model_mode,
            use_ipm_semantic=args.use_ipm_semantic,
            use_pose_token=args.use_pose_token,
            strict_load=args.strict_load,
            n_pose_queries=args.n_pose_queries,
            hybrid_memory_source=args.hybrid_memory_source,
            use_explicit_token_pos=args.use_explicit_token_pos,
        )

    vq = base.PretrainedTokenizer(args.vq_ckpt).to(device)
    vq.eval()
    bev_encoder = model.bev_encoder
    bev_encoder.eval()

    if args.mode in {"default", "fixed5", "interpolated", "zero_shot"}:
        tasks, dataset_cache = build_fixed5_tasks(args)
    elif args.mode == "360sweep":
        tasks, dataset_cache = build_360_tasks(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    rows_by_group: Dict[Tuple[str, int], List[Tuple[int, tuple]]] = {}
    batch_items = []
    batch_times: List[float] = []
    sample_times: List[float] = []
    total_samples = 0

    wall_start = time.time()
    for task_index, task in enumerate(tasks, start=1):
        item = load_prepared_item(
            task,
            dataset_cache,
            vq,
            bev_encoder,
            device,
            use_ipm_semantic=args.use_ipm_semantic,
        )
        if item is None:
            continue

        batch_items.append(item)
        if len(batch_items) >= args.batch_size:
            flush_batch(batch_items, args, model, vq, device, rows_by_group, batch_times, sample_times)
            total_samples += len(batch_items)
            batch_items = []

        if task_index % max(args.batch_size, 10) == 0:
            print(f"[Progress] prepared {task_index}/{len(tasks)} tasks")

    if batch_items:
        flush_batch(batch_items, args, model, vq, device, rows_by_group, batch_times, sample_times)
        total_samples += len(batch_items)

    finalize_outputs(rows_by_group)
    wall_time = time.time() - wall_start
    print_timing_summary(args, total_samples, batch_times, sample_times, wall_time)
    print("\nDone.")


if __name__ == "__main__":
    main()
