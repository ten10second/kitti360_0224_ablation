#!/usr/bin/env python3
"""
Prepare train/test split for KITTI-360 multi-drive training.

Strategy: Reserve last 5% of each drive as test set.
This ensures temporal separation and tests generalization to unseen road segments.
"""

import argparse
from pathlib import Path


def prepare_split(data_root: str, test_ratio: float = 0.05):
    """Prepare train/test split for all drives.

    Args:
        data_root: Root directory containing drive folders
        test_ratio: Fraction of frames to reserve for testing (default 5%)
    """
    data_root = Path(data_root)

    all_drives = sorted(data_root.glob("2013_05_28_drive_*_sync"))

    if not all_drives:
        print(f"No drive folders found in {data_root}")
        return

    print(f"Found {len(all_drives)} drives in {data_root}\n")

    train_frames_total = 0
    test_frames_total = 0

    train_config = []
    test_config = []

    for drive_path in all_drives:
        drive_name = drive_path.name
        poses_file = drive_path / "poses.txt"

        if not poses_file.exists():
            print(f"⚠️  {drive_name}: No poses.txt, skipping")
            continue

        # Read all frame IDs
        frame_ids = []
        with open(poses_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    frame_ids.append(int(parts[0]))
                except Exception:
                    continue

        frame_ids = sorted(list(set(frame_ids)))
        total_frames = len(frame_ids)

        if total_frames == 0:
            print(f"⚠️  {drive_name}: No valid frames, skipping")
            continue

        # Split: last test_ratio% for testing (by frame ID, not by count)
        # This ensures temporal separation - test frames are from later in the drive
        min_frame_id = frame_ids[0]
        max_frame_id = frame_ids[-1]
        frame_id_range = max_frame_id - min_frame_id
        split_frame_id = min_frame_id + int(frame_id_range * (1 - test_ratio))

        train_ids = [fid for fid in frame_ids if fid < split_frame_id]
        test_ids = [fid for fid in frame_ids if fid >= split_frame_id]

        train_frames_total += len(train_ids)
        test_frames_total += len(test_ids)

        print(f"✓ {drive_name}:")
        print(f"    Total: {total_frames} frames")
        print(f"    Train: {len(train_ids)} frames (0-{train_ids[-1]})")
        print(f"    Test:  {len(test_ids)} frames ({test_ids[0]}-{test_ids[-1]})")

        # Save train/test frame lists
        train_file = drive_path / "train_frames.txt"
        test_file = drive_path / "test_frames.txt"

        with open(train_file, "w") as f:
            for fid in train_ids:
                f.write(f"{fid}\n")

        with open(test_file, "w") as f:
            for fid in test_ids:
                f.write(f"{fid}\n")

        print(f"    Saved: {train_file.name}, {test_file.name}\n")

        # Collect for config
        train_config.append(f"  - drive: {drive_name}")
        train_config.append(f"    frames_file: train_frames.txt")

        test_config.append(f"  - drive: {drive_name}")
        test_config.append(f"    frames_file: test_frames.txt")

    print("=" * 60)
    print(f"Summary:")
    print(f"  Total drives: {len(all_drives)}")
    print(f"  Train frames: {train_frames_total}")
    print(f"  Test frames:  {test_frames_total}")
    print(f"  Test ratio:   {test_frames_total / (train_frames_total + test_frames_total) * 100:.1f}%")
    print("=" * 60)

    # Save config template
    config_file = data_root / "train_test_split_config.yaml"
    with open(config_file, "w") as f:
        f.write("# KITTI-360 Train/Test Split Configuration\n")
        f.write(f"# Generated with test_ratio={test_ratio}\n\n")
        f.write("train:\n")
        f.write("\n".join(train_config))
        f.write("\n\ntest:\n")
        f.write("\n".join(test_config))
        f.write("\n")

    print(f"\n✓ Config template saved to: {config_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare KITTI-360 train/test split")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/shizhm/sda1/KITTI-360",
        help="Root directory containing drive folders"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Fraction of frames to reserve for testing (default: 0.05 = 5%%)"
    )

    args = parser.parse_args()
    prepare_split(args.data_root, args.test_ratio)
