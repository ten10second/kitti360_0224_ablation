#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for diffusion-based view synthesis model.

Usage:
    python world3d/train/train_diffusion.py --config configs/diffusion_1d_cond.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.multiprocessing as mp
from world3d.config import load_cfg
from world3d.train.trainer_diffusion import DiffusionTrainer
from utils.distributed import init_distributed_mode


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train diffusion-based view synthesis model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/diffusion_1d_cond.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use only a subset of data (for debugging)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (small batch, single GPU)",
    )

    return parser.parse_args()


def train(rank: int, world_size: int, args, cfg) -> None:
    """
    Main training function called by each process.

    Args:
        rank: process rank
        world_size: total number of processes
        args: parsed command line arguments
        cfg: configuration object
    """
    try:
        init_distributed_mode(rank, world_size, cfg)

        print(f"[Rank {rank}] Training starting...")

        trainer = DiffusionTrainer(
            cfg,
            repo_root=REPO_ROOT,
            resume_ckpt=args.resume_ckpt,
        )

        trainer.train()

        print(f"[Rank {rank}] Training completed")

    except Exception as e:
        print(f"[Rank {rank}] Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_distributed_training(args, cfg):
    """
    Run distributed training using torch.distributed.

    Args:
        args: parsed command line arguments
        cfg: configuration object
    """
    import torch.distributed as dist

    world_size = torch.cuda.device_count()

    print(f"Running distributed training on {world_size} GPUs")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    mp.spawn(
        train,
        args=(world_size, args, cfg),
        nprocs=world_size,
        join=True,
    )


def run_single_gpu_training(args, cfg):
    """
    Run single GPU training.

    Args:
        args: parsed command line arguments
        cfg: configuration object
    """
    print(f"Running single GPU training on {cfg.device}")

    # Disable distributed training
    cfg.dist = None
    cfg.device = args.device or "cuda"

    trainer = DiffusionTrainer(
        cfg,
        repo_root=REPO_ROOT,
        resume_ckpt=args.resume_ckpt,
    )

    trainer.train()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    overrides = {}
    if args.out_dir is not None:
        overrides["out_dir"] = args.out_dir
    if args.data_root is not None:
        overrides["data_root"] = args.data_root
    if args.subset is not None:
        overrides["subset"] = args.subset
    if args.device is not None:
        overrides["device"] = args.device

    cfg = load_cfg(args.config, overrides=overrides)

    # Debug mode modifications
    if args.debug:
        cfg.batch_size = 1
        cfg.steps = 100
        cfg.print_every = 10
        cfg.save_every = 50
        cfg.vis_every = 20
        cfg.eval_every = 100
        cfg.loader.num_workers = 0
        print("Debug mode enabled: Small parameters")

    # Check device availability
    if not torch.cuda.is_available() and cfg.device.startswith("cuda"):
        cfg.device = "cpu"
        print("CUDA not available, using CPU")

    # Start training
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # Multi-GPU training
        run_distributed_training(args, cfg)
    else:
        # Single GPU training
        run_single_gpu_training(args, cfg)


if __name__ == "__main__":
    main()
