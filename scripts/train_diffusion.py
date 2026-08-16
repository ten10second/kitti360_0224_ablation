#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for diffusion-based view synthesis model.

This is the primary entry point for training the diffusion model.
Place in scripts/ directory for consistency with existing project structure.

Usage:
    python scripts/train_diffusion.py --config configs/diffusion_1d_cond.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Use huggingface mirror site for faster downloads in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
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
        "--debug",
        action="store_true",
        help="Enable debug mode (small batch, single GPU)",
    )

    return parser.parse_args()


def train_single_gpu(args, cfg):
    """
    Run single GPU training.

    Args:
        args: parsed command line arguments
        cfg: configuration object
    """
    print(f"Running single GPU training on {cfg.device}")

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
    if args.device is not None:
        overrides["device"] = args.device

    cfg = load_cfg(args.config, overrides=overrides)

    # Debug mode modifications
    if args.debug:
        cfg.batch_size = 2
        cfg.steps = 100
        cfg.print_every = 10
        cfg.save_every = 50
        cfg.vis_every = 20
        cfg.eval_every = 100
        cfg.loader['num_workers'] = 0
        print("Debug mode enabled: Small parameters")

    # Check device availability
    if not torch.cuda.is_available() and cfg.device.startswith("cuda"):
        cfg.device = "cpu"
        print("CUDA not available, using CPU")

    # Start training
    train_single_gpu(args, cfg)


if __name__ == "__main__":
    main()
