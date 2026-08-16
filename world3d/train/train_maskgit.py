#!/usr/bin/env python3
"""MaskGIT training entry point."""

import argparse
import os
import sys
import torch.multiprocessing as mp

# Must be called before any CUDA init so DataLoader workers can use GPU.
mp.set_start_method("spawn", force=True)

import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from world3d.config import load_cfg
from world3d.train.trainer_maskgit import MaskGITTrainer


def parse_args():
    ap = argparse.ArgumentParser(description="Train MaskGIT token predictor")
    ap.add_argument("--config", type=str, default="configs/maskgit.yaml")

    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--warm_start_ckpt", type=str, default=None)
    ap.add_argument("--data_root", type=str, default=None)

    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--accum_steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)

    ap.add_argument("--maskgit_num_steps", type=int, default=None,
                    help="Number of iterative decoding steps during inference")

    ipm_semantic_group = ap.add_mutually_exclusive_group()
    ipm_semantic_group.add_argument(
        "--use-ipm-semantic",
        dest="use_ipm_semantic",
        action="store_true",
        help="Enable IPM semantic features (override config)",
    )
    ipm_semantic_group.add_argument(
        "--no-use-ipm-semantic",
        dest="use_ipm_semantic",
        action="store_false",
        help="Disable IPM semantic features (override config)",
    )
    ap.set_defaults(use_ipm_semantic=None)

    return ap.parse_args()


def main():
    args = parse_args()
    overrides = {
        "device": args.device,
        "out_dir": args.out_dir,
        "resume_ckpt": args.resume_ckpt,
        "warm_start_ckpt": args.warm_start_ckpt,
        "data_root": args.data_root,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "lr": args.lr,
        "maskgit_num_steps": args.maskgit_num_steps,
        "use_ipm_semantic": args.use_ipm_semantic,
    }

    cfg = load_cfg(args.config, overrides=overrides)

    if not torch.cuda.is_available() and cfg.device.startswith("cuda"):
        cfg.device = "cpu"

    trainer = MaskGITTrainer(cfg, repo_root=REPO_ROOT)
    trainer.train()


if __name__ == "__main__":
    main()
