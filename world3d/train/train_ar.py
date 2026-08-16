#!/usr/bin/env python3

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
from world3d.train.trainer_ar import ArTrainer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/ar.yaml")

    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--data_root", type=str, default=None)

    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--accum_steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)

    return ap.parse_args()


def main():
    args = parse_args()
    overrides = {
        "device": args.device,
        "out_dir": args.out_dir,
        "resume_ckpt": args.resume_ckpt,
        "data_root": args.data_root,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "lr": args.lr,
    }

    cfg = load_cfg(args.config, overrides=overrides)

    if not torch.cuda.is_available() and cfg.device.startswith("cuda"):
        cfg.device = "cpu"

    trainer = ArTrainer(cfg, repo_root=REPO_ROOT)
    trainer.train()


if __name__ == "__main__":
    main()

