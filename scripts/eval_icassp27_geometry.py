#!/usr/bin/env python3
"""Geometry (Gate A) binned evaluation: LiDAR-anchored depth AbsRel.

For each held-out val tuple (same deterministic selection as the appearance
eval), at the target frame:
  - sparse STATIC LiDAR GT depth at 640x256 (world3d/eval/k360_lidar)
  - Metric3D ViT-S (offline cached, true intrinsics given) depth of:
      a) the GENERATED image            -> AbsRel_gen   (the model's score)
      b) the REAL target photo          -> AbsRel_real  (instrument ceiling)
      c) the VQ oracle (enc->dec) image -> AbsRel_vq    (Phase-A tokenizer ceiling)
  - scale-aligned AbsRel on static-valid pixels; dynamic pixels excluded.

Usage:
  python -m scripts.eval_icassp27_geometry --ckpt runs/icassp27_b2_pilot/ckpt.pt \
      --num_tuples 72 --out runs/geom_b2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from world3d.data.kitti360_tuple_dataset import Kitti360TupleDataset, collate_tuples
from world3d.models.icassp27_predictor import ICASSP27Predictor
from world3d.eval.k360_lidar import K360Lidar, absrel_scale_aligned
from scripts.eval_icassp27_binned import build_model_from_ckpt

LIDAR_ROOT = Path("/media/shizhm/sda2/KITTI-360/data_3d_raw")
K360_ROOT = Path("/media/shizhm/Lenovo/KITTI-360")


def load_metric3d(device):
    import torch.hub as hub
    repo = Path(hub.get_dir()) / "yvanyin_metric3d_main"
    m = hub.load(str(repo), "metric3d_vit_small", pretrain=True, source="local")
    return m.to(device).eval()


@torch.no_grad()
def metric3d_depth(m3d, img01: torch.Tensor, K: torch.Tensor) -> np.ndarray:
    """img01: (B,3,H,W) [0,1]; K: (B,3,3) at that scale -> (B,H,W) metric depth.

    Follows the official hub example: aspect-preserving resize into (616,1064),
    CENTER pad, ImageNet normalize; output is canonical-space depth -> center
    un-pad -> upsample -> de-canonicalize by fx_scaled/1000.
    """
    B, _, H, W = img01.shape
    dev = img01.device
    IN_H, IN_W = 616, 1064  # vit models
    mean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=dev).view(1, 3, 1, 1)

    out = torch.zeros(B, H, W, device=dev)
    for b in range(B):
        scale = min(IN_H / H, IN_W / W)
        nh, nw = int(H * scale), int(W * scale)
        x = torch.nn.functional.interpolate(img01[b : b + 1], size=(nh, nw), mode="bilinear", align_corners=False)
        x = x * 255.0
        pad_h, pad_w = IN_H - nh, IN_W - nw
        ph0, pw0 = pad_h // 2, pad_w // 2
        xp = torch.full((1, 3, IN_H, IN_W), 0.0, device=dev)
        xp[:, :, ph0 : ph0 + nh, pw0 : pw0 + nw] = x
        xp = (xp - mean) / std
        d, _, _ = m3d.inference({"input": xp})
        d = d[0, 0, ph0 : ph0 + nh, pw0 : pw0 + nw]
        d = torch.nn.functional.interpolate(d[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        fx_scaled = float(K[b, 0, 0]) * scale
        out[b] = d * (fx_scaled / 1000.0)  # canonical -> metric
    return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--num_tuples", type=int, default=72)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    out = Path(args.out) if args.out else Path("runs/geom") / Path(args.ckpt).parent.name
    out.mkdir(parents=True, exist_ok=True)

    model, step = build_model_from_ckpt(args.ckpt, device)
    vq = PretrainedTokenizer("ckpts/maskgit-vqgan-imagenet-f16-256.bin").to(device).eval()
    m3d = load_metric3d(device)
    print(f"[geom] ckpt={args.ckpt} (step {step})")

    ds = Kitti360TupleDataset(args.manifest, mode="eval", seed=args.seed)
    groups = defaultdict(list)
    for i in range(len(ds)):
        k_slot = i % len(ds.eval_k)
        spec = ds.tuples[i // len(ds.eval_k)]
        groups[(next(j for j, (lo, hi) in enumerate(ds.bins) if lo <= spec.dist_m < hi), k_slot)].append(i)
    per_group = max(1, args.num_tuples // len(groups))
    idxs = []
    for g in sorted(groups):
        idxs.extend(groups[g][:per_group])
    print(f"[geom] {len(idxs)} tuples over {len(groups)} groups")

    # lidar handle per drive
    lidar_cache = {}
    def get_lidar(drive):
        if drive not in lidar_cache:
            lidar_cache[drive] = K360Lidar(
                str(K360_ROOT / drive), str(LIDAR_ROOT / drive / "velodyne_points" / "data"))
        return lidar_cache[drive]

    records = []
    t0 = time.time()
    for s in range(0, len(idxs), args.batch_size):
        batch = collate_tuples([ds[i] for i in idxs[s : s + args.batch_size]])
        B = batch["tgt_rgb"].shape[0]
        gt01 = batch["tgt_rgb"].to(device)
        with torch.no_grad():
            tokens = model.generate(
                batch["pose_vec"].to(device),
                max_len=(ds.img_h // 16) * (ds.img_w // 16),
                temperature=args.temperature, top_p=args.top_p,
                sat=batch["sat"].to(device),
                window_origin_xyz=batch["window_origin_xyz"].to(device),
                src_rgbs=batch["src_rgbs"].to(device),
                rel_poses=batch["rel_poses"].to(device),
                src_mask=batch["src_mask"].to(device),
                tgt_K=batch["tgt_K"].to(device),
                tgt_T_cam=batch["tgt_T_cam"].to(device),
            )
            gen01 = ((vq.decode(tokens.view(B, ds.img_h // 16, ds.img_w // 16)) + 1) / 2).clamp(0, 1)
            tok_gt = vq.encode(gt01 * 2 - 1)
            if tok_gt.dim() == 4:
                tok_gt = tok_gt.squeeze(1)
            vq01 = ((vq.decode(tok_gt.view(B, ds.img_h // 16, ds.img_w // 16)) + 1) / 2).clamp(0, 1)

        Kdev = batch["tgt_K"].to(device)
        d_gen = metric3d_depth(m3d, gen01, Kdev)
        d_real = metric3d_depth(m3d, gt01, Kdev)
        d_vq = metric3d_depth(m3d, vq01, Kdev)

        for b in range(B):
            m = batch["meta"][b]
            fid = m["target_fid"]
            L = get_lidar(m["drive"])
            if not L.has_scan(fid):
                continue
            gt_depth, valid = L.sparse_depth(fid, (ds.img_w, ds.img_h), static_only=True)
            if valid.sum() < 500:
                continue
            rec = {
                "ckpt_step": step, "bin": m["bin"], "K": int(batch["n_src"][b]),
                "dist_m": round(m["dist_m"], 2), "drive": m["drive"], "target_fid": fid,
                "absrel_gen": absrel_scale_aligned(d_gen[b], gt_depth, valid),
                "absrel_real": absrel_scale_aligned(d_real[b], gt_depth, valid),
                "absrel_vq": absrel_scale_aligned(d_vq[b], gt_depth, valid),
                "n_valid_px": int(valid.sum()),
            }
            records.append(rec)
        if (s // args.batch_size) % 3 == 0:
            print(f"  {s+B}/{len(idxs)}  {(time.time()-t0)/(s+B):.1f}s/tuple", flush=True)

    with open(out / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n===== geometry summary (ckpt step {step}, n={len(records)}) =====")
    print(f"{'bin':>4} {'K':>2} {'n':>4} {'AbsRel_gen':>10} {'AbsRel_real':>11} {'AbsRel_vq':>10}")
    def show(rs, label):
        if not rs:
            return
        a = lambda k: np.nanmean([r[k] for r in rs])
        print(f"{label:>4} {'':>2} {len(rs):>4} {a('absrel_gen'):>10.4f} {a('absrel_real'):>11.4f} {a('absrel_vq'):>10.4f}")
    for g in sorted({(r["bin"], r["K"]) for r in records}):
        show([r for r in records if (r["bin"], r["K"]) == g], str(g[0]))
    # Gate-1 merged bin: [5,15) m  (bins 1 and 2 restricted <15m)
    merged = [r for r in records if 5.0 <= r["dist_m"] < 15.0]
    show(merged, "G1")
    show(records, "all")
    print(f"[geom] records -> {out/'records.jsonl'}")


if __name__ == "__main__":
    main()
