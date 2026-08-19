#!/usr/bin/env python3
"""Binned AR sampling evaluation for the ICASSP27 framework.

Instrument #1 ("画得像不像"): free-running AR sampling at held-out target
poses, scored against the real photos, stratified by distance bin x K.

Usage:
  python -m scripts.eval_icassp27_binned --ckpt runs/icassp27_b2_pilot/ckpt.pt \
      --manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl \
      --num_tuples 36 --out runs/eval_b2_step<k>
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from world3d.data.kitti360_tuple_dataset import Kitti360TupleDataset, collate_tuples
from world3d.models.icassp27_predictor import ICASSP27Predictor
from metrics import PSNR, LPIPS


def build_model_from_ckpt(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]
    model = ICASSP27Predictor(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"], max_seq_len=cfg["max_seq_len"], pose_dim=cfg["pose_dim"],
        dino_arch=cfg["dino_arch"], sat_encoder=cfg["sat_encoder"], geo=cfg.get("geo", "raymap"),
        use_sat=cfg.get("use_sat", True), use_src=cfg.get("use_src", True),
        fourier_freqs=cfg.get("fourier_freqs", 10), sat_px=cfg.get("sat_px", 512),
        sat_pe_mode=cfg.get("sat_pe_mode", "legacy_fourier"),
        sat_coord_scale_m=cfg.get("sat_coord_scale_m", None),
        sat_m_per_px=cfg.get("sat_m_per_px", 0.196),
    ).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    return model, ckpt.get("step", -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--num_tuples", type=int, default=36)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out = Path(args.out) if args.out else Path("runs/eval") / (Path(args.ckpt).parent.name + f"_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    model, step = build_model_from_ckpt(args.ckpt, device)
    vq = PretrainedTokenizer("ckpts/maskgit-vqgan-imagenet-f16-256.bin").to(device).eval()
    print(f"[eval] ckpt={args.ckpt} (step {step})  manifest={Path(args.manifest).name}")

    ds = Kitti360TupleDataset(args.manifest, mode="eval", seed=args.seed)
    print(f"[eval] tuples={len(ds)} (eval_k={ds.eval_k}, bins={[b for b in ds.bins]})")

    # deterministic spread: for each (bin, k_slot) take num_tuples/(3*2) tuples
    groups = defaultdict(list)
    for i in range(len(ds)):
        m = ds[i // len(ds.eval_k)] if False else None
        # cheap meta without loading images: bin lives in spec.dist_m
        k_slot = i % len(ds.eval_k)
        spec = ds.tuples[i // len(ds.eval_k)]
        groups[(next(j for j, (lo, hi) in enumerate(ds.bins) if lo <= spec.dist_m < hi), k_slot)].append(i)
    per_group = max(1, args.num_tuples // len(groups))
    idxs = []
    for g in sorted(groups):
        idxs.extend(groups[g][:per_group])
    print(f"[eval] selected {len(idxs)} tuples over {len(groups)} (bin,k) groups")

    psnr_fn, lpips_fn = PSNR(reduction="none"), LPIPS(net="alex", reduction="none").to(device).eval()

    records = []
    vis_saved = 0
    t0 = time.time()
    for s in range(0, len(idxs), args.batch_size):
        batch = collate_tuples([ds[i] for i in idxs[s : s + args.batch_size]])
        B = batch["tgt_rgb"].shape[0]
        with torch.no_grad():
            tokens = model.generate(
                batch["pose_vec"].to(device),
                max_len=ds.img_w // 16 * (ds.img_h // 16),
                temperature=args.temperature, top_p=args.top_p,
                sat=batch["sat"].to(device),
                window_origin_xyz=batch["window_origin_xyz"].to(device),
                src_rgbs=batch["src_rgbs"].to(device),
                rel_poses=batch["rel_poses"].to(device),
                src_mask=batch["src_mask"].to(device),
                tgt_K=batch["tgt_K"].to(device),
                tgt_T_cam=batch["tgt_T_cam"].to(device),
            )
            gen = vq.decode(tokens.view(B, ds.img_h // 16, ds.img_w // 16)).clamp(-1, 1)  # (B,3,H,W) [-1,1]
        gt = batch["tgt_rgb"].to(device) * 2 - 1
        gen01 = (gen + 1) / 2
        gt01 = (gt + 1) / 2
        with torch.no_grad():
            p = psnr_fn(gen01, gt01).detach().cpu().numpy()
            l = lpips_fn(gen01, gt01).detach().cpu().numpy()
        for b in range(B):
            m = batch["meta"][b]
            records.append({
                "ckpt_step": step, "bin": m["actual_bin"], "K": int(batch["n_src"][b]),
                "dist_m": round(m["actual_source_target_dist_m"], 2),
                "requested_dist_m": round(m["dist_m"], 2), "dyaw_deg": round(m["dyaw_deg"], 2),
                "drive": m["drive"], "target_fid": m["target_fid"],
                "psnr": float(p[b]), "lpips": float(l[b]),
            })
        if vis_saved < 6:
            import cv2
            rows = []
            for b in range(min(B, 3)):
                src0 = (batch["src_rgbs"][b, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                g = (gen01[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
                t = (gt01[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
                rows.append(np.concatenate([src0[:, :, ::-1], t, g], axis=1))
            grid = np.concatenate(rows, axis=0)
            cv2.imwrite(str(out / f"vis_{vis_saved:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
            vis_saved += 1
        if (s // args.batch_size) % 2 == 0:
            el = time.time() - t0
            print(f"  {s+B}/{len(idxs)}  {el/(s+B):.1f}s/tuple", flush=True)

    with open(out / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # summary: bin x K
    print(f"\n===== summary (ckpt step {step}, n={len(records)}) =====")
    print(f"{'bin':>4} {'K':>2} {'n':>4} {'PSNR':>7} {'LPIPS':>7}")
    key = lambda r: (r["bin"], r["K"])
    for g in sorted({key(r) for r in records}):
        rs = [r for r in records if key(r) == g]
        print(f"{g[0]:>4} {g[1]:>2} {len(rs):>4} {np.mean([r['psnr'] for r in rs]):>7.2f} {np.mean([r['lpips'] for r in rs]):>7.3f}")
    rs = records
    print(f"{'all':>4} {'':>2} {len(rs):>4} {np.mean([r['psnr'] for r in rs]):>7.2f} {np.mean([r['lpips'] for r in rs]):>7.3f}")
    print(f"[eval] records -> {out/'records.jsonl'}")


if __name__ == "__main__":
    main()
