#!/usr/bin/env python3
"""Inference-only paired sweep of complete satellite-memory strength.

For a fixed completed B2 checkpoint, this applies M_sat -> lambda M_sat after
the satellite projection and target-relative PE are added.  It is a diagnostic
for over-weighted satellite memory, not a train-time operation or a formal
distribution-matched benchmark.  Source memory, target rays, selected tuples,
and AR sampling policy are fixed across lambda values.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from metrics import LPIPS, PSNR
from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from scripts.eval_icassp27_sat_ablate import build_model_from_ckpt, select_eval_indices
from world3d.data.kitti360_tuple_dataset import Kitti360TupleDataset, collate_tuples


def parse_scales(text: str) -> list[float]:
    scales = [float(value) for value in text.split(",")]
    if not scales or any(not np.isfinite(value) or value < 0 for value in scales):
        raise ValueError("--scales must be a non-empty comma-separated list of finite non-negative values")
    if len(set(scales)) != len(scales):
        raise ValueError("--scales must not repeat a value")
    return scales


def mean_metrics(records: list[dict]) -> dict:
    return {
        "n": len(records),
        "psnr": float(np.mean([r["psnr"] for r in records])),
        "lpips": float(np.mean([r["lpips"] for r in records])),
    }


def paired_vs_one(records: list[dict], *, seed: int, samples: int = 10_000) -> dict:
    """Paired effects of each lambda and B1 relative to lambda=1.0."""
    rows = defaultdict(dict)
    for record in records:
        rows[record["tuple_index"]][record["condition"]] = record
    baseline = "lambda_1"
    rng = np.random.default_rng(seed)
    output = {}
    for condition in sorted({r["condition"] for r in records}):
        if condition == baseline:
            continue
        pairs = [row for row in rows.values() if baseline in row and condition in row]
        psnr = np.asarray([row[condition]["psnr"] - row[baseline]["psnr"] for row in pairs])
        lpips = np.asarray([row[condition]["lpips"] - row[baseline]["lpips"] for row in pairs])
        idx = rng.integers(0, len(pairs), size=(samples, len(pairs)))
        output[condition] = {
            "n": len(pairs),
            "condition_minus_lambda1_psnr": float(psnr.mean()),
            "condition_minus_lambda1_lpips": float(lpips.mean()),
            "psnr_bootstrap_95ci": [float(x) for x in np.quantile(psnr[idx].mean(axis=1), [0.025, 0.975])],
            "lpips_bootstrap_95ci": [float(x) for x in np.quantile(lpips[idx].mean(axis=1), [0.025, 0.975])],
        }
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/icassp27_tc_b2/ckpt.pt")
    ap.add_argument("--b1_ckpt", default="runs/icassp27_tc_b1/ckpt.pt")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--scales", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--num_tuples", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval_tc_b2_strength_seed0")
    args = ap.parse_args()
    scales = parse_scales(args.scales)
    if 1.0 not in scales:
        raise ValueError("--scales must include 1.0 as the paired baseline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b2, b2_step = build_model_from_ckpt(args.ckpt, device)
    b1, b1_step = build_model_from_ckpt(args.b1_ckpt, device)
    if not (b2.use_sat and b2.use_src and not b1.use_sat and b1.use_src):
        raise ValueError("--ckpt must be B2 and --b1_ckpt must be B1")
    vq = PretrainedTokenizer("ckpts/maskgit-vqgan-imagenet-f16-256.bin").to(device).eval()
    ds = Kitti360TupleDataset(args.manifest, mode="eval", seed=args.seed)
    idxs = select_eval_indices(ds, args.num_tuples)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    psnr_fn = PSNR(reduction="none")
    lpips_fn = LPIPS(net="alex", reduction="none").to(device).eval()
    max_len = ds.img_w // 16 * (ds.img_h // 16)
    conditions = [(f"lambda_{scale:g}", b2, scale) for scale in scales] + [("b1", b1, None)]
    records = []

    print(f"[strength] B2 step={b2_step}, B1 step={b1_step}, tuples={len(idxs)}, seed={args.seed}")
    for condition, model, scale in conditions:
        torch.manual_seed(args.seed)
        t0 = time.time()
        print(f"[strength] {condition}", flush=True)
        for start in range(0, len(idxs), args.batch_size):
            selected = idxs[start : start + args.batch_size]
            batch = collate_tuples([ds[i] for i in selected])
            B = batch["tgt_rgb"].shape[0]
            kwargs = dict(
                max_len=max_len, temperature=args.temperature, top_p=args.top_p,
                sat=batch["sat"].to(device), window_origin_xyz=batch["window_origin_xyz"].to(device),
                src_rgbs=batch["src_rgbs"].to(device), rel_poses=batch["rel_poses"].to(device),
                src_mask=batch["src_mask"].to(device), tgt_K=batch["tgt_K"].to(device),
                tgt_T_cam=batch["tgt_T_cam"].to(device),
            )
            if scale is not None:
                kwargs["sat_memory_scale"] = scale
            with torch.no_grad():
                tokens = model.generate(batch["pose_vec"].to(device), **kwargs)
                gen = vq.decode(tokens.view(B, ds.img_h // 16, ds.img_w // 16)).clamp(-1, 1)
                gen01 = (gen + 1) / 2
                gt01 = batch["tgt_rgb"].to(device)
                psnr = psnr_fn(gen01, gt01).detach().cpu().numpy()
                lpips = lpips_fn(gen01, gt01).detach().cpu().numpy()
            for i in range(B):
                meta = batch["meta"][i]
                records.append({
                    "condition": condition, "sat_memory_scale": scale,
                    "tuple_index": int(selected[i]), "bin": meta["actual_bin"],
                    "dist_m": round(meta["actual_source_target_dist_m"], 2),
                    "psnr": float(psnr[i]), "lpips": float(lpips[i]),
                })
            completed = start + B
            print(f"  {completed}/{len(idxs)}  {(time.time() - t0) / completed:.1f}s/tuple", flush=True)

    with open(out / "records.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    summary = {
        "purpose": "inference-only satellite-memory strength diagnostic; lambda != 1 is distribution shift",
        "b2_checkpoint": args.ckpt, "b1_checkpoint": args.b1_ckpt,
        "b2_step": b2_step, "b1_step": b1_step, "seed": args.seed,
        "scales": scales, "selected_tuples": len(idxs), "conditions": {},
        "paired_effects_vs_lambda1": paired_vs_one(records, seed=args.seed),
    }
    for condition in [name for name, _, _ in conditions]:
        rows = [r for r in records if r["condition"] == condition]
        summary["conditions"][condition] = {
            "overall": mean_metrics(rows),
            "by_bin": {
                str(bin_id): mean_metrics([r for r in rows if r["bin"] == bin_id])
                for bin_id in sorted({r["bin"] for r in rows})
            },
        }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"[strength] records -> {out / 'records.jsonl'}")
    print(f"[strength] summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
