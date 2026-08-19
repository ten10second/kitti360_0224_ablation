#!/usr/bin/env python3
"""Make an interpretable paired table from completed grounding diagnostics.

This is deliberately inference-only: it consumes the same target-centric B2
``real``/``shuffle``/``b1`` records and reconstructs each deterministic tuple
to attach geometry and an observation-coverage proxy.  The proxy is the mean
per-target-patch maximum DINOv2 cosine similarity over all source patches.  It
is *not* ground-truth pixel overlap; it answers the narrower question of how
well the target appearance is represented by the given perspective inputs.

The output has one row per AR seed (96 rows for the completed two-seed run),
plus a seed-averaged per-tuple table for correlation analysis.  Positive
``scs_*`` and ``fusion_*`` signs always favour B2(real).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval_icassp27_sat_ablate import build_model_from_ckpt
from world3d.data.kitti360_tuple_dataset import Kitti360TupleDataset, collate_tuples


REQUIRED_MODES = ("real", "shuffle", "b1")


def record_seed(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    if not match:
        raise ValueError(f"cannot infer AR seed from path: {path}")
    return int(match.group(1))


def load_records(paths: Iterable[str]) -> list[dict]:
    rows = []
    for raw_path in paths:
        path = Path(raw_path)
        seed = record_seed(path)
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["ar_seed"] = seed
                rows.append(row)
    return rows


@torch.no_grad()
def dino_coverage_proxy(model, batch: dict, device: torch.device) -> list[dict]:
    """Return source-observation coverage proxies for every item in ``batch``.

    Source and target use the same frozen DINOv2 feature geometry used by the
    predictor's street branch.  Each target patch chooses its most similar
    patch across every valid source image; larger values mean source images
    provide a closer appearance correspondence.  It deliberately makes no
    claim about dense geometric visibility or semantic correctness.
    """
    tgt = F.interpolate(
        batch["tgt_rgb"].to(device), size=model.dino_src_size,
        mode="bilinear", align_corners=False,
    )
    src = batch["src_rgbs"].to(device)
    B, K = src.shape[:2]
    src = F.interpolate(
        src.flatten(0, 1), size=model.dino_src_size,
        mode="bilinear", align_corners=False,
    )
    tgt_feat = F.normalize(model.dino(tgt), dim=-1)                 # (B,N,D)
    src_feat = F.normalize(model.dino(src), dim=-1).view(B, K, -1, tgt_feat.shape[-1])
    similarities = torch.einsum("bnd,bkmd->bknm", tgt_feat, src_feat)
    valid = batch["src_mask"].to(device)[:, :, None, None]
    similarities = similarities.masked_fill(~valid, -torch.inf)
    best = similarities.amax(dim=(1, 3))                            # (B,N)
    return [
        {
            "dino_patch_overlap_proxy": float(best[i].mean().cpu()),
            "dino_patch_overlap_p25": float(torch.quantile(best[i], 0.25).cpu()),
            "dino_patch_overlap_fraction_ge_0_70": float((best[i] >= 0.70).float().mean().cpu()),
        }
        for i in range(B)
    ]


def build_feature_map(
    dataset: Kitti360TupleDataset,
    indexes: list[int],
    model,
    device: torch.device,
    batch_size: int,
) -> dict[int, dict]:
    output = {}
    for start in range(0, len(indexes), batch_size):
        selected = indexes[start : start + batch_size]
        batch = collate_tuples([dataset[index] for index in selected])
        coverage = dino_coverage_proxy(model, batch, device)
        for local, (item, index, coverage_row) in enumerate(zip(batch["meta"], selected, coverage)):
            actual_distance = float(batch["actual_source_target_dist_m"][local])
            requested_distance = float(item["dist_m"])
            dyaw = float(item["dyaw_deg"])
            output[index] = {
                "tuple_index": int(index),
                "requested_dist_m": requested_distance,
                "actual_dist_m": actual_distance,
                "K": int(batch["n_src"][local]),
                "dyaw_deg": dyaw,
                "curvature_proxy_deg_per_m": dyaw / max(actual_distance, 1e-6),
                "drive": item["drive"],
                "target_fid": int(item["target_fid"]),
                "source_fids": ",".join(str(fid) for fid in item["source_fids"]),
                **coverage_row,
            }
        print(f"[table] DINO coverage {min(start + len(selected), len(indexes))}/{len(indexes)}", flush=True)
    return output


def correlation(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"pearson_r": float("nan"), "spearman_rho": float("nan")}
    # Average ranks (rather than an ordinal rank) makes ties such as K=1/3
    # well-defined without adding an analysis dependency.
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=float)
        sorted_values = values[order]
        left = 0
        while left < len(values):
            right = left + 1
            while right < len(values) and sorted_values[right] == sorted_values[left]:
                right += 1
            result[order[left:right]] = (left + right - 1) / 2.0
            left = right
        return result
    return {
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "spearman_rho": float(np.corrcoef(rank(x), rank(y))[0, 1]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--records", nargs="+", default=[
            "runs/eval_tc_b2_grounding_final_seed0/records.jsonl",
            "runs/eval_tc_b2_grounding_final_seed1/records.jsonl",
        ],
    )
    ap.add_argument("--ckpt", default="runs/icassp27_tc_b2/ckpt.pt")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out", default="runs/analysis_tc_final/per_tuple")
    args = ap.parse_args()

    records = load_records(args.records)
    paired: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for row in records:
        mode = row["sat_memory_mode"]
        if mode in REQUIRED_MODES:
            paired[(int(row["ar_seed"]), int(row["tuple_index"]))][mode] = row
    missing = [key for key, modes in paired.items() if any(mode not in modes for mode in REQUIRED_MODES)]
    if missing:
        raise RuntimeError(f"missing paired conditions for {len(missing)} rows, e.g. {missing[:3]}")

    indexes = sorted({tuple_index for _, tuple_index in paired})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step = build_model_from_ckpt(args.ckpt, device)
    if not model.use_src:
        raise ValueError("--ckpt must expose the frozen DINO street encoder")
    dataset = Kitti360TupleDataset(args.manifest, mode="eval")
    features = build_feature_map(dataset, indexes, model, device, args.batch_size)

    rows = []
    for (seed, index), modes in sorted(paired.items()):
        feature = features[index]
        real, shuffle, b1 = (modes[mode] for mode in REQUIRED_MODES)
        if int(real["target_fid"]) != feature["target_fid"]:
            raise RuntimeError(f"tuple index {index} does not match reconstructed target frame")
        row = {
            "ar_seed": seed,
            **feature,
            "b2_real_psnr": float(real["psnr"]),
            "b2_shuffle_psnr": float(shuffle["psnr"]),
            "b1_psnr": float(b1["psnr"]),
            "scs_psnr": float(real["psnr"] - shuffle["psnr"]),
            "fusion_b2_minus_b1_psnr": float(real["psnr"] - b1["psnr"]),
            "b2_real_lpips": float(real["lpips"]),
            "b2_shuffle_lpips": float(shuffle["lpips"]),
            "b1_lpips": float(b1["lpips"]),
            "scs_lpips": float(shuffle["lpips"] - real["lpips"]),
            "fusion_b2_minus_b1_lpips": float(b1["lpips"] - real["lpips"]),
        }
        rows.append(row)

    aggregate = []
    numeric_means = [
        "requested_dist_m", "actual_dist_m", "K", "dyaw_deg", "curvature_proxy_deg_per_m",
        "dino_patch_overlap_proxy", "dino_patch_overlap_p25", "dino_patch_overlap_fraction_ge_0_70",
        "b2_real_psnr", "b2_shuffle_psnr", "b1_psnr", "scs_psnr", "fusion_b2_minus_b1_psnr",
        "b2_real_lpips", "b2_shuffle_lpips", "b1_lpips", "scs_lpips", "fusion_b2_minus_b1_lpips",
    ]
    by_index: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_index[row["tuple_index"]].append(row)
    for index, samples in sorted(by_index.items()):
        first = samples[0]
        aggregate.append({
            "tuple_index": index,
            "n_ar_seeds": len(samples),
            "drive": first["drive"], "target_fid": first["target_fid"], "source_fids": first["source_fids"],
            **{key: float(np.mean([sample[key] for sample in samples])) for key in numeric_means},
        })

    predictors = [
        "actual_dist_m", "K", "dyaw_deg", "curvature_proxy_deg_per_m",
        "dino_patch_overlap_proxy", "dino_patch_overlap_p25", "dino_patch_overlap_fraction_ge_0_70",
    ]
    outcomes = ["scs_psnr", "fusion_b2_minus_b1_psnr", "scs_lpips", "fusion_b2_minus_b1_lpips"]
    correlations = {
        outcome: {
            predictor: correlation(
                np.asarray([row[predictor] for row in aggregate]),
                np.asarray([row[outcome] for row in aggregate]),
            )
            for predictor in predictors
        }
        for outcome in outcomes
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_seed_paired_samples.csv", rows)
    write_csv(out / "per_tuple_seed_mean.csv", aggregate)
    with (out / "per_seed_paired_samples.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "purpose": "post-hoc paired analysis only; no model or training change",
        "b2_checkpoint": args.ckpt,
        "checkpoint_step": step,
        "n_seeded_paired_samples": len(rows),
        "n_unique_tuples": len(aggregate),
        "coverage_proxy_definition": (
            "mean target-patch maximum cosine similarity against all valid source DINO patches; "
            "an observation-coverage proxy, not ground-truth pixel overlap"
        ),
        "sign_convention": "positive scs/fusion values favour B2(real), including LPIPS",
        "predictor_correlations_seed_averaged_tuples": correlations,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[table] {len(rows)} paired seeded rows / {len(aggregate)} unique tuples -> {out}")
    for outcome, values in correlations.items():
        best = sorted(values.items(), key=lambda item: abs(item[1]["spearman_rho"]), reverse=True)[:2]
        print(f"[table] {outcome}: " + ", ".join(f"{name} rho={row['spearman_rho']:+.3f}" for name, row in best))


if __name__ == "__main__":
    main()
