#!/usr/bin/env python3
"""Distance-sliced paired analysis of completed target-centric diagnostics.

This reads the two completed 48-tuple satellite-grounding runs.  It reports
scene-specificity SCS = B2(real) - B2(shuffle), plus fusion delta =
B2(real) - B1, in actual source-target distance bins.  Two AR seeds are first
averaged per tuple, then confidence intervals bootstrap tuples (not individual
sampling draws), preserving the paired counterfactual design.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BIN_LABELS = {0: "3.5m [2,5)", 1: "7.5m [5,10)", 2: "15m [10,20]"}


def paired_by_bin(records: list[dict], condition: str, *, seed: int, samples: int = 10_000) -> dict:
    """Return real-condition effects, clustered by tuple across AR seeds."""
    per_tuple: dict[tuple[int, int], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        per_tuple[(int(r["bin"]), int(r["tuple_index"]))][r["sat_memory_mode"]].append(r)

    summary = {}
    rng = np.random.default_rng(seed)
    for bin_id in sorted({key[0] for key in per_tuple}):
        pairs = []
        for (current_bin, _), modes in per_tuple.items():
            if current_bin != bin_id or "real" not in modes or condition not in modes:
                continue
            real = modes["real"]
            other = modes[condition]
            if len(real) != len(other):
                raise RuntimeError("unpaired seed count in diagnostic records")
            # Runs are stored in seed order and each mode uses the same reset RNG.
            pairs.append((
                float(np.mean([a["psnr"] - b["psnr"] for a, b in zip(real, other)])),
                float(np.mean([b["lpips"] - a["lpips"] for a, b in zip(real, other)])),
            ))
        values = np.asarray(pairs)
        if not len(values):
            continue
        idx = rng.integers(0, len(values), size=(samples, len(values)))
        boot = values[idx].mean(axis=1)
        summary[BIN_LABELS[bin_id]] = {
            "bin": bin_id,
            "n_tuples": len(values),
            "real_minus_condition_psnr": float(values[:, 0].mean()),
            "condition_minus_real_lpips": float(values[:, 1].mean()),
            "psnr_win_rate": float((values[:, 0] > 0).mean()),
            "lpips_win_rate": float((values[:, 1] > 0).mean()),
            "psnr_bootstrap_95ci": [float(x) for x in np.quantile(boot[:, 0], [0.025, 0.975])],
            "lpips_bootstrap_95ci": [float(x) for x in np.quantile(boot[:, 1], [0.025, 0.975])],
        }
    return summary


def strength_sweep(records: list[dict], *, seed: int, samples: int = 10_000) -> dict:
    """Aggregate lambda effects across AR seeds, resampling tuples as clusters."""
    by_tuple: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_tuple[int(r["tuple_index"])][r["condition"]].append(r)
    rng = np.random.default_rng(seed)
    out = {}
    baseline = "lambda_1"
    for condition in sorted({r["condition"] for r in records}):
        rows = [r for r in records if r["condition"] == condition]
        overall = {
            "n_seeded_samples": len(rows),
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "lpips": float(np.mean([r["lpips"] for r in rows])),
        }
        if condition == baseline:
            out[condition] = {"overall": overall}
            continue
        pairs = []
        for mode_rows in by_tuple.values():
            if baseline not in mode_rows or condition not in mode_rows:
                continue
            base, other = mode_rows[baseline], mode_rows[condition]
            if len(base) != len(other):
                raise RuntimeError("unpaired seed count in strength records")
            pairs.append((
                float(np.mean([b["psnr"] - a["psnr"] for a, b in zip(base, other)])),
                float(np.mean([b["lpips"] - a["lpips"] for a, b in zip(base, other)])),
            ))
        values = np.asarray(pairs)
        idx = rng.integers(0, len(values), size=(samples, len(values)))
        boot = values[idx].mean(axis=1)
        out[condition] = {
            "overall": overall,
            "n_tuples": len(values),
            "condition_minus_lambda1_psnr": float(values[:, 0].mean()),
            "condition_minus_lambda1_lpips": float(values[:, 1].mean()),
            "psnr_bootstrap_95ci": [float(x) for x in np.quantile(boot[:, 0], [0.025, 0.975])],
            "lpips_bootstrap_95ci": [float(x) for x in np.quantile(boot[:, 1], [0.025, 0.975])],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--records", nargs="+", default=[
            "runs/eval_tc_b2_grounding_final_seed0/records.jsonl",
            "runs/eval_tc_b2_grounding_final_seed1/records.jsonl",
        ],
    )
    ap.add_argument(
        "--strength_records", nargs="+", default=[
            "runs/eval_tc_b2_strength_final_seed0/records.jsonl",
            "runs/eval_tc_b2_strength_final_seed1/records.jsonl",
        ],
    )
    ap.add_argument("--out", default="runs/analysis_tc_final/distance_paired.json")
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    records = []
    for path in map(Path, args.records):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    strength_records = []
    for path in map(Path, args.strength_records):
        strength_records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    out = {
        "inputs": args.records,
        "seeded_records": len(records),
        "scs_real_minus_shuffle": paired_by_bin(records, "shuffle", seed=args.seed),
        # Signs deliberately match Δ_fusion = B2(real) - B1.  Thus positive
        # PSNR and positive LPIPS effect both favour B2(real).
        "fusion_b2_real_minus_b1": paired_by_bin(records, "b1", seed=args.seed + 1),
        "strength_sweep_vs_lambda1": strength_sweep(strength_records, seed=args.seed + 2),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    print("distance-sliced paired effects (positive favours B2(real))")
    for title, section in out.items():
        if not isinstance(section, dict) or title == "inputs":
            continue
        print(f"\n{title}")
        for label, row in section.items():
            if title == "strength_sweep_vs_lambda1":
                overall = row["overall"]
                if label == "lambda_1":
                    print(f"  {label}: PSNR={overall['psnr']:.3f} LPIPS={overall['lpips']:.4f}")
                else:
                    print(
                        f"  {label}: ΔPSNR={row['condition_minus_lambda1_psnr']:+.3f} "
                        f"CI={row['psnr_bootstrap_95ci']}; "
                        f"ΔLPIPS={row['condition_minus_lambda1_lpips']:+.4f}"
                    )
                continue
            print(
                f"  {label}: ΔPSNR={row['real_minus_condition_psnr']:+.3f} "
                f"CI={row['psnr_bootstrap_95ci']}; "
                f"ΔLPIPS={row['condition_minus_real_lpips']:+.4f}"
            )
    print(f"[analysis] -> {out_path}")


if __name__ == "__main__":
    main()
