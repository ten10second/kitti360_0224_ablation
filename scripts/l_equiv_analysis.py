#!/usr/bin/env python3
"""Ground-equivalent acquisition length from chunk-mode eval records.

Pairs two eval JSONL record sets (e.g. aligned satellite vs a control) by
(anchor_fid, kept_chunks, query_chunk) and reports, per kept-chunk count K:
  * per-metric paired means for the sparse (ground-only), full (completed),
    and dense branches restricted to missing-chunk queries;
  * L_equiv(K): how many additional whole ground chunks the satellite
    completion is worth, i.e. the smallest K' whose ground-only branch
    reaches the completed branch's quality at K, minus K, times chunk arc.

This reframes "how many frames is one satellite image worth" as meters of
ground-trajectory coverage, the quantity the claim is actually about.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


HIGHER_IS_BETTER = ("psnr", "delta1")


def load(path: str) -> Dict[tuple, dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return {
        (r["drive"], int(r["anchor_fid"]), int(r["kept_chunks"]), int(r["query_chunk"])): r
        for r in rows
    }


def better_is_higher(metric: str) -> bool:
    return any(token in metric for token in HIGHER_IS_BETTER)


def mean_finite(values: List[float]) -> float:
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else float("nan")


def at_least_as_good(a: float, b: float, higher: bool) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return a >= b if higher else a <= b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="treatment records (e.g. aligned satellite)")
    ap.add_argument("--b", default=None, help="optional control records (e.g. random tile)")
    ap.add_argument("--metrics", default=(
        "full_height_hole_core_mae,sparse_height_hole_core_mae,"
        "full_absrel,sparse_absrel,full_psnr,sparse_psnr"
    ))
    args = ap.parse_args()
    a = load(args.a)
    b = load(args.b) if args.b else None
    metrics = [m for m in args.metrics.split(",") if m]
    keys = sorted(set(a) & (set(b) if b else set(a)))
    if not keys:
        raise SystemExit("no paired records")
    kept_values = sorted({k[2] for k in keys})
    chunk_arc = None
    for row in a.values():
        chunk_arc = float(mean_finite(row.get("chunk_arc_m", [float("nan")])))
        break

    print(f"a={args.a}" + (f"\nb={args.b}" if args.b else ""))
    print(f"paired_rows={len(keys)} kept_chunks={kept_values} chunk_arc_m={chunk_arc:.2f}")
    report: Dict[str, dict] = {}
    for K in kept_values:
        rows_a = [a[k] for k in keys if k[2] == K]
        rows_b = [b[k] for k in keys if k[2] == K] if b else None
        block = {"paired_rows": len(rows_a)}
        for metric in metrics:
            va = mean_finite([r[metric] for r in rows_a if r.get(metric) is not None])
            if rows_b is not None:
                vb = mean_finite([r[metric] for r in rows_b if r.get(metric) is not None])
                block[f"{metric}|A"] = round(va, 6)
                block[f"{metric}|B"] = round(vb, 6)
                block[f"{metric}|A-B"] = round(va - vb, 6)
            else:
                block[metric] = round(va, 6)
        report[f"K={K}"] = block

    # L_equiv on the primary hole metric (and any requested paired A/B delta)
    for metric in metrics:
        if "hole_core" not in metric and "absrel" not in metric and "psnr" not in metric:
            continue
        full_curve = {}
        sparse_curve = {}
        for K in kept_values:
            rows = [a[k] for k in keys if k[2] == K]
            full_key = f"full_{metric.split('full_')[-1]}"
            sparse_key = f"sparse_{metric.split('full_')[-1]}"
            if any(r.get(full_key) is not None for r in rows):
                full_curve[K] = mean_finite([
                    r[full_key] for r in rows if r.get(full_key) is not None])
                sparse_curve[K] = mean_finite([
                    r[sparse_key] for r in rows if r.get(sparse_key) is not None])
        if len(full_curve) < 2:
            continue
        higher = better_is_higher(metric)
        for K, value in sorted(full_curve.items()):
            match = next(
                (Kp for Kp in sorted(sparse_curve)
                 if Kp >= K and at_least_as_good(sparse_curve[Kp], value, higher)),
                None,
            )
            delta_chunks = (match - K) if match is not None else None
            report[f"K={K}"][f"L_equiv_{metric}"] = (
                f"{delta_chunks} chunks = {delta_chunks * chunk_arc:.1f} m"
                if delta_chunks is not None else "beyond-curve"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
