#!/usr/bin/env python3
"""Paired tile bootstrap comparison between two eval JSONL record sets."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(path: str) -> dict:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    return {(r["drive"], r["target_fid"]): r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="records A (treatment)")
    ap.add_argument("--b", required=True, help="records B (control)")
    ap.add_argument("--keys", default="full_psnr,full_absrel,full_delta1,full_latent_l1,nadir_l1_full")
    ap.add_argument("--bootstraps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    keys = [(k, "up") if k in ("full_psnr", "full_delta1") else (k, "down")
            for k in args.keys.split(",")]
    paired = [(a[k], b[k]) for k in sorted(set(a) & set(b))]
    print(f"a={args.a}")
    print(f"b={args.b}")
    print(f"paired_tiles={len(paired)}")
    rng = random.Random(args.seed)
    for key, better in keys:
        if key not in paired[0][0] or key not in paired[0][1]:
            continue  # legacy records may lack newer keys (e.g. nadir_l1_*)
        diffs = [float(ra[key]) - float(rb[key]) for ra, rb in paired]
        n = len(diffs)
        mean = sum(diffs) / n
        samples = []
        for _ in range(args.bootstraps):
            picks = [diffs[rng.randrange(n)] for _ in range(n)]
            samples.append(sum(picks) / n)
        samples.sort()
        lo, hi = samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]
        wins = sum(d > 0 for d in diffs)
        significant = (lo > 0) if better == "up" else (hi < 0)
        print(f"{key:>18s}: mean={mean:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  "
              f"wins={wins}/{n}  better={'A' if better == 'up' else 'B(low)'}  "
              f"{'SIGNIFICANT' if significant else 'ns'}")


if __name__ == "__main__":
    main()
