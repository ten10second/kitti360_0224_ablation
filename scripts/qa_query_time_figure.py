#!/usr/bin/env python3
"""Query-time figure: map quality as a function of WHEN you query it.

Persistent chain = a full curve (the state exists and is queryable at every
t); one-shot = a single point at the end (nothing to query before T).
Optional XY-init persistent curve separates "queryable early" (persistence)
from "content worth querying" (satellite gain, E2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="runs/world_state_assim_depth_smoke")
    ap.add_argument("--persistent", default="eval_aligned.jsonl")
    ap.add_argument("--persistent_xy", default="eval_xy.jsonl")
    ap.add_argument("--oneshot", default="eval_one_shot.jsonl")
    ap.add_argument("--metric", default="height_visited_mae",
                    help="row field plotted on the y axis")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.eval_dir)

    per = _load(d / args.persistent)
    one = _load(d / args.oneshot)
    xy = _load(d / args.persistent_xy) if (d / args.persistent_xy).exists() else None

    t_per = [r["version"] for r in per if r.get(args.metric) is not None]
    y_per = [r[args.metric] for r in per if r.get(args.metric) is not None]
    trav = [r.get("traversed_m", r["version"]) for r in per if r.get(args.metric) is not None]

    one_rows = [r for r in one if r.get(args.metric) is not None]
    t_one = max(r["version"] for r in one_rows)
    y_one = [r[args.metric] for r in one_rows if r["version"] == t_one][0]
    trav_one = [r.get("traversed_m", r["version"]) for r in one_rows if r["version"] == t_one][0]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(trav, y_per, "o-", color="#1a6fae", lw=2.0, ms=5, zorder=3,
            label="persistent world state (queryable at every $t$)")
    if xy:
        t_xy = [r["version"] for r in xy if r.get(args.metric) is not None]
        y_xy = [r[args.metric] for r in xy if r.get(args.metric) is not None]
        trav_xy = [r.get("traversed_m", r["version"]) for r in xy if r.get(args.metric) is not None]
        ax.plot(trav_xy, y_xy, "s--", color="#c98a1d", lw=1.5, ms=4, zorder=2,
                label="persistent, XY init (queryable, no geographic content)")
    ax.plot([trav_one], [y_one], "o", color="#b2182b", ms=11, zorder=4,
            label="one-shot assimilation (map first exists at $t=T$)")
    ax.plot([0, trav_one], [y_one, y_one], ":", color="#b2182b", lw=1.2, zorder=1)

    # shade the "free" area: queryable quality before one-shot exists
    xs = np.array(trav)
    ys = np.array(y_per)
    ax.fill_between(xs, ys, y_one, where=ys < y_one, color="#1a6fae", alpha=0.12,
                    label="value of querying early (vs. waiting for $T$)")

    ax.set_xlabel("traversed distance (m)", fontsize=11)
    ylab = {"height_visited_mae": "visited-region height MAE (m)"}.get(args.metric, args.metric)
    ax.set_ylabel(ylab, fontsize=11)
    ax.invert_yaxis()  # lower error = better; up means better
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    scene = per[0]["scene_id"].split("__")[0].replace("2013_05_28_drive_", "").replace("_sync", "")
    ax.set_title(f"map quality at query time — held-out scene {scene} (up = better)", fontsize=11)
    fig.tight_layout()
    out = Path(args.out) if args.out else d / "qa_query_time.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
