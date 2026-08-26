#!/usr/bin/env python3
"""QA the datum-relative height targets of a world-state scene blob.

Renders the satellite tile, the height map (optionally against a buggy
pre-fix blob), the height histogram, footprint contours on the satellite,
and a 3-D surface.  This exists because KITTI-360 world-Z is absolute map
altitude: a height clip placed before datum subtraction flattens the whole
valid tile to one constant, and a single glance at the map/histogram pair
catches that failure mode.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _height_panel(ax, blob, vmin, vmax, title):
    import torch
    h = blob["height"].float().squeeze(0).numpy()
    v = blob["world_valid"].bool().squeeze(0).numpy()
    masked = np.where(v, h, np.nan)
    im = ax.imshow(masked, origin="lower", vmin=vmin, vmax=vmax, cmap="turbo")
    hv = h[v]
    stats = f"std={hv.std():.3f} uniq={len(np.unique(hv))}" if hv.size else "empty"
    ax.set_title(f"{title}\n[{hv.min():.1f}, {hv.max():.1f}] m  {stats}")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True, help="scene blob built by build_world_state_targets.py")
    ap.add_argument("--old_buggy", default=None,
                    help="pre-fix blob for before/after comparison (e.g. buggy_height_backup/)")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside the blob)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    warnings.filterwarnings("ignore")
    blob = torch.load(args.blob, map_location="cpu")
    old = torch.load(args.old_buggy, map_location="cpu") if args.old_buggy else None

    h = blob["height"].float().squeeze(0).numpy()
    v = blob["world_valid"].bool().squeeze(0).numpy()
    hv = h[v]
    datum = float(blob["z_datum_m"])
    vmin, vmax = -5.0, max(5.0, float(np.nanpercentile(hv, 99.5)))

    fig = plt.figure(figsize=(16, 9))
    ax_sat = fig.add_subplot(2, 3, 1)
    sat = blob["satellite_bev"].permute(1, 2, 0).numpy()
    ax_sat.imshow(np.clip(sat, 0, 1), origin="lower")
    ax_sat.set_title(f"satellite BEV (north up)\ndatum = {datum:.2f} m (LiDAR centers median)")

    if old is not None:
        ax_old = fig.add_subplot(2, 3, 2)
        im = _height_panel(ax_old, old, vmin, vmax, "height BEFORE fix (clipped absolute Z)")
        fig.colorbar(im, ax=ax_old, fraction=0.046, label="m rel. datum")
    ax_new = fig.add_subplot(2, 3, 3)
    im = _height_panel(ax_new, blob, vmin, vmax, "height AFTER fix (clip on relative)")
    fig.colorbar(im, ax=ax_new, fraction=0.046, label="m rel. datum")

    ax_hist = fig.add_subplot(2, 3, 4)
    bins = np.linspace(vmin - 3, vmax + 30, 160)
    if old is not None:
        ho = old["height"].float().squeeze(0).numpy()[v]
        ax_hist.hist(ho, bins=bins, alpha=0.55, label=f"before (uniq={len(np.unique(ho))})", color="tab:red")
    ax_hist.hist(hv, bins=bins, alpha=0.6, label=f"after (uniq={len(np.unique(hv))})", color="tab:blue")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("height rel. datum [m]")
    ax_hist.axvline(-2.0, color="k", ls=":", lw=1, label="clip floor -2 m")
    ax_hist.legend(fontsize=8)
    ax_hist.set_title("height histogram, valid cells")

    ax_fp = fig.add_subplot(2, 3, 5)
    gray = sat.mean(axis=2)
    ax_fp.imshow(gray, cmap="gray", origin="lower")
    masked = np.where(v, h, np.nan)
    levels = [2.0, 5.0]
    cs = ax_fp.contour(masked, levels=levels, origin="lower",
                       colors=["yellow", "red"], linewidths=1.2)
    ax_fp.clabel(cs, fmt={lv: f"{lv:g} m" for lv in levels}, fontsize=8)
    ax_fp.set_title("raised-surface footprint on satellite\n(yellow 2 m, red 5 m = roofs/walls)")

    ax3d = fig.add_subplot(2, 3, 6, projection="3d")
    step = 2
    yy, xx = np.mgrid[0:h.shape[0]:step, 0:h.shape[1]:step]
    surf = masked[::step, ::step]
    ax3d.plot_surface(xx, yy, surf, cmap="terrain", vmin=vmin, vmax=vmax,
                      rstride=2, cstride=2, linewidth=0, antialiased=False)
    ax3d.set_title("height surface (rel. datum)")
    ax3d.set_zlim(vmin, vmax)

    for ax in (ax_sat, ax_new, ax_fp):
        ax.set_xticks([])
        ax.set_yticks([])
    if old is not None:
        ax_old.set_xticks([])
        ax_old.set_yticks([])

    out = Path(args.out) if args.out else Path(args.blob).parent / "qa" / "height_target_qa.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[qa] wrote {out}")
    print(f"[qa] datum={datum:.2f} valid={int(v.sum())} "
          f"min={hv.min():.2f} mode={_mode(hv):.2f} p50={np.median(hv):.2f} "
          f"p99={np.percentile(hv, 99):.2f} max={hv.max():.2f} "
          f"floor_frac={float((hv <= -1.999).mean()):.3f} ceil_frac={float((hv >= 39.999).mean()):.3f}")


def _mode(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=96)
    i = int(hist.argmax())
    return float(0.5 * (edges[i] + edges[i + 1]))


if __name__ == "__main__":
    main()
