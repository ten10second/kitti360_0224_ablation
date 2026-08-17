#!/usr/bin/env python3
"""Cross-attention mass diagnostic: is the satellite branch ignored?

Hooks every decoder layer's multihead_attn, captures attention weights over
memory = [e_pose(1) | sat(Ns) | src(K*Nv)], aggregates attention MASS per
segment over the 640 target-query positions.

Interpretation grid (sat_share vs uniform token-count share):
  ratio = sat_share / sat_uniform
    ratio << 1  -> satellite tokens ignored  => injection failure (method fixable)
    ratio >= ~1 -> attended but no gain       => information redundancy (task dead)

Also runs B0 (sat is its ONLY memory besides pose) as the "healthy satellite
usage" reference for what meaningful sat attention looks like.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from world3d.data.kitti360_tuple_dataset import Kitti360TupleDataset, collate_tuples
from scripts.eval_icassp27_binned import build_model_from_ckpt


@torch.no_grad()
def forward_with_attn(model, inp, pose, sat, origin, src, rel, mask, tK, tT):
    """Manual layer loop mirroring ICASSP27Predictor.forward, returning
    per-layer cross-attention weights (B, L, S), head-averaged."""
    memory, key_padding = model.build_memory(pose, sat, origin, src, rel, mask)
    B, L = inp.shape
    x = model.token_embed(inp) + model.pos_embed[:, :L]
    if model.geo == "raymap":
        x = x + model._token_ray_pe(tK, tT, origin, L)
    causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
    weights = []
    for blk in model.blocks:
        sa, _ = blk.self_attn(x, x, x, attn_mask=causal, need_weights=False)
        x = blk.norm1(x + blk.dropout1(sa))
        ca, w = blk.multihead_attn(x, memory, memory, key_padding_mask=key_padding, need_weights=True)
        x = blk.norm2(x + blk.dropout2(ca))
        ff = blk.linear2(blk.dropout(blk.activation(blk.linear1(x))))
        x = blk.norm3(x + blk.dropout3(ff))
        weights.append(w.detach())  # (B, L, S)
    return model.head(model.norm(x)), weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_b2", default="runs/icassp27_b2_pilot/ckpt.pt")
    ap.add_argument("--ckpt_b0", default="runs/icassp27_b0_pilot/ckpt.pt")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--num_tuples", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out", default="runs/diag_attn")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vq = PretrainedTokenizer("ckpts/maskgit-vqgan-imagenet-f16-256.bin").to(device).eval()

    ds = Kitti360TupleDataset(args.manifest, mode="eval", seed=0)
    # spread over bins x K slots (same scheme as eval script)
    groups = defaultdict(list)
    for i in range(len(ds)):
        k_slot = i % len(ds.eval_k)
        spec = ds.tuples[i // len(ds.eval_k)]
        groups[(next(j for j, (lo, hi) in enumerate(ds.bins) if lo <= spec.dist_m < hi), k_slot)].append(i)
    per = max(1, args.num_tuples // len(groups))
    idxs = []
    for g in sorted(groups):
        idxs.extend(groups[g][:per])

    for tag, ckpt in [("B2", args.ckpt_b2), ("B0", args.ckpt_b0)]:
        model, step = build_model_from_ckpt(ckpt, device)
        Ns = model.sat_grid[0] * model.sat_grid[1]
        Nv = model.src_tokens_per_view

        agg = defaultdict(lambda: defaultdict(float))  # K -> seg -> total mass
        agg_n = defaultdict(int)
        layer_sat = defaultdict(float)
        layer_n = 0
        per_bin = defaultdict(lambda: defaultdict(float))
        per_bin_n = defaultdict(int)

        for s in range(0, len(idxs), args.batch_size):
            batch = collate_tuples([ds[i] for i in idxs[s:s + args.batch_size]])
            B = batch["tgt_rgb"].shape[0]
            rgb = batch["tgt_rgb"].to(device) * 2 - 1
            with torch.no_grad():
                tok = vq.encode(rgb)
                if tok.dim() == 4:
                    tok = tok.squeeze(1)
                tok = tok.view(B, ds.img_h // 16, ds.img_w // 16).flatten(1)
                inp, _ = model.make_teacher_forcing(tok)
                _, weights = forward_with_attn(
                    model, inp,
                    batch["pose_vec"].to(device),
                    batch["sat"].to(device),
                    batch["window_origin_xyz"].to(device),
                    batch["src_rgbs"].to(device),
                    batch["rel_poses"].to(device),
                    batch["src_mask"].to(device),
                    batch["tgt_K"].to(device),
                    batch["tgt_T_cam"].to(device),
                )
            for li, w in enumerate(weights):  # w: (B, L, S)
                S = w.shape[-1]
                sat_slice = slice(1, 1 + Ns)
                src_slice = slice(1 + Ns, S)
                pose_mass = w[:, :, 0]  # (B, L) — single pose token
                sat_mass = w[:, :, sat_slice].sum(-1)
                src_mass = w[:, :, src_slice].sum(-1)
                tot = pose_mass + sat_mass + src_mass + 1e-9
                for b in range(B):
                    K = int(batch["n_src"][b])
                    bin_ = batch["meta"][b]["bin"]
                    agg[K]["pose"] += float((pose_mass[b] / tot[b]).mean())
                    agg[K]["sat"] += float((sat_mass[b] / tot[b]).mean())
                    agg[K]["src"] += float((src_mass[b] / tot[b]).mean())
                    agg_n[K] += 1
                    per_bin[bin_]["sat"] += float((sat_mass[b] / tot[b]).mean())
                    per_bin[bin_]["src"] += float((src_mass[b] / tot[b]).mean())
                    per_bin_n[bin_] += 1
                layer_sat[li] += float((sat_mass / tot).mean())
            layer_n += 1

        print(f"\n===== {tag} (step {step}) cross-attention mass (mean over queries/heads/layers) =====")
        print(f"{'K':>2} {'n':>4} | {'pose':>7} {'sat':>7} {'src':>7} | sat_uniform  sat_ratio")
        for K in sorted(agg):
            n = agg_n[K]
            S = 1 + Ns + K * Nv
            u = Ns / S
            r = (agg[K]["sat"] / n) / u
            print(f"{K:>2} {n:>4} | {agg[K]['pose']/n:>7.3f} {agg[K]['sat']/n:>7.3f} {agg[K]['src']/n:>7.3f} | {u:>11.3f}  {r:>7.2f}x")
        print("per-layer sat ratio (uniform=1.0):", {l: round(v / max(1, layer_n), 2) for l, v in sorted(layer_sat.items())})
        if tag == "B2":
            print("per-bin sat share:", {b: round(per_bin[b]["sat"] / max(1, per_bin_n[b]), 3) for b in sorted(per_bin)})
        with open(out / f"{tag.lower()}_attn.json", "w") as f:
            json.dump({str(k): {s: v / agg_n[k] for s, v in agg[k].items()} for k in agg}, f, indent=1)


if __name__ == "__main__":
    main()
