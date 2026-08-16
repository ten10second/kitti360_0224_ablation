#!/bin/bash
# ICASSP27 pilot queue: B2 (main) -> B1 (src-only) -> B0 (sat-only), sequential on one GPU.
# 20k steps each (~2.6 h/config). Sanity checks (doc pitfall #6) run on these ckpts:
#   B0 error curve should be ~flat vs distance; B1 near-bin should beat B0.
set -e
cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/maskgit/bin/python

echo "=== [queue] B2 (sat+src+raymap) start: $(date) ==="
$PY -m world3d.train.train_icassp27 --config configs/icassp27_pilot.yaml \
    --out_dir runs/icassp27_b2_pilot 2>&1 | grep --line-buffered -E "^\[|^step|done"

echo "=== [queue] B1 (src-only) start: $(date) ==="
$PY -m world3d.train.train_icassp27 --config configs/icassp27_pilot.yaml --use_sat false \
    --out_dir runs/icassp27_b1_pilot 2>&1 | grep --line-buffered -E "^\[|^step|done"

echo "=== [queue] B0 (sat-only) start: $(date) ==="
$PY -m world3d.train.train_icassp27 --config configs/icassp27_pilot.yaml --use_src false \
    --out_dir runs/icassp27_b0_pilot 2>&1 | grep --line-buffered -E "^\[|^step|done"

echo "=== [queue] all done: $(date) ==="
