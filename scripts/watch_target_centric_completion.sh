#!/usr/bin/env bash
# Wait for the serial pilot launcher, then evaluate only a successfully completed B0/B1/B2 triplet.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="runs/icassp27_tc_b0/train.log"
while ! grep -q '^done$' "$LOG" 2>/dev/null; do
    sleep 60
done

for variant in b0 b1 b2; do
    test -s "runs/icassp27_tc_${variant}/ckpt.pt"
done

exec scripts/run_target_centric_final_eval.sh
