#!/usr/bin/env bash
# Round-2 TC Stage-A: independent 5k B2/B1/B0 pilots on the same split.
set -euo pipefail
cd "$(dirname "$0")/.."

TC_PY="${TC_PY:-/home/shizhm/anaconda3/envs/maskgit/bin/python}"
TC_CONFIG="configs/icassp27_target_centric_pilot.yaml"
TC_STEPS="${TC_STEPS:-5000}"

run_variant() {
    local name="$1"
    shift
    local output="runs/icassp27_tc_${name}_smoke${TC_STEPS}"
    mkdir -p "$output"
    echo "=== [TC smoke] ${name} (${TC_STEPS} steps) start: $(date) ==="
    "$TC_PY" -m world3d.train.train_icassp27 --config "$TC_CONFIG" --steps "$TC_STEPS" \
        --out_dir "$output" "$@" 2>&1 | tee "$output/train.log"
    echo "=== [TC smoke] ${name} complete: $(date) ==="
}

run_variant b2
run_variant b1 --use_sat false
run_variant b0 --use_src false
