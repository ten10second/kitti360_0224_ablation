#!/usr/bin/env bash
# Post-pilot evaluation for the target-centric B0/B1/B2 run.
set -euo pipefail
cd "$(dirname "$0")/.."

TC_PY="${TC_PY:-/home/shizhm/anaconda3/envs/maskgit/bin/python}"
TC_MANIFEST="${TC_MANIFEST:-dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl}"
TC_BATCH="${TC_BATCH:-4}"

for seed in 0 1; do
    for variant in b2 b1 b0; do
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$TC_PY" -m scripts.eval_icassp27_binned \
            --ckpt "runs/icassp27_tc_${variant}/ckpt.pt" --manifest "$TC_MANIFEST" \
            --num_tuples 48 --batch_size "$TC_BATCH" --seed "$seed" \
            --out "runs/eval_tc_${variant}_final_seed${seed}"
    done
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$TC_PY" -m scripts.eval_satellite_grounding \
        --ckpt runs/icassp27_tc_b2/ckpt.pt --b1_ckpt runs/icassp27_tc_b1/ckpt.pt \
        --manifest "$TC_MANIFEST" --num_tuples 48 --batch_size "$TC_BATCH" --seed "$seed" \
        --vis_batches 2 --out "runs/eval_tc_b2_grounding_final_seed${seed}"
done
