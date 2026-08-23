#!/usr/bin/env bash
# Remaining Stage B chain, fully detached (survives session shutdowns).
set -uo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

PY="conda run --no-capture-output -n maskgit python"
LOG=runs/day_chain.log
echo "[day] start $(date)" >> $LOG

COMMON="--manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl --drive 2013_05_28_drive_0007_sync --max_samples 128 --dense_sources 8 --sparse_sources 2 --max_points 4096 --hidden 256 --ray_samples 48 --min_target_spacing_m 5 --num_workers 6 --device cuda"
EVALCOMMON="--manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl --drive 2013_05_28_drive_0007_sync --max_samples 32 --dense_sources 8 --sparse_sources 2 --max_points 4096 --min_target_spacing_m 5 --device cuda"

run_retry() {  # retry once on transient failures
  for i in 1 2; do
    "$@" && return 0
    echo "[day] retry after failure: $*" >> $LOG
    sleep 20
  done
  return 1
}

# 1. Re-eval residual with paired stats.
run_retry $PY scripts/eval_unified_bev_probe.py \
  --stage_a runs/unified_bev_stage_a_multi_v3/stage_a.pt \
  --stage_b runs/unified_bev_stage_b_multi_residual/stage_b.pt $EVALCOMMON \
  > runs/eval_stage_b_multi_residual.txt 2>&1
echo "[day] residual eval (paired) done $(date)" >> $LOG

# 2. coordinate_only + satellite_only variants.
for FUSION in coordinate_only satellite_only; do
  run_retry $PY scripts/train_unified_bev_stage_b.py \
    --stage_a runs/unified_bev_stage_a_multi_v3/stage_a.pt \
    --out runs/unified_bev_stage_b_multi_${FUSION} --fusion $FUSION \
    --steps 10000 $COMMON \
    > runs/unified_bev_stage_b_multi_${FUSION}.log 2>&1
  run_retry $PY scripts/eval_unified_bev_probe.py \
    --stage_a runs/unified_bev_stage_a_multi_v3/stage_a.pt \
    --stage_b runs/unified_bev_stage_b_multi_${FUSION}/stage_b.pt $EVALCOMMON \
    > runs/eval_stage_b_multi_${FUSION}.txt 2>&1
  echo "[day] stage B $FUSION done $(date)" >> $LOG
done

{
  echo "=== Stage A gate (dense vs sparse) ==="; cat runs/eval_stage_a_multi_v3.txt
  for FUSION in residual coordinate_only satellite_only; do
    echo "=== Stage B $FUSION ==="; cat runs/eval_stage_b_multi_${FUSION}.txt
  done
} > runs/final_summary.txt
echo "[day] ALL DONE $(date)" >> $LOG
