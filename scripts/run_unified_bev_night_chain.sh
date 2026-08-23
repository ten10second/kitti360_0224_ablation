#!/usr/bin/env bash
# Night chain: wait for the 20k Stage A run, then run gate evals + Stage B variants.
set -uo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

PY="conda run --no-capture-output -n maskgit python"
LOG=runs/night_chain.log
echo "[chain] start $(date)" >> $LOG

# 1. Wait for the resumed Stage A run to finish (checkpoint line appears).
while ! grep -q "checkpoint=" runs/unified_bev_stage_a_multi_v3_resume.log 2>/dev/null; do
  sleep 60
done
sleep 30
echo "[chain] stage A resumed run finished $(date)" >> $LOG

COMMON="--manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl --drive 2013_05_28_drive_0007_sync --max_samples 128 --dense_sources 8 --sparse_sources 2 --max_points 4096 --hidden 256 --ray_samples 48 --min_target_spacing_m 5 --num_workers 6 --device cuda"
EVALCOMMON="--manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl --drive 2013_05_28_drive_0007_sync --max_samples 32 --dense_sources 8 --sparse_sources 2 --max_points 4096 --min_target_spacing_m 5 --device cuda"

# 2. Stage A gate eval (dense vs sparse, no stage B).
$PY scripts/eval_unified_bev_probe.py --stage_a runs/unified_bev_stage_a_multi_v3/stage_a.pt $EVALCOMMON \
  > runs/eval_stage_a_multi_v3.txt 2>&1
echo "[chain] stage A eval done $(date)" >> $LOG

# 3. Stage B variants: residual (proposed), coordinate_only (control), satellite_only (diag).
for FUSION in residual coordinate_only satellite_only; do
  $PY scripts/train_unified_bev_stage_b.py \
    --stage_a runs/unified_bev_stage_a_multi_v3/stage_a.pt \
    --out runs/unified_bev_stage_b_multi_${FUSION} --fusion $FUSION \
    --steps 10000 $COMMON \
    > runs/unified_bev_stage_b_multi_${FUSION}.log 2>&1
  $PY scripts/eval_unified_bev_probe.py \
    --stage_a runs/unified_bev_stage_a_multi_v3/stage_a.pt \
    --stage_b runs/unified_bev_stage_b_multi_${FUSION}/stage_b.pt $EVALCOMMON \
    > runs/eval_stage_b_multi_${FUSION}.txt 2>&1
  echo "[chain] stage B $FUSION done $(date)" >> $LOG
done

# 4. Summary.
{
  echo "=== Stage A gate (dense vs sparse) ==="; cat runs/eval_stage_a_multi_v3.txt
  for FUSION in residual coordinate_only satellite_only; do
    echo "=== Stage B $FUSION ==="; cat runs/eval_stage_b_multi_${FUSION}.txt
  done
} > runs/night_summary.txt
echo "[chain] ALL DONE $(date)" >> $LOG
