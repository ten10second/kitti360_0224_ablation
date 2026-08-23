#!/usr/bin/env bash
# Fusion-identity fix chain: retrain Stage B (residual + coordinate_only) on the
# height-fix Stage A encoder, then C1 / identity / B7(road frame) / B8 probes.
set -euo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

PY=(conda run --no-capture-output -n maskgit python)
TAG=fusionfix_20260821
STAGE_A_CKPT="runs/unified_bev_stage_a_heightfix_ns_20260821/stage_a.pt"
LOG="runs/unified_bev_${TAG}_chain.log"
SUMMARY="runs/unified_bev_${TAG}_summary.txt"

COMMON=(
  --manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl
  --drive 2013_05_28_drive_0007_sync
  --max_samples 128
  --dense_sources 8
  --max_points 4096
  --hidden 256
  --ray_samples 48
  --min_target_spacing_m 5
  --num_workers 6
  --seed 0
  --device cuda
)
EVAL_COMMON=(
  --manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl
  --drive 2013_05_28_drive_0007_sync
  --max_samples 32
  --dense_sources 8
  --max_points 4096
  --min_target_spacing_m 5
  --device cuda
)

run_retry() {
  local attempt
  for attempt in 1 2; do
    if "$@"; then
      return 0
    fi
    echo "[chain] retry=${attempt} command=$* $(date)" >> "$LOG"
    sleep 20
  done
  return 1
}

echo "[chain] start tag=${TAG} stage_a=${STAGE_A_CKPT} $(date)" >> "$LOG"

for FUSION in residual coordinate_only; do
  STAGE_B_DIR="runs/unified_bev_stage_b_${TAG}_${FUSION}"
  STAGE_B_CKPT="${STAGE_B_DIR}/stage_b.pt"
  run_retry "${PY[@]}" scripts/train_unified_bev_stage_b.py \
    --stage_a "$STAGE_A_CKPT" --out "$STAGE_B_DIR" --fusion "$FUSION" \
    --steps 10000 --sparse_source_choices 1,2,4,8 "${COMMON[@]}" \
    > "runs/unified_bev_stage_b_${TAG}_${FUSION}.log" 2>&1
  echo "[chain] stage_b_${FUSION}_done $(date)" >> "$LOG"

  for NS in 1 2 4 8; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
      --sparse_sources "$NS" "${EVAL_COMMON[@]}" \
      --records_out "runs/eval_stage_b_${TAG}_${FUSION}_ns${NS}.jsonl" \
      > "runs/eval_stage_b_${TAG}_${FUSION}_ns${NS}.txt" 2>&1
  done
  echo "[chain] stage_b_${FUSION}_c1_done $(date)" >> "$LOG"

  # Identity regression gate: at Ns=8 alpha=0, so full must equal sparse
  # bitwise; their rounded metrics must therefore match exactly.
  "${PY[@]}" -c "
import ast
x = ast.literal_eval(open('runs/eval_stage_b_${TAG}_${FUSION}_ns8.txt').readline())
for key in ('full_latent_l1', 'full_psnr', 'full_absrel', 'full_rmse', 'full_delta1'):
    sparse_key = key.replace('full_', 'sparse_')
    assert abs(x[key] - x[sparse_key]) < 1e-12, (key, x[key], x[sparse_key])
print('[chain] identity_gate_pass fusion=${FUSION}')
"
  echo "[chain] identity_gate_${FUSION}_pass $(date)" >> "$LOG"

  # B7/B8 controls only apply to the proposed residual model.
  if [[ "$FUSION" == "residual" ]]; then
    for SHIFT in 1 2 5 10; do
      run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
        --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
        --sparse_sources 2 --sat_shift_road_m "$SHIFT" "${EVAL_COMMON[@]}" \
        --records_out "runs/eval_b7_${TAG}_shift_road_${SHIFT}m.jsonl" \
        > "runs/eval_b7_${TAG}_shift_road_${SHIFT}m.txt" 2>&1
    done
    for SHIFT in 1 2 5 10; do
      run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
        --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
        --sparse_sources 2 --sat_shift_cross_m "$SHIFT" "${EVAL_COMMON[@]}" \
        --records_out "runs/eval_b7_${TAG}_shift_cross_${SHIFT}m.jsonl" \
        > "runs/eval_b7_${TAG}_shift_cross_${SHIFT}m.txt" 2>&1
    done
    for ROT in 2 5 10; do
      run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
        --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
        --sparse_sources 2 --sat_rotate_deg "$ROT" "${EVAL_COMMON[@]}" \
        --records_out "runs/eval_b7_${TAG}_rotate_ccw_${ROT}deg.jsonl" \
        > "runs/eval_b7_${TAG}_rotate_ccw_${ROT}deg.txt" 2>&1
    done
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
      --sparse_sources 2 --sat_random_tile "${EVAL_COMMON[@]}" \
      --records_out "runs/eval_b8_${TAG}_random_tile.jsonl" \
      > "runs/eval_b8_${TAG}_random_tile.txt" 2>&1
    echo "[chain] b7_b8_controls_done $(date)" >> "$LOG"
  fi
done

{
  for FUSION in residual coordinate_only; do
    echo "=== Stage B ${FUSION} C1 ==="
    for NS in 1 2 4 8; do
      echo "--- Ns=${NS}"; cat "runs/eval_stage_b_${TAG}_${FUSION}_ns${NS}.txt"
    done
  done
  echo "=== B7 road-frame translation ==="
  for SHIFT in 1 2 5 10; do
    echo "--- road(along)=${SHIFT}m"; cat "runs/eval_b7_${TAG}_shift_road_${SHIFT}m.txt"
  done
  for SHIFT in 1 2 5 10; do
    echo "--- cross(left)=${SHIFT}m"; cat "runs/eval_b7_${TAG}_shift_cross_${SHIFT}m.txt"
  done
  echo "=== B7 rotation ==="
  for ROT in 2 5 10; do
    echo "--- ccw=${ROT}deg"; cat "runs/eval_b7_${TAG}_rotate_ccw_${ROT}deg.txt"
  done
  echo "=== B8 random tile ==="
  cat "runs/eval_b8_${TAG}_random_tile.txt"
} > "$SUMMARY"

echo "[chain] ALL_DONE summary=${SUMMARY} $(date)" >> "$LOG"
