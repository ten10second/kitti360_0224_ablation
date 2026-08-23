#!/usr/bin/env bash
# Height-fix baseline + random-Ns Stage B + C1/B7/depth evaluation chain.
set -euo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

PY=(conda run --no-capture-output -n maskgit python)
TAG=heightfix_ns_20260821
STAGE_A_DIR="runs/unified_bev_stage_a_${TAG}"
STAGE_A_CKPT="${STAGE_A_DIR}/stage_a.pt"
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

echo "[chain] start tag=${TAG} $(date)" >> "$LOG"

run_retry "${PY[@]}" scripts/train_unified_bev_stage_a.py \
  --out "$STAGE_A_DIR" --steps 20000 "${COMMON[@]}" \
  > "runs/unified_bev_stage_a_${TAG}.log" 2>&1
echo "[chain] stage_a_done $(date)" >> "$LOG"

for NS in 1 2 4 8; do
  run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
    --stage_a "$STAGE_A_CKPT" --sparse_sources "$NS" "${EVAL_COMMON[@]}" \
    --records_out "runs/eval_stage_a_${TAG}_ns${NS}.jsonl" \
    > "runs/eval_stage_a_${TAG}_ns${NS}.txt" 2>&1
done

# Stage A is the representation gate. Do not spend Stage-B compute when its
# dense reference is not better than the sparse Ns=2 input.
"${PY[@]}" -c \
  "import ast; p='runs/eval_stage_a_${TAG}_ns2.txt'; x=ast.literal_eval(open(p).readline()); assert x['dense_psnr'] > x['sparse_psnr'], x"
echo "[chain] stage_a_gate_pass $(date)" >> "$LOG"

for FUSION in residual coordinate_only satellite_only; do
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

  # B7/B8 controls only apply to the proposed residual model. All metrics,
  # including camera-z depth, are emitted by the same paired evaluator.
  if [[ "$FUSION" == "residual" ]]; then
    for SHIFT in 1 2 5 10; do
      run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
        --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
        --sparse_sources 2 --sat_shift_x_m "$SHIFT" "${EVAL_COMMON[@]}" \
        --records_out "runs/eval_b7_${TAG}_shift_east_${SHIFT}m.jsonl" \
        > "runs/eval_b7_${TAG}_shift_east_${SHIFT}m.txt" 2>&1
    done
    for SHIFT in 1 2 5 10; do
      run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
        --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
        --sparse_sources 2 --sat_shift_y_m "$SHIFT" "${EVAL_COMMON[@]}" \
        --records_out "runs/eval_b7_${TAG}_shift_north_${SHIFT}m.jsonl" \
        > "runs/eval_b7_${TAG}_shift_north_${SHIFT}m.txt" 2>&1
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
  echo "=== Stage A C1 ==="
  for NS in 1 2 4 8; do
    echo "--- Ns=${NS}"; cat "runs/eval_stage_a_${TAG}_ns${NS}.txt"
  done
  for FUSION in residual coordinate_only satellite_only; do
    echo "=== Stage B ${FUSION} C1 ==="
    for NS in 1 2 4 8; do
      echo "--- Ns=${NS}"; cat "runs/eval_stage_b_${TAG}_${FUSION}_ns${NS}.txt"
    done
  done
  echo "=== B7 translation ==="
  for SHIFT in 1 2 5 10; do
    echo "--- east=${SHIFT}m"; cat "runs/eval_b7_${TAG}_shift_east_${SHIFT}m.txt"
  done
  for SHIFT in 1 2 5 10; do
    echo "--- north=${SHIFT}m"; cat "runs/eval_b7_${TAG}_shift_north_${SHIFT}m.txt"
  done
  echo "=== B7 rotation ==="
  for ROT in 2 5 10; do
    echo "--- ccw=${ROT}deg"; cat "runs/eval_b7_${TAG}_rotate_ccw_${ROT}deg.txt"
  done
  echo "=== B8 random tile ==="
  cat "runs/eval_b8_${TAG}_random_tile.txt"
} > "$SUMMARY"

echo "[chain] ALL_DONE summary=${SUMMARY} $(date)" >> "$LOG"
