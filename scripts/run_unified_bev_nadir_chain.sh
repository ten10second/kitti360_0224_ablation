#!/usr/bin/env bash
# Nadir supervision chain (family-5 variant B): 2x2 input x supervision plus
# a loss-weight ablation, on top of the heightfix Stage A and fusionfix
# identity.  Diagnostics that motivated this: decoder readout gain ratio
# declines 0.92->0.84->0.76 over Ns=1/2/4; nadir round-trip correlation is
# weak but positive on 32/32 tiles.
set -euo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

PY=(conda run --no-capture-output -n maskgit python)
TAG=nadir_20260822
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
  --eval_nadir
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

# name:fusion:nadir_weight:render_weight
RUNS=(
  "yy01:residual:0.1:0.1"
  "yy03:residual:0.3:0.1"
  "ny01:coordinate_only:0.1:0.1"
  "rw10:residual:0.0:1.0"
)

echo "[chain] start tag=${TAG} $(date)" >> "$LOG"

for SPEC in "${RUNS[@]}"; do
  IFS=':' read -r NAME FUSION NADIR_W RENDER_W <<< "$SPEC"
  STAGE_B_DIR="runs/unified_bev_stage_b_${TAG}_${NAME}"
  STAGE_B_CKPT="${STAGE_B_DIR}/stage_b.pt"
  EXTRA=(--nadir_weight "$NADIR_W")
  if [[ "$NAME" == "rw10" ]]; then
    EXTRA+=(--render_weight 1.0 --depth_weight 0.5)
  fi
  run_retry "${PY[@]}" scripts/train_unified_bev_stage_b.py \
    --stage_a "$STAGE_A_CKPT" --out "$STAGE_B_DIR" --fusion "$FUSION" \
    --steps 10000 --sparse_source_choices 1,2,4,8 "${EXTRA[@]}" "${COMMON[@]}" \
    > "runs/unified_bev_stage_b_${TAG}_${NAME}.log" 2>&1
  echo "[chain] stage_b_${NAME}_done $(date)" >> "$LOG"

  for NS in 1 2 4 8; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
      --sparse_sources "$NS" "${EVAL_COMMON[@]}" \
      --records_out "runs/eval_stage_b_${TAG}_${NAME}_ns${NS}.jsonl" \
      > "runs/eval_stage_b_${TAG}_${NAME}_ns${NS}.txt" 2>&1
  done
  "${PY[@]}" -c "
import ast
x = ast.literal_eval(open('runs/eval_stage_b_${TAG}_${NAME}_ns8.txt').readline())
for key in ('full_latent_l1', 'full_psnr', 'full_absrel'):
    sparse_key = key.replace('full_', 'sparse_')
    assert abs(x[key] - x[sparse_key]) < 1e-12, (key, x[key], x[sparse_key])
print('[chain] identity_gate_pass ${NAME}')
"
  echo "[chain] identity_gate_${NAME}_pass $(date)" >> "$LOG"
done

# B7/B8 road-frame controls for both nadir doses of the residual variant.
for NAME in yy01 yy03; do
  STAGE_B_CKPT="runs/unified_bev_stage_b_${TAG}_${NAME}/stage_b.pt"
  for SHIFT in 1 2 5 10; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
      --sparse_sources 2 --sat_shift_cross_m "$SHIFT" "${EVAL_COMMON[@]}" \
      --records_out "runs/eval_b7_${TAG}_${NAME}_shift_cross_${SHIFT}m.jsonl" \
      > "runs/eval_b7_${TAG}_${NAME}_shift_cross_${SHIFT}m.txt" 2>&1
  done
  for ROT in 5 10; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
      --sparse_sources 2 --sat_rotate_deg "$ROT" "${EVAL_COMMON[@]}" \
      --records_out "runs/eval_b7_${TAG}_${NAME}_rotate_ccw_${ROT}deg.jsonl" \
      > "runs/eval_b7_${TAG}_${NAME}_rotate_ccw_${ROT}deg.txt" 2>&1
  done
  run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
    --stage_a "$STAGE_A_CKPT" --stage_b "$STAGE_B_CKPT" \
    --sparse_sources 2 --sat_random_tile "${EVAL_COMMON[@]}" \
    --records_out "runs/eval_b8_${TAG}_${NAME}_random_tile.jsonl" \
    > "runs/eval_b8_${TAG}_${NAME}_random_tile.txt" 2>&1
  echo "[chain] b7_b8_${NAME}_done $(date)" >> "$LOG"
done

# Paired bootstrap comparisons (positive = first argument better).
compare() {
  echo "=== $1 vs $2 (Ns=$3) ==="
  "${PY[@]}" scripts/compare_unified_bev_paired.py \
    --a "runs/eval_stage_b_${TAG}_$1_ns$3.jsonl" \
    --b "runs/eval_stage_b_${TAG}_$2_ns$3.jsonl" \
    --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,nadir_l1_full
}
compare yy01 ny01 2
compare yy01 yy03 2
compare yy01 rw10 2
compare yy03 rw10 2
compare yy01 ny01 1
compare yy01 rw10 1

NN_NS2="runs/eval_stage_b_coordfixed_20260821_coordinate_only_ns2.jsonl"
YN_NS2="runs/eval_stage_b_fusionfix_20260821_residual_ns2.jsonl"
{
  echo "=== yy01 vs NN_xy(coordfixed, no sat) Ns=2 ==="
  "${PY[@]}" scripts/compare_unified_bev_paired.py \
    --a "runs/eval_stage_b_${TAG}_yy01_ns2.jsonl" --b "$NN_NS2" \
    --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,nadir_l1_full
  echo "=== yy01 vs YN_residual(fusionfix, input-only) Ns=2 ==="
  "${PY[@]}" scripts/compare_unified_bev_paired.py \
    --a "runs/eval_stage_b_${TAG}_yy01_ns2.jsonl" --b "$YN_NS2" \
    --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,nadir_l1_full
  echo "=== yy03 vs NN_xy Ns=2 ==="
  "${PY[@]}" scripts/compare_unified_bev_paired.py \
    --a "runs/eval_stage_b_${TAG}_yy03_ns2.jsonl" --b "$NN_NS2" \
    --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,nadir_l1_full
  echo "=== rw10 vs YN_residual (loss-weight ablation) Ns=2 ==="
  "${PY[@]}" scripts/compare_unified_bev_paired.py \
    --a "runs/eval_stage_b_${TAG}_rw10_ns2.jsonl" --b "$YN_NS2" \
    --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,nadir_l1_full
} >> "$SUMMARY" 2>&1

{
  for NAME in yy01 yy03 ny01 rw10; do
    echo "=== Stage B ${NAME} C1 ==="
    for NS in 1 2 4 8; do
      echo "--- Ns=${NS}"; cat "runs/eval_stage_b_${TAG}_${NAME}_ns${NS}.txt"
    done
  done
  for NAME in yy01 yy03; do
    echo "=== B7 cross-road ${NAME} ==="
    for SHIFT in 1 2 5 10; do
      echo "--- cross=${SHIFT}m"; cat "runs/eval_b7_${TAG}_${NAME}_shift_cross_${SHIFT}m.txt"
    done
    echo "=== B7/B8 rot+random ${NAME} ==="
    for ROT in 5 10; do
      echo "--- ccw=${ROT}deg"; cat "runs/eval_b7_${TAG}_${NAME}_rotate_ccw_${ROT}deg.txt"
    done
    echo "--- random tile"; cat "runs/eval_b8_${TAG}_${NAME}_random_tile.txt"
  done
} > "$SUMMARY.body"

cat "$SUMMARY.body" >> "$SUMMARY"
echo "[chain] ALL_DONE summary=${SUMMARY} $(date)" >> "$LOG"
