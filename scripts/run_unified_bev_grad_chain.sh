#!/usr/bin/env bash
# Graduation chain: full geographic isolation after the Step-0 collapse.
# Stage A + Stage B train on the 5-drive train split (42k frames); all
# evaluation happens on never-seen drives: test 0003 and (now also unseen)
# val 0007.  Recipe = combo (nadir 0.1 + render 1.0 + depth 0.5) after the
# zs0003 evening comparison.
set -uo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

PY=(conda run --no-capture-output -n maskgit python)
TAG=grad_20260822
TRAIN_MANIFEST="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl"
STAGE_A_DIR="runs/unified_bev_stage_a_${TAG}"
STAGE_A_CKPT="${STAGE_A_DIR}/stage_a.pt"
LOG="runs/unified_bev_${TAG}_chain.log"

TRAIN_COMMON=(
  --manifest "$TRAIN_MANIFEST"
  --max_samples 2048 --dense_sources 8 --max_points 4096
  --hidden 256 --ray_samples 48 --min_target_spacing_m 5
  --batch_size 2 --num_workers 0 --cache runs/cache_grad_2048_dir --device cuda
)
EVAL_0003=(
  --manifest dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl
  --drive 2013_05_28_drive_0003_sync
  --max_samples 32 --dense_sources 8 --max_points 4096
  --min_target_spacing_m 5 --device cuda --eval_ssim_lpips
)
EVAL_0007=(
  --manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl
  --drive 2013_05_28_drive_0007_sync
  --max_samples 32 --dense_sources 8 --max_points 4096
  --min_target_spacing_m 5 --device cuda --eval_ssim_lpips
)

run_retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then return 0; fi
    echo "[chain] retry=${attempt} command=$* $(date)" >> "$LOG"
    sleep 30
  done
  return 1
}

echo "[chain] start grad $(date)" >> "$LOG"

# --- Stage A on the 5-drive train split (resume if a partial checkpoint exists) ---
STAGE_A_RESUME=()
if [[ -f "$STAGE_A_CKPT" ]]; then
  STAGE_A_RESUME=(--resume "$STAGE_A_CKPT")
fi
run_retry "${PY[@]}" scripts/train_unified_bev_stage_a.py \
  --out "$STAGE_A_DIR" --steps 20000 --seed 0 "${STAGE_A_RESUME[@]}" "${TRAIN_COMMON[@]}" \
  > "runs/unified_bev_stage_a_${TAG}.log" 2>&1
echo "[chain] stage_a_done $(date)" >> "$LOG"

# Sanity: does the multi-drive space render the unseen test drive at all?
run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
  --stage_a "$STAGE_A_CKPT" --sparse_sources 2 "${EVAL_0003[@]}" \
  --records_out "runs/eval_grad_stagea_only_ns2.jsonl" \
  > "runs/eval_grad_stagea_only_ns2.txt" 2>&1
echo "[chain] stage_a_transfer_check_done $(date)" >> "$LOG"

# --- Stage B: combo recipe x {residual, coordinate_only} x 3 seeds ---
for SEED in 0 1; do
  for FUSION in residual coordinate_only; do
    DIR="runs/unified_bev_stage_b_${TAG}_${FUSION}_s${SEED}"
    run_retry "${PY[@]}" scripts/train_unified_bev_stage_b.py \
      --stage_a "$STAGE_A_CKPT" --out "$DIR" --fusion "$FUSION" \
      --steps 10000 --sparse_source_choices 1,2,4,8 --seed "$SEED" \
      --nadir_weight 0.1 --render_weight 1.0 --depth_weight 0.5 \
      "${TRAIN_COMMON[@]}" > "runs/unified_bev_stage_b_${TAG}_${FUSION}_s${SEED}.log" 2>&1
    for NS in 1 2 4 8; do
      run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
        --stage_a "$STAGE_A_CKPT" --stage_b "$DIR/stage_b.pt" \
        --sparse_sources "$NS" "${EVAL_0003[@]}" \
        --records_out "runs/eval_grad_${FUSION}_s${SEED}_0003_ns${NS}.jsonl" \
        > "runs/eval_grad_${FUSION}_s${SEED}_0003_ns${NS}.txt" 2>&1
    done
    "${PY[@]}" -c "
import ast
lines = [l for l in open('runs/eval_grad_${FUSION}_s${SEED}_0003_ns8.txt') if l.startswith('{')]
x = ast.literal_eval(lines[0])
for key in ('full_latent_l1', 'full_psnr'):
    assert abs(x[key] - x[key.replace('full_', 'sparse_')]) < 1e-12, (key, x[key])
print('[chain] identity_gate_pass ${FUSION}_s${SEED}')
" >> "$LOG" 2>&1
    echo "[chain] stage_b_${FUSION}_s${SEED}_done $(date)" >> "$LOG"
  done
done

# --- B7/B8 + val-0007 second venue for seed-0 residual ---
for NS in 1 2 4 8; do
  run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
    --stage_a "$STAGE_A_CKPT" \
    --stage_b "runs/unified_bev_stage_b_${TAG}_residual_s0/stage_b.pt" \
    --sparse_sources "$NS" "${EVAL_0007[@]}" \
    --records_out "runs/eval_grad_residual_s0_0007_ns${NS}.jsonl" \
    > "runs/eval_grad_residual_s0_0007_ns${NS}.txt" 2>&1
  run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
    --stage_a "$STAGE_A_CKPT" \
    --stage_b "runs/unified_bev_stage_b_${TAG}_coordinate_only_s0/stage_b.pt" \
    --sparse_sources "$NS" "${EVAL_0007[@]}" \
    --records_out "runs/eval_grad_coord_s0_0007_ns${NS}.jsonl" \
    > "runs/eval_grad_coord_s0_0007_ns${NS}.txt" 2>&1
done
for CTRL in "--sat_shift_cross_m 2" "--sat_shift_cross_m 5" "--sat_random_tile"; do
  run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
    --stage_a "$STAGE_A_CKPT" \
    --stage_b "runs/unified_bev_stage_b_${TAG}_residual_s0/stage_b.pt" \
    --sparse_sources 2 $CTRL "${EVAL_0003[@]}" \
    --records_out "runs/eval_grad_b7_${CTRL// /_}_0003.jsonl" \
    > "runs/eval_grad_b7_${CTRL// /_}_0003.txt" 2>&1
done
echo "[chain] b7_val_done $(date)" >> "$LOG"

# --- paired compares on 0003 across seeds ---
for SEED in 0 1; do
  for NS in 1 2; do
    echo "=== grad residual vs coord s${SEED} @0003 Ns=${NS} ===" >> runs/grad_compares.txt
    "${PY[@]}" scripts/compare_unified_bev_paired.py \
      --a "runs/eval_grad_residual_s${SEED}_0003_ns${NS}.jsonl" \
      --b "runs/eval_grad_coordinate_only_s${SEED}_0003_ns${NS}.jsonl" \
      --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,full_ssim,full_lpips \
      >> runs/grad_compares.txt 2>&1
  done
done

echo "[chain] ALL_DONE $(date)" >> "$LOG"
