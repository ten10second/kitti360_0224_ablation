#!/usr/bin/env bash
# v2 satellite branch: CVS-pattern height-map prior (cross-attention to the
# street latent + dense-LiDAR h_mean as per-tile DEM supervision + exact
# orthographic BEV anchor).  Identical protocol to the vit/cnn graduation
# chains; the ONLY change is the satellite prior pathway.
set -uo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

PY=(conda run --no-capture-output -n maskgit python)
TAG=v2sat_20260823
STAGE_A_CKPT="runs/unified_bev_stage_a_grad_20260822/stage_a.pt"
LOG="runs/unified_bev_${TAG}_chain.log"

TRAIN_COMMON=(
  --manifest dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl
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

echo "[chain] start v2sat $(date)" >> "$LOG"

for SEED in 0 1; do
  DIR="runs/unified_bev_stage_b_${TAG}_residual_s${SEED}"
  run_retry "${PY[@]}" scripts/train_unified_bev_stage_b.py \
    --stage_a "$STAGE_A_CKPT" --out "$DIR" --fusion residual --sat_encoder heightmap \
    --steps 10000 --sparse_source_choices 1,2,4,8 --seed "$SEED" \
    --height_weight 1.0 --nadir_weight 0.1 --render_weight 1.0 --depth_weight 0.5 \
    "${TRAIN_COMMON[@]}" > "runs/unified_bev_stage_b_${TAG}_residual_s${SEED}.log" 2>&1
  for NS in 1 2 4 8; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$DIR/stage_b.pt" \
      --sparse_sources "$NS" "${EVAL_0003[@]}" \
      --records_out "runs/eval_v2sat_residual_s${SEED}_0003_ns${NS}.jsonl" \
      > "runs/eval_v2sat_residual_s${SEED}_0003_ns${NS}.txt" 2>&1
  done
  "${PY[@]}" -c "
import ast
lines = [l for l in open('runs/eval_v2sat_residual_s${SEED}_0003_ns8.txt') if l.startswith('{')]
x = ast.literal_eval(lines[0])
for key in ('full_latent_l1', 'full_psnr'):
    assert abs(x[key] - x[key.replace('full_', 'sparse_')]) < 1e-12, (key, x[key])
print('[chain] identity_gate_pass v2sat_s${SEED}')
" >> "$LOG" 2>&1
  for SHIFT in 2 5; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$DIR/stage_b.pt" \
      --sparse_sources 2 --sat_shift_cross_m "$SHIFT" "${EVAL_0003[@]}" \
      --records_out "runs/eval_v2sat_b7_cross${SHIFT}_s${SEED}_0003.jsonl" \
      > "runs/eval_v2sat_b7_cross${SHIFT}_s${SEED}_0003.txt" 2>&1
  done
  run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
    --stage_a "$STAGE_A_CKPT" --stage_b "$DIR/stage_b.pt" \
    --sparse_sources 2 --sat_random_tile "${EVAL_0003[@]}" \
    --records_out "runs/eval_v2sat_b8_random_s${SEED}_0003.jsonl" \
    > "runs/eval_v2sat_b8_random_s${SEED}_0003.txt" 2>&1
  for NS in 1 2; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$DIR/stage_b.pt" \
      --sparse_sources "$NS" "${EVAL_0007[@]}" \
      --records_out "runs/eval_v2sat_residual_s${SEED}_0007_ns${NS}.jsonl" \
      > "runs/eval_v2sat_residual_s${SEED}_0007_ns${NS}.txt" 2>&1
  done
  echo "[chain] v2sat_s${SEED}_done $(date)" >> "$LOG"
done

for SEED in 0 1; do
  for NS in 1 2; do
    for B in "vit:eval_vit_residual_s${SEED}_0003_ns${NS}" "cnn:eval_grad_residual_s${SEED}_0003_ns${NS}" "coord:eval_grad_coordinate_only_s${SEED}_0003_ns${NS}"; do
      NAME="${B%%:*}"; F="runs/${B#*:}.jsonl"
      echo "=== v2sat vs ${NAME} s${SEED} @0003 Ns=${NS} ===" >> runs/v2sat_compares.txt
      "${PY[@]}" scripts/compare_unified_bev_paired.py \
        --a "runs/eval_v2sat_residual_s${SEED}_0003_ns${NS}.jsonl" \
        --b "$F" \
        --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,full_ssim,full_lpips \
        >> runs/v2sat_compares.txt 2>&1
    done
  done
done

echo "[chain] ALL_DONE $(date)" >> "$LOG"
