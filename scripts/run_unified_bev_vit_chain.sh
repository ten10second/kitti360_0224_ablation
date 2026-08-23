#!/usr/bin/env bash
# ViT satellite encoder chain: the graduation verdict named encoder capacity
# as the binding constraint (B7 gone, render conversion negative on unseen
# drives).  Same grad Stage A + cache + combo recipe as grad_20260822; the
# ONLY change is the satellite encoder family (cnn -> vit, ~3.4M params,
# global receptive field, fixed metric-XY sin-cos positions).
set -uo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

PY=(conda run --no-capture-output -n maskgit python)
TAG=vit_20260822
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

echo "[chain] start vit $(date)" >> "$LOG"

for SEED in 0 1; do
  DIR="runs/unified_bev_stage_b_${TAG}_residual_s${SEED}"
  run_retry "${PY[@]}" scripts/train_unified_bev_stage_b.py \
    --stage_a "$STAGE_A_CKPT" --out "$DIR" --fusion residual --sat_encoder vit \
    --steps 10000 --sparse_source_choices 1,2,4,8 --seed "$SEED" \
    --nadir_weight 0.1 --render_weight 1.0 --depth_weight 0.5 \
    "${TRAIN_COMMON[@]}" > "runs/unified_bev_stage_b_${TAG}_residual_s${SEED}.log" 2>&1
  for NS in 1 2 4 8; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$DIR/stage_b.pt" \
      --sparse_sources "$NS" "${EVAL_0003[@]}" \
      --records_out "runs/eval_vit_residual_s${SEED}_0003_ns${NS}.jsonl" \
      > "runs/eval_vit_residual_s${SEED}_0003_ns${NS}.txt" 2>&1
  done
  "${PY[@]}" -c "
import ast
lines = [l for l in open('runs/eval_vit_residual_s${SEED}_0003_ns8.txt') if l.startswith('{')]
x = ast.literal_eval(lines[0])
for key in ('full_latent_l1', 'full_psnr'):
    assert abs(x[key] - x[key.replace('full_', 'sparse_')]) < 1e-12, (key, x[key])
print('[chain] identity_gate_pass vit_s${SEED}')
" >> "$LOG" 2>&1
  # B7/B8 on 0003
  for SHIFT in 2 5; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$DIR/stage_b.pt" \
      --sparse_sources 2 --sat_shift_cross_m "$SHIFT" "${EVAL_0003[@]}" \
      --records_out "runs/eval_vit_b7_cross${SHIFT}_s${SEED}_0003.jsonl" \
      > "runs/eval_vit_b7_cross${SHIFT}_s${SEED}_0003.txt" 2>&1
  done
  run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
    --stage_a "$STAGE_A_CKPT" --stage_b "$DIR/stage_b.pt" \
    --sparse_sources 2 --sat_random_tile "${EVAL_0003[@]}" \
    --records_out "runs/eval_vit_b8_random_s${SEED}_0003.jsonl" \
    > "runs/eval_vit_b8_random_s${SEED}_0003.txt" 2>&1
  # 0007 second venue
  for NS in 1 2; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$DIR/stage_b.pt" \
      --sparse_sources "$NS" "${EVAL_0007[@]}" \
      --records_out "runs/eval_vit_residual_s${SEED}_0007_ns${NS}.jsonl" \
      > "runs/eval_vit_residual_s${SEED}_0007_ns${NS}.txt" 2>&1
  done
  echo "[chain] vit_s${SEED}_done $(date)" >> "$LOG"
done

# Paired compares: ViT vs CNN (same everything else) and ViT vs XY control
for SEED in 0 1; do
  for NS in 1 2; do
    echo "=== vit vs cnn(grad) s${SEED} @0003 Ns=${NS} ===" >> runs/vit_compares.txt
    "${PY[@]}" scripts/compare_unified_bev_paired.py \
      --a "runs/eval_vit_residual_s${SEED}_0003_ns${NS}.jsonl" \
      --b "runs/eval_grad_residual_s${SEED}_0003_ns${NS}.jsonl" \
      --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,full_ssim,full_lpips \
      >> runs/vit_compares.txt 2>&1
    echo "=== vit vs coord(grad) s${SEED} @0003 Ns=${NS} ===" >> runs/vit_compares.txt
    "${PY[@]}" scripts/compare_unified_bev_paired.py \
      --a "runs/eval_vit_residual_s${SEED}_0003_ns${NS}.jsonl" \
      --b "runs/eval_grad_coordinate_only_s${SEED}_0003_ns${NS}.jsonl" \
      --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,full_ssim,full_lpips \
      >> runs/vit_compares.txt 2>&1
  done
done

echo "[chain] ALL_DONE $(date)" >> "$LOG"
