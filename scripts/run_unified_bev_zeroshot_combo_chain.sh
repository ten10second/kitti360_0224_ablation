#!/usr/bin/env bash
# Evening chain: (1) zero-shot transfer eval on test drive 0003, (2) perceptual
# (SSIM/LPIPS) rerun of the val main table, (3) combo recipe training
# (nadir 0.1 + render/depth weights x10) with C1 evals and paired compares.
set -uo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

PY=(conda run --no-capture-output -n maskgit python)
TAG=zs0003_20260822
STAGE_A_CKPT="runs/unified_bev_stage_a_heightfix_ns_20260821/stage_a.pt"
YY01="runs/unified_bev_stage_b_nadir_20260822_yy01/stage_b.pt"
NN="runs/unified_bev_stage_b_coordfixed_20260821_coordinate_only/stage_b.pt"
YN="runs/unified_bev_stage_b_fusionfix_20260821_residual/stage_b.pt"
LOG="runs/unified_bev_${TAG}_chain.log"

VAL_EVAL=(
  --manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl
  --drive 2013_05_28_drive_0007_sync
  --max_samples 32 --dense_sources 8 --max_points 4096
  --min_target_spacing_m 5 --device cuda --eval_ssim_lpips
)
TEST_EVAL=(
  --manifest dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl
  --drive 2013_05_28_drive_0003_sync
  --max_samples 32 --dense_sources 8 --max_points 4096
  --min_target_spacing_m 5 --device cuda --eval_ssim_lpips
)
COMMON_TRAIN=(
  --manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl
  --drive 2013_05_28_drive_0007_sync
  --max_samples 128 --dense_sources 8 --max_points 4096
  --hidden 256 --ray_samples 48 --min_target_spacing_m 5
  --num_workers 6 --seed 0 --device cuda
)

run_retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then return 0; fi
    echo "[chain] retry=${attempt} command=$* $(date)" >> "$LOG"
    sleep 20
  done
  return 1
}

echo "[chain] start zeroshot+combo $(date)" >> "$LOG"

# --- 1. zero-shot on test drive 0003 (relative-effect survival) ---
run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
  --stage_a "$STAGE_A_CKPT" --stage_b "$YY01" --sparse_sources 1 "${TEST_EVAL[@]}" \
  --records_out "runs/eval_zs0003_yy01_ns1.jsonl" > runs/eval_zs0003_yy01_ns1.txt 2>&1
run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
  --stage_a "$STAGE_A_CKPT" --stage_b "$YY01" --sparse_sources 2 "${TEST_EVAL[@]}" \
  --records_out "runs/eval_zs0003_yy01_ns2.jsonl" > runs/eval_zs0003_yy01_ns2.txt 2>&1
run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
  --stage_a "$STAGE_A_CKPT" --stage_b "$NN" --sparse_sources 1 "${TEST_EVAL[@]}" \
  --records_out "runs/eval_zs0003_nn_ns1.jsonl" > runs/eval_zs0003_nn_ns1.txt 2>&1
run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
  --stage_a "$STAGE_A_CKPT" --stage_b "$NN" --sparse_sources 2 "${TEST_EVAL[@]}" \
  --records_out "runs/eval_zs0003_nn_ns2.jsonl" > runs/eval_zs0003_nn_ns2.txt 2>&1
run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
  --stage_a "$STAGE_A_CKPT" --stage_b "$YY01" --sparse_sources 2 --sat_shift_cross_m 5 "${TEST_EVAL[@]}" \
  --records_out "runs/eval_zs0003_yy01_ns2_cross5.jsonl" > runs/eval_zs0003_yy01_ns2_cross5.txt 2>&1
run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
  --stage_a "$STAGE_A_CKPT" --stage_b "$YY01" --sparse_sources 2 --sat_random_tile "${TEST_EVAL[@]}" \
  --records_out "runs/eval_zs0003_yy01_ns2_random.jsonl" > runs/eval_zs0003_yy01_ns2_random.txt 2>&1
echo "[chain] zeroshot_0003_done $(date)" >> "$LOG"

"${PY[@]}" -c "
import ast
for tag in ('yy01_ns1','yy01_ns2','nn_ns1','nn_ns2'):
    x = ast.literal_eval(open(f'runs/eval_zs0003_{tag}.txt').readline())
    print(tag, {k: round(x[k],4) for k in ('full_psnr','sparse_psnr','dense_psnr','full_absrel','sparse_absrel','full_ssim','full_lpips') if k in x})
" >> "$LOG" 2>&1

# --- 2. val main table with SSIM/LPIPS ---
for SPEC in "yy01:$YY01" "nn:$NN" "yn:$YN"; do
  NAME="${SPEC%%:*}"; CKPT="${SPEC#*:}"
  run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
    --stage_a "$STAGE_A_CKPT" --stage_b "$CKPT" --sparse_sources 2 "${VAL_EVAL[@]}" \
    --records_out "runs/eval_val_percep_${NAME}_ns2.jsonl" > "runs/eval_val_percep_${NAME}_ns2.txt" 2>&1
done
echo "[chain] val_perceptual_done $(date)" >> "$LOG"

# --- 3. combo recipe training + C1 ---
COMBO_DIR="runs/unified_bev_stage_b_combo_20260822_residual"
run_retry "${PY[@]}" scripts/train_unified_bev_stage_b.py \
  --stage_a "$STAGE_A_CKPT" --out "$COMBO_DIR" --fusion residual \
  --steps 10000 --sparse_source_choices 1,2,4,8 \
  --nadir_weight 0.1 --render_weight 1.0 --depth_weight 0.5 \
  "${COMMON_TRAIN[@]}" > runs/unified_bev_stage_b_combo_20260822_residual.log 2>&1
echo "[chain] combo_train_done $(date)" >> "$LOG"

for NS in 1 2 4 8; do
  run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
    --stage_a "$STAGE_A_CKPT" --stage_b "$COMBO_DIR/stage_b.pt" \
    --sparse_sources "$NS" "${VAL_EVAL[@]}" \
    --records_out "runs/eval_combo_residual_ns${NS}.jsonl" \
    > "runs/eval_combo_residual_ns${NS}.txt" 2>&1
done
"${PY[@]}" -c "
import ast
x = ast.literal_eval(open('runs/eval_combo_residual_ns8.txt').readline())
for key in ('full_latent_l1', 'full_psnr', 'full_absrel'):
    assert abs(x[key] - x[key.replace('full_', 'sparse_')]) < 1e-12, (key, x[key])
print('[chain] identity_gate_pass combo')
" >> "$LOG" 2>&1
echo "[chain] combo_eval_done $(date)" >> "$LOG"

for NS in 1 2; do
  echo "=== combo vs NN @Ns=$NS ===" >> runs/combo_compares.txt
  "${PY[@]}" scripts/compare_unified_bev_paired.py \
    --a "runs/eval_combo_residual_ns${NS}.jsonl" \
    --b "runs/eval_stage_b_coordfixed_20260821_coordinate_only_ns${NS}.jsonl" \
    --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,full_ssim,full_lpips >> runs/combo_compares.txt 2>&1
  echo "=== combo vs yy01 @Ns=$NS ===" >> runs/combo_compares.txt
  "${PY[@]}" scripts/compare_unified_bev_paired.py \
    --a "runs/eval_combo_residual_ns${NS}.jsonl" \
    --b "runs/eval_stage_b_nadir_20260822_yy01_ns${NS}.jsonl" \
    --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1 >> runs/combo_compares.txt 2>&1
done

echo "[chain] ALL_DONE $(date)" >> "$LOG"
