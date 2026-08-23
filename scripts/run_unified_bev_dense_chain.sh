#!/usr/bin/env bash
# E2: Metric3D dense-lift graduation chain.
# Waits for the train M3D cache, builds the two eval caches, retrains Stage A
# with GroundDenseBEVEncoder, retrains Stage B (satellite heightmap + XY
# control), then runs the full battery with the C2 multi-chain ratio as the
# headline gate (pre-registered: Ns=1 ratio < 0.70 -> true convergence).
set -uo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

PY=(conda run --no-capture-output -n maskgit python)
TAG=dense_20260823
LOG="runs/unified_bev_${TAG}_chain.log"
TRAIN_CACHE=runs/cache_grad_2048_dir
M3D_TRAIN=runs/cache_m3d_street
M3D_0003=runs/cache_m3d_0003
M3D_0007=runs/cache_m3d_0007
STAGE_A_DIR="runs/unified_bev_stage_a_${TAG}"
STAGE_A_CKPT="${STAGE_A_DIR}/stage_a.pt"

TRAIN_COMMON=(
  --manifest dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl
  --max_samples 2048 --dense_sources 8 --max_points 4096
  --hidden 256 --ray_samples 48 --min_target_spacing_m 5
  --batch_size 2 --num_workers 0 --cache "$TRAIN_CACHE" --m3d_cache "$M3D_TRAIN"
  --device cuda --seed 0
)
EVAL_0003=(
  --manifest dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl
  --drive 2013_05_28_drive_0003_sync
  --max_samples 32 --dense_sources 8 --max_points 4096
  --min_target_spacing_m 5 --device cuda --eval_ssim_lpips --m3d_cache "$M3D_0003"
)
EVAL_0007=(
  --manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl
  --drive 2013_05_28_drive_0007_sync
  --max_samples 32 --dense_sources 8 --max_points 4096
  --min_target_spacing_m 5 --device cuda --eval_ssim_lpips --m3d_cache "$M3D_0007"
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

echo "[chain] waiting for train m3d cache $(date)" >> "$LOG"
while [ "$(ls "$M3D_TRAIN" 2>/dev/null | wc -l)" -lt 2048 ]; do sleep 120; done
echo "[chain] train cache complete $(date)" >> "$LOG"
sleep 30

# eval caches (~10 min each)
run_retry "${PY[@]}" scripts/build_metric3d_street_cache.py --out "$M3D_0003" \
  --eval_split --manifest dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl \
  --drive 2013_05_28_drive_0003_sync --max_samples 32 \
  > runs/build_m3d_0003.log 2>&1
run_retry "${PY[@]}" scripts/build_metric3d_street_cache.py --out "$M3D_0007" \
  --eval_split --manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl \
  --drive 2013_05_28_drive_0007_sync --max_samples 32 \
  > runs/build_m3d_0007.log 2>&1
echo "[chain] eval caches done $(date)" >> "$LOG"

# Stage A dense
run_retry "${PY[@]}" scripts/train_unified_bev_stage_a.py \
  --out "$STAGE_A_DIR" --steps 20000 "${TRAIN_COMMON[@]}" \
  > "runs/unified_bev_stage_a_${TAG}.log" 2>&1
echo "[chain] stage_a_dense_done $(date)" >> "$LOG"

# quick sanity: dense space on unseen 0003
run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
  --stage_a "$STAGE_A_CKPT" --sparse_sources 2 "${EVAL_0003[@]}" \
  --records_out "runs/eval_${TAG}_stagea_only.jsonl" \
  > "runs/eval_${TAG}_stagea_only.txt" 2>&1
echo "[chain] stage_a_transfer_check_done $(date)" >> "$LOG"

# Stage B x2
for FUSION in residual coordinate_only; do
  DIR="runs/unified_bev_stage_b_${TAG}_${FUSION}"
  run_retry "${PY[@]}" scripts/train_unified_bev_stage_b.py \
    --stage_a "$STAGE_A_CKPT" --out "$DIR" --fusion "$FUSION" --sat_encoder heightmap \
    --steps 10000 --sparse_source_choices 1,2,4,8 \
    --height_weight 1.0 --nadir_weight 0.1 --render_weight 1.0 --depth_weight 0.5 \
    "${TRAIN_COMMON[@]}" > "runs/unified_bev_stage_b_${TAG}_${FUSION}.log" 2>&1
  for NS in 1 2 4 8; do
    run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
      --stage_a "$STAGE_A_CKPT" --stage_b "$DIR/stage_b.pt" \
      --sparse_sources "$NS" "${EVAL_0003[@]}" \
      --records_out "runs/eval_${TAG}_${FUSION}_0003_ns${NS}.jsonl" \
      > "runs/eval_${TAG}_${FUSION}_0003_ns${NS}.txt" 2>&1
  done
  "${PY[@]}" -c "
import ast
lines = [l for l in open('runs/eval_${TAG}_${FUSION}_0003_ns8.txt') if l.startswith('{')]
x = ast.literal_eval(lines[0])
for key in ('full_latent_l1', 'full_psnr'):
    assert abs(x[key] - x[key.replace('full_', 'sparse_')]) < 1e-12, (key, x[key])
print('[chain] identity_gate_pass ${FUSION}')
" >> "$LOG" 2>&1
  echo "[chain] stage_b_${FUSION}_done $(date)" >> "$LOG"
done

# B7 controls (residual)
SB="runs/unified_bev_stage_b_${TAG}_residual/stage_b.pt"
for S in 2 5; do
  run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
    --stage_a "$STAGE_A_CKPT" --stage_b "$SB" \
    --sparse_sources 2 --sat_shift_cross_m "$S" "${EVAL_0003[@]}" \
    --records_out "runs/eval_${TAG}_b7_cross${S}.jsonl" \
    > "runs/eval_${TAG}_b7_cross${S}.txt" 2>&1
done

# HEADLINE: C2 multi-chain convergence on dense lift
run_retry "${PY[@]}" scripts/consistency_unified_bev_multichain.py \
  --stage_a "$STAGE_A_CKPT" \
  --stage_b_sat "runs/unified_bev_stage_b_${TAG}_residual/stage_b.pt" \
  --stage_b_xy "runs/unified_bev_stage_b_${TAG}_coordinate_only/stage_b.pt" \
  --ns_list 1,2,4 --m3d_cache "$M3D_0003" \
  --records_out "runs/${TAG}_c2_dense.jsonl" \
  > "runs/${TAG}_c2_dense.txt" 2>&1
echo "[chain] C2_HEADLINE_DONE $(date)" >> "$LOG"

# paired compares vs E0 sparse baseline
for NS in 1 2; do
  echo "=== dense vs sparse(grad) residual @0003 Ns=${NS} ===" >> runs/dense_compares.txt
  "${PY[@]}" scripts/compare_unified_bev_paired.py \
    --a "runs/eval_${TAG}_residual_0003_ns${NS}.jsonl" \
    --b "runs/eval_grad_residual_s0_0003_ns${NS}.jsonl" \
    --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,full_ssim,full_lpips >> runs/dense_compares.txt 2>&1
  echo "=== dense vs dense-xy @0003 Ns=${NS} ===" >> runs/dense_compares.txt
  "${PY[@]}" scripts/compare_unified_bev_paired.py \
    --a "runs/eval_${TAG}_residual_0003_ns${NS}.jsonl" \
    --b "runs/eval_${TAG}_coordinate_only_0003_ns${NS}.jsonl" \
    --keys full_psnr,full_absrel,full_delta1,full_rmse,full_latent_l1,full_ssim,full_lpips >> runs/dense_compares.txt 2>&1
done

echo "[chain] ALL_DONE $(date)" >> "$LOG"
