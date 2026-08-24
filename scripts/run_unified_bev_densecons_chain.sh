#!/usr/bin/env bash
# Consistency-weighted splat verdict chain: retrain the dense lift with the
# MVS visibility check and re-measure the C2 headline (pre-registered: pull
# the Ns=1 ratio below the dense-no-consistency 1.115, ideally toward the
# sparse baseline 1.02 or below), while confirming no regression in the
# ceiling (stage-A-only dense PSNR) or C1/identity.
set -uo pipefail
cd /media/shizhm/Lenovo/kitti360_0224_ablation

# sda2 self-heal (drops on every reboot)
mountpoint -q /media/shizhm/sda2 || udisksctl mount -b /dev/sda83 || true
sleep 2

PY=(conda run --no-capture-output -n maskgit python)
TAG=densecons_20260824
LOG="runs/unified_bev_${TAG}_chain.log"
STAGE_A_DIR="runs/unified_bev_stage_a_${TAG}"
STAGE_A_CKPT="${STAGE_A_DIR}/stage_a.pt"

TRAIN_COMMON=(
  --manifest dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl
  --max_samples 2048 --dense_sources 8 --max_points 4096
  --hidden 256 --ray_samples 48 --min_target_spacing_m 5
  --batch_size 2 --num_workers 0 --cache runs/cache_grad_2048_dir --m3d_cache runs/cache_m3d_street
  --device cuda --seed 0
)
EVAL_0003=(
  --manifest dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl
  --drive 2013_05_28_drive_0003_sync
  --max_samples 32 --dense_sources 8 --max_points 4096
  --min_target_spacing_m 5 --device cuda --eval_ssim_lpips --m3d_cache runs/cache_m3d_0003
)

run_retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then return 0; fi
    echo "[chain] retry=${attempt} $(date)" >> "$LOG"
    sleep 30
  done
  return 1
}
# artifact audit: fail loudly if a product contains a traceback
audit() {
  if grep -q Traceback "$1" 2>/dev/null; then
    echo "[chain] AUDIT_FAIL $1" >> "$LOG"; return 1
  fi
}

echo "[chain] start $(date)" >> "$LOG"

run_retry "${PY[@]}" scripts/train_unified_bev_stage_a.py \
  --out "$STAGE_A_DIR" --steps 20000 "${TRAIN_COMMON[@]}" \
  > "runs/unified_bev_stage_a_${TAG}.log" 2>&1
echo "[chain] stage_a_done $(date)" >> "$LOG"

run_retry "${PY[@]}" scripts/eval_unified_bev_probe.py \
  --stage_a "$STAGE_A_CKPT" --sparse_sources 2 "${EVAL_0003[@]}" \
  --records_out "runs/eval_${TAG}_stagea_only.jsonl" \
  > "runs/eval_${TAG}_stagea_only.txt" 2>&1
audit "runs/eval_${TAG}_stagea_only.txt" && echo "[chain] transfer_check_ok $(date)" >> "$LOG"

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
    audit "runs/eval_${TAG}_${FUSION}_0003_ns${NS}.txt" || exit 1
  done
  echo "[chain] stage_b_${FUSION}_done $(date)" >> "$LOG"
done

run_retry "${PY[@]}" scripts/consistency_unified_bev_multichain.py \
  --stage_a "$STAGE_A_CKPT" \
  --stage_b_sat "runs/unified_bev_stage_b_${TAG}_residual/stage_b.pt" \
  --stage_b_xy "runs/unified_bev_stage_b_${TAG}_coordinate_only/stage_b.pt" \
  --ns_list 1,2,4 --m3d_cache runs/cache_m3d_0003 \
  --records_out "runs/${TAG}_c2.jsonl" \
  > "runs/${TAG}_c2.txt" 2>&1
audit "runs/${TAG}_c2.txt" && echo "[chain] C2_HEADLINE_DONE $(date)" >> "$LOG"

echo "[chain] ALL_DONE $(date)" >> "$LOG"
