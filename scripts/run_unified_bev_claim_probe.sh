#!/usr/bin/env bash
# Claim-aligned first probe:
# dense-ground Stage A defines one RGB/depth + relative-height interface;
# Stage B compares aligned satellite, fixed XY, random-tile, shifted satellite,
# and sparse-ground-only controls on a geographically held-out drive.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=(conda run --no-capture-output -n maskgit python)
TAG="${TAG:-claim_probe_20260824}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-256}"
EVAL_SAMPLES="${EVAL_SAMPLES:-32}"
WITH_PERCEPTUAL="${WITH_PERCEPTUAL:-1}"
STAGE_A_STEPS="${STAGE_A_STEPS:-20000}"
STAGE_B_STEPS="${STAGE_B_STEPS:-10000}"
VGGT_WEIGHTS="${VGGT_WEIGHTS:-/home/shizhm/Downloads/vggt.pt}"
LIDAR_ROOT="${LIDAR_ROOT:-/media/shizhm/sda2/KITTI360_lidar/data_3d_raw}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl}"
TEST_DRIVE="${TEST_DRIVE:-2013_05_28_drive_0003_sync}"

ROOT="runs/unified_bev_${TAG}"
LOG="${ROOT}/chain.log"
VGGT_TRAIN="${ROOT}/cache_vggt_train"
VGGT_TEST="${ROOT}/cache_vggt_test"
STAGE_A_DIR="${ROOT}/stage_a"
STAGE_A="${STAGE_A_DIR}/stage_a.pt"
STAGE_B_SAT_DIR="${ROOT}/stage_b_aligned_satellite"
STAGE_B_XY_DIR="${ROOT}/stage_b_fixed_xy"
STAGE_B_SAT="${STAGE_B_SAT_DIR}/stage_b.pt"
STAGE_B_XY="${STAGE_B_XY_DIR}/stage_b.pt"
mkdir -p "$ROOT"
: >"${ROOT}/paired_claim_comparisons.txt"
cat >>"${ROOT}/paired_claim_comparisons.txt" <<'EOF'
# VGGT scale audit: Ns=1 is a camera-rig-scaled diagnostic because one frame
# has no vehicle motion. Ns=2 uses one vehicle-motion baseline. Every JSONL
# record includes source/pair-count/MAD/RMSE/reliability fields.
EOF

[[ -f "$VGGT_WEIGHTS" ]] || { echo "missing VGGT weights: $VGGT_WEIGHTS" >&2; exit 2; }
[[ -d "$LIDAR_ROOT" ]] || { echo "missing LiDAR root: $LIDAR_ROOT" >&2; exit 2; }

run_logged() {
  local output="$1"
  shift
  local attempt
  for attempt in 1 2; do
    if "$@" >"$output" 2>&1; then
      return 0
    fi
    echo "[chain] failed attempt=${attempt} output=${output} command=$*" >>"$LOG"
  done
  return 1
}

echo "[chain] start tag=${TAG} $(date --iso-8601=seconds)" >>"$LOG"

# Exact subset inference is mandatory for joint-view VGGT. The C2 B chain
# uses source frames 4.., so those subsets are cached independently too.
VGGT_TRAIN_SOURCE=(
  --raw_dataset --manifest "$TRAIN_MANIFEST" --drive none
  --lidar_root "$LIDAR_ROOT" --max_samples "$TRAIN_SAMPLES"
)
TRAIN_DATA=(
  --manifest "$TRAIN_MANIFEST" --lidar_root "$LIDAR_ROOT"
  --max_samples "$TRAIN_SAMPLES"
)
if [[ -n "${TRAIN_SAMPLE_CACHE:-}" ]]; then
  VGGT_TRAIN_SOURCE=(--cache "$TRAIN_SAMPLE_CACHE" --max_samples "$TRAIN_SAMPLES")
  TRAIN_DATA=(--cache "$TRAIN_SAMPLE_CACHE" --max_samples "$TRAIN_SAMPLES")
fi

run_logged "${ROOT}/build_vggt_train.log" "${PY[@]}" scripts/build_vggt_street_cache.py \
  --weights "$VGGT_WEIGHTS" --out "$VGGT_TRAIN" \
  --subset_specs 0:1,0:2,0:4,0:8 --dense_sources 8 --device cuda \
  "${VGGT_TRAIN_SOURCE[@]}"

run_logged "${ROOT}/build_vggt_test.log" "${PY[@]}" scripts/build_vggt_street_cache.py \
  --weights "$VGGT_WEIGHTS" --out "$VGGT_TEST" --raw_dataset \
  --manifest "$TEST_MANIFEST" --drive "$TEST_DRIVE" --lidar_root "$LIDAR_ROOT" \
  --max_samples "$EVAL_SAMPLES" --dense_sources 8 \
  --subset_specs 0:1,0:2,0:4,0:8,4:1,4:2 --device cuda
echo "[chain] exact-subset VGGT caches complete $(date --iso-8601=seconds)" >>"$LOG"

run_logged "${ROOT}/stage_a.log" "${PY[@]}" scripts/train_unified_bev_stage_a.py \
  --out "$STAGE_A_DIR" --steps "$STAGE_A_STEPS" --dense_sources 8 \
  --max_points 4096 --hidden 256 --ray_samples 48 --batch_size 2 \
  --num_workers 0 --geometry_cache "$VGGT_TRAIN" --geometry_weight 0.1 \
  --device cuda --seed 0 "${TRAIN_DATA[@]}"
echo "[chain] Stage A + shared geometry readout complete $(date --iso-8601=seconds)" >>"$LOG"

STAGE_B_COMMON=(
  --stage_a "$STAGE_A" --steps "$STAGE_B_STEPS" --dense_sources 8
  --sparse_source_choices 1,2,4 --max_points 4096
  --batch_size 2 --num_workers 0 --geometry_cache "$VGGT_TRAIN"
  --sat_encoder heightmap --height_weight 0.0 --latent_weight 0.01
  --anchor_weight 1.0 --geometry_fill_weight 1.0
  --rgb_lowfreq_weight 0.1 --rgb_observed_weight 0.1
  --device cuda --seed 0
)
run_logged "${ROOT}/stage_b_aligned_satellite.log" "${PY[@]}" scripts/train_unified_bev_stage_b.py \
  --out "$STAGE_B_SAT_DIR" --fusion residual "${STAGE_B_COMMON[@]}" "${TRAIN_DATA[@]}"
run_logged "${ROOT}/stage_b_fixed_xy.log" "${PY[@]}" scripts/train_unified_bev_stage_b.py \
  --out "$STAGE_B_XY_DIR" --fusion coordinate_only "${STAGE_B_COMMON[@]}" "${TRAIN_DATA[@]}"
echo "[chain] Stage B controls complete $(date --iso-8601=seconds)" >>"$LOG"

EVAL_COMMON=(
  --stage_a "$STAGE_A" --manifest "$TEST_MANIFEST" --drive "$TEST_DRIVE"
  --lidar_root "$LIDAR_ROOT" --max_samples "$EVAL_SAMPLES" --dense_sources 8
  --max_points 4096 --geometry_cache "$VGGT_TEST" --device cuda
)
if [[ "$WITH_PERCEPTUAL" == "1" ]]; then
  EVAL_COMMON+=(--eval_ssim_lpips)
fi

# Keep Ns=1 to measure the extreme sparse regime, but do not describe its
# geometry as vehicle-motion-scaled: that is mathematically impossible with
# one temporal frame and is labeled explicitly in the output records.
for ns in 1 2; do
  run_logged "${ROOT}/eval_ground_ns${ns}.log" "${PY[@]}" scripts/eval_unified_bev_probe.py \
    "${EVAL_COMMON[@]}" --sparse_sources "$ns" \
    --records_out "${ROOT}/eval_ground_ns${ns}.jsonl"
  run_logged "${ROOT}/eval_aligned_ns${ns}.log" "${PY[@]}" scripts/eval_unified_bev_probe.py \
    "${EVAL_COMMON[@]}" --stage_b "$STAGE_B_SAT" --sparse_sources "$ns" \
    --records_out "${ROOT}/eval_aligned_ns${ns}.jsonl"
  run_logged "${ROOT}/eval_fixed_xy_ns${ns}.log" "${PY[@]}" scripts/eval_unified_bev_probe.py \
    "${EVAL_COMMON[@]}" --stage_b "$STAGE_B_XY" --sparse_sources "$ns" \
    --records_out "${ROOT}/eval_fixed_xy_ns${ns}.jsonl"
  run_logged "${ROOT}/eval_random_tile_ns${ns}.log" "${PY[@]}" scripts/eval_unified_bev_probe.py \
    "${EVAL_COMMON[@]}" --stage_b "$STAGE_B_SAT" --sparse_sources "$ns" \
    --sat_random_tile --records_out "${ROOT}/eval_random_tile_ns${ns}.jsonl"
  run_logged "${ROOT}/eval_cross5m_ns${ns}.log" "${PY[@]}" scripts/eval_unified_bev_probe.py \
    "${EVAL_COMMON[@]}" --stage_b "$STAGE_B_SAT" --sparse_sources "$ns" \
    --sat_shift_cross_m 5 --records_out "${ROOT}/eval_cross5m_ns${ns}.jsonl"

  for control in ground fixed_xy random_tile cross5m; do
    "${PY[@]}" scripts/compare_unified_bev_paired.py \
      --a "${ROOT}/eval_aligned_ns${ns}.jsonl" \
      --b "${ROOT}/eval_${control}_ns${ns}.jsonl" \
      --keys full_height_fill_mae,full_height_fill_rmse,full_rgb_lowfreq_psnr,full_rgb_supported_psnr,full_psnr,full_absrel \
      >>"${ROOT}/paired_claim_comparisons.txt"
  done
done

run_logged "${ROOT}/c2_frozen_query.log" "${PY[@]}" scripts/consistency_unified_bev_multichain.py \
  --stage_a "$STAGE_A" --stage_b_sat "$STAGE_B_SAT" --stage_b_xy "$STAGE_B_XY" \
  --manifest "$TEST_MANIFEST" --drive "$TEST_DRIVE" --eval_samples "$EVAL_SAMPLES" \
  --lidar_root "$LIDAR_ROOT" --ns_list 1,2 --geometry_cache "$VGGT_TEST" \
  --records_out "${ROOT}/c2_frozen_query.jsonl" --device cuda

echo "[chain] ALL_DONE root=${ROOT} $(date --iso-8601=seconds)" >>"$LOG"
echo "$ROOT"
