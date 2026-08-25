#!/usr/bin/env bash
# Chunk-mode claim probe: the unit of ground evidence is a route chunk.
# Dense Stage A = all N_c chunks; sparse Stage B drops the middle chunks
# (kept K of N_c) with a guard band; the satellite completes the hole.
# Evaluates aligned / random-tile / cross-shifted satellite controls per K,
# disjoint-chunk C2, and ground-equivalent acquisition length.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=(conda run --no-capture-output -n maskgit python)
TAG="${TAG:-claim_chunk_20260825}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-256}"
EVAL_SAMPLES="${EVAL_SAMPLES:-32}"
WITH_PERCEPTUAL="${WITH_PERCEPTUAL:-0}"
STAGE_A_STEPS="${STAGE_A_STEPS:-20000}"
STAGE_B_STEPS="${STAGE_B_STEPS:-10000}"
VGGT_WEIGHTS="${VGGT_WEIGHTS:-/home/shizhm/Downloads/vggt.pt}"
LIDAR_ROOT="${LIDAR_ROOT:-/media/shizhm/sda2/KITTI360_lidar/data_3d_raw}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl}"
TEST_DRIVE="${TEST_DRIVE:-2013_05_28_drive_0003_sync}"
KEPT_CHOICES="${KEPT_CHOICES:-1,2,3}"

ROOT="runs/unified_bev_${TAG}"
LOG="${ROOT}/chain.log"
VGGT_TRAIN="${ROOT}/cache_vggt_chunk_train"
VGGT_TEST="${ROOT}/cache_vggt_chunk_test"
STAGE_A_DIR="${ROOT}/stage_a"
STAGE_A="${STAGE_A_DIR}/stage_a.pt"
STAGE_B_SAT_DIR="${ROOT}/stage_b_aligned_satellite"
STAGE_B_XY_DIR="${ROOT}/stage_b_fixed_xy"
STAGE_B_SAT="${STAGE_B_SAT_DIR}/stage_b.pt"
STAGE_B_XY="${STAGE_B_XY_DIR}/stage_b.pt"
mkdir -p "$ROOT"
: >"${ROOT}/paired_claim_comparisons.txt"

[[ -f "$VGGT_WEIGHTS" ]] || { echo "missing VGGT weights: $VGGT_WEIGHTS" >&2; exit 2; }

run_logged() {
  local output="$1"; shift
  local attempt
  for attempt in 1 2; do
    if "$@" >"$output" 2>&1; then return 0; fi
    echo "[chain] failed attempt=${attempt} output=${output} command=$*" >>"$LOG"
  done
  return 1
}

echo "[chain] start tag=${TAG} $(date --iso-8601=seconds)" >>"$LOG"

# Chunk caches: one independent joint forward per chunk (c0..c3); every
# condition assembles lift rows from these entries, no extra inference.
run_logged "${ROOT}/build_vggt_train.log" "${PY[@]}" scripts/build_vggt_street_cache.py \
  --chunked --raw_dataset --weights "$VGGT_WEIGHTS" --out "$VGGT_TRAIN" \
  --manifest "$TRAIN_MANIFEST" --drive none --lidar_root "$LIDAR_ROOT" \
  --max_samples "$TRAIN_SAMPLES" --subset_specs 0,1,2,3 --device cuda
run_logged "${ROOT}/build_vggt_test.log" "${PY[@]}" scripts/build_vggt_street_cache.py \
  --chunked --raw_dataset --weights "$VGGT_WEIGHTS" --out "$VGGT_TEST" \
  --manifest "$TEST_MANIFEST" --drive "$TEST_DRIVE" --lidar_root "$LIDAR_ROOT" \
  --max_samples "$EVAL_SAMPLES" --subset_specs 0,1,2,3 --device cuda
echo "[chain] chunk caches complete $(date --iso-8601=seconds)" >>"$LOG"

run_logged "${ROOT}/stage_a.log" "${PY[@]}" scripts/train_unified_bev_stage_a.py \
  --chunked --out "$STAGE_A_DIR" --steps "$STAGE_A_STEPS" \
  --max_samples "$TRAIN_SAMPLES" --geometry_cache "$VGGT_TRAIN" \
  --geometry_weight 0.1 --batch_size 2 --num_workers 0 --device cuda --seed 0 \
  --manifest "$TRAIN_MANIFEST" --drive none --lidar_root "$LIDAR_ROOT"
echo "[chain] Stage A (all-chunk dense interface) complete" >>"$LOG"

STAGE_B_COMMON=(
  --chunked --stage_a "$STAGE_A" --steps "$STAGE_B_STEPS"
  --sparse_source_choices "$KEPT_CHOICES"
  --manifest "$TRAIN_MANIFEST" --drive none --lidar_root "$LIDAR_ROOT"
  --max_samples "$TRAIN_SAMPLES" --geometry_cache "$VGGT_TRAIN"
  --batch_size 2 --num_workers 0 --device cuda --seed 0
)
run_logged "${ROOT}/stage_b_aligned_satellite.log" "${PY[@]}" scripts/train_unified_bev_stage_b.py \
  --out "$STAGE_B_SAT_DIR" --fusion residual --sat_encoder vit \
  --latent_weight 0.01 --anchor_weight 1.0 --geometry_fill_weight 1.0 \
  --rgb_lowfreq_weight 0.1 --rgb_observed_weight 0.1 "${STAGE_B_COMMON[@]}"
run_logged "${ROOT}/stage_b_fixed_xy.log" "${PY[@]}" scripts/train_unified_bev_stage_b.py \
  --out "$STAGE_B_XY_DIR" --fusion coordinate_only \
  --latent_weight 0.01 --anchor_weight 1.0 --geometry_fill_weight 1.0 \
  --rgb_lowfreq_weight 0.1 --rgb_observed_weight 0.1 "${STAGE_B_COMMON[@]}"
echo "[chain] Stage B controls complete" >>"$LOG"

EVAL_COMMON=(
  --stage_a "$STAGE_A" --manifest "$TEST_MANIFEST" --drive "$TEST_DRIVE"
  --lidar_root "$LIDAR_ROOT" --max_samples "$EVAL_SAMPLES"
  --geometry_cache "$VGGT_TEST" --kept_choices "$KEPT_CHOICES" --device cuda
)

run_logged "${ROOT}/eval_aligned.log" "${PY[@]}" scripts/eval_unified_bev_chunk_probe.py \
  "${EVAL_COMMON[@]}" --stage_b "$STAGE_B_SAT" \
  --records_out "${ROOT}/eval_aligned.jsonl"
run_logged "${ROOT}/eval_ground.log" "${PY[@]}" scripts/eval_unified_bev_chunk_probe.py \
  "${EVAL_COMMON[@]}" --records_out "${ROOT}/eval_ground.jsonl"
run_logged "${ROOT}/eval_fixed_xy.log" "${PY[@]}" scripts/eval_unified_bev_chunk_probe.py \
  "${EVAL_COMMON[@]}" --stage_b "$STAGE_B_XY" \
  --records_out "${ROOT}/eval_fixed_xy.jsonl"
run_logged "${ROOT}/eval_random_tile.log" "${PY[@]}" scripts/eval_unified_bev_chunk_probe.py \
  "${EVAL_COMMON[@]}" --stage_b "$STAGE_B_SAT" --sat_random_tile \
  --records_out "${ROOT}/eval_random_tile.jsonl"
run_logged "${ROOT}/eval_cross5m.log" "${PY[@]}" scripts/eval_unified_bev_chunk_probe.py \
  "${EVAL_COMMON[@]}" --stage_b "$STAGE_B_SAT" --sat_shift_cross_m 5 \
  --records_out "${ROOT}/eval_cross5m.jsonl"
echo "[chain] chunk evaluations complete" >>"$LOG"

for control in ground fixed_xy random_tile cross5m; do
  "${PY[@]}" scripts/l_equiv_analysis.py \
    --a "${ROOT}/eval_aligned.jsonl" --b "${ROOT}/eval_${control}.jsonl" \
    >>"${ROOT}/paired_claim_comparisons.txt"
done

run_logged "${ROOT}/c2_chunk.log" "${PY[@]}" scripts/consistency_unified_bev_chunk.py \
  --stage_a "$STAGE_A" --stage_b_sat "$STAGE_B_SAT" --stage_b_xy "$STAGE_B_XY" \
  --manifest "$TEST_MANIFEST" --drive "$TEST_DRIVE" --eval_samples "$EVAL_SAMPLES" \
  --geometry_cache "$VGGT_TEST" --chain_a_chunks 0,2 --chain_b_chunks 1,3 \
  --records_out "${ROOT}/c2_chunk.jsonl" --device cuda

echo "[chain] ALL_DONE root=${ROOT} $(date --iso-8601=seconds)" >>"$LOG"
echo "$ROOT"
