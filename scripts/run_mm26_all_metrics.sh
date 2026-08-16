#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RESULTS_ROOT="${RESULTS_ROOT:-/media/zhimiao/Lenovo/mm26_results}"

COMMON_FLAGS=(
  --psnr
  --ssim
  --lpips
  --p_squeeze
  --dino
  --depth
  --lrce
  --fid
  --mv-depth
  --mv-mask-dino
)

run_eval() {
  local model_name="$1"
  local mode_name="$2"
  local subset_name="$3"
  local result_root="$4"
  local output_csv="$5"

  echo "[Run] model=${model_name} mode=${mode_name} subset=${subset_name}"
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/compute_all_metrics.py" \
    --results-root "${result_root}" \
    --mode "${mode_name}" \
    --subsets "${subset_name}" \
    --output-path "${output_csv}" \
    "${COMMON_FLAGS[@]}"
}

# Historical evaluations (kept here for reference, currently disabled)
# run_eval \
#   "ar_direct" \
#   "direct" \
#   "fixed5" \
#   "${RESULTS_ROOT}/ar_direct/direct" \
#   "${RESULTS_ROOT}/ar_direct/metrics_fixed5.csv"
#
# run_eval \
#   "ar_direct" \
#   "direct" \
#   "zero_shot" \
#   "${RESULTS_ROOT}/ar_direct/direct" \
#   "${RESULTS_ROOT}/ar_direct/metrics_zero_shot.csv"
#
# run_eval \
#   "ar_hybrid_archor" \
#   "hybrid" \
#   "fixed5" \
#   "${RESULTS_ROOT}/ar_hybrid_archor/hybrid" \
#   "${RESULTS_ROOT}/ar_hybrid_archor/metrics_fixed5.csv"
#
# run_eval \
#   "ar_hybrid_archor" \
#   "hybrid" \
#   "zero_shot" \
#   "${RESULTS_ROOT}/ar_hybrid_archor/hybrid" \
#   "${RESULTS_ROOT}/ar_hybrid_archor/metrics_zero_shot.csv"
#
# run_eval \
#   "ar_hybrid_enhanced" \
#   "hybrid" \
#   "fixed5" \
#   "${RESULTS_ROOT}/ar_hybrid_enhanced/hybrid" \
#   "${RESULTS_ROOT}/ar_hybrid_enhanced/metrics_fixed5.csv"
#
# run_eval \
#   "ar_hybrid_enhanced" \
#   "hybrid" \
#   "zero_shot" \
#   "${RESULTS_ROOT}/ar_hybrid_enhanced/hybrid" \
#   "${RESULTS_ROOT}/ar_hybrid_enhanced/metrics_zero_shot.csv"

run_eval \
  "ar_direct_consistency_v2" \
  "direct" \
  "fixed5" \
  "${RESULTS_ROOT}/ar_direct_consistency_v2/direct" \
  "${RESULTS_ROOT}/ar_direct_consistency_v2/metrics_fixed5.csv"

run_eval \
  "ar_direct_consistency_v2" \
  "direct" \
  "zero_shot" \
  "${RESULTS_ROOT}/ar_direct_consistency_v2/direct" \
  "${RESULTS_ROOT}/ar_direct_consistency_v2/metrics_zero_shot.csv"

run_eval \
  "ar_enhanced_consistency_v2" \
  "hybrid" \
  "fixed5" \
  "${RESULTS_ROOT}/ar_enhanced_consistency_v2/hybrid" \
  "${RESULTS_ROOT}/ar_enhanced_consistency_v2/metrics_fixed5.csv"

run_eval \
  "ar_enhanced_consistency_v2" \
  "hybrid" \
  "zero_shot" \
  "${RESULTS_ROOT}/ar_enhanced_consistency_v2/hybrid" \
  "${RESULTS_ROOT}/ar_enhanced_consistency_v2/metrics_zero_shot.csv"

echo "[Done] Consistency-v2 metric evaluations finished."
