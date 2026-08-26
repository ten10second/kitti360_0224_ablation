#!/usr/bin/env bash
# Persistent georeferenced world-state probe.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=(conda run --no-capture-output -n maskgit python)
TAG="${TAG:-world_state_$(date +%Y%m%d)}"
MAX_SCENES="${MAX_SCENES:-2}"
INTERFACE_STEPS="${INTERFACE_STEPS:-20}"
ASSIM_STEPS="${ASSIM_STEPS:-20}"
LIDAR_ROOT="${LIDAR_ROOT:-/media/shizhm/sda2/KITTI360_lidar/data_3d_raw}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl}"
TEST_DRIVE="${TEST_DRIVE:-2013_05_28_drive_0003_sync}"

ROOT="runs/world_state_${TAG}"
mkdir -p "$ROOT"
python3 - <<PY
import json, subprocess, pathlib
p = pathlib.Path("$ROOT/run_metadata.json")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
p.write_text(json.dumps({
    "task_family": "persistent_world_state",
    "schema_version": "world_state_v1",
    "git_commit": commit,
    "tag": "$TAG",
}, indent=2))
PY

"${PY[@]}" scripts/build_world_state_targets.py \
  --manifest "$TRAIN_MANIFEST" --split train --lidar_root "$LIDAR_ROOT" \
  --out "$ROOT/targets_train" --max_scenes "$MAX_SCENES" --drive none || \
"${PY[@]}" scripts/build_world_state_targets.py \
  --manifest "$TRAIN_MANIFEST" --split train --lidar_root "$LIDAR_ROOT" \
  --out "$ROOT/targets_train" --max_scenes "$MAX_SCENES"

"${PY[@]}" scripts/build_world_state_targets.py \
  --manifest "$TEST_MANIFEST" --split test --lidar_root "$LIDAR_ROOT" \
  --drive "$TEST_DRIVE" --out "$ROOT/targets_test" --max_scenes 1

"${PY[@]}" scripts/build_world_vggt_cache.py \
  --scenes "$ROOT/targets_train" --manifest "$TRAIN_MANIFEST" \
  --lidar_root "$LIDAR_ROOT" --out "$ROOT/vggt_cache_train" --device cuda

"${PY[@]}" scripts/build_world_vggt_cache.py \
  --scenes "$ROOT/targets_test" --manifest "$TEST_MANIFEST" \
  --lidar_root "$LIDAR_ROOT" --out "$ROOT/vggt_cache_test" --device cuda

"${PY[@]}" scripts/train_world_state_interface.py \
  --scenes "$ROOT/targets_train" --out "$ROOT/interface" \
  --steps "$INTERFACE_STEPS" --device cuda

for branch in sat_ground xy_ground ground_only one_shot; do
  "${PY[@]}" scripts/train_world_state_assimilation.py \
    --scenes "$ROOT/targets_train" --interface "$ROOT/interface/world_interface.pt" \
    --vggt_cache "$ROOT/vggt_cache_train" \
    --out "$ROOT/assim_${branch}" --branch "$branch" --steps "$ASSIM_STEPS" --device cuda
done

for control in aligned xy random shift_cross sat_only ground_only one_shot world_upper; do
  assim="$ROOT/assim_sat_ground/assimilation.pt"
  case "$control" in
    xy) assim="$ROOT/assim_xy_ground/assimilation.pt" ;;
    ground_only) assim="$ROOT/assim_ground_only/assimilation.pt" ;;
    one_shot) assim="$ROOT/assim_one_shot/assimilation.pt" ;;
  esac
  "${PY[@]}" scripts/eval_world_state_trajectory.py \
    --scenes "$ROOT/targets_test" --interface "$ROOT/interface/world_interface.pt" \
    --assimilation "$assim" --control "$control" \
    --vggt_cache "$ROOT/vggt_cache_test" \
    --records_out "$ROOT/eval_${control}.jsonl" --device cuda
done

"${PY[@]}" scripts/compare_world_state_paired.py \
  --a "$ROOT/eval_aligned.jsonl" --b "$ROOT/eval_xy.jsonl" \
  | tee "$ROOT/paired_aligned_vs_xy.json"

echo "[world-state] ALL_DONE root=$ROOT"
echo "$ROOT"
