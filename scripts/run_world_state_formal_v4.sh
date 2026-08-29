#!/usr/bin/env bash
# Formal chain v4: LABEL_POLICY v2 targets rebuild -> VGGT cache rebuild ->
# interface -> shared assimilation (DGM anchor + depth consistency) ->
# full E1-E4 control battery -> paired verdict.  One artifact root.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=(conda run --no-capture-output -n maskgit python)
TAG="${TAG:-v4_formal_$(date +%Y%m%d)}"
INTERFACE_STEPS="${INTERFACE_STEPS:-5000}"
ASSIM_STEPS="${ASSIM_STEPS:-8000}"
LIDAR_ROOT="${LIDAR_ROOT:-/media/shizhm/sda2/KITTI360_lidar/data_3d_raw}"
DGM_TILES="${DGM_TILES:-/media/shizhm/sda1/proposal/Cross-View Conditional Coding of Route-Specific Gaussian Scenes/outputs/bw_dgm_dom/tiles}"
KITTI_ROOT="${KITTI_ROOT:-/media/shizhm/sda1/KITTI-360}"
TEST_DRIVE="${TEST_DRIVE:-2013_05_28_drive_0003_sync}"

ROOT="runs/world_state_${TAG}"
TRAIN_TARGETS="$ROOT/targets_train_v2"
TRAIN_VGGT="$ROOT/vggt_cache_train_v2"
TEST_TARGETS="$ROOT/targets_test_v2"
TEST_VGGT="$ROOT/vggt_cache_test_v2"
mkdir -p "$ROOT"
python3 - "$ROOT" "$TAG" <<'PY'
import json, subprocess, sys, pathlib
root, tag = sys.argv[1], sys.argv[2]
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
pathlib.Path(root, "run_metadata.json").write_text(json.dumps({
    "task_family": "persistent_world_state",
    "world_target_version": "official_semantics_surface_v3+labels_v2",
    "git_commit": commit, "tag": tag,
    "interface_steps": 5000, "assimilation_steps": 8000,
    "dgm_anchor": True, "depth_consistency": True,
}, indent=2))
PY
echo "[chain] root=$ROOT"

"${PY[@]}" scripts/build_world_state_targets.py \
  --manifest dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl --split train \
  --lidar_root "$LIDAR_ROOT" --out "$TRAIN_TARGETS" --max_scenes 64 2>&1 | tee "$ROOT/build_targets_train.log"
"${PY[@]}" scripts/build_world_state_targets.py \
  --manifest dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl --split test \
  --lidar_root "$LIDAR_ROOT" --drive "$TEST_DRIVE" --out "$TEST_TARGETS" --max_scenes 1 2>&1 | tee "$ROOT/build_targets_test.log"

"${PY[@]}" scripts/build_world_vggt_cache.py \
  --scenes "$TRAIN_TARGETS" --manifest dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl \
  --lidar_root "$LIDAR_ROOT" --out "$TRAIN_VGGT" --device cuda 2>&1 | tee "$ROOT/build_vggt_train.log"
"${PY[@]}" scripts/build_world_vggt_cache.py \
  --scenes "$TEST_TARGETS" --manifest dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl \
  --lidar_root "$LIDAR_ROOT" --out "$TEST_VGGT" --device cuda 2>&1 | tee "$ROOT/build_vggt_test.log"

"${PY[@]}" scripts/train_world_state_interface.py \
  --scenes "$TRAIN_TARGETS" --out "$ROOT/interface" \
  --steps "$INTERFACE_STEPS" --device cuda 2>&1 | tee "$ROOT/interface.log"

"${PY[@]}" scripts/train_world_state_assimilation.py \
  --scenes "$TRAIN_TARGETS" --interface "$ROOT/interface/world_interface.pt" \
  --vggt_cache "$TRAIN_VGGT" --dgm_tiles "$DGM_TILES" --kitti360_root "$KITTI_ROOT" \
  --out "$ROOT/assim_shared" --steps "$ASSIM_STEPS" --device cuda 2>&1 | tee "$ROOT/assim_shared.log"

for control in aligned xy random shift_cross shift_road sat_only ground_only one_shot world_upper; do
  "${PY[@]}" scripts/eval_world_state_trajectory.py \
    --scenes "$TEST_TARGETS" --interface "$ROOT/interface/world_interface.pt" \
    --assimilation "$ROOT/assim_shared/assimilation.pt" --control "$control" \
    --vggt_cache "$TEST_VGGT" --dgm_tiles "$DGM_TILES" --kitti360_root "$KITTI_ROOT" \
    --records_out "$ROOT/eval_${control}.jsonl" --device cuda 2>&1 | tee -a "$ROOT/eval.log"
done

"${PY[@]}" scripts/compare_world_state_paired.py \
  --a "$ROOT/eval_aligned.jsonl" --b "$ROOT/eval_xy.jsonl" \
  | tee "$ROOT/paired_aligned_vs_xy.json"

python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for f in sorted(root.glob("eval_*.jsonl")):
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    name = f.stem.replace("eval_", "")
    t0 = next((r for r in rows if r["version"] == 0), None)
    last = rows[-1] if rows else None
    def pick(r, keys):
        return {k: r.get(k) for k in keys if r and r.get(k) is not None}
    summary[name] = {
        "t0": pick(t0, ["height_ahead_mae", "height_visited_mae", "visited_fraction", "ahead_fraction"]),
        "final": pick(last, ["version", "height_ahead_mae", "height_visited_mae",
                             "g_update_height", "outside_latent_max",
                             "measurement_target_overlap", "depth_absrel"]),
    }
(root / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
echo "[chain] ALL_DONE root=$ROOT"
