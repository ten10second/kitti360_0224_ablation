#!/usr/bin/env bash
# Overnight formal chain on world_target_v3: interface -> shared assimilation
# -> full E1-E4 control battery -> paired verdict.  Artifacts under one root.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=(conda run --no-capture-output -n maskgit python)
TAG="${TAG:-v3_formal_$(date +%Y%m%d)}"
INTERFACE_STEPS="${INTERFACE_STEPS:-5000}"
ASSIM_STEPS="${ASSIM_STEPS:-8000}"
TRAIN_TARGETS="${TRAIN_TARGETS:-runs/world_state_e0/targets_train_v3}"
TRAIN_VGGT="${TRAIN_VGGT:-runs/world_state_e0/vggt_cache_train_v3}"
TEST_TARGETS="${TEST_TARGETS:-runs/world_state_targets_smoke}"
TEST_VGGT="${TEST_VGGT:-runs/world_state_vggt_smoke}"

ROOT="runs/world_state_${TAG}"
mkdir -p "$ROOT"
python3 - <<PY
import json, subprocess, pathlib
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
pathlib.Path("$ROOT/run_metadata.json").write_text(json.dumps({
    "task_family": "persistent_world_state",
    "world_target_version": "official_semantics_surface_v3",
    "git_commit": commit,
    "tag": "$TAG",
    "interface_steps": $INTERFACE_STEPS,
    "assimilation_steps": $ASSIM_STEPS,
}, indent=2))
PY
echo "[chain] root=$ROOT"

"${PY[@]}" scripts/train_world_state_interface.py \
  --scenes "$TRAIN_TARGETS" --out "$ROOT/interface" \
  --steps "$INTERFACE_STEPS" --device cuda 2>&1 | tee "$ROOT/interface.log"

"${PY[@]}" scripts/train_world_state_assimilation.py \
  --scenes "$TRAIN_TARGETS" --interface "$ROOT/interface/world_interface.pt" \
  --vggt_cache "$TRAIN_VGGT" \
  --out "$ROOT/assim_shared" --steps "$ASSIM_STEPS" --device cuda 2>&1 | tee "$ROOT/assim_shared.log"

for control in aligned xy random shift_cross shift_road sat_only ground_only one_shot world_upper; do
  "${PY[@]}" scripts/eval_world_state_trajectory.py \
    --scenes "$TEST_TARGETS" --interface "$ROOT/interface/world_interface.pt" \
    --assimilation "$ROOT/assim_shared/assimilation.pt" --control "$control" \
    --vggt_cache "$TEST_VGGT" \
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
