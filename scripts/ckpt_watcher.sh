#!/bin/bash
# Snapshot each run's overwritten ckpt.pt as ckpt_step<N>.pt at every save,
# so the sample-quality-vs-step (training sufficiency) curve can be computed.
PY=~/anaconda3/envs/maskgit/bin/python
declare -A LAST

snapshot_pass() {
  for d in runs/icassp27_b2_pilot runs/icassp27_b1_pilot runs/icassp27_b0_pilot; do
    [ -f "$d/ckpt.pt" ] || continue
    m=$(stat -c %Y "$d/ckpt.pt")
    [ "${LAST[$d]:-0}" = "$m" ] && continue
    LAST[$d]=$m
    step=$($PY -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)['step'])" "$d/ckpt.pt" 2>/dev/null)
    [ -z "$step" ] && continue
    if [ ! -f "$d/ckpt_step${step}.pt" ]; then
      cp "$d/ckpt.pt" "$d/ckpt_step${step}.pt"
      echo "$(date +%H:%M:%S) snap $d -> ckpt_step${step}.pt"
    fi
  done
}

while true; do
  snapshot_pass
  if grep -q "all done" runs/logs/pilot_queue.log 2>/dev/null; then
    sleep 30
    snapshot_pass
    echo "$(date +%H:%M:%S) queue finished; watcher exiting"
    break
  fi
  sleep 45
done
