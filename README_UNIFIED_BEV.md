# Unified KITTI-360 BEV latent probe

This is the new two-stage path described in
`todo/kitti360_unified_bev_latent_implementation_spec.md`.  It is kept
separate from the legacy VQ/AR path.

The local KITTI-360 satellite convention is `512x512`, north-up, vehicle
-centered, and `0.196 m/px`.  LiDAR is read from:

```text
/media/shizhm/sda2/KITTI360_lidar/data_3d_raw
```

## Geometry conventions (fixed 2026-08-20)

The canonical BEV raster is **south-up**: row 0 = min world y, col 0 = min
world x, pixel-center convention (cell `r` center = `origin + (r+0.5)*res`).
`bilinear_splat`, the decoder sampling grid (`bev_grid_from_world_xy`) and
the satellite crop (vertically flipped from the north-up source image) all
share this convention; a unit test locks it in.  The first probe round had
a north/south mirror between the splat writer and the decoder/satellite
readers, plus a half-cell offset — both fixed and QA-verified with
LiDAR-on-satellite overlays (`scripts/qa_unified_bev_alignment.py`).

Ground sources now include the two fisheyes: each of `image_02`/`image_03`
is sampled with three 90-degree virtual perspective crops (yaw -45/0/+45)
via a hand-rolled MEI warp (`world3d/unified_bev/fisheye.py`; cv2.omnidir
is unavailable in this env).  Virtual-crop geometry is QA-verified with
LiDAR overlays (`runs/unified_bev_qa/fisheye_virtual_qa.png`).  Targets
remain the front perspective camera.

## Smoke checks

```bash
conda run -n maskgit python scripts/smoke_unified_bev.py \
  --manifest dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl \
  --drive 2013_05_28_drive_0007_sync

conda run -n maskgit python scripts/check_unified_bev_replay.py \
  --stage_a runs/unified_bev_stage_a_gpu_smoke/stage_a.pt
```

## Probe status (fixed-XY control, 2026-08-21)

- Engineering checks (data/geometry, backprop, replay, frozen decoder): pass.
- Stage A was retrained for 20k steps after fixing the double-divided height
  mean/variance.  Its `Ns=1/2/4/8` RGB, depth and latent metrics strictly
  approach the dense upper bound; dense PSNR is 15.19.
- Stage B now uses the ground-anchored identity-preserving fusion.  At Ns=2,
  aligned residual improves sparse PSNR 13.63 -> 14.33 and AbsRel
  0.181 -> 0.099; Ns=8 is bitwise equal to the ground branch.
- The original coordinate-only result used a trainable `64x128x128` spatial
  template and is retained only as a `learned-template` diagnostic.  B3 now
  uses deterministic metric-relative XY/Fourier features with zero trainable
  positional parameters.  At Ns=2 it reaches 14.30 PSNR versus residual
  14.33; the paired difference (+0.031 dB) is not significant, while residual
  latent L1 is significantly better and depth metrics remain mixed.
- B7/B8 and camera-z depth evaluation are implemented.  Cross-road shifts
  and rotations degrade cleanly; along-road shifts are tolerant, consistent
  with road-layout anisotropy.
- The current gate is therefore nuanced rather than fully passed: satellite
  content improves latent recovery, but the frozen renderer does not turn it
  into a consistent RGB/depth win over fixed XY.  See
  `runs/unified_bev_coordfixed_20260821_REPORT.md` for the corrected baseline,
  paired bootstrap CIs, and exact verdict; the earlier heightfix/fusionfix
  reports are historical.
- Still required for a full-scale claim: geographically held-out drives,
  trajectory-normal/tangent B7 controls, a distant-drive B8 donor and >=3
  seeds with tile-level confidence intervals.
- Operational notes: the external data drive throws transient PIL IO errors
  (reads are retried in `data.py`); run long jobs with `setsid nohup`;
  checkpoints save every 500 steps and `train_unified_bev_stage_a.py
  --resume` continues from them.
