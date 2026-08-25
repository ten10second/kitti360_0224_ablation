# Step 1 — Decompose the Novelty

Timestamp: 2026-08-24T19:21:47+08:00

## Research problem

Can a georeferenced urban scene be represented in one ground-generative BEV latent space learned from dense street-level observations, and can the same latent be recovered feed-forward from a registered satellite image plus sparse street-level observations so that a frozen decoder supports novel-view RGB and geometric queries without per-scene optimization?

## Proposed novelty

Learn and freeze a dense-ground reference latent space and its RGB/geometry decoder first; then treat satellite imagery as a globally registered overhead prior and sparse ground images as local metric evidence, and train a second-stage encoder/completion model to recover the dense-ground reference latent rather than directly predict a task-specific output.

## Four atomic axes

- **Problem framing:** Cross-view, cross-modal recovery of the same georeferenced scene representation from dense ground, sparse ground, and satellite inputs. Evaluation is feed-forward, geographically held out, and uses a frozen decoder for RGB and geometry.
- **Core mechanism:** Two-stage teacher/reference construction. Stage A learns `Z* = E_ground(G_dense)` and freezes the ground-generative decoder. Stage B predicts `Z_hat = C(E_sat(I_sat), E_ground(G_sparse))` under latent, frozen-render, and frozen-geometry supervision.
- **Key insight:** Satellite imagery should not merely condition a target-view renderer or predict a BEV task map. It should provide globally registered priors that fill spatial/geometry uncertainty left by sparse local observations inside a latent space whose semantics were defined independently by dense ground evidence.
- **Application domain:** Large-scale urban street scenes on KITTI-360-like data with registered satellite crops, known vehicle/camera motion, sparse LiDAR, novel-view RGB, and depth/geometry evaluation.

## Claim boundaries

- VGGT plus vehicle-motion scale anchoring is treated as geometry infrastructure, not the principal novelty.
- A generic BEV encoder, latent distillation loss, DPT head, or frozen decoder alone is not novel.
- The candidate contribution is the combination of a dense-ground-defined generative world latent, cross-modal recovery into that exact frozen space, and causal/geometry evidence that registered satellite content repairs sparse-ground uncertainty.
