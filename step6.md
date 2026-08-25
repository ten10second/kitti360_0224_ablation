# Step 6 — Structured Comparison and Novelty Level

Timestamp: 2026-08-24T19:37:00+08:00

- **Proposed work**
  - Title: Dense-ground-defined cross-modal recovery of a ground-generative BEV world latent
  - Date: —
  - Source: —
  - Problem framing: Learn one georeferenced latent from dense ground observations, then recover it from registered satellite plus sparse ground observations; use one frozen pose-query decoder for RGB and geometry.
  - Core mechanism: Asymmetric two-stage teacher/reference construction, frozen world decoder, and cross-modal completion into the exact teacher space.
  - Key insight: Dense ground evidence should define what is true/renderable; satellite should act as a registered global prior only where sparse local evidence is missing or ambiguous.
  - Application domain: KITTI-360-like urban driving scenes with satellite, street imagery, camera motion and LiDAR.

- **Prior work A — Cross-View Splatter**
  - Title: Cross-View Splatter: Feed-Forward View Synthesis with Georeferenced Images
  - Date: 2026-05
  - Source: arXiv:2605.19656
  - Problem framing: Registered satellite plus sparse ground images feed-forward to a georeferenced 3D scene and NVS.
  - Core mechanism: Joint VGGT/satellite transformer with DPT depth/height and merged pixel-aligned Gaussians.
  - Key insight: Satellite supplies global geometry/coverage absent from sparse ground views.
  - Application domain: Outdoor georeferenced scenes.
  - Axes matching: 3/4 (problem, insight, domain); core mechanism differs.
  - Level: **Level 2 — High Overlap** (*one axis differs*).

- **Prior work B — Geo2**
  - Title: Geo2: Geometry-Guided Cross-view Geo-Localization and Image Synthesis
  - Date: 2026-06
  - Source: CVPR 2026
  - Problem framing: Shared satellite/ground representation for localization and bidirectional synthesis.
  - Core mechanism: VGGT geometry features, shared geometry-aware embeddings, flow-matching image generator, consistency loss.
  - Key insight: Shared geometry-aware latent bridges aerial/ground views and tasks.
  - Application domain: Urban satellite/panorama cross-view datasets.
  - Axes matching: 3/4 (mechanism at shared-latent level, insight, domain); persistent world-query framing differs.
  - Level: **Level 2 — High Overlap** (*one axis differs*).

- **Prior work C — Sat2Density**
  - Title: Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs
  - Date: 2023-10
  - Source: ICCV 2023
  - Problem framing: Satellite-conditioned, geometry-aware ground-view synthesis.
  - Core mechanism: Satellite-to-explicit-density volume plus volume/render network.
  - Key insight: Geometry is the crucial information learned from satellite-ground pairs.
  - Application domain: Urban satellite/panorama imagery.
  - Axes matching: 3/4 (problem at satellite-to-ground level, insight, domain); teacher-recovery mechanism differs.
  - Level: **Level 2 — High Overlap** (*one axis differs*).

- **Prior work D — Sat2Scene**
  - Title: Sat2Scene: 3D Urban Scene Generation from Satellite Images with Diffusion
  - Date: 2024-06
  - Source: CVPR 2024
  - Problem framing: Satellite-conditioned queryable urban 3D scene generation.
  - Core mechanism: Satellite-derived geometry, sparse 3D diffusion, point features and neural renderer.
  - Key insight: Explicit scene representation gives arbitrary-view and temporal consistency.
  - Application domain: Urban satellite/ground scene generation.
  - Axes matching: 3/4 (problem, insight, domain); dense-ground teacher recovery differs.
  - Level: **Level 2 — High Overlap** (*one axis differs*).

- **Prior work E — CrossModalityDiffusion**
  - Title: CrossModalityDiffusion: Multi-Modal Novel View Synthesis with Unified Intermediate Representation
  - Date: 2025-02
  - Source: WACV 2025 workshop
  - Problem framing: Multi-modal sparse-input NVS.
  - Core mechanism: Modality-specific encoders trained into one geometry-aware feature volume, common rendering, modality-specific decoders.
  - Key insight: Heterogeneous sensors can share one intermediate scene representation.
  - Application domain: Synthetic EO/LiDAR/SAR cars.
  - Axes matching: 2/4 (core mechanism at abstract level, key insight); asymmetric teacher and real geospatial domain differ.
  - Level: **Level 3 — Medium Overlap** (*two axes differ*).

- **Prior work F — LVSM**
  - Title: LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias
  - Date: 2024-10
  - Source: arXiv:2410.17242
  - Problem framing: Sparse posed-image feed-forward NVS.
  - Core mechanism: Fixed latent scene tokens plus target-ray decoder.
  - Key insight: A learned independent scene latent can support fast rendering.
  - Application domain: Generic object and scene datasets.
  - Axes matching: 2/4 (problem at sparse NVS level, latent/decoder mechanism); satellite insight and domain differ.
  - Level: **Level 3 — Medium Overlap** (*two axes differ*).

- **Prior work G — AerialMegaDepth**
  - Title: AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis
  - Date: 2025-04
  - Source: arXiv:2504.13157
  - Problem framing: Aerial-ground geometry, registration and NVS.
  - Core mechanism: Co-registered hybrid dataset and GFM/NVS fine-tuning.
  - Key insight: An overhead image can act as a map that stitches sparse ground observations.
  - Application domain: Outdoor aerial-ground reconstruction.
  - Axes matching: 2/4 (problem/domain); latent-recovery mechanism and satellite-specific insight differ.
  - Level: **Level 3 — Medium Overlap** (*two axes differ*).

## Overall verdict

**Level 2 — High Overlap.** Cross-View Splatter, Geo2, Sat2Density, and Sat2Scene each overlap on three of the four novelty axes. The only defensible remaining axis is not “satellite as geometry,” “shared satellite-ground latent,” “VGGT geometry,” “feed-forward NVS,” or “queryable urban scene” separately. It is the asymmetric construction of a dense-ground-defined and frozen generative world space, followed by cross-modal recovery into that exact space and frozen RGB/geometry evidence.
