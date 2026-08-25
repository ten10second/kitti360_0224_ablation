# Step 4 — High-Potential Candidates

Timestamp: 2026-08-24T19:32:00+08:00

Seven papers were selected after abstract/title triage. Selection favors mechanism overlap, the same satellite-ground scene domain, and recent work.

1. **Cross-View Splatter: Feed-Forward View Synthesis with Georeferenced Images (2026).** Highest threat: same satellite+sparse-ground inputs, same global-geometric-prior motivation, feed-forward scene reconstruction, unified georeferenced frame, VGGT/DPT geometry, and NVS.
2. **Geo2: Geometry-Guided Cross-view Geo-Localization and Image Synthesis (2026).** Highest shared-latent threat: explicitly uses VGGT features and a shared geometry-aware satellite/ground latent with consistency training.
3. **Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs (2023).** Strong conceptual threat to the claim that satellite injects geometry: directly learns a renderable density/depth representation from paired satellite-ground imagery.
4. **Sat2Scene: 3D Urban Scene Generation from Satellite Images with Diffusion (2024).** Strong scene-representation threat: feed-forward satellite-conditioned 3D representation with arbitrary-view RGB/depth rendering and no per-scene optimization.
5. **CrossModalityDiffusion: Multi-Modal Novel View Synthesis with Unified Intermediate Representation (2025).** Strong formulation threat: modality-specific encoders are explicitly trained to produce one modality-agnostic geometry-aware feature volume usable by common rendering and modality decoders.
6. **LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias (2024).** Strong latent/decoder threat: fixed-size latent scene tokens independently support target-view querying, testing whether “ground-generative latent” alone is novel.
7. **AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis (2025).** Strong geometry/data threat: demonstrates that an aerial view can act as a map to register sparse ground views and materially improve geometry/NVS.

P24 Sat3DGen was not promoted because its indexed record is newer and less methodologically accessible than Sat2Scene/Cross-View Splatter; P32 SG-BEV was not promoted because its output is semantic segmentation rather than a generative scene representation; A03 GroundScape was not promoted because it focuses on 2D diffusion sequence consistency rather than a queryable 3D/world latent.
