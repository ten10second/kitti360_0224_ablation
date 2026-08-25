# Step 5 — Full-Paper Deep Dive

Timestamp: 2026-08-24T19:35:00+08:00

Seven PDFs were downloaded and text-extracted under `papers/`. The records below supersede Step 3 for these candidates.

## 1. Cross-View Splatter: Feed-Forward View Synthesis with Georeferenced Images

- **Problem framing (verified):** Given one or more GPS/heading-tagged ground images plus one registered orthorectified satellite image, predict a feed-forward 3D Gaussian scene in a shared coordinate frame and render novel ground views.
- **Core mechanism (verified):** VGGT/DINO ground tokens and satellite tokens exchange information through repeated bidirectional cross-attention. DPT heads predict ground depth/confidence and satellite height/confidence; separate ground and satellite Gaussian attributes are backprojected and merged. Training uses camera, depth, height, RGB/perceptual, sky, and Gaussian consistency losses.
- **Key insight (verified):** Tiled web-map imagery supplies global structure and geometry beyond narrow/occluded street observations, with the largest benefit at low context-target overlap.
- **Application domain (verified):** Outdoor georeferenced imagery, including augmented Tanks & Temples and DL3DV-style data plus satellite/terrain supervision.
- **Venue:** arXiv preprint, 2026 (arXiv:2605.19656).
- **Assumptions & scope:** Known ground GPS/heading, known satellite resolution and alignment, reference-camera zero altitude; terrain height supervision is used in training. It predicts visible/covered content rather than a general hallucinated city model.
- **Closest-passage evidence:** `papers/cross_view_splatter_2605_19656.txt:200–220` defines the same satellite+sparse-ground feed-forward reconstruction problem; `:222–340` specifies VGGT, cross-attention, DPT depth/height and unified Gaussians; `:594–607` reports strongest satellite gains at low overlap.
- **Refined overlap:** Problem framing **match**; core mechanism **partial/different** (direct joint Gaussian prediction, no dense-ground teacher latent or frozen decoder); key insight **match**; application domain **match**.

## 2. Geo2: Geometry-Guided Cross-view Geo-Localization and Image Synthesis

- **Problem framing (verified):** Learn shared satellite/ground representations for cross-view geolocalization and bidirectional single-image synthesis.
- **Core mechanism (verified):** VGGT geometry features and CNN semantics feed a dual-branch GeoMap, optimized with InfoNCE into shared geometry-aware embeddings. GeoFlow conditions a flow-matching image generator operating in a pretrained representation-autoencoder latent. Joint training adds a cross-direction consistency loss.
- **Key insight (verified):** A shared geometry-aware latent can bridge the satellite-ground appearance gap and serve both matching and synthesis.
- **Application domain (verified):** CVUSA, CVACT, and VIGOR paired satellite/panorama benchmarks.
- **Venue:** CVPR 2026.
- **Assumptions & scope:** Paired/aligned cross-view images; output is retrieval embedding or a generated 2D counterpart, not a persistent metric scene queried at arbitrary camera poses. No dense-ground reference construction and no frozen world decoder.
- **Closest-passage evidence:** `papers/geo2_cvpr2026.txt:226–340` introduces GeoMap and the shared geometry-aware latent; `:417–497` describes RAE-latent GeoFlow and joint consistency; `:644–655` states the localization/synthesis scope.
- **Refined overlap:** Problem framing **partial**; core mechanism **partial**; key insight **match**; application domain **match**.

## 3. Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs

- **Problem framing (verified):** Predict a 3D-aware ground panorama/video and render depth from a satellite image, learned from paired satellite-ground images without depth supervision.
- **Core mechanism (verified):** DensityNet maps a satellite image to an explicit `H×W×N` density volume. Volumetric rendering produces depth, opacity, and satellite-projected color, followed by RenderNet. Non-sky opacity and illumination supervision are designed to make density geometric rather than appearance-only.
- **Key insight (verified):** Geometry, topology, and geography are the critical transferable information in satellite-ground pairs; an explicit density field enables multi-view-consistent synthesis.
- **Application domain (verified):** CVUSA and CVACT urban satellite/panorama pairs.
- **Venue:** ICCV 2023.
- **Assumptions & scope:** Satellite-only inference; fixed centered panorama/trajectory regime; no sparse ground observations, dense-ground teacher space, or explicit depth ground truth.
- **Closest-passage evidence:** `papers/sat2density_iccv2023.txt:18–105` states the geometry claim; `:154–240` defines the explicit density field and rendering; `:429–450` concludes that the learned satellite-conditioned density is the geometric representation.
- **Refined overlap:** Problem framing **partial/match**; core mechanism **different**; key insight **match**; application domain **match**.

## 4. Sat2Scene: 3D Urban Scene Generation from Satellite Images with Diffusion

- **Problem framing (verified):** Generate a queryable urban 3D scene from satellite-derived geometry and render arbitrary ground/bird views without per-scene test-time optimization.
- **Core mechanism (verified):** Given point geometry, a sparse 3D diffusion model colors foreground points, a 2D diffusion model produces sky, a 3D encoder attaches point features, and a neural volume renderer emits RGB/depth at arbitrary poses.
- **Key insight (verified):** Generating in an explicit 3D sparse representation provides inter-view consistency and arbitrary-view rendering that 2D satellite-to-ground generators lack.
- **Application domain (verified):** HoliCity and OmniCity urban scenes.
- **Venue:** CVPR 2024.
- **Assumptions & scope:** Requires geometry associated with the scene (satellite predictions or dataset geometry); satellite-only scene generation, not sparse-ground-conditioned recovery of a dense-ground-defined latent.
- **Closest-passage evidence:** `papers/sat2scene_cvpr2024.txt:18–115` defines the arbitrary-view scene objective; `:134–205` specifies point representation, 3D diffusion and renderer; `:465–478` emphasizes no test-time optimization.
- **Refined overlap:** Problem framing **partial/match**; core mechanism **different**; key insight **match**; application domain **match**.

## 5. CrossModalityDiffusion: Multi-Modal Novel View Synthesis with Unified Intermediate Representation

- **Problem framing (verified):** Few-shot novel-view generation across heterogeneous EO, LiDAR and SAR input/output modalities.
- **Core mechanism (verified):** Modality-specific encoders create camera-frustum feature volumes; overlapping volumes are combined through one shared MLP and volume renderer into a feature image; modality-specific diffusion decoders generate outputs. Random input/target modalities jointly train encoders to emit the same modality-agnostic feature field.
- **Key insight (verified):** A common geometry-aware intermediate volume lets arbitrary sensor modalities contribute to and decode from one scene representation.
- **Application domain (verified):** Synthetic ShapeNet Cars rendered as EO, LiDAR and SAR; not real urban satellite-ground scenes.
- **Venue:** WACV 2025 workshop.
- **Assumptions & scope:** Known camera poses and synthetic modality alignment; jointly learned common space, not a dense-ground-defined/frozen teacher space; diffusion decoder is modality-specific.
- **Closest-passage evidence:** `papers/cross_modality_diffusion_wacv2025.txt:124–171` defines feature-volume rendering; `:172–260` describes shared MLP/modality modules and random-modality joint training; `:321–344` states the common intermediate representation claim and compute limitation.
- **Refined overlap:** Problem framing **partial**; core mechanism **partial/match**; key insight **match**; application domain **different**.

## 6. LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias

- **Problem framing (verified):** Feed-forward NVS from sparse posed images on object and generic scene datasets.
- **Core mechanism (verified):** The encoder-decoder variant maps image/ray tokens to a fixed number of latent scene tokens; a target-ray-conditioned transformer decoder predicts RGB. It is trained end-to-end with MSE and perceptual rendering losses. A decoder-only variant removes the scene latent and performs better.
- **Key insight (verified):** A learned latent scene representation can support independent fast rendering without explicit NeRF/3DGS structure, although direct decoding may yield higher quality.
- **Application domain (verified):** Objaverse/ABO/GSO and RealEstate10K; no satellite or metric urban geometry.
- **Venue:** arXiv preprint, 2024 (arXiv:2410.17242).
- **Assumptions & scope:** Posed source and target cameras; RGB-only rendering objective; no geometry output, cross-modal recovery, teacher latent, or geographic registration.
- **Closest-passage evidence:** `papers/lvsm_2410_17242.txt:20–115` defines latent scene tokens; `:205–292` details encoder/decoder and photometric loss; its own experiments show the decoder-only variant outperforming the latent variant.
- **Refined overlap:** Problem framing **partial**; core mechanism **partial**; key insight **partial/match**; application domain **different**.

## 7. AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis

- **Problem framing (verified):** Improve geometry, registration and NVS for mixed aerial/ground imagery under extreme viewpoint changes.
- **Core mechanism (verified):** Construct a large hybrid dataset by co-registering real ground images with pseudo-synthetic aerial views/depth from city meshes, then fine-tune DUSt3R/MASt3R and ZeroNVS.
- **Key insight (verified):** Missing co-registered aerial-ground training data is the major bottleneck; an aerial image can serve as an overhead map that stitches sparse ground images into a common frame.
- **Application domain (verified):** Real landmarks and aerial/ground outdoor datasets; aerial perspective rather than orthorectified satellite maps.
- **Venue:** arXiv preprint, 2025 (arXiv:2504.13157).
- **Assumptions & scope:** Hybrid real/pseudo training data and aerial views with overlap; mainly data and geometry-model adaptation, not a new persistent latent representation.
- **Closest-passage evidence:** `papers/aerial_megadepth_2504_13157.txt:22–120` defines the data gap and construction; `:309–347` shows aerial-as-map registration; `:368–390` evaluates NVS and states remaining difficulty.
- **Refined overlap:** Problem framing **partial**; core mechanism **different**; key insight **partial/match**; application domain **partial/match**.
