# Step 2 — Search and Deduplicate

Timestamp: 2026-08-24T19:26:41+08:00

## Queries

1. Original-problem: `satellite street-view sparse ground unified scene latent novel view synthesis geometry`
2. Broad-domain: `cross-view satellite ground neural scene representation BEV rendering`
3. Method-signature: `dense-view teacher latent sparse-view reconstruction frozen decoder scene representation distillation`

Window: 2020–2026. Sources requested: arXiv, DBLP, OpenAlex, OpenReview, Semantic Scholar, Crossref. Ten results per source and query.

## Source failures surfaced verbatim

- `[openreview] Error: openreview not installed. pip install openreview-py`
- `[open_alex] Error: 504 Server Error: Gateway Timeout ...` on all three long semantic queries.
- DBLP returned zero results for all three queries.
- Semantic Scholar and Crossref rate-limited individual requests but returned results after their built-in waits.

## Deduplicated search pool

Title normalization used lowercase plus collapsed whitespace. Near-identical arXiv/Crossref/Semantic Scholar records were merged.

| ID | Title | Year | Provenance |
|---:|---|---:|---|
| P01 | Sparse-View Visual-Acoustic Latent Learning for Novel-View Audio Synthesis | 2026 | Crossref Q1 |
| P02 | Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations | 2022 | Crossref Q1 |
| P03 | wildNeRF: Novel view synthesis of in-the-wild dynamic scenes using sparse monocular view data | — | Crossref Q1 |
| P04 | A Unified Deep Learning Approach for Foveated Rendering & Novel View Synthesis from Sparse RGB-D Light Fields | 2020 | Crossref Q1 |
| P05 | Pseudo-View-Driven Gaussian Optimization for Sparse Novel View Synthesis | — | Crossref Q1 |
| P06 | Cross-View Token-Enhanced Neural Radiance Fields for Sparse-View Novel View Synthesis | 2026 | Crossref Q1 |
| P07 | Sparse Input Novel View Synthesis using Gaussian Splatting for Power Distribution Network Scene | 2025 | Crossref Q1 |
| P08 | Geometry-Aware Scene Configurations for Novel View Synthesis | 2026 | Crossref Q1 |
| P09 | MetaSplats: Rapid Sparse 2D View to 3D Novel View Synthesis | 2024 | Crossref Q1 |
| P10 | Segmentation-Guided Neural Radiance Fields for Novel Street View Synthesis | 2025 | Crossref Q1 |
| P11 | CrossViewDiff: A Cross-View Diffusion Model for Satellite-to-Street View Synthesis | 2024 | arXiv Q1 |
| P12 | DENSER: Depth-Guided Ensemble with Staged EFA-GS Reconstruction for Soccer Novel View Synthesis | 2026 | arXiv Q1 |
| P13 | Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization | 2021 | arXiv Q1 |
| P14 | Geometry-Guided Street-View Panorama Synthesis from Satellite Imagery | 2021 | arXiv Q1 |
| P15 | Sat2Vid: Street-view Panoramic Video Synthesis from a Single Satellite Image | 2020 | arXiv Q1 |
| P16 | Seeing through Satellite Images at Street Views | 2025 | arXiv Q1 |
| P17 | Latent-Y: A Lab-Validated Autonomous Agent for De Novo Drug Design | 2026 | arXiv Q1/Q3 |
| P18 | FreeGen: Feed-Forward Reconstruction-Generation Co-Training for Free-Viewpoint Driving Scene Synthesis | 2025 | arXiv Q1 |
| P19 | Latent-X: An Atom-level Frontier Model for De Novo Protein Binder Design | 2025 | arXiv Q1/Q3 |
| P20 | Drug-like antibodies with low immunogenicity in human panels designed with Latent-X2 | 2025 | arXiv Q1/Q3 |
| P21 | CrossModalityDiffusion: Multi-Modal Novel View Synthesis with Unified Intermediate Representation | 2025 | Semantic Scholar Q1; CVF verified |
| P22 | UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis | 2025 | Semantic Scholar Q1 |
| P23 | Geometry-Aware Satellite-to-Ground Image Synthesis for Urban Areas | 2020 | Semantic Scholar Q1; CVF verified |
| P24 | Sat3DGen: Comprehensive Street-Level 3D Scene Generation from Single Satellite Image | 2026 | Semantic Scholar Q1 |
| P25 | Geo-EVS: Geometry-Conditioned Extrapolative View Synthesis for Autonomous Driving | 2026 | Semantic Scholar Q1 |
| P26 | LangFlash: Feed-forward 3D Language Gaussian Splatting from Sparse Unposed Images | 2026 | Semantic Scholar Q1 |
| P27 | ReX-Shot: Single-Image Rephotography via Geometry- and Camera-Grounded Generation | 2026 | Semantic Scholar Q1 |
| P28 | StreetForward: Perceiving Dynamic Street with Feedforward Causal Attention | 2026 | Semantic Scholar Q1 |
| P29 | Feed-Forward Gaussian Splatting from Sparse Aerial Views | 2026 | Semantic Scholar Q1/Q3 |
| P30 | AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views | 2025 | Semantic Scholar Q1 |
| P31 | Neural Scene Graphs for Dynamic Scenes | 2020 | arXiv Q2 |
| P32 | SG-BEV: Satellite-Guided BEV Fusion for Cross-View Semantic Segmentation | 2024 | arXiv/Crossref Q2 |
| P33 | Cross-View Splatter: Feed-Forward View Synthesis with Georeferenced Images | 2026 | arXiv Q2; primary page verified |
| P34 | Window-to-Window BEV Representation Learning for Limited FoV Cross-View Geo-localization | 2024/2026 | arXiv/Crossref Q2 |
| P35 | Bird's-Eye-View Scene Graph for Vision-Language Navigation | 2023 | arXiv Q2 |
| P36 | 3D-Aware Multi-Task Learning with Cross-View Correlations for Dense Scene Understanding | 2025 | arXiv Q2 |
| P37 | Cross-view image geo-localization with Panorama-BEV Co-Retrieval Network | 2024 | arXiv Q2 |
| P38 | Monocular BEV Perception of Road Scenes via Front-to-Top View Projection | 2022 | arXiv Q2 |
| P39 | TiG-BEV: Multi-view BEV 3D Object Detection via Target Inner-Geometry Learning | 2022 | arXiv Q2 |
| P40 | Call to Protect the Dark and Quiet Sky from Harmful Interference by Satellite Constellations | 2024 | arXiv Q2 |
| P41 | A Review of Neural Radiance Fields and Gaussian Splatting Techniques for Scene Representation and Rendering | 2026 | Crossref Q2 |
| P42 | FB-BEV: BEV Representation from Forward-Backward View Transformations | 2023 | Crossref Q2 |
| P43 | Virtual Kitchen Scene Modelling Based on 3ds Max With Rendering View | 2022 | Crossref Q2 |
| P44 | ORSA-T: Multi-View Object-Centric Scene Representation Learning with Slot Attention and Transformer | 2025 | Crossref Q2 |
| P45 | From Satellite to Ground: Satellite Assisted Visual Localization with Cross-view Semantic Matching | 2024 | Crossref Q2 |
| P46 | MPM-GS: Optimizing Sparse-View 3D Scene Reconstruction with Virtual View Rendering and Multimodal Regularization | 2025 | Crossref Q2 |
| P47 | Dynamic Scene Representation in the Era of Neural Rendering: From NeRFs to 3DGSs | 2026 | Crossref Q2 |
| P48 | RETRACTED CHAPTER: CrossViewDiff: A Cross-View Diffusion Model for Satellite-to-Ground Image Synthesis | 2024 | Crossref Q2 |
| P49 | NeuralFloors++: Consistent Street-Level Scene Generation From BEV Semantic Maps | 2024 | Semantic Scholar Q2 |
| P50 | NavBEV: Empowering Self-Supervised UAV-Based Visual Navigation Through 3D BEV Representation | 2025 | Semantic Scholar Q2 |
| P51 | Unifying UAV Cross-View Geo-Localization via 3D Geometric Perception | 2026 | Semantic Scholar Q2 |
| P52 | Sat2Scene: 3D Urban Scene Generation from Satellite Images with Diffusion | 2024 | Semantic Scholar Q2; CVF verified |
| P53 | Neural Groundplans: Persistent Neural Scene Representations from a Single Image | 2022 | Semantic Scholar Q2 |
| P54 | Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs | 2023 | Semantic Scholar Q2; CVF verified |
| P55 | Enhancing 3D Gaussian Splatting with Semantic and Geometric Priors: Bridging Neural Rendering and 3D City Modeling | 2026 | Semantic Scholar Q2 |
| P56 | Satellite True Digital Orthophoto Map Generation Without Elevation Data: A New NeRF-Based Method | 2024 | Semantic Scholar Q2 |
| P57 | Epipolar Focus Spectrum: A Novel Light Field Representation and Application in Dense-view Reconstruction | 2022 | arXiv Q3 |
| P58 | X-LRM: X-ray Large Reconstruction Model for Extremely Sparse-View Computed Tomography Recovery in One Second | 2025 | arXiv Q3 |
| P59 | DGTR: Distributed Gaussian Turbo-Reconstruction for Sparse-View Vast Scenes | 2024 | arXiv Q3 |
| P60 | ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model | 2024 | arXiv Q3 |
| P61 | UniForward: Unified 3D Scene and Semantic Field Reconstruction via Feed-Forward Gaussian Splatting from Only Sparse-View Images | 2025 | arXiv/Semantic Scholar Q3 |
| P62 | Omni-Scene: Omni-Gaussian Representation for Ego-Centric Sparse-View Scene Reconstruction | 2024/2025 | arXiv/Crossref Q3 |
| P63 | Sp2360: Sparse-view 360 Scene Reconstruction using Cascaded 2D Diffusion Priors | 2024 | arXiv Q3 |
| P64 | Energy-Efficient 3D Scene Representation: A Review of Sparse-View Gaussian Splatting for Sustainable AI | 2026 | Crossref Q3 |
| P65 | SparseDet: Towards Efficient Multi-View 3D Object Detection Via Sparse Scene Representation | — | Crossref Q3 |
| P66 | Vvbpnet: Deep Learning Model in View-by-View Backprojection Domain for Sparse-View CBCT Reconstruction | — | Crossref Q3 |
| P67 | Efficient Sparse-View 3D Scene Reconstruction in Underwater Using Optimized Gaussian Splatting | 2025 | Crossref Q3 |
| P68 | PE-INeR: Prior-Embedded Implicit Neural Representation for Sparse-View CBCT Reconstruction | 2024 | Crossref Q3 |
| P69 | Training-Free Instance-Aware 3D Scene Reconstruction and Diffusion-Based View Synthesis from Sparse Images | 2025 | Crossref Q3 |
| P70 | Optimizing View Angles for Sparse-View 3D Reconstruction Using Elastic Registration | — | Crossref Q3 |
| P71 | Frequency-Regularized Neural Representation Method for Sparse-View Tomographic Reconstruction | 2024 | Crossref Q3 |
| P72 | LRR-CED: Low-Resolution Reconstruction-Aware Convolutional Encoder–Decoder Network for Direct Sparse-View CT Image Reconstruction | 2022 | Crossref Q3 |
| P73 | Rethinking Image-to-3D Generation with Sparse Queries: Efficiency, Capacity, and Input-View Bias | 2026 | Semantic Scholar Q3 |
| P74 | 4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere | 2026 | Semantic Scholar Q3 |
| P75 | SemanticSplat: Feed-Forward 3D Scene Understanding with Language-Aware Gaussian Fields | 2025 | Semantic Scholar Q3 |
| P76 | Bridging 3D Gaussians and Semantic Occupancy for Comprehensive Open-Vocabulary Scene Understanding from Unposed Images | 2026 | Semantic Scholar Q3 |
| P77 | ID-NeRF: Indirect Diffusion-Guided Neural Radiance Fields for Generalizable View Synthesis | 2024 | Semantic Scholar Q3 |
| P78 | F-RNG: Feed-Forward Relightable Neural Gaussians | 2026 | Semantic Scholar Q3 |

## Search augmentation

Primary-source web verification added papers missed by the keyword APIs:

| ID | Title | Year | Source |
|---:|---|---:|---|
| A01 | Geo2: Geometry-Guided Cross-view Geo-Localization and Image Synthesis | 2026 | CVPR open access |
| A02 | AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis | 2025 | arXiv:2504.13157 |
| A03 | Satellite to GroundScape — Large-scale Consistent Ground View Generation from Satellite Views | 2025 | arXiv:2504.15786 |
| A04 | LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias | 2024 | arXiv:2410.17242 |
| A05 | SparseFusion: Distilling View-conditioned Diffusion for 3D Reconstruction | 2022 | arXiv:2212.00792 |

Model-recall additions, to be treated as unverified until the full-paper stage:

| ID | Title | Year | Source |
|---:|---|---:|---|
| M01 | pixelNeRF: Neural Radiance Fields from One or Few Images | 2021 | model-recall |
| M02 | IBRNet: Learning Multi-View Image-Based Rendering | 2021 | model-recall |
| M03 | MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo | 2021 | model-recall |

Total deduplicated pool: 86 papers (78 query hits, 5 primary-source augmentations, 3 model-recall additions).
