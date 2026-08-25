# Step 3 — Abstract-Level Triage

Timestamp: 2026-08-24T19:32:00+08:00

The overlap score counts plausible matches among problem framing, core mechanism, key insight, and application domain. For candidates later deep-read, this table is superseded by Step 5. For low-ranked API false positives whose abstracts were unavailable in the CLI output, the record is explicitly title/metadata-level rather than pretending a full abstract was read.

| ID | Title / date | Problem framing | Core mechanism | Key insight | Application domain | Score | Source |
|---:|---|---|---|---|---|---:|---|
| P01 | Sparse-View Visual-Acoustic Latent Learning… / 2026 | sparse-view audio synthesis | audiovisual latent | acoustics from sparse views | audio | 0 | Crossref |
| P02 | Scene Representation Transformer / 2022 | sparse-input NVS | set-latent scene representation + renderer | scene can live in learned latent tokens | generic scenes | 2 | DOI/CVPR |
| P03 | wildNeRF / — | sparse monocular NVS | NeRF variant | robust in-the-wild rendering | dynamic scenes | 1 | Crossref |
| P04 | Unified Foveated Rendering & NVS… / 2020 | sparse RGB-D light-field NVS | encoder-decoder | jointly solve foveation/NVS | light fields | 1 | Crossref |
| P05 | Pseudo-View-Driven Gaussian Optimization… / — | sparse NVS | pseudo views + optimized 3DGS | densify supervision | generic scenes | 1 | Crossref |
| P06 | Cross-View Token-Enhanced NeRF… / 2026 | sparse-view NVS | cross-view tokens + NeRF | exchange view evidence | generic scenes | 1 | Crossref |
| P07 | Sparse Input NVS using GS for Power Distribution… / 2025 | sparse NVS | Gaussian splatting | explicit representation | power infrastructure | 1 | Crossref |
| P08 | Geometry-Aware Scene Configurations for NVS / 2026 | NVS | geometry-aware configuration | geometry helps view synthesis | generic/VR | 1 | Crossref |
| P09 | MetaSplats / 2024 | sparse 2D-to-3D NVS | rapid Gaussian prediction | feed-forward reconstruction | generic scenes | 1 | Crossref |
| P10 | Segmentation-Guided NeRF for Novel Street View… / 2025 | street NVS | segmentation-guided NeRF | semantics regularize rendering | street scenes | 1 | Crossref |
| P11 | CrossViewDiff / 2024 | satellite-to-street synthesis | cross-view diffusion | satellite context predicts street appearance | urban cross-view | 2 | arXiv |
| P12 | DENSER / 2026 | soccer NVS | depth-guided staged 3DGS | depth ensembles improve reconstruction | sports | 1 | arXiv |
| P13 | Coming Down to Earth / 2021 | satellite-to-street synthesis | conditional generation | overhead context predicts street view | geolocalization | 2 | arXiv |
| P14 | Geometry-Guided Street-View Panorama Synthesis… / 2021 | satellite-to-panorama | explicit geometry projection + generation | geometry bridges extreme view gap | urban cross-view | 2 | arXiv |
| P15 | Sat2Vid / 2020 | satellite-to-street panoramic video | satellite geometry + 3D/2D correspondences | explicit geometry improves temporal consistency | urban cross-view | 2 | arXiv |
| P16 | Seeing through Satellite Images at Street Views / 2025 | satellite-to-street generation | 3D-aware generative model | satellite contains scene layout | urban cross-view | 2 | arXiv |
| P17 | Latent-Y / 2026 | drug design | molecular foundation model | autonomous design | biology | 0 | arXiv |
| P18 | FreeGen / 2025 | driving-scene free-view synthesis | reconstruction-generation co-training | feed-forward scene generation | driving | 1 | arXiv |
| P19 | Latent-X / 2025 | protein binder design | atom-level model | molecular latent | biology | 0 | arXiv |
| P20 | Drug-like antibodies… Latent-X2 / 2025 | antibody design | molecular generation | immunogenicity optimization | biology | 0 | arXiv |
| P21 | CrossModalityDiffusion / 2025 | sparse multi-modal NVS | modality encoders → common geometry-aware feature volume → modality decoder | force heterogeneous sensors into one intermediate scene representation | synthetic geospatial sensors | 2 | WACV workshop/CVF |
| P22 | UMAMI / 2025 | view synthesis | masked autoregression + deterministic rendering | combine generation and rendering | generic scenes | 1 | Semantic Scholar |
| P23 | Geometry-Aware Satellite-to-Ground Image Synthesis / 2020 | satellite-to-ground image synthesis | explicit ground-object geometry + generator | geometry preserves structure across views | urban cross-view | 2 | CVPR/CVF |
| P24 | Sat3DGen / 2026 | street-level 3D generation from one satellite image | satellite-conditioned 3D generation | overhead image supplies global scene structure | urban 3D | 3 | Semantic Scholar |
| P25 | Geo-EVS / 2026 | extrapolative driving NVS | geometry conditioning | geometry reduces extrapolation ambiguity | driving | 1 | Semantic Scholar |
| P26 | LangFlash / 2026 | sparse unposed 3D reconstruction | language-aware Gaussian splatting | semantics aid feed-forward geometry | generic scenes | 1 | Semantic Scholar |
| P27 | ReX-Shot / 2026 | single-image rephotography | geometry/camera-grounded generation | explicit camera grounding | generic | 1 | Semantic Scholar |
| P28 | StreetForward / 2026 | dynamic street prediction | causal feed-forward attention | temporal causality | driving | 1 | Semantic Scholar |
| P29 | Feed-Forward GS from Sparse Aerial Views / 2026 | sparse aerial reconstruction | feed-forward 3DGS | aerial-view generalization | aerial scenes | 1 | Semantic Scholar |
| P30 | AnySplat / 2025 | feed-forward NVS from unconstrained views | geometry foundation model → 3DGS | strong pretrained geometry generalizes | generic scenes | 1 | TOG/Semantic Scholar |
| P31 | Neural Scene Graphs / 2020 | dynamic NVS | object-compositional NeRF graph | factor dynamic scenes | driving | 1 | arXiv |
| P32 | SG-BEV / 2024 | satellite+ground BEV segmentation | align/fuse satellite and street BEV features | satellite supplies global context missing locally | urban BEV perception | 2 | CVPR |
| P33 | Cross-View Splatter / 2026 | georeferenced satellite+sparse-ground feed-forward NVS | joint VGGT/satellite transformer predicts ground+BEV Gaussians in one frame | satellite is a global geometric prior improving low-coverage ground reconstruction | outdoor georeferenced scenes | 3 | arXiv:2605.19656 |
| P34 | Window-to-Window BEV Representation Learning / 2024/26 | cross-view geolocalization | BEV window correspondence | local BEV alignment | geolocalization | 1 | arXiv/Neural Networks |
| P35 | BEV Scene Graph for VLN / 2023 | navigation | BEV scene graph | spatial memory aids navigation | embodied navigation | 1 | arXiv |
| P36 | 3D-Aware Multi-Task Learning with Cross-View Correlations / 2025 | dense scene understanding | cross-view 3D multi-task features | shared 3D cues help tasks | perception | 1 | arXiv |
| P37 | Panorama-BEV Co-Retrieval Network / 2024 | cross-view geolocalization | panorama/BEV retrieval | shared BEV aids matching | geolocalization | 1 | arXiv |
| P38 | Monocular BEV Perception… / 2022 | monocular BEV perception | front-to-top projection | explicit view transform | driving perception | 1 | arXiv |
| P39 | TiG-BEV / 2022 | multi-view 3D detection | target inner geometry BEV | geometry-aware aggregation | driving detection | 1 | arXiv |
| P40 | Protect the Dark and Quiet Sky… / 2024 | astronomy policy | — | constellation interference | astronomy | 0 | arXiv |
| P41 | Review of NeRF and Gaussian Splatting… / 2026 | survey | literature review | summarize rendering representations | generic scenes | 0 | Crossref |
| P42 | FB-BEV / 2023 | camera BEV perception | forward/backward view transforms | inverse mapping repairs BEV features | driving perception | 1 | ICCV |
| P43 | Virtual Kitchen Scene Modelling… / 2022 | 3ds Max tutorial | manual modeling | rendering workflow | graphics education | 0 | Crossref |
| P44 | ORSA-T / 2025 | object-centric multi-view representation | slot attention + transformer | object factorization | synthetic objects | 1 | Crossref |
| P45 | Satellite Assisted Visual Localization… / 2024 | ground localization using satellite | cross-view semantic matching | satellite as map | localization | 1 | ICRA |
| P46 | MPM-GS / 2025 | sparse reconstruction | virtual views + multimodal regularization | priors fill missing views | generic scenes | 1 | Crossref |
| P47 | Dynamic Scene Representation… / 2026 | survey | NeRF/3DGS review | organize dynamic rendering | generic | 0 | Crossref |
| P48 | RETRACTED CrossViewDiff chapter / 2024 | satellite-to-ground synthesis | cross-view diffusion | overhead conditioning | urban cross-view | 1 | Crossref; retracted |
| P49 | NeuralFloors++ / 2024 | consistent street generation from BEV maps | BEV semantic-conditioned generation | top-down layout stabilizes street sequence | driving/urban | 2 | IROS |
| P50 | NavBEV / 2025 | UAV navigation | self-supervised 3D BEV | geometry aids navigation | UAV | 1 | Semantic Scholar |
| P51 | Unifying UAV Cross-View Geolocalization via 3D Geometry / 2026 | cross-view localization | 3D geometric perception | shared geometry bridges views | UAV | 1 | Semantic Scholar |
| P52 | Sat2Scene / 2024 | arbitrary-view urban 3D scene generation from satellite | satellite-inferred geometry + 3D point diffusion + feed-forward neural renderer | explicit scene representation ensures view consistency | urban cross-view | 3 | CVPR/CVF |
| P53 | Neural Groundplans / 2022 | persistent scene representation from one image | neural groundplan decoded into views | persistent top-down latent supports queries | indoor scenes | 2 | ICLR |
| P54 | Sat2Density / 2023 | satellite-to-ground NVS and geometry | satellite encoder predicts explicit density volume rendered to depth/RGB | geometry is the key transferable satellite information | urban cross-view | 3 | ICCV/CVF |
| P55 | Enhancing 3DGS with Semantic and Geometric Priors… / 2026 | city reconstruction | semantic/geometric 3DGS priors | priors aid city modeling | urban | 1 | Semantic Scholar |
| P56 | Satellite True Orthophoto Generation without Elevation… / 2024 | satellite orthorectification | NeRF | infer elevation implicitly | remote sensing | 1 | Semantic Scholar |
| P57 | Epipolar Focus Spectrum / 2022 | dense-view light-field reconstruction | focus spectrum | encode dense light field | light fields | 1 | arXiv |
| P58 | X-LRM / 2025 | sparse-view CT | reconstruction model | rapid CT recovery | medical | 0 | arXiv |
| P59 | DGTR / 2024 | sparse vast-scene reconstruction | distributed Gaussian reconstruction | scale reconstruction | generic outdoor | 1 | arXiv |
| P60 | ReconX / 2024 | sparse-view reconstruction | video diffusion prior → 3D reconstruction | generative prior fills unseen content | generic scenes | 1 | arXiv |
| P61 | UniForward / 2025 | sparse-view scene+semantic field reconstruction | feed-forward 3DGS | one representation supports RGB+semantic outputs | generic scenes | 2 | arXiv |
| P62 | Omni-Scene / 2024/25 | ego-centric sparse-view reconstruction | omni Gaussian representation | model surround geometry | driving | 1 | CVPR |
| P63 | Sp2360 / 2024 | sparse 360 reconstruction | cascaded 2D diffusion priors | generative priors repair sparsity | generic 360 | 1 | arXiv |
| P64 | Energy-Efficient Sparse-View GS Review / 2026 | survey | review | sustainability | generic | 0 | Crossref |
| P65 | SparseDet / — | multi-view 3D detection | sparse scene representation | efficiency | driving detection | 1 | Crossref |
| P66 | Vvbpnet / — | sparse-view CBCT | backprojection CNN | direct medical reconstruction | medical | 0 | Crossref |
| P67 | Sparse-View Underwater 3DGS / 2025 | sparse underwater reconstruction | optimized 3DGS | domain-specific regularization | underwater | 1 | Crossref |
| P68 | PE-INeR / 2024 | sparse-view CBCT | prior-embedded INR | prior repairs missing tomography | medical | 0 | Crossref |
| P69 | Training-Free Instance-Aware Reconstruction… / 2025 | sparse reconstruction/NVS | diffusion-based generation | generative completion | generic objects/scenes | 1 | SIGGRAPH Asia |
| P70 | Optimizing View Angles for Sparse Reconstruction… / — | acquisition design | elastic registration | choose informative views | tomography | 0 | Crossref |
| P71 | Frequency-Regularized Neural Representation for Sparse CT / 2024 | sparse tomography | frequency regularization | suppress artifacts | medical | 0 | Crossref |
| P72 | LRR-CED / 2022 | sparse CBCT | reconstruction-aware encoder-decoder | low-resolution prior | medical | 0 | Crossref |
| P73 | Rethinking Image-to-3D with Sparse Queries / 2026 | image-to-3D | sparse query architecture | control input-view bias | generic 3D | 1 | Semantic Scholar |
| P74 | 4RC / 2026 | 4D reconstruction | conditional queries | query space-time flexibly | dynamic scenes | 1 | Semantic Scholar |
| P75 | SemanticSplat / 2025 | RGB+language scene understanding | language-aware Gaussian field | one field supports semantics | generic scenes | 1 | Semantic Scholar |
| P76 | Bridging 3DGS and Semantic Occupancy… / 2026 | geometry+semantic reconstruction | Gaussian/occupancy bridge | shared representation for multiple queries | generic scenes | 1 | Semantic Scholar |
| P77 | ID-NeRF / 2024 | generalizable NVS | indirect diffusion guidance | improve sparse NVS | generic scenes | 1 | Semantic Scholar |
| P78 | F-RNG / 2026 | relightable NVS | feed-forward neural Gaussians | factor lighting | generic scenes | 1 | Semantic Scholar |
| A01 | Geo2 / 2026 | satellite-ground localization and bidirectional synthesis | VGGT geometry features → shared geometry-aware latent; flow-matching image decoder | geometry-aware cross-view latent bridges tasks/views | geospatial cross-view | 3 | CVPR/CVF |
| A02 | AerialMegaDepth / 2025 | aerial-ground geometry and NVS | hybrid co-registered dataset + fine-tuned DUSt3R/MASt3R/ZeroNVS | aerial image can act as overhead map stitching sparse ground views | aerial-ground outdoor | 2 | arXiv:2504.13157 |
| A03 | Satellite to GroundScape / 2025 | consistent multi-view ground generation from satellite | fixed latent diffusion + satellite/layout and temporal conditioning | shared satellite prior improves sequence consistency | urban cross-view | 2 | arXiv:2504.15786 |
| A04 | LVSM / 2024 | sparse-view feed-forward NVS | encoder maps images to fixed latent scene tokens; decoder queries target rays | learned scene latent can replace explicit 3D representation | generic objects/scenes | 2 | arXiv:2410.17242 |
| A05 | SparseFusion / 2022 | sparse-view 3D reconstruction | distill view-conditioned diffusion into 3D scene representation | generative prior plus 3D consistency | object-centric scenes | 1 | arXiv:2212.00792 |
| M01 | pixelNeRF / 2021 | one/few-view NVS | image-conditioned NeRF | generalizable radiance field from sparse input | generic objects/scenes | 1 | model-recall; title verified by LVSM refs |
| M02 | IBRNet / 2021 | multi-view NVS | ray transformer over source features | aggregate evidence at render time | generic scenes | 1 | model-recall |
| M03 | MVSNeRF / 2021 | few-view NVS | MVS cost volume → NeRF | geometry-aware generalization | generic scenes | 1 | model-recall |

## Triage result

- Score 3: P24, P33, P52, P54, A01.
- Score 2: P02, P11, P13–P16, P21, P23, P32, P49, P53, P61, A02–A04.
- All remaining papers match at most one axis or are false positives.
