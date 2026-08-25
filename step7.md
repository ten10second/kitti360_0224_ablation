# Step 7 — Delta, C2 Necessity, and Falsification Gates

Timestamp: 2026-08-24T19:39:00+08:00

## Delta

> Unlike Cross-View Splatter, which jointly maps satellite and ground features directly to Gaussian outputs under task losses, the proposed work first fixes a metric, dense-ground-defined BEV latent and its pose-query RGB/depth decoder, then learns satellite-plus-sparse-ground amortized recovery into that exact space, with the intended measurable benefit of improved frozen-decoded geometry in unobserved cells while remaining stable across source subsets.

This delta is **contingent**, not yet an established contribution. If frozen decoded geometry, aligned-satellite controls, and held-out source-subset recovery do not pass, the sentence collapses to an architectural variant of Cross-View Splatter/Sat2Density.

## Does the broad idea stand?

- **No, not as a broad claim.** “Satellite is a global geometric prior for sparse ground reconstruction/NVS,” “VGGT+DPT for satellite-ground geometry,” “shared satellite-ground geometry-aware latent,” and “satellite-conditioned queryable urban 3D scene” are already covered by Cross-View Splatter, Geo2, Sat2Density, and Sat2Scene.
- **Potentially yes, as a narrower representation claim.** The remaining defensible claim is that the *semantics of the latent are defined asymmetrically by dense ground evidence and a frozen world decoder*, and other modalities are evaluated as recovery mechanisms into that fixed space rather than jointly defining a task-specific representation.
- **Current empirical status:** not yet established. Existing results show a useful dense-geometry Stage-A gain, but aligned satellite content has not yet decisively beaten the fixed-XY control, and the earlier C2 result was reversed. The small positive B-only signal is directional evidence, not a passed headline claim.

## Is current C2 necessary?

### Not necessary in its strict form

The statement `Z_A ≈ Z_B` over every latent channel is too strong and not the correct definition of “same latent space.” Sparse observations leave real appearance ambiguity, and latent coordinates can differ while a frozen decoder produces the same geometry. Full-tensor A/B L1 should therefore not be a headline acceptance gate.

### A revised source-subset test is necessary for the representation paper

If the paper claims only better NVS, C2 can be omitted—but that broad territory is already occupied by Cross-View Splatter. To defend the narrower *recovery into one fixed world space* claim, at least one disjoint-source stability test is needed:

1. Build two non-overlapping source subsets A and B from the same tile.
2. Infer each subset independently, including independent VGGT forwards.
3. Anchor both outputs to the same dense-ground reference `Z*`.
4. Make frozen decoded geometry the primary metric:
   - `D_geom(G(Z_A), G(Z*))`;
   - `D_geom(G(Z_B), G(Z*))`;
   - `D_geom(G(Z_A), G(Z_B))` on shared/B-only/A-only cells.
5. Use raw latent L1/cosine only as a secondary diagnostic.
6. Include different-location pairs to rule out a constant/collapsed latent.

This changes C2 from “different inputs must produce identical tensors” into “different inputs recover compatible geometry in the same frozen decoder space.”

## DPT decoder requirement

A DPT head helps only if it is part of the Stage-A world interface:

- train `RGB/depth = D_frozen(Z, query_pose)` from dense ground;
- freeze it before Stage B;
- prohibit direct satellite/source features from bypassing `Z` into the decoder;
- evaluate unseen poses and LiDAR depth/height/occupancy.

If the DPT head is trained jointly with Stage B, consumes satellite features directly, or predicts only a satellite height map, it does not establish latent recovery and overlaps heavily with Cross-View Splatter/Sat2Density.

### Current-code audit

`scripts/dpt_unified_bev_readout.py` currently trains four independent heads from scratch for `star`, `gnd`, `xy`, and `sat` (`heads = {k: DPThead() ...}`). That experiment establishes branch-wise geometry *decodability*, but it cannot establish a shared latent space because each head can adapt to a different latent distribution.

The representation claim requires exactly one geometry head:

1. Train one DPT geometry decoder only on dense-ground `Z*`.
2. Freeze it permanently.
3. Apply the identical weights, with no adaptation, to `Z*`, sparse ground, XY control, aligned satellite, and misaligned/random satellite latents.
4. Attribute improvements only when aligned satellite improves this frozen head's held-out geometry metrics.

This single-head protocol is more central to the claim than strict raw-latent A/B equality.

## Minimal value/falsification sequence

1. **Geometry backend probe, no training:** Compare sparse LiDAR lift, Metric3D lift, and independently inferred motion-scaled VGGT on the same 8–16 tiles. Gate on LiDAR depth and cross-subset decoded/cell geometry, not coverage alone.
2. **Stage-A probe:** Retrain with VGGT geometry. Require dense-ground `Z*` to be a real upper bound over sparse ground in both held-out RGB and depth. Freeze encoder/decoder after this point.
3. **Stage-B causal probe:** Train ground-only, fixed-XY, aligned satellite, shifted/rotated satellite, and random-tile satellite under identical protocols.
4. **Primary success condition:** On geographically held-out tiles, aligned satellite must beat ground-only, fixed-XY, and misaligned/random satellite in paired frozen-depth/geometry metrics, especially B-only/unobserved cells; RGB-only gains are insufficient.
5. **Representation property:** Revised source-subset recovery and source-count trends must hold at frozen decoded geometry outputs. Exact full-latent equality is not required.
6. **Stop condition:** If aligned satellite cannot beat fixed XY and misaligned/random controls after a clean VGGT Stage A, stop claiming cross-modal latent recovery; report it as a strong ground geometry encoder or a negative finding instead.

## Final verdict

**Level 2 — High Overlap.** The project can still stand, but only around the dense-ground-defined frozen world interface and causal recovery evidence. C2 is strategically necessary in revised decoded-geometry form, not as strict raw-latent equality. VGGT is a needed measurement repair and fair closest-baseline component, not the novelty.
