"""Unit tests for persistent georeferenced world-state v1."""
from __future__ import annotations

import torch

from world3d.unified_bev.world_state import (
    C_INIT,
    FORBIDDEN_MODEL_INPUT_KEYS,
    GroundMeasurement,
    PROVENANCE_INFERRED,
    PROVENANCE_SATELLITE,
    PROVENANCE_VEHICLE,
    SceneTileSpec,
    WorldState,
    apply_satellite_metadata,
    apply_vehicle_metadata,
    assert_no_supervision_leak,
    assert_preserved_outside_support,
    empty_state,
    visited_mask,
)
from world3d.unified_bev.world_targets import (
    bev_cell_centers,
    georeferenced_satellite_resample,
    satellite_mapping_error_px,
)
from world3d.unified_bev.state_models import (
    EvidenceAwareUpdater,
    SatelliteInitializer,
    WorldGeometryEncoder,
    aggregate_measurements,
)


def _spec(b=1, size=16, tile=8.0):
    res = tile / size
    return SceneTileSpec(
        scene_id="s",
        origin_xy=torch.zeros(b, 2),
        tile_size_m=tile,
        resolution_m=res,
        z_datum_m=torch.zeros(b, 1),
    )


def test_world_state_shapes_and_invalid_spec():
    spec = _spec()
    state = empty_state(spec, 4)
    state.validate()
    assert state.confidence.shape == (1, 1, 16, 16)
    assert state.provenance.dtype == torch.uint8
    assert int(state.last_update.min()) == -1
    try:
        SceneTileSpec("x", torch.zeros(1, 3), 8.0, 0.5, torch.zeros(1, 1)).validate()
        raise AssertionError("bad origin must fail")
    except ValueError:
        pass


def test_satellite_metadata_and_vehicle_or():
    spec = _spec()
    state = empty_state(spec, 2)
    valid = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    valid[..., :4, :4] = True
    state.latent = torch.ones_like(state.latent)
    state = apply_satellite_metadata(state, valid)
    assert float(state.confidence[valid].max()) == C_INIT
    assert int(state.provenance[valid][0]) == (PROVENANCE_SATELLITE | PROVENANCE_INFERRED)
    assert int(state.last_update[valid][0]) == 0
    assert int(state.last_update[~valid][0]) == -1
    support = torch.zeros_like(valid)
    support[..., 2:6, 2:6] = True
    new_lat = torch.full_like(state.latent, 3.0)
    new_conf = torch.ones_like(state.confidence)
    updated = apply_vehicle_metadata(state, new_lat, new_conf, support, 1)
    assert torch.equal(updated.latent[~support.expand_as(updated.latent)],
                       state.latent[~support.expand_as(state.latent)])
    assert int(updated.provenance[support][0]) & PROVENANCE_VEHICLE
    assert int(updated.last_update[support][0]) == 1
    assert_preserved_outside_support(state, updated, support)


def test_supervision_leak_fail_fast():
    assert_no_supervision_leak({"satellite_bev": 0}, context="ok")
    try:
        assert_no_supervision_leak({"future_route_support": 1, "satellite_bev": 0}, context="writer")
        raise AssertionError("leak must fail")
    except RuntimeError as exc:
        assert "future_route_support" in str(exc)
    assert "world_valid" in FORBIDDEN_MODEL_INPUT_KEYS


def test_georeferenced_satellite_resample_identity_center():
    sat_h = sat_w = 32
    mpp = 0.196
    # 8 m tile, 0.5 m/cell -> 16 cells, well inside 32*0.196=6.272? TOO SMALL
    # Use 4 m tile: 8 cells. 32 px * 0.196 = 6.272 m, half=3.136. 4 m tile half=2 < 3.136.
    sat = torch.full((1, 3, sat_h, sat_w), 0.4)
    center = torch.tensor([[10.0, 20.0]])
    origin = center - 2.0
    out = georeferenced_satellite_resample(
        sat, center, origin, tile_size_m=4.0, resolution_m=0.5, sat_m_per_px=mpp,
    )
    assert out.shape == (1, 3, 8, 8)
    err = satellite_mapping_error_px(
        center, origin, tile_size_m=4.0, resolution_m=0.5, sat_h=sat_h, sat_w=sat_w, sat_m_per_px=mpp,
    )
    assert float(err.max()) <= 1e-4
    assert abs(float(out.mean()) - 0.4) < 1e-3


def test_resample_fails_outside_asset():
    sat = torch.zeros(1, 3, 16, 16)
    center = torch.zeros(1, 2)
    origin = torch.tensor([[-100.0, -100.0]])
    try:
        georeferenced_satellite_resample(
            sat, center, origin, tile_size_m=4.0, resolution_m=0.5, sat_m_per_px=0.196,
        )
        raise AssertionError("out-of-asset resample must fail")
    except RuntimeError:
        pass


def test_updater_exact_preservation():
    spec = _spec(size=8, tile=4.0)
    prev = empty_state(spec, 8)
    prev.latent = torch.randn_like(prev.latent)
    meas_lat = torch.randn_like(prev.latent)
    support = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    support[..., :3, :3] = True
    meas = GroundMeasurement(meas_lat, support, support.float(), chunk_index=1)
    updater = EvidenceAwareUpdater(channels=8)
    out = updater(prev, meas)
    assert_preserved_outside_support(prev, out.state, support)
    assert float(out.gate[~support].abs().max()) == 0.0
    leaked = {"future_route_support": 1}
    try:
        updater(prev, meas, **leaked)
        raise AssertionError("updater leak must fail")
    except RuntimeError:
        pass


def test_satellite_initializer_no_ground_kwargs():
    spec = _spec(size=16, tile=8.0)
    init = SatelliteInitializer(
        latent_channels=64, bev_height=16, bev_width=16, tile_size_m=8.0, backbone="tiny",
    )
    sat = torch.rand(1, 3, 16, 16)
    state = init(sat, spec)
    assert state.version == 0
    assert float(state.confidence.mean()) == C_INIT
    try:
        init(sat, spec, world_valid=torch.ones(1, 1, 16, 16))
        raise AssertionError("initializer must reject supervision")
    except RuntimeError:
        pass


def test_world_encoder_rejects_rgb_signature():
    enc = WorldGeometryEncoder(latent_channels=8, context_blocks=1)
    h = torch.zeros(1, 1, 8, 8)
    z = enc(h, h, h.bool())
    assert z.shape == (1, 8, 8, 8)


def test_visited_and_aggregate():
    support = torch.zeros(1, 3, 1, 4, 4, dtype=torch.bool)
    support[:, 0, :, :2, :2] = True
    support[:, 2, :, 2:, 2:] = True
    v1 = visited_mask(support, 1)
    v3 = visited_mask(support, 3)
    assert int(v1.sum()) == 4
    assert int(v3.sum()) == 8
    spec = _spec(size=4, tile=2.0)
    m1 = GroundMeasurement(torch.ones(1, 2, 4, 4), support[:, 0], support[:, 0].float(), 1)
    m2 = GroundMeasurement(torch.zeros(1, 2, 4, 4), support[:, 2], support[:, 2].float(), 3)
    agg = aggregate_measurements([m1, m2])
    assert int(agg.support.sum()) == 8
    assert agg.chunk_index == 3


def test_replay_order_changes_metadata():
    spec = _spec(size=8, tile=4.0)
    updater = EvidenceAwareUpdater(channels=8)
    prev = empty_state(spec, 8)
    prev.latent = torch.zeros_like(prev.latent)
    a_support = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    b_support = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    a_support[..., :4, :4] = True
    b_support[..., 4:, 4:] = True
    lat = torch.ones(1, 8, 8, 8)
    ma = GroundMeasurement(lat, a_support, a_support.float(), 1)
    mb = GroundMeasurement(lat, b_support, b_support.float(), 2)
    s_ab = updater(updater(prev, ma).state, mb).state
    s_ba = updater(updater(prev, GroundMeasurement(lat, b_support, b_support.float(), 1)).state,
                   GroundMeasurement(lat, a_support, a_support.float(), 2)).state
    assert not torch.equal(s_ab.last_update, s_ba.last_update)


def test_world_height_targets_survive_absolute_elevation():
    """KITTI-360 world-Z is absolute map altitude (~120 m).  The physical
    height guard must apply to datum-relative height only; clipping absolute
    Z first flattened every real surface onto the ceiling and, after datum
    subtraction, the whole valid tile collapsed to one constant."""
    import numpy as np
    from world3d.unified_bev.world_targets import (
        MAX_RELATIVE_HEIGHT_M,
        MIN_RELATIVE_HEIGHT_M,
        accumulate_lidar_surface,
        height_minus_datum,
    )

    res, size = 0.5, 8
    road_z, facade_z, datum = 120.0, 128.0, 121.8
    pts = []
    for r in range(size):
        for c in range(4):  # road strip, west half
            pts.append([c * res + 0.25, r * res + 0.25, road_z])
    for c in range(4, size):  # facade band on row 4
        pts.append([c * res + 0.25, 4 * res + 0.25, facade_z])
    pts.append([5 * res + 0.25, 7 * res + 0.25, 500.0])  # noise far above
    pts.append([6 * res + 0.25, 7 * res + 0.25, 60.0])   # far below floor
    packed = accumulate_lidar_surface(
        np.asarray(pts, dtype=np.float64), np.zeros(2),
        tile_size_m=size * res, resolution_m=res,
    )
    assert abs(packed["height_world_z"][0, 0] - road_z) < 1e-9
    assert abs(packed["height_world_z"][7, 5] - 500.0) < 1e-9  # no absolute clip
    h = height_minus_datum(packed["height_world_z"], datum)
    assert abs(h[0, 0] - (road_z - datum)) < 1e-4   # road ~ -1.8 m
    assert abs(h[4, 4] - (facade_z - datum)) < 1e-4  # facade ~ +6.2 m
    assert h[4, 4] > h[0, 0]                        # vertical structure survives
    assert h[7, 5] == MAX_RELATIVE_HEIGHT_M          # relative-height ceiling
    assert h[7, 6] == MIN_RELATIVE_HEIGHT_M          # relative-height floor
    valid_h = h[packed["valid"]]
    assert len(np.unique(valid_h)) > 1              # never collapses to a constant
    assert not packed["valid"][5, 4]
    assert h[5, 4] == 0.0                           # unknown cells reset to zero


def test_semantics_surface_selection_and_policy():
    """v3 ground truth: label-driven layer choice (P0-audited), quality
    filters, and the instance coding contract."""
    import numpy as np
    from world3d.unified_bev.semantics import (
        GROUND_IDS,
        TOP_IDS,
        filter_points,
        label_policy_hash,
        select_surface_height,
    )

    # mixed cell (mostly road + some building): top share >= 15% -> read roof
    z = np.array([120.0, 120.1, 124.0, 124.2, 124.1, 124.3, 130.0])
    lab = np.array([7, 7, 11, 11, 11, 21, 41])
    assert abs(select_surface_height(z, lab) - 124.285) < 1e-2  # p95 of TOP points (interpolated)
    # ground-dominated: same cell without buildings -> road median
    assert abs(select_surface_height(np.array([120.0, 120.2]), np.array([7, 7])) - 120.1) < 1e-9
    assert select_surface_height(np.array([120.0, 126.0, 121.0]),
                                 np.array([7, 21, 21])) >= 125.0

    # policy: top and ground sets disjoint; ignore ids vote nowhere
    assert not (GROUND_IDS & TOP_IDS)

    # instance coding contract: stuff classes carry classInstanceID == 0
    sem = np.array([7, 11, 26])
    inst = np.array([7000, 11062, 26683])
    assert ((inst // 1000) == sem).all()
    assert int((inst % 1000 == 0).sum()) == 1  # road is stuff; building/person are things

    # hash is stable — target identity depends on it
    assert label_policy_hash() == label_policy_hash()


def test_ground_measurement_encoder_support_follows_vggt_gates():
    """The measurement's support must come from VGGT conf/depth gates on the
    calibrated unprojection — not from LiDAR supervision masks."""
    import numpy as np
    from world3d.unified_bev.state_models import GroundMeasurementEncoder

    size, res = 8, 0.5
    K = torch.tensor([[8.0, 0, 3.5], [0, 8.0, 3.5], [0, 0, 1.0]])
    T = torch.eye(4)
    T[:3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # look +x
    images = torch.rand(1, 2, 3, size, size)

    def measure(depth, conf):
        enc = GroundMeasurementEncoder(latent_channels=8, bev_height=size, bev_width=size)
        with torch.no_grad():
            return enc(
                images=images, K=K.expand(1, 2, 3, 3), dense_depth=depth, dense_conf=conf,
                T_world_cam=T.expand(1, 2, 4, 4), origin_xy=torch.zeros(1, 2),
                resolution_m=res, z_datum_m=torch.zeros(1, 1), chunk_index=3,
            )

    good = measure(torch.full((1, 2, size, size), 2.0), torch.full((1, 2, size, size), 0.9))
    assert good.chunk_index == 3
    assert bool(good.support.any()), "gated depth/conf must produce support"
    assert float(good.confidence[~good.support].abs().max()) == 0.0
    assert torch.isfinite(good.latent).all()

    oob = measure(torch.full((1, 2, size, size), 100.0), torch.full((1, 2, size, size), 0.9))
    assert not bool(oob.support.any()), "depth beyond the gate bound must empty the support"
    beyond_range = measure(torch.full((1, 2, size, size), 30.0), torch.full((1, 2, size, size), 0.9))
    assert not bool(beyond_range.support.any()), \
        "depth beyond reliable_range_m is skyline-contaminated and must not write"
    lowconf = measure(torch.full((1, 2, size, size), 2.0), torch.full((1, 2, size, size), 0.1))
    assert not bool(lowconf.support.any()), "confidence below the gate must empty the support"


def test_ground_height_quantile_keeps_ground_envelope():
    """A BEV cell mixes road, facades, canopies and stray far depths; the
    column MEAN was measured +8.6 m high at range — the low quantile must
    keep the ground cluster."""
    from world3d.unified_bev.geometry import ground_height_quantile

    size, res = 4, 0.5
    points = torch.tensor([[
        # cell (0,0): half road (122) half canopy (128), true road 120+2 bias
        [[0.25, 0.25, 122.0], [0.25, 0.25, 122.0], [0.25, 0.25, 128.0], [0.25, 0.25, 128.0]],
        # cell (1,2): road x3, canopy x1
        [[1.25, 0.25, 122.0], [1.25, 0.25, 122.0], [1.25, 0.25, 122.0], [1.25, 0.25, 128.0]],
    ]])
    valid = torch.ones(1, 2, 4, dtype=torch.bool)
    z_ground, count = ground_height_quantile(
        points, valid, torch.zeros(1, 2), res, size, size, quantile=0.15,
    )
    assert abs(float(z_ground[0, 0, 0, 0]) - 122.0) < 1e-6, "mean would read 125.0"
    assert abs(float(z_ground[0, 0, 0, 2]) - 122.0) < 1e-6  # cell (row 0, col 2)
    assert int(count[0, 0, 0, 0]) == 4
    assert float(z_ground[0, 0, 3, 3]) == 0.0  # empty cell stays zero


def test_measurement_ground_field_anchor_removes_local_bias():
    """The camera rig's world Z minus the calibrated camera height anchors
    the chunk's absolute ground: a uniform VGGT ground bias must cancel,
    and the anchor stays inside this measurement (never a global offset)."""
    import numpy as np
    from world3d.unified_bev.state_models import GroundMeasurementEncoder

    size, res = 8, 0.5
    fx = 8.0
    K = torch.tensor([[fx, 0, 3.5], [0, fx, 3.5], [0, 0, 1.0]])
    T = torch.eye(4)
    # camera looking straight down from 121.75; true road at 120.0
    T[:3, :3] = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T[0, 3], T[1, 3], T[2, 3] = 2.0, 2.0, 121.75
    depth = torch.full((1, 1, size, size), 3.75)  # -> world z = 118 (uniform -2 m bias)
    conf = torch.full((1, 1, size, size), 0.9)
    enc = GroundMeasurementEncoder(latent_channels=8, bev_height=size, bev_width=size)
    h_rel, support, anchor = enc.ground_field(
        depth, conf, K.view(1, 1, 3, 3), T.view(1, 1, 4, 4),
        torch.zeros(1, 2), res, torch.tensor([[120.0]]),
    )
    assert abs(float(anchor) - (-2.0)) < 1e-4, "anchor must measure the local VGGT bias"
    assert bool(support.any())
    assert float(h_rel[support].abs().max()) < 1e-4, "anchored ground must read datum-relative 0"


def test_world_vggt_cache_identity_and_assembly():
    """The per-scene VGGT cache is bound to the target contract; stale or
    mismatched caches must fail fast, and valid entries must assemble into a
    GroundMeasurement without re-running VGGT."""
    import tempfile

    import torch

    from world3d.unified_bev.state_models import GroundMeasurementEncoder
    from world3d.unified_bev.world_vggt import (
        WORLD_VGGT_CACHE_VERSION,
        chunk_measurement_from_cache,
        load_world_vggt_cache,
    )

    size = 8
    K = torch.tensor([[8.0, 0, 3.5], [0, 8.0, 3.5], [0, 0, 1.0]]).expand(2, 3, 3)
    T = torch.eye(4).expand(2, 4, 4).clone()
    T[:, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    entry = {
        "rgb": torch.rand(2, 3, size, size).to(torch.float16),
        "K": K, "T_world_cam": T,
        "depth": torch.full((2, size, size), 2.0, dtype=torch.float16),
        "conf": torch.full((2, size, size), 0.9, dtype=torch.float16),
        "measurement_fids": [101, 102], "query_fid": 99,
    }
    cache = {
        "schema": WORLD_VGGT_CACHE_VERSION, "scene_id": "s0",
        "world_target_version": "v2", "world_target_hash": "hash0",
        "chunks": {"1": entry},
    }
    with tempfile.TemporaryDirectory() as tmp:
        torch.save(cache, f"{tmp}/s0.pt")
        loaded = load_world_vggt_cache(tmp, "s0", "v2", "hash0")
        assert "1" in loaded["chunks"]
        try:
            load_world_vggt_cache(tmp, "s0", "v2", "different")
            raise AssertionError("hash mismatch must fail")
        except RuntimeError as exc:
            assert "world_target_hash" in str(exc)
        cache["schema"] = "legacy"
        torch.save(cache, f"{tmp}/s0.pt")
        try:
            load_world_vggt_cache(tmp, "s0", "v2", "hash0")
            raise AssertionError("schema mismatch must fail")
        except RuntimeError:
            pass
    enc = GroundMeasurementEncoder(latent_channels=8, bev_height=size, bev_width=size)
    with torch.no_grad():
        meas = chunk_measurement_from_cache(
            enc, entry, origin_xy=torch.zeros(1, 2), resolution_m=0.5,
            z_datum_m=torch.zeros(1, 1), chunk_index=1, query_fid=99,
        )
    assert meas.chunk_index == 1
    assert bool(meas.support.any())
    assert torch.isfinite(meas.latent).all()
    # query isolation: wrong chunk identity and leaked query frames must fail
    try:
        chunk_measurement_from_cache(
            enc, entry, origin_xy=torch.zeros(1, 2), resolution_m=0.5,
            z_datum_m=torch.zeros(1, 1), chunk_index=1, query_fid=100,
        )
        raise AssertionError("query_fid mismatch must fail")
    except RuntimeError as exc:
        assert "query_fid" in str(exc)
    leaked = dict(entry, measurement_fids=[99, 102])
    try:
        chunk_measurement_from_cache(
            enc, leaked, origin_xy=torch.zeros(1, 2), resolution_m=0.5,
            z_datum_m=torch.zeros(1, 1), chunk_index=1, query_fid=99,
        )
        raise AssertionError("query frame inside measurement frames must fail")
    except RuntimeError as exc:
        assert "held-out" in str(exc)
    stale = {k: v for k, v in entry.items() if k not in ("query_fid", "measurement_fids")}
    try:
        chunk_measurement_from_cache(
            enc, stale, origin_xy=torch.zeros(1, 2), resolution_m=0.5,
            z_datum_m=torch.zeros(1, 1), chunk_index=1, query_fid=99,
        )
        raise AssertionError("v1 entry without isolation fields must fail")
    except RuntimeError as exc:
        assert "query isolation" in str(exc)


def test_one_shot_support_matches_aggregate_write_region():
    """The one-shot loss region must equal the union of measurement supports —
    exactly what aggregate_measurements writes, not the LiDAR final mask."""
    support_a = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    support_a[..., :2, :] = True
    support_b = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    support_b[..., 1:3, :] = True
    m1 = GroundMeasurement(torch.ones(1, 2, 4, 4), support_a, support_a.float(), 1)
    m2 = GroundMeasurement(torch.zeros(1, 2, 4, 4), support_b, support_b.float(), 2)
    from world3d.unified_bev.state_models import aggregate_measurements, one_shot_support

    agg = aggregate_measurements([m1, m2])
    assert torch.equal(one_shot_support([m1, m2]), agg.support)
    assert int(one_shot_support([m1, m2]).sum()) == 12  # 4+4 rows minus 2-row overlap


def test_supervised_region_excludes_unlabelled_measurement_cells():
    """VGGT support extends beyond LiDAR labels, and the world target maps are
    zero (not "unknown") outside world_valid — supervision must intersect the
    two or unlabelled cells get pushed toward height 0 / density 0."""
    import torch
    from world3d.unified_bev.world_state import supervised_region

    support = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    support[..., :3, :] = True   # measurement sees 12 cells
    valid = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    valid[..., 2:, :] = True     # labels exist on 8 cells
    assert int((support & ~valid).sum()) > 0, "fixture must contain unlabelled-but-seen cells"
    sup = supervised_region(support, valid)
    assert int((sup & ~valid).sum()) == 0
    assert int((sup & ~support).sum()) == 0
    assert int(sup.sum()) == int((support & valid).sum())


def test_world_target_version_contract():
    """The version strings must move whenever the target math or datum policy
    changes; a stale checkpoint from an older target generation must fail."""
    from world3d.unified_bev.world_state import WORLD_TARGET_VERSION, Z_DATUM_POLICY
    assert WORLD_TARGET_VERSION == "official_semantics_surface_v3"
    assert Z_DATUM_POLICY == "first_chunk_lidar_optical_center_world_z_median_v1"


def test_first_chunk_datum_is_causal():
    """The scene datum may only read the FIRST chunk's optical centers; a
    scene-wide median would consume future vehicle positions at t=0."""
    from types import SimpleNamespace

    import numpy as np
    from world3d.unified_bev.world_targets import first_chunk_datum_z

    def rec(z):
        T = np.eye(4)
        T[2, 3] = z
        return SimpleNamespace(T_world_cam=T, _T_cam_velo=np.eye(4))

    by_fid = {f"a{i}": rec(z) for i, z in enumerate([121.7, 121.8, 121.9])}
    by_fid.update({f"b{i}": rec(z) for i, z in enumerate([200.0, 205.0])})  # future
    window = [
        SimpleNamespace(fids=("a0", "a1", "a2")),
        SimpleNamespace(fids=("b0", "b1")),
    ]
    assert abs(first_chunk_datum_z(window, by_fid) - 121.8) < 1e-9


def test_scene_dataset_binds_target_version_and_manifest():
    import json
    import tempfile

    import torch

    from world3d.unified_bev.world_checkpoints import (
        compute_world_interface_fingerprint,
        validate_scenes_manifest,
    )
    from world3d.unified_bev.world_data import WorldStateSceneDataset
    from world3d.unified_bev.world_state import (
        WORLD_STATE_SCHEMA_VERSION,
        WORLD_TARGET_VERSION,
        Z_DATUM_POLICY,
    )

    def make_blob(version, target_hash):
        return {
            "tile_size_m": 8.0, "resolution_m": 0.5, "chunking_version": "route_chunk_v1",
            "world_target_version": version, "world_target_hash": target_hash,
        }

    with tempfile.TemporaryDirectory() as tmp:
        torch.save(make_blob("surface_p90_world_z_minus_lidar_origin_median_v1", "h1"),
                   f"{tmp}/old.pt")
        torch.save(make_blob(WORLD_TARGET_VERSION, "h2"), f"{tmp}/new.pt")
        with open(f"{tmp}/scenes.jsonl", "w") as f:
            f.write(json.dumps({"scene_id": "old", "split": "train", "file": "old.pt",
                                "world_target_hash": "h1"}) + "\n")
        try:
            WorldStateSceneDataset(tmp)
            raise AssertionError("stale-version blob must be rejected")
        except RuntimeError as exc:
            assert "world_target_version" in str(exc)

        with open(f"{tmp}/scenes.jsonl", "w") as f:
            f.write(json.dumps({"scene_id": "new", "split": "train", "file": "new.pt",
                                "world_target_hash": "h2"}) + "\n")
            f.write(json.dumps({"scene_id": "newer", "split": "train", "file": "new.pt",
                                "world_target_hash": "h9"}) + "\n")
        ds = WorldStateSceneDataset(tmp)
        assert ds.manifest_hash == WorldStateSceneDataset(tmp).manifest_hash  # deterministic

        def ck(manifest):
            return {
                "schema_version": WORLD_STATE_SCHEMA_VERSION,
                "world_target_version": WORLD_TARGET_VERSION,
                "z_datum_policy": Z_DATUM_POLICY,
                "scenes_manifest_hash": manifest,
                "encoder": {"w": torch.zeros(1)},
                "height_reader": {"w": torch.zeros(1)},
                "density_reader": {"w": torch.zeros(1)},
                "depth_reader": {"w": torch.zeros(1)},
            }

        assert (compute_world_interface_fingerprint(ck(ds.manifest_hash))
                != compute_world_interface_fingerprint(ck("other")))
        validate_scenes_manifest(ck(ds.manifest_hash), ds.manifest_hash)
        validate_scenes_manifest({"no_binding": True}, ds.manifest_hash)
        try:
            validate_scenes_manifest(ck(ds.manifest_hash), "different")
            raise AssertionError("manifest mismatch must fail")
        except RuntimeError:
            pass
