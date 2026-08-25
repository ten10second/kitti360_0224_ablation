from __future__ import annotations

from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch

from world3d.geo.sat_alignment import SatSpec
from world3d.unified_bev.checkpoints import (
    GEOMETRY_TARGET_VERSION,
    STAGE_A_SCHEMA_VERSION,
    STAGE_B_SCHEMA_VERSION,
    compute_stage_a_fingerprint,
    validate_stage_a_checkpoint,
    validate_stage_a_dataset,
    validate_stage_b_checkpoint,
)
from world3d.unified_bev.geometry import (
    bev_grid_from_world_xy,
    bilinear_splat,
    geometry_supervision_support,
    height_statistics,
    image_uv_to_grid,
    observation_partition,
    ray_distance_to_camera_z,
    relative_height_map,
    se3_inverse,
    target_pixels_supported_by_bev,
)
from world3d.unified_bev.losses import (
    high_frequency_masked_l1,
    low_frequency_l1,
    masked_smooth_l1,
)
from world3d.unified_bev.data import (
    FRONT_CROP_OVERLAP,
    FRONT_CROP_WIDTH,
    VIEW_CAMERA_IDS,
    VIEW_LAYOUT_VERSION,
    centered_two_crop_starts,
    scaled_crop_intrinsics,
)
from world3d.unified_bev.models import (
    CompletionOutput,
    ColumnFieldDecoder,
    GroundBEVEncoder,
    LatentCompletion,
    fixed_relative_xy_encoding,
    nadir_distance,
    satellite_bev_crop,
)
from world3d.unified_bev.readouts import BEVHeightDecoder, freeze_module
from scripts.eval_unified_bev_probe import (
    depth_metrics,
    per_item_ssim,
    perturb_satellite,
    road_frame_shift,
    road_headings,
)
from scripts.train_unified_bev_stage_b import parse_source_choices


def test_se3_inverse_round_trip():
    T = torch.eye(4)
    T[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = torch.tensor([3.0, -2.0, 1.0])
    points = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    ph = torch.cat([points, torch.ones(2, 1)], dim=-1)
    recovered = (se3_inverse(T) @ (T @ ph.T)).T
    assert torch.allclose(recovered, ph, atol=1e-6)


def test_bilinear_splat_weight_conservation():
    values = torch.ones(1, 1, 1, 2)
    xy = torch.tensor([[[[1.25, 1.75]]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    bev, count = bilinear_splat(
        values, xy, valid, origin_xy=torch.zeros(1, 2), resolution_m=1.0, height=4, width=4,
    )
    assert torch.isclose(count.sum(), torch.tensor(1.0))
    assert torch.allclose((bev * count).sum(), torch.tensor(2.0), atol=1e-6)


def test_image_uv_grid_matches_align_corners_false():
    uv = torch.tensor([[0.0, 0.0], [159.0, 95.0], [79.5, 47.5]])
    grid = image_uv_to_grid(uv, (160, 96))
    expected = torch.tensor([
        [-1.0 + 1.0 / 160.0, -1.0 + 1.0 / 96.0],
        [1.0 - 1.0 / 160.0, 1.0 - 1.0 / 96.0],
        [0.0, 0.0],
    ])
    assert torch.allclose(grid, expected, atol=1e-6)


def test_decoder_and_ground_shapes():
    B, N, P, H, W = 1, 2, 8, 32, 48
    images = torch.rand(B, N, 3, H, W)
    points = torch.rand(B, N, P, 3)
    points[..., 0] = points[..., 0] * 10 + 20
    points[..., 1] = points[..., 1] * 10 + 20
    uv = torch.rand(B, N, P, 2)
    uv[..., 0] *= W - 1; uv[..., 1] *= H - 1
    valid = torch.ones(B, N, P, dtype=torch.bool)
    enc = GroundBEVEncoder(latent_channels=8, bev_height=16, bev_width=16)
    z, mask = enc(images, points, uv, valid, torch.zeros(B, 2), 2.0)
    K = torch.tensor([[[30.0, 0.0, W / 2], [0.0, 30.0, H / 2], [0.0, 0.0, 1.0]]])
    T = torch.eye(4).unsqueeze(0)
    dec = ColumnFieldDecoder(latent_channels=8, hidden=16, samples=4)
    rgb, depth, opacity = dec.render(z, K, T, torch.zeros(B, 2), tile_size_m=32.0, image_size=(W, H), far_m=10.0)
    assert rgb.shape == (B, 3, H, W)
    assert depth.shape == opacity.shape == (B, H, W)
    assert torch.isfinite(rgb).all() and torch.isfinite(depth).all()
    rgb2, depth2, opacity2 = dec.render(
        z, K[:, None].expand(-1, 2, -1, -1), T[:, None].expand(-1, 2, -1, -1),
        torch.zeros(B, 2), tile_size_m=32.0, image_size=(W, H), far_m=10.0,
    )
    assert rgb2.shape == (B, 2, 3, H, W)
    assert depth2.shape == opacity2.shape == (B, 2, H, W)


def test_bev_grid_matches_splat_convention():
    """A point splatted at world XY must be read back by the decoder grid at
    the same world XY, and must not leak to the Y-mirrored location."""
    import torch.nn.functional as F

    H = W = 64
    origin = torch.zeros(1, 2)
    p_world = torch.tensor([[[[20.0, 40.0]]]])
    bev, _ = bilinear_splat(
        torch.ones(1, 1, 1, 1), p_world, torch.ones(1, 1, 1, dtype=torch.bool),
        origin_xy=origin, resolution_m=1.0, height=H, width=W,
    )

    def read_at(xy):
        grid = bev_grid_from_world_xy(xy, origin, float(H)).view(1, 1, 1, 2)
        return F.grid_sample(bev, grid, mode="bilinear", align_corners=False)

    mirrored = torch.tensor([[[[20.0, float(H) - 40.0]]]])
    assert float(read_at(p_world)) > 0.9
    assert float(read_at(mirrored)) < 0.1


def test_height_statistics_are_true_mean_and_variance():
    """Regression: bilinear_splat already returns weighted means, so the
    height channels must not be divided by the count map again (the old
    double division shrank h_mean by a per-cell factor of count)."""
    # Points at exact cell centers (pixel-center convention): heights 10, 20
    # in cell (row 0, col 0); height 6 in cell (row 1, col 2).
    points = torch.tensor([[[[0.5, 0.5, 10.0], [0.5, 0.5, 20.0], [2.5, 1.5, 6.0]]]])
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    h_mean, h_var = height_statistics(points, valid, torch.zeros(1, 2), 1.0, 4, 4)
    assert torch.isclose(h_mean[0, 0, 0, 0], torch.tensor(15.0), atol=1e-5)
    assert torch.isclose(h_var[0, 0, 0, 0], torch.tensor(25.0), atol=1e-4)
    assert torch.isclose(h_mean[0, 0, 1, 2], torch.tensor(6.0), atol=1e-5)
    assert torch.isclose(h_var[0, 0, 1, 2], torch.tensor(0.0), atol=1e-5)
    assert float(h_mean[0, 0, 3, 3]) == 0.0  # empty cell stays zero


def test_relative_height_is_translation_invariant_and_ignores_empty_cells():
    points = torch.tensor([[[
        [0.5, 0.5, 100.0], [1.5, 0.5, 101.0],
        [0.5, 1.5, 105.0], [1.5, 1.5, 110.0],
    ]]])
    valid = torch.ones(1, 1, 4, dtype=torch.bool)
    args = (valid, torch.zeros(1, 2), 1.0, 4, 4)
    relative, support, ground = relative_height_map(points, *args)
    shifted, shifted_support, shifted_ground = relative_height_map(
        points + points.new_tensor([0.0, 0.0, 100.0]), *args,
    )
    assert torch.equal(support, shifted_support)
    assert torch.allclose(relative, shifted, atol=1e-5)
    assert torch.allclose(shifted_ground - ground, torch.tensor(100.0), atol=1e-5)
    assert torch.equal(relative[~support], torch.zeros_like(relative[~support]))
    assert not torch.all(relative[support] == 30.0), "absolute KITTI altitude must not saturate the target"


def test_observation_partition_is_disjoint_and_geometry_bounded():
    sparse = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.bool)
    dense = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.bool)
    observed, fill = observation_partition(sparse, dense)
    assert torch.equal(observed, sparse)
    assert torch.equal(fill, torch.tensor([[[[0, 1], [0, 0]]]], dtype=torch.bool))
    assert not (observed & fill).any()
    assert not (fill & ~dense).any()


def test_geometry_supervision_requires_dense_lift_and_height_label():
    dense_lift = torch.tensor([[[[1, 1], [0, 1]]]], dtype=torch.bool)
    height_label = torch.tensor([[[[1, 0], [1, 1]]]], dtype=torch.bool)
    support = geometry_supervision_support(dense_lift, height_label)
    assert torch.equal(
        support, torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.bool),
    )
    sparse = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.bool)
    _, fill = observation_partition(sparse, support)
    assert torch.equal(fill, torch.tensor([[[[0, 0], [0, 1]]]], dtype=torch.bool))


def test_target_pixel_support_uses_backprojected_world_xy():
    depth = torch.ones(1, 2, 2)
    valid = torch.ones_like(depth, dtype=torch.bool)
    # Pixel centers (0.5,0.5)..(1.5,1.5) at z=1 backproject to the same
    # world XY with this unit-intrinsic convention.
    K = torch.eye(3).unsqueeze(0)
    T = torch.eye(4).unsqueeze(0)
    support = torch.zeros(1, 1, 2, 2)
    support[0, 0, 0, 0] = 1.0
    mask = target_pixels_supported_by_bev(
        depth, valid, K, T, torch.zeros(1, 2), 2.0, support,
    )
    assert mask.shape == (1, 1, 2, 2)
    assert mask[0, 0, 0, 0]
    assert int(mask.sum()) == 1
    multi = target_pixels_supported_by_bev(
        depth[:, None].expand(-1, 2, -1, -1),
        valid[:, None].expand(-1, 2, -1, -1),
        K[:, None].expand(-1, 2, -1, -1),
        T[:, None].expand(-1, 2, -1, -1),
        torch.zeros(1, 2), 2.0, support,
    )
    assert multi.shape == (1, 2, 1, 2, 2)
    assert int(multi.sum()) == 2


def test_ray_distance_is_converted_to_camera_z():
    distance = torch.tensor([10.0])
    direction = torch.tensor([[0.6, 0.0, 0.8]])
    assert torch.allclose(ray_distance_to_camera_z(distance, direction), torch.tensor([8.0]))


def test_depth_metrics_known_values():
    pred = torch.tensor([[[1.0, 2.0]]])
    target = torch.ones_like(pred)
    metric = depth_metrics(pred, target, torch.ones_like(pred, dtype=torch.bool))[0]
    assert abs(metric["absrel"] - 0.5) < 1e-6
    assert abs(metric["rmse"] - 2 ** -0.5) < 1e-6
    assert abs(metric["delta1"] - 0.5) < 1e-6


def test_satellite_metric_perturbations_have_expected_direction():
    sat = torch.zeros(1, 1, 9, 9)
    sat[0, 0, 4, 4] = 1.0
    east = perturb_satellite(sat, meters_per_pixel=1.0, shift_x_m=2.0)
    north = perturb_satellite(sat, meters_per_pixel=1.0, shift_y_m=2.0)
    assert torch.nonzero(east == east.max(), as_tuple=False)[0, -2:].tolist() == [4, 6]
    assert torch.nonzero(north == north.max(), as_tuple=False)[0, -2:].tolist() == [2, 4]

    sat.zero_()
    sat[0, 0, 4, 6] = 1.0
    ccw = perturb_satellite(sat, meters_per_pixel=1.0, rotate_deg=90.0)
    assert torch.nonzero(ccw == ccw.max(), as_tuple=False)[0, -2:].tolist() == [2, 4]


def test_random_sparse_source_choices_are_validated():
    assert parse_source_choices("1,2,4", fixed=2, dense=8) == (1, 2, 4)
    assert parse_source_choices(None, fixed=2, dense=8) == (2,)
    try:
        parse_source_choices("1,16", fixed=2, dense=8)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range source choice must fail")
    try:
        parse_source_choices("8", fixed=2, dense=8)
    except ValueError:
        pass
    else:
        raise AssertionError("dense identity source count must be evaluation-only")


def test_kitti360_satellite_spec():
    spec = SatSpec()
    assert spec.width == 512 and spec.height == 512
    assert spec.meters_per_pixel == 0.196
    assert spec.cx == spec.cy == 256.0


def _make_completion(mode: str, channels: int = 8, size: int = 12) -> LatentCompletion:
    torch.manual_seed(0)
    module = LatentCompletion(channels=channels, mode=mode, bev_height=size, bev_width=size)
    # The delta branch is zero-initialised on purpose; activate it so the
    # routing tests exercise a live correction path.
    with torch.no_grad():
        module.delta[-1].weight.normal_(std=0.05)
        module.delta[-1].bias.normal_(std=0.05)
    return module


def test_completion_identity_at_dense_sources():
    """alpha(Ns=N_dense)=0 must return the ground latent bitwise.  The old
    identity z_sat + mask*gate*(z_gnd-z_sat) forced z_hat=z_sat on every
    uncovered cell, so the dense-convergence gate failed by construction."""
    z_gnd = torch.randn(2, 8, 12, 12)
    z_sat = torch.randn(2, 8, 12, 12)
    coverage = (torch.rand(2, 1, 12, 12) > 0.6).float()
    for mode in ("residual", "coordinate_only"):
        module = _make_completion(mode)
        out = module(z_sat, z_gnd, coverage, n_sparse=8, dense_sources=8)
        assert isinstance(out, CompletionOutput)
        assert torch.equal(out.latent, z_gnd), mode
        assert torch.equal(out.correction, torch.zeros_like(z_gnd))
        assert torch.equal(out.ground_support, coverage)
    assert torch.equal(
        _make_completion("satellite_only")(z_sat, z_gnd, coverage, 8, 8).latent, z_sat
    )
    assert torch.equal(
        _make_completion("ground_only")(z_sat, z_gnd, coverage, 8, 8).latent, z_gnd
    )
    module = _make_completion("residual")
    for bad in (0, 9):
        try:
            module(z_sat, z_gnd, coverage, bad, 8)
        except ValueError:
            pass
        else:
            raise AssertionError(f"n_sparse={bad} must be rejected")


def test_completion_write_gate_range_and_prior_routing():
    z_gnd = torch.randn(1, 8, 12, 12)
    coverage = (torch.rand(1, 1, 12, 12) > 0.6).float()
    z_sat_a, z_sat_b = torch.randn(1, 8, 12, 12), torch.randn(1, 8, 12, 12)

    residual = _make_completion("residual")
    output_a = residual(z_sat_a, z_gnd, coverage, 2, 8)
    output_b = residual(z_sat_b, z_gnd, coverage, 2, 8)
    gate = output_a.write_gate
    assert gate.shape == (1, 1, 12, 12)
    assert 0.0 <= float(gate.min()) and float(gate.max()) <= 1.0
    assert not torch.allclose(gate, output_b.write_gate), \
        "the auditable write gate must be able to react to satellite content"
    assert torch.allclose(output_a.latent, z_gnd + output_a.correction)
    assert not torch.allclose(
        output_a.latent,
        output_b.latent,
    ), "residual mode must consume satellite content"

    coord = _make_completion("coordinate_only")
    first = coord(z_sat_a, z_gnd, coverage, 2, 8).latent
    assert torch.equal(first, coord(z_sat_b, z_gnd, coverage, 2, 8).latent), \
        "coordinate_only must be satellite-blind"


def test_coordinate_only_uses_fixed_metric_relative_xy():
    encoding = fixed_relative_xy_encoding(8, 12, 12, tile_size_m=24.0)
    assert encoding.shape == (1, 8, 12, 12)
    # Canonical BEV is south-up: x increases west->east across columns and y
    # increases south->north across rows, exactly like z_sat after its flip.
    assert float(encoding[0, 0, 6, 0]) < 0 < float(encoding[0, 0, 6, -1])
    assert float(encoding[0, 1, 0, 6]) < 0 < float(encoding[0, 1, -1, 6])

    torch.manual_seed(1)
    first = LatentCompletion(channels=8, mode="coordinate_only", bev_height=12, bev_width=12,
                             tile_size_m=24.0)
    torch.manual_seed(999)
    second = LatentCompletion(channels=8, mode="coordinate_only", bev_height=12, bev_width=12,
                              tile_size_m=24.0)
    assert torch.equal(first.coord_embed, second.coord_embed)
    assert "coord_embed" in dict(first.named_buffers())
    assert "coord_embed" not in dict(first.named_parameters())
    assert sum(p.numel() for p in first.parameters()) < 1_000_000

    z_gnd = torch.randn(1, 8, 12, 12)
    coverage = (torch.rand(1, 1, 12, 12) > 0.6).float()
    z_sat = torch.randn_like(z_gnd)
    with_xy = _make_completion("coordinate_only").eval()
    output_with_xy = with_xy(z_sat, z_gnd, coverage, 2, 8).latent
    with torch.no_grad():
        with_xy.coord_embed.zero_()
    assert not torch.equal(output_with_xy, with_xy(z_sat, z_gnd, coverage, 2, 8).latent), \
        "fixed XY must be routed through the shared delta path"


def test_completion_alpha_schedule_scales_correction():
    """alpha(2)=0.75 and alpha(4)=0.5 share the same conf/correction path, so
    the applied correction must scale by exactly the alpha ratio."""
    z_gnd = torch.randn(1, 8, 12, 12)
    z_sat = torch.randn(1, 8, 12, 12)
    coverage = (torch.rand(1, 1, 12, 12) > 0.6).float()
    module = _make_completion("residual")
    at2 = module(z_sat, z_gnd, coverage, 2, 8).correction
    at4 = module(z_sat, z_gnd, coverage, 4, 8).correction
    assert torch.allclose(at2, 1.5 * at4, atol=1e-6)


def test_frozen_bev_height_decoder_is_shared_readout_not_a_gradient_barrier():
    decoder = freeze_module(BEVHeightDecoder(latent_channels=8, width=16))
    latent = torch.randn(2, 8, 16, 16, requires_grad=True)
    prediction = decoder(latent)
    assert prediction.shape == (2, 1, 16, 16)
    prediction.square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_observation_aware_losses_handle_empty_masks():
    pred = torch.randn(1, 3, 16, 16, requires_grad=True)
    target = torch.randn_like(pred)
    empty = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    loss = masked_smooth_l1(pred, target, empty)
    assert float(loss) == 0.0 and loss.requires_grad
    loss.backward(retain_graph=True)
    assert pred.grad is not None
    assert torch.isfinite(low_frequency_l1(pred, target, scale=8))
    assert float(high_frequency_masked_l1(pred, target, empty, scale=8)) == 0.0
    pred_views = pred[:, None].expand(-1, 2, -1, -1, -1)
    target_views = target[:, None].expand_as(pred_views)
    empty_views = empty[:, None].expand(-1, 2, -1, -1, -1)
    assert torch.isfinite(low_frequency_l1(pred_views, target_views, scale=8))
    assert float(high_frequency_masked_l1(
        pred_views, target_views, empty_views, scale=8,
    )) == 0.0


def test_checkpoint_schema_binds_stage_b_to_exact_stage_a():
    stage_a = {
        "schema_version": STAGE_A_SCHEMA_VERSION,
        "ground": {"weight": torch.arange(4, dtype=torch.float32)},
        "decoder": {"weight": torch.ones(2)},
        "geometry_decoder": {"weight": torch.zeros(3)},
        "ground_config": {"family": "dense", "latent_channels": 64},
        "renderer_config": {"latent_channels": 64, "hidden": 16, "samples": 4},
        "geometry_decoder_config": {"latent_channels": 64, "width": 8},
        "geometry_target_version": GEOMETRY_TARGET_VERSION,
        "grid_config": {
            "bev_size": 128, "bev_resolution_m": 0.5,
            "tile_size_m": 64.0, "views_per_frame": 8, "target_views": 2,
            "target_view_layout_version": "front2_v1",
        },
    }
    stage_a["fingerprint"] = compute_stage_a_fingerprint(stage_a)
    fingerprint = validate_stage_a_checkpoint(stage_a)
    dataset = SimpleNamespace(
        bev_size=128, bev_resolution_m=0.5, tile_size_m=64.0,
        views_per_frame=8, target_views=2,
        target_view_layout_version="front2_v1",
    )
    validate_stage_a_dataset(stage_a, dataset, dense_geometry_attached=True)
    try:
        validate_stage_a_dataset(stage_a, dataset, dense_geometry_attached=False)
    except RuntimeError as error:
        assert "ground family" in str(error)
    else:
        raise AssertionError("dense Stage A must reject a sparse-ground data path")
    stage_b = {
        "schema_version": STAGE_B_SCHEMA_VERSION,
        "stage_a_fingerprint": fingerprint,
    }
    validate_stage_b_checkpoint(stage_b, fingerprint)
    try:
        validate_stage_b_checkpoint(stage_b, "different-stage-a")
    except RuntimeError as error:
        assert "different Stage-A" in str(error)
    else:
        raise AssertionError("Stage B must be bound to its exact Stage A")

    stage_a["ground"]["weight"] = stage_a["ground"]["weight"] + 1.0
    try:
        validate_stage_a_checkpoint(stage_a)
    except RuntimeError as error:
        assert "fingerprint mismatch" in str(error)
    else:
        raise AssertionError("mutated Stage-A weights must invalidate the checkpoint")


def test_road_frame_shift_decomposes_along_and_cross():
    east, north = road_frame_shift(np.array([1.0, 0.0]), road_m=3.0, cross_m=4.0)
    assert abs(east - 3.0) < 1e-9 and abs(north - 4.0) < 1e-9
    east, north = road_frame_shift(np.array([0.0, 1.0]), road_m=3.0, cross_m=4.0)
    assert abs(east + 4.0) < 1e-9 and abs(north - 3.0) < 1e-9


def test_road_headings_follow_trajectory_order():
    def rec(x: float, y: float) -> SimpleNamespace:
        pose = np.eye(4)
        pose[0, 3], pose[1, 3] = x, y
        return SimpleNamespace(T_world_imu=pose)

    class StraightDS:
        samples = [(rec(i * 10.0, 0.0), None) for i in range(4)]

    headings = road_headings(StraightDS())
    assert np.allclose(headings, np.tile([1.0, 0.0], (4, 1)))

    class DegenerateDS:
        samples = [(rec(5.0, 5.0), None) for _ in range(3)]

    assert np.allclose(road_headings(DegenerateDS()), np.tile([0.0, 1.0], (3, 1)))


def test_satellite_bev_crop_flips_north_to_south():
    """North-up source content must land in the south rows of the BEV raster,
    matching the splat/decoder convention."""
    sat = torch.zeros(1, 1, 8, 8)
    sat[0, 0, 1, :] = 1.0  # bright strip near the top (north) row
    out = satellite_bev_crop(sat, tile_size_m=8.0, sat_m_per_px=1.0, size=8)
    row_max = int(out[0, 0].argmax(dim=0).float().mean().round())
    assert row_max >= 5, f"north content must flip to south rows, got row {row_max}"


def test_nadir_render_shapes_and_range():
    dec = ColumnFieldDecoder(latent_channels=8, hidden=16, samples=4)
    z = torch.randn(2, 8, 16, 16)
    rgb, opacity = dec.render_nadir(
        z, torch.zeros(2, 2), tile_size_m=16.0, bev_size=16, z_top_m=16.0,
    )
    assert rgb.shape == (2, 3, 16, 16)
    assert opacity.shape == (2, 16, 16)
    assert torch.isfinite(rgb).all() and torch.isfinite(opacity).all()
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0


def test_nadir_distance_ignores_masked_out_cells():
    ref = torch.rand(1, 3, 8, 8)
    pred = ref.clone()
    pred[..., :2] += 0.5  # differs only in the far-left columns
    compare = torch.zeros(1, 1, 8, 8)
    compare[..., 5:] = 1.0  # compare only on the far right, with a gap so no
    # finite-difference pair spans a changed cell
    assert float(nadir_distance(pred, ref, compare)) == 0.0
    changed = torch.zeros(1, 1, 8, 8)
    changed[..., :2] = 1.0
    assert float(nadir_distance(pred, ref, changed)) > 0.0


def test_ssim_identity_and_range():
    torch.manual_seed(0)
    img = torch.rand(2, 3, 32, 48)
    values = per_item_ssim(img, img)
    assert all(abs(v - 1.0) < 1e-5 for v in values)
    other = torch.rand_like(img)
    # SSIM lives in [-1, 1]; uncorrelated noise pairs sit near zero and can
    # dip slightly negative, so only the upper bound is a hard invariant.
    assert all(-1.0 <= v <= 1.0 + 1e-4 for v in per_item_ssim(img, other))


def test_lpips_identity_is_zero():
    try:
        import lpips
    except ImportError:
        return  # perceptual metric optional in minimal environments
    from scripts.eval_unified_bev_probe import per_item_lpips
    net = lpips.LPIPS(net="alex")
    x = torch.rand(1, 3, 32, 48)
    assert abs(per_item_lpips(x, x, net)[0]) < 1e-4


def test_satellite_vit_output_contract():
    from world3d.unified_bev.models import SatelliteViTEncoder
    enc = SatelliteViTEncoder(latent_channels=16, bev_height=64, bev_width=64,
                              dim=64, depth=2, heads=4, patch=8, tile_size_m=64.0)
    sat = torch.rand(2, 3, 256, 256)
    z = enc(sat, tile_size_m=64.0, sat_m_per_px=0.196)
    assert z.shape == (2, 16, 64, 64)
    assert torch.isfinite(z).all()
    try:
        enc(sat, tile_size_m=32.0, sat_m_per_px=0.196)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched tile size must be rejected")


def test_satellite_vit_position_encoding_is_fixed():
    from world3d.unified_bev.models import SatelliteViTEncoder
    enc = SatelliteViTEncoder(latent_channels=16, bev_height=64, bev_width=64,
                              dim=64, depth=2, heads=4, patch=8, tile_size_m=64.0)
    param_names = {n for n, _ in enc.named_parameters()}
    assert not any(n.startswith("pos") for n in param_names), "position must never be learned"
    assert "pos" in dict(enc.named_buffers())
    other = SatelliteViTEncoder(latent_channels=16, bev_height=64, bev_width=64,
                                dim=64, depth=2, heads=4, patch=8, tile_size_m=64.0)
    assert torch.equal(enc.pos, other.pos)
    # distinct patches must carry distinct positions
    assert not torch.allclose(enc.pos[0], enc.pos[-1])


def test_satellite_vit_respects_orientation():
    from world3d.unified_bev.models import SatelliteViTEncoder
    torch.manual_seed(0)
    enc = SatelliteViTEncoder(latent_channels=16, bev_height=64, bev_width=64,
                              dim=64, depth=2, heads=4, patch=8, tile_size_m=64.0)
    sat = torch.rand(1, 3, 256, 256)
    z1 = enc(sat, 64.0, 0.196)
    z2 = enc(torch.flip(sat, dims=[-2]), 64.0, 0.196)  # north<->south flip of content
    assert not torch.allclose(z1, z2), "flipping satellite content must change the latent"


def test_heightmap_prior_contract_and_cross_attention():
    from world3d.unified_bev.models import HeightMapSatellitePrior
    torch.manual_seed(0)
    enc = HeightMapSatellitePrior(latent_channels=16, bev_height=64, bev_width=64,
                                  dim=64, depth=2, heads=4, patch=8, tile_size_m=64.0)
    sat = torch.rand(1, 3, 256, 256)
    z_gnd = torch.randn(1, 16, 64, 64)
    output = enc(sat, z_gnd, 64.0, 0.196)
    assert len(output) == 2, "the satellite branch must not emit fake zero uncertainty"
    prior, h_pred = output
    assert prior.shape == (1, 16, 64, 64)
    assert h_pred.shape == (1, 1, 64, 64)
    assert torch.isfinite(h_pred).all()
    # cross-attention is live: changing the street latent must change the prior
    z2 = z_gnd + torch.randn_like(z_gnd) * 0.1
    assert not torch.allclose(enc(sat, z_gnd, 64.0, 0.196)[0], enc(sat, z2, 64.0, 0.196)[0])
    # satellite content is consumed
    assert not torch.allclose(enc(sat, z_gnd, 64.0, 0.196)[0], enc(torch.rand_like(sat), z_gnd, 64.0, 0.196)[0])
    try:
        enc(sat, z_gnd, 32.0, 0.196)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched tile size must be rejected")


def test_dense_lift_unprojection_roundtrip():
    """v2 lift geometry: project known world points through a synthetic pinhole,
    unproject the z-depth map, recover the same points."""
    from world3d.unified_bev.models import unproject_dense
    torch.manual_seed(0)
    fx, cx, cy = 200.0, 80.0, 48.0
    K = torch.tensor([[[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]]])
    T = torch.eye(4).view(1, 1, 4, 4)
    T[0, 0, 2, 3] = 1.5  # camera 1.5m above ground, looking +z
    pts = torch.tensor([
        [0.0, 0.0, 10.0], [-3.0, 0.5, 15.0], [4.0, -1.0, 20.0], [1.0, 2.0, 30.0],
    ])
    cam = pts.clone(); cam[:, 2] -= 1.5  # world->cam: camera sits 1.5m above (z offset)
    u = cam[:, 0] / cam[:, 2] * fx + cx
    v = cam[:, 1] / cam[:, 2] * fx + cy
    H, W = 96, 160
    depth = torch.zeros(1, 1, H, W)
    ui = u.round().long().clamp(0, W - 1); vi = v.round().long().clamp(0, H - 1)
    depth[0, 0, vi, ui] = cam[:, 2]
    rec = unproject_dense(depth, K.expand(1, 1, 3, 3), T)[0, 0]
    for k in range(pts.shape[0]):
        got = rec[vi[k], ui[k]]
        # sub-pixel quantization: nearest-pixel rounding at fx=200 is ~4cm at 15m
        assert torch.allclose(got, pts[k], atol=0.1), (k, got, pts[k])


def test_dense_encoder_shapes_and_coverage():
    from world3d.unified_bev.models import GroundDenseBEVEncoder
    torch.manual_seed(0)
    enc = GroundDenseBEVEncoder(latent_channels=8, bev_height=16, bev_width=16)
    B, N, H, W = 1, 2, 12, 20
    images = torch.rand(B, N, 3, H, W)
    fx = 15.0
    K = torch.tensor([[[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]]]).expand(B, N, 3, 3)
    # Camera looks along world +y (horizontal), 1.5m above ground: pixel rays
    # spread over the BEV (x, y) plane as depth varies.
    T = torch.zeros(B, N, 4, 4)
    T[..., 0, 0] = 1.0
    T[..., 1, 2] = 1.0   # cam z -> world y
    T[..., 2, 1] = -1.0  # cam y -> world -z
    T[..., 3, 3] = 1.0
    T[..., 1, 3] = 1.5
    rows = torch.linspace(2.0, 32.0, H).view(1, 1, H, 1).expand(B, N, H, W)
    depth = rows.clone()                     # nearer rows (bottom) to far rows
    conf = torch.ones(B, N, H, W)
    origin = torch.tensor([[-16.0, 2.0]])  # tile y in [2, 34]: matches the 2..33.5m point span
    z, cov = enc(images, K, depth, conf, T, origin, 2.0)
    assert z.shape == (B, 8, 16, 16)
    assert float(cov.mean()) > 0.3, "a 10m wall ahead should cover many cells"
    # low-confidence pixels must be excluded
    conf2 = torch.zeros_like(conf)
    _, cov2 = enc(images, K, depth, conf2, T, origin, 2.0)
    assert float(cov2.mean()) == 0.0


def test_metric3d_two_crops_are_centered_on_optical_axis():
    starts = centered_two_crop_starts(
        1408, FRONT_CROP_WIDTH, FRONT_CROP_OVERLAP, 682.049453,
    )
    assert starts == [158, 646]
    union_center = (starts[0] + starts[1] + 560) / 2.0
    assert abs(union_center - 682.049453) < 0.5


def test_front2_intrinsics_share_cam0_center_and_shift_principal_point():
    K = np.array([
        [552.554261, 0.0, 682.049453],
        [0.0, 552.554261, 238.769549],
        [0.0, 0.0, 1.0],
    ])
    starts = centered_two_crop_starts(1408, 560, 72, K[0, 2])
    left = scaled_crop_intrinsics(K, (1408, 376), starts[0], 560, (160, 96))
    right = scaled_crop_intrinsics(K, (1408, 376), starts[1], 560, (160, 96))
    assert VIEW_LAYOUT_VERSION == "front2_left3_right3_v1"
    assert VIEW_CAMERA_IDS == (0, 0, 1, 1, 1, 2, 2, 2)
    assert float(left[0, 2]) > 80.0
    assert float(right[0, 2]) < 80.0
    assert torch.isclose(left[0, 0], right[0, 0])
    assert torch.isclose(left[1, 1], right[1, 1])


def test_vggt_motion_scale_recovers_metric_gauge():
    from scripts.build_vggt_street_cache import estimate_motion_metric_scale

    pred_centers = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],  # duplicate virtual-view center must not bias the fit
    ])
    pred_w2c = torch.eye(4).repeat(4, 1, 1)[:, :3]
    pred_w2c[:, :3, 3] = -pred_centers
    R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    gt_centers = 3.5 * (pred_centers @ R.T) + torch.tensor([10.0, -4.0, 2.0])
    gt_world_cam = torch.eye(4).repeat(4, 1, 1)
    gt_world_cam[:, :3, 3] = gt_centers

    fit = estimate_motion_metric_scale(pred_w2c, gt_world_cam)
    assert abs(float(fit["scale"]) - 3.5) < 1e-5
    assert float(fit["alignment_rmse_m"]) < 1e-5
    aligned = (
        pred_centers * fit["scale"] @ fit["alignment_rotation"].T
        + fit["alignment_translation_m"]
    )
    assert torch.allclose(aligned[:3], gt_centers[:3], atol=1e-5)
    assert fit["anchor_indices"].tolist() == [0, 1, 2]
    assert fit["scale_source"] == "camera_rig"


def test_vggt_scale_uses_vehicle_motion_for_multiframe_subset():
    from scripts.build_vggt_street_cache import estimate_motion_metric_scale

    # Eight views per frame but only three physical camera centers.  The
    # predicted physical-camera centroid moves 2 VGGT units.  Ground-truth
    # camera-center averages are deliberately inconsistent: metric scale must
    # come from the explicit 6 m vehicle displacement, yielding exactly 3.
    pred_centers = torch.tensor([
        [-0.3, 0.0, 0.0], [-0.3, 0.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0],
        [1.7, 0.0, 0.0], [1.7, 0.0, 0.0],
        [2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [2.3, 0.0, 0.0], [2.3, 0.0, 0.0], [2.3, 0.0, 0.0],
    ])
    pred_w2c = torch.eye(4).repeat(16, 1, 1)[:, :3]
    pred_w2c[:, :3, 3] = -pred_centers
    gt_world_cam = torch.eye(4).repeat(16, 1, 1)
    gt_world_cam[:, :3, 3] = pred_centers * 9.0
    gt_world_vehicle = torch.eye(4).repeat(2, 1, 1)
    gt_world_vehicle[1, 0, 3] = 6.0

    fit = estimate_motion_metric_scale(
        pred_w2c, gt_world_cam, views_per_frame=8,
        gt_world_vehicle=gt_world_vehicle,
        view_camera_ids=torch.tensor(VIEW_CAMERA_IDS),
    )
    assert abs(float(fit["scale"]) - 3.0) < 1e-5
    assert fit["scale_source"] == "vehicle_motion"
    assert fit["pair_count"] == 1


def test_vggt_single_frame_averages_virtual_views_per_physical_camera():
    from scripts.build_vggt_street_cache import estimate_motion_metric_scale

    pred_centers = torch.tensor([
        [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0],
        [0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0],
        [0.9, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0],
    ])
    pred_w2c = torch.eye(4).repeat(8, 1, 1)[:, :3]
    pred_w2c[:, :3, 3] = -pred_centers
    gt_world_cam = torch.eye(4).repeat(8, 1, 1)
    gt_world_cam[:2, 0, 3] = 0.0
    gt_world_cam[2:5, 0, 3] = 1.0
    gt_world_cam[5:, 0, 3] = 2.0

    fit = estimate_motion_metric_scale(
        pred_w2c, gt_world_cam, views_per_frame=8,
        view_camera_ids=torch.tensor(VIEW_CAMERA_IDS),
    )
    assert abs(float(fit["scale"]) - 2.0) < 1e-5
    assert fit["scale_source"] == "camera_rig"


def test_vggt_scale_reliability_labels_single_frame_and_motion_evidence():
    from world3d.unified_bev.data import geometry_scale_reliability

    assert geometry_scale_reliability("camera_rig", 3) == (
        "single_frame_camera_rig_fallback"
    )
    assert geometry_scale_reliability("vehicle_motion", 1) == (
        "single_baseline_vehicle_motion"
    )
    assert geometry_scale_reliability("vehicle_motion", 3) == (
        "multi_baseline_vehicle_motion"
    )


def test_vggt_preprocessing_keeps_unit_range_and_aspect():
    from scripts.build_vggt_street_cache import (
        _resize_for_vggt,
        parse_subset_specs,
        vggt_confidence_score,
    )

    images = torch.rand(8, 3, 96, 160)
    resized, original_hw = _resize_for_vggt(images, 518)
    assert original_hw == (96, 160)
    assert resized.shape == (8, 3, 308, 518)
    assert float(resized.min()) >= 0.0 and float(resized.max()) <= 1.0
    assert parse_subset_specs("0:1,0:2,0:2,4:4", 8) == [(0, 1), (0, 2), (4, 4)]
    score = vggt_confidence_score(torch.tensor([1.0, 2.0, 11.0]))
    assert 0.0 <= float(score.min()) and float(score.max()) <= 1.0
    assert score[0] < score[1] < score[2]
    constant = vggt_confidence_score(torch.ones(4, 8, 8))
    assert torch.allclose(constant, torch.full_like(constant, 0.5))


def test_joint_geometry_cache_requires_exact_subset():
    from world3d.unified_bev.data import (
        dense_geometry_from_blob,
        dense_geometry_subset_qa,
    )

    entry = {
        "depth": torch.ones(8, 2, 3),
        "conf": torch.ones(8, 2, 3),
        "metric_scale": torch.tensor(2.5),
        "scale_source": "camera_rig",
        "scale_pair_count": 3,
        "scale_relative_mad": torch.tensor(0.1),
        "pose_alignment_rmse_m": torch.tensor(0.2),
    }
    joint = {"geometry_model": "vggt", "subsets": {"s0_n1": entry}}
    depth, conf = dense_geometry_from_blob(joint, 0, 1, 8)
    assert depth.shape == conf.shape == (8, 2, 3)
    qa = dense_geometry_subset_qa(joint, 0, 1)
    assert qa["metric_scale"] == 2.5
    assert qa["scale_reliability"] == "single_frame_camera_rig_fallback"
    try:
        dense_geometry_from_blob(joint, 0, 2, 7)
    except KeyError:
        pass
    else:
        raise AssertionError("joint geometry must not be sliced from a larger inference")

    legacy = {"depth": torch.arange(32).view(16, 1, 2), "conf": torch.ones(16, 1, 2)}
    depth, _ = dense_geometry_from_blob(legacy, 1, 1, 8)
    assert torch.equal(depth, legacy["depth"][8:16].float())


def test_geometry_cache_identity_rejects_index_mismatch():
    from world3d.unified_bev.data import validate_geometry_blob_identity

    expected = {
        "drive": "drive_a", "target_fid": 10, "source_fids": [1, 2, 3],
        "view_layout_version": VIEW_LAYOUT_VERSION,
    }
    blob = {"sample_identity": dict(expected)}
    validate_geometry_blob_identity(blob, expected)
    wrong = dict(expected)
    wrong["target_fid"] = 11
    try:
        validate_geometry_blob_identity(blob, wrong)
    except RuntimeError as exc:
        assert "sample mismatch" in str(exc)
    else:
        raise AssertionError("misindexed geometry cache must fail before use")
    wrong_layout = dict(expected)
    wrong_layout["view_layout_version"] = "legacy_front1"
    try:
        validate_geometry_blob_identity(blob, wrong_layout)
    except RuntimeError as exc:
        assert "sample mismatch" in str(exc)
    else:
        raise AssertionError("geometry cache view-layout mismatch must fail before use")


def test_dense_geometry_attach_infers_frame_count_from_source_views():
    from world3d.unified_bev.data import attach_dense_geometry

    class CachedSample:
        views_per_frame = 8

        def __len__(self):
            return 1

        def __getitem__(self, _):
            return {
                "source_rgb": torch.zeros(16, 3, 2, 2),
                "meta": {
                    "drive": "drive_a", "target_fid": 10,
                    "source_fids": [1, 2],
                    "view_layout_version": VIEW_LAYOUT_VERSION,
                },
            }

    with TemporaryDirectory() as tmp:
        blob = {
            "sample_identity": CachedSample()[0]["meta"],
            "subsets": {
                "s0_n2": {
                    "depth": torch.ones(16, 2, 2),
                    "conf": torch.ones(16, 2, 2),
                },
            },
        }
        torch.save(blob, Path(tmp) / "000000.pt")
        sample = attach_dense_geometry(CachedSample(), tmp)[0]
        assert sample["dense_depth"].shape == (16, 2, 2)


# ---------------------------------------------------------------------------
# route-chunk primitives and the chunk-mode experiment contract
# ---------------------------------------------------------------------------

def _straight_trajectory(n, step=1.0):
    import numpy as np
    return np.stack([np.arange(n) * step, np.zeros(n)], axis=1)


def test_route_chunks_cut_by_arc_and_split_at_jumps():
    import numpy as np
    from world3d.unified_bev.chunks import build_route_chunks

    pos = _straight_trajectory(40)
    fids = list(range(100, 140))
    chunks = build_route_chunks(pos, fids, chunk_arc_m=12.0, max_step_m=5.0)
    assert [len(c.fids) for c in chunks] == [13, 13, 13, 1]
    assert all(abs(c.arc_length - 12.0) < 1e-9 for c in chunks[:3])
    assert chunks[0].fids[0] == 100 and chunks[1].fids[0] == 113

    pos_jump = pos.copy()
    pos_jump[20:] += np.array([500.0, 0.0])
    jumped = build_route_chunks(pos_jump, fids, chunk_arc_m=12.0, max_step_m=5.0)
    assert {c.segment for c in jumped} == {0, 1}
    assert jumped[0].fids == list(range(100, 113))
    assert jumped[2].segment == 1 and jumped[2].fids[0] == 120


def test_chunk_hole_patterns_and_guard_band():
    import numpy as np
    from world3d.unified_bev.chunks import (
        build_chunk_windows,
        build_route_chunks,
        core_member_index,
        guard_keep_mask,
        missing_chunks,
    )

    pos = _straight_trajectory(52)
    chunks = build_route_chunks(pos, list(range(100, 152)))
    windows = build_chunk_windows(chunks, chunks_per_window=4,
                                  min_frames_per_chunk=6, max_window_span_m=52.0)
    assert len(windows) == 1
    w = windows[0]
    # holes are contiguous interior blocks starting at chunk 1
    assert [c.index for c in missing_chunks(w, 3)] == [1]
    assert [c.index for c in missing_chunks(w, 2)] == [1, 2]
    assert [c.index for c in missing_chunks(w, 1)] == [1, 2, 3]
    # guard drops kept frames within guard of the hole interval
    hole = (12.0, 13.0 + 12.0)
    kept2 = guard_keep_mask(w[3], hole, 4.0)
    arcs = w[3].member_arcs()
    assert (kept2 == (arcs >= hole[1] + 4.0)).all()
    # the query core frame sits deeper than the guard inside its chunk
    for c in w:
        i = core_member_index(c, 4.0)
        assert min(abs(arcs[i] - c.arc_start) for arcs in [c.member_arcs()]) >= 0
        assert min(
            c.member_arcs()[i] - c.arc_start, c.arc_end - c.member_arcs()[i],
        ) >= 4.0


def test_select_chunk_frames_guard_safe_subset_of_geometry():
    import numpy as np
    from world3d.unified_bev.chunks import (
        build_route_chunks,
        missing_chunks,
        select_chunk_frames,
    )

    pos = _straight_trajectory(52)
    chunks = build_route_chunks(pos, list(range(100, 152)))
    w = chunks[:4]
    # chunk 0 guards only its right side (the first hole always starts at c1)
    lift0, geom0 = select_chunk_frames(
        w[0], 2, 4.0, 8, guard_left=False, guard_right_arc=w[1].arc_start)
    assert len(lift0) == 2 and set(lift0) <= set(geom0) and len(geom0) <= 8
    # other chunks guard only their left side
    for c in w[1:]:
        lift_c, geom_c = select_chunk_frames(c, 2, 4.0, 8)
        assert len(lift_c) == 2 and set(lift_c) <= set(geom_c) and len(geom_c) <= 8
    # the guard property holds against EVERY hole pattern
    for K in (1, 2, 3):
        missing = missing_chunks(w, K)
        hole = (min(c.arc_start for c in missing), max(c.arc_end for c in missing))
        kept = [c for c in w if c not in missing]
        lifts = {}
        lifts[w[0].index] = select_chunk_frames(
            w[0], 2, 4.0, 8, guard_left=False, guard_right_arc=w[1].arc_start)[0]
        for c in w[1:]:
            lifts[c.index] = select_chunk_frames(c, 2, 4.0, 8)[0]
        for c in kept:
            for m in lifts[c.index]:
                arc = c.member_arcs()[m]
                dist = max(hole[0] - arc, arc - hole[1], 0.0)
                assert dist >= 4.0 - 1e-6, (K, c.index, dist)


def test_chunk_identity_is_idempotent():
    from world3d.unified_bev.data import (
        geometry_sample_identity,
        validate_geometry_blob_identity,
    )

    meta = {
        "drive": "d", "target_fid": 7,
        "chunk_table": [{
            "index": 0, "fids": [1, 2, 3], "geometry_fids": [1, 3],
            "lift_fids": [1, 3], "arc_start": 0.0, "arc_end": 2.0, "core_fid": 2,
        }],
        "query_fids": [2], "guard_m": 4.0, "frames_per_chunk": 2,
        "chunking_version": "route_chunk_v1",
        "view_layout_version": "front2_left3_right3_v1",
    }
    identity = geometry_sample_identity(meta)
    assert identity["geometry_fids"] == [[1, 3]]
    validate_geometry_blob_identity({"sample_identity": identity}, meta)
    other = dict(meta, target_fid=8)
    try:
        validate_geometry_blob_identity({"sample_identity": identity}, other)
        raise AssertionError("identity mismatch must fail")
    except RuntimeError:
        pass


def test_completion_alpha_schedule_over_chunk_counts():
    import torch
    from world3d.unified_bev.models import LatentCompletion

    torch.manual_seed(0)
    n_chunks = 4
    completion = LatentCompletion(mode="residual", channels=8,
                                  bev_height=16, bev_width=16, tile_size_m=48.0)
    z_gnd = torch.randn(1, 8, 16, 16)
    z_sat = torch.randn(1, 8, 16, 16)
    coverage = torch.zeros(1, 1, 16, 16)
    # K = Nc is the dense identity, bit-for-bit
    out = completion(z_sat, z_gnd, coverage, n_chunks, n_chunks)
    assert out.latent is z_gnd
    assert float(out.correction.abs().sum()) == 0.0
    # fewer kept chunks leave progressively more room for the correction
    magnitudes = []
    for k in (1, 2, 3):
        out = completion(z_sat, z_gnd, coverage, k, n_chunks)
        magnitudes.append(float(out.correction.abs().mean()))
    assert magnitudes[0] >= magnitudes[1] >= magnitudes[2]
