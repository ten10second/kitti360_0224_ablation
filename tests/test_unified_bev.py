from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from world3d.geo.sat_alignment import SatSpec
from world3d.unified_bev.geometry import (
    bev_grid_from_world_xy,
    bilinear_splat,
    height_statistics,
    image_uv_to_grid,
    ray_distance_to_camera_z,
    se3_inverse,
)
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    LatentCompletion,
    fixed_relative_xy_encoding,
    nadir_distance,
    satellite_bev_crop,
)
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
    assert parse_source_choices("1,2,4,8", fixed=2, dense=8) == (1, 2, 4, 8)
    assert parse_source_choices(None, fixed=2, dense=8) == (2,)
    try:
        parse_source_choices("1,16", fixed=2, dense=8)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range source choice must fail")


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
        assert torch.equal(out, z_gnd), mode
    assert torch.equal(
        _make_completion("satellite_only")(z_sat, z_gnd, coverage, 8, 8), z_sat
    )
    assert torch.equal(
        _make_completion("ground_only")(z_sat, z_gnd, coverage, 8, 8), z_gnd
    )
    module = _make_completion("residual")
    for bad in (0, 9):
        try:
            module(z_sat, z_gnd, coverage, bad, 8)
        except ValueError:
            pass
        else:
            raise AssertionError(f"n_sparse={bad} must be rejected")


def test_completion_conf_range_and_prior_routing():
    z_gnd = torch.randn(1, 8, 12, 12)
    coverage = (torch.rand(1, 1, 12, 12) > 0.6).float()
    z_sat_a, z_sat_b = torch.randn(1, 8, 12, 12), torch.randn(1, 8, 12, 12)

    residual = _make_completion("residual")
    conf = residual.conf(torch.cat([z_gnd, coverage], dim=1))
    assert conf.shape == (1, 1, 12, 12)
    assert 0.0 <= float(conf.min()) and float(conf.max()) <= 1.0
    assert not torch.allclose(
        residual(z_sat_a, z_gnd, coverage, 2, 8),
        residual(z_sat_b, z_gnd, coverage, 2, 8),
    ), "residual mode must consume satellite content"

    coord = _make_completion("coordinate_only")
    first = coord(z_sat_a, z_gnd, coverage, 2, 8)
    assert torch.equal(first, coord(z_sat_b, z_gnd, coverage, 2, 8)), \
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
    output_with_xy = with_xy(z_sat, z_gnd, coverage, 2, 8)
    with torch.no_grad():
        with_xy.coord_embed.zero_()
    assert not torch.equal(output_with_xy, with_xy(z_sat, z_gnd, coverage, 2, 8)), \
        "fixed XY must be routed through the shared delta path"


def test_completion_alpha_schedule_scales_correction():
    """alpha(2)=0.75 and alpha(4)=0.5 share the same conf/correction path, so
    the applied correction must scale by exactly the alpha ratio."""
    z_gnd = torch.randn(1, 8, 12, 12)
    z_sat = torch.randn(1, 8, 12, 12)
    coverage = (torch.rand(1, 1, 12, 12) > 0.6).float()
    module = _make_completion("residual")
    at2 = module(z_sat, z_gnd, coverage, 2, 8) - z_gnd
    at4 = module(z_sat, z_gnd, coverage, 4, 8) - z_gnd
    assert torch.allclose(at2, 1.5 * at4, atol=1e-6)


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
    prior, h_pred, _ = enc(sat, z_gnd, 64.0, 0.196)
    assert prior.shape == (1, 16, 64, 64)
    assert h_pred.shape == (1, 1, 64, 64)
    assert float(h_pred.min()) >= 0.0 and float(h_pred.max()) <= 60.0
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
