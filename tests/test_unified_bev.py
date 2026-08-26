"""Contracts for the modules the persistent world-state mainline consumes.

Retired with the frame-centred completion experiment (2026-08-26): the tests
for LatentCompletion/alpha identity, coordinate-only completion, the satellite
ViT and heightmap priors, nadir round-trip, M3D crops, observation-aware RGB
losses, road-frame B7 controls, LPIPS/SSIM, the frame Stage-A/B checkpoint
schema, and the v6 frame-cache attach validators.  What remains guards the
geometry conventions, view builders, VGGT scaling, dense lift, chunk
primitives, readers, and render math that the world-state chain imports.
"""
from __future__ import annotations

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
from world3d.unified_bev.losses import masked_smooth_l1
from world3d.unified_bev.data import (
    VIEW_CAMERA_IDS,
    VIEW_LAYOUT_VERSION,
    centered_two_crop_starts,
    scaled_crop_intrinsics,
)
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    fixed_relative_xy_encoding,
)
from world3d.unified_bev.readouts import BEVHeightDecoder, freeze_module


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


def test_depth_reader_render_shapes():
    """ColumnFieldDecoder is the frozen depth reader: single- and multi-view
    renders must be finite with the documented shapes."""
    B, H, W = 1, 32, 48
    z = torch.randn(B, 8, 16, 16)
    K = torch.tensor([[[30.0, 0.0, W / 2], [0.0, 30.0, H / 2], [0.0, 0.0, 1.0]]])
    T = torch.eye(4).unsqueeze(0)
    dec = ColumnFieldDecoder(latent_channels=8, hidden=16, samples=4)
    rgb, depth, opacity = dec.render(
        z, K, T, torch.zeros(B, 2), tile_size_m=32.0, image_size=(W, H), far_m=10.0,
    )
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


def test_ray_distance_is_converted_to_camera_z():
    distance = torch.tensor([10.0])
    direction = torch.tensor([[0.6, 0.0, 0.8]])
    assert torch.allclose(ray_distance_to_camera_z(distance, direction), torch.tensor([8.0]))


def test_kitti360_satellite_spec():
    spec = SatSpec()
    assert spec.width == 512 and spec.height == 512
    assert spec.meters_per_pixel == 0.196
    assert spec.cx == spec.cy == 256.0


def test_fixed_relative_xy_encoding_orientation():
    """The XY prior buffer of both writers: south-up orientation, zero-mean,
    fixed (never learned)."""
    encoding = fixed_relative_xy_encoding(8, 12, 12, tile_size_m=24.0)
    assert encoding.shape == (1, 8, 12, 12)
    assert float(encoding[0, 0, 6, 0]) < 0 < float(encoding[0, 0, 6, -1])
    assert float(encoding[0, 1, 0, 6]) < 0 < float(encoding[0, 1, -1, 6])
    again = fixed_relative_xy_encoding(8, 12, 12, tile_size_m=24.0)
    assert torch.equal(encoding, again)


def test_frozen_bev_height_decoder_is_shared_readout_not_a_gradient_barrier():
    decoder = freeze_module(BEVHeightDecoder(latent_channels=8, width=16))
    latent = torch.randn(2, 8, 16, 16, requires_grad=True)
    prediction = decoder(latent)
    assert prediction.shape == (2, 1, 16, 16)
    prediction.square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_masked_smooth_l1_handles_empty_masks():
    """Empty supervision regions are expected (e.g. no valid cells yet); the
    loss must be a finite differentiable zero, not NaN."""
    pred = torch.randn(1, 3, 16, 16, requires_grad=True)
    target = torch.randn_like(pred)
    empty = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    loss = masked_smooth_l1(pred, target, empty)
    assert float(loss) == 0.0 and loss.requires_grad
    loss.backward()
    assert pred.grad is not None


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
        vggt_confidence_score,
    )

    images = torch.rand(8, 3, 96, 160)
    resized, original_hw = _resize_for_vggt(images, 518)
    assert original_hw == (96, 160)
    assert resized.shape == (8, 3, 308, 518)
    assert float(resized.min()) >= 0.0 and float(resized.max()) <= 1.0
    score = vggt_confidence_score(torch.tensor([1.0, 2.0, 11.0]))
    assert 0.0 <= float(score.min()) and float(score.max()) <= 1.0
    assert score[0] < score[1] < score[2]
    constant = vggt_confidence_score(torch.ones(4, 8, 8))
    assert torch.allclose(constant, torch.full_like(constant, 0.5))


# ---------------------------------------------------------------------------
# route-chunk primitives (measurement packets for world-state assimilation)
# ---------------------------------------------------------------------------

def _straight_trajectory(n, step=1.0):
    return np.stack([np.arange(n) * step, np.zeros(n)], axis=1)


def test_route_chunks_cut_by_arc_and_split_at_jumps():
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


def test_select_chunk_frames_lift_subset_of_geometry():
    from world3d.unified_bev.chunks import (
        build_route_chunks,
        core_member_index,
        select_chunk_frames,
    )

    pos = _straight_trajectory(52)
    chunks = build_route_chunks(pos, list(range(100, 152)))
    c = chunks[0]
    lift, geom = select_chunk_frames(c, 4, 0.0, 8)
    assert len(lift) == 4 and set(lift) <= set(geom) and len(geom) <= 8
    i = core_member_index(c, 0.0)
    assert 0 <= i < len(c.fids)
