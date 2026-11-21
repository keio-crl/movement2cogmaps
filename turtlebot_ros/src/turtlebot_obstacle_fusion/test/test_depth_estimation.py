from math import radians

import numpy as np

from turtlebot_obstacle_fusion.depth_estimation import (
    DepthFit,
    backproject,
    bin_ranges,
    extrinsic_from_pose,
    fit_inverse_depth,
    obstacle_points,
    pinhole_from_hfov,
    project_to_image,
    ranges_to_points,
    render_panel,
)


def test_extrinsic_maps_optical_axis_forward_and_pitch_down() -> None:
    rotation, translation = extrinsic_from_pose(0.1, 0.0, 0.3, 0.0, 0.0)
    assert np.allclose(rotation @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    assert np.allclose(rotation @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0])
    assert np.allclose(rotation @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0])
    pitched, _ = extrinsic_from_pose(0.0, 0.0, 0.3, 0.0, radians(30.0))
    assert pitched @ [0.0, 0.0, 1.0] @ [0.0, 0.0, 1.0] < 0.0


def test_projection_and_backprojection_round_trip() -> None:
    intrinsics = pinhole_from_hfov(320, 240, 69.0)
    points = np.array([[0.2, -0.1, 1.5], [-0.3, 0.2, 0.8], [0.0, 0.0, -1.0], [50.0, 0.0, 1.0]])
    uv, kept = project_to_image(points, intrinsics, 320, 240)
    assert list(kept) == [0, 1]
    depth = np.full((240, 320), np.nan)
    depth[uv[:, 1], uv[:, 0]] = points[kept, 2]
    back, pixels = backproject(depth, intrinsics, 1)
    order = np.argsort(back[:, 2])[::-1]
    assert np.allclose(back[order], points[kept], atol=0.01)
    assert len(pixels) == 2


def test_fit_recovers_scale_and_offset_with_outliers() -> None:
    rng = np.random.default_rng(0)
    metres = rng.uniform(0.4, 3.0, 60)
    relative = (1.0 / metres - 0.05) / 0.02
    relative[:8] *= 3.0
    fit = fit_inverse_depth(relative, metres)
    assert fit is not None
    assert abs(fit.a - 0.02) < 1e-6 and abs(fit.b - 0.05) < 1e-4
    assert fit.inliers == 52
    assert np.allclose(fit.to_metres(relative[8:]), metres[8:])


def test_fit_with_equal_distances_fits_scale_only() -> None:
    metres = np.full(30, 1.2) + np.linspace(-0.01, 0.01, 30)
    relative = 1.0 / metres / 0.5
    fit = fit_inverse_depth(relative, metres)
    assert fit is not None and fit.b == 0.0 and abs(fit.a - 0.5) < 1e-3


def test_fit_rejects_too_few_pairs_and_inconsistent_data() -> None:
    assert fit_inverse_depth(np.ones(5), np.ones(5)) is None
    rng = np.random.default_rng(1)
    assert fit_inverse_depth(rng.uniform(0, 1, 40), rng.uniform(0.5, 3.0, 40)) is None


def test_to_metres_marks_non_positive_inverse_depth_as_infinite() -> None:
    assert np.isinf(DepthFit(1.0, -1.0, 0.0, 0, 0).to_metres(np.array([0.5]))[0])


def test_obstacle_points_drop_floor_and_high_objects() -> None:
    width, height = 160, 120
    intrinsics = pinhole_from_hfov(width, height, 90.0)
    rotation, translation = extrinsic_from_pose(0.0, 0.0, 0.3, 0.0, 0.0)
    fx, fy, cx, cy = intrinsics
    depth = np.full((height, width), 1.0)
    rows_z = 0.3 - (np.arange(height) - cy) / fy * 1.0
    depth[rows_z < 0.0] = np.inf
    xy, pixels = obstacle_points(depth, intrinsics, rotation, translation, 1, (0.1, 2.5), (0.05, 0.40))
    heights = 0.3 - (pixels[:, 1] - cy) / fy
    assert len(xy) > 0
    assert heights.min() > 0.05 and heights.max() < 0.40
    assert np.allclose(xy[:, 0], 1.0)


def test_bin_ranges_uses_low_percentile_and_min_points() -> None:
    points = np.array([[1.0, 0.001]] * 5 + [[1.2, 0.001]] * 5 + [[0.0, 2.0]] * 2)
    ranges = bin_ranges(points, 360, 5)
    assert abs(ranges[180] - 1.0) < 1e-5
    assert np.isinf(ranges[270])
    back = ranges_to_points(ranges, -np.pi, 2.0 * np.pi / 360)
    assert back.shape == (1, 2) and abs(back[0, 0] - 1.0) < 0.01


def test_render_panel_produces_an_image() -> None:
    bgr = np.zeros((120, 160, 3), dtype=np.uint8)
    panel = render_panel(bgr, np.full((120, 160), 1.0), np.array([[10, 10]]), np.array([[20, 30]]), np.array([1.0]), np.array([[1.0, 0.0]]), np.array([[0.8, 0.0]]), 0.17, 0.22, 2.5, ["a", "b"])
    assert panel.ndim == 3 and panel.shape[0] == 320


def test_distorted_projection_and_backprojection_round_trip() -> None:
    intrinsics = (202.96, 202.96, 121.68, 129.17)
    distortion = np.array([-3.983, -36.99, -0.000357, -0.000214, 334.49, -4.152, -34.94, 325.65])
    points = np.array([[0.3, -0.2, 1.0], [-0.4, 0.25, 1.5], [0.05, 0.1, 0.8]])
    uv, kept = project_to_image(points, intrinsics, 250, 250, distortion=distortion)
    assert list(kept) == [0, 1, 2]
    pinhole_uv, _ = project_to_image(points, intrinsics, 250, 250)
    assert np.abs(uv - pinhole_uv).max() >= 1
    depth = np.full((250, 250), np.nan)
    depth[uv[:, 1], uv[:, 0]] = points[:, 2]
    back, pixels = backproject(depth, intrinsics, 1, distortion)
    for point in points:
        assert np.min(np.linalg.norm(back - point, axis=1)) < 0.01


def test_floor_pairs_give_exact_floor_depth() -> None:
    from turtlebot_obstacle_fusion.depth_estimation import floor_pairs

    width, height = 160, 120
    intrinsics = pinhole_from_hfov(width, height, 90.0)
    rotation, translation = extrinsic_from_pose(0.0, 0.0, 0.3, 0.0, radians(20.0))
    relative, metres, pixels = floor_pairs(np.zeros((height, width)), intrinsics, rotation, translation, 4, (0.3, 3.0))
    assert len(metres) > 50
    fx, fy, cx, cy = intrinsics
    rays = np.column_stack([(pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels))]) * metres[:, None]
    floor = rays @ rotation.T + translation
    assert np.allclose(floor[:, 2], 0.0, atol=1e-9)
    assert np.all((np.hypot(floor[:, 0], floor[:, 1]) >= 0.3) & (np.hypot(floor[:, 0], floor[:, 1]) <= 3.0))


def test_fit_rejects_when_too_few_pairs_agree() -> None:
    rng = np.random.default_rng(2)
    metres = rng.uniform(0.4, 3.0, 100)
    relative = (1.0 / metres - 0.05) / 0.02
    relative[:50] = rng.uniform(0.0, 50.0, 50)
    assert fit_inverse_depth(relative, metres) is not None
    assert fit_inverse_depth(relative, metres, min_inlier_ratio=0.6) is None


def test_obstacle_memory_keeps_points_in_odom_frame_until_expiry() -> None:
    from turtlebot_obstacle_fusion.depth_estimation import ObstacleMemory, base_to_odom, odom_to_base

    pose = (1.0, 2.0, np.pi / 2)
    seen = np.array([[0.8, 0.0], [0.8, 0.01]])
    world = base_to_odom(seen, pose)
    assert np.allclose(world[0], [1.0, 2.8])
    memory = ObstacleMemory(0.03, 8.0, 3.0)
    memory.add(world, 0.0)
    assert len(memory.points(1.0, pose[:2])) == 1
    moved = (1.0, 2.5, np.pi / 2)
    assert np.allclose(odom_to_base(memory.points(5.0, moved[:2]), moved), [[0.3, 0.0]], atol=0.02)
    assert len(memory.points(9.0, moved[:2])) == 0
    memory.add(world, 10.0)
    assert len(memory.points(10.0, (10.0, 10.0))) == 0


def test_fuse_ranges_keeps_the_nearer_sensor_per_angle() -> None:
    from turtlebot_obstacle_fusion.scan_depth_fusion_node import fuse_ranges

    lidar = np.array([[1.25, 0.0], [0.0, 2.0]])
    depth = np.array([[0.88, 0.0], [0.0, 3.0]])
    ranges = fuse_ranges(lidar, depth, 360)
    assert abs(ranges[180] - 0.88) < 1e-9
    assert abs(ranges[270] - 2.0) < 1e-9
    assert np.isinf(ranges[0])


def test_decode_image_handles_raw_encodings_and_jpeg() -> None:
    import cv2

    from turtlebot_obstacle_fusion.depth_estimation import decode_image

    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[..., 0], bgr[..., 2] = 10, 200
    assert np.array_equal(decode_image(bgr.tobytes(), "bgr8", 6, 4, 18), bgr)
    assert np.array_equal(decode_image(bgr[..., ::-1].copy().tobytes(), "rgb8", 6, 4, 18), bgr)
    padded = np.zeros((4, 20, 3), dtype=np.uint8)
    padded[:, :6] = bgr
    assert np.array_equal(decode_image(padded.tobytes(), "bgr8", 6, 4, 60), bgr)
    assert decode_image(b"\x00" * 10, "bgr8", 6, 4, 18) is None
    assert decode_image(b"", "16UC1", 6, 4, 12) is None
    jpeg = cv2.imencode(".png", bgr)[1].tobytes()
    assert np.array_equal(decode_image(jpeg), bgr)


def test_lidar_pairs_recover_a_known_scale() -> None:
    from turtlebot_obstacle_fusion.depth_estimation import lidar_pairs

    width, height = 320, 240
    intrinsics = pinhole_from_hfov(width, height, 80.0)
    rotation, translation = extrinsic_from_pose(0.0, 0.0, 0.25, 0.0, radians(10.0))
    angles = np.radians(np.linspace(-30.0, 30.0, 61))
    distances = np.linspace(0.8, 2.0, 61)
    lidar_xy = np.stack([distances * np.cos(angles), distances * np.sin(angles)], axis=1)
    lidar_cam = (np.column_stack([lidar_xy, np.full(61, 0.19)]) - translation) @ rotation
    relative = np.zeros((height, width))
    uv, kept = project_to_image(lidar_cam, intrinsics, width, height)
    relative[uv[:, 1], uv[:, 0]] = (1.0 / lidar_cam[kept, 2] - 0.05) / 0.02
    rel, metres, pixels = lidar_pairs(relative, lidar_xy, 0.19, intrinsics, rotation, translation, 0)
    assert len(metres) == len(kept) > 40
    fit = fit_inverse_depth(rel, metres)
    assert fit is not None and abs(fit.a - 0.02) < 1e-4 and abs(fit.b - 0.05) < 1e-3


def test_decode_compressed_depth_reads_png_after_header() -> None:
    import cv2

    from turtlebot_obstacle_fusion.depth_estimation import decode_compressed_depth

    depth = np.zeros((6, 8), dtype=np.uint16)
    depth[2, 3], depth[4, 7] = 812, 2500
    png = cv2.imencode(".png", depth)[1].tobytes()
    assert np.array_equal(decode_compressed_depth(b"\x00" * 12 + png), depth)
    assert decode_compressed_depth(b"\x00" * 12) is None
    colour = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))[1].tobytes()
    assert decode_compressed_depth(b"\x00" * 12 + colour) is None


def test_tilt_correction_flattens_a_floor_seen_by_a_tilted_camera() -> None:
    from turtlebot_obstacle_fusion.depth_estimation import floor_tilt, tilt_correction

    rng = np.random.default_rng(3)
    floor = np.column_stack([rng.uniform(0.5, 1.8, 500), rng.uniform(-0.5, 0.5, 500), np.zeros(500)])
    camera = np.array([-0.06, 0.0, 0.244])
    true_tilt = tilt_correction(6.5, -1.1)
    seen = (floor - camera) @ true_tilt + camera
    pitch, roll, _ = floor_tilt(seen)
    assert abs(pitch - 6.5) < 0.3 and abs(roll + 1.1) < 0.3
    corrected = (seen - camera) @ tilt_correction(pitch, roll).T + camera
    assert np.abs(corrected[:, 2]).max() < 0.01
    assert all(np.isnan(floor_tilt(seen[:10])))
