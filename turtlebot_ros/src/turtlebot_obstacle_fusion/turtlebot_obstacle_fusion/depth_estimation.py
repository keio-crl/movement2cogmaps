"""Torch-free geometry for turning a Depth Anything V2 relative depth map into 2D obstacle ranges.

Camera points use the optical convention (x right, y down, z forward). Depth Anything V2 returns
relative inverse depth ``r``; it is converted to metres with ``1 / z = a r + b`` where ``a`` and
``b`` are fitted every frame against LiDAR points projected into the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, radians, sin, tan
from typing import Optional, Tuple

import numpy as np

Intrinsics = Tuple[float, float, float, float]
Distortion = Optional[np.ndarray]

OPTICAL_TO_BODY = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def pinhole_from_hfov(width: int, height: int, hfov_deg: float) -> Intrinsics:
    """Return (fx, fy, cx, cy) of an undistorted pinhole camera with square pixels."""
    fx = (width / 2.0) / tan(radians(hfov_deg) / 2.0)
    return fx, fx, (width - 1) / 2.0, (height - 1) / 2.0


def extrinsic_from_pose(x: float, y: float, z: float, yaw: float, pitch: float, roll: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, t) mapping optical-frame points into the base frame.

    The camera body frame (x forward, y left, z up) is placed at (x, y, z) and rotated by
    ``roll``, then ``pitch`` (positive tilts the view down), then ``yaw`` (positive turns left).
    """
    cr, sr, cp, sp, cy, sy = cos(roll), sin(roll), cos(pitch), sin(pitch), cos(yaw), sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx @ OPTICAL_TO_BODY, np.array([x, y, z], dtype=float)


def quaternion_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Return the rotation matrix of a unit quaternion."""
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ])


def camera_matrix(intrinsics: Intrinsics) -> np.ndarray:
    """Return the 3x3 camera matrix of (fx, fy, cx, cy)."""
    fx, fy, cx, cy = intrinsics
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def project_to_image(points_cam: np.ndarray, intrinsics: Intrinsics, width: int, height: int, min_depth: float = 0.05, distortion: Distortion = None) -> Tuple[np.ndarray, np.ndarray]:
    """Project optical-frame points to integer pixels; return (uv, index of the kept points).

    ``distortion`` holds OpenCV coefficients (plumb_bob or rational_polynomial); None = pinhole.
    """
    fx, fy, cx, cy = intrinsics
    points_cam = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    z = points_cam[:, 2]
    front = z > min_depth
    u = np.full(len(z), -1.0)
    v = np.full(len(z), -1.0)
    if distortion is not None and front.any():
        import cv2

        projected, _ = cv2.projectPoints(points_cam[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix(intrinsics), np.asarray(distortion, dtype=float))
        u[front], v[front] = projected[:, 0, 0], projected[:, 0, 1]
    else:
        u[front] = fx * points_cam[front, 0] / z[front] + cx
        v[front] = fy * points_cam[front, 1] / z[front] + cy
    keep = np.flatnonzero(front & (u >= 0.0) & (u <= width - 1) & (v >= 0.0) & (v <= height - 1))
    return np.rint(np.stack([u[keep], v[keep]], axis=1)).astype(int), keep


def sample_median(image: np.ndarray, uv: np.ndarray, radius: int) -> np.ndarray:
    """Median of ``image`` in a (2 radius + 1)^2 window around every pixel ``uv``."""
    height, width = image.shape
    values = np.empty(len(uv))
    for index, (u, v) in enumerate(uv):
        window = image[max(0, v - radius): min(height, v + radius + 1), max(0, u - radius): min(width, u + radius + 1)]
        values[index] = float(np.median(window))
    return values


@dataclass(frozen=True)
class DepthFit:
    """Affine relation ``1 / z = a r + b`` between relative inverse depth ``r`` and metres ``z``."""

    a: float
    b: float
    residual: float
    inliers: int
    pairs: int

    def to_metres(self, relative: np.ndarray) -> np.ndarray:
        """Convert relative inverse depth to metres; non-positive inverse depth becomes inf."""
        inverse = self.a * np.asarray(relative, dtype=float) + self.b
        with np.errstate(divide="ignore"):
            return np.where(inverse > 0.0, 1.0 / np.where(inverse > 0.0, inverse, 1.0), np.inf)


def fit_inverse_depth(relative: np.ndarray, metres: np.ndarray, min_pairs: int = 15, inlier_tolerance: float = 0.15, max_residual: float = 0.1, iterations: int = 5, min_inlier_ratio: float = 0.0) -> Optional[DepthFit]:
    """Robustly fit ``1 / z = a r + b`` and return None when the fit is unreliable.

    Pairs whose relative depth error exceeds ``inlier_tolerance`` are dropped and the fit repeated.
    When the LiDAR distances are nearly equal the offset is unobservable, so only ``a`` is fitted.
    The fit is rejected with fewer than ``min_pairs`` inliers or a fraction of inliers below
    ``min_inlier_ratio`` (e.g. the floor is mostly hidden by an obstacle), a non-positive ``a`` or
    a median relative error above ``max_residual``.
    """
    relative = np.asarray(relative, dtype=float)
    metres = np.asarray(metres, dtype=float)
    valid = np.isfinite(relative) & np.isfinite(metres) & (metres > 0.0)
    relative, inverse = relative[valid], 1.0 / metres[valid]
    pairs = len(relative)
    if pairs < min_pairs:
        return None
    inliers = np.ones(pairs, dtype=bool)
    a = b = 0.0
    for _ in range(iterations):
        r, w = relative[inliers], inverse[inliers]
        if len(r) < min_pairs:
            return None
        if np.std(w) < 0.1 * np.mean(w) or np.std(r) == 0.0:
            a, b = float(np.dot(r, w) / max(np.dot(r, r), 1e-12)), 0.0
        else:
            a, b = (float(value) for value in np.polyfit(r, w, 1))
        if a <= 0.0:
            return None
        predicted = DepthFit(a, b, 0.0, 0, pairs).to_metres(relative)
        error = np.abs(predicted - 1.0 / inverse) * inverse
        updated = error < inlier_tolerance
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    if inliers.sum() < max(min_pairs, min_inlier_ratio * pairs):
        return None
    residual = float(np.median(error[inliers]))
    if residual > max_residual:
        return None
    return DepthFit(a, b, residual, int(inliers.sum()), pairs)


def backproject(depth: np.ndarray, intrinsics: Intrinsics, stride: int, distortion: Distortion = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return optical-frame points and their pixels for every ``stride``-th finite depth pixel.

    ``depth`` is the distance along the optical axis; with ``distortion`` the pixels are undistorted first.
    """
    fx, fy, cx, cy = intrinsics
    v, u = np.mgrid[0: depth.shape[0]: stride, 0: depth.shape[1]: stride]
    z = depth[v, u]
    keep = np.isfinite(z)
    u, v, z = u[keep], v[keep], z[keep]
    if distortion is not None and len(u):
        import cv2

        pixels = np.stack([u, v], axis=1).astype(float).reshape(-1, 1, 2)
        normalised = cv2.undistortPoints(pixels, camera_matrix(intrinsics), np.asarray(distortion, dtype=float))[:, 0]
        x, y = normalised[:, 0], normalised[:, 1]
    else:
        x, y = (u - cx) / fx, (v - cy) / fy
    points = np.stack([x * z, y * z, z], axis=1)
    return points, np.stack([u, v], axis=1)


def obstacle_points(depth: np.ndarray, intrinsics: Intrinsics, rotation: np.ndarray, translation: np.ndarray, stride: int, depth_range: Tuple[float, float], height_range: Tuple[float, float], distortion: Distortion = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return base-frame (x, y) of pixels inside the obstacle height band, and those pixels.

    Pixels below ``height_range[0]`` (floor) or above ``height_range[1]`` (higher than the robot)
    and depths outside ``depth_range`` are dropped.
    """
    clipped = np.where((depth >= depth_range[0]) & (depth <= depth_range[1]), depth, np.nan)
    points_cam, pixels = backproject(clipped, intrinsics, stride, distortion)
    points_base = points_cam @ rotation.T + translation
    keep = (points_base[:, 2] > height_range[0]) & (points_base[:, 2] < height_range[1]) & (points_base[:, 0] > 0.0)
    return points_base[keep, :2], pixels[keep]


def floor_pairs(relative: np.ndarray, intrinsics: Intrinsics, rotation: np.ndarray, translation: np.ndarray, stride: int, distance_range: Tuple[float, float], distortion: Distortion = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair relative depth with the exact optical-axis depth of the floor plane (z = 0 in the base frame).

    Returns (relative values, metric depths, pixels) for every ``stride``-th pixel whose ray hits
    the floor at a horizontal distance inside ``distance_range``. Pixels showing an obstacle instead
    of the floor are outliers that the robust fit removes.
    """
    height, width = relative.shape
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    u, v = u.ravel(), v.ravel()
    if distortion is not None:
        import cv2

        pixels = np.stack([u, v], axis=1).astype(float).reshape(-1, 1, 2)
        normalised = cv2.undistortPoints(pixels, camera_matrix(intrinsics), np.asarray(distortion, dtype=float))[:, 0]
    else:
        fx, fy, cx, cy = intrinsics
        normalised = np.stack([(u - cx) / fx, (v - cy) / fy], axis=1)
    rays = np.column_stack([normalised, np.ones(len(u))]) @ rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(rays[:, 2] < -1e-3, -translation[2] / rays[:, 2], np.nan)
    floor_xy = translation[:2] + rays[:, :2] * scale[:, None]
    distance = np.nan_to_num(np.hypot(floor_xy[:, 0], floor_xy[:, 1]), nan=-1.0)
    keep = np.isfinite(scale) & (distance >= distance_range[0]) & (distance <= distance_range[1])
    return relative[v[keep], u[keep]], scale[keep], np.stack([u[keep], v[keep]], axis=1)


def bin_ranges(points_xy: np.ndarray, bin_count: int, min_points: int, percentile: float = 10.0) -> np.ndarray:
    """Per-angle-bin conservative range over [-pi, pi); bins with fewer than ``min_points`` are inf."""
    ranges = np.full(bin_count, np.inf)
    if len(points_xy) == 0:
        return ranges
    angles = np.arctan2(points_xy[:, 1], points_xy[:, 0])
    distances = np.hypot(points_xy[:, 0], points_xy[:, 1])
    bins = np.floor((angles + pi) / (2.0 * pi / bin_count)).astype(int) % bin_count
    for index in np.unique(bins):
        values = distances[bins == index]
        if len(values) >= min_points:
            ranges[index] = float(np.percentile(values, percentile))
    return ranges


def ranges_to_points(ranges: np.ndarray, angle_min: float, angle_increment: float) -> np.ndarray:
    """Convert the finite entries of a LaserScan-like range array to (x, y) points."""
    ranges = np.asarray(ranges, dtype=float)
    index = np.flatnonzero(np.isfinite(ranges))
    angles = angle_min + index * angle_increment
    return np.stack([ranges[index] * np.cos(angles), ranges[index] * np.sin(angles)], axis=1)


def front_minimum(points_xy: np.ndarray, half_angle: float) -> float:
    """Minimum distance of the points within +-``half_angle`` of the forward direction."""
    if len(points_xy) == 0:
        return float("inf")
    front = np.abs(np.arctan2(points_xy[:, 1], points_xy[:, 0])) <= half_angle
    return float(np.hypot(points_xy[front, 0], points_xy[front, 1]).min()) if front.any() else float("inf")


def _colour(distance: float, max_distance: float) -> Tuple[int, int, int]:
    import cv2

    level = int(np.clip(255.0 * (1.0 - distance / max_distance), 0, 255))
    return tuple(int(c) for c in cv2.applyColorMap(np.array([[level]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0])


def render_panel(bgr: np.ndarray, depth: Optional[np.ndarray], obstacle_pixels: np.ndarray, lidar_uv: np.ndarray, lidar_metres: np.ndarray, lidar_xy: np.ndarray, depth_xy: np.ndarray, robot_radius: float, body_radius: float, max_distance: float, lines: list) -> np.ndarray:
    """Compose the diagnostic image: overlay, metric depth, top view and text, side by side.

    Overlay: obstacle pixels in red, projected LiDAR points coloured by distance (red = near).
    Top view: base frame with x up; grey = LiDAR, red = depth obstacles, circles = robot body.
    """
    import cv2

    size = 320
    scale = size / bgr.shape[0]
    width = int(round(bgr.shape[1] * scale))

    overlay = bgr.copy()
    if len(obstacle_pixels):
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        mask[obstacle_pixels[:, 1], obstacle_pixels[:, 0]] = 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    overlay = cv2.resize(overlay, (width, size))
    for (u, v), distance in zip(lidar_uv, lidar_metres):
        cv2.circle(overlay, (int(u * scale), int(v * scale)), 3, _colour(distance, max_distance), -1)
    cv2.putText(overlay, "red=depth obstacle, dots=LiDAR", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if depth is None:
        depth_tile = np.zeros((size, width, 3), dtype=np.uint8)
    else:
        level = np.clip(255.0 * (1.0 - np.nan_to_num(depth, nan=max_distance, posinf=max_distance) / max_distance), 0, 255).astype(np.uint8)
        depth_tile = cv2.resize(cv2.applyColorMap(level, cv2.COLORMAP_JET), (width, size))
    cv2.putText(depth_tile, f"metric depth: red=0 m, blue>={max_distance:.1f} m", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    top = np.full((size, size, 3), 255, dtype=np.uint8)
    pixels_per_m = size / (2.0 * max_distance)
    origin = (size // 2, size - int(0.25 * size))

    def to_top(xy: np.ndarray) -> np.ndarray:
        return np.stack([origin[0] - xy[:, 1] * pixels_per_m, origin[1] - xy[:, 0] * pixels_per_m], axis=1).astype(int)

    for metres in np.arange(0.5, 2.0 * max_distance, 0.5):
        cv2.circle(top, origin, int(metres * pixels_per_m), (230, 230, 230), 1)
    for x, y in to_top(np.asarray(lidar_xy).reshape(-1, 2)):
        cv2.circle(top, (int(x), int(y)), 1, (120, 120, 120), -1)
    for x, y in to_top(np.asarray(depth_xy).reshape(-1, 2)):
        cv2.circle(top, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.circle(top, origin, int(robot_radius * pixels_per_m), (0, 0, 0), 1)
    cv2.circle(top, origin, int(body_radius * pixels_per_m), (0, 160, 0), 1)
    cv2.line(top, origin, (origin[0], origin[1] - int(robot_radius * pixels_per_m)), (0, 0, 0), 1)
    cv2.putText(top, "top view (rings 0.5 m)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    text = np.full((size, 300, 3), 255, dtype=np.uint8)
    for row, line in enumerate(lines):
        cv2.putText(text, str(line), (5, 20 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return np.hstack([overlay, depth_tile, top, text])


def base_to_odom(points_xy: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
    """Transform base-frame (x, y) points to the odometry frame given the robot pose (x, y, yaw)."""
    c, s = cos(pose[2]), sin(pose[2])
    points_xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    return points_xy @ np.array([[c, s], [-s, c]]) + np.array(pose[:2])


def odom_to_base(points_xy: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
    """Transform odometry-frame (x, y) points to the base frame given the robot pose (x, y, yaw)."""
    c, s = cos(pose[2]), sin(pose[2])
    points_xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    return (points_xy - np.array(pose[:2])) @ np.array([[c, -s], [s, c]])


@dataclass
class ObstacleMemory:
    """Short-term memory of obstacle points in the odometry frame.

    Points are kept on a ``cell_m`` grid (newest observation per cell) for ``max_age_s`` seconds,
    so obstacles seen from afar stay known after they leave the camera view.
    """

    cell_m: float
    max_age_s: float
    max_range_m: float
    cells: dict = None

    def __post_init__(self) -> None:
        self.cells = {}

    def add(self, points_odom: np.ndarray, now_s: float) -> None:
        """Insert points observed at ``now_s``."""
        for x, y in np.asarray(points_odom, dtype=float).reshape(-1, 2):
            self.cells[(int(np.floor(x / self.cell_m)), int(np.floor(y / self.cell_m)))] = (x, y, now_s)

    def points(self, now_s: float, centre: Tuple[float, float]) -> np.ndarray:
        """Drop expired or distant cells and return the remaining points in the odometry frame."""
        keep = {
            key: value for key, value in self.cells.items()
            if now_s - value[2] <= self.max_age_s and np.hypot(value[0] - centre[0], value[1] - centre[1]) <= self.max_range_m
        }
        self.cells = keep
        return np.array([value[:2] for value in keep.values()], dtype=float).reshape(-1, 2)
