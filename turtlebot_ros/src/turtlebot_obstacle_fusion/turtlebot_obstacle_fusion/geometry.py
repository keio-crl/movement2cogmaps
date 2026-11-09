"""Pure geometry routines used by the ROS obstacle fusion node."""

from __future__ import annotations

from math import hypot, sin
from typing import Sequence, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def adaptive_break_threshold(range_m: float, angle_increment_rad: float, base_m: float, scale: float) -> float:
    """Return a distance-sensitive breakpoint threshold for adjacent laser beams."""
    return base_m + scale * 2.0 * range_m * sin(abs(angle_increment_rad) / 2.0)


def cluster_scan(points: Sequence[Point], ranges: Sequence[float], angle_increment_rad: float, base_m: float, scale: float, min_points: int) -> list[list[Point]]:
    """Split angularly ordered scan points at adaptive Euclidean gaps."""
    if len(points) != len(ranges):
        raise ValueError("points and ranges must have identical lengths")
    if not points:
        return []
    clusters: list[list[Point]] = [[points[0]]]
    for previous, current, current_range in zip(points, points[1:], ranges[1:]):
        gap = hypot(current[0] - previous[0], current[1] - previous[1])
        if gap > adaptive_break_threshold(current_range, angle_increment_rad, base_m, scale):
            clusters.append([current])
        else:
            clusters[-1].append(current)
    return [cluster for cluster in clusters if len(cluster) >= min_points]


def _point_line_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    if length == 0.0:
        return hypot(point[0] - start[0], point[1] - start[1])
    return abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / length


def rdp_segments(points: Sequence[Point], epsilon_m: float) -> list[Segment]:
    """Approximate an ordered point sequence as Ramer-Douglas-Peucker line segments."""
    if len(points) < 2:
        return []
    start, end = points[0], points[-1]
    max_index, max_distance = 0, -1.0
    for index, point in enumerate(points[1:-1], start=1):
        distance = _point_line_distance(point, start, end)
        if distance > max_distance:
            max_index, max_distance = index, distance
    if max_distance <= epsilon_m:
        return [(start, end)]
    return rdp_segments(points[: max_index + 1], epsilon_m) + rdp_segments(points[max_index:], epsilon_m)
