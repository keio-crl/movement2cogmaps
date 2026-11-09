"""ROS 2 node that fuses RPLiDAR and OAK-D obstacle observations."""

from __future__ import annotations

from math import atan2, cos, floor, hypot, inf, isfinite, pi, sin

import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .geometry import cluster_scan, rdp_segments


class ObstacleFusion(Node):
    """Publish a conservative 2D obstacle scan in the robot base frame."""

    def __init__(self) -> None:
        super().__init__("obstacle_fusion")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("depth_cloud_topic", "/oakd/points")
        self.declare_parameter("output_topic", "/fused_scan")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("angle_bin_degrees", 1.0)
        self.declare_parameter("depth_min_m", 0.4)
        self.declare_parameter("depth_max_m", 4.0)
        self.declare_parameter("obstacle_min_height_m", 0.03)
        self.declare_parameter("obstacle_max_height_m", 0.40)
        self.declare_parameter("min_depth_points_per_bin", 4)
        self.declare_parameter("max_depth_age_s", 0.30)
        self.declare_parameter("breakpoint_base_m", 0.03)
        self.declare_parameter("breakpoint_scale", 2.25)
        self.declare_parameter("minimum_cluster_points", 4)
        self.declare_parameter("rdp_epsilon_m", 0.04)

        self.base_frame = self.get_parameter("base_frame").value
        self.bin_width = float(self.get_parameter("angle_bin_degrees").value) * pi / 180.0
        self.bin_count = round(2.0 * pi / self.bin_width)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_depth: list[float] | None = None
        self.latest_depth_stamp = None
        self.scan_sub = self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._scan_callback, 10)
        self.depth_sub = self.create_subscription(PointCloud2, self.get_parameter("depth_cloud_topic").value, self._depth_callback, 5)
        self.scan_pub = self.create_publisher(LaserScan, self.get_parameter("output_topic").value, 10)
        self.segment_pub = self.create_publisher(MarkerArray, "/fused_segments", 10)

    def _transform(self, frame_id: str, stamp):
        try:
            return self.tf_buffer.lookup_transform(self.base_frame, frame_id, stamp, timeout=Duration(seconds=0.05))
        except TransformException as error:
            self.get_logger().warn(f"Cannot transform {frame_id} to {self.base_frame}: {error}", throttle_duration_sec=2.0)
            return None

    @staticmethod
    def _apply_transform(x: float, y: float, z: float, transform) -> tuple[float, float, float]:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)
        return (
            x + qw * tx + (qy * tz - qz * ty) + translation.x,
            y + qw * ty + (qz * tx - qx * tz) + translation.y,
            z + qw * tz + (qx * ty - qy * tx) + translation.z,
        )

    def _bin_index(self, angle: float) -> int:
        return int(floor((angle + pi) / self.bin_width)) % self.bin_count

    def _depth_callback(self, cloud: PointCloud2) -> None:
        transform = self._transform(cloud.header.frame_id, rclpy.time.Time.from_msg(cloud.header.stamp))
        if transform is None:
            return
        depth_min = float(self.get_parameter("depth_min_m").value)
        depth_max = float(self.get_parameter("depth_max_m").value)
        height_min = float(self.get_parameter("obstacle_min_height_m").value)
        height_max = float(self.get_parameter("obstacle_max_height_m").value)
        per_bin: list[list[float]] = [[] for _ in range(self.bin_count)]
        for x, y, z in point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
            x_base, y_base, z_base = self._apply_transform(float(x), float(y), float(z), transform)
            distance = hypot(x_base, y_base)
            if x_base > 0.0 and height_min < z_base < height_max and depth_min <= distance <= depth_max:
                per_bin[self._bin_index(atan2(y_base, x_base))].append(distance)
        minimum_points = int(self.get_parameter("min_depth_points_per_bin").value)
        self.latest_depth = [sorted(values)[max(0, int(0.1 * (len(values) - 1)))] if len(values) >= minimum_points else inf for values in per_bin]
        self.latest_depth_stamp = rclpy.time.Time.from_msg(cloud.header.stamp)

    def _scan_callback(self, scan: LaserScan) -> None:
        stamp = rclpy.time.Time.from_msg(scan.header.stamp)
        transform = self._transform(scan.header.frame_id, stamp)
        if transform is None:
            return
        ranges = [inf] * self.bin_count
        points: list[tuple[float, float]] = []
        valid_ranges: list[float] = []
        for index, distance in enumerate(scan.ranges):
            if not isfinite(distance) or distance < scan.range_min or distance > scan.range_max:
                continue
            angle = scan.angle_min + index * scan.angle_increment
            x, y, _ = self._apply_transform(distance * cos(angle), distance * sin(angle), 0.0, transform)
            horizontal_distance = hypot(x, y)
            bin_index = self._bin_index(atan2(y, x))
            ranges[bin_index] = min(ranges[bin_index], horizontal_distance)
            points.append((x, y))
            valid_ranges.append(horizontal_distance)
        clusters = cluster_scan(points, valid_ranges, scan.angle_increment, float(self.get_parameter("breakpoint_base_m").value), float(self.get_parameter("breakpoint_scale").value), int(self.get_parameter("minimum_cluster_points").value))
        self._publish_segments(clusters, scan.header)
        if self.latest_depth is not None and self.latest_depth_stamp is not None:
            max_age = float(self.get_parameter("max_depth_age_s").value)
            if abs((stamp - self.latest_depth_stamp).nanoseconds) <= max_age * 1e9:
                ranges = [min(lidar, depth) for lidar, depth in zip(ranges, self.latest_depth)]
        message = LaserScan()
        message.header.stamp = scan.header.stamp
        message.header.frame_id = self.base_frame
        message.angle_min = -pi
        message.angle_max = pi - self.bin_width
        message.angle_increment = self.bin_width
        message.range_min = 0.0
        message.range_max = scan.range_max
        message.ranges = ranges
        self.scan_pub.publish(message)

    def _publish_segments(self, clusters: list[list[tuple[float, float]]], header) -> None:
        markers = MarkerArray()
        epsilon = float(self.get_parameter("rdp_epsilon_m").value)
        marker_id = 0
        for cluster in clusters:
            for start, end in rdp_segments(cluster, epsilon):
                marker = Marker()
                marker.header = header
                marker.header.frame_id = self.base_frame
                marker.ns = "lidar_rdp"
                marker.id = marker_id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.02
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 1.0
                marker.points = [Point(x=start[0], y=start[1], z=0.05), Point(x=end[0], y=end[1], z=0.05)]
                markers.markers.append(marker)
                marker_id += 1
        self.segment_pub.publish(markers)


def main() -> None:
    rclpy.init()
    node = ObstacleFusion()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
