"""ROS 2 node that fuses RPLiDAR scans and OAK-D depth images."""

from __future__ import annotations

from math import atan2, cos, floor, hypot, inf, isfinite, pi, sin

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .geometry import cluster_scan, rdp_segments


class ObstacleFusion(Node):
    """Publish a conservative 2D obstacle scan in the robot base frame."""

    def __init__(self) -> None:
        super().__init__("obstacle_fusion")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("depth_topic", "/stereo/depth")
        self.declare_parameter("camera_info_topic", "/stereo/camera_info")
        self.declare_parameter("output_topic", "/fused_scan")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("angle_bin_degrees", 1.0)
        self.declare_parameter("depth_min_m", 0.4)
        self.declare_parameter("depth_max_m", 4.0)
        self.declare_parameter("obstacle_min_height_m", 0.03)
        self.declare_parameter("obstacle_max_height_m", 0.40)
        self.declare_parameter("pixel_stride", 4)
        self.declare_parameter("min_depth_points_per_bin", 3)
        self.declare_parameter("max_depth_age_s", 0.30)
        self.declare_parameter("breakpoint_base_m", 0.03)
        self.declare_parameter("breakpoint_scale", 2.25)
        self.declare_parameter("minimum_cluster_points", 4)
        self.declare_parameter("rdp_epsilon_m", 0.04)

        self.base_frame = self.get_parameter("base_frame").value
        self.bin_width = float(self.get_parameter("angle_bin_degrees").value) * pi / 180.0
        self.bin_count = round(2.0 * pi / self.bin_width)
        self.camera_matrix: tuple[float, float, float, float] | None = None
        self.latest_depth: list[float] | None = None
        self.latest_depth_stamp = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._scan_callback, 10)
        self.create_subscription(Image, self.get_parameter("depth_topic").value, self._depth_callback, 5)
        self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value, self._camera_info_callback, 1)
        self.scan_pub = self.create_publisher(LaserScan, self.get_parameter("output_topic").value, 10)
        self.segment_pub = self.create_publisher(MarkerArray, "/fused_segments", 10)

    def _camera_info_callback(self, message: CameraInfo) -> None:
        fx, fy, cx, cy = message.k[0], message.k[4], message.k[2], message.k[5]
        if fx > 0.0 and fy > 0.0:
            self.camera_matrix = (fx, fy, cx, cy)

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

    @staticmethod
    def _depth_array(message: Image) -> tuple[np.ndarray, float] | None:
        encoding = message.encoding.upper()
        if encoding in {"16UC1", "MONO16"}:
            dtype, scale = (">u2" if message.is_bigendian else "<u2"), 0.001
        elif encoding == "32FC1":
            dtype, scale = (">f4" if message.is_bigendian else "<f4"), 1.0
        else:
            return None
        elements_per_row = message.step // np.dtype(dtype).itemsize
        image = np.frombuffer(message.data, dtype=dtype).reshape(message.height, elements_per_row)
        return image[:, : message.width], scale

    def _depth_callback(self, message: Image) -> None:
        if self.camera_matrix is None:
            self.get_logger().warn("Waiting for CameraInfo before using depth images.", throttle_duration_sec=5.0)
            return
        decoded = self._depth_array(message)
        if decoded is None:
            self.get_logger().warn(f"Unsupported depth encoding: {message.encoding}", throttle_duration_sec=5.0)
            return
        transform = self._transform(message.header.frame_id, rclpy.time.Time.from_msg(message.header.stamp))
        if transform is None:
            return
        image, scale = decoded
        fx, fy, cx, cy = self.camera_matrix
        stride = int(self.get_parameter("pixel_stride").value)
        depth_min = float(self.get_parameter("depth_min_m").value)
        depth_max = float(self.get_parameter("depth_max_m").value)
        height_min = float(self.get_parameter("obstacle_min_height_m").value)
        height_max = float(self.get_parameter("obstacle_max_height_m").value)
        per_bin: list[list[float]] = [[] for _ in range(self.bin_count)]
        for v in range(0, message.height, stride):
            for u in range(0, message.width, stride):
                depth = float(image[v, u]) * scale
                if not isfinite(depth) or not depth_min <= depth <= depth_max:
                    continue
                x_base, y_base, z_base = self._apply_transform((u - cx) * depth / fx, (v - cy) * depth / fy, depth, transform)
                if x_base > 0.0 and height_min < z_base < height_max:
                    per_bin[self._bin_index(atan2(y_base, x_base))].append(hypot(x_base, y_base))
        minimum_points = int(self.get_parameter("min_depth_points_per_bin").value)
        self.latest_depth = [sorted(values)[max(0, int(0.1 * (len(values) - 1)))] if len(values) >= minimum_points else inf for values in per_bin]
        self.latest_depth_stamp = rclpy.time.Time.from_msg(message.header.stamp)

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
            x_base, y_base, _ = self._apply_transform(distance * cos(angle), distance * sin(angle), 0.0, transform)
            horizontal_distance = hypot(x_base, y_base)
            bin_index = self._bin_index(atan2(y_base, x_base))
            ranges[bin_index] = min(ranges[bin_index], horizontal_distance)
            points.append((x_base, y_base))
            valid_ranges.append(horizontal_distance)
        clusters = cluster_scan(points, valid_ranges, scan.angle_increment, float(self.get_parameter("breakpoint_base_m").value), float(self.get_parameter("breakpoint_scale").value), int(self.get_parameter("minimum_cluster_points").value))
        self._publish_segments(clusters, scan.header)
        if self.latest_depth is not None and self.latest_depth_stamp is not None:
            max_age = float(self.get_parameter("max_depth_age_s").value)
            if abs((stamp - self.latest_depth_stamp).nanoseconds) <= max_age * 1e9:
                ranges = [min(lidar, depth) for lidar, depth in zip(ranges, self.latest_depth)]
        output = LaserScan()
        output.header.stamp = scan.header.stamp
        output.header.frame_id = self.base_frame
        output.angle_min = -pi
        output.angle_max = pi - self.bin_width
        output.angle_increment = self.bin_width
        output.range_min = 0.0
        output.range_max = scan.range_max
        output.ranges = ranges
        self.scan_pub.publish(output)

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
