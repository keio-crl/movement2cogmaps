"""Fuse the LiDAR scan with Depth Anything V2 obstacles into one 360-bin scan for rat_walker.

For every 1-degree bin the nearer of the LiDAR and the depth obstacles is kept. Depth obstacles are
remembered in the odometry frame for ``depth_memory_s`` so that low obstacles stay known after they
leave the camera view (the camera cannot see the floor right in front of the robot). Depth
received while the robot turned faster than ``depth_max_turn_rate`` within the last
``depth_motion_window_s`` is discarded: motion blur makes Depth Anything V2 report phantom
obstacles next to the robot.
"""

from __future__ import annotations

from math import atan2, cos, isfinite, pi, sin
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .depth_estimation import ObstacleMemory, base_to_odom, bin_ranges, odom_to_base, ranges_to_points


def fuse_ranges(lidar_xy: np.ndarray, depth_xy: np.ndarray, bin_count: int) -> np.ndarray:
    """Per-bin minimum range over [-pi, pi) of the LiDAR and depth points (inf where neither has one)."""
    return np.minimum(bin_ranges(lidar_xy, bin_count, 1, 0.0), bin_ranges(depth_xy, bin_count, 1, 0.0))


class ScanDepthFusion(Node):
    """Publish ``/fused_scan`` = min(LiDAR, remembered depth obstacles) per angle in the base frame."""

    def __init__(self) -> None:
        super().__init__("scan_depth_fusion")
        defaults = {
            "scan_topic": "/scan",
            "depth_topic": "/depth_obstacles",
            "odom_topic": "/odom",
            "output_topic": "/fused_scan",
            "base_frame": "base_link",
            "angle_bin_degrees": 1.0,
            "depth_timeout_s": 0.6,
            "depth_memory_s": 8.0,
            "depth_memory_cell_m": 0.03,
            "depth_memory_range_m": 3.0,
            "depth_max_turn_rate": 0.3,
            "depth_motion_window_s": 0.6,
            "marker_topic": "/scan_depth_fusion/markers",
            "robot_radius_m": 0.1705,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        self.bin_count = round(360.0 / float(self.p["angle_bin_degrees"]))
        self.memory = ObstacleMemory(float(self.p["depth_memory_cell_m"]), float(self.p["depth_memory_s"]), float(self.p["depth_memory_range_m"]))
        self.pose: Optional[tuple] = None
        self.depth_time: Optional[Time] = None
        self.last_fast_turn: Optional[Time] = None
        self.latest_depth_odom = np.zeros((0, 2))
        self.depth_discarded = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(LaserScan, self.p["output_topic"], 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.p["marker_topic"], 10)
        self.create_subscription(LaserScan, self.p["scan_topic"], self._scan_callback, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.p["depth_topic"], self._depth_callback, 10)
        self.create_subscription(Odometry, self.p["odom_topic"], self._odom_callback, qos_profile_sensor_data)

    def _odom_callback(self, odom: Odometry) -> None:
        q = odom.pose.pose.orientation
        yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (odom.pose.pose.position.x, odom.pose.pose.position.y, yaw)
        if abs(odom.twist.twist.angular.z) > float(self.p["depth_max_turn_rate"]):
            self.last_fast_turn = self.get_clock().now()

    def _depth_callback(self, scan: LaserScan) -> None:
        if self.pose is None:
            return
        now = self.get_clock().now()
        self.depth_discarded = self.last_fast_turn is not None and (now - self.last_fast_turn).nanoseconds <= float(self.p["depth_motion_window_s"]) * 1e9
        if self.depth_discarded:
            return
        points = ranges_to_points(np.array(scan.ranges, dtype=float), scan.angle_min, scan.angle_increment)
        self.depth_time = now
        self.latest_depth_odom = base_to_odom(points, self.pose)
        self.memory.add(self.latest_depth_odom, self.depth_time.nanoseconds * 1e-9)

    def _scan_callback(self, scan: LaserScan) -> None:
        if self.pose is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(self.p["base_frame"], scan.header.frame_id, Time(), timeout=Duration(seconds=0.05))
        except TransformException as error:
            self.get_logger().warn(f"Cannot transform {scan.header.frame_id} to {self.p['base_frame']}: {error}", throttle_duration_sec=2.0)
            return
        q = transform.transform.rotation
        offset = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        tx, ty = transform.transform.translation.x, transform.transform.translation.y
        lidar = [
            (tx + distance * cos(offset + scan.angle_min + index * scan.angle_increment), ty + distance * sin(offset + scan.angle_min + index * scan.angle_increment))
            for index, distance in enumerate(scan.ranges)
            if isfinite(distance) and scan.range_min <= distance <= scan.range_max
        ]
        now = self.get_clock().now()
        if self.depth_time is None or (now - self.depth_time).nanoseconds > float(self.p["depth_timeout_s"]) * 1e9:
            self.get_logger().warn("Depth is stale or discarded while turning; using LiDAR and remembered depth obstacles only.", throttle_duration_sec=5.0)
        remembered, ages = self.memory.entries(now.nanoseconds * 1e-9, self.pose[:2])
        depth = odom_to_base(remembered, self.pose)
        output = LaserScan()
        output.header.stamp = scan.header.stamp
        output.header.frame_id = self.p["base_frame"]
        width = 2.0 * pi / self.bin_count
        output.angle_min, output.angle_max, output.angle_increment = -pi + width / 2.0, pi - width / 2.0, width
        output.range_min, output.range_max = 0.0, scan.range_max
        output.ranges = [float(value) for value in fuse_ranges(np.array(lidar, dtype=float).reshape(-1, 2), depth, self.bin_count)]
        self.publisher.publish(output)
        depth_fresh = self.depth_time is not None and (now - self.depth_time).nanoseconds <= float(self.p["depth_timeout_s"]) * 1e9
        self._publish_markers(scan.header.stamp, depth, ages, odom_to_base(self.latest_depth_odom, self.pose) if depth_fresh else np.zeros((0, 2)), depth_fresh)

    def _publish_markers(self, stamp, remembered: np.ndarray, ages: np.ndarray, live: np.ndarray, depth_fresh: bool) -> None:
        """RViz markers in the base frame: remembered depth (orange, fading with age), latest depth (red), robot, status."""
        max_age = float(self.p["depth_memory_s"])

        def marker(namespace: str, kind: int) -> Marker:
            item = Marker()
            item.header.stamp, item.header.frame_id = stamp, self.p["base_frame"]
            item.ns, item.id, item.type, item.action = namespace, 0, kind, Marker.ADD
            item.pose.orientation.w = 1.0
            return item

        memory = marker("depth_memory", Marker.POINTS)
        memory.scale.x = memory.scale.y = 0.03
        memory.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in remembered]
        memory.colors = [ColorRGBA(r=1.0, g=0.55, b=0.0, a=float(max(0.15, 1.0 - age / max_age))) for age in ages]
        latest = marker("depth_latest", Marker.POINTS)
        latest.scale.x = latest.scale.y = 0.04
        latest.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        latest.points = [Point(x=float(x), y=float(y), z=0.03) for x, y in live]
        body = marker("robot", Marker.CYLINDER)
        body.scale.x = body.scale.y = 2.0 * float(self.p["robot_radius_m"])
        body.scale.z = 0.01
        body.color = ColorRGBA(r=0.3, g=0.6, b=1.0, a=0.3)
        heading = marker("heading", Marker.ARROW)
        heading.scale.x, heading.scale.y, heading.scale.z = 0.02, 0.04, 0.04
        heading.color = ColorRGBA(r=0.3, g=0.6, b=1.0, a=1.0)
        heading.points = [Point(x=0.0, y=0.0, z=0.05), Point(x=float(self.p["robot_radius_m"]), y=0.0, z=0.05)]
        status = marker("status", Marker.TEXT_VIEW_FACING)
        status.pose.position.x, status.pose.position.y, status.pose.position.z = -0.45, 0.0, 0.3
        status.scale.z = 0.08
        status.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        state = "discarded (turning)" if self.depth_discarded else ("fresh" if depth_fresh else "STALE")
        oldest = f"{ages.max():.1f}" if len(ages) else "-"
        status.text = f"depth {state} | memory {len(remembered)} pts, oldest {oldest} s"
        self.marker_pub.publish(MarkerArray(markers=[memory, latest, body, heading, status]))


def main() -> None:
    rclpy.init()
    node = ScanDepthFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
