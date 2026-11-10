"""ROS 2 node that drives the TurtleBot with the scaled RatInABox motion model and records it."""

from __future__ import annotations

import csv
from datetime import datetime
from math import atan2, cos, isfinite, pi, sin
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from .geometry import cluster_scan, rdp_segments
from .rat_motion import BEHAVIOURS, RatMotion, scaled_params

IDLE, RUN, TURN = "IDLE", "RUN", "TURN"
CSV_HEADER = ["t", "behaviour", "x", "y", "yaw", "cmd_linear", "cmd_angular", "v_net_x", "v_net_y", "rot_velocity", "closest_wall"]


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + pi) % (2.0 * pi) - pi


def scan_to_segments(scan: LaserScan, base_m: float, scale: float, epsilon_m: float) -> np.ndarray:
    """Convert a scan to obstacle line segments in the scan frame.

    Every cluster is kept: clusters of a single point become zero-length segments so that thin
    obstacles such as chair legs are still repelled from and collided with.
    """
    points, ranges = [], []
    for index, distance in enumerate(scan.ranges):
        if isfinite(distance) and scan.range_min <= distance <= scan.range_max:
            angle = scan.angle_min + index * scan.angle_increment
            points.append((distance * cos(angle), distance * sin(angle)))
            ranges.append(distance)
    segments = []
    for cluster in cluster_scan(points, ranges, scan.angle_increment, base_m, scale, 1):
        segments.extend(rdp_segments(cluster, epsilon_m) if len(cluster) > 1 else [(cluster[0], cluster[0])])
    return np.array(segments, dtype=float).reshape(-1, 2, 2)


class RatWalker(Node):
    """Drive with the RatInABox OU model, bounce at obstacles and log the trajectory."""

    def __init__(self) -> None:
        super().__init__("rat_walker")
        self.declare_parameter("scan_topic", "/fused_scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("behaviour_topic", "/rat_behaviour")
        self.declare_parameter("status_topic", "/rat_walker/status")
        self.declare_parameter("length_scale", 8.161)
        self.declare_parameter("time_scale", 7.248)
        self.declare_parameter("robot_radius_m", 0.1705)
        self.declare_parameter("control_period_k", 0.02)
        self.declare_parameter("save_period_k", 0.1)
        self.declare_parameter("record_duration_s", 5129.0)
        self.declare_parameter("max_linear_m_s", 0.31)
        self.declare_parameter("max_angular_rad_s", 1.9)
        self.declare_parameter("turn_tolerance_deg", 3.0)
        self.declare_parameter("collision_margin_m", 0.05)
        self.declare_parameter("sensor_timeout_s", 0.5)
        self.declare_parameter("breakpoint_base_m", 0.03)
        self.declare_parameter("breakpoint_scale", 2.25)
        self.declare_parameter("rdp_epsilon_m", 0.04)
        self.declare_parameter("log_dir", str(Path.home() / "rat_walker_logs"))
        self.declare_parameter("seed", -1)

        value = lambda name: self.get_parameter(name).value  # noqa: E731
        self.length_scale = float(value("length_scale"))
        self.time_scale = float(value("time_scale"))
        self.radius = float(value("robot_radius_m"))
        self.dt = float(value("control_period_k")) * self.time_scale
        self.save_every = max(1, round(float(value("save_period_k")) / float(value("control_period_k"))))
        self.duration = float(value("record_duration_s"))
        self.total_steps = int(round(self.duration / self.dt))
        self.max_linear = float(value("max_linear_m_s"))
        self.max_angular = float(value("max_angular_rad_s"))
        self.turn_tolerance = float(value("turn_tolerance_deg")) * pi / 180.0
        self.collision_margin = float(value("collision_margin_m"))
        self.sensor_timeout = float(value("sensor_timeout_s"))
        self.segment_args = (float(value("breakpoint_base_m")), float(value("breakpoint_scale")), float(value("rdp_epsilon_m")))
        self.log_dir = Path(str(value("log_dir"))).expanduser()
        seed = int(value("seed"))
        self.rng = np.random.default_rng(None if seed < 0 else seed)

        self.state = IDLE
        self.behaviour: Optional[str] = None
        self.pending: Optional[str] = None
        self.model: Optional[RatMotion] = None
        self.record_steps = 0
        self.log_file = None
        self.writer = None
        self.walls = np.zeros((0, 2, 2))
        self.scan_time = None
        self.pose: Optional[List[float]] = None
        self.odom_time = None

        self.cmd_pub = self.create_publisher(Twist, value("cmd_vel_topic"), 10)
        self.status_pub = self.create_publisher(String, value("status_topic"), 10)
        self.create_subscription(LaserScan, value("scan_topic"), self._scan_callback, 10)
        self.create_subscription(Odometry, value("odom_topic"), self._odom_callback, 10)
        self.create_subscription(String, value("behaviour_topic"), self._behaviour_callback, 10)
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(f"dt={self.dt:.5f}s save={self.save_every * self.dt:.4f}s duration={self.duration}s; waiting for a behaviour.")

    def _scan_callback(self, scan: LaserScan) -> None:
        self.walls = scan_to_segments(scan, *self.segment_args)
        self.scan_time = self.get_clock().now()

    def _odom_callback(self, odom: Odometry) -> None:
        q = odom.pose.pose.orientation
        yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = [odom.pose.pose.position.x, odom.pose.pose.position.y, yaw]
        self.odom_time = self.get_clock().now()

    def _behaviour_callback(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == "stop":
            self._end_session("stopped by user")
        elif command in BEHAVIOURS:
            self._end_session(f"switching to {command}")
            self.pending = command
        else:
            self.get_logger().warn(f"Unknown behaviour command: {message.data!r}")

    def _sensors_fresh(self) -> bool:
        now = self.get_clock().now()
        return all(stamp is not None and (now - stamp).nanoseconds <= self.sensor_timeout * 1e9 for stamp in (self.scan_time, self.odom_time))

    def _start_session(self, behaviour: str) -> None:
        params = scaled_params(behaviour, self.length_scale, self.time_scale)
        self.model = RatMotion(params, self.radius, self.pose[2], self.rng, self.collision_margin)
        self.behaviour, self.state = behaviour, RUN
        self.record_steps = 0
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{behaviour}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_file = open(path, "w", newline="")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(CSV_HEADER)
        self.get_logger().info(f"Started {behaviour}: {params}; logging to {path}")

    def _end_session(self, reason: str) -> None:
        self._publish(0.0, 0.0)
        if self.log_file is not None:
            self.log_file.close()
            self.get_logger().info(f"Ended {self.behaviour} at t={self.record_time:.2f}s ({reason}).")
        self.log_file, self.writer, self.model = None, None, None
        self.behaviour, self.state = None, IDLE

    @property
    def record_time(self) -> float:
        """Recorded (moving) time of the current session in seconds."""
        return self.record_steps * self.dt

    def _publish(self, linear: float, angular: float) -> None:
        twist = Twist()
        twist.linear.x, twist.angular.z = linear, angular
        self.cmd_pub.publish(twist)

    def _tick(self) -> None:
        self._tick_state()
        self.status_pub.publish(String(data=f"state={self.state} behaviour={self.behaviour} t={self.record_time:.1f}/{self.duration:.0f}s"))

    def _tick_state(self) -> None:
        if not self._sensors_fresh():
            if self.state != IDLE or self.pending is not None:
                self.get_logger().warn("Scan or odometry is stale; holding still.", throttle_duration_sec=2.0)
            self._publish(0.0, 0.0)
            return
        if self.pending is not None:
            self._start_session(self.pending)
            self.pending = None
        if self.state == IDLE:
            return
        yaw = self.pose[2]
        c, s = cos(yaw), sin(yaw)
        walls = self.walls @ np.array([[c, s], [-s, c]])
        if self.state == TURN:
            self._turn(walls)
        else:
            self._run(walls)

    def _turn(self, walls: np.ndarray) -> None:
        error = wrap_angle(atan2(self.model.velocity[1], self.model.velocity[0]) - self.pose[2])
        if abs(error) > self.turn_tolerance:
            self._publish(0.0, float(np.clip(error / self.dt, -self.max_angular, self.max_angular)))
            return
        wall = self.model.colliding_wall(walls, self.model.velocity * self.dt)
        if wall is not None:
            self.model.bounce(wall)
            return
        self.state = RUN
        self._drive(self.model.velocity, float("nan"))

    def _run(self, walls: np.ndarray) -> None:
        result = self.model.step(walls, self.dt)
        if result.collision_wall is not None:
            self._publish(0.0, 0.0)
            self.model.bounce(result.collision_wall)
            self.state = TURN
            self.get_logger().info(f"Collision ahead at t={self.record_time:.2f}s; turning to the mirrored heading.")
            return
        self._drive(result.net_velocity, result.closest_wall_distance)

    def _drive(self, velocity: np.ndarray, closest_wall: float) -> None:
        error = wrap_angle(atan2(velocity[1], velocity[0]) - self.pose[2])
        angular = float(np.clip(error / self.dt, -self.max_angular, self.max_angular))
        linear = float(min(np.linalg.norm(velocity) * max(cos(error), 0.0), self.max_linear))
        self._publish(linear, angular)
        if self.record_steps % self.save_every == 0:
            x, y, yaw = self.pose
            self.writer.writerow([f"{self.record_time:.4f}", self.behaviour, x, y, yaw, linear, angular, velocity[0], velocity[1], self.model.rotational_velocity, closest_wall])
            self.log_file.flush()
        self.record_steps += 1
        if self.record_steps >= self.total_steps:
            self._end_session("duration reached")

    def shutdown(self) -> None:
        """Stop the robot and close the log."""
        self._end_session("shutdown")


def main() -> None:
    rclpy.init()
    node = RatWalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()
