"""ROS 2 node that drives the TurtleBot with the scaled RatInABox motion model and records it."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime
from math import atan2, cos, isfinite, pi, sin
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from .geometry import points_to_segments
from .rat_motion import BEHAVIOURS, RatMotion, scaled_params

IDLE, RUN, BACKUP, TURN = "IDLE", "RUN", "BACKUP", "TURN"
HAZARD_BUMP = 1
BUMP_ANGLES_DEG = {"bump_left": 70.0, "bump_front_left": 35.0, "bump_front_center": 0.0, "bump_front_right": -35.0, "bump_right": -70.0}
CSV_HEADER = ["row", "t", "behaviour", "x", "y", "yaw", "cmd_linear", "cmd_angular", "v_net_x", "v_net_y", "rot_velocity", "closest_wall"]


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + pi) % (2.0 * pi) - pi


def scan_to_segments(scan: LaserScan, base_m: float, scale: float, epsilon_m: float) -> np.ndarray:
    """Convert a scan to obstacle line segments in the scan frame."""
    points, ranges = [], []
    for index, distance in enumerate(scan.ranges):
        if isfinite(distance) and scan.range_min <= distance <= scan.range_max:
            angle = scan.angle_min + index * scan.angle_increment
            points.append((distance * cos(angle), distance * sin(angle)))
            ranges.append(distance)
    segments = points_to_segments(points, ranges, scan.angle_increment, base_m, scale, epsilon_m)
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
        self.declare_parameter("marker_topic", "/rat_walker/markers")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("length_scale", 8.161)
        self.declare_parameter("time_scale", 7.248)
        self.declare_parameter("robot_radius_m", 0.1705)
        self.declare_parameter("control_period_k", 0.02)
        self.declare_parameter("save_period_k", 0.1)
        self.declare_parameter("record_duration_s", 5218.6)
        self.declare_parameter("max_linear_m_s", 0.31)
        self.declare_parameter("max_angular_rad_s", 1.9)
        self.declare_parameter("turn_tolerance_deg", 3.0)
        self.declare_parameter("heading_gain", 2.0)
        self.declare_parameter("collision_margin_m", 0.05)
        self.declare_parameter("merge_wall_chains", True)
        self.declare_parameter("sensor_timeout_s", 0.5)
        self.declare_parameter("use_bumper", True)
        self.declare_parameter("hazard_topic", "/hazard_detection")
        self.declare_parameter("backup_distance_m", 0.05)
        self.declare_parameter("backup_speed_m_s", 0.05)
        self.declare_parameter("breakpoint_base_m", 0.03)
        self.declare_parameter("breakpoint_scale", 2.25)
        self.declare_parameter("rdp_epsilon_m", 0.04)
        self.declare_parameter("log_dir", str(Path.home() / "rat_walker_logs"))
        self.declare_parameter("seed", -1)
        self.declare_parameter("record_images", False)
        self.declare_parameter("sample_topic", "/rat_walker/sample")
        self.declare_parameter("recorder_status_topic", "/rat_recorder/status")
        self.declare_parameter("recorder_timeout_s", 0.5)

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
        self.heading_gain = float(value("heading_gain"))
        self.collision_margin = float(value("collision_margin_m"))
        self.merge_wall_chains = bool(value("merge_wall_chains"))
        self.sensor_timeout = float(value("sensor_timeout_s"))
        self.use_bumper = bool(value("use_bumper"))
        self.backup_distance = float(value("backup_distance_m"))
        self.backup_speed = float(value("backup_speed_m_s"))
        self.segment_args = (float(value("breakpoint_base_m")), float(value("breakpoint_scale")), float(value("rdp_epsilon_m")))
        self.log_dir = Path(str(value("log_dir"))).expanduser()
        seed = int(value("seed"))
        self.rng = np.random.default_rng(None if seed < 0 else seed)
        self.record_images = bool(value("record_images"))
        self.recorder_timeout = float(value("recorder_timeout_s"))

        self.state = IDLE
        self.behaviour: Optional[str] = None
        self.pending: Optional[str] = None
        self.pending_repel: Optional[float] = None
        self.model: Optional[RatMotion] = None
        self.record_steps = 0
        self.log_file = None
        self.writer = None
        self.walls = np.zeros((0, 2, 2))
        self.scan_time = None
        self.pose: Optional[List[float]] = None
        self.odom_time = None
        self.hazard_time = None
        self.bump: Optional[str] = None
        self.backup_start: Optional[List[float]] = None
        self.backup_since = None
        self.session: Optional[str] = None
        self.recorder_status = "no status received"
        self.recorder_time = None

        self.cmd_pub = self.create_publisher(Twist, value("cmd_vel_topic"), 10)
        self.status_pub = self.create_publisher(String, value("status_topic"), 10)
        self.marker_pub = self.create_publisher(MarkerArray, value("marker_topic"), 10)
        self.base_frame = str(value("base_frame"))
        self.create_subscription(LaserScan, value("scan_topic"), self._scan_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, value("odom_topic"), self._odom_callback, qos_profile_sensor_data)
        self.create_subscription(String, value("behaviour_topic"), self._behaviour_callback, 10)
        if self.use_bumper:
            from irobot_create_msgs.msg import HazardDetectionVector

            self.create_subscription(HazardDetectionVector, value("hazard_topic"), self._hazard_callback, qos_profile_sensor_data)
        if self.record_images:
            self.sample_pub = self.create_publisher(String, value("sample_topic"), 50)
            self.create_subscription(String, value("recorder_status_topic"), self._recorder_callback, 10)
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
        words = message.data.strip().lower().split()
        command = words[0] if words else ""
        repel = None
        if len(words) == 2:
            try:
                repel = float(words[1])
            except ValueError:
                repel = -1.0
        if command == "stop" and len(words) == 1:
            self._end_session("stopped by user")
        elif command in BEHAVIOURS and len(words) <= 2 and (repel is None or repel > 0.0):
            self._end_session(f"switching to {command}")
            self.pending, self.pending_repel = command, repel
        else:
            self.get_logger().warn(f"Unknown behaviour command: {message.data!r}")

    def _hazard_callback(self, message) -> None:
        for detection in message.detections:
            if detection.type == HAZARD_BUMP and self.bump is None:
                self.bump = detection.header.frame_id or "bump_front_center"
        self.hazard_time = self.get_clock().now()

    def _recorder_callback(self, message: String) -> None:
        self.recorder_status = message.data
        self.recorder_time = self.get_clock().now()

    def _recorder_ready(self) -> bool:
        """True when image recording is off, or the recorder reported ``ready`` recently."""
        if not self.record_images:
            return True
        fresh = self.recorder_time is not None and (self.get_clock().now() - self.recorder_time).nanoseconds <= self.recorder_timeout * 1e9
        return fresh and self.recorder_status == "ready"

    def _sensors_fresh(self) -> bool:
        now = self.get_clock().now()
        stamps = (self.scan_time, self.odom_time, self.hazard_time) if self.use_bumper else (self.scan_time, self.odom_time)
        return all(stamp is not None and (now - stamp).nanoseconds <= self.sensor_timeout * 1e9 for stamp in stamps)

    def _start_session(self, behaviour: str, repel: Optional[float] = None) -> None:
        """Start recording ``behaviour``; ``repel`` overrides the scaled wall repulsion distance [m]."""
        params = scaled_params(behaviour, self.length_scale, self.time_scale)
        if repel is not None:
            params = replace(params, wall_repel_distance=repel)
        self.model = RatMotion(params, self.radius, self.pose[2], self.rng, self.collision_margin, self.merge_wall_chains)
        self.behaviour, self.state = behaviour, RUN
        self.bump = None
        self.record_steps = 0
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{behaviour}_repel{params.wall_repel_distance:.3f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.session = path.stem
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
        self._publish_markers()
        self.status_pub.publish(String(data=f"state={self.state} behaviour={self.behaviour} t={self.record_time:.1f}/{self.duration:.0f}s"))

    def _tick_state(self) -> None:
        if not self._sensors_fresh():
            if self.state != IDLE or self.pending is not None:
                self.get_logger().warn("Scan, odometry or hazard input is stale; holding still.", throttle_duration_sec=2.0)
            self._publish(0.0, 0.0)
            return
        if not self._recorder_ready():
            if self.state != IDLE or self.pending is not None:
                self.get_logger().warn(f"Image recorder not ready ({self.recorder_status}); holding still.", throttle_duration_sec=2.0)
            self._publish(0.0, 0.0)
            return
        if self.pending is not None:
            self._start_session(self.pending, self.pending_repel)
            self.pending, self.pending_repel = None, None
        if self.state == IDLE:
            return
        yaw = self.pose[2]
        c, s = cos(yaw), sin(yaw)
        walls = self.walls @ np.array([[c, s], [-s, c]])
        if self.bump is not None and self.state == RUN:
            self._start_backup()
        elif self.bump is not None and self.state != BACKUP:
            self.bump = None
        if self.state == BACKUP:
            self._backup(walls)
        elif self.state == TURN:
            self._turn(walls)
        else:
            self._run(walls)

    def _start_backup(self) -> None:
        """Stop recording, remember where the bumper touched and back up before turning away."""
        self._publish(0.0, 0.0)
        self.get_logger().info(f"Bumper hit ({self.bump}) at t={self.record_time:.2f}s; backing up {self.backup_distance:.2f} m.")
        self.state = BACKUP
        self.backup_start = list(self.pose)
        self.backup_since = self.get_clock().now()

    def _backup(self, walls: np.ndarray) -> None:
        travelled = float(np.hypot(self.pose[0] - self.backup_start[0], self.pose[1] - self.backup_start[1]))
        heading = np.array([cos(self.pose[2]), sin(self.pose[2])])
        rear_blocked = self.model.colliding_wall(walls, -heading * self.backup_speed * self.dt) is not None
        timed_out = (self.get_clock().now() - self.backup_since).nanoseconds > 3.0 * self.backup_distance / self.backup_speed * 1e9
        if travelled < self.backup_distance and not rear_blocked and not timed_out:
            self._publish(-self.backup_speed, 0.0)
            return
        self._publish(0.0, 0.0)
        angle = self.pose[2] + BUMP_ANGLES_DEG.get(self.bump, 0.0) * pi / 180.0
        self.model.bounce_from_point(self.radius * np.array([cos(angle), sin(angle)]))
        self.get_logger().info(f"Backed up {travelled:.3f} m; turning away from the {self.bump} contact.")
        self.bump = None
        self.state = TURN

    def _turn(self, walls: np.ndarray) -> None:
        error = wrap_angle(atan2(self.model.velocity[1], self.model.velocity[0]) - self.pose[2])
        if abs(error) > self.turn_tolerance:
            self._publish(0.0, float(np.clip(self.heading_gain * error, -self.max_angular, self.max_angular)))
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
        angular = float(np.clip(self.model.rotational_velocity + self.heading_gain * error, -self.max_angular, self.max_angular))
        linear = float(min(np.linalg.norm(velocity) * max(cos(error), 0.0), self.max_linear))
        self._publish(linear, angular)
        if self.record_steps % self.save_every == 0:
            x, y, yaw = self.pose
            row = self.record_steps // self.save_every
            if self.record_images:
                self.sample_pub.publish(String(data=f"{self.session},{row},{self.record_time:.4f}"))
            self.writer.writerow([row, f"{self.record_time:.4f}", self.behaviour, x, y, yaw, linear, angular, velocity[0], velocity[1], self.model.rotational_velocity, closest_wall])
            self.log_file.flush()
        self.record_steps += 1
        if self.record_steps >= self.total_steps:
            self._end_session("duration reached")

    def _publish_markers(self) -> None:
        """RViz markers in the base frame: walls used by the model (green), repulsion range (circle), net velocity (arrow)."""
        stamp = self.get_clock().now().to_msg()

        def marker(namespace: str, kind: int) -> Marker:
            item = Marker()
            item.header.stamp, item.header.frame_id = stamp, self.base_frame
            item.ns, item.id, item.type, item.action = namespace, 0, kind, Marker.ADD
            item.pose.orientation.w = 1.0
            return item

        walls = marker("walls", Marker.LINE_LIST)
        walls.scale.x = 0.015
        walls.color = ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0)
        for start, end in self.walls:
            if np.allclose(start, end):
                end = start + np.array([0.01, 0.0])
            walls.points += [Point(x=float(start[0]), y=float(start[1]), z=0.04), Point(x=float(end[0]), y=float(end[1]), z=0.04)]
        markers = [walls]
        if self.model is not None and self.pose is not None:
            reach = marker("repel_range", Marker.LINE_STRIP)
            reach.scale.x = 0.01
            reach.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.6)
            radius = self.radius + self.model.params.wall_repel_distance
            reach.points = [Point(x=radius * cos(a), y=radius * sin(a), z=0.04) for a in np.linspace(0.0, 2.0 * pi, 73)]
            velocity = self.model.velocity
            c, s = cos(self.pose[2]), sin(self.pose[2])
            local = np.array([c * velocity[0] + s * velocity[1], -s * velocity[0] + c * velocity[1]])
            arrow = marker("model_velocity", Marker.ARROW)
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.02, 0.05, 0.05
            arrow.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
            tip = local / max(float(np.linalg.norm(local)), 1e-9) * 0.4
            arrow.points = [Point(x=0.0, y=0.0, z=0.06), Point(x=float(tip[0]), y=float(tip[1]), z=0.06)]
            markers += [reach, arrow]
        self.marker_pub.publish(MarkerArray(markers=markers))

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
