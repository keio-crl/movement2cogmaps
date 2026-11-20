"""Real-robot test of LiDAR collision avoidance: drive straight, stop before an obstacle, turn 180 deg.

The obstacle model and collision check are the ones used by ``rat_walker``: scan points are
clustered, approximated by RDP segments and a disc of ``robot_radius_m + collision_margin_m`` is
tested against the distance the robot needs to stop: one control step, the command latency and
the braking distance v^2 / (2 a). Each detection logs the clearance at detection,
the braking distance measured by odometry and the minimum clearance reached, so the margin can be
tuned on the real robot.
"""

from __future__ import annotations

import csv
from datetime import datetime
from math import atan2, cos, hypot, isfinite, pi, sin
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from .depth_estimation import ObstacleMemory, base_to_odom, odom_to_base, ranges_to_points
from .geometry import points_to_segments
from .rat_motion import closest_points, colliding_wall

WAIT, FORWARD, BRAKE, BACKUP, TURN, DONE = "WAIT", "FORWARD", "BRAKE", "BACKUP", "TURN", "DONE"
HAZARD_BUMP = 1
CSV_HEADER = ["time", "state", "x", "y", "yaw", "odom_v", "cmd_linear", "cmd_angular", "clearance", "front_clearance", "depth_clearance", "trigger"]


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + pi) % (2.0 * pi) - pi


class CollisionTest(Node):
    """Bounce back and forth between obstacles using only the LiDAR scan."""

    def __init__(self) -> None:
        super().__init__("collision_test")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("linear_speed_m_s", 0.1)
        self.declare_parameter("angular_speed_rad_s", 0.8)
        self.declare_parameter("max_linear_m_s", 0.31)
        self.declare_parameter("max_angular_rad_s", 1.9)
        self.declare_parameter("control_period_s", 0.14496)
        self.declare_parameter("robot_radius_m", 0.1705)
        self.declare_parameter("collision_margin_m", 0.05)
        self.declare_parameter("latency_s", 0.1)
        self.declare_parameter("braking_decel_m_s2", 1.0)
        self.declare_parameter("turn_tolerance_deg", 3.0)
        self.declare_parameter("stopped_speed_m_s", 0.01)
        self.declare_parameter("sensor_timeout_s", 0.5)
        self.declare_parameter("start_delay_s", 3.0)
        self.declare_parameter("max_turns", 4)
        self.declare_parameter("breakpoint_base_m", 0.03)
        self.declare_parameter("breakpoint_scale", 2.25)
        self.declare_parameter("rdp_epsilon_m", 0.04)
        self.declare_parameter("log_dir", str(Path.home() / "collision_test_logs"))
        self.declare_parameter("use_depth", False)
        self.declare_parameter("depth_topic", "/depth_obstacles")
        self.declare_parameter("depth_timeout_s", 0.6)
        self.declare_parameter("depth_lost_turn_s", 2.0)
        self.declare_parameter("depth_memory_s", 8.0)
        self.declare_parameter("depth_memory_cell_m", 0.03)
        self.declare_parameter("depth_memory_range_m", 3.0)
        self.declare_parameter("use_bumper", False)
        self.declare_parameter("hazard_topic", "/hazard_detection")
        self.declare_parameter("backup_distance_m", 0.05)
        self.declare_parameter("backup_speed_m_s", 0.05)

        value = lambda name: self.get_parameter(name).value  # noqa: E731
        self.base_frame = str(value("base_frame"))
        self.linear_speed = min(float(value("linear_speed_m_s")), float(value("max_linear_m_s")))
        self.angular_speed = min(float(value("angular_speed_rad_s")), float(value("max_angular_rad_s")))
        self.dt = float(value("control_period_s"))
        self.radius = float(value("robot_radius_m"))
        self.body_radius = self.radius + float(value("collision_margin_m"))
        self.turn_tolerance = float(value("turn_tolerance_deg")) * pi / 180.0
        v = self.linear_speed
        self.lookahead = v * (self.dt + float(value("latency_s"))) + v * v / (2.0 * float(value("braking_decel_m_s2")))
        self.stopped_speed = float(value("stopped_speed_m_s"))
        self.sensor_timeout = float(value("sensor_timeout_s"))
        self.start_delay = float(value("start_delay_s"))
        self.max_turns = int(value("max_turns"))
        self.segment_args = (float(value("breakpoint_base_m")), float(value("breakpoint_scale")), float(value("rdp_epsilon_m")))
        self.use_depth = bool(value("use_depth"))
        self.depth_timeout = float(value("depth_timeout_s"))
        self.depth_lost_turn = float(value("depth_lost_turn_s"))
        self.memory = ObstacleMemory(float(value("depth_memory_cell_m")), float(value("depth_memory_s")), float(value("depth_memory_range_m")))
        self.use_bumper = bool(value("use_bumper"))
        self.backup_distance = float(value("backup_distance_m"))
        self.backup_speed = float(value("backup_speed_m_s"))
        self.backup_step = self.backup_speed * (self.dt + float(value("latency_s")))

        self.state = WAIT
        self.ready_since: Optional[Time] = None
        self.walls = np.zeros((0, 2, 2))
        self.points = np.zeros((0, 2))
        self.scan_time: Optional[Time] = None
        self.pose: Optional[list] = None
        self.odom_speed = 0.0
        self.odom_time: Optional[Time] = None
        self.turns = 0
        self.turn_remaining = 0.0
        self.last_yaw = 0.0
        self.detect_pose: Optional[list] = None
        self.detect_clearance = 0.0
        self.min_clearance = float("inf")
        self.depth_points = np.zeros((0, 2))
        self.depth_time: Optional[Time] = None
        self.depth_lost_since: Optional[Time] = None
        self.trigger = ""
        self.hazard_time: Optional[Time] = None
        self.bump: Optional[str] = None
        self.backup_start: Optional[list] = None
        self.backup_since: Optional[Time] = None

        log_dir = Path(str(value("log_dir"))).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"collision_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_file = open(path, "w", newline="")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(CSV_HEADER)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, value("cmd_vel_topic"), 10)
        self.create_subscription(LaserScan, value("scan_topic"), self._scan_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, value("odom_topic"), self._odom_callback, qos_profile_sensor_data)
        if self.use_depth:
            self.create_subscription(LaserScan, value("depth_topic"), self._depth_callback, 10)
        if self.use_bumper:
            from irobot_create_msgs.msg import HazardDetectionVector

            self.create_subscription(HazardDetectionVector, value("hazard_topic"), self._hazard_callback, qos_profile_sensor_data)
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"speed={self.linear_speed} m/s, turn={self.angular_speed} rad/s, body radius={self.body_radius:.3f} m, lookahead={self.lookahead:.3f} m, "
            f"max_turns={self.max_turns}, obstacles from {'LiDAR + depth' if self.use_depth else 'LiDAR only'}, bumper {'on' if self.use_bumper else 'off'}; logging to {path}"
        )

    def _scan_callback(self, scan: LaserScan) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(self.base_frame, scan.header.frame_id, Time(), timeout=Duration(seconds=0.05))
        except TransformException as error:
            self.get_logger().warn(f"Cannot transform {scan.header.frame_id} to {self.base_frame}: {error}", throttle_duration_sec=2.0)
            return
        q = transform.transform.rotation
        offset_yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        tx, ty = transform.transform.translation.x, transform.transform.translation.y
        points, ranges = [], []
        for index, distance in enumerate(scan.ranges):
            if isfinite(distance) and scan.range_min <= distance <= scan.range_max:
                angle = offset_yaw + scan.angle_min + index * scan.angle_increment
                x, y = tx + distance * cos(angle), ty + distance * sin(angle)
                points.append((x, y))
                ranges.append(hypot(x, y))
        segments = points_to_segments(points, ranges, scan.angle_increment, *self.segment_args)
        self.walls = np.array(segments, dtype=float).reshape(-1, 2, 2)
        self.points = np.array(points, dtype=float).reshape(-1, 2)
        self.scan_time = self.get_clock().now()

    def _odom_callback(self, odom: Odometry) -> None:
        q = odom.pose.pose.orientation
        yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = [odom.pose.pose.position.x, odom.pose.pose.position.y, yaw]
        self.odom_speed = hypot(odom.twist.twist.linear.x, odom.twist.twist.linear.y)
        self.odom_time = self.get_clock().now()

    def _depth_callback(self, scan: LaserScan) -> None:
        if self.pose is None:
            return
        points = ranges_to_points(np.array(scan.ranges, dtype=float), scan.angle_min, scan.angle_increment)
        self.depth_time = self.get_clock().now()
        self.memory.add(base_to_odom(points, self.pose), self.depth_time.nanoseconds * 1e-9)

    def _depth_fresh(self) -> bool:
        return self.depth_time is not None and (self.get_clock().now() - self.depth_time).nanoseconds <= self.depth_timeout * 1e9

    def _hazard_callback(self, message) -> None:
        for detection in message.detections:
            if detection.type == HAZARD_BUMP and self.bump is None:
                self.bump = detection.header.frame_id or "bump"
        self.hazard_time = self.get_clock().now()

    def _stale_sensors(self) -> list:
        """Names of the inputs that are missing or older than their timeout."""
        now = self.get_clock().now()
        inputs = [("scan", self.scan_time, self.sensor_timeout), ("odom", self.odom_time, self.sensor_timeout)]
        if self.use_bumper:
            inputs.append(("hazard", self.hazard_time, self.sensor_timeout))
        return [name for name, stamp, timeout in inputs if stamp is None or (now - stamp).nanoseconds > timeout * 1e9]

    def _depth_walls(self) -> np.ndarray:
        """Depth obstacle points as zero-length walls."""
        return np.repeat(self.depth_points[:, None, :], 2, axis=1)

    def _hit(self) -> str:
        """Return which sensor predicts a collision within the lookahead: '', 'lidar', 'depth' or 'both'."""
        displacement = np.array([self.lookahead, 0.0])
        lidar = colliding_wall(self.walls, displacement, self.body_radius) is not None
        depth = self.use_depth and colliding_wall(self._depth_walls(), displacement, self.body_radius) is not None
        return {(True, True): "both", (True, False): "lidar", (False, True): "depth"}.get((lidar, depth), "")

    def _depth_clearance(self) -> float:
        """Distance from the robot surface to the nearest depth obstacle point."""
        if not self.use_depth or len(self.depth_points) == 0:
            return float("inf")
        return float(np.hypot(self.depth_points[:, 0], self.depth_points[:, 1]).min()) - self.radius

    def _clearance(self) -> float:
        """Distance from the robot surface to the nearest obstacle segment."""
        if len(self.walls) == 0:
            return float("inf")
        return float(np.linalg.norm(closest_points(np.zeros(2), self.walls), axis=1).min()) - self.radius

    def _front_clearance(self) -> float:
        """Distance from the robot surface to the nearest point inside the forward corridor."""
        ahead = self.points[(self.points[:, 0] > 0.0) & (np.abs(self.points[:, 1]) < self.radius)] if len(self.points) else self.points
        if len(ahead) == 0:
            return float("inf")
        return float(np.min(ahead[:, 0] - np.sqrt(self.radius ** 2 - ahead[:, 1] ** 2)))

    def _publish(self, linear: float, angular: float) -> None:
        twist = Twist()
        twist.linear.x, twist.angular.z = linear, angular
        self.cmd_pub.publish(twist)
        if self.pose is not None:
            x, y, yaw = self.pose
            self.writer.writerow([f"{self.get_clock().now().nanoseconds * 1e-9:.3f}", self.state, x, y, yaw, self.odom_speed, linear, angular, self._clearance(), self._front_clearance(), self._depth_clearance(), self.trigger])

    def _tick(self) -> None:
        if self.state == DONE:
            self._publish(0.0, 0.0)
            return
        stale = self._stale_sensors()
        if stale:
            self.get_logger().warn(f"Stale input ({', '.join(stale)}); holding still.", throttle_duration_sec=2.0)
            self.ready_since = None
            self._publish(0.0, 0.0)
            return
        if self.use_depth:
            remembered = self.memory.points(self.get_clock().now().nanoseconds * 1e-9, self.pose[:2])
            self.depth_points = odom_to_base(remembered, self.pose)
        clearance = self._clearance()
        self.min_clearance = min(self.min_clearance, clearance)
        if self.bump is not None and self.state in (FORWARD, BRAKE):
            self._start_backup()
        elif self.bump is not None and self.state != BACKUP:
            self.bump = None
        if self.state == WAIT:
            self._wait()
        elif self.state == FORWARD:
            self._forward(clearance)
        elif self.state == BRAKE:
            self._brake(clearance)
        elif self.state == BACKUP:
            self._backup()
        elif self.state == TURN:
            self._turn()

    def _wait(self) -> None:
        now = self.get_clock().now()
        self._publish(0.0, 0.0)
        if self.use_depth and not self._depth_fresh():
            self.ready_since = None
            self.depth_lost_since = self.depth_lost_since or now
            if (now - self.depth_lost_since).nanoseconds <= self.depth_lost_turn * 1e9:
                self.get_logger().warn("Waiting for depth.", throttle_duration_sec=2.0)
                return
            self.get_logger().info(f"No depth for {self.depth_lost_turn:.1f} s (floor hidden?); turning first.")
            self.depth_lost_since = None
            self.trigger = "depth_lost"
            self._start_turn()
            return
        self.depth_lost_since = None
        if self.ready_since is None:
            self.ready_since = now
            self.get_logger().info(f"Sensors ready; starting in {self.start_delay:.1f} s.")
        if (now - self.ready_since).nanoseconds >= self.start_delay * 1e9:
            if self._hit():
                self.get_logger().info("Obstacle directly ahead at start; turning first.")
                self._start_turn()
            else:
                self.state = FORWARD
        self._publish(0.0, 0.0)

    def _forward(self, clearance: float) -> None:
        if self.use_depth and not self._depth_fresh():
            now = self.get_clock().now()
            self.depth_lost_since = self.depth_lost_since or now
            if (now - self.depth_lost_since).nanoseconds <= self.depth_lost_turn * 1e9:
                self.get_logger().warn("Depth is stale; holding still.", throttle_duration_sec=1.0)
                self._publish(0.0, 0.0)
                return
            trigger = "depth_lost"
        else:
            trigger = self._hit()
        self.depth_lost_since = None
        if not trigger:
            self._publish(self.linear_speed, 0.0)
            return
        self.trigger = trigger
        self.detect_pose = list(self.pose)
        self.detect_clearance = clearance
        self.min_clearance = clearance
        self.state = BRAKE
        depth_text = f", depth {self._depth_clearance():.3f} m" if self.use_depth else ""
        self.get_logger().info(f"Obstacle detected by {trigger}: clearance {clearance:.3f} m, front {self._front_clearance():.3f} m{depth_text}, odom speed {self.odom_speed:.3f} m/s.")
        self._publish(0.0, 0.0)

    def _brake(self, clearance: float) -> None:
        self._publish(0.0, 0.0)
        if self.odom_speed > self.stopped_speed:
            return
        braking = hypot(self.pose[0] - self.detect_pose[0], self.pose[1] - self.detect_pose[1])
        self.get_logger().info(f"Stopped: braking distance {braking:.3f} m, clearance {clearance:.3f} m (at detection {self.detect_clearance:.3f} m).")
        self._start_turn()

    def _start_backup(self) -> None:
        self.trigger = "bump"
        self.get_logger().info(f"Bumper hit ({self.bump}) at clearance {self._clearance():.3f} m, odom speed {self.odom_speed:.3f} m/s; backing up {self.backup_distance:.2f} m.")
        self.state = BACKUP
        self.backup_start = list(self.pose)
        self.backup_since = self.get_clock().now()

    def _backup(self) -> None:
        travelled = hypot(self.pose[0] - self.backup_start[0], self.pose[1] - self.backup_start[1])
        walls = np.concatenate([self.walls, self._depth_walls()]) if self.use_depth else self.walls
        rear_blocked = colliding_wall(walls, np.array([-self.backup_step, 0.0]), self.body_radius) is not None
        timed_out = (self.get_clock().now() - self.backup_since).nanoseconds > 3.0 * self.backup_distance / self.backup_speed * 1e9
        if travelled < self.backup_distance and not rear_blocked and not timed_out:
            self._publish(-self.backup_speed, 0.0)
            return
        reason = "obstacle behind" if rear_blocked else "timeout" if timed_out else "done"
        self.get_logger().info(f"Backed up {travelled:.3f} m ({reason}); turning.")
        self.bump = None
        self._publish(0.0, 0.0)
        self._start_turn()

    def _start_turn(self) -> None:
        self.state = TURN
        self.turn_remaining = pi
        self.last_yaw = self.pose[2]
        self.min_clearance = float("inf")

    def _turn(self) -> None:
        self.turn_remaining -= abs(wrap_angle(self.pose[2] - self.last_yaw))
        self.last_yaw = self.pose[2]
        if self.turn_remaining > self.turn_tolerance:
            self._publish(0.0, min(self.angular_speed, self.turn_remaining / self.dt))
            return
        self.turns += 1
        self.get_logger().info(f"Turn {self.turns} done (min clearance while turning {self.min_clearance:.3f} m).")
        if self.max_turns > 0 and self.turns >= self.max_turns:
            self.state = DONE
            self.get_logger().info("Reached max_turns; test finished. Robot holds still.")
        else:
            self.state = FORWARD
        self._publish(0.0, 0.0)

    def shutdown(self) -> None:
        """Stop the robot and close the log."""
        self.state = DONE
        self._publish(0.0, 0.0)
        self.log_file.close()


def main() -> None:
    rclpy.init()
    node = CollisionTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()
