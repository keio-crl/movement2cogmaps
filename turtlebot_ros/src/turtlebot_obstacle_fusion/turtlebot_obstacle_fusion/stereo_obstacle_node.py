"""ROS 2 node that turns OAK-D stereo depth into 2D obstacle ranges.

Publishes the same 360-bin LaserScan in the base frame as depth_obstacle_node, so collision_test
can use either with ``use_depth: true``. Needs no torch; runs with the system Python.
"""

from __future__ import annotations

import csv
from datetime import datetime
from math import pi
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from .depth_estimation import backproject, bin_ranges, decode_compressed_depth, floor_tilt, quaternion_matrix, tilt_correction

CSV_HEADER = ["time", "image_stamp", "age_s", "valid_ratio", "floor_points", "floor_height_median_m", "floor_pitch_deg", "floor_roll_deg", "obstacle_points", "min_obstacle_m", "npz"]


class StereoObstacle(Node):
    """Publish OAK-D stereo depth obstacles as a 360-bin LaserScan in the base frame."""

    def __init__(self) -> None:
        super().__init__("stereo_obstacle")
        defaults = {
            "depth_topic": "/stereo/depth/compressedDepth",
            "camera_info_topic": "/stereo/camera_info",
            "output_topic": "/depth_obstacles",
            "base_frame": "base_link",
            "camera_frame": "oakd_pro_right_camera_optical_frame",
            "use_distortion": False,
            "camera_pitch_down_deg": 0.0,
            "camera_roll_deg": 0.0,
            "depth_min_m": 0.1,
            "depth_max_m": 2.5,
            "obstacle_min_height_m": 0.05,
            "obstacle_max_height_m": 0.40,
            "pixel_stride": 2,
            "angle_bin_degrees": 1.0,
            "min_points_per_bin": 5,
            "save_period_s": 1.0,
            "log_dir": str(Path.home() / "depth_obstacle_logs"),
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        self.bin_count = round(360.0 / float(self.p["angle_bin_degrees"]))
        self.bin_width = 2.0 * pi / self.bin_count
        self.tilt = tilt_correction(float(self.p["camera_pitch_down_deg"]), float(self.p["camera_roll_deg"]))

        self.intrinsics = None
        self.distortion = None
        self.info_size = None
        self.last_save: Optional[Time] = None
        self.frame_index = 0

        self.log_dir = Path(str(self.p["log_dir"])).expanduser() / datetime.now().strftime("stereo_%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_dir / "frames.csv", "w", newline="")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(CSV_HEADER)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(LaserScan, self.p["output_topic"], 10)
        self.create_subscription(CameraInfo, self.p["camera_info_topic"], self._camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(CompressedImage, self.p["depth_topic"], self._depth_callback, qos_profile_sensor_data)
        self.get_logger().info(f"Waiting for {self.p['depth_topic']} and {self.p['camera_info_topic']}; saving diagnostics to {self.log_dir}")

    def _camera_info_callback(self, message: CameraInfo) -> None:
        if message.k[0] > 0.0:
            self.intrinsics = (message.k[0], message.k[4], message.k[2], message.k[5])
            self.info_size = (message.width, message.height)
            distortion = np.array(message.d, dtype=float)
            self.distortion = distortion if distortion.size in (4, 5, 8, 12, 14) and np.any(distortion != 0.0) else None

    def _camera_pose(self):
        """Return (R, t) from the depth optical frame to the base frame, or None."""
        try:
            transform = self.tf_buffer.lookup_transform(self.p["base_frame"], self.p["camera_frame"], Time(), timeout=Duration(seconds=0.05))
        except TransformException as error:
            self.get_logger().warn(f"Cannot transform {self.p['camera_frame']} to {self.p['base_frame']}: {error}", throttle_duration_sec=2.0)
            return None
        r, t = transform.transform.rotation, transform.transform.translation
        return self.tilt @ quaternion_matrix(r.x, r.y, r.z, r.w), np.array([t.x, t.y, t.z])

    def _depth_callback(self, message: CompressedImage) -> None:
        now = self.get_clock().now()
        depth_mm = decode_compressed_depth(message.data)
        if depth_mm is None:
            self.get_logger().warn(f"Cannot decode depth (format {message.format!r}).", throttle_duration_sec=5.0)
            return
        if self.intrinsics is None:
            self.get_logger().warn(f"Waiting for {self.p['camera_info_topic']}.", throttle_duration_sec=5.0)
            return
        height, width = depth_mm.shape
        if self.info_size != (width, height):
            self.get_logger().warn(f"CameraInfo is {self.info_size} but the depth is {(width, height)}; skipping.", throttle_duration_sec=5.0)
            return
        pose = self._camera_pose()
        if pose is None:
            return
        rotation, translation = pose

        depth = depth_mm.astype(np.float32) * 1e-3
        depth[(depth_mm == 0) | (depth < float(self.p["depth_min_m"])) | (depth > float(self.p["depth_max_m"]))] = np.nan
        distortion = self.distortion if self.p["use_distortion"] else None
        points_cam, pixels = backproject(depth, self.intrinsics, int(self.p["pixel_stride"]), distortion)
        points = points_cam @ rotation.T + translation
        floor = points[:, 2] <= float(self.p["obstacle_min_height_m"])
        obstacle = ~floor & (points[:, 2] < float(self.p["obstacle_max_height_m"])) & (points[:, 0] > 0.0)
        obstacle_xy, obstacle_pixels = points[obstacle, :2], pixels[obstacle]
        self._publish(bin_ranges(obstacle_xy, self.bin_count, int(self.p["min_points_per_bin"])), now)

        if self.last_save is None or (now - self.last_save).nanoseconds * 1e-9 >= float(self.p["save_period_s"]):
            self.last_save = now
            stamp = Time.from_msg(message.header.stamp)
            floor_height = float(np.median(points[floor, 2])) if floor.any() else float("nan")
            self._save(now, stamp, depth_mm, floor.sum(), floor_height, floor_tilt(points)[:2], obstacle_xy, obstacle_pixels, rotation, translation)

    def _publish(self, ranges: np.ndarray, now: Time) -> None:
        scan = LaserScan()
        scan.header.stamp = now.to_msg()
        scan.header.frame_id = self.p["base_frame"]
        scan.angle_min, scan.angle_max, scan.angle_increment = -pi + self.bin_width / 2.0, pi - self.bin_width / 2.0, self.bin_width
        scan.range_min, scan.range_max = 0.0, float(self.p["depth_max_m"])
        scan.ranges = [float(value) for value in ranges]
        self.publisher.publish(scan)

    def _save(self, now, stamp, depth_mm, floor_points, floor_height, residual_tilt, obstacle_xy, obstacle_pixels, rotation, translation) -> None:
        name = f"stereo_{self.frame_index:05d}"
        self.frame_index += 1
        np.savez_compressed(
            self.log_dir / f"{name}.npz",
            depth_mm=depth_mm, obstacle_xy=obstacle_xy, obstacle_pixels=obstacle_pixels,
            intrinsics=np.array(self.intrinsics), rotation=rotation, translation=translation,
        )
        self._save_image(name, depth_mm, obstacle_pixels)
        distances = np.hypot(obstacle_xy[:, 0], obstacle_xy[:, 1])
        self.writer.writerow([
            f"{now.nanoseconds * 1e-9:.3f}", f"{stamp.nanoseconds * 1e-9:.3f}", f"{(now - stamp).nanoseconds * 1e-9:.3f}",
            f"{np.count_nonzero(depth_mm) / depth_mm.size:.4f}", int(floor_points), f"{floor_height:.3f}", f"{residual_tilt[0]:.2f}", f"{residual_tilt[1]:.2f}",
            len(obstacle_xy), f"{distances.min():.3f}" if len(distances) else "", f"{name}.npz",
        ])
        self.log_file.flush()
        self.get_logger().info(
            f"valid {100 * np.count_nonzero(depth_mm) / depth_mm.size:.1f}%, floor points {int(floor_points)} (median height {floor_height:+.3f} m, remaining tilt pitch {residual_tilt[0]:+.1f} roll {residual_tilt[1]:+.1f} deg), "
            f"obstacle points {len(obstacle_xy)}" + (f", nearest {distances.min():.2f} m" if len(distances) else ""))

    def _save_image(self, name: str, depth_mm: np.ndarray, obstacle_pixels: np.ndarray) -> None:
        """Half-size colour map (near = red, 0 to depth_max_m) with obstacle pixels in white."""
        scaled = np.clip(depth_mm.astype(np.float32) / (1000.0 * float(self.p["depth_max_m"])), 0.0, 1.0)
        colour = cv2.applyColorMap((255 * (1.0 - scaled)).astype(np.uint8), cv2.COLORMAP_JET)
        colour[depth_mm == 0] = 0
        colour[obstacle_pixels[:, 1], obstacle_pixels[:, 0]] = 255
        colour = cv2.dilate(colour, np.ones((3, 3), np.uint8))
        cv2.imwrite(str(self.log_dir / f"{name}.png"), cv2.resize(colour, (depth_mm.shape[1] // 2, depth_mm.shape[0] // 2), interpolation=cv2.INTER_NEAREST))

    def shutdown(self) -> None:
        """Close the log."""
        self.log_file.close()


def main() -> None:
    rclpy.init()
    node = StereoObstacle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
