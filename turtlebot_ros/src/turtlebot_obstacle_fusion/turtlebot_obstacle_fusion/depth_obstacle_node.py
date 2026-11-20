"""ROS 2 node that turns RGB images into 2D obstacle ranges with Depth Anything V2 and LiDAR scaling.

Run it with a Python that has torch and depth_anything_v2 installed (see README), e.g.
``~/tomita-turtlebot/.venv/bin/python -m turtlebot_obstacle_fusion.depth_obstacle_node``.
"""

from __future__ import annotations

import csv
from datetime import datetime
from math import atan2, cos, isfinite, pi, radians, sin
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

from .depth_estimation import (
    DepthFit,
    bin_ranges,
    extrinsic_from_pose,
    fit_inverse_depth,
    floor_pairs,
    front_minimum,
    obstacle_points,
    pinhole_from_hfov,
    project_to_image,
    quaternion_matrix,
    render_panel,
    sample_median,
)

CSV_HEADER = ["time", "scale_source", "a", "b", "pairs", "inliers", "residual", "fit_age_s", "lidar_check_err", "front_min_depth", "front_min_lidar", "infer_ms", "png"]


def load_model(weights_path: str, device: str):
    """Build Depth Anything V2 ViT-S and load the checkpoint."""
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    checkpoint = torch.load(str(Path(weights_path).expanduser()), map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    state = {key[len("module."):] if key.startswith("module.") else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


class DepthObstacle(Node):
    """Publish Depth Anything V2 obstacles as a 360-bin LaserScan in the base frame."""

    def __init__(self) -> None:
        super().__init__("depth_obstacle")
        defaults = {
            "image_topic": "/image_raw/compressed",
            "camera_info_topic": "",
            "scan_topic": "/scan",
            "output_topic": "/depth_obstacles",
            "base_frame": "base_link",
            "camera_frame": "",
            "hfov_deg": 69.0,
            "cam_x": 0.0,
            "cam_y": 0.0,
            "cam_z": 0.3,
            "cam_yaw_deg": 0.0,
            "cam_pitch_deg": 0.0,
            "cam_roll_deg": 0.0,
            "weights_path": str(Path.home() / "tomita-turtlebot" / "depth_anything_v2_vits.pth"),
            "device": "cuda:0",
            "input_size": 518,
            "infer_hz": 5.0,
            "scale_source": "floor",
            "floor_min_m": 0.3,
            "floor_max_m": 3.0,
            "max_scan_age_s": 0.3,
            "fit_hold_s": 2.0,
            "fit_min_pairs": 15,
            "fit_inlier_tolerance": 0.15,
            "fit_max_residual": 0.1,
            "fit_min_inlier_ratio": 0.6,
            "sample_radius_px": 2,
            "depth_min_m": 0.1,
            "depth_max_m": 2.5,
            "obstacle_min_height_m": 0.05,
            "obstacle_max_height_m": 0.40,
            "pixel_stride": 4,
            "angle_bin_degrees": 1.0,
            "min_points_per_bin": 5,
            "robot_radius_m": 0.1705,
            "collision_margin_m": 0.05,
            "save_period_s": 1.0,
            "log_dir": str(Path.home() / "depth_obstacle_logs"),
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        self.bin_count = round(360.0 / float(self.p["angle_bin_degrees"]))
        self.bin_width = 2.0 * pi / self.bin_count

        self.get_logger().info(f"Loading Depth Anything V2 from {self.p['weights_path']} on {self.p['device']} ...")
        self.model = load_model(self.p["weights_path"], self.p["device"])
        self.get_logger().info("Model loaded.")

        self.image: Optional[CompressedImage] = None
        self.image_received: Optional[Time] = None
        self.lidar_xy = np.zeros((0, 2))
        self.lidar_height = 0.0
        self.scan_received: Optional[Time] = None
        self.intrinsics_from_info = None
        self.distortion_from_info = None
        self.info_size = None
        self.fit: Optional[DepthFit] = None
        self.fit_time: Optional[Time] = None
        self.last_save: Optional[Time] = None
        self.frame_index = 0

        self.log_dir = Path(str(self.p["log_dir"])).expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_dir / "frames.csv", "w", newline="")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(CSV_HEADER)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(LaserScan, self.p["output_topic"], 10)
        self.create_subscription(CompressedImage, self.p["image_topic"], self._image_callback, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.p["scan_topic"], self._scan_callback, qos_profile_sensor_data)
        if self.p["camera_info_topic"]:
            self.create_subscription(CameraInfo, self.p["camera_info_topic"], self._camera_info_callback, qos_profile_sensor_data)
        self.create_timer(1.0 / float(self.p["infer_hz"]), self._tick)
        self.get_logger().info(f"Waiting for {self.p['image_topic']} and {self.p['scan_topic']}; saving diagnostics to {self.log_dir}")

    def _image_callback(self, message: CompressedImage) -> None:
        self.image = message
        self.image_received = self.get_clock().now()

    def _camera_info_callback(self, message: CameraInfo) -> None:
        if message.k[0] > 0.0:
            self.intrinsics_from_info = (message.k[0], message.k[4], message.k[2], message.k[5])
            self.info_size = (message.width, message.height)
            distortion = np.array(message.d, dtype=float)
            self.distortion_from_info = distortion if distortion.size in (4, 5, 8, 12, 14) and np.any(distortion != 0.0) else None

    def _lookup(self, frame: str):
        try:
            return self.tf_buffer.lookup_transform(self.p["base_frame"], frame, Time(), timeout=Duration(seconds=0.05))
        except TransformException as error:
            self.get_logger().warn(f"Cannot transform {frame} to {self.p['base_frame']}: {error}", throttle_duration_sec=2.0)
            return None

    def _scan_callback(self, scan: LaserScan) -> None:
        transform = self._lookup(scan.header.frame_id)
        if transform is None:
            return
        q = transform.transform.rotation
        offset = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        tx, ty = transform.transform.translation.x, transform.transform.translation.y
        points = []
        for index, distance in enumerate(scan.ranges):
            if isfinite(distance) and scan.range_min <= distance <= scan.range_max:
                angle = offset + scan.angle_min + index * scan.angle_increment
                points.append((tx + distance * cos(angle), ty + distance * sin(angle)))
        self.lidar_xy = np.array(points, dtype=float).reshape(-1, 2)
        self.lidar_height = transform.transform.translation.z
        self.scan_received = self.get_clock().now()

    def _camera_pose(self):
        """Return (R, t) from optical frame to base frame, preferring TF when ``camera_frame`` is set."""
        if self.p["camera_frame"]:
            transform = self._lookup(self.p["camera_frame"])
            if transform is None:
                return None
            r = transform.transform.rotation
            t = transform.transform.translation
            return quaternion_matrix(r.x, r.y, r.z, r.w), np.array([t.x, t.y, t.z])
        return extrinsic_from_pose(
            float(self.p["cam_x"]), float(self.p["cam_y"]), float(self.p["cam_z"]),
            radians(float(self.p["cam_yaw_deg"])), radians(float(self.p["cam_pitch_deg"])), radians(float(self.p["cam_roll_deg"])),
        )

    def _age(self, stamp: Optional[Time], now: Time) -> float:
        return float("inf") if stamp is None else (now - stamp).nanoseconds * 1e-9

    def _infer(self, bgr: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        image, (height, width) = self.model.image2tensor(bgr, int(self.p["input_size"]))
        with torch.no_grad():
            relative = self.model(image.to(self.p["device"]))
            relative = F.interpolate(relative[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
        return relative.cpu().numpy()

    def _tick(self) -> None:
        now = self.get_clock().now()
        if self.image is None or self._age(self.image_received, now) > 1.0:
            self.get_logger().warn(f"No recent image on {self.p['image_topic']}.", throttle_duration_sec=5.0)
            return
        message, self.image = self.image, None
        bgr = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(f"Cannot decode image (format {message.format!r}).", throttle_duration_sec=5.0)
            return
        pose = self._camera_pose()
        if pose is None:
            return
        rotation, translation = pose
        height, width = bgr.shape[:2]
        if self.p["camera_info_topic"]:
            if self.intrinsics_from_info is None:
                self.get_logger().warn(f"Waiting for {self.p['camera_info_topic']}.", throttle_duration_sec=5.0)
                return
            if self.info_size != (width, height):
                self.get_logger().warn(f"CameraInfo is {self.info_size} but the image is {(width, height)}; skipping.", throttle_duration_sec=5.0)
                return
            intrinsics, distortion = self.intrinsics_from_info, self.distortion_from_info
        else:
            intrinsics, distortion = pinhole_from_hfov(width, height, float(self.p["hfov_deg"])), None

        started = self.get_clock().now()
        relative = self._infer(bgr)
        infer_ms = (self.get_clock().now() - started).nanoseconds * 1e-6

        lidar_uv, lidar_metres, lidar_relative = np.zeros((0, 2), int), np.zeros(0), np.zeros(0)
        scan_fresh = self._age(self.scan_received, now) <= float(self.p["max_scan_age_s"])
        if scan_fresh and len(self.lidar_xy):
            lidar_base = np.column_stack([self.lidar_xy, np.full(len(self.lidar_xy), self.lidar_height)])
            lidar_cam = (lidar_base - translation) @ rotation
            lidar_uv, kept = project_to_image(lidar_cam, intrinsics, width, height, distortion=distortion)
            lidar_metres = lidar_cam[kept, 2]
            lidar_relative = sample_median(relative, lidar_uv, int(self.p["sample_radius_px"]))
        if self.p["scale_source"] == "lidar":
            pair_relative, pair_metres = lidar_relative, lidar_metres
        else:
            pair_relative, pair_metres, _ = floor_pairs(relative, intrinsics, rotation, translation, int(self.p["pixel_stride"]), (float(self.p["floor_min_m"]), float(self.p["floor_max_m"])), distortion)
        fit = fit_inverse_depth(pair_relative, pair_metres, int(self.p["fit_min_pairs"]), float(self.p["fit_inlier_tolerance"]), float(self.p["fit_max_residual"]), min_inlier_ratio=float(self.p["fit_min_inlier_ratio"]))
        if fit is not None:
            self.fit, self.fit_time = fit, now
        fit_age = self._age(self.fit_time, now)
        usable = self.fit is not None and fit_age <= float(self.p["fit_hold_s"])
        if not usable:
            self.get_logger().warn(f"No reliable {self.p['scale_source']} scale fit ({len(pair_metres)} pairs); not publishing.", throttle_duration_sec=2.0)

        depth, depth_xy, pixels = None, np.zeros((0, 2)), np.zeros((0, 2), int)
        if usable:
            depth = self.fit.to_metres(relative)
            depth_xy, pixels = obstacle_points(
                depth, intrinsics, rotation, translation, int(self.p["pixel_stride"]),
                (float(self.p["depth_min_m"]), float(self.p["depth_max_m"])),
                (float(self.p["obstacle_min_height_m"]), float(self.p["obstacle_max_height_m"])),
                distortion,
            )
            self._publish(bin_ranges(depth_xy, self.bin_count, int(self.p["min_points_per_bin"])), now)

        if self.last_save is None or self._age(self.last_save, now) >= float(self.p["save_period_s"]):
            self.last_save = now
            self._save(now, bgr, relative, depth, pixels, lidar_uv, lidar_metres, lidar_relative, depth_xy, fit, fit_age if usable else float("nan"), intrinsics, distortion, rotation, translation, infer_ms)

    def _publish(self, ranges: np.ndarray, now: Time) -> None:
        scan = LaserScan()
        scan.header.stamp = now.to_msg()
        scan.header.frame_id = self.p["base_frame"]
        scan.angle_min, scan.angle_max, scan.angle_increment = -pi, pi - self.bin_width, self.bin_width
        scan.range_min, scan.range_max = 0.0, float(self.p["depth_max_m"])
        scan.ranges = [float(value) for value in ranges]
        self.publisher.publish(scan)

    def _save(self, now, bgr, relative, depth, pixels, lidar_uv, lidar_metres, lidar_relative, depth_xy, fit, fit_age, intrinsics, distortion, rotation, translation, infer_ms) -> None:
        used = fit if fit is not None else (self.fit if depth is not None else None)
        lidar_check = float(np.median(np.abs(used.to_metres(lidar_relative) - lidar_metres) / lidar_metres)) if used is not None and len(lidar_metres) else float("nan")
        front_depth = front_minimum(depth_xy, radians(15.0))
        front_lidar = front_minimum(self.lidar_xy, radians(15.0))
        lines = [
            datetime.now().strftime("%H:%M:%S.%f")[:-3],
            f"scale from {self.p['scale_source']}: " + (f"a={fit.a:.3f} b={fit.b:.3f}" if fit else "FAILED"),
            (f"inliers {fit.inliers}/{fit.pairs} err {100 * fit.residual:.1f}%" if fit else ""),
            "depth used: " + ("yes" if depth is not None else "NO (not published)"),
            (f"fit age {fit_age:.2f} s" if depth is not None else ""),
            f"LiDAR pts {len(lidar_metres)}, depth vs LiDAR err {100 * lidar_check:.0f}%",
            "front +-15deg min (from base_link):",
            f"  depth {front_depth:.2f} m",
            f"  LiDAR {front_lidar:.2f} m",
            f"obstacle band {self.p['obstacle_min_height_m']}-{self.p['obstacle_max_height_m']} m",
            f"depth range {self.p['depth_min_m']}-{self.p['depth_max_m']} m",
            f"infer {infer_ms:.0f} ms",
            f"camera {self.p['image_topic']}"[:40],
        ]
        panel = render_panel(
            bgr, depth, pixels, lidar_uv, lidar_metres, self.lidar_xy, depth_xy,
            float(self.p["robot_radius_m"]), float(self.p["robot_radius_m"]) + float(self.p["collision_margin_m"]),
            float(self.p["depth_max_m"]), lines,
        )
        name = f"frame_{self.frame_index:05d}"
        self.frame_index += 1
        cv2.imwrite(str(self.log_dir / f"{name}.png"), panel)
        np.savez_compressed(
            self.log_dir / f"{name}.npz",
            relative=relative[::2, ::2].astype(np.float16),
            depth_m=(depth[::2, ::2] if depth is not None else np.zeros((0, 0))).astype(np.float32),
            obstacle_xy=depth_xy, obstacle_pixels=pixels,
            lidar_uv=lidar_uv, lidar_m=lidar_metres, lidar_relative=lidar_relative, lidar_xy=self.lidar_xy,
            fit=np.array([used.a, used.b] if used else [np.nan, np.nan]),
            intrinsics=np.array(intrinsics), distortion=np.zeros(0) if distortion is None else distortion, rotation=rotation, translation=translation,
        )
        self.writer.writerow([
            f"{now.nanoseconds * 1e-9:.3f}", self.p["scale_source"],
            f"{fit.a:.6f}" if fit else "", f"{fit.b:.6f}" if fit else "",
            fit.pairs if fit else 0, fit.inliers if fit else 0, f"{fit.residual:.4f}" if fit else "",
            f"{fit_age:.3f}", f"{lidar_check:.3f}", f"{front_depth:.3f}", f"{front_lidar:.3f}", f"{infer_ms:.1f}", f"{name}.png",
        ])
        self.log_file.flush()

    def shutdown(self) -> None:
        """Close the log."""
        self.log_file.close()


def main() -> None:
    rclpy.init()
    node = DepthObstacle()
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
