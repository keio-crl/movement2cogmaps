"""Save the camera image and its Depth Anything V2 depth for every row that rat_walker logs.

rat_walker publishes ``"<session>,<row>,<t>"`` on ``sample_topic`` each time it writes a CSV row.
For each sample the latest image is saved with its relative depth (raw model output) and, when the
scale fit succeeds, its metric depth, into ``log_dir/<session>/``. ``status_topic`` reports
``ready`` while the model is loaded and images are fresh; rat_walker holds still otherwise.

Run it with a Python that has torch and depth_anything_v2 installed (see README).
"""

from __future__ import annotations

import csv
from math import atan2, cos, isfinite, sin
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .depth_estimation import decode_image, extrinsic_from_pose, fit_inverse_depth, floor_pairs, lidar_pairs, pinhole_from_hfov, quaternion_matrix
from .depth_obstacle_node import load_model

CSV_HEADER = ["row", "t", "image_stamp", "image_age_s", "width", "height", "fit_ok", "a", "b", "pairs", "inliers", "residual", "infer_ms", "rgb", "depth"]


class RatRecorder(Node):
    """Write rgb_XXXXX.png and depth_XXXXX.npz for each rat_walker row."""

    def __init__(self) -> None:
        super().__init__("rat_recorder")
        defaults = {
            "image_topic": "/color/preview/image/compressed",
            "image_type": "compressed",
            "camera_info_topic": "/color/preview/camera_info",
            "camera_frame": "oakd_pro_rgb_camera_optical_frame",
            "hfov_deg": 69.0,
            "cam_x": 0.0, "cam_y": 0.0, "cam_z": 0.3, "cam_yaw_deg": 0.0, "cam_pitch_deg": 0.0, "cam_roll_deg": 0.0,
            "scan_topic": "/scan",
            "base_frame": "base_link",
            "sample_topic": "/rat_walker/sample",
            "status_topic": "/rat_recorder/status",
            "weights_path": str(Path.home() / "tomita-turtlebot" / "depth_anything_v2_vits.pth"),
            "device": "cuda:0",
            "input_size": 518,
            "max_image_age_s": 0.3,
            "scale_source": "floor",
            "floor_min_m": 0.3,
            "floor_max_m": 3.0,
            "pixel_stride": 4,
            "max_scan_age_s": 0.3,
            "sample_radius_px": 2,
            "fit_min_pairs": 15,
            "fit_inlier_tolerance": 0.15,
            "fit_max_residual": 0.1,
            "fit_min_inlier_ratio": 0.0,
            "log_dir": str(Path.home() / "rat_walker_logs"),
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self.p = {name: self.get_parameter(name).value for name in defaults}

        self.ready = False
        self.image = None
        self.image_received: Optional[Time] = None
        self.intrinsics = None
        self.distortion = None
        self.info_size = None
        self.lidar_xy = np.zeros((0, 2))
        self.lidar_height = 0.0
        self.scan_received: Optional[Time] = None
        self.session: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.log_file = None
        self.writer = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.status_pub = self.create_publisher(String, self.p["status_topic"], 10)
        self.create_timer(0.2, self._publish_status)

        self.get_logger().info(f"Loading Depth Anything V2 from {self.p['weights_path']} on {self.p['device']} ...")
        self.model = load_model(self.p["weights_path"], self.p["device"])
        self.ready = True
        self.get_logger().info("Model loaded.")

        image_type = CompressedImage if self.p["image_type"] == "compressed" else Image
        self.create_subscription(image_type, self.p["image_topic"], self._image_callback, qos_profile_sensor_data)
        if self.p["camera_info_topic"]:
            self.create_subscription(CameraInfo, self.p["camera_info_topic"], self._camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.p["scan_topic"], self._scan_callback, qos_profile_sensor_data)
        self.create_subscription(String, self.p["sample_topic"], self._sample_callback, 50)
        self.get_logger().info(f"Recording {self.p['image_topic']} for each row on {self.p['sample_topic']} into {self.p['log_dir']}/<session>/")

    def _now(self) -> Time:
        return self.get_clock().now()

    def _age(self, stamp: Optional[Time]) -> float:
        return float("inf") if stamp is None else (self._now() - stamp).nanoseconds * 1e-9

    def _status(self) -> str:
        if not self.ready:
            return "loading model"
        if self._age(self.image_received) > float(self.p["max_image_age_s"]):
            return f"no image on {self.p['image_topic']}"
        if self.p["camera_info_topic"] and self.intrinsics is None:
            return f"no camera info on {self.p['camera_info_topic']}"
        return "ready"

    def _publish_status(self) -> None:
        self.status_pub.publish(String(data=self._status()))

    def _image_callback(self, message) -> None:
        self.image = message
        self.image_received = self._now()

    def _camera_info_callback(self, message: CameraInfo) -> None:
        if message.k[0] > 0.0:
            self.intrinsics = (message.k[0], message.k[4], message.k[2], message.k[5])
            self.info_size = (message.width, message.height)
            distortion = np.array(message.d, dtype=float)
            self.distortion = distortion if distortion.size in (4, 5, 8, 12, 14) and np.any(distortion != 0.0) else None

    def _lookup(self, frame: str):
        try:
            return self.tf_buffer.lookup_transform(self.p["base_frame"], frame, Time(), timeout=Duration(seconds=0.05))
        except TransformException as error:
            self.get_logger().warn(f"Cannot transform {frame} to {self.p['base_frame']}: {error}", throttle_duration_sec=5.0)
            return None

    def _scan_callback(self, scan: LaserScan) -> None:
        transform = self._lookup(scan.header.frame_id)
        if transform is None:
            return
        q = transform.transform.rotation
        offset = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        tx, ty = transform.transform.translation.x, transform.transform.translation.y
        points = [
            (tx + d * cos(offset + scan.angle_min + i * scan.angle_increment), ty + d * sin(offset + scan.angle_min + i * scan.angle_increment))
            for i, d in enumerate(scan.ranges) if isfinite(d) and scan.range_min <= d <= scan.range_max
        ]
        self.lidar_xy = np.array(points, dtype=float).reshape(-1, 2)
        self.lidar_height = transform.transform.translation.z
        self.scan_received = self._now()

    def _camera_pose(self):
        if self.p["camera_frame"]:
            transform = self._lookup(self.p["camera_frame"])
            if transform is None:
                return None
            r, t = transform.transform.rotation, transform.transform.translation
            return quaternion_matrix(r.x, r.y, r.z, r.w), np.array([t.x, t.y, t.z])
        return extrinsic_from_pose(float(self.p["cam_x"]), float(self.p["cam_y"]), float(self.p["cam_z"]), np.radians(float(self.p["cam_yaw_deg"])), np.radians(float(self.p["cam_pitch_deg"])), np.radians(float(self.p["cam_roll_deg"])))

    def _open_session(self, session: str) -> None:
        if self.log_file is not None:
            self.log_file.close()
        self.session = session
        self.session_dir = Path(str(self.p["log_dir"])).expanduser() / session
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / "frames.csv"
        is_new = not path.exists()
        self.log_file = open(path, "a", newline="")
        self.writer = csv.writer(self.log_file)
        if is_new:
            self.writer.writerow(CSV_HEADER)
        self.get_logger().info(f"Saving images of session {session} to {self.session_dir}")

    def _infer(self, bgr: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        image, (height, width) = self.model.image2tensor(bgr, int(self.p["input_size"]))
        with torch.no_grad():
            relative = self.model(image.to(self.p["device"]))
            relative = F.interpolate(relative[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
        return relative.cpu().numpy().astype(np.float32)

    def _decode(self, message) -> Optional[np.ndarray]:
        if isinstance(message, CompressedImage):
            return decode_image(message.data)
        return decode_image(message.data, message.encoding, message.width, message.height, message.step)

    def _fit(self, relative: np.ndarray):
        """Return (fit or None, pairs) using the floor plane or the LiDAR, as configured."""
        height, width = relative.shape
        if self.p["camera_info_topic"]:
            if self.intrinsics is None or self.info_size != (width, height):
                return None, 0, None, None, None
            intrinsics, distortion = self.intrinsics, self.distortion
        else:
            intrinsics, distortion = pinhole_from_hfov(width, height, float(self.p["hfov_deg"])), None
        pose = self._camera_pose()
        if pose is None:
            return None, 0, intrinsics, None, None
        rotation, translation = pose
        if self.p["scale_source"] == "lidar":
            if self._age(self.scan_received) > float(self.p["max_scan_age_s"]):
                return None, 0, intrinsics, rotation, translation
            rel, metres, _ = lidar_pairs(relative, self.lidar_xy, self.lidar_height, intrinsics, rotation, translation, int(self.p["sample_radius_px"]), distortion)
        else:
            rel, metres, _ = floor_pairs(relative, intrinsics, rotation, translation, int(self.p["pixel_stride"]), (float(self.p["floor_min_m"]), float(self.p["floor_max_m"])), distortion)
        fit = fit_inverse_depth(rel, metres, int(self.p["fit_min_pairs"]), float(self.p["fit_inlier_tolerance"]), float(self.p["fit_max_residual"]), min_inlier_ratio=float(self.p["fit_min_inlier_ratio"]))
        return fit, len(metres), intrinsics, rotation, translation

    def _sample_callback(self, message: String) -> None:
        try:
            session, row_text, t_text = message.data.split(",")
            row, t = int(row_text), float(t_text)
        except ValueError:
            self.get_logger().warn(f"Malformed sample {message.data!r}")
            return
        if session != self.session:
            self._open_session(session)
        status = self._status()
        image = self.image
        if status != "ready" or image is None:
            self.get_logger().error(f"Row {row}: cannot save an image ({status}); rat_walker should have been holding still.")
            self.writer.writerow([row, t_text, "", "", "", "", 0, "", "", 0, 0, "", "", "", ""])
            self.log_file.flush()
            return
        image_age = self._age(self.image_received)
        bgr = self._decode(image)
        if bgr is None:
            self.get_logger().error(f"Row {row}: cannot decode the image.")
            return
        started = self._now()
        relative = self._infer(bgr)
        infer_ms = (self._now() - started).nanoseconds * 1e-6
        fit, pairs, intrinsics, rotation, translation = self._fit(relative)
        depth_m = fit.to_metres(relative).astype(np.float32) if fit is not None else None

        name = f"{row:05d}"
        cv2.imwrite(str(self.session_dir / f"rgb_{name}.png"), bgr)
        stamp = image.header.stamp.sec + image.header.stamp.nanosec * 1e-9
        np.savez_compressed(
            self.session_dir / f"depth_{name}.npz",
            relative=relative,
            depth_m=depth_m if depth_m is not None else np.zeros((0, 0), np.float32),
            fit=np.array([fit.a, fit.b, fit.residual, fit.inliers, fit.pairs] if fit is not None else [np.nan] * 5),
            scale_source=str(self.p["scale_source"]),
            intrinsics=np.array(intrinsics if intrinsics is not None else [np.nan] * 4),
            distortion=self.distortion if self.distortion is not None else np.zeros(0),
            rotation=rotation if rotation is not None else np.full((3, 3), np.nan),
            translation=translation if translation is not None else np.full(3, np.nan),
            row=row, t=t, image_stamp=stamp, image_age_s=image_age,
        )
        self.writer.writerow([
            row, t_text, f"{stamp:.3f}", f"{image_age:.3f}", bgr.shape[1], bgr.shape[0], int(fit is not None),
            f"{fit.a:.6f}" if fit else "", f"{fit.b:.6f}" if fit else "", pairs, fit.inliers if fit else 0,
            f"{fit.residual:.4f}" if fit else "", f"{infer_ms:.1f}", f"rgb_{name}.png", f"depth_{name}.npz",
        ])
        self.log_file.flush()

    def shutdown(self) -> None:
        """Close the frame index."""
        if self.log_file is not None:
            self.log_file.close()


def main() -> None:
    rclpy.init()
    node = RatRecorder()
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
