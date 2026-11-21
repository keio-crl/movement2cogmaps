"""Interactive launcher for the collision test: LiDAR only, + Depth Anything V2, or + OAK-D stereo depth.

Usage: ``ros2 run turtlebot_obstacle_fusion collision_test_cli [--dry-run]``.
``--dry-run`` sends velocity commands to ``/cmd_vel_dryrun`` so the robot does not move.
The Depth Anything node runs with the Python in ``$DEPTH_PYTHON`` (default ``~/tomita-turtlebot/.venv/bin/python``).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from ament_index_python.packages import get_package_share_directory

PACKAGE = "turtlebot_obstacle_fusion"
DEFAULT_DEPTH_PYTHON = str(Path.home() / "tomita-turtlebot" / ".venv" / "bin" / "python")
DEPTH_READY_TIMEOUT_S = 90.0
CAMERA_CONFIGS = {"1": "depth_obstacle_oakd.yaml", "2": "depth_obstacle_wide.yaml"}
# Dense OAK-D stereo depth reaches the PC only about every 2 s with the Pi defaults (PNG encoding
# of 1280x720 on the Pi), so collision_test accepts older depth and leans on its odom-frame memory.
STEREO_DEPTH_TIMEOUT_S = 2.5
STEREO_DEPTH_LOST_TURN_S = 5.0


def ask(prompt: str, default: str, valid=None) -> str:
    """Ask until the answer (or the default on an empty line) passes ``valid``."""
    while True:
        answer = input(f"{prompt} [{default}]: ").strip() or default
        if valid is None or valid(answer):
            return answer
        print("  入力が正しくありません。")


def is_float(text: str, low: float, high: float) -> bool:
    try:
        return low <= float(text) <= high
    except ValueError:
        return False


def stop(process: Optional[subprocess.Popen], name: str) -> None:
    """Send SIGINT to ``process`` and kill it if it does not exit within 10 s."""
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        print(f"{name} did not exit; killing it.")
        process.kill()
        process.wait()


def wait_for_depth(process: subprocess.Popen, topic: str) -> bool:
    """Wait until the depth node has loaded its model and advertises ``topic``.

    The first message is not awaited: with the floor hidden (an obstacle filling the view) no
    depth is published, and collision_test then turns in place until the floor is visible.
    """
    import rclpy

    rclpy.init()
    node = rclpy.create_node("collision_test_cli_wait")
    deadline = time.monotonic() + DEPTH_READY_TIMEOUT_S
    ready = False
    try:
        while not ready and process.poll() is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
            ready = node.count_publishers(topic) > 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return ready


def main() -> None:
    dry_run = "--dry-run" in sys.argv[1:]
    config = Path(get_package_share_directory(PACKAGE)) / "config"
    print("=== LiDAR 衝突回避テスト ===" + ("（ドライラン: /cmd_vel_dryrun に出力、ロボットは動きません）" if dry_run else ""))
    print("障害物センサ:\n  1) LiDAR のみ\n  2) LiDAR + 深度 (Depth Anything V2)\n  3) LiDAR + OAK-D ステレオ深度")
    sensor = ask("番号", "1", lambda a: a in ("1", "2", "3"))
    use_depth, use_stereo = sensor in ("2", "3"), sensor == "3"
    depth_config = None
    if sensor == "2":
        print("深度推定に使うカメラ:\n  1) OAK-D カラー画像（camera_info と TF を使用）\n  2) 広角 USB カメラ（depth_obstacle_wide.yaml の画角・取付位置を使用）")
        depth_config = CAMERA_CONFIGS[ask("番号", "1", lambda a: a in CAMERA_CONFIGS)]
    print("バンパー（Create 3）に当たったとき:\n  1) 止まって少し下がり、180°回る\n  2) 何もしない")
    use_bumper = ask("番号", "1", lambda a: a in ("1", "2")) == "1"
    speed = ask("直進速度 m/s (0.05-0.31)", "0.1", lambda a: is_float(a, 0.05, 0.31))
    turns = ask("反転回数 (0 = Ctrl+C まで)", "4", lambda a: a.isdigit())

    test_command = [
        "ros2", "run", PACKAGE, "collision_test", "--ros-args",
        "--params-file", str(config / "collision_test.yaml"),
        "-p", f"linear_speed_m_s:={float(speed)}", "-p", f"max_turns:={int(turns)}",
        "-p", f"use_depth:={'true' if use_depth else 'false'}",
        "-p", f"use_bumper:={'true' if use_bumper else 'false'}",
    ]
    if use_stereo:
        test_command += ["-p", f"depth_timeout_s:={STEREO_DEPTH_TIMEOUT_S}", "-p", f"depth_lost_turn_s:={STEREO_DEPTH_LOST_TURN_S}"]
    if dry_run:
        test_command += ["-p", "cmd_vel_topic:=/cmd_vel_dryrun"]

    depth_process: Optional[subprocess.Popen] = None
    test_process: Optional[subprocess.Popen] = None
    try:
        if use_stereo:
            depth_process = subprocess.Popen(["ros2", "run", PACKAGE, "stereo_obstacle", "--ros-args", "--params-file", str(config / "stereo_obstacle.yaml")])
            print("ステレオ深度ノードを起動しました。")
            if not wait_for_depth(depth_process, "/depth_obstacles"):
                print("ステレオ深度ノードが起動しませんでした。上のログを確認してください。")
                return
            print("深度の準備ができました。テストを開始します。")
        elif use_depth:
            python = os.environ.get("DEPTH_PYTHON", DEFAULT_DEPTH_PYTHON)
            if not Path(python).exists():
                print(f"深度ノード用の Python が見つかりません: {python}（環境変数 DEPTH_PYTHON で指定できます）")
                return
            depth_process = subprocess.Popen([python, "-m", f"{PACKAGE}.depth_obstacle_node", "--ros-args", "--params-file", str(config / depth_config)])
            print("深度ノードを起動しました。モデルの読み込みを待っています...")
            if not wait_for_depth(depth_process, "/depth_obstacles"):
                print("深度ノードが起動しませんでした。上のログを確認してください。")
                return
            print("深度の準備ができました。テストを開始します。")
        test_process = subprocess.Popen(test_command)
        test_process.wait()
    except KeyboardInterrupt:
        print("\n停止しています...")
    finally:
        stop(test_process, "collision_test")
        stop(depth_process, "depth_obstacle")


if __name__ == "__main__":
    main()
