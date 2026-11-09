#!/usr/bin/env python3
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist


class CmdVelRelay(Node):
    def __init__(self, input_topic: str, output_topic: str) -> None:
        super().__init__("cmd_vel_relay")
        qos = QoSProfile(depth=10)
        self._pub = self.create_publisher(Twist, output_topic, qos)
        self.create_subscription(Twist, input_topic, self._callback, qos)
        self.get_logger().info(
            f"Relaying {input_topic} -> {output_topic} (QoS=RELIABLE)."
        )

    def _callback(self, msg: Twist) -> None:
        self._pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/cmd_vel_remote")
    parser.add_argument("--output", default="/cmd_vel")
    args = parser.parse_args()

    rclpy.init()
    node = CmdVelRelay(args.input, args.output)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
