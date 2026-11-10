"""Interactive terminal to select the rat behaviour (developmental stage) driven by rat_walker."""

from __future__ import annotations

import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from .rat_motion import BEHAVIOURS

HELP = f"Commands: {' | '.join(BEHAVIOURS)} (starts a new recording session) | stop | status | help | quit"


class BehaviourCli(Node):
    """Publish behaviour commands and keep the latest walker status."""

    def __init__(self) -> None:
        super().__init__("rat_behaviour_cli")
        self.declare_parameter("behaviour_topic", "/rat_behaviour")
        self.declare_parameter("status_topic", "/rat_walker/status")
        self.publisher = self.create_publisher(String, self.get_parameter("behaviour_topic").value, 10)
        self.status = "no status received yet"
        self.create_subscription(String, self.get_parameter("status_topic").value, self._status_callback, 10)

    def _status_callback(self, message: String) -> None:
        self.status = message.data

    def send(self, command: str) -> None:
        if self.publisher.get_subscription_count() == 0:
            print("Warning: rat_walker is not subscribed; the command may be lost.")
        self.publisher.publish(String(data=command))


def main() -> None:
    rclpy.init()
    node = BehaviourCli()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    print(HELP)
    try:
        while True:
            command = input("behaviour> ").strip().lower()
            if command in ("quit", "exit"):
                break
            if command == "status":
                print(node.status)
            elif command in BEHAVIOURS or command == "stop":
                node.send(command)
                print(f"Sent {command}.")
            elif command:
                print(HELP)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        executor.shutdown()
        spinner.join()
        node.destroy_node()
        rclpy.shutdown()
