from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config_dir = Path(get_package_share_directory("turtlebot_obstacle_fusion")) / "config"
    return LaunchDescription([
        Node(
            package="turtlebot_obstacle_fusion",
            executable="obstacle_fusion",
            name="obstacle_fusion",
            output="screen",
            parameters=[str(config_dir / "fusion.yaml")],
        ),
        Node(
            package="turtlebot_obstacle_fusion",
            executable="rat_walker",
            name="rat_walker",
            output="screen",
            parameters=[str(config_dir / "rat_walker.yaml")],
        ),
    ])
