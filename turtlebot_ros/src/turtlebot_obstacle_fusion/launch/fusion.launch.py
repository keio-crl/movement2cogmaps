from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    config_path = Path(get_package_share_directory("turtlebot_obstacle_fusion")) / "config" / "fusion.yaml"
    return LaunchDescription([
        Node(
            package="turtlebot_obstacle_fusion",
            executable="obstacle_fusion",
            name="obstacle_fusion",
            output="screen",
            parameters=[str(config_path)],
        )
    ])
