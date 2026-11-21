"""rat_walker with the LiDAR/depth fusion node; ``record_images:=true`` also saves camera images and depth.

The image recorder needs torch; its Python is the ``depth_python`` launch argument (default
``$DEPTH_PYTHON`` or ``~/tomita-turtlebot/.venv/bin/python``).
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    config_dir = Path(get_package_share_directory("turtlebot_obstacle_fusion")) / "config"
    default_python = os.environ.get("DEPTH_PYTHON", str(Path.home() / "tomita-turtlebot" / ".venv" / "bin" / "python"))
    return LaunchDescription([
        DeclareLaunchArgument("depth_python", default_value=default_python),
        DeclareLaunchArgument("record_images", default_value="false", description="Save the camera image and Depth Anything V2 depth for every CSV row"),
        DeclareLaunchArgument("recorder_config", default_value=str(config_dir / "rat_recorder.yaml")),
        ExecuteProcess(
            cmd=[LaunchConfiguration("depth_python"), "-m", "turtlebot_obstacle_fusion.rat_recorder_node", "--ros-args", "--params-file", LaunchConfiguration("recorder_config")],
            output="screen",
            condition=IfCondition(LaunchConfiguration("record_images")),
        ),
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
            parameters=[str(config_dir / "rat_walker.yaml"), {"record_images": ParameterValue(LaunchConfiguration("record_images"), value_type=bool)}],
        ),
    ])
