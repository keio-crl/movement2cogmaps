"""rat_walker with obstacles from the LiDAR and Depth Anything V2 on the OAK-D colour preview.

Runs on the development PC. The depth node needs torch; its Python is the ``depth_python`` launch
argument (default ``$DEPTH_PYTHON`` or ``~/tomita-turtlebot/.venv/bin/python``).
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
        DeclareLaunchArgument("depth_config", default_value=str(config_dir / "depth_obstacle_oakd.yaml")),
        DeclareLaunchArgument("record_images", default_value="false", description="Save the camera image and Depth Anything V2 depth for every CSV row"),
        DeclareLaunchArgument("recorder_config", default_value=str(config_dir / "rat_recorder.yaml")),
        ExecuteProcess(
            cmd=[LaunchConfiguration("depth_python"), "-m", "turtlebot_obstacle_fusion.rat_recorder_node", "--ros-args", "--params-file", LaunchConfiguration("recorder_config")],
            output="screen",
            condition=IfCondition(LaunchConfiguration("record_images")),
        ),
        DeclareLaunchArgument("rviz", default_value="false", description="Show LiDAR, fused scan, depth memory and walls in RViz"),
        ExecuteProcess(
            cmd=[LaunchConfiguration("depth_python"), "-m", "turtlebot_obstacle_fusion.depth_obstacle_node", "--ros-args", "--params-file", LaunchConfiguration("depth_config")],
            output="screen",
        ),
        Node(
            package="turtlebot_obstacle_fusion",
            executable="scan_depth_fusion",
            name="scan_depth_fusion",
            output="screen",
            parameters=[str(config_dir / "scan_depth_fusion.yaml")],
        ),
        Node(
            package="turtlebot_obstacle_fusion",
            executable="rat_walker",
            name="rat_walker",
            output="screen",
            parameters=[str(config_dir / "rat_walker.yaml"), {"record_images": ParameterValue(LaunchConfiguration("record_images"), value_type=bool)}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", str(config_dir / "obstacles.rviz")],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
