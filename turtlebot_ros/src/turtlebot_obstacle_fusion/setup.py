from glob import glob

from setuptools import find_packages, setup


package_name = "turtlebot_obstacle_fusion"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/turtlebot_obstacle_fusion"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml") + glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TurtleBot maintainer",
    maintainer_email="maintainer@example.invalid",
    description="LiDAR and OAK-D obstacle fusion and a RatInABox-style random walker.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "obstacle_fusion = turtlebot_obstacle_fusion.node:main",
            "rat_walker = turtlebot_obstacle_fusion.rat_walker_node:main",
            "rat_behaviour_cli = turtlebot_obstacle_fusion.behaviour_cli:main",
            "collision_test = turtlebot_obstacle_fusion.collision_test_node:main",
            "collision_test_cli = turtlebot_obstacle_fusion.collision_test_cli:main",
            "scan_depth_fusion = turtlebot_obstacle_fusion.scan_depth_fusion_node:main",
            "stereo_obstacle = turtlebot_obstacle_fusion.stereo_obstacle_node:main",
        ]
    },
)
