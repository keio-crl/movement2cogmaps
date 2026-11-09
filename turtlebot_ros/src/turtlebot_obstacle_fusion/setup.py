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
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TurtleBot maintainer",
    maintainer_email="maintainer@example.invalid",
    description="Conservative LiDAR and OAK-D obstacle range fusion.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["obstacle_fusion = turtlebot_obstacle_fusion.node:main"]},
)
