# TurtleBot ROS 2 workspace

This directory is the self-contained ROS 2 workspace to deploy to the TurtleBot 4 Raspberry Pi.
It contains only on-robot sensing and safety-adjacent processing. Keep RViz, rosbag analysis, and
offline experiments on the development PC.

## Deploy on the Raspberry Pi

Copy this `turtlebot_ros` directory to the Pi, then run:

```bash
cd turtlebot_ros
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch turtlebot_obstacle_fusion fusion.launch.py
```

Inspect the installed drivers first and override the input topics and frames if necessary:

```bash
ros2 topic list | rg 'scan|point'
ros2 run tf2_ros tf2_echo base_link <point-cloud-frame>
```

`/scan` must be a `sensor_msgs/msg/LaserScan`. The OAK-D driver must publish an XYZ
`sensor_msgs/msg/PointCloud2`; set `depth_cloud_topic` to that topic. The node transforms both
inputs into `base_link`, so the corresponding TF transforms must be available.

The node publishes:

- `/fused_scan`: 360-degree, 1-degree-bin conservative obstacle ranges in `base_link`.
- `/fused_segments`: RDP line segments derived from the LiDAR clusters, for RViz diagnostics.

Before connecting `/fused_scan` to motion control, verify it with a stationary robot, RViz, and
known-distance obstacles. Tune the YAML parameters after recording a rosbag from the actual robot.
