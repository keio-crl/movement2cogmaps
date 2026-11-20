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

`/scan` must be a `sensor_msgs/msg/LaserScan`. The OAK-D driver must publish a metric depth
`sensor_msgs/msg/Image` and a matching `sensor_msgs/msg/CameraInfo`; set `depth_topic` and
`camera_info_topic` to their actual names. The node transforms both inputs into `base_link`, so
the corresponding TF transforms must be available.

The node publishes:

- `/fused_scan`: 360-degree, 1-degree-bin conservative obstacle ranges in `base_link`.
- `/fused_segments`: RDP line segments derived from the LiDAR clusters, for RViz diagnostics.

Before connecting `/fused_scan` to motion control, verify it with a stationary robot, RViz, and
known-distance obstacles. Tune the YAML parameters after recording a rosbag from the actual robot.

## Rat-like random walk (RatInABox OU model)

`rat_walker` drives the robot with the RatInABox 2D motion model (OU processes on the rotational
velocity and on the Rayleigh-distributed speed, plus wall repulsion with thigmotaxis), scaled to
the robot with `length_scale` (s = 8.161) and `time_scale` (k = 7.248). Obstacles are the RDP line
segments of `/fused_scan` (per-angle minimum of LiDAR and depth). The model update runs every
0.02 k s, a CSV row is logged every 0.1 k s, and a session ends after 5129 s of recorded time.

When the next step would bring the robot body (radius + `collision_margin_m`) into an obstacle,
the robot stops, pauses recording, turns in place to the mirrored heading, then resumes at
0.5 × speed_mean and continues recording.

```bash
ros2 launch turtlebot_obstacle_fusion rat_walker.launch.py   # fusion + walker; robot stays still
ros2 run turtlebot_obstacle_fusion rat_behaviour_cli         # in another terminal
behaviour> adult      # crawl | walk | run | adult: start a new session with that behaviour
behaviour> status     # state, behaviour and recorded time
behaviour> stop       # stop the robot and close the log
```

Logs are written to `~/rat_walker_logs/<behaviour>_<date>.csv` with columns
`t, behaviour, x, y, yaw, cmd_linear, cmd_angular, v_net_x, v_net_y, rot_velocity, closest_wall`
(`x, y, yaw` from `/odom`). The robot holds still whenever `/fused_scan` or `/odom` is older than
`sensor_timeout_s`. Parameters are in `config/rat_walker.yaml`; check the actual publishers with
`ros2 topic info -v /odom` and `ros2 topic info -v /cmd_vel` on the robot before the first run.

## LiDAR collision-avoidance test

`collision_test` drives straight using only `/scan`, stops when the robot body
(radius + `collision_margin_m`) would hit an obstacle within the stopping distance
`v (control_period + latency) + v^2 / (2 braking_decel)`, turns 180 deg in place and repeats.

```bash
ros2 run turtlebot_obstacle_fusion collision_test --ros-args \
  --params-file install/turtlebot_obstacle_fusion/share/turtlebot_obstacle_fusion/config/collision_test.yaml \
  -p linear_speed_m_s:=0.1
```

It stops after `max_turns` turns (0 = until Ctrl+C). Each detection logs the clearance at
detection, the braking distance from odometry and the clearance after stopping; the full trace is
in `~/collision_test_logs/*.csv`. Start at 0.1 m/s and raise the speed only after the logged
clearance stays positive, recalibrating `braking_decel_m_s2` from the measured braking distance.

## Optional: LiDAR + Depth Anything V2 (camera depth)

LiDAR sees a single horizontal plane, so obstacles that overhang it (a chair seat above thin legs)
are missed. `depth_obstacle` estimates depth from `/image_raw/compressed` with Depth Anything V2
(ViT-S), converts the relative inverse depth `r` to metres with `1/z = a r + b` fitted every frame
against the floor (`scale_source: floor`: the metric depth of every floor pixel follows from the
camera pose; obstacles on the floor are rejected as outliers), keeps pixels between `obstacle_min_height_m`
and `obstacle_max_height_m` above the floor and publishes them on `/depth_obstacles` (360-bin
`LaserScan` in `base_link`). With `use_depth:=true`, `collision_test` also stops for these points
and holds still if they are older than `depth_timeout_s`.

Run it on the development PC (needs the GPU) through the interactive launcher:

```bash
source /opt/ros/galactic/setup.bash && source install/setup.bash
ros2 run turtlebot_obstacle_fusion collision_test_cli            # add --dry-run to publish to /cmd_vel_dryrun
# 1) LiDAR only   2) LiDAR + depth, then speed and number of turns
```

The depth node runs with `$DEPTH_PYTHON` (default `~/tomita-turtlebot/.venv/bin/python`, which has
torch and `depth_anything_v2`). The launcher asks for the camera:

- `config/depth_obstacle_oakd.yaml`: OAK-D colour preview (`/color/preview/image/compressed`). The
  intrinsics and lens distortion come from `/color/preview/camera_info` and the pose from TF
  (`oakd_pro_rgb_camera_optical_frame`); start the camera on the Pi with
  `ros2 launch turtlebot4_bringup oakd.launch.py`. Do not subscribe to the raw `/stereo/depth` from
  the PC: it saturates Wi-Fi and delays `/scan`.
- `config/depth_obstacle_wide.yaml`: wide USB camera (`/image_raw/compressed`, uncalibrated). Set
  `hfov_deg` and the mounting pose `cam_x/y/z`, `cam_yaw_deg`, `cam_pitch_deg`.

The camera model is correct when the projected LiDAR dots in the saved images lie on the obstacles
at LiDAR height.

Diagnostics go to `~/depth_obstacle_logs/<date>/` every `save_period_s`:

- `frame_XXXXX.png`: (1) camera image with depth obstacle pixels in red and projected LiDAR points
  coloured by distance, (2) metric depth (red = near, blue >= `depth_max_m`), (3) top view of LiDAR
  (grey) and depth obstacles (red) with the robot and collision body circles, (4) fit and distances.
- `frame_XXXXX.npz`: relative and metric depth (half resolution), obstacle points and pixels, LiDAR
  pairs used for the fit, `a, b`, intrinsics and camera pose.
`scale_source: lidar` fits against the LiDAR points projected into the image instead; it fails in
exactly the scenes that matter, because the LiDAR row in the image runs through the overhanging
parts (a chair base) that the LiDAR passes over.

- `frames.csv`: per saved frame `a, b`, fit pairs and inliers, fit residual, median depth-vs-LiDAR
  error at the projected LiDAR points (`lidar_check_err`, a sanity check), front minimum
  distance from depth and from LiDAR, inference time. Times match the `collision_test` CSV, whose
  `trigger` column says whether `lidar`, `depth` or `both` caused each stop.

## Optional: bumper reaction

With `use_bumper:=true` (asked by `collision_test_cli`), `collision_test` also listens to the
Create 3 `/hazard_detection` and, on a `BUMP` while driving or braking, stops, backs up
`backup_distance_m` at `backup_speed_m_s` (stopping early if LiDAR/depth sees something behind),
then turns 180 deg. It catches low obstacles that neither the LiDAR plane nor the camera sees (chair
bases), but not overhangs above the bumper. The robot holds still if `/hazard_detection` is stale.
The CSV `trigger` column is `bump` for these events.
