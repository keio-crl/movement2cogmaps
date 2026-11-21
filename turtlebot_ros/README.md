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
0.02 k s, a CSV row is logged every 0.1 k s, and a session ends after 5218.6 s of recorded time (7200 rows, matching the 720 s x 10 fps simulations).

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

Logs are written to `movement2cogmaps/rat_walker_logs/<behaviour>_<date>.csv` with columns
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
in `movement2cogmaps/collision_test_logs/*.csv`. Start at 0.1 m/s and raise the speed only after the logged
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
# 1) LiDAR only   2) LiDAR + Depth Anything   3) LiDAR + OAK-D stereo, then speed and number of turns
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

Diagnostics go to `movement2cogmaps/depth_obstacle_logs/<date>/` every `save_period_s`:

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

## Optional: LiDAR + OAK-D stereo depth

Option 3 of `collision_test_cli` starts `stereo_obstacle` (system Python, no torch) with
`config/stereo_obstacle.yaml`. It reads `/stereo/depth/compressedDepth` (16UC1 mm as PNG, about
10 KB per frame, about 6.6 Hz over Wi-Fi) and `/stereo/camera_info`, takes the pose of
`oakd_pro_right_camera_optical_frame` from TF, and publishes the same `/depth_obstacles` scan as the
Depth Anything node, so `collision_test` is unchanged.

With the Pi defaults (`dotProjectormA: 0`, IR dot projector off) under 1 % of the pixels have depth,
mostly on edges; plain walls and floors are not measured, and objects closer than about 0.35 m to
the camera (the 720p stereo minimum) give no depth or wrong depth.

The TF of the camera is level, but the floor seen by the stereo camera is tilted about 6.5 deg
(camera looking down); `camera_pitch_down_deg` / `camera_roll_deg` correct it. Without the
correction the floor beyond about 0.6 m rises above `obstacle_min_height_m` and becomes an obstacle.
Dense depth (a textured view) arrives only about every 2 s with the Pi defaults, so the launcher
runs `collision_test` with `depth_timeout_s:=2.5` and `depth_lost_turn_s:=5.0` for this option.

Diagnostics go to `movement2cogmaps/depth_obstacle_logs/stereo_<date>/` every `save_period_s`:
`stereo_XXXXX.png` (half-size depth map, red = near, black = no depth, white = obstacle pixels),
`stereo_XXXXX.npz` (`depth_mm`, obstacle points and pixels, intrinsics, camera pose) and
`frames.csv` (valid pixel ratio, number and median height of floor points, which should be near
0 m when the camera model is right, the floor tilt left after the correction (a rough per-frame
fit, +-2 deg), number of obstacle points, nearest obstacle).

## Optional: bumper reaction

With `use_bumper:=true` (asked by `collision_test_cli`), `collision_test` also listens to the
Create 3 `/hazard_detection` and, on a `BUMP` while driving or braking, stops, backs up
`backup_distance_m` at `backup_speed_m_s` (stopping early if LiDAR/depth sees something behind),
then turns 180 deg. It catches low obstacles that neither the LiDAR plane nor the camera sees (chair
bases), but not overhangs above the bumper. The robot holds still if `/hazard_detection` is stale.
The CSV `trigger` column is `bump` for these events.

## rat_walker with LiDAR + Depth Anything V2

`rat_walker_depth.launch.py` runs, on the development PC, `depth_obstacle` (OAK-D colour preview,
see above), `scan_depth_fusion` and `rat_walker`. `scan_depth_fusion` publishes `/fused_scan`: for
every 1-degree bin the nearer of the LiDAR (`/scan`, transformed to `base_link`) and the depth
obstacles, which are remembered in the odometry frame for `depth_memory_s` so low obstacles stay
known after they leave the camera view. `rat_walker` turns `/fused_scan` into RDP segments and applies
the RatInABox wall repulsion and the stop / mirror-turn / resume-at-half-speed bounce to them.

```bash
# Pi: LiDAR (turtlebot4 service) and `ros2 launch turtlebot4_bringup oakd.launch.py`
ros2 launch turtlebot_obstacle_fusion rat_walker_depth.launch.py
ros2 run turtlebot_obstacle_fusion rat_behaviour_cli     # another terminal: crawl | walk | run | adult | stop
```

Two deviations from RatInABox are needed on the robot (both in `config/rat_walker.yaml`):

- `merge_wall_chains`: perceived surfaces arrive as chains of short segments; summing the repulsion
  of every piece multiplied it (a wall in 30 pieces pushed about 7 times harder) and saturated the
  commands. A chain now contributes once per local distance minimum; separate walls (a box corner)
  still add up as in RatInABox.
- `heading_gain`: the heading command is `rotational_velocity + heading_gain * heading_error`
  instead of `heading_error / dt`, which overshot with the command latency and oscillated at the
  angular limit.

`rat_behaviour_cli` asks for the wall repulsion distance after a behaviour (Enter keeps the scaled
`0.1 x length_scale` = 0.816 m). A smaller value lets the robot move between closely spaced
obstacles but departs from the scaled RatInABox model; it is sent as `"<behaviour> <metres>"` and
recorded in the log file name (`<behaviour>_repel<metres>_<date>.csv`).

With `use_bumper: true` (default in `config/rat_walker.yaml`), a Create 3 `BUMP` while running stops
the robot and the recording, backs up `backup_distance_m`, bounces the model velocity off the contact
point (bumper zone direction at the robot radius; the new heading always leads away from it), turns
to that heading and resumes at 0.5 x speed_mean, as for a predicted collision. The robot holds still
while `/hazard_detection` is stale.

### Recording camera images and depth with rat_walker

`record_images:=true` (both `rat_walker.launch.py` and `rat_walker_depth.launch.py`) starts
`rat_recorder` with `$DEPTH_PYTHON`. Every time rat_walker writes a CSV row it publishes
`"<session>,<row>,<t>"` on `/rat_walker/sample`; the recorder saves the newest OAK-D colour preview
and its Depth Anything V2 depth. rat_walker holds still and stops the recorded time while
`/rat_recorder/status` is not `ready` (model loading, or no image newer than `max_image_age_s`), so
every CSV row has its image.

```bash
ros2 launch turtlebot_obstacle_fusion rat_walker_depth.launch.py record_images:=true
```

Files go to `<log_dir>/<session>/`, where `<session>` is the CSV file name without `.csv`, and
`XXXXX` is the `row` column of the CSV:

- `rgb_XXXXX.png`: the camera image at its original resolution.
- `depth_XXXXX.npz`: `relative` (Depth Anything V2 output, relative inverse depth, float32),
  `depth_m` (metres, empty when the scale fit failed), `fit` (`a, b, residual, inliers, pairs` of
  `1/z = a r + b`), `scale_source`, `intrinsics`, `distortion`, camera `rotation`/`translation` in
  `base_link`, `row`, `t`, `image_stamp`, `image_age_s`.
- `frames.csv`: one line per row with the image age, fit result and inference time.

The image topic, camera frame and scale settings are in `config/rat_recorder.yaml` (defaults
match `depth_obstacle_oakd.yaml`); its `log_dir` must equal rat_walker's `log_dir`.

### Visualising the obstacles in RViz

`ros2 launch turtlebot_obstacle_fusion rat_walker_depth.launch.py rviz:=true` (or, separately,
`rviz2 -d install/turtlebot_obstacle_fusion/share/turtlebot_obstacle_fusion/config/obstacles.rviz`)
shows, in `base_link` from above: LiDAR `/scan` (grey), `/fused_scan` (blue), remembered depth
obstacles (orange, fading with age), the latest depth obstacles (red), the walls rat_walker uses
(green), the repulsion range (yellow circle), the model heading (magenta arrow) and a status text.

The same RViz configuration also shows `collision_test` (`/collision_test/markers`): the depth memory
(orange) and latest depth (red), the collision body (green circle), the body at the lookahead that
decides the stop (red circle) and the state / trigger text. Open it in another terminal with the
`rviz2 -d ...` command above while `collision_test_cli` runs.
