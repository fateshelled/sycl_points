# sycl_points_ros2

This package provides ROS 2 nodes for three types of odometry: LiDAR odometry (LO), LiDAR-inertial odometry (LIO), and sliding-window graph odometry. \
The nodes use SYCL for acceleration. Each type supports live topics and offline rosbag evaluation.

## Prerequisites

- ROS 2 (the examples use Jazzy), C++20, Eigen, and Intel oneAPI DPC++ or AdaptiveCpp.
- A SYCL device with unified shared memory: an Intel CPU or GPU, or an NVIDIA GPU.
  This project has not tested AMD GPUs and does not support them.

For installation and device selection, see the repository [README](../../README.md). \
For the two compiler implementations, see [AdaptiveCpp setup](../../docs/install_adaptive_cpp.md) and [DPC++ NVIDIA limitations](../../docs/install_dpcpp_nvidia.md).

## Build

Clone the repository into a ROS 2 workspace. \
To use DPC++ (the default), build the package with these commands:

```bash
source /opt/intel/oneapi/setvars.sh
source /opt/ros/jazzy/setup.bash

cd ~/ros2_ws
colcon build --symlink-install \
  --packages-up-to sycl_points_ros2 \
  --cmake-args -DSYCL_IMPL=IntelDPCPP
```

To use AdaptiveCpp instead, build the package with these commands.  \
Do not source oneAPI:

```bash
source /opt/ros/jazzy/setup.bash

cd ~/ros2_ws
colcon build --symlink-install \
  --packages-up-to sycl_points_ros2 \
  --cmake-args -DSYCL_IMPL=AdaptiveCpp
```

The `--packages-up-to` option also builds `sycl_points`. \
If you change the SYCL implementation or AdaptiveCpp targets, use a new build directory or clear the CMake cache for the package. \
Before you run a node, source the workspace:

```bash
source ~/ros2_ws/install/setup.bash
```

## Choose A Mode

| Mode | Live executable | Launch file | IMU use |
| ---- | --------------- | ----------- | --- |
| LO | `lidar_odometry_node` | `lidar_odometry_launch.py` | Only if `imu/enable=true` |
| LIO | `lidar_inertial_odometry_node` | `lidar_inertial_odometry_launch.py` | Must have IMU data |
| Graph | `graph_odometry_node` | `graph_odometry_launch.py` | Must have IMU data |

For graph mode, set `imu/enable`, `imu/initial_alignment/enable`, and `imu/deskew/enable` to `true`. \
Set `motion_prediction/mode=IMU_SE3` as well.

All modes also provide ROS 2 components and bag-evaluation nodes.

## Save The Submap On Exit

For each mode, live or bag evaluation, use these parameters:

```yaml
submap/save_on_exit: true
submap/output_path: "/path/to/submap.pcd" # .pcd or .ply
submap/output_binary: true
```

On normal shutdown, including Ctrl+C, the node saves all points that remain in its internal map in the odom frame. \
The `submap/max_distance_range` parameter does not limit this output. The node cannot recover removed voxels or raw scans. \
In graph mode, the output does not include scans that are still in the optimization window.

PCD and PLY files contain RGB, intensity, and covariance when these data are available. \
Covariance has six fields: `cov_xx`, `cov_xy`, `cov_xz`, `cov_yy`, `cov_yz`, and `cov_zz`. \
The readers use these fields to make the symmetric 3x3 matrix.

The node default for `submap/save_on_exit` is `false`. For launch defaults, examine the YAML file for each mode. \
The node creates parent directories. It resolves relative paths from the process working directory and overwrites existing files. \
It does not save empty maps. After a crash or SIGKILL, a save is not guaranteed. \
For bag evaluation, the node saves the map when the node is destroyed, not only when it finishes reading the bag.

## Live Quick Start

```bash
ros2 launch sycl_points_ros2 lidar_odometry_launch.py
ros2 launch sycl_points_ros2 lidar_inertial_odometry_launch.py \
  points_topic:=/lidar/points imu_topic:=/imu/data
ros2 launch sycl_points_ros2 graph_odometry_launch.py \
  points_topic:=/lidar/points imu_topic:=/imu/data
```

The first scan can take more time because the SYCL runtime can JIT-compile device kernels when it first uses them. \
Startup time depends on the backend and device.

These are common launch defaults: `points_topic=points`, `imu_topic=imu/data`,
`odom_frame_id=odom`, `base_link_id=base_link`, `lidar_frame_id=os_sensor`,
`rviz2=true`, `use_sim_time=false`. You can use a launch argument with the same name to override a YAML parameter. \
For example, use `sycl/device_type:=gpu`. Direct execution without a parameter file uses different defaults. \
To use the same settings each time, use launch files.

### Live Rosbag Playback

From any live launch, start the in-process rosbag player:

```bash
ros2 launch sycl_points_ros2 lidar_inertial_odometry_launch.py \
  rosbag/play:=true \
  rosbag/uri:=/data/run_01 \
  rosbag/start_offset/sec:=5 \
  use_sim_time:=true
```

The default for `rosbag/play` is `false`, and the default for `rosbag/start_offset/sec` is `0`. \
If the bag conflicts with the static TF from the launch file, use `rosbag/exclude_topics:=/tf_static`. \
Odometry estimation does not use TF.

## Input Requirements

### Point Clouds

The `points_topic` topic must carry `sensor_msgs/msg/PointCloud2` messages that are not empty:

| Field | Type | Notes |
| ----- | ---- | ----- |
| `x`, `y`, `z` | All `FLOAT32` or all `FLOAT64` | Required |
| `rgb` / `rgba` | `FLOAT32` / `UINT32` | Requires `input/convert_rgb=true` |
| `intensity` | `FLOAT32` / `UINT16` | Requires `input/convert_intensity=true` |
| `reflectivity` | `UINT16` | Preferred when `input/use_reflectivity_as_intensity=true` |
| `t`, `time`, `timestamp`, `offset_time` | `UINT32` / `FLOAT32` / `FLOAT64` | Per-point time |

For Ouster enhanced reflectivity, the input must have raw `intensity`, `ambient: UINT16`, and `ring: UINT8` or `UINT16`. \
Set `input/convert_intensity=true`. \
Set `input/use_reflectivity_as_intensity=false`. \
Set `scan/enhanced_reflectivity/enable=true`.

The node converts per-point time to milliseconds relative to the message timestamp:

| Type                         | Interpretation                     |
| ---------------------------- | ---------------------------------- |
| `UINT32`                     | Nanoseconds relative to scan start |
| `FLOAT32`                    | Seconds relative to scan start     |
| `FLOAT64` greater than `1e8` | Absolute Unix seconds              |
| Other non-negative `FLOAT64` | Seconds relative to scan start     |

For deskew, the input must have per-point time. \
LIO and graph mode wait for IMU coverage of the frame. If the points have time values, they wait for coverage of the full scan.

### IMU

The `imu_topic` topic must carry `sensor_msgs/msg/Imu` messages. \
The node uses only the timestamp, angular velocity, and linear acceleration. It does not use orientation or the frame ID. \
For your sensor, set `T_imu_to_lidar/*` and `imu/accel_unit`. \
LO uses IMU only if `imu/enable=true`. LIO and graph mode must have IMU data.

## ROS Topics And Frames

All modes subscribe to `points_topic`. LIO and graph mode also subscribe to `imu_topic`. \
LO subscribes to `imu_topic` only if `imu/enable=true`. \
For subscription QoS, set `points_qos/*` and `imu_qos/*`.  \
The YAML defaults use best-effort QoS with keep-last depths of 10 and 100.

### Publications

Replace `<prefix>` with `sycl_lo` for LO, `sycl_lio` for LIO, or `sycl_go` for graph mode:

| Topic                    | Type                            | Content                                          |
| ------------------------ | ------------------------------- | ------------------------------------------------ |
| `<prefix>/preprocessed`  | `sensor_msgs/msg/PointCloud2`   | Preprocessed scan in the input message frame     |
| `<prefix>/submap`        | `sensor_msgs/msg/PointCloud2`   | Registration submap relabeled to `odom_frame_id` |
| `<prefix>/odom`          | `nav_msgs/msg/Odometry`         | Current base-link pose                           |
| `<prefix>/keyframe/pose` | `nav_msgs/msg/Odometry`         | Accepted keyframe base-link pose                 |
| `<prefix>/pose`          | `geometry_msgs/msg/PoseStamped` | Current base-link pose                           |

The nodes publish debug clouds only when a subscriber is present. LO also publishes scan covariance markers (`vis/covariance_markers/scan/topic_name`). \
The launch file publishes static `base_link_id -> lidar_frame_id` TF. The node publishes dynamic `odom_frame_id -> base_link_id` TF. \
Estimation uses LiDAR poses. Base-link outputs apply `T_base_link_to_lidar`. The nodes do **not** transform incoming clouds with TF. \
Offline evaluators publish no topics or TF. They write TUM poses and log statistics.

## Offline Bag Evaluation

Bag evaluators read messages directly. You do not have to use `ros2 bag play`:

```bash
ros2 launch sycl_points_ros2 lidar_odometry_bag_eval_launch.py \
  rosbag/uri:=/data/run_01 \
  points_topic:=/lidar/points

ros2 launch sycl_points_ros2 lidar_inertial_odometry_bag_eval_launch.py \
  rosbag/uri:=/data/run_01 \
  points_topic:=/lidar/points \
  imu_topic:=/imu/data

ros2 launch sycl_points_ros2 graph_odometry_bag_eval_launch.py \
  rosbag/uri:=/data/run_01 \
  points_topic:=/lidar/points \
  imu_topic:=/imu/data
```

The default output files are `sycl_lo_odom.tum`, `sycl_lio_odom.tum`, and `sycl_go_odom.tum`. \
To change the path, set `eval/output_tum`. To include the first pose, set `eval/write_first_frame` to `true` (the default). \
Each TUM row contains the base-link pose:

```text
timestamp tx ty tz qx qy qz qw
```

The default for `rosbag/start_offset/sec` is `0.0`. Evaluation runs once. \
With `eval/exit_on_end=false`, the process remains active after evaluation. \
Graph mode keeps clouds in a buffer until it has IMU coverage. The `eval/max_pending_point_clouds` limit defaults to `10000`. \
If frames are dropped or incomplete, the output can be partial. Examine the logs for errors. \
The exit status alone does not show that evaluation succeeded.

## Configuration

These files contain launch defaults and all parameters:

- [config/lidar_odometry.yaml](config/lidar_odometry.yaml)
- [config/lidar_inertial_odometry.yaml](config/lidar_inertial_odometry.yaml)
- [config/graph_odometry.yaml](config/graph_odometry.yaml)

For `sycl/device_vendor`, use `default`, `intel`, `nvidia`, or `amd`. With AdaptiveCpp, you can also use `omp`. \
For `sycl/device_type`, use `cpu` or `gpu`. If a device is not available, startup fails. \
For runtime device filters, see the repository [README](../../README.md).

### Deskew Selection

- `registration/velocity_update/enable`: registration-based deskew for sequential scans with time values. For random-scan sensors, this feature can give no improvement.
- `imu/deskew/enable`: IMU-based deskew before registration. To use it, the input must have per-point time and IMU coverage.
  Set this parameter to `true` in graph mode. Its default is `true` in LIO.
