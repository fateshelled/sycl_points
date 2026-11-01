# sycl_points_ros2

ROS 2 nodes for SYCL-accelerated LiDAR odometry (LO), LiDAR-inertial odometry
(LIO), and sliding-window graph odometry. Each supports live topics and offline
rosbag evaluation.

## Prerequisites

- ROS 2 (examples use Jazzy), C++20, Eigen, Intel oneAPI DPC++ or AdaptiveCpp.
- A SYCL device with unified shared memory: Intel CPU/GPU or NVIDIA GPU.
  AMD GPUs are untested and unsupported.

See the repository [README](../../README.md) for installation and device selection,
[AdaptiveCpp setup](../../docs/install_adaptive_cpp.md), and
[DPC++ NVIDIA limitations](../../docs/install_dpcpp_nvidia.md).

## Build

Clone the repository into a ROS 2 workspace. Build with DPC++ (the default):

```bash
source /opt/intel/oneapi/setvars.sh
source /opt/ros/jazzy/setup.bash

cd ~/ros2_ws
colcon build --symlink-install \
  --packages-up-to sycl_points_ros2 \
  --cmake-args -DSYCL_IMPL=IntelDPCPP
```

Or use AdaptiveCpp (without sourcing oneAPI):

```bash
source /opt/ros/jazzy/setup.bash

cd ~/ros2_ws
colcon build --symlink-install \
  --packages-up-to sycl_points_ros2 \
  --cmake-args -DSYCL_IMPL=AdaptiveCpp
```

`--packages-up-to` also builds `sycl_points`. When changing SYCL implementations
or AdaptiveCpp targets, use a clean build directory or clear the affected CMake
cache. Before running, source the workspace:

```bash
source ~/ros2_ws/install/setup.bash
```

## Choose A Mode

| Mode | Live executable | Launch file | IMU |
| ---- | --------------- | ----------- | --- |
| LO | `lidar_odometry_node` | `lidar_odometry_launch.py` | Optional (`imu/enable=true`) |
| LIO | `lidar_inertial_odometry_node` | `lidar_inertial_odometry_launch.py` | Required |
| Graph | `graph_odometry_node` | `graph_odometry_launch.py` | Required |

Graph mode requires `imu/enable`, `imu/initial_alignment/enable`, and
`imu/deskew/enable` to be true, plus `motion_prediction/mode=IMU_SE3`.
All modes also provide ROS 2 components and bag-evaluation nodes.

## Save The Submap On Exit

For any mode (live or bag evaluation):

```yaml
submap/save_on_exit: true
submap/output_path: "/path/to/submap.pcd" # .pcd or .ply
submap/output_binary: true
```

On normal shutdown (including Ctrl+C), save all points retained in the internal
map in the odom frame. This is not limited by `submap/max_distance_range`, but
removed voxels and raw scans cannot be recovered. Graph mode excludes scans
still in the optimization window.

PCD and PLY include RGB, intensity, and covariance when available. Covariance
uses six fields (`cov_xx`, `cov_xy`, `cov_xz`, `cov_yy`, `cov_yz`, `cov_zz`);
the readers reconstruct the symmetric 3x3 matrix.

The node default is off; check each mode's YAML for launch defaults. Parent
directories are created, relative paths use the process working directory, and
existing files are overwritten. Empty maps are skipped. Crashes and SIGKILL
cannot guarantee a save. Bag evaluation saves on node destruction, not just
when bag reading ends.

## Live Quick Start

```bash
ros2 launch sycl_points_ros2 lidar_odometry_launch.py
ros2 launch sycl_points_ros2 lidar_inertial_odometry_launch.py \
  points_topic:=/lidar/points imu_topic:=/imu/data
ros2 launch sycl_points_ros2 graph_odometry_launch.py \
  points_topic:=/lidar/points imu_topic:=/imu/data
```

The first scan may take longer to process because the SYCL runtime can JIT-compile
device kernels on first use. Startup time depends on the backend and device.

Common launch defaults: `points_topic=points`, `imu_topic=imu/data`,
`odom_frame_id=odom`, `base_link_id=base_link`, `lidar_frame_id=os_sensor`,
`rviz2=true`, `use_sim_time=false`. Any YAML parameter can be overridden by a
same-named launch argument (e.g. `sycl/device_type:=gpu`). Direct execution
without a parameter file uses different defaults; use launch files for
reproducible settings.

### Live Rosbag Playback

Start an in-process rosbag player from any live launch:

```bash
ros2 launch sycl_points_ros2 lidar_inertial_odometry_launch.py \
  rosbag/play:=true \
  rosbag/uri:=/data/run_01 \
  rosbag/start_offset/sec:=5 \
  use_sim_time:=true
```

`rosbag/play` defaults to false; `rosbag/start_offset/sec` defaults to 0.
Use `rosbag/exclude_topics:=/tf_static` if the bag conflicts with the launch
file's static TF. TF is not used for odometry estimation.

## Input Requirements

### Point Clouds

`points_topic` must carry nonempty `sensor_msgs/msg/PointCloud2` messages:

| Field | Type | Notes |
| ----- | ---- | ----- |
| `x`, `y`, `z` | All `FLOAT32` or all `FLOAT64` | Required |
| `rgb` / `rgba` | `FLOAT32` / `UINT32` | Requires `input/convert_rgb=true` |
| `intensity` | `FLOAT32` / `UINT16` | Requires `input/convert_intensity=true` |
| `reflectivity` | `UINT16` | Preferred when `input/use_reflectivity_as_intensity=true` |
| `t`, `time`, `timestamp`, `offset_time` | `UINT32` / `FLOAT32` / `FLOAT64` | Per-point time |

Ouster enhanced reflectivity needs raw `intensity`, `ambient: UINT16`, and
`ring: UINT8` or `UINT16`. Set `input/convert_intensity=true`,
`input/use_reflectivity_as_intensity=false`, and
`scan/enhanced_reflectivity/enable=true`.

Per-point time is converted to milliseconds relative to the message timestamp:

| Type                         | Interpretation                     |
| ---------------------------- | ---------------------------------- |
| `UINT32`                     | Nanoseconds relative to scan start |
| `FLOAT32`                    | Seconds relative to scan start     |
| `FLOAT64` greater than `1e8` | Absolute Unix seconds              |
| Other non-negative `FLOAT64` | Seconds relative to scan start     |

Deskew needs per-point time. LIO and graph mode wait for IMU coverage of the
frame and, when timed points are present, the full scan.

### IMU

`imu_topic` must carry `sensor_msgs/msg/Imu`. Only timestamp, angular velocity,
and linear acceleration are used (not orientation or frame ID). Set
`T_imu_to_lidar/*` and `imu/accel_unit` for your sensor. LO uses IMU only when
`imu/enable=true`; LIO and graph mode require it.

## ROS Topics And Frames

All modes subscribe to `points_topic`; LIO and graph mode also subscribe to
`imu_topic` (LO only with `imu/enable=true`). Set `points_qos/*` and `imu_qos/*`
for subscription QoS; YAML defaults are best-effort, keep-last depths 10 and 100.

### Publications

Replace `<prefix>` with `sycl_lo`, `sycl_lio`, or `sycl_go` (LO/LIO/graph):

| Topic                    | Type                            | Content                                          |
| ------------------------ | ------------------------------- | ------------------------------------------------ |
| `<prefix>/preprocessed`  | `sensor_msgs/msg/PointCloud2`   | Preprocessed scan in the input message frame     |
| `<prefix>/submap`        | `sensor_msgs/msg/PointCloud2`   | Registration submap relabeled to `odom_frame_id` |
| `<prefix>/odom`          | `nav_msgs/msg/Odometry`         | Current base-link pose                           |
| `<prefix>/keyframe/pose` | `nav_msgs/msg/Odometry`         | Accepted keyframe base-link pose                 |
| `<prefix>/pose`          | `geometry_msgs/msg/PoseStamped` | Current base-link pose                           |

Debug clouds publish only when subscribed. LO also publishes scan covariance
markers (`vis/covariance_markers/scan/topic_name`). The launch file publishes
static `base_link_id -> lidar_frame_id` TF; the node publishes dynamic
`odom_frame_id -> base_link_id` TF. Estimation uses LiDAR poses; base-link outputs
apply `T_base_link_to_lidar`. Incoming clouds are **not** transformed via TF.
Offline evaluators publish no topics or TF; they write TUM poses and log statistics.

## Offline Bag Evaluation

Bag evaluators read messages directly; no `ros2 bag play` is needed:

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

Default outputs are `sycl_lo_odom.tum`, `sycl_lio_odom.tum`, and
`sycl_go_odom.tum`. Set `eval/output_tum` to change the path and
`eval/write_first_frame` (default `true`) to include the first pose.
Each TUM row contains the base-link pose:

```text
timestamp tx ty tz qx qy qz qw
```

`rosbag/start_offset/sec` defaults to `0.0`. Evaluation runs once;
`eval/exit_on_end=false` keeps the process alive afterward. Graph mode buffers
clouds awaiting IMU coverage (limit: `eval/max_pending_point_clouds`, default
`10000`); dropped or incomplete frames can make output partial. Check logs for
errors; exit status alone is not a success indicator.

## Configuration

Launch defaults and all parameters are in:

- [config/lidar_odometry.yaml](config/lidar_odometry.yaml)
- [config/lidar_inertial_odometry.yaml](config/lidar_inertial_odometry.yaml)
- [config/graph_odometry.yaml](config/graph_odometry.yaml)

`sycl/device_vendor` accepts `default`, `intel`, `nvidia`, `amd` (also `omp`
with AdaptiveCpp); `sycl/device_type` accepts `cpu` or `gpu`. Unavailable
devices cause startup failure. See the repository [README](../../README.md) for
runtime device filters.

### Deskew Selection

- `registration/velocity_update/enable`: registration-based deskew for timed,
  sequential scans; may not help random-scan sensors.
- `imu/deskew/enable`: IMU-based deskew before registration; needs per-point
  time and IMU coverage. Required in graph mode, enabled by default in LIO.
