# sycl_points_ros2

`sycl_points_ros2` provides ROS 2 components and executables for SYCL-accelerated
LiDAR odometry (LO), LiDAR-inertial odometry (LIO), and sliding-window graph
odometry. Each mode supports live ROS topics and direct offline rosbag evaluation.

## Prerequisites

- A ROS 2 installation. The examples below use ROS 2 Jazzy.
- A C++20 compiler and Eigen.
- Intel oneAPI DPC++ or AdaptiveCpp.
- A SYCL device with unified shared memory support.

See the repository [README](../../README.md) for supported devices, runtime device
filtering, and general installation instructions. AdaptiveCpp setup is documented
in [install_adaptive_cpp.md](../../docs/install_adaptive_cpp.md). The current
DPC++ NVIDIA limitation is documented in
[install_dpcpp_nvidia.md](../../docs/install_dpcpp_nvidia.md).

Intel CPUs, Intel GPUs, and NVIDIA GPUs are the supported targets. AMD GPUs may be
exposed by AdaptiveCpp, but they are untested and unsupported by this project.

## Build

Clone this repository into a ROS 2 workspace. Building `sycl_points_ros2` with
`--packages-up-to` also builds the local `sycl_points` package; a separate install
of that package is not required.

Intel oneAPI DPC++ is the default SYCL implementation:

```bash
source /opt/intel/oneapi/setvars.sh
source /opt/ros/jazzy/setup.bash

cd ~/ros2_ws
colcon build --symlink-install \
  --packages-up-to sycl_points_ros2 \
  --cmake-args -DSYCL_IMPL=IntelDPCPP
```

For AdaptiveCpp, do not source oneAPI solely for this package:

```bash
source /opt/ros/jazzy/setup.bash

cd ~/ros2_ws
colcon build --symlink-install \
  --packages-up-to sycl_points_ros2 \
  --cmake-args -DSYCL_IMPL=AdaptiveCpp
```

Use a clean CMake build directory, or remove the affected CMake cache, when
switching between `IntelDPCPP` and `AdaptiveCpp` or changing AdaptiveCpp targets.
CMake caches the compiler and target selection.

After building, source the workspace before running any example:

```bash
source ~/ros2_ws/install/setup.bash
```

The package manifest lists its core ROS dependencies, but is not an exhaustive
list of every tool used by launch files, such as RViz and Python launch support.

## Choose A Mode

| Mode                          | Live executable                | Live launch file                    | Use case                                 | IMU requirement                                         |
| ----------------------------- | ------------------------------ | ----------------------------------- | ---------------------------------------- | ------------------------------------------------------- |
| LiDAR odometry (LO)           | `lidar_odometry_node`          | `lidar_odometry_launch.py`          | Frame-to-submap LiDAR odometry           | LiDAR-only by default; optional when `imu/enable:=true` |
| LiDAR-inertial odometry (LIO) | `lidar_inertial_odometry_node` | `lidar_inertial_odometry_launch.py` | Tightly coupled LiDAR and IMU estimation | Required                                                |
| Graph odometry                | `graph_odometry_node`          | `graph_odometry_launch.py`          | Sliding-window graph LIO                 | Required                                                |

Graph odometry is not a LiDAR-only mode. It validates that `imu/enable=true`,
`imu/initial_alignment/enable=true`, `imu/deskew/enable=true`, and
`motion_prediction/mode=IMU_SE3` at startup.

The same library also registers the corresponding component plugins:
`sycl_points::ros2::LiDAROdometryNode`,
`sycl_points::ros2::LidarInertialOdometryNode`, and
`sycl_points::ros2::GraphOdometryNode`. The evaluator plugins are
`sycl_points::ros2::LiDAROdometryBagEvalNode`,
`sycl_points::ros2::LidarInertialOdometryBagEvalNode`, and
`sycl_points::ros2::GraphOdometryBagEvalNode`.

## Save The Submap On Exit

For LO, LIO, and graph LIO (both live and bag evaluation nodes), configure:

```yaml
submap/save_on_exit: true
submap/output_path: "/path/to/submap.pcd" # .pcd or .ply
submap/output_binary: true
```

On node destruction (including normal Ctrl+C shutdown), the node saves all exportable
points still retained in the internal map in the odom frame, not just the local
registration target limited by `submap/max_distance_range`. Existing voxel
aggregation, occupancy thresholds, sampling, and stale-voxel removal remain
unchanged; previously removed points and raw scan history are not recovered.
Graph LIO exports only points already inserted into the internal map, not scans
still in the active optimization window.
The existing writer includes RGB when present; intensity is also included in PLY
(the PCD writer exports XYZ and optional RGB only). Saving is disabled by default.
Parent directories are created automatically, relative paths use the process
working directory, and an existing output file is overwritten. Empty maps are
skipped with a warning; write failures are logged without throwing from shutdown.
SIGKILL or a process crash cannot guarantee saving. Bag evaluation saves when
the node is destroyed, not merely when bag reading finishes with `eval/exit_on_end=false`.

## Live Quick Start

Launch LO with its defaults:

```bash
ros2 launch sycl_points_ros2 lidar_odometry_launch.py
```

Launch LIO with LiDAR and IMU topics:

```bash
ros2 launch sycl_points_ros2 lidar_inertial_odometry_launch.py \
  points_topic:=/lidar/points \
  imu_topic:=/imu/data
```

Launch graph odometry:

```bash
ros2 launch sycl_points_ros2 graph_odometry_launch.py \
  points_topic:=/lidar/points \
  imu_topic:=/imu/data
```

Common launch defaults are `points_topic=points`, `imu_topic=imu/data`,
`odom_frame_id=odom`, `base_link_id=base_link`, `lidar_frame_id=os_sensor`,
`rviz2=true`, and `use_sim_time=false`. Representative overrides include:

```bash
ros2 launch sycl_points_ros2 lidar_odometry_launch.py \
  points_topic:=/lidar/points \
  odom_frame_id:=odom \
  base_link_id:=base_link \
  lidar_frame_id:=lidar \
  sycl/device_vendor:=intel \
  sycl/device_type:=gpu \
  rviz2:=false
```

Every entry under `ros__parameters` in the mode's YAML file is exposed as a
same-named launch argument. The defaults documented here are launch/YAML defaults.
Running an executable directly uses constructor fallbacks for any parameter not
provided. Notable fallback differences are point QoS depth 1 instead of 10, IMU
keep-all instead of keep-last depth 100, enabled intensity and reflectivity
conversion instead of disabled, and an LIO timer period of 1 ms instead of 5 ms.
Prefer the launch files or pass a complete parameter file when reproducibility
matters.

The live launch files run the components with intra-process communication in a
multi-threaded component container. Container startup is delayed by one second so
the rosbag player can load first when enabled.

### Live Rosbag Playback

Each live launch file can start an in-process rosbag player:

```bash
ros2 launch sycl_points_ros2 lidar_inertial_odometry_launch.py \
  rosbag/play:=true \
  rosbag/uri:=/data/run_01 \
  rosbag/start_offset/sec:=5 \
  use_sim_time:=true
```

| Argument                  | Description                              | Default |
| ------------------------- | ---------------------------------------- | ------- |
| `rosbag/play`             | Start the in-process rosbag player       | `false` |
| `rosbag/uri`              | Bag URI                                  | Empty   |
| `rosbag/start_offset/sec` | Integer playback start offset in seconds | `0`     |
| `rosbag/exclude_topics`   | Comma-separated topics to exclude        | Empty   |

Playback uses a read-ahead queue of 1000 messages, rate `1.0`, no loop, and an
unpaused start. If the bag contains a `/tf_static` transform that conflicts with
the transform created by the launch file, exclude it:

```bash
ros2 launch sycl_points_ros2 graph_odometry_launch.py \
  rosbag/play:=true \
  rosbag/uri:=/data/run_01 \
  rosbag/exclude_topics:=/tf_static \
  use_sim_time:=true
```

TF exclusion affects the displayed TF tree, not odometry estimation; the
odometry pipeline does not consume TF.

## Input Requirements

### Point Clouds

The topic selected by `points_topic` must use
`sensor_msgs/msg/PointCloud2`. Empty clouds are rejected. Supported fields are:

| Field                                      | Accepted type                                      | Control and behavior                                                                                               |
| ------------------------------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `x`, `y`, `z`                              | All three must be `FLOAT32` or all three `FLOAT64` | Required                                                                                                           |
| `rgb` or `rgba`                            | `FLOAT32` or `UINT32`                              | Copied when `input/convert_rgb=true`                                                                               |
| `intensity`                                | `FLOAT32` or `UINT16`                              | Copied when `input/convert_intensity=true`                                                                         |
| `reflectivity`                             | `UINT16`                                           | Preferred over `intensity` when both `input/convert_intensity=true` and `input/use_reflectivity_as_intensity=true` |
| `t`, `time`, `timestamp`, or `offset_time` | `UINT32`, `FLOAT32`, or `FLOAT64`                  | Converted to milliseconds relative to the message header stamp; see the type-specific interpretation below         |

Ouster enhanced reflectivity, enabled with
`scan/enhanced_reflectivity/enable=true`, operates on converted raw `intensity`
and additionally requires `ambient: UINT16` and `ring: UINT8` or `UINT16` in the
input message. Set `input/convert_intensity=true` and keep
`input/use_reflectivity_as_intensity=false` to use the raw `intensity` field.
Missing or invalid supporting fields cause enhanced reflectivity to be skipped
with a warning.

The accepted per-point time types are interpreted as follows before conversion
to milliseconds relative to the message header stamp:

| Type                         | Interpretation                     |
| ---------------------------- | ---------------------------------- |
| `UINT32`                     | Nanoseconds relative to scan start |
| `FLOAT32`                    | Seconds relative to scan start     |
| `FLOAT64` greater than `1e8` | Absolute Unix seconds              |
| Other non-negative `FLOAT64` | Seconds relative to scan start     |

Per-point time is needed for scan-time deskew. LIO and graph odometry also wait
for IMU samples covering the frame timestamp and, when timestamped points are
present, the full scan interval.

### IMU

The topic selected by `imu_topic` must use `sensor_msgs/msg/Imu`. The nodes use
the header timestamp, angular velocity, and linear acceleration. The message
orientation and frame ID are not consumed. Configure `T_imu_to_lidar/*` with the
sensor extrinsic and set `imu/accel_unit` to match the input units.

LO subscribes to IMU only when `imu/enable=true`. LIO and graph odometry always
require both point cloud and IMU streams.

## ROS Topics And Frames

### Subscriptions

| Mode  | Point cloud subscription                       | IMU subscription                                           |
| ----- | ---------------------------------------------- | ---------------------------------------------------------- |
| LO    | `points_topic` (`sensor_msgs/msg/PointCloud2`) | `imu_topic` (`sensor_msgs/msg/Imu`) when `imu/enable=true` |
| LIO   | `points_topic` (`sensor_msgs/msg/PointCloud2`) | `imu_topic` (`sensor_msgs/msg/Imu`)                        |
| Graph | `points_topic` (`sensor_msgs/msg/PointCloud2`) | `imu_topic` (`sensor_msgs/msg/Imu`)                        |

Subscription history, depth, and reliability are controlled by `points_qos/*`
and `imu_qos/*`. The YAML defaults use keep-last depths of 10 for points and 100
for IMU, both with best-effort reliability.

### Publications

Replace `<prefix>` with `sycl_lo`, `sycl_lio`, or `sycl_go` for LO, LIO, or graph
odometry respectively. Standard live publishers use QoS depth 5.

| Topic                    | Type                            | Content                                          |
| ------------------------ | ------------------------------- | ------------------------------------------------ |
| `<prefix>/preprocessed`  | `sensor_msgs/msg/PointCloud2`   | Preprocessed scan in the input message frame     |
| `<prefix>/submap`        | `sensor_msgs/msg/PointCloud2`   | Registration submap relabeled to `odom_frame_id` |
| `<prefix>/odom`          | `nav_msgs/msg/Odometry`         | Current base-link pose                           |
| `<prefix>/keyframe/pose` | `nav_msgs/msg/Odometry`         | Accepted keyframe base-link pose                 |
| `<prefix>/pose`          | `geometry_msgs/msg/PoseStamped` | Current base-link pose                           |

Debug point clouds are converted and published only while they have subscribers.
LO also publishes covariance markers on the configurable
`vis/covariance_markers/scan/topic_name`, whose YAML default is
`sycl_lo/scan_covariance_markers`.

The estimator state is the LiDAR pose. ROS odometry, pose, keyframe, TF, and TUM
outputs convert it to the base-link pose using `T_base_link_to_lidar`.

For live launches:

- The launch file publishes the static transform `base_link_id -> lidar_frame_id`
  from `T_base_link_to_lidar/*`.
- The odometry node publishes the dynamic transform
  `odom_frame_id -> base_link_id`.
- `lidar_frame_id` selects the child frame of the launch-created static transform;
  it does not transform or validate incoming clouds.
- The pipeline performs no TF lookup and does not transform the input cloud based
  on its frame ID.

Offline bag evaluators do not publish ROS topics or TF. They write a TUM file and
processing statistics to the log.

## Offline Bag Evaluation

Bag evaluators read messages directly in bag order without running
`ros2 bag play`:

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

| Mode  | Evaluator executable                    | Default TUM output  |
| ----- | --------------------------------------- | ------------------- |
| LO    | `lidar_odometry_bag_eval_node`          | `sycl_lo_odom.tum`  |
| LIO   | `lidar_inertial_odometry_bag_eval_node` | `sycl_lio_odom.tum` |
| Graph | `graph_odometry_bag_eval_node`          | `sycl_go_odom.tum`  |

| Argument                        | Description                                     | Default                   |
| ------------------------------- | ----------------------------------------------- | ------------------------- |
| `rosbag/uri`                    | Input bag URI; must not be empty                | Empty                     |
| `rosbag/start_offset/sec`       | Start offset in seconds                         | `0.0`                     |
| `eval/output_tum`               | Output TUM path                                 | Mode-specific value above |
| `eval/write_first_frame`        | Include the initialized first-frame pose        | `true`                    |
| `eval/exit_on_end`              | Shut down after the one-shot evaluation         | `true`                    |
| `eval/max_pending_point_clouds` | Graph-only limit while waiting for IMU coverage | `10000`                   |

TUM rows contain the base-link pose in this exact order:

```text
timestamp tx ty tz qx qy qz qw
```

The evaluator creates the output's parent directories and truncates an existing
file. With `eval/exit_on_end=false`, evaluation still runs once, then the process
continues spinning.

LO and LIO evaluators consume clouds in bag order. Graph evaluation queues clouds
until matching IMU coverage is available, bounded by
`eval/max_pending_point_clouds`. It reports expired frames and clouds left
incomplete at overflow or bag end; the resulting TUM file is partial. Evaluation
exceptions are caught and logged internally, so process exit status alone is not
a reliable success signal. Inspect the final statistics and error log.

## Configuration

Launch defaults and the complete parameter inventories are in:

- [config/lidar_odometry.yaml](config/lidar_odometry.yaml)
- [config/lidar_inertial_odometry.yaml](config/lidar_inertial_odometry.yaml)
- [config/graph_odometry.yaml](config/graph_odometry.yaml)

The graph launch merges wildcard (`/**`) parameters with node-specific overrides.
Important configuration groups include:

- Device, input conversion, subscription QoS, frames, initial pose, and sensor
  extrinsics.
- Scan preprocessing, intensity processing, downsampling, covariance estimation,
  and submapping.
- LO registration and motion prediction, LIO optimization and IMU settings, or
  graph factors and sliding-window optimization.

`sycl/device_vendor` accepts case-insensitive `default`, `intel`, `nvidia`, and
`amd`; AdaptiveCpp also accepts `omp`. `sycl/device_type` accepts `cpu` or `gpu`.
Selection fails if the requested device is invalid or unavailable. Runtime
visibility filters such as `ONEAPI_DEVICE_SELECTOR` and `ACPP_VISIBILITY_MASK`
are described in the repository [README](../../README.md).

### Deskew Selection

Two different mechanisms must not be confused:

- `registration/velocity_update/enable` is LiDAR registration velocity-update
  deskew. It is useful for sequential spinning-LiDAR scans with per-point time.
  Random-scan sensors may not benefit because nearby points need not be adjacent
  in time.
- `imu/deskew/enable` uses buffered IMU measurements before downsampling and
  registration. It requires per-point time plus IMU coverage for the scan
  interval. Graph odometry requires it; LIO enables it in the default YAML.

Tune from the mode's YAML rather than treating scenario-specific values as
universal defaults.
