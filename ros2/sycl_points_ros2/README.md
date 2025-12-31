# sycl_points_ros2

This package provides a ROS 2 wrapper for the `sycl_points` library, offering a LiDAR odometry node that leverages SYCL for high-performance point cloud processing on heterogeneous systems (CPU, Intel iGPU, NVIDIA GPU).

## Prerequisites

Before building this package, ensure that you have installed `sycl_points` and its dependencies (Intel oneAPI, etc.). For detailed instructions, please refer to the main sycl_points README.

- ROS 2 (Jazzy or later recommended)
- `sycl_points` library

## Build

To build the ROS 2 wrapper, follow these steps. First, source the Intel oneAPI environment setup script. Then, build the workspace using `colcon`.

```bash
# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# ROS 2 setup script
source /opt/ros/jazzy/setup.bash

# Make workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone the sycl_points repository
git clone https://github.com/fateshelled/sycl_points.git

# Build workspace
cd ~/ros2_ws
colcon build --symlink-install --packages-up-to sycl_points_ros2
```

## Launching the Node

Source your workspace setup file and launch the odometry node.

```bash
# Setup environment
source /opt/intel/oneapi/setvars.sh
source ~/ros2_ws/install/setup.bash

# Set own parameters
POINT_TOPIC=/your/pointcloud/topic
FRAME_ID=/your/lidar/frame
ROSBAG=/path/to/your/rosbag

# Launch
ros2 launch sycl_points_ros2 lidar_odometry_launch.py \
    point_topic:=${POINT_TOPIC} \
    lidar_frame_id:=${FRAME_ID} \
    sycl/device_type:=gpu \
    sycl/device_vendor:=intel \
    rviz2:=true \
    rosbag/play:=true \
    rosbag/uri:=${ROSBAG} \
    use_sim_time:=true
```

## Node Details

### `lidar_odometry_node`

This is a composable node that performs LiDAR odometry estimation.

#### Subscribed Topics

- **`points`** (`sensor_msgs/msg/PointCloud2`)
  - The input point cloud. The topic name can be remapped via the `point_topic` launch argument.

#### Published Topics

- **`sycl_lo/preprocessed`** (`sensor_msgs/msg/PointCloud2`)
  - The preprocessed align point cloud.
- **`sycl_lo/submap`** (`sensor_msgs/msg/PointCloud2`)
  - The local map point cloud used for registration.
- **`sycl_lo/odom`** (`nav_msgs/msg/Odometry`)
  - The estimated odometry of the sensor.
- **`sycl_lo/keyframe/pose`** (`nav_msgs/msg/Odometry`)
  - The keyframe pose of the sensor.
- **`sycl_lo/pose`** (`geometry_msgs::msg::PoseStamped`)
  - The estimated odometry pose of the sensor.

## Parameters tuning

### LiDAR

| parameter                              | recommend value | description                                                                                    |
| -------------------------------------- | --------------: | ---------------------------------------------------------------------------------------------- |
| scan/preprocess/box_filter/max         |           100.0 | remove points far away from the sensor                                                         |
| scan/preprocess/box_filter/min         |             1.0 | remove points near the sensor (e.g., vehicle body, operator)                                   |
| registration/velocity_update/enable    |            true | enable deskew ※                                                                                |
| scan/downsampling/polar/coord_system   |           LIDAR | coordinate system                                                                              |
| scan/downsampling/polar/elevation_size |         0.01745 | elevation grid size (1 degrees)                                                                |
| scan/downsampling/polar/azimuth_size   |         0.05236 | azimuth grid size  (recommend 0.01745 ~ 0.05236 radians, 1 ~ 3 degrees )                       |
| submap/voxel_size                      |             0.5 | Larger values reduce processing time but result in a coarser map. (recommend 0.2 ~ 1.0　meter) |

```bash
DESKEW=true # e.g. Ouster, Velodyne
# DESKEW=false # e.g. Livox, Solid state

ros2 launch sycl_points_ros2 lidar_odometry_launch.py \
    point_topic:=${POINT_TOPIC} \
    lidar_frame_id:=${FRAME_ID} \
    sycl/device_type:=gpu \
    sycl/device_vendor:=intel \
    rviz2:=true \
    rosbag/play:=true \
    use_sim_time:=true \
    rosbag/uri:=${ROSBAG} \
    scan/preprocess/box_filter/max:=100.0 \
    scan/preprocess/box_filter/min:=1.0 \
    registration/velocity_update/enable:=${DESKEW} \
    scan/downsampling/polar/coord_system:=LIDAR \
    scan/downsampling/polar/elevation_size:=0.017453292519943295 \
    scan/downsampling/polar/azimuth_size:=0.05235987755982989 \
    submap/voxel_size:=0.5
```

### Depth camera

| parameter                              | recommend value | description                                                                                    |
| -------------------------------------- | --------------: | ---------------------------------------------------------------------------------------------- |
| scan/preprocess/box_filter/max         |            10.0 | remove points far away from the sensor                                                         |
| scan/preprocess/box_filter/min         |             0.3 | remove points near the sensor (e.g., vehicle body, operator)                                   |
| registration/velocity_update/enable    |           false | disable deskew                                                                                 |
| scan/downsampling/polar/coord_system   |          CAMERA | coordinate system                                                                              |
| scan/downsampling/polar/elevation_size |         0.01745 | elevation grid size (1 degrees)                                                                |
| scan/downsampling/polar/azimuth_size   |         0.01745 | azimuth grid size (1 degrees)                                                                  |
| submap/voxel_size                      |             0.2 | Larger values reduce processing time but result in a coarser map. (recommend 0.2 ~ 1.0　meter) |

```bash
ros2 launch sycl_points_ros2 lidar_odometry_launch.py \
    point_topic:=${POINT_TOPIC} \
    lidar_frame_id:=${FRAME_ID} \
    sycl/device_type:=gpu \
    sycl/device_vendor:=intel \
    rviz2:=true \
    rosbag/play:=true \
    use_sim_time:=true \
    rosbag/uri:=${ROSBAG} \
    scan/preprocess/box_filter/max:=10.0 \
    scan/preprocess/box_filter/min:=0.0 \
    registration/velocity_update/enable:=false \
    scan/downsampling/polar/coord_system:=CAMERA \
    scan/downsampling/polar/elevation_size:=0.01745 \
    scan/downsampling/polar/azimuth_size:=0.01745 \
    submap/voxel_size:=0.2
```

#### ※ registration/velocity_update/enable (Deskew)

`registration/velocity_update/enable` parameter should be set according to your LiDAR type.

- **Spinning LiDAR (e.g., Ouster, Velodyne):** Set to `true`. This is effective because physically adjacent points have similar timestamps.
  - `registration/velocity_update/enable:=true`
- **Random Scan LiDAR (e.g., Livox):** Set to `false`. Since physically adjacent points do not necessarily have similar timestamps, enabling this may degrade accuracy.
  - `registration/velocity_update/enable:=false`
