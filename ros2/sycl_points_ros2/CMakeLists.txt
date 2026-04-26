cmake_minimum_required(VERSION 3.15)

# SYCL implementation selection (must be set before project())
# Options: IntelDPCPP, AdaptiveCpp
set(SYCL_IMPL "IntelDPCPP" CACHE STRING "SYCL implementation to use (IntelDPCPP or AdaptiveCpp)")
set_property(CACHE SYCL_IMPL PROPERTY STRINGS "IntelDPCPP" "AdaptiveCpp")

# Set SYCL compiler before project()
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/SYCLCompiler.cmake")

project(sycl_points_ros2 LANGUAGES CXX)

# Initialize SYCL environment after compiler identification
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/SYCLSetup.cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(Eigen3 REQUIRED)

# ROS 2
ament_auto_find_build_dependencies()

ament_auto_add_library(lidar_odometry SHARED
  src/lidar_odometry_base_node.cpp
  src/lidar_odometry_bag_eval_node.cpp
  src/lidar_odometry_node.cpp
  src/lidar_inertial_odometry_node.cpp
)

target_include_directories(lidar_odometry PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(lidar_odometry
  Eigen3::Eigen
)

apply_sycl_settings(lidar_odometry)

rclcpp_components_register_node(
  lidar_odometry
  PLUGIN "sycl_points::ros2::LiDAROdometryNode"
  EXECUTABLE lidar_odometry_node
)

rclcpp_components_register_node(
  lidar_odometry
  PLUGIN "sycl_points::ros2::LiDAROdometryBagEvalNode"
  EXECUTABLE lidar_odometry_bag_eval_node
)

rclcpp_components_register_node(
  lidar_odometry
  PLUGIN "sycl_points::ros2::LidarInertialOdometryNode"
  EXECUTABLE lidar_inertial_odometry_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
  rviz2
  USE_SCOPED_HEADER_INSTALL_DIR
)
