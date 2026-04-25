#pragma once

#include <rclcpp/node.hpp>
#include <sycl_points/pipeline/lidar_inertial_odometry_params.hpp>

#include "sycl_points_ros2/declare_lidar_odometry_params.hpp"

namespace sycl_points {
namespace ros2 {

inline pipeline::lidar_inertial_odometry::Parameters declare_lidar_inertial_odometry_parameters(
    rclcpp::Node* node) {
    // Declare all base lidar_odometry parameters (scan, submap, registration, IMU, …)
    pipeline::lidar_inertial_odometry::Parameters params;
    static_cast<pipeline::lidar_odometry::Parameters&>(params) = declare_lidar_odometry_parameters(node);

    // LIO-specific optimization parameters
    params.lio.max_iterations =
        static_cast<size_t>(node->declare_parameter<int64_t>("lio/max_iterations", params.lio.max_iterations));
    params.lio.rotation_convergence =
        node->declare_parameter<double>("lio/rotation_convergence", params.lio.rotation_convergence);
    params.lio.position_convergence =
        node->declare_parameter<double>("lio/position_convergence", params.lio.position_convergence);

    return params;
}

}  // namespace ros2
}  // namespace sycl_points
