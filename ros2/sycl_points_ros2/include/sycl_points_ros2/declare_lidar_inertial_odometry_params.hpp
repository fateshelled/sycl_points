#pragma once

#include <rclcpp/node.hpp>
#include <sycl_points/pipeline/lidar_inertial_odometry_params.hpp>

#include "sycl_points_ros2/declare_lidar_odometry_params.hpp"

namespace sycl_points {
namespace ros2 {

inline pipeline::lidar_inertial_odometry::Parameters declare_lidar_inertial_odometry_parameters(rclcpp::Node* node) {
    // Declare all base lidar_odometry parameters (scan, submap, registration, IMU, …)
    pipeline::lidar_inertial_odometry::Parameters params;
    static_cast<pipeline::lidar_odometry::Parameters&>(params) = declare_lidar_odometry_parameters(node);

    // IMU noise densities for 15×15 covariance propagation (not in base declaration)
    params.imu.preintegration.gyro_noise_density = static_cast<float>(node->declare_parameter<double>(
        "imu/preintegration/gyro_noise_density", params.imu.preintegration.gyro_noise_density));
    params.imu.preintegration.accel_noise_density = static_cast<float>(node->declare_parameter<double>(
        "imu/preintegration/accel_noise_density", params.imu.preintegration.accel_noise_density));
    params.imu.preintegration.gyro_bias_rw_density = static_cast<float>(node->declare_parameter<double>(
        "imu/preintegration/gyro_bias_rw_density", params.imu.preintegration.gyro_bias_rw_density));
    params.imu.preintegration.accel_bias_rw_density = static_cast<float>(node->declare_parameter<double>(
        "imu/preintegration/accel_bias_rw_density", params.imu.preintegration.accel_bias_rw_density));

    // LIO-specific optimization parameters
    params.lio.max_iterations =
        static_cast<size_t>(node->declare_parameter<int64_t>("lio/max_iterations", params.lio.max_iterations));
    params.lio.criteria.rotation =
        node->declare_parameter<double>("lio/criteria/rotation", params.lio.criteria.rotation);
    params.lio.criteria.translation =
        node->declare_parameter<double>("lio/criteria/translation", params.lio.criteria.translation);
    params.lio.invalid_regularization_factor =
        node->declare_parameter<double>("lio/invalid_regularization_factor", params.lio.invalid_regularization_factor);
    params.lio.fd_velocity_sigma =
        static_cast<float>(node->declare_parameter<double>("lio/fd_velocity_sigma", params.lio.fd_velocity_sigma));
    params.lio.icp_rotation_sigma =
        static_cast<float>(node->declare_parameter<double>("lio/icp_rotation_sigma", params.lio.icp_rotation_sigma));

    return params;
}

}  // namespace ros2
}  // namespace sycl_points
