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
    params.lio.total_iterations =
        static_cast<size_t>(node->declare_parameter<int64_t>("lio/total_iterations", params.lio.total_iterations));
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
    params.lio.velocity_fd_blend =
        static_cast<float>(node->declare_parameter<double>("lio/velocity_fd_blend", params.lio.velocity_fd_blend));
    params.lio.directional_icp_weighting.enable = node->declare_parameter<bool>(
        "lio/directional_icp_weighting/enable", params.lio.directional_icp_weighting.enable);
    params.lio.directional_icp_weighting.trans_min_eigenvalue_per_inlier = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/trans_min_eigenvalue_per_inlier",
                                        params.lio.directional_icp_weighting.trans_min_eigenvalue_per_inlier));
    params.lio.directional_icp_weighting.rot_min_eigenvalue_per_inlier = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/rot_min_eigenvalue_per_inlier",
                                        params.lio.directional_icp_weighting.rot_min_eigenvalue_per_inlier));
    params.lio.directional_icp_weighting.weak_direction_scale =
        static_cast<float>(node->declare_parameter<double>("lio/directional_icp_weighting/weak_direction_scale",
                                                           params.lio.directional_icp_weighting.weak_direction_scale));
    params.lio.directional_icp_weighting.max_icp_to_imu_ratio =
        static_cast<float>(node->declare_parameter<double>("lio/directional_icp_weighting/max_icp_to_imu_ratio",
                                                           params.lio.directional_icp_weighting.max_icp_to_imu_ratio));
    params.lio.directional_icp_weighting.imu_information_floor =
        static_cast<float>(node->declare_parameter<double>("lio/directional_icp_weighting/imu_information_floor",
                                                           params.lio.directional_icp_weighting.imu_information_floor));

    // Bias-estimation safeguards
    params.lio.bias_estimation.freeze_on_low_excitation = node->declare_parameter<bool>(
        "lio/bias_estimation/freeze_on_low_excitation", params.lio.bias_estimation.freeze_on_low_excitation);
    params.lio.bias_estimation.gyro_excitation_threshold = static_cast<float>(node->declare_parameter<double>(
        "lio/bias_estimation/gyro_excitation_threshold", params.lio.bias_estimation.gyro_excitation_threshold));
    params.lio.bias_estimation.accel_excitation_threshold = static_cast<float>(node->declare_parameter<double>(
        "lio/bias_estimation/accel_excitation_threshold", params.lio.bias_estimation.accel_excitation_threshold));
    params.lio.bias_estimation.max_accel_bias = static_cast<float>(node->declare_parameter<double>(
        "lio/bias_estimation/max_accel_bias", params.lio.bias_estimation.max_accel_bias));
    params.lio.bias_estimation.max_gyro_bias = static_cast<float>(
        node->declare_parameter<double>("lio/bias_estimation/max_gyro_bias", params.lio.bias_estimation.max_gyro_bias));
    return params;
}

}  // namespace ros2
}  // namespace sycl_points
