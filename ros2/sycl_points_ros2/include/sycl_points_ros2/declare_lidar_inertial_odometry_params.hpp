#pragma once

#include <rclcpp/node.hpp>
#include <sycl_points/pipeline/lidar_inertial_odometry_params.hpp>

#include "sycl_points_ros2/declare_odometry_common_params.hpp"

namespace sycl_points {
namespace ros2 {

inline pipeline::lidar_inertial_odometry::Parameters declare_lidar_inertial_odometry_parameters(rclcpp::Node* node) {
    // Declare shared odometry parameters (scan, submap, registration, IMU, ...).
    // LIO uses directional ICP weighting, so LiDAR-only degenerate regularization
    // is intentionally not declared here.
    pipeline::lidar_inertial_odometry::Parameters params;
    static_cast<pipeline::odometry::CommonParameters&>(params) = declare_odometry_common_parameters(node);

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
    auto& registration = params.lio.registration;
    declare_registration_optimization_parameters(node, registration.optimization);
    registration.robust.init_scale =
        node->declare_parameter<double>("registration/robust/init_scale", registration.robust.init_scale);
    registration.robust.auto_scale =
        node->declare_parameter<bool>("registration/robust/auto_scale", registration.robust.auto_scale);
    registration.robust.min_scale =
        node->declare_parameter<double>("registration/robust/min_scale", registration.robust.min_scale);
    registration.robust.auto_scaling_iter = node->declare_parameter<int64_t>("registration/robust/auto_scaling_iter",
                                                                             registration.robust.auto_scaling_iter);
    registration.robust.rotation_init_scale = node->declare_parameter<double>(
        "registration/rotation_constraint/robust/init_scale", registration.robust.rotation_init_scale);
    registration.robust.rotation_min_scale = node->declare_parameter<double>(
        "registration/rotation_constraint/robust/min_scale", registration.robust.rotation_min_scale);
    registration.total_iterations =
        static_cast<size_t>(node->declare_parameter<int64_t>("lio/total_iterations", registration.total_iterations));
    registration.criteria.rotation =
        node->declare_parameter<double>("lio/criteria/rotation", registration.criteria.rotation);
    registration.criteria.translation =
        node->declare_parameter<double>("lio/criteria/translation", registration.criteria.translation);
    registration.invalid_regularization_factor = node->declare_parameter<double>(
        "lio/invalid_regularization_factor", registration.invalid_regularization_factor);
    params.lio.preintegration_reset.fd_velocity_sigma = static_cast<float>(
        node->declare_parameter<double>("lio/fd_velocity_sigma", params.lio.preintegration_reset.fd_velocity_sigma));
    params.lio.preintegration_reset.icp_rotation_sigma = static_cast<float>(
        node->declare_parameter<double>("lio/icp_rotation_sigma", params.lio.preintegration_reset.icp_rotation_sigma));
    registration.directional_icp_weighting.enable = node->declare_parameter<bool>(
        "lio/directional_icp_weighting/enable", registration.directional_icp_weighting.enable);
    registration.directional_icp_weighting.trans_min_eigenvalue_per_inlier = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/trans_min_eigenvalue_per_inlier",
                                        registration.directional_icp_weighting.trans_min_eigenvalue_per_inlier));
    registration.directional_icp_weighting.rot_min_eigenvalue_per_inlier = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/rot_min_eigenvalue_per_inlier",
                                        registration.directional_icp_weighting.rot_min_eigenvalue_per_inlier));
    registration.directional_icp_weighting.trans_weak_direction_scale = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/trans_weak_direction_scale",
                                        registration.directional_icp_weighting.trans_weak_direction_scale));
    registration.directional_icp_weighting.rot_weak_direction_scale = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/rot_weak_direction_scale",
                                        registration.directional_icp_weighting.rot_weak_direction_scale));

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
