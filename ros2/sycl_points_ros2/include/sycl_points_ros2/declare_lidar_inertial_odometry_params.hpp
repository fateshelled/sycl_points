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
    registration.icp_information_scale = static_cast<float>(
        node->declare_parameter<double>("lio/icp_information_scale", registration.icp_information_scale));
    params.lio.preintegration_reset.fd_velocity_sigma = static_cast<float>(
        node->declare_parameter<double>("lio/fd_velocity_sigma", params.lio.preintegration_reset.fd_velocity_sigma));
    params.lio.preintegration_reset.icp_rotation_sigma = static_cast<float>(
        node->declare_parameter<double>("lio/icp_rotation_sigma", params.lio.preintegration_reset.icp_rotation_sigma));
    params.lio.initial_covariance.accel_bias_sigma = static_cast<float>(node->declare_parameter<double>(
        "lio/initial_covariance/accel_bias_sigma", params.lio.initial_covariance.accel_bias_sigma));
    params.lio.initial_covariance.gyro_bias_sigma = static_cast<float>(node->declare_parameter<double>(
        "lio/initial_covariance/gyro_bias_sigma", params.lio.initial_covariance.gyro_bias_sigma));
    registration.directional_icp_weighting.enable = node->declare_parameter<bool>(
        "lio/directional_icp_weighting/enable", registration.directional_icp_weighting.enable);
    registration.directional_icp_weighting.verbose = node->declare_parameter<bool>(
        "lio/directional_icp_weighting/verbose", registration.directional_icp_weighting.verbose);
    registration.directional_icp_weighting.type = algorithms::lio::DirectionalIcpWeightingType_from_string(
        node->declare_parameter<std::string>("lio/directional_icp_weighting/type", "SCALE"));
    registration.directional_icp_weighting.trans_min_information_ratio = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/trans_min_information_ratio",
                                        registration.directional_icp_weighting.trans_min_information_ratio));
    registration.directional_icp_weighting.rot_min_information_ratio = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/rot_min_information_ratio",
                                        registration.directional_icp_weighting.rot_min_information_ratio));
    registration.directional_icp_weighting.trans_imu_information_floor_per_inlier = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/trans_imu_information_floor_per_inlier",
                                        registration.directional_icp_weighting.trans_imu_information_floor_per_inlier));
    registration.directional_icp_weighting.rot_imu_information_floor_per_inlier = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/rot_imu_information_floor_per_inlier",
                                        registration.directional_icp_weighting.rot_imu_information_floor_per_inlier));
    registration.directional_icp_weighting.trans_weak_direction_scale = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/trans_weak_direction_scale",
                                        registration.directional_icp_weighting.trans_weak_direction_scale));
    registration.directional_icp_weighting.rot_weak_direction_scale = static_cast<float>(
        node->declare_parameter<double>("lio/directional_icp_weighting/rot_weak_direction_scale",
                                        registration.directional_icp_weighting.rot_weak_direction_scale));
    registration.directional_icp_weighting.use_schur_complement = node->declare_parameter<bool>(
        "lio/directional_icp_weighting/use_schur_complement",
        registration.directional_icp_weighting.use_schur_complement);
    registration.directional_icp_weighting.schur_relative_cutoff = node->declare_parameter<double>(
        "lio/directional_icp_weighting/schur_relative_cutoff",
        registration.directional_icp_weighting.schur_relative_cutoff);
    registration.directional_icp_weighting.schur_absolute_cutoff = node->declare_parameter<double>(
        "lio/directional_icp_weighting/schur_absolute_cutoff",
        registration.directional_icp_weighting.schur_absolute_cutoff);

    // Constant-velocity prior on the world-frame velocity state
    registration.constant_velocity_prior.enable = node->declare_parameter<bool>(
        "lio/constant_velocity_prior/enable", registration.constant_velocity_prior.enable);
    registration.constant_velocity_prior.verbose = node->declare_parameter<bool>(
        "lio/constant_velocity_prior/verbose", registration.constant_velocity_prior.verbose);
    registration.constant_velocity_prior.min_eigenvalue_ratio = static_cast<float>(node->declare_parameter<double>(
        "lio/constant_velocity_prior/min_eigenvalue_ratio", registration.constant_velocity_prior.min_eigenvalue_ratio));
    registration.constant_velocity_prior.min_information_per_inlier = static_cast<float>(
        node->declare_parameter<double>("lio/constant_velocity_prior/min_information_per_inlier",
                                        registration.constant_velocity_prior.min_information_per_inlier));
    registration.constant_velocity_prior.degenerate_velocity_sigma = static_cast<float>(
        node->declare_parameter<double>("lio/constant_velocity_prior/degenerate_velocity_sigma",
                                        registration.constant_velocity_prior.degenerate_velocity_sigma));
    registration.constant_velocity_prior.observable_velocity_sigma = static_cast<float>(
        node->declare_parameter<double>("lio/constant_velocity_prior/observable_velocity_sigma",
                                        registration.constant_velocity_prior.observable_velocity_sigma));

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
