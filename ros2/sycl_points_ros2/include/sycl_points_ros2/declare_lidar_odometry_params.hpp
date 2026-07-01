#pragma once

#include <sycl_points/pipeline/lidar_odometry_params.hpp>

#include "sycl_points_ros2/declare_odometry_common_params.hpp"

namespace sycl_points {
namespace ros2 {

inline void declare_registration_robust_schedule_parameters(
    rclcpp::Node* node, algorithms::registration::RegistrationRobustScheduleParams& params) {
    params.init_scale = node->declare_parameter<double>("registration/robust/init_scale", params.init_scale);
    params.auto_scale = node->declare_parameter<bool>("registration/robust/auto_scale", params.auto_scale);
    params.min_scale = node->declare_parameter<double>("registration/robust/min_scale", params.min_scale);
    params.auto_scaling_iter =
        node->declare_parameter<int64_t>("registration/robust/auto_scaling_iter", params.auto_scaling_iter);
    params.rotation_init_scale = node->declare_parameter<double>("registration/rotation_constraint/robust/init_scale",
                                                                 params.rotation_init_scale);
    params.rotation_min_scale =
        node->declare_parameter<double>("registration/rotation_constraint/robust/min_scale", params.rotation_min_scale);
}

inline pipeline::lidar_odometry::Parameters declare_lidar_odometry_parameters(rclcpp::Node* node) {
    // Declare shared odometry parameters (scan, submap, registration, IMU, ...).
    pipeline::lidar_odometry::Parameters params;
    static_cast<pipeline::odometry::CommonParameters&>(params) = declare_odometry_common_parameters(node);

    // Declare LiDAR-only odometry parameters.
    params.imu.enable = node->declare_parameter<bool>("imu/enable", params.imu.enable);

    // Motion prediction without tightly-coupled LIO.
    {
        params.motion_prediction.verbose =
            node->declare_parameter<bool>("motion_prediction/verbose", params.motion_prediction.verbose);
        params.motion_prediction.velocity_ema_alpha = node->declare_parameter<double>(
            "motion_prediction/velocity/ema_alpha", params.motion_prediction.velocity_ema_alpha);

        params.motion_prediction.adaptive.rotation.factor_min = node->declare_parameter<double>(
            "motion_prediction/adaptive/rotation/factor/min", params.motion_prediction.adaptive.rotation.factor_min);
        params.motion_prediction.adaptive.rotation.factor_max = node->declare_parameter<double>(
            "motion_prediction/adaptive/rotation/factor/max", params.motion_prediction.adaptive.rotation.factor_max);
        params.motion_prediction.adaptive.rotation.min_eigenvalue_low =
            node->declare_parameter<double>("motion_prediction/adaptive/rotation/min_eigenvalue/low",
                                            params.motion_prediction.adaptive.rotation.min_eigenvalue_low);
        params.motion_prediction.adaptive.rotation.min_eigenvalue_high =
            node->declare_parameter<double>("motion_prediction/adaptive/rotation/min_eigenvalue/high",
                                            params.motion_prediction.adaptive.rotation.min_eigenvalue_high);
        params.motion_prediction.adaptive.translation.factor_min =
            node->declare_parameter<double>("motion_prediction/adaptive/translation/factor/min",
                                            params.motion_prediction.adaptive.translation.factor_min);
        params.motion_prediction.adaptive.translation.factor_max =
            node->declare_parameter<double>("motion_prediction/adaptive/translation/factor/max",
                                            params.motion_prediction.adaptive.translation.factor_max);
        params.motion_prediction.adaptive.translation.min_eigenvalue_low =
            node->declare_parameter<double>("motion_prediction/adaptive/translation/min_eigenvalue/low",
                                            params.motion_prediction.adaptive.translation.min_eigenvalue_low);
        params.motion_prediction.adaptive.translation.min_eigenvalue_high =
            node->declare_parameter<double>("motion_prediction/adaptive/translation/min_eigenvalue/high",
                                            params.motion_prediction.adaptive.translation.min_eigenvalue_high);
    }

    auto& registration = params.lo.registration;
    auto& pipeline = params.lo.pipeline;

    declare_registration_optimization_parameters(node, registration.optimization);
    declare_registration_robust_schedule_parameters(node, pipeline.robust);

    // LiDAR-only registration pipeline loop controls.
    {
        registration.max_iterations =
            node->declare_parameter<int64_t>("registration/solver_iterations", registration.max_iterations);
        registration.criteria.translation =
            node->declare_parameter<double>("registration/criteria/translation", registration.criteria.translation);
        registration.criteria.rotation =
            node->declare_parameter<double>("registration/criteria/rotation", registration.criteria.rotation);

        auto& velocity_update = pipeline.velocity_update;
        velocity_update.enable =
            node->declare_parameter<bool>("registration/velocity_update/enable", velocity_update.enable);
        velocity_update.iter =
            node->declare_parameter<int64_t>("registration/velocity_update/iter", velocity_update.iter);
    }

    // MAP prior using the previous LiDAR odometry frame Hessian.
    {
        auto& map_prior = registration.map_prior;
        map_prior.enabled = node->declare_parameter<bool>("registration/map_prior/enabled", map_prior.enabled);
        map_prior.rot_vel_sigma =
            node->declare_parameter<double>("registration/map_prior/rot_vel_sigma", map_prior.rot_vel_sigma);
        map_prior.trans_vel_sigma =
            node->declare_parameter<double>("registration/map_prior/trans_vel_sigma", map_prior.trans_vel_sigma);
        map_prior.rot_base_sigma =
            node->declare_parameter<double>("registration/map_prior/rot_base_sigma", map_prior.rot_base_sigma);
        map_prior.trans_base_sigma =
            node->declare_parameter<double>("registration/map_prior/trans_base_sigma", map_prior.trans_base_sigma);
    }

    // LiDAR-only degeneracy regularization. LIO uses directional ICP weighting.
    {
        auto& degenerate_reg = registration.degenerate_regularization;

        const std::string degenerate_reg_type =
            node->declare_parameter<std::string>("registration/degenerate_regularization/type", "NONE");
        degenerate_reg.type = algorithms::registration::DegenerateRegularizationType_from_string(degenerate_reg_type);

        degenerate_reg.base_factor = node->declare_parameter<double>(
            "registration/degenerate_regularization/nl_reg/base_factor", degenerate_reg.base_factor);
        degenerate_reg.trans_eigenvalue_threshold =
            node->declare_parameter<double>("registration/degenerate_regularization/nl_reg/trans_eigenvalue_threshold",
                                            degenerate_reg.trans_eigenvalue_threshold);
        degenerate_reg.rot_eigenvalue_threshold =
            node->declare_parameter<double>("registration/degenerate_regularization/nl_reg/rot_eigenvalue_threshold",
                                            degenerate_reg.rot_eigenvalue_threshold);
    }

    return params;
}
}  // namespace ros2

}  // namespace sycl_points
