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

/// @brief Declare graph factor registration / linearization parameters under `graph/factor/*`.
/// Graph-only: the single-frame align path keeps its own `registration/*` keys, so this must
/// NOT be called from declare_lidar_odometry_parameters (shared with lidar_odometry / lidar_inertial_odometry).
inline void declare_graph_registration_parameters(rclcpp::Node* node,
                                                  pipeline::lidar_odometry::Parameters& params) {
    auto& reg = params.graph.registration;
    auto& factor = reg.factor;

    reg.min_num_points = node->declare_parameter<int64_t>("graph/factor/min_num_points", reg.min_num_points);

    auto& random_sampling = reg.random_sampling;
    random_sampling.enable =
        node->declare_parameter<bool>("graph/factor/random_sampling/enable", random_sampling.enable);
    random_sampling.num = node->declare_parameter<int64_t>("graph/factor/random_sampling/num", random_sampling.num);
    random_sampling.use_intensities = node->declare_parameter<bool>(
        "graph/factor/random_sampling/use_intensities", random_sampling.use_intensities);
    random_sampling.weighted_ratio = static_cast<float>(node->declare_parameter<double>(
        "graph/factor/random_sampling/weighted_ratio", random_sampling.weighted_ratio));
    if (random_sampling.weighted_ratio < 0.0f || random_sampling.weighted_ratio > 1.0f) {
        throw std::invalid_argument(
            "[declare_graph_registration_parameters] `graph/factor/random_sampling/weighted_ratio` must be "
            "within [0.0, 1.0]");
    }

    const std::string reg_type = node->declare_parameter<std::string>("graph/factor/type", "gicp");
    factor.reg_type = algorithms::registration::RegType_from_string(reg_type);
    factor.verbose = node->declare_parameter<bool>("graph/factor/verbose", factor.verbose);

    factor.max_correspondence_distance = node->declare_parameter<double>(
        "graph/factor/max_correspondence_distance", factor.max_correspondence_distance);

    auto& rotation_constraint = factor.rotation_constraint;
    auto& rotation_robust = rotation_constraint.robust;
    rotation_constraint.enable =
        node->declare_parameter<bool>("graph/factor/rotation_constraint/enable", rotation_constraint.enable);
    rotation_constraint.weight =
        node->declare_parameter<double>("graph/factor/rotation_constraint/weight", rotation_constraint.weight);
    rotation_robust.default_scale = node->declare_parameter<double>(
        "graph/factor/rotation_constraint/robust/default_scale", rotation_robust.default_scale);
}

/// @brief Declare the tip-only constant-velocity deskew parameters under
/// `graph/velocity_update/*`. Graph-only mirror of the align path's
/// `registration/velocity_update/*`; must NOT be called from
/// declare_lidar_odometry_parameters (shared with lidar_odometry / LIO).
inline void declare_graph_velocity_update_parameters(rclcpp::Node* node,
                                                     pipeline::lidar_odometry::Parameters& params) {
    auto& velocity_update = params.graph.velocity_update;
    velocity_update.enable =
        node->declare_parameter<bool>("graph/velocity_update/enable", velocity_update.enable);
    velocity_update.iter = static_cast<size_t>(node->declare_parameter<int64_t>(
        "graph/velocity_update/iter", static_cast<int64_t>(velocity_update.iter)));
    if (velocity_update.iter == 0) {
        throw std::invalid_argument(
            "[declare_graph_velocity_update_parameters] `graph/velocity_update/iter` must be >= 1");
    }
}

inline pipeline::lidar_odometry::Parameters declare_lidar_odometry_parameters(rclcpp::Node* node) {
    // Declare shared odometry parameters (scan, submap, registration, IMU, ...).
    pipeline::lidar_odometry::Parameters params;
    static_cast<pipeline::odometry::CommonParameters&>(params) = declare_odometry_common_parameters(node);

    // Declare LiDAR-only odometry parameters.
    params.imu.enable = node->declare_parameter<bool>("imu/enable", params.imu.enable);

    // Motion prediction without tightly-coupled LIO.
    {
        const std::string prediction_mode = node->declare_parameter<std::string>(
            "motion_prediction/mode",
            pipeline::lidar_odometry::MotionPredictionMode_to_string(params.motion_prediction.mode));
        params.motion_prediction.mode =
            pipeline::lidar_odometry::MotionPredictionMode_from_string(prediction_mode);
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
