#pragma once

#include <rclcpp/node.hpp>
#include <sycl_points/pipeline/lidar_odometry_params.hpp>

namespace sycl_points {
namespace ros2 {

inline pipeline::lidar_odometry::Parameters declare_lidar_odometry_parameters(rclcpp::Node* node) {
    pipeline::lidar_odometry::Parameters params;

    // SYCL
    {
        params.device.vendor = node->declare_parameter<std::string>("sycl/device_vendor", params.device.vendor);
        params.device.type = node->declare_parameter<std::string>("sycl/device_type", params.device.type);
    }

    // scan
    {
        params.scan.intensity_correction.enable =
            node->declare_parameter<bool>("scan/intensity_correction/enable", params.scan.intensity_correction.enable);
        params.scan.intensity_correction.exp =
            node->declare_parameter<double>("scan/intensity_correction/exp", params.scan.intensity_correction.exp);
        params.scan.intensity_correction.scale =
            node->declare_parameter<double>("scan/intensity_correction/scale", params.scan.intensity_correction.scale);
        params.scan.intensity_correction.min_intensity = node->declare_parameter<double>(
            "scan/intensity_correction/min_intensity", params.scan.intensity_correction.min_intensity);
        params.scan.intensity_correction.max_intensity = node->declare_parameter<double>(
            "scan/intensity_correction/max_intensity", params.scan.intensity_correction.max_intensity);
        params.scan.downsampling.voxel.enable =
            node->declare_parameter<bool>("scan/downsampling/voxel/enable", params.scan.downsampling.voxel.enable);
        params.scan.downsampling.voxel.size =
            node->declare_parameter<double>("scan/downsampling/voxel/voxel_size", params.scan.downsampling.voxel.size);

        params.scan.downsampling.polar.enable =
            node->declare_parameter<bool>("scan/downsampling/polar/enable", params.scan.downsampling.polar.enable);
        params.scan.downsampling.polar.distance_size = node->declare_parameter<double>(
            "scan/downsampling/polar/distance_size", params.scan.downsampling.polar.distance_size);
        params.scan.downsampling.polar.elevation_size = node->declare_parameter<double>(
            "scan/downsampling/polar/elevation_size", params.scan.downsampling.polar.elevation_size);
        params.scan.downsampling.polar.azimuth_size = node->declare_parameter<double>(
            "scan/downsampling/polar/azimuth_size", params.scan.downsampling.polar.azimuth_size);
        params.scan.downsampling.polar.coord_system = node->declare_parameter<std::string>(
            "scan/downsampling/polar/coord_system", params.scan.downsampling.polar.coord_system);
        params.scan.downsampling.random.enable =
            node->declare_parameter<bool>("scan/downsampling/random/enable", params.scan.downsampling.random.enable);
        params.scan.downsampling.random.num =
            node->declare_parameter<int64_t>("scan/downsampling/random/num", params.scan.downsampling.random.num);

        params.scan.preprocess.box_filter.enable = node->declare_parameter<bool>(
            "scan/preprocess/box_filter/enable", params.scan.preprocess.box_filter.enable);
        params.scan.preprocess.box_filter.min =
            node->declare_parameter<double>("scan/preprocess/box_filter/min", params.scan.preprocess.box_filter.min);
        params.scan.preprocess.box_filter.max =
            node->declare_parameter<double>("scan/preprocess/box_filter/max", params.scan.preprocess.box_filter.max);

        params.scan.preprocess.angle_incidence_filter.enable = node->declare_parameter<bool>(
            "scan/preprocess/angle_incidence_filter/enable", params.scan.preprocess.angle_incidence_filter.enable);
        params.scan.preprocess.angle_incidence_filter.min_angle =
            node->declare_parameter<double>("scan/preprocess/angle_incidence_filter/min_angle",
                                            params.scan.preprocess.angle_incidence_filter.min_angle);
        params.scan.preprocess.angle_incidence_filter.max_angle =
            node->declare_parameter<double>("scan/preprocess/angle_incidence_filter/max_angle",
                                            params.scan.preprocess.angle_incidence_filter.max_angle);
    }

    // submapping
    {
        params.submap.map_type =
            pipeline::lidar_odometry::SubmapMapType_from_string(node->declare_parameter<std::string>(
                "submap/map_type", pipeline::lidar_odometry::SubmapMapType_to_string(params.submap.map_type)));
        params.submap.voxel_size = node->declare_parameter<double>("submap/voxel_size", params.submap.voxel_size);
        params.submap.max_distance_range =
            node->declare_parameter<double>("submap/max_distance_range", params.submap.max_distance_range);
        params.submap.point_random_sampling_num = node->declare_parameter<int64_t>(
            "submap/point_random_sampling_num", params.submap.point_random_sampling_num);
        params.submap.weighted_sampling_ratio =
            node->declare_parameter<double>("submap/weighted_sampling_ratio", params.submap.weighted_sampling_ratio);
        if (params.submap.weighted_sampling_ratio < 0.0f || params.submap.weighted_sampling_ratio > 1.0f) {
            throw std::invalid_argument(
                "[declare_lidar_odometry_params] `submap/weighted_sampling_ratio` must be "
                "within [0.0, 1.0]");
        }
        params.submap.covariance_aggregation_mode =
            algorithms::mapping::CovarianceAggregationMode_from_string(node->declare_parameter<std::string>(
                "submap/covariance_aggregation_mode",
                algorithms::mapping::CovarianceAggregationMode_to_string(params.submap.covariance_aggregation_mode)));

        params.submap.keyframe.inlier_ratio_threshold = node->declare_parameter<double>(
            "submap/keyframe/inlier_ratio_threshold", params.submap.keyframe.inlier_ratio_threshold);
        params.submap.keyframe.distance_threshold = node->declare_parameter<double>(
            "submap/keyframe/distance_threshold", params.submap.keyframe.distance_threshold);
        params.submap.keyframe.angle_threshold_degrees = node->declare_parameter<double>(
            "submap/keyframe/angle_threshold_degrees", params.submap.keyframe.angle_threshold_degrees);
        params.submap.keyframe.time_threshold_seconds = node->declare_parameter<double>(
            "submap/keyframe/time_threshold_seconds", params.submap.keyframe.time_threshold_seconds);

        params.submap.occupancy_grid_map.log_odds_hit = node->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_hit", params.submap.occupancy_grid_map.log_odds_hit);
        params.submap.occupancy_grid_map.log_odds_miss = node->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_miss", params.submap.occupancy_grid_map.log_odds_miss);
        params.submap.occupancy_grid_map.log_odds_limits_min = node->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_limits/min", params.submap.occupancy_grid_map.log_odds_limits_min);
        params.submap.occupancy_grid_map.log_odds_limits_max = node->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_limits/max", params.submap.occupancy_grid_map.log_odds_limits_max);
        params.submap.occupancy_grid_map.occupied_threshold = node->declare_parameter<double>(
            "submap/occupancy_grid_map/occupied_threshold", params.submap.occupancy_grid_map.occupied_threshold);
        params.submap.occupancy_grid_map.enable_free_space_updates =
            node->declare_parameter<bool>("submap/occupancy_grid_map/enable_free_space_update",
                                          params.submap.occupancy_grid_map.enable_free_space_updates);
        params.submap.occupancy_grid_map.enable_pruning = node->declare_parameter<bool>(
            "submap/occupancy_grid_map/enable_pruning", params.submap.occupancy_grid_map.enable_pruning);
        params.submap.occupancy_grid_map.stale_frame_threshold = node->declare_parameter<int64_t>(
            "submap/occupancy_grid_map/stale_frame_threshold", params.submap.occupancy_grid_map.stale_frame_threshold);
    }

    // Covariances
    {
        params.covariance_estimation.neighbor_num = node->declare_parameter<int64_t>(
            "covariance_estimation/neighbor_num", params.covariance_estimation.neighbor_num);

        params.covariance_estimation.m_estimation.enable = node->declare_parameter<bool>(
            "covariance_estimation/m_estimation/enable", params.covariance_estimation.m_estimation.enable);
        const std::string robust_loss =
            node->declare_parameter<std::string>("covariance_estimation/m_estimation/type", "HUBER");
        params.covariance_estimation.m_estimation.type = algorithms::robust::RobustLossType_from_string(robust_loss);
        if (params.covariance_estimation.m_estimation.type == algorithms::robust::RobustLossType::NONE) {
            params.covariance_estimation.m_estimation.enable = false;
        }
        params.covariance_estimation.m_estimation.mad_scale = node->declare_parameter<double>(
            "covariance_estimation/m_estimation/mad_scale", params.covariance_estimation.m_estimation.mad_scale);
        params.covariance_estimation.m_estimation.min_robust_scale =
            node->declare_parameter<double>("covariance_estimation/m_estimation/min_robust_scale",
                                            params.covariance_estimation.m_estimation.min_robust_scale);
        params.covariance_estimation.m_estimation.max_iterations =
            node->declare_parameter<int64_t>("covariance_estimation/m_estimation/max_iterations",
                                             params.covariance_estimation.m_estimation.max_iterations);
    }

    // motion predictor
    {
        params.motion_prediction.static_factor =
            node->declare_parameter<double>("motion_prediction/static_factor", params.motion_prediction.static_factor);
        params.motion_prediction.verbose =
            node->declare_parameter<bool>("motion_prediction/verbose", params.motion_prediction.verbose);

        params.motion_prediction.adaptive.rotation.enable = node->declare_parameter<bool>(
            "motion_prediction/adaptive/rotation/enable", params.motion_prediction.adaptive.rotation.enable);
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

        params.motion_prediction.adaptive.translation.enable = node->declare_parameter<bool>(
            "motion_prediction/adaptive/translation/enable", params.motion_prediction.adaptive.translation.enable);
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

    // Registration
    {
        auto& reg = params.registration;
        auto& pipeline = reg.pipeline;
        auto& solver = pipeline.registration;

        // common
        {
            auto& random_sampling = pipeline.random_sampling;

            reg.min_num_points = node->declare_parameter<int64_t>("registration/min_num_points", reg.min_num_points);
            random_sampling.enable =
                node->declare_parameter<bool>("registration/random_sampling/enable", random_sampling.enable);
            random_sampling.num =
                node->declare_parameter<int64_t>("registration/random_sampling/num", random_sampling.num);

            const std::string reg_type = node->declare_parameter<std::string>("registration/type", "gicp");
            solver.reg_type = algorithms::registration::RegType_from_string(reg_type);
            solver.max_iterations =
                node->declare_parameter<int64_t>("registration/max_iterations", solver.max_iterations);
            solver.criteria.translation =
                node->declare_parameter<double>("registration/criteria/translation", solver.criteria.translation);
            solver.criteria.rotation =
                node->declare_parameter<double>("registration/criteria/rotation", solver.criteria.rotation);
            solver.verbose = node->declare_parameter<bool>("registration/verbose", solver.verbose);
        }
        // Outlier removal
        {
            solver.max_correspondence_distance = node->declare_parameter<double>(
                "registration/max_correspondence_distance", solver.max_correspondence_distance);
            solver.mahalanobis_distance_threshold = node->declare_parameter<double>(
                "registration/mahalanobis_distance_threshold", solver.mahalanobis_distance_threshold);
        }

        // robust
        {
            auto& robust = solver.robust;
            auto& pipeline_robust = pipeline.robust;

            const std::string robust_loss = node->declare_parameter<std::string>("registration/robust/type", "NONE");
            robust.type = algorithms::robust::RobustLossType_from_string(robust_loss);
            robust.default_scale =
                node->declare_parameter<double>("registration/robust/default_scale", robust.default_scale);
            pipeline_robust.init_scale =
                node->declare_parameter<double>("registration/robust/init_scale", pipeline_robust.init_scale);
            pipeline_robust.auto_scale =
                node->declare_parameter<bool>("registration/robust/auto_scale", pipeline_robust.auto_scale);
            pipeline_robust.min_scale =
                node->declare_parameter<double>("registration/robust/min_scale", pipeline_robust.min_scale);
            pipeline_robust.auto_scaling_iter = node->declare_parameter<int64_t>(
                "registration/robust/auto_scaling_iter", pipeline_robust.auto_scaling_iter);
        }
        // deskew
        {
            auto& velocity_update = pipeline.velocity_update;

            velocity_update.enable =
                node->declare_parameter<bool>("registration/velocity_update/enable", velocity_update.enable);
            velocity_update.iter =
                node->declare_parameter<int64_t>("registration/velocity_update/iter", velocity_update.iter);
        }
        // photometric
        {
            auto& photometric = solver.photometric;

            photometric.enable = node->declare_parameter<bool>("registration/photometric/enable", photometric.enable);
            photometric.weight = node->declare_parameter<double>("registration/photometric/weight", photometric.weight);
            photometric.robust_scale =
                node->declare_parameter<double>("registration/photometric/robust_scale", photometric.robust_scale);
        }
        // GenZ
        {
            auto& genz = solver.genz;

            genz.planarity_threshold =
                node->declare_parameter<double>("registration/genz/planarity_threshold", genz.planarity_threshold);
        }
        // Rotation Constraint
        {
            auto& rotation_constraint = solver.rotation_constraint;
            auto& rotation_robust = rotation_constraint.robust;
            auto& pipeline_robust = pipeline.robust;

            rotation_constraint.enable =
                node->declare_parameter<bool>("registration/rotation_constraint/enable", rotation_constraint.enable);
            rotation_constraint.weight =
                node->declare_parameter<double>("registration/rotation_constraint/weight", rotation_constraint.weight);
            rotation_robust.default_scale = node->declare_parameter<double>(
                "registration/rotation_constraint/robust/default_scale", rotation_robust.default_scale);
            pipeline_robust.rotation_init_scale = node->declare_parameter<double>(
                "registration/rotation_constraint/robust/init_scale", pipeline_robust.rotation_init_scale);
            pipeline_robust.rotation_min_scale = node->declare_parameter<double>(
                "registration/rotation_constraint/robust/min_scale", pipeline_robust.rotation_min_scale);
        }

        // Optimization
        {
            auto& gn = solver.gn;
            auto& lm = solver.lm;
            auto& dogleg = solver.dogleg;

            const std::string optimization_method =
                node->declare_parameter<std::string>("registration/optimization_method", "GN");
            solver.optimization_method = algorithms::registration::OptimizationMethod_from_string(optimization_method);

            gn.lambda = node->declare_parameter<double>("registration/gn/lambda", gn.lambda);

            lm.max_inner_iterations =
                node->declare_parameter<int64_t>("registration/lm/max_inner_iterations", lm.max_inner_iterations);
            lm.init_lambda = node->declare_parameter<double>("registration/lm/init_lambda", lm.init_lambda);
            lm.max_lambda = node->declare_parameter<double>("registration/lm/max_lambda", lm.max_lambda);
            lm.min_lambda = node->declare_parameter<double>("registration/lm/min_lambda", lm.min_lambda);

            dogleg.initial_trust_region_radius = node->declare_parameter<double>(
                "registration/dogleg/initial_trust_region_radius", dogleg.initial_trust_region_radius);
            dogleg.max_trust_region_radius = node->declare_parameter<double>(
                "registration/dogleg/max_trust_region_radius", dogleg.max_trust_region_radius);
            dogleg.min_trust_region_radius = node->declare_parameter<double>(
                "registration/dogleg/min_trust_region_radius", dogleg.min_trust_region_radius);
            dogleg.eta1 = node->declare_parameter<double>("registration/dogleg/eta1", dogleg.eta1);
            dogleg.eta2 = node->declare_parameter<double>("registration/dogleg/eta2", dogleg.eta2);
            dogleg.gamma_decrease =
                node->declare_parameter<double>("registration/dogleg/gamma_decrease", dogleg.gamma_decrease);
            dogleg.gamma_increase =
                node->declare_parameter<double>("registration/dogleg/gamma_increase", dogleg.gamma_increase);
        }
        // Degenerate Regularization
        {
            auto& degenerate_reg = solver.degenerate_reg;

            const std::string degenerate_reg_type =
                node->declare_parameter<std::string>("registration/degenerate_regularization/type", "NONE");
            degenerate_reg.type =
                algorithms::registration::DegenerateRegularizationType_from_string(degenerate_reg_type);

            degenerate_reg.base_factor = node->declare_parameter<double>(
                "registration/degenerate_regularization/nl_reg/base_factor", degenerate_reg.base_factor);
            degenerate_reg.trans_eigenvalue_threshold = node->declare_parameter<double>(
                "registration/degenerate_regularization/nl_reg/trans_eigenvalue_threshold",
                degenerate_reg.trans_eigenvalue_threshold);
            degenerate_reg.rot_eigenvalue_threshold = node->declare_parameter<double>(
                "registration/degenerate_regularization/nl_reg/rot_eigenvalue_threshold",
                degenerate_reg.rot_eigenvalue_threshold);
        }
    }

    // IMU
    {
        params.imu.enable = node->declare_parameter<bool>("imu/enable", params.imu.enable);

        // Extrinsic: T_imu_to_lidar
        {
            const auto x = node->declare_parameter<double>("T_imu_to_lidar/x", 0.0);
            const auto y = node->declare_parameter<double>("T_imu_to_lidar/y", 0.0);
            const auto z = node->declare_parameter<double>("T_imu_to_lidar/z", 0.0);
            const auto qx = node->declare_parameter<double>("T_imu_to_lidar/qx", 0.0);
            const auto qy = node->declare_parameter<double>("T_imu_to_lidar/qy", 0.0);
            const auto qz = node->declare_parameter<double>("T_imu_to_lidar/qz", 0.0);
            const auto qw = node->declare_parameter<double>("T_imu_to_lidar/qw", 1.0);
            params.imu.T_imu_to_lidar.setIdentity();
            params.imu.T_imu_to_lidar.translation() << static_cast<float>(x), static_cast<float>(y),
                static_cast<float>(z);
            const Eigen::Quaternionf quat(static_cast<float>(qw), static_cast<float>(qx), static_cast<float>(qy),
                                          static_cast<float>(qz));
            params.imu.T_imu_to_lidar.matrix().block<3, 3>(0, 0) = quat.normalized().matrix();
        }

        // Gravity vector in world frame [m/s^2]
        {
            const auto gx =
                node->declare_parameter<double>("imu/preintegration/gravity/x", params.imu.preintegration.gravity.x());
            const auto gy =
                node->declare_parameter<double>("imu/preintegration/gravity/y", params.imu.preintegration.gravity.y());
            const auto gz =
                node->declare_parameter<double>("imu/preintegration/gravity/z", params.imu.preintegration.gravity.z());
            params.imu.preintegration.gravity << static_cast<float>(gx), static_cast<float>(gy), static_cast<float>(gz);
        }

        // Initial/fixed bias
        {
            const auto bgx = node->declare_parameter<double>("imu/bias/gyro/x", params.imu.bias.gyro_bias.x());
            const auto bgy = node->declare_parameter<double>("imu/bias/gyro/y", params.imu.bias.gyro_bias.y());
            const auto bgz = node->declare_parameter<double>("imu/bias/gyro/z", params.imu.bias.gyro_bias.z());
            params.imu.bias.gyro_bias << static_cast<float>(bgx), static_cast<float>(bgy), static_cast<float>(bgz);

            const auto bax = node->declare_parameter<double>("imu/bias/accel/x", params.imu.bias.accel_bias.x());
            const auto bay = node->declare_parameter<double>("imu/bias/accel/y", params.imu.bias.accel_bias.y());
            const auto baz = node->declare_parameter<double>("imu/bias/accel/z", params.imu.bias.accel_bias.z());
            params.imu.bias.accel_bias << static_cast<float>(bax), static_cast<float>(bay), static_cast<float>(baz);
        }

        params.imu.buffer_duration_sec =
            node->declare_parameter<double>("imu/buffer_duration_sec", params.imu.buffer_duration_sec);

        params.imu.deskew.enable = node->declare_parameter<bool>("imu/deskew/enable", params.imu.deskew.enable);
    }

    return params;
}
}  // namespace ros2

}  // namespace sycl_points
