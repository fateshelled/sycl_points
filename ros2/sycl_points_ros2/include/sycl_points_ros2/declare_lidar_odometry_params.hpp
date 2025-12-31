#pragma once

#include <rclcpp/node.hpp>
#include <sycl_points/pipeline/lidar_odometry_params.hpp>

namespace sycl_points {
namespace ros2 {

pipeline::lidar_odometry::Parameters declare_lidar_odometry_parameters(rclcpp::Node* node) {
    pipeline::lidar_odometry::Parameters params;

    // SYCL
    {
        params.sycl_device_vendor =
            node->declare_parameter<std::string>("sycl/device_vendor", params.sycl_device_vendor);
        params.sycl_device_type = node->declare_parameter<std::string>("sycl/device_type", params.sycl_device_type);
    }

    // scan
    {
        params.scan_intensity_correction_enable =
            node->declare_parameter<bool>("scan/intensity_correction/enable", params.scan_intensity_correction_enable);
        params.scan_intensity_correction_exp =
            node->declare_parameter<double>("scan/intensity_correction/exp", params.scan_intensity_correction_exp);
        params.scan_intensity_correction_scale =
            node->declare_parameter<double>("scan/intensity_correction/scale", params.scan_intensity_correction_scale);
        params.scan_intensity_correction_min_intensity = node->declare_parameter<double>(
            "scan/intensity_correction/min_intensity", params.scan_intensity_correction_min_intensity);
        params.scan_intensity_correction_max_intensity = node->declare_parameter<double>(
            "scan/intensity_correction/max_intensity", params.scan_intensity_correction_max_intensity);
        params.scan_downsampling_voxel_enable =
            node->declare_parameter<bool>("scan/downsampling/voxel/enable", params.scan_downsampling_voxel_enable);
        params.scan_downsampling_voxel_size =
            node->declare_parameter<double>("scan/downsampling/voxel/voxel_size", params.scan_downsampling_voxel_size);

        params.scan_downsampling_polar_enable =
            node->declare_parameter<bool>("scan/downsampling/polar/enable", params.scan_downsampling_polar_enable);
        params.scan_downsampling_polar_distance_size = node->declare_parameter<double>(
            "scan/downsampling/polar/distance_size", params.scan_downsampling_polar_distance_size);
        params.scan_downsampling_polar_elevation_size = node->declare_parameter<double>(
            "scan/downsampling/polar/elevation_size", params.scan_downsampling_polar_elevation_size);
        params.scan_downsampling_polar_azimuth_size = node->declare_parameter<double>(
            "scan/downsampling/polar/azimuth_size", params.scan_downsampling_polar_azimuth_size);
        params.scan_downsampling_polar_coord_system = node->declare_parameter<std::string>(
            "scan/downsampling/polar/coord_system", params.scan_downsampling_polar_coord_system);
        params.scan_downsampling_random_enable =
            node->declare_parameter<bool>("scan/downsampling/random/enable", params.scan_downsampling_random_enable);
        params.scan_downsampling_random_num =
            node->declare_parameter<int64_t>("scan/downsampling/random/num", params.scan_downsampling_random_num);

        params.scan_covariance_neighbor_num =
            node->declare_parameter<int64_t>("scan/covariance/neighbor_num", params.scan_covariance_neighbor_num);
        params.scan_preprocess_box_filter_enable = node->declare_parameter<bool>(
            "scan/preprocess/box_filter/enable", params.scan_preprocess_box_filter_enable);
        params.scan_preprocess_box_filter_min =
            node->declare_parameter<double>("scan/preprocess/box_filter/min", params.scan_preprocess_box_filter_min);
        params.scan_preprocess_box_filter_max =
            node->declare_parameter<double>("scan/preprocess/box_filter/max", params.scan_preprocess_box_filter_max);
    }

    // submapping
    {
        params.submap_voxel_size = node->declare_parameter<double>("submap/voxel_size", params.submap_voxel_size);
        params.submap_covariance_neighbor_num =
            node->declare_parameter<int64_t>("submap/covariance/neighbor_num", params.submap_covariance_neighbor_num);
        params.submap_covariance_update_to_plane = node->declare_parameter<bool>(
            "submap/covariance/update_to_plane", params.submap_covariance_update_to_plane);
        params.submap_color_gradient_neighbor_num = node->declare_parameter<int64_t>(
            "submap/color_gradient/neighbor_num", params.submap_color_gradient_neighbor_num);
        params.submap_max_distance_range =
            node->declare_parameter<double>("submap/max_distance_range", params.submap_max_distance_range);
        params.submap_point_random_sampling_num = node->declare_parameter<int64_t>(
            "submap/point_random_sampling_num", params.submap_point_random_sampling_num);

        params.keyframe_inlier_ratio_threshold = node->declare_parameter<double>(
            "submap/keyframe/inlier_ratio_threshold", params.keyframe_inlier_ratio_threshold);
        params.keyframe_distance_threshold =
            node->declare_parameter<double>("submap/keyframe/distance_threshold", params.keyframe_distance_threshold);
        params.keyframe_angle_threshold_degrees = node->declare_parameter<double>(
            "submap/keyframe/angle_threshold_degrees", params.keyframe_angle_threshold_degrees);
        params.keyframe_time_threshold_seconds = node->declare_parameter<double>(
            "submap/keyframe/time_threshold_seconds", params.keyframe_time_threshold_seconds);

        params.occupancy_grid_map_enable =
            node->declare_parameter<bool>("submap/occupancy_grid_map/enable", params.occupancy_grid_map_enable);
        params.occupancy_grid_map_log_odds_hit = node->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_hit", params.occupancy_grid_map_log_odds_hit);
        params.occupancy_grid_map_log_odds_miss = node->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_miss", params.occupancy_grid_map_log_odds_miss);
        params.occupancy_grid_map_log_odds_limits_min = node->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_limits/min", params.occupancy_grid_map_log_odds_limits_min);
        params.occupancy_grid_map_log_odds_limits_max = node->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_limits/max", params.occupancy_grid_map_log_odds_limits_max);
        params.occupancy_grid_map_occupied_threshold = node->declare_parameter<double>(
            "submap/occupancy_grid_map/occupied_threshold", params.occupancy_grid_map_occupied_threshold);
        params.occupancy_grid_map_enable_pruning = node->declare_parameter<bool>(
            "submap/occupancy_grid_map/enable_pruning", params.occupancy_grid_map_enable_pruning);
        params.occupancy_grid_map_stale_frame_threshold = node->declare_parameter<int64_t>(
            "submap/occupancy_grid_map/stale_frame_threshold", params.occupancy_grid_map_stale_frame_threshold);
    }

    // motion predictor
    {
        params.motion_prediction_static_factor =
            node->declare_parameter<double>("motion_prediction/static_factor", params.motion_prediction_static_factor);
        params.motion_prediction_verbose =
            node->declare_parameter<bool>("motion_prediction/verbose", params.motion_prediction_verbose);

        params.motion_prediction_adaptive_rot_enable = node->declare_parameter<bool>(
            "motion_prediction/adaptive/rotation/enable", params.motion_prediction_adaptive_rot_enable);
        params.motion_prediction_adaptive_rot_factor_min = node->declare_parameter<double>(
            "motion_prediction/adaptive/rotation/factor/min", params.motion_prediction_adaptive_rot_factor_min);
        params.motion_prediction_adaptive_rot_factor_max = node->declare_parameter<double>(
            "motion_prediction/adaptive/rotation/factor/max", params.motion_prediction_adaptive_rot_factor_max);
        params.motion_prediction_adaptive_rot_eigen_low = node->declare_parameter<double>(
            "motion_prediction/adaptive/rotation/eigen/low", params.motion_prediction_adaptive_rot_eigen_low);
        params.motion_prediction_adaptive_rot_eigen_high = node->declare_parameter<double>(
            "motion_prediction/adaptive/rotation/eigen/high", params.motion_prediction_adaptive_rot_eigen_high);

        params.motion_prediction_adaptive_trans_enable = node->declare_parameter<bool>(
            "motion_prediction/adaptive/translation/enable", params.motion_prediction_adaptive_trans_enable);
        params.motion_prediction_adaptive_trans_factor_min = node->declare_parameter<double>(
            "motion_prediction/adaptive/translation/factor/min", params.motion_prediction_adaptive_trans_factor_min);
        params.motion_prediction_adaptive_trans_factor_max = node->declare_parameter<double>(
            "motion_prediction/adaptive/translation/factor/max", params.motion_prediction_adaptive_trans_factor_max);
        params.motion_prediction_adaptive_trans_eigen_low = node->declare_parameter<double>(
            "motion_prediction/adaptive/translation/eigen/low", params.motion_prediction_adaptive_trans_eigen_low);
        params.motion_prediction_adaptive_trans_eigen_high = node->declare_parameter<double>(
            "motion_prediction/adaptive/translation/eigen/high", params.motion_prediction_adaptive_trans_eigen_high);
    }

    // Registration
    {
        // common
        {
            params.registration_min_num_points =
                node->declare_parameter<int64_t>("registration/min_num_points", params.registration_min_num_points);
            params.registration_random_sampling_enable = node->declare_parameter<bool>(
                "registration/random_sampling/enable", params.registration_random_sampling_enable);
            params.registration_random_sampling_num = node->declare_parameter<int64_t>(
                "registration/random_sampling/num", params.registration_random_sampling_num);

            const std::string reg_type = node->declare_parameter<std::string>("registration/type", "gicp");
            params.reg_params.reg_type = algorithms::registration::RegType_from_string(reg_type);
            params.reg_params.max_iterations =
                node->declare_parameter<int64_t>("registration/max_iterations", params.reg_params.max_iterations);
            params.reg_params.lambda = node->declare_parameter<double>("registration/lambda", params.reg_params.lambda);
            params.reg_params.criteria.translation = node->declare_parameter<double>(
                "registration/criteria/translation", params.reg_params.criteria.translation);
            params.reg_params.criteria.rotation =
                node->declare_parameter<double>("registration/criteria/rotation", params.reg_params.criteria.rotation);

            params.reg_params.verbose =
                node->declare_parameter<bool>("registration/verbose", params.reg_params.verbose);
        }
        // Outlier removal
        {
            params.reg_params.max_correspondence_distance = node->declare_parameter<double>(
                "registration/max_correspondence_distance", params.reg_params.max_correspondence_distance);
            params.reg_params.max_correspondence_distance = node->declare_parameter<double>(
                "registration/mahalanobis_distance_threshold", params.reg_params.mahalanobis_distance_threshold);
        }

        // robust
        {
            const std::string robust_loss = node->declare_parameter<std::string>("registration/robust/type", "NONE");
            params.reg_params.robust.type = algorithms::registration::RobustLossType_from_string(robust_loss);
            params.reg_params.robust.auto_scale =
                node->declare_parameter<bool>("registration/robust/auto_scale", params.reg_params.robust.auto_scale);
            params.reg_params.robust.init_scale =
                node->declare_parameter<double>("registration/robust/init_scale", params.reg_params.robust.init_scale);
            params.reg_params.robust.min_scale =
                node->declare_parameter<double>("registration/robust/min_scale", params.reg_params.robust.min_scale);
            params.reg_params.robust.scaling_iter = node->declare_parameter<int64_t>(
                "registration/robust/scaling_iter", params.reg_params.robust.scaling_iter);
        }
        // deskew
        {
            params.registration_velocity_update_enable = node->declare_parameter<bool>(
                "registration/velocity_update/enable", params.registration_velocity_update_enable);
            params.registration_velocity_update_iter = node->declare_parameter<int64_t>(
                "registration/velocity_update/iter", params.registration_velocity_update_iter);
        }
        // photometric
        {
            params.reg_params.photometric.enable =
                node->declare_parameter<bool>("registration/photometric/enable", params.reg_params.photometric.enable);
            params.reg_params.photometric.photometric_weight = node->declare_parameter<double>(
                "registration/photometric/weight", params.reg_params.photometric.photometric_weight);
        }
        // GenZ
        {
            params.reg_params.genz.planarity_threshold = node->declare_parameter<double>(
                "registration/genz/planarity_threshold", params.reg_params.genz.planarity_threshold);
        }

        // optimization
        {
            const std::string optimization_method =
                node->declare_parameter<std::string>("registration/optimization_method", "GN");
            params.reg_params.optimization_method =
                algorithms::registration::OptimizationMethod_from_string(optimization_method);

            params.reg_params.lm.max_inner_iterations = node->declare_parameter<int64_t>(
                "registration/lm/max_inner_iterations", params.reg_params.lm.max_inner_iterations);
            params.reg_params.lm.lambda_factor =
                node->declare_parameter<double>("registration/lm/lambda_factor", params.reg_params.lm.lambda_factor);
            params.reg_params.lm.max_lambda =
                node->declare_parameter<double>("registration/lm/max_lambda", params.reg_params.lm.max_lambda);
            params.reg_params.lm.min_lambda =
                node->declare_parameter<double>("registration/lm/min_lambda", params.reg_params.lm.min_lambda);

            params.reg_params.dogleg.initial_trust_region_radius =
                node->declare_parameter<double>("registration/dogleg/initial_trust_region_radius",
                                                params.reg_params.dogleg.initial_trust_region_radius);
            params.reg_params.dogleg.max_trust_region_radius = node->declare_parameter<double>(
                "registration/dogleg/max_trust_region_radius", params.reg_params.dogleg.max_trust_region_radius);
            params.reg_params.dogleg.min_trust_region_radius = node->declare_parameter<double>(
                "registration/dogleg/min_trust_region_radius", params.reg_params.dogleg.min_trust_region_radius);
            params.reg_params.dogleg.eta1 =
                node->declare_parameter<double>("registration/dogleg/eta1", params.reg_params.dogleg.eta1);
            params.reg_params.dogleg.eta2 =
                node->declare_parameter<double>("registration/dogleg/eta2", params.reg_params.dogleg.eta2);
            params.reg_params.dogleg.gamma_decrease = node->declare_parameter<double>(
                "registration/dogleg/gamma_decrease", params.reg_params.dogleg.gamma_decrease);
            params.reg_params.dogleg.gamma_increase = node->declare_parameter<double>(
                "registration/dogleg/gamma_increase", params.reg_params.dogleg.gamma_increase);
        }
        // Degenerate Regularization
        {
            const std::string degenerate_reg_type =
                node->declare_parameter<std::string>("registration/degenerate_regularization/type", "NONE");
            params.reg_params.degenerate_reg.type =
                algorithms::registration::DegenerateRegularizationType_from_string(degenerate_reg_type);

            params.reg_params.degenerate_reg.base_factor =
                node->declare_parameter<double>("registration/degenerate_regularization/nl_reg/base_factor",
                                                params.reg_params.degenerate_reg.base_factor);
            params.reg_params.degenerate_reg.trans_eig_threshold =
                node->declare_parameter<double>("registration/degenerate_regularization/nl_reg/trans_eig_threshold",
                                                params.reg_params.degenerate_reg.trans_eig_threshold);
            params.reg_params.degenerate_reg.rot_eig_threshold =
                node->declare_parameter<double>("registration/degenerate_regularization/nl_reg/rot_eig_threshold",
                                                params.reg_params.degenerate_reg.rot_eig_threshold);
        }
    }

    // tf and pose
    {
        params.odom_frame_id = node->declare_parameter<std::string>("odom_frame_id", "odom");
        params.base_link_id = node->declare_parameter<std::string>("base_link_id", "base_link");
        {
            // x, y, z, qx, qy, qz, qw
            const auto x = node->declare_parameter<double>("T_base_link_to_lidar/x", 0.0);
            const auto y = node->declare_parameter<double>("T_base_link_to_lidar/y", 0.0);
            const auto z = node->declare_parameter<double>("T_base_link_to_lidar/z", 0.0);
            const auto qx = node->declare_parameter<double>("T_base_link_to_lidar/qx", 0.0);
            const auto qy = node->declare_parameter<double>("T_base_link_to_lidar/qy", 0.0);
            const auto qz = node->declare_parameter<double>("T_base_link_to_lidar/qz", 0.0);
            const auto qw = node->declare_parameter<double>("T_base_link_to_lidar/qw", 1.0);
            params.T_base_link_to_lidar.setIdentity();
            params.T_base_link_to_lidar.translation() << x, y, z;
            const Eigen::Quaternionf quat(qw, qx, qy, qz);
            params.T_base_link_to_lidar.matrix().block<3, 3>(0, 0) = quat.matrix();

            params.T_lidar_to_base_link = params.T_base_link_to_lidar.inverse();
        }

        {
            // x, y, z, qx, qy, qz, qw
            const auto T =
                node->declare_parameter<std::vector<double>>("initial_pose", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
            if (T.size() != 7) throw std::runtime_error("invalid initial_pose");
            params.initial_pose.setIdentity();
            params.initial_pose.translation() << T[0], T[1], T[2];
            const Eigen::Quaternionf quat(T[6], T[3], T[4], T[5]);
            params.initial_pose.matrix().block<3, 3>(0, 0) = quat.matrix();
        }
    }
    return params;
}
}  // namespace ros2

}  // namespace sycl_points
