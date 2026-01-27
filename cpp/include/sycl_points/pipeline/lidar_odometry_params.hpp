#pragma once

#include <Eigen/Geometry>

#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/ros2/covariance_marker_publisher.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_odometry {

struct Parameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::string sycl_device_vendor = "intel";
    std::string sycl_device_type = "gpu";
    bool scan_intensity_correction_enable = true;
    float scan_intensity_correction_exp = 2.0f;
    float scan_intensity_correction_scale = 1e-3f;
    float scan_intensity_correction_min_intensity = 0.0f;
    float scan_intensity_correction_max_intensity = 1.0f;
    bool scan_downsampling_voxel_enable = false;
    float scan_downsampling_voxel_size = 1.0f;
    bool scan_downsampling_polar_enable = true;
    float scan_downsampling_polar_distance_size = 1.0f;
    float scan_downsampling_polar_elevation_size = 3.0f * M_PIf / 180.0f;
    float scan_downsampling_polar_azimuth_size = 3.0f * M_PIf / 180.0f;
    std::string scan_downsampling_polar_coord_system = "CAMERA";
    bool scan_downsampling_random_enable = true;
    size_t scan_downsampling_random_num = 5000;

    bool scan_preprocess_box_filter_enable = true;
    float scan_preprocess_box_filter_min = 2.0f;
    float scan_preprocess_box_filter_max = 50.0f;
    bool scan_preprocess_angle_incidence_filter_enable = true;
    float scan_preprocess_angle_incidence_filter_min_angle = 0.0f;                    // 0.0 degrees
    float scan_preprocess_angle_incidence_filter_max_angle = 80.0f * M_PIf / 180.0f;  // 80.0 degrees

    float submap_voxel_size = 1.0f;
    float submap_max_distance_range = 30.0f;
    size_t submap_point_random_sampling_num = 2000;
    float keyframe_inlier_ratio_threshold = 0.7f;
    float keyframe_distance_threshold = 2.0f;
    float keyframe_angle_threshold_degrees = 20.0f;
    float keyframe_time_threshold_seconds = 1.0f;

    size_t covariance_estimation_neighbor_num = 10;
    bool covariance_estimation_m_estimation_enable = true;
    algorithms::robust::RobustLossType covariance_estimation_m_estimation_type =
        algorithms::robust::RobustLossType::GEMAN_MCCLURE;
    float covariance_estimation_m_estimation_mad_scale = 1.0f;
    float covariance_estimation_m_estimation_min_robust_scale = 5.0f;
    size_t covariance_estimation_m_estimation_max_iterations = 1;

    bool occupancy_grid_map_enable = true;
    float occupancy_grid_map_log_odds_hit = 0.8f;
    float occupancy_grid_map_log_odds_miss = -0.05f;
    float occupancy_grid_map_log_odds_limits_min = -1.0f;
    float occupancy_grid_map_log_odds_limits_max = 4.0f;
    float occupancy_grid_map_occupied_threshold = 0.5f;
    bool occupancy_grid_map_enable_free_space_updates = true;
    bool occupancy_grid_map_enable_pruning = true;
    size_t occupancy_grid_map_stale_frame_threshold = 100U;

    float motion_prediction_static_factor = 0.5f;
    bool motion_prediction_verbose = false;
    bool motion_prediction_adaptive_rot_enable = true;
    float motion_prediction_adaptive_rot_factor_min = 0.2f;
    float motion_prediction_adaptive_rot_factor_max = 1.0f;
    float motion_prediction_adaptive_rot_min_eigenvalue_low = 5.0f;
    float motion_prediction_adaptive_rot_min_eigenvalue_high = 10.0f;
    bool motion_prediction_adaptive_trans_enable = true;
    float motion_prediction_adaptive_trans_factor_min = 0.2f;
    float motion_prediction_adaptive_trans_factor_max = 1.0f;
    float motion_prediction_adaptive_trans_min_eigenvalue_low = 1.0f;
    float motion_prediction_adaptive_trans_min_eigenvalue_high = 10.0f;

    size_t registration_min_num_points = 100;
    bool registration_velocity_update_enable = true;
    size_t registration_velocity_update_iter = 1;
    bool registration_random_sampling_enable = true;
    size_t registration_random_sampling_num = 1000;
    algorithms::registration::RegistrationParams reg_params;

    std::string odom_frame_id = "odom";
    std::string base_link_id = "base_link";

    Eigen::Isometry3f T_base_link_to_lidar = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f T_lidar_to_base_link = Eigen::Isometry3f::Identity();

    Eigen::Isometry3f initial_pose = Eigen::Isometry3f::Identity();  // map to base_link

    // Visualizer
    ros2::CovarianceMarkerConfig scan_cov_marker_config;
};

}  // namespace lidar_odometry
}  // namespace pipeline
}  // namespace sycl_points
