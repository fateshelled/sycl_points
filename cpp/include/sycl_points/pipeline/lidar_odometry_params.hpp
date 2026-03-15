#pragma once

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>

#include "sycl_points/algorithms/mapping/covariance_aggregation_mode.hpp"
#include "sycl_points/algorithms/registration/registration_pipeline_params.hpp"
#include "sycl_points/ros2/covariance_marker_publisher.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_odometry {

enum class SubmapMapType {
    OCCUPANCY_GRID_MAP = 0,
    VOXEL_HASH_MAP,
};

inline SubmapMapType SubmapMapType_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });
    if (upper == "OCCUPANCY_GRID_MAP") {
        return SubmapMapType::OCCUPANCY_GRID_MAP;
    }
    if (upper == "VOXEL_HASH_MAP") {
        return SubmapMapType::VOXEL_HASH_MAP;
    }
    throw std::runtime_error("[SubmapMapType_from_string] Invalid submap map type '" + str + "'");
}

inline std::string SubmapMapType_to_string(const SubmapMapType type) {
    switch (type) {
        case SubmapMapType::OCCUPANCY_GRID_MAP:
            return "OCCUPANCY_GRID_MAP";
        case SubmapMapType::VOXEL_HASH_MAP:
            return "VOXEL_HASH_MAP";
    }
    throw std::runtime_error("[SubmapMapType_to_string] Invalid submap map type");
}

struct Parameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Device {
        std::string vendor = "intel";
        std::string type = "gpu";
    };

    struct Scan {
        struct IntensityCorrection {
            bool enable = true;
            float exp = 2.0f;
            float scale = 1e-3f;
            float min_intensity = 0.0f;
            float max_intensity = 1.0f;
        };

        struct Downsampling {
            struct Voxel {
                bool enable = false;
                float size = 1.0f;
            };

            struct Polar {
                bool enable = true;
                float distance_size = 1.0f;
                float elevation_size = 3.0f * M_PIf / 180.0f;
                float azimuth_size = 3.0f * M_PIf / 180.0f;
                std::string coord_system = "CAMERA";
            };

            struct Random {
                bool enable = true;
                size_t num = 5000;
            };

            Voxel voxel;
            Polar polar;
            Random random;
        };

        struct Preprocess {
            struct BoxFilter {
                bool enable = true;
                float min = 2.0f;
                float max = 50.0f;
            };

            struct AngleIncidenceFilter {
                bool enable = true;
                float min_angle = 0.0f;
                float max_angle = 80.0f * M_PIf / 180.0f;
            };

            BoxFilter box_filter;
            AngleIncidenceFilter angle_incidence_filter;
        };

        IntensityCorrection intensity_correction;
        Downsampling downsampling;
        Preprocess preprocess;
    };

    struct Submap {
        struct Keyframe {
            float inlier_ratio_threshold = 0.7f;
            float distance_threshold = 2.0f;
            float angle_threshold_degrees = 20.0f;
            float time_threshold_seconds = 1.0f;
        };

        struct OccupancyGridMap {
            float log_odds_hit = 0.8f;
            float log_odds_miss = -0.05f;
            float log_odds_limits_min = -1.0f;
            float log_odds_limits_max = 4.0f;
            float occupied_threshold = 0.5f;
            bool enable_free_space_updates = true;
            bool enable_pruning = true;
            size_t stale_frame_threshold = 100U;
        };

        SubmapMapType map_type = SubmapMapType::OCCUPANCY_GRID_MAP;
        float voxel_size = 1.0f;
        float max_distance_range = 30.0f;
        size_t point_random_sampling_num = 2000;
        algorithms::mapping::CovarianceAggregationMode covariance_aggregation_mode =
            algorithms::mapping::CovarianceAggregationMode::ARITHMETIC;
        Keyframe keyframe;
        OccupancyGridMap occupancy_grid_map;
    };

    struct CovarianceEstimation {
        struct MEstimation {
            bool enable = true;
            algorithms::robust::RobustLossType type = algorithms::robust::RobustLossType::GEMAN_MCCLURE;
            float mad_scale = 1.0f;
            float min_robust_scale = 5.0f;
            size_t max_iterations = 1;
        };

        size_t neighbor_num = 10;
        MEstimation m_estimation;
    };

    struct MotionPrediction {
        struct AdaptiveAxis {
            bool enable = true;
            float factor_min = 0.2f;
            float factor_max = 1.0f;
            float min_eigenvalue_low = 1.0f;
            float min_eigenvalue_high = 10.0f;
        };

        struct Adaptive {
            AdaptiveAxis rotation = {.enable = true,
                                     .factor_min = 0.2f,
                                     .factor_max = 1.0f,
                                     .min_eigenvalue_low = 5.0f,
                                     .min_eigenvalue_high = 10.0f};
            AdaptiveAxis translation;
        };

        float static_factor = 0.5f;
        bool verbose = false;
        Adaptive adaptive;
    };

    struct Registration {
        size_t min_num_points = 100;
        algorithms::registration::RegistrationPipelineParams pipeline;
    };

    struct Frames {
        std::string odom_frame_id = "odom";
        std::string base_link_id = "base_link";
        Eigen::Isometry3f T_base_link_to_lidar = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f T_lidar_to_base_link = Eigen::Isometry3f::Identity();
    };

    struct Pose {
        Eigen::Isometry3f initial = Eigen::Isometry3f::Identity();
    };

    struct Visualization {
        ros2::CovarianceMarkerConfig scan_covariance_markers;
    };

    Device device;
    Scan scan;
    Submap submap;
    CovarianceEstimation covariance_estimation;
    MotionPrediction motion_prediction;
    Registration registration;
    Frames frames;
    Pose pose;
    Visualization visualization;
};

}  // namespace lidar_odometry
}  // namespace pipeline
}  // namespace sycl_points
