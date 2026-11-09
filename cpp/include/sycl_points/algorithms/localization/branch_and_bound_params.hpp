#pragma once

#include <Eigen/Core>
#include <chrono>
#include <cstddef>
#include <numbers>

namespace sycl_points::algorithms::localization {

/// @brief Grid settings for scans with gravity alignment (z-up).
/// Roll and pitch ranges describe small IMU uncertainty.
/// @note Each level uses a voxel occupancy map. An occupied voxel also marks its negative neighbors,
///       so a parent score is an upper bound for the scores of its child nodes.
struct BranchAndBoundParams {
    float min_resolution = 1.0f;
    int max_level = 6;
    int branch_factor = 2;
    Eigen::Vector3f min_translation = Eigen::Vector3f::Zero();
    Eigen::Vector3f max_translation = Eigen::Vector3f::Zero();
    Eigen::Vector3f min_rpy = Eigen::Vector3f::Zero();
    Eigen::Vector3f max_rpy = Eigen::Vector3f(0.0f, 0.0f, 2.0f * std::numbers::pi_v<float>);
    float score_threshold_ratio = 0.0f;
    bool enable_timeout = false;
    std::chrono::milliseconds timeout{10000};
    size_t score_batch_size = 10000;
    bool auto_translation_range = true;
};

}  // namespace sycl_points::algorithms::localization
