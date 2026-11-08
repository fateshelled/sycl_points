#pragma once

#include <Eigen/Geometry>
#include <cstddef>
#include <cstdint>

namespace sycl_points::algorithms::localization {

/// @brief Best leaf pose and localization statistics.
struct BranchAndBoundResult {
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    uint32_t inlier = 0;
    float score_ratio = 0.0f;
    bool converged = false;
    bool timed_out = false;
    double elapsed_ms = 0.0;
    size_t evaluated_candidates = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace sycl_points::algorithms::localization
