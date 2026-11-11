#pragma once

#include <Eigen/Geometry>
#include <cstddef>
#include <cstdint>

#include "sycl_points/algorithms/localization/branch_and_bound_profile.hpp"

namespace sycl_points::algorithms::localization {

/// @brief Best leaf pose and statistics for the localization result.
struct BranchAndBoundResult {
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    uint32_t inlier = 0;
    float score_ratio = 0.0f;
    bool converged = false;
    bool timed_out = false;
    double elapsed_ms = 0.0;
    size_t evaluated_candidates = 0;
    BranchAndBoundProfile profile;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace sycl_points::algorithms::localization
