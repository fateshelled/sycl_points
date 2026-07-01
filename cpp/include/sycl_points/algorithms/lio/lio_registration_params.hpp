#pragma once

#include <cstddef>

#include "sycl_points/algorithms/registration/registration_params.hpp"

namespace sycl_points {
namespace algorithms {
namespace lio {

struct LIORobustScheduleParams {
    bool auto_scale = false;
    float init_scale = 10.0f;
    float min_scale = 0.5f;
};

/// @brief Direction-wise ICP information shaping for degenerate LIO frames.
///
/// The reduced-chi² scalar weight handles globally bad alignments, but geometric
/// degeneracy is directional: an ICP frame can be very confident in wall-normal
/// motion while providing almost no information along a corridor. This filter
/// detects weak directions separately in the translation and rotation 3x3
/// blocks, then applies the resulting scales consistently to the full 6-DOF
/// pose factor before the IMU prior is added.
struct DirectionalIcpWeightingParams {
    bool enable = true;
    /// Treat ICP translation eigen-directions below this per-inlier information as weak.
    float trans_min_eigenvalue_per_inlier = 3.0f;
    /// Treat ICP rotation eigen-directions below this per-inlier information as weak.
    float rot_min_eigenvalue_per_inlier = 3.0f;
    /// Multiplicative scale applied to weak directions. 0 removes them entirely.
    float weak_direction_scale = 0.05f;
};

/// @brief Parameters for the tightly-coupled ICP/IMU optimization loop.
struct LIORegistrationParams {
    size_t total_iterations = 10;
    registration::RegistrationConvergenceCriteria criteria;
    registration::RegistrationOptimizationParams optimization;
    LIORobustScheduleParams robust;
    float invalid_regularization_factor = 1e4f;
    DirectionalIcpWeightingParams directional_icp_weighting;
};

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
