#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "sycl_points/algorithms/registration/registration_params.hpp"

namespace sycl_points {
namespace algorithms {
namespace lio {

struct LIORobustScheduleParams {
    bool auto_scale = false;
    float init_scale = 10.0f;
    float min_scale = 0.5f;
    float rotation_init_scale = 10.0f;
    float rotation_min_scale = 0.5f;
    size_t auto_scaling_iter = 4;
};

struct BiasUpdateMask {
    bool accel = true;
    bool gyro = true;
};

enum class DirectionalIcpWeightingType {
    scale,
    tsvd,
};

inline DirectionalIcpWeightingType DirectionalIcpWeightingType_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (upper == "SCALE" || upper == "SCALED") return DirectionalIcpWeightingType::scale;
    if (upper == "TSVD") return DirectionalIcpWeightingType::tsvd;
    throw std::runtime_error("Invalid LIO directional ICP weighting type [" + str + "]");
}

/// @brief Direction-wise ICP information shaping for degenerate LIO frames.
///
/// The reduced-chi² scalar weight handles globally bad alignments, but geometric
/// degeneracy is directional: an ICP frame can be very confident in wall-normal
/// motion while providing almost no information along a corridor. This filter
/// detects weak directions separately in the translation and rotation 3x3
/// blocks, then applies the resulting scales consistently to the full 6-DOF
/// pose factor before the IMU prior is added.
///
/// The filter only changes the balance between the ICP and IMU factors in the
/// joint solve (it scales H and b by the same factor along an eigen-direction,
/// leaving the ICP-only solution unchanged).  The comparison must therefore be
/// made against the IMU prior's information in that same direction, not against
/// an absolute ICP magnitude: a per-inlier ICP eigenvalue of 10 means nothing
/// until it is compared with what the IMU already provides.  Using absolute
/// thresholds flagged well-observed axes whose ICP information merely happened
/// to sit below the fixed number.
struct DirectionalIcpWeightingParams {
    bool enable = true;
    bool verbose = false;
    /// SCALE continuously attenuates weak information; TSVD removes it.
    DirectionalIcpWeightingType type = DirectionalIcpWeightingType::scale;
    /// An ICP eigen-direction is weak when its information is below this multiple
    /// of the IMU prior's information projected onto the same direction.
    float trans_min_information_ratio = 0.5f;
    /// Same as trans_min_information_ratio for the rotation block.
    float rot_min_information_ratio = 0.5f;
    /// Floor on the IMU per-inlier information used as the comparison baseline.
    /// The previous frame's degeneracy feeds back through P_post -> P_pred ->
    /// H_imu, so the measured IMU information collapses along exactly the axes
    /// that are already degenerate.  Without a floor that feedback makes every
    /// direction look degenerate.  5.0 per inlier corresponds to sigma ~= 0.01
    /// (rad / m) at ~2000 inliers, matching the icp_rotation_sigma and
    /// fd_velocity_sigma P_initial floors.
    float trans_imu_information_floor_per_inlier = 5.0f;
    /// Same as trans_imu_information_floor_per_inlier for the rotation block.
    float rot_imu_information_floor_per_inlier = 5.0f;
    /// Minimum information scale applied to weak translation directions. 0 allows full removal.
    float trans_weak_direction_scale = 0.2f;
    /// Minimum information scale applied to weak rotation directions. 0 allows full removal.
    float rot_weak_direction_scale = 0.2f;
};

/// @brief Parameters for the tightly-coupled ICP/IMU optimization loop.
struct LIORegistrationParams {
    /// Maximum number of solver iterations summed across all robust levels.
    size_t total_iterations = 10;
    registration::RegistrationConvergenceCriteria criteria;
    registration::RegistrationOptimizationParams optimization;
    LIORobustScheduleParams robust;
    float invalid_regularization_factor = 1e4f;
    /// Explicit information multiplier for the robustified ICP factor.
    float icp_information_scale = 1.0f;
    DirectionalIcpWeightingParams directional_icp_weighting;
};

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
