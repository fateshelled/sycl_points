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
    /// Ceiling on the IMU per-inlier information used as the comparison baseline.
    /// Without it an unusually confident IMU prior inflates the threshold until
    /// every direction is weak and the filter becomes a uniform down-scale.
    /// Must be >= the matching floor; <= 0 disables the cap.
    float trans_max_imu_information_per_inlier = 50.0f;
    /// Same as trans_max_imu_information_per_inlier for the rotation block.
    float rot_max_imu_information_per_inlier = 50.0f;
    /// Minimum information scale applied to weak translation directions. 0 allows full removal.
    float trans_weak_direction_scale = 0.2f;
    /// Minimum information scale applied to weak rotation directions. 0 allows full removal.
    float rot_weak_direction_scale = 0.2f;
    /// Analyse the coupled translation/rotation modes with a unit-balanced full
    /// 6x6 eigendecomposition instead of the independent 3x3 diagonal blocks.
    /// This catches combined degenerate motions (e.g. translation along, and
    /// rotation about, a cylinder axis) that the block-diagonal analysis misses.
    /// The default keeps the existing block-diagonal behaviour.
    bool use_coupled_degeneracy = false;
    /// Representative length [m] balancing the rotation and translation blocks
    /// of the coupled analysis. `<= 0` estimates it per frame from the Hessian
    /// trace ratio (~= weighted RMS point range); a positive value is used fixed.
    float coupled_representative_length = 0.0f;
    /// Weak-mode ratio on the balanced, inlier-normalised information used by
    /// the coupled analysis (replaces the per-block ratios for that path).
    float coupled_min_information_ratio = 0.5f;
    /// Per-inlier IMU information floor in the balanced, inlier-normalised units
    /// used by the coupled weak-direction gate.
    float coupled_imu_information_floor_per_inlier = 5.0f;
    /// Ceiling on the balanced, inlier-normalised IMU information used as the
    /// coupled comparison baseline (mirrors the block-path ceilings). Without it
    /// an over-confident IMU prior makes every coupled direction look weak.
    /// Must be >= the floor; <= 0 disables the cap.
    float coupled_max_imu_information_per_inlier = 50.0f;
    /// Information scale applied to a weak coupled direction (0 removes it).
    float coupled_weak_direction_scale = 0.2f;
};

/// @brief Constant-velocity prior on the world-frame velocity state.
///
/// The LiDAR cannot observe motion along a degenerate axis, and the IMU prior is
/// a position anchor rather than a driver: its gradient vanishes at the prediction,
/// so neither factor can carry the velocity forward through a corridor.  Without
/// that the velocity state absorbs whatever residual gradient the weak ICP leaves
/// and the next prediction inherits it, which shows up as a frame that stalls or
/// reverses while the platform keeps moving.
///
/// This prior anchors the velocity state to the last accepted velocity, which is
/// the constant-velocity assumption applied to the state the optimiser solves for
/// rather than to the prediction.  The information is applied per eigen-direction:
/// only directions whose ICP position information is weak are anchored strongly, so
/// well-observed axes still take their velocity from the measurement.
///
/// A direction must fail BOTH gates to count as degenerate.  The ratio gate is
/// self-referenced and catches the corridor case where one axis is orders of
/// magnitude below the others, but on its own it fires in any environment because
/// the weakest axis is always some fraction of the strongest.  The absolute gate
/// requires the axis to be weak in an absolute sense as well, so a well-conditioned
/// frame is left alone regardless of its eigenvalue spread.
struct ConstantVelocityPriorParams {
    bool enable = false;
    bool verbose = false;
    /// An ICP position eigen-direction is degenerate when its information is below
    /// this fraction of the largest eigenvalue.  Self-referenced so it does not
    /// depend on the IMU information, which itself collapses along degenerate axes
    /// through the P_post -> P_pred -> H_imu feedback.
    float min_eigenvalue_ratio = 0.05f;
    /// Absolute per-inlier information floor for the degeneracy gate.  A direction
    /// must be below this AND below min_eigenvalue_ratio * lambda_max.  ~0.5 rejects
    /// the corridor weak axis (observed ~0.5 per inlier) while leaving normal frames
    /// untouched.  Set <= 0 to rely on the ratio gate alone.
    float min_information_per_inlier = 0.5f;
    /// Velocity std-dev [m/s] of the anchor in degenerate directions.  The prior
    /// information is 1 / sigma^2, an absolute quantity on the velocity state: the
    /// expected velocity change over one frame is a_max * dt, so this should be the
    /// largest acceleration the platform is expected to undergo times the frame
    /// period (e.g. 1 m/s^2 * 0.1 s = 0.1 m/s).  Do NOT scale this with the inlier
    /// count - that gives a velocity information ~1e4, far above what the ICP
    /// position information can supply as velocity (lambda_p * dt^2 ~ 10), which
    /// freezes the velocity to whatever error it already carried and integrates
    /// that error into a slow positional drift.
    float degenerate_velocity_sigma = 0.1f;
    /// Velocity std-dev [m/s] of the anchor in well-observed directions.  <= 0 leaves
    /// those directions entirely to the LiDAR/IMU factors.
    float observable_velocity_sigma = 0.0f;
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
    ConstantVelocityPriorParams constant_velocity_prior;
};

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
