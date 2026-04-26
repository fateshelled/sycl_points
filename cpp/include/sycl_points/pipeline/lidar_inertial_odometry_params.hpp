#pragma once

#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/pipeline/lidar_odometry_params.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_inertial_odometry {

/// @brief Parameters for the LiDAR-Inertial Odometry pipeline.
///
/// Extends lidar_odometry::Parameters with an LIO-specific optimization block.
/// IMU is always required; params_.imu.enable is forced true in the pipeline.
struct Parameters : public lidar_odometry::Parameters {
    struct LIO {
        /// Maximum number of Gauss-Newton iterations per frame.
        size_t max_iterations = 10;
        // Convergence threshold
        algorithms::registration::RegistrationParams::Criteria criteria;
        /// Regularization factor for velocity and bias when P_pred is singular.
        float invalid_regularization_factor = 1e4f;
        /// Scale factor α for the H_icp-based P_initial floor (dimensionless).
        ///
        /// At each IMU reset, P_initial is augmented by:
        ///   P_floor[φ,φ] = α × H_icp[rot,rot]⁻¹
        ///   P_floor[v,v] = α × (R × H_icp[t,t] × Rᵀ)⁻¹ / dt²
        ///
        /// This ensures H_imu ≲ (1/α) × H_icp in each direction, automatically
        /// adapting to scene geometry and point density without manual tuning.
        /// In degenerate directions (H_icp small), the inverse is large so H_imu
        /// is weakened — allowing the IMU to compensate where ICP cannot.
        ///
        ///   α = 1.0  → H_imu ≈ H_icp  (balanced)
        ///   α > 1.0  → ICP dominates
        ///   α < 1.0  → IMU can dominate
        float icp_floor_scale = 1.0f;

        /// Fallback velocity std-dev [m/s] used on the first frame (before H_icp
        /// is available) and when the H_icp translation block is ill-conditioned.
        float fd_velocity_sigma = 0.1f;

        /// Fallback rotation std-dev [rad] used on the first frame (before H_icp
        /// is available) and when the H_icp rotation block is ill-conditioned.
        float icp_rotation_sigma = 0.01f;
    };

    LIO lio;
};

}  // namespace lidar_inertial_odometry
}  // namespace pipeline
}  // namespace sycl_points
