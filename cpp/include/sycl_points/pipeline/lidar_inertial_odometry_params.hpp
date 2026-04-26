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
        /// Standard deviation of the finite-difference velocity estimate [m/s].
        ///
        /// Added to the velocity block of the initial covariance at every
        /// IMU preintegration reset to reflect that finite-difference velocity
        ///   v = Δp / dt
        /// has an inherent uncertainty of roughly σ_icp / dt, where σ_icp is
        /// the ICP position accuracy.  This prevents H_imu from overwhelming
        /// H_icp in the LIO optimization when accel_noise_density is small,
        /// because P_pred[p,p] ≈ dt² × fd_velocity_sigma² ≈ σ_icp².
        ///
        /// Typical value: σ_icp / dt  (e.g. 0.01m / 0.1s = 0.1 m/s)
        float fd_velocity_sigma = 0.1f;
    };

    LIO lio;
};

}  // namespace lidar_inertial_odometry
}  // namespace pipeline
}  // namespace sycl_points
