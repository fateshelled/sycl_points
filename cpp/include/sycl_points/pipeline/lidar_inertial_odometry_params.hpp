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
        /// Total number of IEKF/Gauss-Newton iterations per frame.
        /// The IEKF loop is flat: each iteration runs one linearize+solve, and the
        /// robust scale is annealed across these iterations. So this value is the
        /// total linearize+solve count (unlike the nested LO solver loop).
        size_t total_iterations = 10;
        // Convergence threshold
        algorithms::registration::RegistrationParams::Criteria criteria;
        /// Regularization factor for velocity and bias when P_pred is singular.
        float invalid_regularization_factor = 1e4f;
        /// Velocity std-dev [m/s] for the P_initial floor at each IMU reset.
        /// Ensures P_pred[p,p] ≳ (fd_velocity_sigma × dt)² so H_imu[p,p]
        /// stays on the same scale as H_icp regardless of accel_noise_density.
        /// Rule of thumb: σ_icp_position / dt  (e.g. 0.01 m / 0.1 s = 0.1 m/s)
        float fd_velocity_sigma = 0.1f;

        /// Rotation std-dev [rad] for the P_initial floor at each IMU reset.
        /// Same mechanism as fd_velocity_sigma for H_imu[φ,φ] vs gyro_noise_density.
        /// Choose so that 1/icp_rotation_sigma² ≲ H_icp[rot,rot]
        /// (≈ 1e4–1e5 for outdoor LiDAR with ~1000 inliers → σ ≈ 0.003–0.01 rad).
        float icp_rotation_sigma = 0.01f;

        /// Blend factor in [0,1] for the post-optimization world-frame velocity.
        /// The final velocity is a convex combination of the IEKF filter velocity
        /// (x_op.velocity, smooth and IMU-consistent) and a finite-difference
        /// velocity (position delta + ½·a·dt, responsive but noisy):
        ///   v = (1 − blend)·v_filter + blend·v_finite_difference
        /// 0.0 trusts the filter; 1.0 reproduces the legacy pure-FD behavior.
        /// Because this velocity seeds the next window's IMU prediction, a lower
        /// blend reduces position-difference noise leaking into the prior.
        float velocity_fd_blend = 1.0f;
    };

    LIO lio;
};

}  // namespace lidar_inertial_odometry
}  // namespace pipeline
}  // namespace sycl_points
