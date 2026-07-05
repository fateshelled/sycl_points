#pragma once

#include "sycl_points/algorithms/lio/lio_registration_params.hpp"
#include "sycl_points/pipeline/odometry_common_params.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_inertial_odometry {

/// @brief Parameters for the LiDAR-Inertial Odometry pipeline.
///
/// Extends the parameters shared by the odometry pipelines with an
/// LIO-specific optimization block.
/// IMU is always required; params_.imu.enable is forced true in the pipeline.
struct Parameters : public odometry::CommonParameters {
    struct LIO {
        algorithms::lio::LIORegistrationParams registration;

        struct PreintegrationReset {
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
        };

        /// Bias-estimation safeguards for the weakly-observable IMU bias states.
        struct BiasEstimation {
            /// Skip accel/gyro bias updates when the IMU excitation within the
            /// window is below the thresholds below.  Biases are weakly observable
            /// without motion, so updating them while near-stationary mostly
            /// absorbs measurement noise and drives slow drift.  Default off to
            /// preserve behavior; enable for long stationary periods.
            bool freeze_on_low_excitation = false;
            /// Window gyro variation (max |ω − mean ω|) above which the gyro is
            /// considered excited [rad/s].
            float gyro_excitation_threshold = 0.03f;
            /// Window specific-force variation (max ||a| − mean |a||) above which
            /// the accelerometer is considered excited [m/s²] (raw sensor units).
            float accel_excitation_threshold = 0.3f;
            /// Hard clamp on the estimated bias L2 norm; ≤0 disables.  Bounds
            /// runaway from unobservable directions or numerical drift.
            float max_accel_bias = 0.0f;  ///< [m/s²]
            float max_gyro_bias = 0.0f;   ///< [rad/s]
        };
        PreintegrationReset preintegration_reset;
        BiasEstimation bias_estimation;
    };

    LIO lio;
};

}  // namespace lidar_inertial_odometry
}  // namespace pipeline
}  // namespace sycl_points
