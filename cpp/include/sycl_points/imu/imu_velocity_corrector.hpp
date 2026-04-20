#pragma once

#include <Eigen/Dense>
#include <memory>

#include "sycl_points/imu/imu_preintegration.hpp"

namespace sycl_points {
namespace imu {

/// @brief Maintains an ICP-corrected instantaneous velocity estimate for IMU window resets.
///
/// Each LiDAR frame the pipeline calls two methods in order:
///
///   1. get_reset_velocity()  — called before reset(), inside the IMU mutex.
///      Returns the best available world-frame velocity for the window start.
///      Saves a snapshot of the current preintegration state (Delta_p, Delta_v, dt).
///
///   2. update()              — called after ICP, outside the mutex.
///      Uses the ICP displacement to back-solve the window-start velocity and
///      propagate it to the window end via IMU dynamics:
///
///        v_reset_corrected = (disp_icp - 0.5·g·dt² - R·Δp) / dt
///        v_k               = v_reset_corrected + g·dt + R·Δv
///
///      The result is stored and returned on the next get_reset_velocity() call.
///
/// On the first frame (no prior correction) get_reset_velocity() falls back to the
/// supplied @p fallback_v_world (typically the LiDAR-period average velocity).
class IMUVelocityCorrector {
public:
    using Ptr = std::shared_ptr<IMUVelocityCorrector>;

    /// @brief Returns the velocity to use for the next IMU window reset and saves a
    ///        snapshot of the preintegration state for the subsequent update() call.
    ///
    /// @param preintegration   Current integrator (read-only; snapshotted here).
    /// @param bias             Current bias estimate used for bias correction.
    /// @param fallback_v_world Fallback world-frame velocity (e.g. LiDAR average).
    ///                         Used when no ICP-corrected velocity is available yet.
    /// @return World-frame velocity to pass to IMUPreintegration::reset().
    Eigen::Vector3f get_reset_velocity(const IMUPreintegration& preintegration, const IMUBias& bias,
                                       const Eigen::Vector3f& fallback_v_world) {
        const Eigen::Vector3f v_reset = corrected_v_valid_ ? corrected_v_world_ : fallback_v_world;
        corrected_v_valid_ = false;

        const PreintegrationResult snap = preintegration.get_corrected(bias);
        snapshot_delta_v_ = snap.Delta_v;
        snapshot_delta_p_ = snap.Delta_p;
        snapshot_dt_ = static_cast<float>(snap.dt_total);
        snapshot_valid_ = true;

        return v_reset;
    }

    /// @brief Computes and stores the ICP-corrected end-of-window velocity.
    ///        Must be called after ICP with the corrected poses.
    ///
    /// @param disp_icp    World-frame LiDAR displacement: T_k.translation() - T_{k-1}.translation().
    /// @param R_world_imu World-frame rotation of the IMU body at window start (T_{k-1}).
    /// @param gravity     Gravity vector in the world frame [m/s²].
    void update(const Eigen::Vector3f& disp_icp, const Eigen::Matrix3f& R_world_imu, const Eigen::Vector3f& gravity) {
        if (!snapshot_valid_ || snapshot_dt_ <= 0.0f) return;

        const float dt = snapshot_dt_;
        const Eigen::Vector3f v_reset_corrected =
            (disp_icp - 0.5f * gravity * dt * dt - R_world_imu * snapshot_delta_p_) / dt;
        corrected_v_world_ = v_reset_corrected + gravity * dt + R_world_imu * snapshot_delta_v_;
        corrected_v_valid_ = true;
        snapshot_valid_ = false;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Eigen::Vector3f snapshot_delta_v_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f snapshot_delta_p_ = Eigen::Vector3f::Zero();
    float snapshot_dt_ = 0.0f;
    bool snapshot_valid_ = false;

    Eigen::Vector3f corrected_v_world_ = Eigen::Vector3f::Zero();
    bool corrected_v_valid_ = false;
};

}  // namespace imu
}  // namespace sycl_points
