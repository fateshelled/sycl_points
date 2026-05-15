#pragma once

#include <Eigen/Dense>
#include <deque>
#include <string>

#include "sycl_points/algorithms/imu/imu_preintegration.hpp"

namespace sycl_points {
namespace imu {

/// @brief Parameters for stationary-IMU initial roll/pitch alignment.
struct InitialAlignmentParams {
    /// Enable automatic gravity-aligned roll/pitch estimation at startup.
    bool enable = true;

    /// Required time span [s] of IMU samples in the buffer before alignment is attempted.
    /// Must be ≤ buffer_duration_sec or alignment can never trigger.
    float required_duration_sec = 1.0f;

    /// Maximum per-axis gyroscope standard deviation [rad/s] for the stationarity check.
    /// MEMS at rest is typically below 0.01 rad/s; loosen for noisier sensors.
    float max_gyro_std = 0.01f;

    /// Maximum per-axis accelerometer standard deviation [m/s²] for the stationarity check.
    /// MEMS at rest is typically below 0.1 m/s²; raise if engine vibration is unavoidable.
    float max_accel_std = 0.2f;

    /// Allowed deviation of ||a_mean|| from |gravity_world| [m/s²].
    /// Detects large unmodelled accel bias or that the robot is not actually at rest.
    float max_accel_norm_error = 0.5f;

    /// Replace the configured gyro bias with the mean gyro reading on successful alignment.
    bool estimate_gyro_bias = true;

    /// Maximum time [s] to keep waiting for a stationary window before forcing alignment.
    /// Counted from the first scan that triggered alignment.  Set ≤ 0 to wait indefinitely.
    /// Once exceeded, the stationarity checks are bypassed and alignment is computed from
    /// whatever samples are in the buffer (the gyro bias estimate may be polluted).
    float max_wait_sec = 5.0f;
};

/// @brief Result of an alignment attempt.  R_world_imu is the rotation that maps IMU body
///        vectors into the world frame so that the body-frame "up" direction aligns with
///        the negated gravity vector.  Yaw is intentionally not constrained by gravity and
///        is left as the minimum-rotation yaw (essentially zero).
struct InitialAlignmentResult {
    bool success = false;
    Eigen::Matrix3f R_world_imu = Eigen::Matrix3f::Identity();
    Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();
    Eigen::Vector3f accel_mean = Eigen::Vector3f::Zero();
    Eigen::Vector3f gyro_std = Eigen::Vector3f::Zero();
    Eigen::Vector3f accel_std = Eigen::Vector3f::Zero();
    float accel_norm = 0.0f;
    float roll_rad = 0.0f;
    float pitch_rad = 0.0f;
    std::string error_message;
};

namespace detail {

/// @brief Extract the yaw component (ZYX convention) of a rotation matrix using atan2 on
///        the first column.  Avoids gimbal-lock corner cases at pitch = ±π/2 by returning
///        0 when the horizontal projection of the first column collapses.
inline float yaw_from_rotation(const Eigen::Matrix3f& R) {
    const float cy = R(0, 0);
    const float sy = R(1, 0);
    if (std::hypot(cy, sy) < 1e-6f) return 0.0f;
    return std::atan2(sy, cy);
}

inline Eigen::Matrix3f rotation_z(float yaw) {
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);
    Eigen::Matrix3f Rz;
    Rz << c, -s, 0.0f, s, c, 0.0f, 0.0f, 0.0f, 1.0f;
    return Rz;
}

}  // namespace detail

/// @brief Estimate the gravity-aligned roll/pitch of the IMU body frame from a window of
///        stationary IMU samples.
///
/// Physics:  for a stationary device,  a_meas - b_a  ≈  R_wb^T · (-g_world)
///         where the LHS is the specific-force reading in the body frame and  -g_world
///         points "up" in the world frame.  Normalising both sides gives a single
///         pair of unit vectors (one in body, one in world) and the minimum rotation
///         between them defines the gravity-aligned R_world_imu (yaw unobservable).
///
/// @param imu_buffer    Recent IMU samples (chronological order).  Latest sample sets the
///                      window end; samples within required_duration_sec of the end are used.
/// @param gravity_world Gravity vector in the world frame [m/s²].  May be non-(0,0,-g) for
///                      tilted-world setups; the magnitude is also used as the expected
///                      ||a_meas|| reference.
/// @param params        Stationarity and duration thresholds.
/// @param current_bias  Bias estimate to subtract from the mean accel before normalising.
///                      Pass the user-configured initial bias; accel_bias cannot be
///                      separated from gravity by this routine.
inline InitialAlignmentResult estimate_initial_alignment(const std::deque<IMUMeasurement>& imu_buffer,
                                                         const Eigen::Vector3f& gravity_world,
                                                         const InitialAlignmentParams& params,
                                                         const IMUBias& current_bias,
                                                         bool bypass_stationarity = false) {
    InitialAlignmentResult res;

    const float gravity_norm = gravity_world.norm();
    if (gravity_norm < 1e-3f) {
        res.error_message = "gravity vector is (near) zero";
        return res;
    }

    if (imu_buffer.size() < 2) {
        res.error_message = "IMU buffer has fewer than 2 samples";
        return res;
    }

    // The buffer itself must span the required duration; the per-window filter below
    // will inevitably have a span ≤ required due to discrete sampling, so we cannot
    // check window_span ≥ required directly.
    const double t_end = imu_buffer.back().timestamp;
    const double buffer_span = t_end - imu_buffer.front().timestamp;
    if (buffer_span + 1e-6 < static_cast<double>(params.required_duration_sec)) {
        res.error_message = "IMU buffer spans less than required_duration_sec";
        return res;
    }

    // Collect samples in [t_end - required, t_end].  Extend by one earlier sample
    // when available so the window's effective span actually reaches `required`.
    const double t_required_start = t_end - static_cast<double>(params.required_duration_sec);
    std::vector<const IMUMeasurement*> window;
    window.reserve(imu_buffer.size());
    const IMUMeasurement* pre_sample = nullptr;
    for (const auto& m : imu_buffer) {
        if (m.timestamp >= t_required_start) {
            window.push_back(&m);
        } else {
            pre_sample = &m;
        }
    }
    if (window.empty()) {
        res.error_message = "no IMU samples in required window";
        return res;
    }
    if (pre_sample != nullptr && window.front()->timestamp > t_required_start + 1e-6) {
        window.insert(window.begin(), pre_sample);
    }

    // Mean.
    Eigen::Vector3d gyro_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_sum = Eigen::Vector3d::Zero();
    for (const auto* m : window) {
        gyro_sum += m->gyro.cast<double>();
        accel_sum += m->accel.cast<double>();
    }
    const double n = static_cast<double>(window.size());
    const Eigen::Vector3d gyro_mean = gyro_sum / n;
    const Eigen::Vector3d accel_mean = accel_sum / n;

    // Variance.
    Eigen::Vector3d gyro_var = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_var = Eigen::Vector3d::Zero();
    for (const auto* m : window) {
        const Eigen::Vector3d dg = m->gyro.cast<double>() - gyro_mean;
        const Eigen::Vector3d da = m->accel.cast<double>() - accel_mean;
        gyro_var += dg.cwiseProduct(dg);
        accel_var += da.cwiseProduct(da);
    }
    gyro_var /= n;
    accel_var /= n;
    const Eigen::Vector3f gyro_std = gyro_var.cwiseSqrt().cast<float>();
    const Eigen::Vector3f accel_std = accel_var.cwiseSqrt().cast<float>();

    res.gyro_std = gyro_std;
    res.accel_std = accel_std;
    res.accel_mean = accel_mean.cast<float>();
    res.accel_norm = static_cast<float>(accel_mean.norm());

    // Stationarity checks.  Bypassed on timeout so alignment can still proceed.
    if (!bypass_stationarity) {
        if ((gyro_std.array() > params.max_gyro_std).any()) {
            res.error_message = "gyro_std exceeds threshold (robot not stationary?)";
            return res;
        }
        if ((accel_std.array() > params.max_accel_std).any()) {
            res.error_message = "accel_std exceeds threshold (robot not stationary?)";
            return res;
        }
        const float accel_norm_error = std::abs(res.accel_norm - gravity_norm);
        if (accel_norm_error > params.max_accel_norm_error) {
            res.error_message = "|a_mean| - |gravity| exceeds threshold (unmodelled accel bias?)";
            return res;
        }
    }

    // Bias-corrected specific force points opposite to world gravity.
    const Eigen::Vector3f a_unbiased = res.accel_mean - current_bias.accel_bias;
    const float a_norm = a_unbiased.norm();
    if (a_norm < 1e-3f) {
        res.error_message = "bias-corrected accel magnitude is (near) zero";
        return res;
    }

    // body_up (in IMU body frame) := bias-corrected specific force, normalised.
    // world_up (in world frame)    := -gravity_world, normalised.
    // R_world_imu * body_up  =  world_up   ⇒  use minimum rotation (FromTwoVectors).
    const Eigen::Vector3f body_up = a_unbiased / a_norm;
    const Eigen::Vector3f world_up = (-gravity_world).normalized();
    const Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(body_up, world_up);
    res.R_world_imu = q.normalized().toRotationMatrix();

    // Diagnostics: roll/pitch (ZYX) of R_world_imu. Yaw is ≈ 0 by construction.
    res.roll_rad = std::atan2(res.R_world_imu(2, 1), res.R_world_imu(2, 2));
    res.pitch_rad = std::asin(-std::clamp(res.R_world_imu(2, 0), -1.0f, 1.0f));

    res.gyro_bias = params.estimate_gyro_bias ? gyro_mean.cast<float>() : current_bias.gyro_bias;
    res.success = true;
    return res;
}

}  // namespace imu
}  // namespace sycl_points
