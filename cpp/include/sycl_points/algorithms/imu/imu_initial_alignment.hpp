#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
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

/// @brief Stationary-IMU initial roll/pitch alignment with user-specified yaw preserved.
///
/// Wraps estimate_initial_alignment and owns:
///   - the wait/timeout clock (started at the first try),
///   - the gravity-aligned → yaw-overlayed composition R_world_lidar,
///   - the corresponding R_world_imu_final for IMU preintegration relinearization.
///
/// The caller polls try_align() once per scan while is_done() is false.  On success
/// the result must be applied by the caller to its own state (pose, gyro bias, IMU
/// preintegration reset) because state layouts differ between LO and LIO.
class InitialAlignmentEstimator {
public:
    using Ptr = std::shared_ptr<InitialAlignmentEstimator>;

    enum class Status : std::int8_t {
        success = 0,
        waiting,  ///< not enough data / not stationary yet — keep polling
    };

    struct Output {
        Status status = Status::waiting;
        std::string error_message;  ///< populated on Status::waiting (diagnostic)

        // Valid only when status == success.
        Eigen::Matrix3f R_world_lidar = Eigen::Matrix3f::Identity();  ///< gravity-aligned + user yaw
        Eigen::Matrix3f R_world_imu = Eigen::Matrix3f::Identity();    ///< matching IMU rotation
        Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();
        float roll_rad = 0.0f;
        float pitch_rad = 0.0f;
        float yaw_preserved_rad = 0.0f;
        float accel_norm = 0.0f;
    };

    InitialAlignmentEstimator(const InitialAlignmentParams& params, const Eigen::Vector3f& gravity_world,
                              const Eigen::Isometry3f& T_imu_to_lidar)
        : params_(params), gravity_world_(gravity_world), T_imu_to_lidar_(T_imu_to_lidar) {}

    bool enabled() const { return this->params_.enable; }
    bool is_done() const { return this->done_; }

    /// @brief Attempt one alignment iteration.
    ///
    /// @param scan_timestamp      Timestamp of the current scan; used for the timeout clock.
    /// @param imu_buffer          Snapshot of buffered IMU samples (chronological).
    /// @param current_bias        Current IMU bias estimate (subtracted from mean accel).
    /// @param R_world_lidar_user  User-specified initial LiDAR rotation; yaw is extracted
    ///                            and preserved while roll/pitch are replaced by the
    ///                            gravity-aligned estimate.
    Output try_align(double scan_timestamp, const std::deque<IMUMeasurement>& imu_buffer,
                     const IMUBias& current_bias, const Eigen::Matrix3f& R_world_lidar_user) {
        Output out;
        if (this->done_) {
            out.status = Status::success;
            return out;
        }

        if (this->alignment_start_timestamp_ < 0.0) {
            this->alignment_start_timestamp_ = scan_timestamp;
        }
        const double elapsed = scan_timestamp - this->alignment_start_timestamp_;
        const bool timeout_reached =
            this->params_.max_wait_sec > 0.0f && elapsed >= static_cast<double>(this->params_.max_wait_sec);

        auto result = estimate_initial_alignment(imu_buffer, this->gravity_world_, this->params_, current_bias,
                                                 /*bypass_stationarity=*/false);

        if (!result.success && timeout_reached) {
            result = estimate_initial_alignment(imu_buffer, this->gravity_world_, this->params_, current_bias,
                                                /*bypass_stationarity=*/true);
            if (result.success) {
                std::cerr << "[InitialAlignment] initial alignment FORCED after " << elapsed
                          << "s (robot was not detected stationary). gyro_bias may be biased; "
                          << "drift performance can degrade until convergence." << std::endl;
            }
        }

        if (!result.success) {
            out.status = Status::waiting;
            out.error_message = result.error_message;
            const double span =
                imu_buffer.size() >= 2 ? (imu_buffer.back().timestamp - imu_buffer.front().timestamp) : 0.0;
            std::cerr << "[InitialAlignment] waiting initial alignment: " << result.error_message
                      << " (samples=" << imu_buffer.size() << ", buffer_span=" << span
                      << "s, required=" << this->params_.required_duration_sec << "s, elapsed=" << elapsed << "s/"
                      << this->params_.max_wait_sec << "s, accel_mean_norm=" << result.accel_norm << ", gyro_std=["
                      << result.gyro_std.transpose() << "], accel_std=[" << result.accel_std.transpose() << "])"
                      << std::endl;
            return out;
        }

        // Compose: R_world_lidar = R_world_imu_aligned * R_imu_to_lidar^T (yaw≈0),
        // then overlay user-requested yaw.
        const Eigen::Matrix3f R_imu_to_lidar = this->T_imu_to_lidar_.linear();
        const Eigen::Matrix3f R_lidar_grav_only = result.R_world_imu * R_imu_to_lidar.transpose();
        const float yaw_grav = detail::yaw_from_rotation(R_lidar_grav_only);
        const Eigen::Matrix3f R_lidar_grav_no_yaw = detail::rotation_z(-yaw_grav) * R_lidar_grav_only;

        const float yaw_user = detail::yaw_from_rotation(R_world_lidar_user);
        const Eigen::Matrix3f R_world_lidar = detail::rotation_z(yaw_user) * R_lidar_grav_no_yaw;
        const Eigen::Matrix3f R_world_imu_final = R_world_lidar * R_imu_to_lidar;

        out.status = Status::success;
        out.R_world_lidar = R_world_lidar;
        out.R_world_imu = R_world_imu_final;
        out.gyro_bias = result.gyro_bias;
        out.roll_rad = result.roll_rad;
        out.pitch_rad = result.pitch_rad;
        out.yaw_preserved_rad = yaw_user;
        out.accel_norm = result.accel_norm;

        this->done_ = true;

        std::cout << "[InitialAlignment] initial alignment done: "
                  << "roll=" << result.roll_rad * 180.0f / static_cast<float>(M_PI) << " deg, "
                  << "pitch=" << result.pitch_rad * 180.0f / static_cast<float>(M_PI) << " deg, "
                  << "yaw_preserved=" << yaw_user * 180.0f / static_cast<float>(M_PI) << " deg, "
                  << "|a|=" << result.accel_norm << " m/s^2, "
                  << "gyro_bias=[" << result.gyro_bias.transpose() << "]" << std::endl;
        return out;
    }

private:
    InitialAlignmentParams params_;
    Eigen::Vector3f gravity_world_;
    Eigen::Isometry3f T_imu_to_lidar_;
    bool done_ = false;
    double alignment_start_timestamp_ = -1.0;
};

}  // namespace imu
}  // namespace sycl_points
