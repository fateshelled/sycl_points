#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <vector>

#include "sycl_points/points/types.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace imu {

/// @brief A single IMU measurement sample.
struct IMUMeasurement {
    double timestamp = 0.0;          ///< Absolute wall time [s]
    Eigen::Vector3f gyro  = Eigen::Vector3f::Zero();   ///< Angular velocity [rad/s], body frame, raw
    Eigen::Vector3f accel = Eigen::Vector3f::Zero();   ///< Linear acceleration [m/s^2], body frame, raw

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Gyroscope and accelerometer biases.
struct IMUBias {
    Eigen::Vector3f gyro_bias  = Eigen::Vector3f::Zero();  ///< [rad/s]
    Eigen::Vector3f accel_bias = Eigen::Vector3f::Zero();  ///< [m/s^2]

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief First-order Jacobians of the preintegrated state w.r.t. biases.
struct PreintegrationJacobians {
    Eigen::Matrix3f J_R_bg = Eigen::Matrix3f::Zero();  ///< d(Delta_R_log) / d(b_g)
    Eigen::Matrix3f J_v_bg = Eigen::Matrix3f::Zero();  ///< d(Delta_v) / d(b_g)
    Eigen::Matrix3f J_v_ba = Eigen::Matrix3f::Zero();  ///< d(Delta_v) / d(b_a)
    Eigen::Matrix3f J_p_bg = Eigen::Matrix3f::Zero();  ///< d(Delta_p) / d(b_g)
    Eigen::Matrix3f J_p_ba = Eigen::Matrix3f::Zero();  ///< d(Delta_p) / d(b_a)

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Accumulated preintegrated IMU state between two keyframes.
struct PreintegrationResult {
    Eigen::Matrix3f Delta_R = Eigen::Matrix3f::Identity();  ///< Relative rotation SO(3)
    Eigen::Vector3f Delta_v = Eigen::Vector3f::Zero();      ///< Relative velocity [m/s]
    Eigen::Vector3f Delta_p = Eigen::Vector3f::Zero();      ///< Relative position [m]
    double dt_total = 0.0;                                  ///< Total integrated duration [s]
    PreintegrationJacobians J;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Parameters for IMU preintegration.
struct IMUPreintegrationParams {
    /// Gravity vector in the world frame [m/s^2]. Default: z-down.
    Eigen::Vector3f gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief IMU preintegration on SO(3) with midpoint (RK2) integration.
///
/// Implements Forster et al. 2017 "On-Manifold Preintegration for Real-Time
/// Visual-Inertial Odometry" (TRO). The class accumulates IMU measurements
/// between two LiDAR keyframes and provides a predicted relative transform
/// suitable as an ICP initial guess.
///
/// Usage:
///   1. Call reset() at each new LiDAR keyframe.
///   2. Call integrate() for every arriving IMU sample.
///   3. Before running ICP, call predict_relative_transform() to get the
///      initial guess, then reset() again.
class IMUPreintegration {
public:
    using Ptr = std::shared_ptr<IMUPreintegration>;

    explicit IMUPreintegration(const IMUPreintegrationParams& params = IMUPreintegrationParams())
        : params_(params) {}

    /// @brief Reset the integrator (call at every new LiDAR keyframe).
    /// @param bias           Current bias estimate used as the linearization point.
    /// @param R_world_body_i Rotation from body to world frame at the window start.
    ///                       Required to compensate gravity in predict_relative_transform().
    void reset(const IMUBias& bias = IMUBias(),
               const Eigen::Matrix3f& R_world_body_i = Eigen::Matrix3f::Identity()) {
        bias_lin_         = bias;
        result_           = PreintegrationResult{};
        has_prev_         = false;
        num_measurements_ = 0;
        step_count_       = 0;
        R_world_body_i_   = R_world_body_i;
    }

    /// @brief Feed one IMU sample.  Integration starts after the second call.
    ///        Out-of-order or duplicate samples (non-increasing timestamp) are silently dropped
    ///        without corrupting the previous measurement used for midpoint integration.
    void integrate(const IMUMeasurement& meas) {
        if (!has_prev_) {
            prev_meas_ = meas;
            has_prev_  = true;
            ++num_measurements_;
            return;
        }
        if (meas.timestamp <= prev_meas_.timestamp) {
            return;  // drop without overwriting prev_meas_
        }
        integrate_step(prev_meas_, meas);
        prev_meas_ = meas;
        ++num_measurements_;
    }

    /// @brief Convenience wrapper to integrate a batch of measurements.
    void integrate_batch(
        const std::vector<IMUMeasurement, Eigen::aligned_allocator<IMUMeasurement>>& measurements) {
        for (const auto& m : measurements) {
            integrate(m);
        }
    }

    /// @brief Return bias-corrected preintegrated state using first-order Jacobians.
    ///        This avoids full re-integration when the bias estimate changes slightly.
    PreintegrationResult get_corrected(const IMUBias& new_bias) const {
        const Eigen::Vector3f d_bg = new_bias.gyro_bias  - bias_lin_.gyro_bias;
        const Eigen::Vector3f d_ba = new_bias.accel_bias - bias_lin_.accel_bias;

        PreintegrationResult corrected = result_;

        // Correct rotation: Delta_R_corr = Delta_R * Exp(J_R_bg * d_bg)
        const Eigen::Vector3f phi_corr = result_.J.J_R_bg * d_bg;
        const Eigen::Vector4f q_corr   = eigen_utils::lie::so3_exp(phi_corr);
        const Eigen::Matrix3f R_corr   = eigen_utils::geometry::quaternion_to_rotation_matrix(q_corr);
        corrected.Delta_R = result_.Delta_R * R_corr;

        corrected.Delta_v = result_.Delta_v + result_.J.J_v_bg * d_bg + result_.J.J_v_ba * d_ba;
        corrected.Delta_p = result_.Delta_p + result_.J.J_p_bg * d_bg + result_.J.J_p_ba * d_ba;

        return corrected;
    }

    /// @brief Return the raw preintegrated state at the linearization-point bias.
    const PreintegrationResult& get_raw() const { return result_; }

    /// @brief Predict the absolute world-frame pose at the end of the window.
    ///
    /// @param T_world_body_i  World-to-body transform at window start (4x4 SE3).
    /// @param v_i_world       Body velocity in the world frame at window start [m/s].
    /// @param current_bias    Current bias estimate (may differ from linearization bias).
    /// @return Predicted world-frame pose T_world_body_j (TransformMatrix = Matrix4f).
    TransformMatrix predict_transform(
        const TransformMatrix& T_world_body_i,
        const Eigen::Vector3f& v_i_world,
        const IMUBias& current_bias) const {
        const PreintegrationResult c = get_corrected(current_bias);
        const float dt_f = static_cast<float>(c.dt_total);

        const Eigen::Matrix3f R_i = T_world_body_i.block<3, 3>(0, 0);
        const Eigen::Vector3f p_i = T_world_body_i.block<3, 1>(0, 3);

        const Eigen::Matrix3f R_j = R_i * c.Delta_R;
        const Eigen::Vector3f p_j = p_i
            + v_i_world * dt_f
            + 0.5f * params_.gravity * dt_f * dt_f
            + R_i * c.Delta_p;

        TransformMatrix T_j = TransformMatrix::Identity();
        T_j.block<3, 3>(0, 0) = R_j;
        T_j.block<3, 1>(0, 3) = p_j;
        return T_j;
    }

    /// @brief Predict the relative transform from window start to window end.
    ///        Gravity is compensated using the initial orientation supplied to reset(),
    ///        making the returned translation purely motion-induced.
    ///        Suitable for direct use as an ICP initial_guess when the absolute
    ///        world-frame velocity is not tracked.
    /// @return T_body_i_to_body_j  (TransformMatrix = Matrix4f)
    TransformMatrix predict_relative_transform(const IMUBias& current_bias) const {
        const PreintegrationResult c = get_corrected(current_bias);
        const float dt_f = static_cast<float>(c.dt_total);

        // Delta_p accumulates the specific force (accelerometer reading), which includes
        // gravity.  Subtract the gravity contribution expressed in the window-start body
        // frame so that a stationary device returns zero translation.
        // Gravity term in window-start body frame: R_i^T * g_world.
        // Its integral over [0, dt]: 0.5 * R_i^T * g_world * dt^2.
        // Adding it cancels the gravity already baked into Delta_p.
        const Eigen::Vector3f delta_p_grav_free =
            c.Delta_p + 0.5f * (R_world_body_i_.transpose() * params_.gravity) * dt_f * dt_f;

        TransformMatrix T_rel = TransformMatrix::Identity();
        T_rel.block<3, 3>(0, 0) = c.Delta_R;
        T_rel.block<3, 1>(0, 3) = delta_p_grav_free;
        return T_rel;
    }

    double get_dt_total() const { return result_.dt_total; }
    bool has_measurements() const { return num_measurements_ > 0; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    /// @brief Right Jacobian of SO(3): Jr(phi) = d Exp(phi) / d phi.
    static Eigen::Matrix3f right_jacobian_so3(const Eigen::Vector3f& phi) {
        const float theta = phi.norm();
        const Eigen::Matrix3f S  = eigen_utils::lie::skew(phi);
        const Eigen::Matrix3f S2 = S * S;
        if (theta < 1e-4f) {
            // Second-order Taylor: Jr ≈ I - 0.5*S + (1/6)*S²
            // Avoids catastrophic cancellation in (1-cos θ)/θ² and (θ-sin θ)/θ³
            // for small θ in single-precision float (machine eps ~1.2e-7).
            return Eigen::Matrix3f::Identity() - 0.5f * S + (1.0f / 6.0f) * S2;
        }
        return Eigen::Matrix3f::Identity()
            - (1.0f - std::cos(theta)) / (theta * theta) * S
            + (theta - std::sin(theta)) / (theta * theta * theta) * S2;
    }

    /// @brief Integrate one step from measurement m0 to m1 using midpoint (RK2).
    void integrate_step(const IMUMeasurement& m0, const IMUMeasurement& m1) {
        const double dt = m1.timestamp - m0.timestamp;
        if (dt < 1e-9) return;
        const float dt_f = static_cast<float>(dt);

        // Bias-corrected measurements
        const Eigen::Vector3f omega_0 = m0.gyro  - bias_lin_.gyro_bias;
        const Eigen::Vector3f omega_1 = m1.gyro  - bias_lin_.gyro_bias;
        const Eigen::Vector3f a_0     = m0.accel - bias_lin_.accel_bias;
        const Eigen::Vector3f a_1     = m1.accel - bias_lin_.accel_bias;

        // Midpoint values
        const Eigen::Vector3f omega_mid = 0.5f * (omega_0 + omega_1);
        const Eigen::Vector3f a_mid     = 0.5f * (a_0 + a_1);

        // Full-step rotation increment  Exp(omega_mid * dt)
        const Eigen::Vector3f phi_mid  = omega_mid * dt_f;
        const Eigen::Vector4f q_step   = eigen_utils::lie::so3_exp(phi_mid);
        const Eigen::Matrix3f R_step   = eigen_utils::geometry::quaternion_to_rotation_matrix(q_step);

        // Half-step rotation for accel integration (use omega at m0 start)
        const Eigen::Vector3f phi_half    = omega_0 * (0.5f * dt_f);
        const Eigen::Vector4f q_half      = eigen_utils::lie::so3_exp(phi_half);
        const Eigen::Matrix3f R_half      = eigen_utils::geometry::quaternion_to_rotation_matrix(q_half);
        const Eigen::Matrix3f Delta_R_mid = result_.Delta_R * R_half;

        // Save pre-update values for Jacobian propagation
        const Eigen::Matrix3f J_R_bg_old = result_.J.J_R_bg;
        const Eigen::Matrix3f J_v_bg_old = result_.J.J_v_bg;
        const Eigen::Matrix3f J_v_ba_old = result_.J.J_v_ba;
        const Eigen::Vector3f Delta_v_old = result_.Delta_v;

        // Rotate midpoint accel into the navigation frame of the window start
        const Eigen::Vector3f a_nav = Delta_R_mid * a_mid;

        // --- State update ---
        result_.Delta_R  = result_.Delta_R * R_step;
        result_.Delta_p += Delta_v_old * dt_f + 0.5f * a_nav * dt_f * dt_f;
        result_.Delta_v  = Delta_v_old + a_nav * dt_f;
        result_.dt_total += dt;

        // --- Jacobian propagation (first-order discrete-time recurrence) ---
        const Eigen::Matrix3f Jr     = right_jacobian_so3(phi_mid);
        const Eigen::Matrix3f skew_a = eigen_utils::lie::skew(a_mid);

        result_.J.J_R_bg = R_step.transpose() * J_R_bg_old - Jr * dt_f;

        result_.J.J_v_bg = J_v_bg_old - Delta_R_mid * skew_a * J_R_bg_old * dt_f;
        result_.J.J_v_ba = result_.J.J_v_ba - Delta_R_mid * dt_f;

        result_.J.J_p_bg = result_.J.J_p_bg + J_v_bg_old * dt_f
                           - 0.5f * Delta_R_mid * skew_a * J_R_bg_old * dt_f * dt_f;
        result_.J.J_p_ba = result_.J.J_p_ba + J_v_ba_old * dt_f
                           - 0.5f * Delta_R_mid * dt_f * dt_f;

        // --- Periodically renormalize Delta_R to stay on SO(3) ---
        ++step_count_;
        if (step_count_ % 100 == 0) {
            const Eigen::Vector4f q_norm =
                eigen_utils::geometry::rotation_matrix_to_quaternion(result_.Delta_R);
            result_.Delta_R = eigen_utils::geometry::quaternion_to_rotation_matrix(q_norm);
        }
    }

    IMUPreintegrationParams params_;
    IMUBias bias_lin_;
    PreintegrationResult result_;
    IMUMeasurement prev_meas_;
    Eigen::Matrix3f R_world_body_i_ = Eigen::Matrix3f::Identity();
    bool has_prev_         = false;
    int  num_measurements_ = 0;
    int  step_count_       = 0;
};

}  // namespace imu
}  // namespace sycl_points
