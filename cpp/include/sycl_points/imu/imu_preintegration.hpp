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
    double timestamp = 0.0;                           ///< Absolute wall time [s]
    Eigen::Vector3f gyro = Eigen::Vector3f::Zero();   ///< Angular velocity [rad/s], body frame, raw
    Eigen::Vector3f accel = Eigen::Vector3f::Zero();  ///< Linear acceleration [m/s^2], body frame, raw

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Gyroscope and accelerometer biases.
struct IMUBias {
    Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();   ///< [rad/s]
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

    /// @brief 15×15 covariance matrix of the navigation state.
    ///
    /// Propagated from the initial_covariance supplied to reset() by the
    /// discrete error-state dynamics:
    ///
    ///   Σ_{k+1} = F_k · Σ_k · F_k^T  +  G_k · Q_d · G_k^T
    ///
    /// State-vector ordering (matches imu::State in imu_factor.hpp):
    ///   indices  0– 2  position           (world frame)
    ///   indices  3– 5  rotation           (so(3) tangent, right-perturbation)
    ///   indices  6– 8  velocity           (world frame)
    ///   indices  9–11  accelerometer bias (body frame)
    ///   indices 12–14  gyroscope bias     (body frame)
    ///
    /// Pass directly as P_pred to compute_imu_hessian_gradient().
    /// Remains zero if all IMUPreintegrationParams noise densities are zero.
    Eigen::Matrix<float, 15, 15> covariance = Eigen::Matrix<float, 15, 15>::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Parameters for IMU preintegration.
struct IMUPreintegrationParams {
    /// Gravity vector in the world frame [m/s^2]. Default: z-down.
    Eigen::Vector3f gravity = Eigen::Vector3f(0.0f, 0.0f, -9.80665f);

    /// Scale factor applied to raw accelerometer measurements before integration.
    /// Set to 9.80665 when the IMU reports acceleration in [G] instead of [m/s^2].
    float accel_scale = 1.0f;

    /// @name IMU noise parameters for 15×15 covariance propagation.
    ///
    /// Set non-zero values to enable covariance propagation.  The noise model
    /// follows the standard IMU calibration convention (e.g. Kalibr):
    ///   - Measurement noise  PSD: σ²  → discrete variance = σ² / dt per step.
    ///   - Bias random-walk   PSD: σ²  → discrete variance = σ² * dt per step.
    ///
    /// Typical MEMS IMU values (order-of-magnitude):
    ///   gyro_noise_density    ≈ 1e-3  rad/s/√Hz
    ///   accel_noise_density   ≈ 1e-2  m/s²/√Hz
    ///   gyro_bias_rw_density  ≈ 1e-5  rad/s²/√Hz
    ///   accel_bias_rw_density ≈ 1e-4  m/s³/√Hz
    /// @{
    float gyro_noise_density = 0.0f;     ///< Gyroscope white noise density [rad/s/√Hz]
    float accel_noise_density = 0.0f;    ///< Accelerometer white noise density [m/s²/√Hz]
    float gyro_bias_rw_density = 0.0f;   ///< Gyroscope bias random-walk density [rad/s²/√Hz]
    float accel_bias_rw_density = 0.0f;  ///< Accelerometer bias random-walk density [m/s³/√Hz]
    /// @}

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

    explicit IMUPreintegration(const IMUPreintegrationParams& params = IMUPreintegrationParams()) : params_(params) {}

    /// @brief Reset the integrator (call at every new LiDAR keyframe).
    /// @param bias               Current bias estimate used as the linearization point.
    /// @param R_world_body_i     Rotation from body to world frame at the window start.
    ///                           Required to compensate gravity in predict_relative_transform().
    /// @param v_world_i          Linear velocity in the world frame at the window start [m/s].
    ///                           Used by predict_relative_transform() to add the constant-
    ///                           velocity contribution (v * dt) to the predicted displacement.
    ///                           Defaults to zero.
    /// @param initial_covariance 15×15 state covariance at the window start.
    ///                           Typically the posterior covariance P from the previous
    ///                           keyframe optimisation.  Propagated forward together with
    ///                           the IMU process noise; pass as P_pred to
    ///                           compute_imu_hessian_gradient().
    ///                           Defaults to zero (uncertainty from IMU noise only).
    void reset(const IMUBias& bias = IMUBias(), const Eigen::Matrix3f& R_world_body_i = Eigen::Matrix3f::Identity(),
               const Eigen::Vector3f& v_world_i = Eigen::Vector3f::Zero(),
               const Eigen::Matrix<float, 15, 15>& initial_covariance = Eigen::Matrix<float, 15, 15>::Zero()) {
        bias_lin_ = bias;
        result_ = PreintegrationResult{};
        result_.covariance = initial_covariance;
        has_prev_ = false;
        num_measurements_ = 0;
        step_count_ = 0;
        R_world_body_i_ = R_world_body_i;
        v_world_i_ = v_world_i;
    }

    /// @brief Feed one IMU sample.  Integration starts after the second call.
    ///        Out-of-order or duplicate samples (non-increasing timestamp) are silently dropped
    ///        without corrupting the previous measurement used for midpoint integration.
    void integrate(const IMUMeasurement& meas) {
        if (!has_prev_) {
            prev_meas_ = meas;
            has_prev_ = true;
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
    void integrate_batch(const std::vector<IMUMeasurement, Eigen::aligned_allocator<IMUMeasurement>>& measurements) {
        for (const auto& m : measurements) {
            integrate(m);
        }
    }

    /// @brief Return bias-corrected preintegrated state using first-order Jacobians.
    ///        This avoids full re-integration when the bias estimate changes slightly.
    PreintegrationResult get_corrected(const IMUBias& new_bias) const {
        const Eigen::Vector3f d_bg = new_bias.gyro_bias - bias_lin_.gyro_bias;
        const Eigen::Vector3f d_ba = new_bias.accel_bias - bias_lin_.accel_bias;

        // copy
        PreintegrationResult corrected = result_;

        // Correct rotation: Delta_R_corr = Delta_R * Exp(J_R_bg * d_bg)
        const Eigen::Vector3f phi_corr = result_.J.J_R_bg * d_bg;
        const Eigen::Vector4f q_corr = eigen_utils::lie::so3_exp(phi_corr);
        const Eigen::Matrix3f R_corr = eigen_utils::geometry::quaternion_to_rotation_matrix(q_corr);
        corrected.Delta_R *= R_corr;

        corrected.Delta_v += result_.J.J_v_bg * d_bg + result_.J.J_v_ba * d_ba;
        corrected.Delta_p += result_.J.J_p_bg * d_bg + result_.J.J_p_ba * d_ba;

        return corrected;
    }

    /// @brief Return the raw preintegrated state at the linearization-point bias.
    const PreintegrationResult& get_raw() const { return result_; }

    /// @brief Predict the absolute world-frame pose at the end of the window.
    ///
    /// @param T_world_body_i  World-to-body transform at window start (4x4 SE3).
    /// @param v_world_i       Body velocity in the world frame at window start [m/s].
    /// @param current_bias    Current bias estimate (may differ from linearization bias).
    /// @return Predicted world-frame pose T_world_body_j (TransformMatrix = Matrix4f).
    TransformMatrix predict_transform(const TransformMatrix& T_world_body_i, const Eigen::Vector3f& v_world_i,
                                      const IMUBias& current_bias) const {
        const PreintegrationResult c = get_corrected(current_bias);
        const float dt_f = static_cast<float>(c.dt_total);

        const Eigen::Matrix3f R_i = T_world_body_i.block<3, 3>(0, 0);
        const Eigen::Vector3f p_i = T_world_body_i.block<3, 1>(0, 3);

        const Eigen::Matrix3f R_j = R_i * c.Delta_R;
        const Eigen::Vector3f p_j = p_i + v_world_i * dt_f + 0.5f * params_.gravity * dt_f * dt_f + R_i * c.Delta_p;

        TransformMatrix T_j = TransformMatrix::Identity();
        T_j.block<3, 3>(0, 0) = R_j;
        T_j.block<3, 1>(0, 3) = p_j;
        return T_j;
    }

    /// @brief Predict the relative transform from window start to window end.
    ///        Gravity is compensated using the initial orientation supplied to reset().
    ///        The initial velocity supplied to reset() is also included, so the returned
    ///        translation captures both constant-velocity motion and acceleration-induced
    ///        displacement.  Suitable for direct use as an ICP initial guess.
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

        // Add the initial velocity contribution in window-start body frame.
        // v_world * dt is expressed in body frame as R_i^T * v_world * dt.
        const Eigen::Vector3f delta_p = delta_p_grav_free + R_world_body_i_.transpose() * v_world_i_ * dt_f;

        TransformMatrix T_rel = TransformMatrix::Identity();
        T_rel.block<3, 3>(0, 0) = c.Delta_R;
        T_rel.block<3, 1>(0, 3) = delta_p;
        return T_rel;
    }

    double get_dt_total() const { return result_.dt_total; }
    bool has_measurements() const { return num_measurements_ > 0; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    /// @brief Right Jacobian of SO(3): Jr(phi) = d Exp(phi) / d phi.
    static Eigen::Matrix3f right_jacobian_so3(const Eigen::Vector3f& phi) {
        const float theta = phi.norm();
        const Eigen::Matrix3f S = eigen_utils::lie::skew(phi);
        const Eigen::Matrix3f S2 = S * S;
        if (theta < 1e-4f) {
            // Second-order Taylor: Jr ≈ I - 0.5*S + (1/6)*S²
            // Avoids catastrophic cancellation in (1-cos θ)/θ² and (θ-sin θ)/θ³
            // for small θ in single-precision float (machine eps ~1.2e-7).
            return Eigen::Matrix3f::Identity() - 0.5f * S + (1.0f / 6.0f) * S2;
        }
        return Eigen::Matrix3f::Identity()                       //
               - (1.0f - std::cos(theta)) / (theta * theta) * S  //
               + (theta - std::sin(theta)) / (theta * theta * theta) * S2;
    }

    /// @brief Integrate one step from measurement m0 to m1 using midpoint (RK2).
    void integrate_step(const IMUMeasurement& m0, const IMUMeasurement& m1) {
        const double dt = m1.timestamp - m0.timestamp;
        if (dt < 1e-9) return;
        const float dt_f = static_cast<float>(dt);

        // Bias-corrected measurements (accel scaled to [m/s^2] if sensor outputs [G])
        const Eigen::Vector3f omega_0 = m0.gyro - bias_lin_.gyro_bias;
        const Eigen::Vector3f omega_1 = m1.gyro - bias_lin_.gyro_bias;
        const Eigen::Vector3f a_0 = m0.accel * params_.accel_scale - bias_lin_.accel_bias;
        const Eigen::Vector3f a_1 = m1.accel * params_.accel_scale - bias_lin_.accel_bias;

        // Midpoint values
        const Eigen::Vector3f omega_mid = 0.5f * (omega_0 + omega_1);
        const Eigen::Vector3f a_mid = 0.5f * (a_0 + a_1);

        // Full-step rotation increment  Exp(omega_mid * dt)
        const Eigen::Vector3f phi_mid = omega_mid * dt_f;
        const Eigen::Vector4f q_step = eigen_utils::lie::so3_exp(phi_mid);
        const Eigen::Matrix3f R_step = eigen_utils::geometry::quaternion_to_rotation_matrix(q_step);

        // Half-step rotation for accel integration (use omega at m0 start)
        const Eigen::Vector3f phi_half = omega_0 * (0.5f * dt_f);
        const Eigen::Vector4f q_half = eigen_utils::lie::so3_exp(phi_half);
        const Eigen::Matrix3f R_half = eigen_utils::geometry::quaternion_to_rotation_matrix(q_half);
        const Eigen::Matrix3f Delta_R_mid = result_.Delta_R * R_half;

        // Save pre-update values for Jacobian propagation
        const Eigen::Matrix3f J_R_bg_old = result_.J.J_R_bg;
        const Eigen::Matrix3f J_v_bg_old = result_.J.J_v_bg;
        const Eigen::Matrix3f J_v_ba_old = result_.J.J_v_ba;
        const Eigen::Vector3f Delta_v_old = result_.Delta_v;

        // Rotate midpoint accel into the navigation frame of the window start
        const Eigen::Vector3f a_nav = Delta_R_mid * a_mid;

        // --- State update ---
        result_.Delta_R = result_.Delta_R * R_step;
        result_.Delta_p += Delta_v_old * dt_f + 0.5f * a_nav * dt_f * dt_f;
        result_.Delta_v = Delta_v_old + a_nav * dt_f;
        result_.dt_total += dt;

        // --- Jacobian propagation (first-order discrete-time recurrence) ---
        const Eigen::Matrix3f Jr = right_jacobian_so3(phi_mid);
        const Eigen::Matrix3f skew_a = eigen_utils::lie::skew(a_mid);

        result_.J.J_R_bg = R_step.transpose() * J_R_bg_old - Jr * dt_f;

        result_.J.J_v_bg = J_v_bg_old - Delta_R_mid * skew_a * J_R_bg_old * dt_f;
        result_.J.J_v_ba = result_.J.J_v_ba - Delta_R_mid * dt_f;

        result_.J.J_p_bg =
            result_.J.J_p_bg + J_v_bg_old * dt_f - 0.5f * Delta_R_mid * skew_a * J_R_bg_old * dt_f * dt_f;
        result_.J.J_p_ba = result_.J.J_p_ba + J_v_ba_old * dt_f - 0.5f * Delta_R_mid * dt_f * dt_f;

        // --- 15×15 covariance propagation ---
        //
        // Error-state ordering: [δp(0:3), δφ(3:6), δv(6:9), δba(9:12), δbg(12:15)]
        //
        // Discrete state-transition matrix F (15×15):
        //   δp_{k+1} = δp_k + δv_k·dt  − 0.5·ΔRm·[am×]·dt²·δφ_k − 0.5·ΔRm·dt²·δba
        //   δφ_{k+1} = R_step^T·δφ_k   − Jr·dt·δbg
        //   δv_{k+1} = δv_k             − ΔRm·[am×]·dt·δφ_k − ΔRm·dt·δba
        //   δba_{k+1}= δba_k
        //   δbg_{k+1}= δbg_k
        //
        // Note: Jr only appears in F[δφ, δbg].  The rotation-error effect on position
        // and velocity (F[δp,δφ] and F[δv,δφ]) arises from the rotation of the
        // accelerometer measurement, which does not involve the exponential-map Jacobian.
        //
        // Process noise Q = G·Q_d·G^T (computed analytically, sparse):
        //   Accel white noise (σ_a, PSD σ_a²):
        //     G[δp, na] = 0.5·ΔRm·dt²,  G[δv, na] = ΔRm·dt,  Q_na = σ_a²/dt · I
        //   Gyro white noise (σ_g, PSD σ_g²):
        //     G[δφ, ng] = Jr·dt,                              Q_ng = σ_g²/dt · I
        //   Bias random-walk noise:
        //     G[δba, nba] = I,  Q_nba = σ_ba²·dt · I
        //     G[δbg, nbg] = I,  Q_nbg = σ_bg²·dt · I
        //
        // F is always applied so that existing uncertainty (e.g. velocity error from
        // initial_covariance) propagates through state dynamics even when all noise
        // parameters are zero.  Q is only non-zero when noise parameters are set.
        // Skip the entire block only when both covariance and Q are trivially zero.
        const bool has_noise = (params_.gyro_noise_density > 0.0f || params_.accel_noise_density > 0.0f ||
                                params_.gyro_bias_rw_density > 0.0f || params_.accel_bias_rw_density > 0.0f);
        if (has_noise || !result_.covariance.isZero()) {
            // --- Build F (15×15) ---
            Eigen::Matrix<float, 15, 15> F = Eigen::Matrix<float, 15, 15>::Identity();

            // δp row  (Jr is NOT used here; only needed for the bias→rotation block)
            F.block<3, 3>(0, 3) = -0.5f * Delta_R_mid * skew_a * dt_f * dt_f;
            F.block<3, 3>(0, 6) = Eigen::Matrix3f::Identity() * dt_f;
            F.block<3, 3>(0, 9) = -0.5f * Delta_R_mid * dt_f * dt_f;

            // δφ row  (Jr applies only to the bias term)
            F.block<3, 3>(3, 3) = R_step.transpose();
            F.block<3, 3>(3, 12) = -Jr * dt_f;

            // δv row  (Jr is NOT used here)
            F.block<3, 3>(6, 3) = -Delta_R_mid * skew_a * dt_f;
            F.block<3, 3>(6, 9) = -Delta_R_mid * dt_f;

            // --- Build process noise Q = G·Q_d·G^T (sparse, closed-form) ---
            const float dt2 = dt_f * dt_f;
            const float dt3 = dt2 * dt_f;

            Eigen::Matrix<float, 15, 15> Q = Eigen::Matrix<float, 15, 15>::Zero();

            if (has_noise) {
                const float sa2 = params_.accel_noise_density * params_.accel_noise_density;
                const float sg2 = params_.gyro_noise_density * params_.gyro_noise_density;
                const float sba2 = params_.accel_bias_rw_density * params_.accel_bias_rw_density;
                const float sbg2 = params_.gyro_bias_rw_density * params_.gyro_bias_rw_density;

                // Accel noise contributions (Q_na = sa2/dt·I):
                //   [δp,δp]: G[δp,na]·Q_na·G[δp,na]^T = 0.5·dt²·(sa2/dt)·0.5·dt²·I = sa2·dt³/4·I
                //   [δp,δv]: G[δp,na]·Q_na·G[δv,na]^T = 0.5·dt²·(sa2/dt)·dt·I      = sa2·dt²/2·I
                //   [δv,δv]: G[δv,na]·Q_na·G[δv,na]^T = dt·(sa2/dt)·dt·I            = sa2·dt·I
                Q.block<3, 3>(0, 0) += (sa2 * dt3 / 4.0f) * Eigen::Matrix3f::Identity();
                Q.block<3, 3>(0, 6) += (sa2 * dt2 / 2.0f) * Eigen::Matrix3f::Identity();
                Q.block<3, 3>(6, 0) += (sa2 * dt2 / 2.0f) * Eigen::Matrix3f::Identity();
                Q.block<3, 3>(6, 6) += (sa2 * dt_f) * Eigen::Matrix3f::Identity();

                // Gyro noise contribution (Q_ng = sg2/dt·I):
                //   [δφ,δφ]: Jr·dt·(sg2/dt)·dt·Jr^T = sg2·dt·Jr·Jr^T
                Q.block<3, 3>(3, 3) += (sg2 * dt_f) * (Jr * Jr.transpose());

                // Bias random-walk contributions (Q_nba = sba2·dt·I, Q_nbg = sbg2·dt·I):
                Q.block<3, 3>(9, 9) += (sba2 * dt_f) * Eigen::Matrix3f::Identity();
                Q.block<3, 3>(12, 12) += (sbg2 * dt_f) * Eigen::Matrix3f::Identity();
            }

            // Propagate: Σ_{k+1} = F·Σ_k·F^T + Q
            // Enforce symmetry explicitly to prevent numerical drift in single precision.
            result_.covariance = eigen_utils::ensure_symmetric<15>(F * result_.covariance * F.transpose() + Q);
        }

        // --- Periodically renormalize Delta_R to stay on SO(3) ---
        ++step_count_;
        if (step_count_ % 100 == 0) {
            const Eigen::Vector4f q_norm = eigen_utils::geometry::rotation_matrix_to_quaternion(result_.Delta_R);
            result_.Delta_R = eigen_utils::geometry::quaternion_to_rotation_matrix(q_norm);
        }
    }

    IMUPreintegrationParams params_;
    IMUBias bias_lin_;
    PreintegrationResult result_;
    IMUMeasurement prev_meas_;
    Eigen::Matrix3f R_world_body_i_ = Eigen::Matrix3f::Identity();
    Eigen::Vector3f v_world_i_ = Eigen::Vector3f::Zero();
    bool has_prev_ = false;
    int num_measurements_ = 0;
    int step_count_ = 0;
};

}  // namespace imu
}  // namespace sycl_points
