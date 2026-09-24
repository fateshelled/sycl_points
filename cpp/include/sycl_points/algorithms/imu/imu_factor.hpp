#pragma once

#include <Eigen/Dense>

#include "sycl_points/utils/eigen_utils.hpp"

// ---------------------------------------------------------------------------
// IMU factor for tightly-coupled LiDAR-IMU Odometry (LIO)
//
// This header provides:
//   - State        : 15-DOF navigation state (position, SO(3) rotation,
//                    velocity, accelerometer bias, gyroscope bias)
//   - compute_imu_hessian_gradient()
//                  : Linearises the IMU prior cost around a given operating
//                    point for use in the LIO Gauss-Newton optimisation loop.
//
// State-vector ordering (15-D error-state / tangent-space):
//   indices  0– 2  position           (3-D, world frame)
//   indices  3– 5  rotation           (3-D, so(3) tangent, right-perturbation)
//   indices  6– 8  velocity           (3-D, world frame)
//   indices  9–11  accelerometer bias (3-D, body frame)
//   indices 12–14  gyroscope bias     (3-D, body frame)
//
// The LiDAR-IMU extrinsic is no longer part of the optimised state.  It is
// held as a static calibration value at the pipeline level (params_.imu.T_imu_to_lidar).
// ---------------------------------------------------------------------------

namespace sycl_points {
namespace imu {

// ---------------------------------------------------------------------------
// State  – full 15-DOF navigation state
// ---------------------------------------------------------------------------

/// @brief Full navigation state used by the LIO optimisation back-end.
///
/// The rotation is stored as a 3×3 matrix on SO(3).  Inside the optimisation
/// all perturbations are expressed in the 3-D tangent space (Lie algebra so(3))
/// using a right-perturbation convention.
///
/// The named index constants (kIdx*) identify the start of each 3-D block
/// in the 15-D error-state vector and should be used instead of magic numbers.
struct State {
    /// Start indices of each 3-D block in the 15-D error-state / tangent vector.
    static constexpr int kIdxPos = 0;       ///< position           (indices  0– 2)
    static constexpr int kIdxRot = 3;       ///< rotation           (indices  3– 5)
    static constexpr int kIdxVel = 6;       ///< velocity           (indices  6– 8)
    static constexpr int kIdxAccBias = 9;   ///< accel bias         (indices  9–11)
    static constexpr int kIdxGyrBias = 12;  ///< gyro bias          (indices 12–14)

    static constexpr int kDOF = 15;

    Eigen::Vector3f position = Eigen::Vector3f::Zero();      ///< World-frame position [m]
    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();  ///< Body-to-world rotation R ∈ SO(3)
    Eigen::Vector3f velocity = Eigen::Vector3f::Zero();      ///< World-frame velocity [m/s]
    Eigen::Vector3f accel_bias = Eigen::Vector3f::Zero();    ///< Accelerometer bias [m/s²]
    Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();     ///< Gyroscope bias [rad/s]

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ---------------------------------------------------------------------------
// compute_manifold_residual
// ---------------------------------------------------------------------------

/// @brief Compute the 15-D manifold residual r = x_op ⊖ x_pred.
///
/// Vector quantities use plain subtraction; SO(3) quantities use the group
/// logarithm (right-perturbation convention).  Shared by both
/// compute_imu_hessian_gradient().
inline Eigen::Matrix<float, 15, 1> compute_manifold_residual(const State& x_pred, const State& x_op) {
    Eigen::Matrix<float, 15, 1> r;

    r.segment<3>(State::kIdxPos) = x_op.position - x_pred.position;

    const Eigen::Matrix3f R_relative = x_pred.rotation.transpose() * x_op.rotation;
    const Eigen::Vector4f q_relative = eigen_utils::geometry::rotation_matrix_to_quaternion(R_relative);
    r.segment<3>(State::kIdxRot) = eigen_utils::lie::so3_log(q_relative);

    r.segment<3>(State::kIdxVel) = x_op.velocity - x_pred.velocity;
    r.segment<3>(State::kIdxAccBias) = x_op.accel_bias - x_pred.accel_bias;
    r.segment<3>(State::kIdxGyrBias) = x_op.gyro_bias - x_pred.gyro_bias;

    return r;
}

// ---------------------------------------------------------------------------
// compute_imu_hessian_gradient
// ---------------------------------------------------------------------------

inline Eigen::Matrix3f so3_right_jacobian_inverse(const Eigen::Vector3f& phi) {
    const float theta_sq = phi.squaredNorm();
    const Eigen::Matrix3f Phi = eigen_utils::lie::skew(phi);
    float coefficient = 1.0f / 12.0f;
    if (theta_sq > 1e-8f) {
        const float theta = std::sqrt(theta_sq);
        const float half_theta = 0.5f * theta;
        coefficient = (1.0f - half_theta * std::cos(half_theta) / std::sin(half_theta)) / theta_sq;
    }
    return Eigen::Matrix3f::Identity() + 0.5f * Phi + coefficient * Phi * Phi;
}

inline bool compute_imu_information(const Eigen::Matrix<float, 15, 15>& P_pred,
                                    Eigen::Matrix<float, 15, 15>& information) {
    if (!P_pred.allFinite()) return false;
    Eigen::LDLT<Eigen::Matrix<float, 15, 15>> ldlt(P_pred);
    if (ldlt.info() != Eigen::Success || ldlt.vectorD().minCoeff() <= 0.0f) return false;
    information.setIdentity();
    ldlt.solveInPlace(information);
    return information.allFinite();
}

inline void linearize_imu_prior(const State& x_pred, const State& x_op,
                                const Eigen::Matrix<float, 15, 15>& information,
                                Eigen::Matrix<float, 15, 15>& H_imu, Eigen::Matrix<float, 15, 1>& b_imu) {
    const Eigen::Matrix<float, 15, 1> r = compute_manifold_residual(x_pred, x_op);
    Eigen::Matrix<float, 15, 15> J = Eigen::Matrix<float, 15, 15>::Identity();
    J.block<3, 3>(State::kIdxRot, State::kIdxRot) =
        so3_right_jacobian_inverse(r.segment<3>(State::kIdxRot));
    H_imu = J.transpose() * information * J;
    b_imu = J.transpose() * information * r;
}

/// @brief Compute the Hessian and gradient of the IMU prior term.
///
/// The IMU prior cost is the Mahalanobis distance between the current
/// operating-point state x_op and the IMU-preintegration prediction x_pred:
///
///   J_imu(x) = ½ · rᵀ · P_pred⁻¹ · r
///
/// where  r = x_op ⊖ x_pred  is computed on the state manifold (SO(3)
/// part uses the group logarithm; all other parts use plain subtraction).
///
/// Linearising around x_op gives the Gauss-Newton normal equations
///
///   H_imu · δx = −b_imu
///
/// with J containing the inverse right Jacobian of the SO(3) Log residual:
///   H_imu = Jᵀ · P_pred⁻¹ · J
///   b_imu = Jᵀ · P_pred⁻¹ · r
///
/// @param x_pred   IMU-preintegration prediction (prior mean).
/// @param x_op     Current Gauss-Newton operating point (linearisation point).
/// @param P_pred   15×15 prior covariance.  Must be symmetric positive-definite.
/// @param[out] H_imu  15×15 Gauss-Newton Hessian.
/// @param[out] b_imu  15×1 gradient vector.
/// @return true on success; false if P_pred is ill-conditioned (H_imu and
///         b_imu are set to zero in that case).
inline bool compute_imu_hessian_gradient(const State& x_pred, const State& x_op,
                                         const Eigen::Matrix<float, 15, 15>& P_pred,
                                         Eigen::Matrix<float, 15, 15>& H_imu, Eigen::Matrix<float, 15, 1>& b_imu) {
    // ------------------------------------------------------------------
    // 1. Information matrix  H_imu = P_pred⁻¹
    // ------------------------------------------------------------------
    Eigen::Matrix<float, 15, 15> information;
    if (!compute_imu_information(P_pred, information)) {
        H_imu.setZero();
        b_imu.setZero();
        return false;
    }
    linearize_imu_prior(x_pred, x_op, information, H_imu, b_imu);
    return H_imu.allFinite() && b_imu.allFinite();
}

}  // namespace imu
}  // namespace sycl_points
