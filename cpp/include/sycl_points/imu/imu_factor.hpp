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
//                    point for use inside an Iterated Extended Kalman Filter
//                    (IEKF) or a Gauss-Newton optimisation loop.
//
// State-vector ordering (15-D error-state / tangent-space):
//   indices  0– 2  position           (3-D, world frame)
//   indices  3– 5  rotation           (3-D, so(3) tangent, right-perturbation)
//   indices  6– 8  velocity           (3-D, world frame)
//   indices  9–11  accelerometer bias (3-D, body frame)
//   indices 12–14  gyroscope bias     (3-D, body frame)
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
    static constexpr int kIdxPos = 0;       ///< position    (indices  0– 2)
    static constexpr int kIdxRot = 3;       ///< rotation    (indices  3– 5)
    static constexpr int kIdxVel = 6;       ///< velocity    (indices  6– 8)
    static constexpr int kIdxAccBias = 9;   ///< accel bias  (indices  9–11)
    static constexpr int kIdxGyrBias = 12;  ///< gyro bias   (indices 12–14)

    Eigen::Vector3f position = Eigen::Vector3f::Zero();      ///< World-frame position [m]
    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();  ///< Body-to-world rotation R ∈ SO(3)
    Eigen::Vector3f velocity = Eigen::Vector3f::Zero();      ///< World-frame velocity [m/s]
    Eigen::Vector3f accel_bias = Eigen::Vector3f::Zero();    ///< Accelerometer bias [m/s²]
    Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();     ///< Gyroscope bias [rad/s]

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ---------------------------------------------------------------------------
// compute_imu_hessian_gradient
// ---------------------------------------------------------------------------

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
/// with
///   H_imu = P_pred⁻¹      (information matrix)
///   b_imu = H_imu · r     (gradient of the cost)
///
/// These are accumulated together with the LiDAR geometry term inside the
/// IEKF update step.
///
/// @param x_pred   IMU-preintegration prediction (prior mean).
/// @param x_op     Current IEKF operating point (linearisation point).
/// @param P_pred   15×15 prior covariance propagated by IMU kinematics.
///                 Must be symmetric positive-definite.
/// @param[out] H_imu  15×15 information matrix  (= P_pred⁻¹).
/// @param[out] b_imu  15×1  gradient vector     (= H_imu · r).
/// @return true on success; false if P_pred is ill-conditioned (H_imu and
///         b_imu are set to zero in that case).
inline bool compute_imu_hessian_gradient(const State& x_pred, const State& x_op,
                                         const Eigen::Matrix<float, 15, 15>& P_pred,
                                         Eigen::Matrix<float, 15, 15>& H_imu, Eigen::Matrix<float, 15, 1>& b_imu) {
    // ------------------------------------------------------------------
    // 1. Information matrix  H_imu = P_pred⁻¹
    //
    //    Use LDLT (Bunch-Kaufman) factorisation: more robust than LLT for
    //    single-precision floats because it handles positive semi-definite
    //    matrices and detects numerical failures.
    //    solveInPlace(I) overwrites the identity in-place, yielding P⁻¹.
    // ------------------------------------------------------------------
    Eigen::LDLT<Eigen::Matrix<float, 15, 15>> ldlt(P_pred);
    // Eigen's LDLT::isPositive() returns true for PSD matrices (D_i >= 0),
    // which includes degenerate cases like the zero matrix.  Instead, check
    // that every diagonal entry of D is strictly positive to ensure P_pred
    // is truly invertible.
    if (ldlt.info() != Eigen::Success || ldlt.vectorD().minCoeff() <= 0.0f) {
        H_imu.setZero();
        b_imu.setZero();
        return false;
    }
    H_imu.setIdentity();
    ldlt.solveInPlace(H_imu);

    // ------------------------------------------------------------------
    // 2. Manifold residual  r = x_op ⊖ x_pred  (15×1)
    //
    //    For vector quantities (position, velocity, biases) the manifold
    //    subtraction reduces to ordinary vector subtraction.
    //    For the rotation the subtraction lives in so(3):
    //      r_R = Log(R_pred^T · R_op)
    //    which measures the rotation that maps R_pred onto R_op.
    // ------------------------------------------------------------------
    Eigen::Matrix<float, 15, 1> r;

    // Indices kIdxPos (0–2): position residual
    r.segment<3>(State::kIdxPos) = x_op.position - x_pred.position;

    // Indices kIdxRot (3–5): rotation residual on SO(3)
    //   R_pred^T · R_op  is the relative rotation from R_pred to R_op.
    //   Convert to a quaternion and apply the existing so3_log to obtain
    //   the 3-D tangent-space error vector.
    const Eigen::Matrix3f R_relative = x_pred.rotation.transpose() * x_op.rotation;
    const Eigen::Vector4f q_relative = eigen_utils::geometry::rotation_matrix_to_quaternion(R_relative);
    r.segment<3>(State::kIdxRot) = eigen_utils::lie::so3_log(q_relative);

    // Indices kIdxVel (6–8): velocity residual
    r.segment<3>(State::kIdxVel) = x_op.velocity - x_pred.velocity;

    // Indices kIdxAccBias (9–11): accelerometer bias residual
    r.segment<3>(State::kIdxAccBias) = x_op.accel_bias - x_pred.accel_bias;

    // Indices kIdxGyrBias (12–14): gyroscope bias residual
    r.segment<3>(State::kIdxGyrBias) = x_op.gyro_bias - x_pred.gyro_bias;

    // ------------------------------------------------------------------
    // 3. Gradient  b_imu = P_pred⁻¹ · r
    //
    //    Reuse the LDLT factorisation already computed in step 1 to solve
    //    P_pred · b = r directly, instead of forming H_imu = P_pred⁻¹
    //    explicitly and then computing H_imu · r.  This avoids one extra
    //    matrix–vector product and is more numerically stable.
    // ------------------------------------------------------------------
    b_imu = ldlt.solve(r);
    return true;
}

}  // namespace imu
}  // namespace sycl_points
