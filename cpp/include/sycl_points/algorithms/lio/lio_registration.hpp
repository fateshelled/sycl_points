#pragma once

#include <Eigen/Dense>

#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

// ---------------------------------------------------------------------------
// LIO (LiDAR-Inertial Odometry) registration
//
// Combines ICP and IMU prior Hessian/gradient into a unified 15-DOF normal
// equation and solves with LDLT decomposition.
//
// State-vector ordering (15-D, same as imu_factor.hpp):
//   indices  0– 2  position           δp
//   indices  3– 5  rotation           δφ  (so(3) tangent, right-perturbation)
//   indices  6– 8  velocity           δv
//   indices  9–11  accelerometer bias δb_a
//   indices 12–14  gyroscope bias     δb_g
//
// ICP 6-DOF delta ordering (registration.hpp convention):
//   indices  0– 2  rotation  δω
//   indices  3– 5  translation δt
//
// Embedding ICP into LIO (index permutation):
//   ICP δω (0–2) → LIO δφ (3–5)
//   ICP δt (3–5) → LIO δp (0–2)
// ---------------------------------------------------------------------------

namespace sycl_points {
namespace algorithms {
namespace lio {

// ---------------------------------------------------------------------------
// LIOLinearizedResult
// ---------------------------------------------------------------------------

/// @brief Combined LIO linearized system (ICP + IMU prior).
struct LIOLinearizedResult {
    Eigen::Matrix<float, 15, 15> H = Eigen::Matrix<float, 15, 15>::Zero();
    Eigen::Matrix<float, 15, 1> b = Eigen::Matrix<float, 15, 1>::Zero();
    float error_icp = 0.0f;
    float error_imu = 0.0f;
    uint32_t inlier = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ---------------------------------------------------------------------------
// add_icp_factor
// ---------------------------------------------------------------------------

/// @brief Embed ICP 6×6 Hessian/gradient into the LIO 15×15 system.
///
/// The ICP result lives in the 6-D pose tangent space [δω, δt], while the
/// LIO state is ordered [δp, δφ, δv, δb_a, δb_g].  Only the (δp, δφ) 6×6
/// block of the LIO Hessian is updated; velocity and bias rows/cols are
/// unaffected.
///
/// @param result  LIO system to accumulate into.
/// @param icp     ICP linearized result from registration::Registration.
inline void add_icp_factor(LIOLinearizedResult& result, const registration::LinearizedResult& icp) {
    // Permutation: ICP[ω=0:3, t=3:6] → LIO[φ=3:6, p=0:3]
    //
    // H_lio[p, p] += H_icp[t, t]
    result.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos) += icp.H.block<3, 3>(3, 3);
    // H_lio[p, φ] += H_icp[t, ω]
    result.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot) += icp.H.block<3, 3>(3, 0);
    // H_lio[φ, p] += H_icp[ω, t]
    result.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxPos) += icp.H.block<3, 3>(0, 3);
    // H_lio[φ, φ] += H_icp[ω, ω]
    result.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot) += icp.H.block<3, 3>(0, 0);

    // b_lio[p] += b_icp[t]
    result.b.segment<3>(imu::State::kIdxPos) += icp.b.segment<3>(3);
    // b_lio[φ] += b_icp[ω]
    result.b.segment<3>(imu::State::kIdxRot) += icp.b.segment<3>(0);

    result.error_icp += icp.error;
    result.inlier += icp.inlier;
}

// ---------------------------------------------------------------------------
// add_imu_factor
// ---------------------------------------------------------------------------

/// @brief Add IMU prior 15×15 Hessian/gradient into the LIO 15×15 system.
///
/// The IMU factor shares the same 15-D state ordering as LIOLinearizedResult,
/// so no permutation is needed.
///
/// @param result  LIO system to accumulate into.
/// @param H_imu   15×15 IMU information matrix from compute_imu_hessian_gradient().
/// @param b_imu   15×1  IMU gradient vector   from compute_imu_hessian_gradient().
/// @param error   Optional scalar IMU error to record (default 0).
inline void add_imu_factor(LIOLinearizedResult& result, const Eigen::Matrix<float, 15, 15>& H_imu,
                           const Eigen::Matrix<float, 15, 1>& b_imu, float error = 0.0f) {
    result.H += H_imu;
    result.b += b_imu;
    result.error_imu = error;
}

// ---------------------------------------------------------------------------
// solve_ldlt
// ---------------------------------------------------------------------------

/// @brief Solve the combined LIO normal equation H·δx = −b with LDLT.
///
/// Uses Eigen's Bunch-Kaufman LDLT, which is robust for positive semi-definite
/// matrices and detects numerical failures via vectorD().minCoeff().
///
/// @param[in]  lio_result  Combined LIO Hessian/gradient.
/// @param[out] delta       15-D state update δx on success; zero on failure.
/// @return true on success; false if H is ill-conditioned.
inline bool solve_ldlt(const LIOLinearizedResult& lio_result, Eigen::Matrix<float, 15, 1>& delta) {
    Eigen::LDLT<Eigen::Matrix<float, 15, 15>> ldlt(lio_result.H);
    if (ldlt.info() != Eigen::Success || ldlt.vectorD().minCoeff() <= 0.0f) {
        delta.setZero();
        return false;
    }
    delta = ldlt.solve(-lio_result.b);
    return true;
}

// ---------------------------------------------------------------------------
// retract
// ---------------------------------------------------------------------------

/// @brief Apply a 15-D tangent-space update δx to a navigation state.
///
/// Vector quantities (position, velocity, biases) use ordinary addition.
/// The rotation update uses right-perturbation on SO(3):
///   R_new = R_old · Exp(δφ)
///
/// @param state  Current navigation state (linearisation point).
/// @param delta  15-D update from solve_ldlt().
/// @return Updated navigation state.
inline imu::State retract(const imu::State& state, const Eigen::Matrix<float, 15, 1>& delta) {
    imu::State updated = state;

    updated.position += delta.segment<3>(imu::State::kIdxPos);

    // Right-perturbation: R_new = R_old · Exp(δφ)
    const Eigen::Vector4f dq = eigen_utils::lie::so3_exp(delta.segment<3>(imu::State::kIdxRot));
    const Eigen::Matrix3f dR = eigen_utils::geometry::quaternion_to_rotation_matrix(dq);
    updated.rotation = state.rotation * dR;

    updated.velocity += delta.segment<3>(imu::State::kIdxVel);
    updated.accel_bias += delta.segment<3>(imu::State::kIdxAccBias);
    updated.gyro_bias += delta.segment<3>(imu::State::kIdxGyrBias);

    return updated;
}

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
