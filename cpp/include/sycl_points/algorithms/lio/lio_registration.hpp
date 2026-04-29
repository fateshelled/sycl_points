#pragma once

#include <Eigen/Dense>

#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

// ---------------------------------------------------------------------------
// LIO (LiDAR-Inertial Odometry) registration utilities
//
// Provides the building blocks for the LIO Gauss-Newton optimisation loop:
//
//   1. add_icp_factor()   — embed the 6×6 ICP linearization into the 21-DOF
//                           normal equation, handling the perturbation-convention
//                           mismatch between ICP (body-frame δt) and LIO
//                           (world-frame δp).
//   2. add_imu_factor()   — add the 21×21 IMU prior Hessian/gradient.
//   3. solve_ldlt()       — solve H·δx = −b and optionally compute P_post = H⁻¹.
//   4. retract()          — apply a 21-D tangent-space update to a State,
//                           respecting SO(3) for the two rotation components.
//   5. imu_to_lidar_jacobian() / transform_covariance_*()
//                         — convert the 15-D IMU-frame preintegration covariance
//                           to the 15-D LiDAR-frame used by the optimiser, and
//                           back again for the next IMU reset window.
//
// State-vector ordering (21-D, same as imu_factor.hpp):
//   indices  0– 2  position           δp   (world frame)
//   indices  3– 5  rotation           δφ   (so(3) tangent, right-perturbation)
//   indices  6– 8  velocity           δv   (world frame)
//   indices  9–11  accelerometer bias δb_a (body frame)
//   indices 12–14  gyroscope bias     δb_g (body frame)
//   indices 15–17  extrinsic rotation δφ_ex (so(3) tangent, right-perturbation)
//   indices 18–20  extrinsic transl.  δt_ex
//
// ICP 6-DOF delta ordering (registration.hpp convention):
//   indices  0– 2  rotation    δω  (body frame)
//   indices  3– 5  translation δt  (body frame)
//
// Index mapping when embedding ICP into the LIO system:
//   ICP δω (0–2) → LIO δφ   (3–5)   body frame matches directly
//   ICP δt (3–5) → LIO δp   (0–2)   requires rotation to world frame
// ---------------------------------------------------------------------------

namespace sycl_points {
namespace algorithms {
namespace lio {

// ---------------------------------------------------------------------------
// LIOLinearizedResult
// ---------------------------------------------------------------------------

/// @brief Accumulated Gauss-Newton normal equation for one LIO iteration.
///
/// Holds the combined 21×21 Hessian H and 21×1 gradient b that result from
/// accumulating one or more factors (ICP geometry, IMU prior).  After all
/// factors have been added, pass H and b to solve_ldlt() to obtain the state
/// update δx and the posterior covariance P_post = H⁻¹.
///
/// The normal equation is formulated as:
///
///   H · δx = −b
///
/// where:
///   H = Σ Jᵢᵀ · Ωᵢ · Jᵢ   (sum of information-weighted outer products)
///   b = Σ Jᵢᵀ · Ωᵢ · rᵢ   (sum of information-weighted residuals)
///
/// Diagnostic fields (error_icp, error_imu, inlier) are for logging only.
struct LIOLinearizedResult {
    Eigen::Matrix<float, 21, 21> H = Eigen::Matrix<float, 21, 21>::Zero();  ///< Combined Hessian
    Eigen::Matrix<float, 21, 1> b = Eigen::Matrix<float, 21, 1>::Zero();    ///< Combined gradient
    float error_icp = 0.0f;                                                 ///< Accumulated ICP cost (for diagnostics)
    float error_imu = 0.0f;                                                 ///< IMU prior cost (for diagnostics)
    uint32_t inlier = 0;                                                    ///< Number of valid ICP correspondences

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ---------------------------------------------------------------------------
// add_icp_factor
// ---------------------------------------------------------------------------

/// @brief Embed an ICP 6×6 Hessian/gradient into the LIO 21×21 normal equation.
///
/// ## Perturbation-convention mismatch and how it is resolved
///
/// The ICP linearization uses SE(3) right-perturbation:
///
///   T_new = T · Exp([δω; δt])
///
/// where δω and δt are both expressed in the *LiDAR body frame*:
///   - δω : rotation increment (body frame)
///   - δt : translation increment (body frame) → p_new = p + R · δt
///
/// The LIO 21-D error-state uses:
///   - δφ : rotation increment (body frame)  — same as ICP δω ✓
///   - δp : position increment (world frame) — p_new = p + δp ✗
///
/// For rotation the conventions agree directly.
/// For translation they differ by R_world_lidar:
///
///   δp_world = R_world_lidar · δt_body
///
/// Consequence: the ICP translation gradient and Hessian blocks must be
/// rotated into the world frame before accumulation:
///
///   b_lio[p]     = R · b_icp[t]
///   H_lio[p,p]   = R · H_icp[t,t] · Rᵀ
///   H_lio[p,φ]   = R · H_icp[t,ω]         (φ stays in body frame)
///   H_lio[φ,p]   = H_icp[ω,t] · Rᵀ
///
/// The extrinsic blocks (rows/columns 15–20) of H and b are unaffected because
/// the ICP measurement depends only on the LiDAR-frame pose, not the extrinsic.
///
/// @param result        LIO normal equation to accumulate into.
/// @param icp           ICP linearization from registration::Registration.
/// @param R_world_lidar Current body-to-world rotation (x_op.rotation).
inline void add_icp_factor(LIOLinearizedResult& result, const registration::LinearizedResult& icp,
                           const Eigen::Matrix3f& R_world_lidar) {
    // Rotation block: δω (ICP body frame) == δφ (LIO body frame) — embed directly
    result.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot) += icp.H.block<3, 3>(0, 0);
    result.b.segment<3>(imu::State::kIdxRot) += icp.b.segment<3>(0);

    // Translation block: rotate ICP body-frame δt into LIO world-frame δp
    const Eigen::Matrix3f& R = R_world_lidar;
    result.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos) += R * icp.H.block<3, 3>(3, 3) * R.transpose();
    result.b.segment<3>(imu::State::kIdxPos) += R * icp.b.segment<3>(3);

    // Cross terms: φ remains body-frame, p becomes world-frame
    result.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot) += R * icp.H.block<3, 3>(3, 0);
    result.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxPos) += icp.H.block<3, 3>(0, 3) * R.transpose();

    result.error_icp += icp.error;
    result.inlier += icp.inlier;
}

// ---------------------------------------------------------------------------
// add_imu_factor
// ---------------------------------------------------------------------------

/// @brief Add the IMU prior 21×21 Hessian/gradient into the LIO normal equation.
///
/// The IMU factor is already expressed in the LIO 21-D state ordering
/// (computed by compute_imu_hessian_gradient()), so no index permutation is
/// needed.  When extrinsic estimation is enabled the caller is responsible for
/// adding the cross-terms between the navigation state and the extrinsic blocks
/// before calling this function.
///
/// @param result  LIO normal equation to accumulate into.
/// @param H_imu   21×21 IMU information matrix from compute_imu_hessian_gradient().
/// @param b_imu   21×1  IMU gradient vector   from compute_imu_hessian_gradient().
/// @param error   Optional scalar IMU prior cost for logging (default 0).
inline void add_imu_factor(LIOLinearizedResult& result, const Eigen::Matrix<float, 21, 21>& H_imu,
                           const Eigen::Matrix<float, 21, 1>& b_imu, float error = 0.0f) {
    result.H += H_imu;
    result.b += b_imu;
    result.error_imu = error;
}

// ---------------------------------------------------------------------------
// solve_ldlt
// ---------------------------------------------------------------------------

/// @brief Solve the combined LIO normal equation H·δx = −b with LDLT.
///
/// Uses Bunch-Kaufman (LDLT) factorisation for numerical stability with
/// single-precision floats.  The posterior covariance P_post = H⁻¹ is
/// computed at negligible extra cost by solving H · P_post = I in-place.
///
/// P_post is used as the initial covariance for the next frame's IMU
/// preintegration window (passed to IMUPreintegration::reset() after
/// converting back to IMU frame via transform_covariance_lidar_to_imu()).
///
/// @param[in]  H      Combined LIO Hessian (21×21), may include a Levenberg–
///                    Marquardt damping term λI added by the caller.
/// @param[in]  b      Combined LIO gradient (21×1).
/// @param[out] delta  21-D state update δx on success; zero on failure.
/// @param[out] P_post Posterior covariance H⁻¹ (optional, pass nullptr to skip).
/// @return true on success; false if H is numerically ill-conditioned.
inline bool solve_ldlt(const Eigen::Matrix<float, 21, 21>& H, const Eigen::Matrix<float, 21, 1>& b,
                       Eigen::Matrix<float, 21, 1>& delta, Eigen::Matrix<float, 21, 21>* P_post = nullptr) {
    Eigen::LDLT<Eigen::Matrix<float, 21, 21>> ldlt(H);
    if (ldlt.info() != Eigen::Success || ldlt.vectorD().minCoeff() <= 0.0f) {
        delta.setZero();
        if (P_post) P_post->setZero();
        return false;
    }
    delta = ldlt.solve(-b);
    if (P_post) {
        P_post->setIdentity();
        ldlt.solveInPlace(*P_post);  // P_post = H⁻¹
    }
    return true;
}

// ---------------------------------------------------------------------------
// retract
// ---------------------------------------------------------------------------

/// @brief Apply a 21-D tangent-space update δx to a navigation state.
///
/// Implements the manifold retraction  x_new = x ⊕ δx:
///
///   position     : p_new = p + δp                         (additive)
///   rotation     : R_new = R · Exp(δφ)                   (SO(3) right-perturbation)
///   velocity     : v_new = v + δv                         (additive)
///   accel_bias   : b_a_new = b_a + δb_a                  (additive)
///   gyro_bias    : b_g_new = b_g + δb_g                  (additive)
///   offset_R_L_I : R_ex_new = R_ex · Exp(δφ_ex)          (SO(3) right-perturbation)
///   offset_T_L_I : t_ex_new = t_ex + δt_ex               (additive)
///
/// The right-perturbation convention is consistent throughout: the same
/// convention is used by add_icp_factor() and compute_imu_hessian_gradient().
///
/// @param state  Current navigation state (linearisation point x_op).
/// @param delta  21-D tangent-space update δx from solve_ldlt().
/// @return Updated navigation state x_op ⊕ δx.
inline imu::State retract(const imu::State& state, const Eigen::Matrix<float, 21, 1>& delta) {
    imu::State updated = state;

    updated.position += delta.segment<3>(imu::State::kIdxPos);

    const Eigen::Vector4f dq = eigen_utils::lie::so3_exp(delta.segment<3>(imu::State::kIdxRot));
    updated.rotation = state.rotation * eigen_utils::geometry::quaternion_to_rotation_matrix(dq);

    updated.velocity += delta.segment<3>(imu::State::kIdxVel);
    updated.accel_bias += delta.segment<3>(imu::State::kIdxAccBias);
    updated.gyro_bias += delta.segment<3>(imu::State::kIdxGyrBias);

    // Extrinsic rotation: right-perturbation on SO(3)
    const Eigen::Vector4f dq_ex = eigen_utils::lie::so3_exp(delta.segment<3>(imu::State::kIdxExRot));
    updated.offset_R_L_I = state.offset_R_L_I * eigen_utils::geometry::quaternion_to_rotation_matrix(dq_ex);

    updated.offset_T_L_I += delta.segment<3>(imu::State::kIdxExTrans);

    return updated;
}

// ---------------------------------------------------------------------------
// imu_to_lidar_jacobian / transform_covariance_imu_to_lidar (15-DOF)
//
// The IMU preintegration propagates a 15-D covariance in the IMU body
// error-state [δp_imu, δφ_imu, δv, δb_a, δb_g].  The LIO optimiser maintains
// the state in the LiDAR body frame [δp_lidar, δφ_lidar, δv, δb_a, δb_g].
//
// When T_imu_to_lidar ≠ Identity the two frames differ, and the covariance
// must be transformed before it can be used as the IMU prior.  These functions
// implement that transformation via a first-order Jacobian J.
//
// The extrinsic blocks (indices 15–20) are not part of IMU preintegration and
// are therefore handled separately (see lio_registration() in
// lidar_inertial_odometry.hpp).
// ---------------------------------------------------------------------------

/// @brief Jacobian that maps the 15-D IMU error-state to the 15-D LiDAR error-state.
///
/// First-order relationships (right-perturbation convention):
///
///   δφ_lidar = R_lidar_imu · δφ_imu
///   δp_lidar = δp_imu − R_world_imu · skew(t_lidar_in_imu) · δφ_imu
///   δv, δb_a, δb_g  unchanged
///
/// where:
///   R_lidar_imu    = T_imu_to_lidar.rotation()
///   t_lidar_in_imu = T_imu_to_lidar.inverse().translation()
///                  = position of the LiDAR origin in the IMU body frame
///   R_world_imu    = R_world_lidar · R_lidar_imu
///
/// The lever-arm term −R_world_imu · skew(t_lidar_in_imu) is zero when the
/// IMU and LiDAR are co-located (T_imu_to_lidar = Identity).
///
/// @param T_imu_to_lidar  Extrinsic: pose of IMU body expressed in LiDAR body frame.
/// @param R_world_lidar   Current LiDAR-to-world rotation from the LIO state.
/// @return 15×15 Jacobian  J  such that  δx_lidar = J · δx_imu.
inline Eigen::Matrix<float, 15, 15> imu_to_lidar_jacobian(const Eigen::Isometry3f& T_imu_to_lidar,
                                                          const Eigen::Matrix3f& R_world_lidar) {
    Eigen::Matrix<float, 15, 15> J = Eigen::Matrix<float, 15, 15>::Identity();

    const Eigen::Matrix3f R_lidar_imu = T_imu_to_lidar.rotation();
    const Eigen::Vector3f t_lidar_in_imu = T_imu_to_lidar.inverse().translation();
    const Eigen::Matrix3f R_world_imu = R_world_lidar * R_lidar_imu;

    // J[φ, φ]: rotation error transforms by R_lidar_imu
    J.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot) = R_lidar_imu;

    // J[p, φ]: lever-arm coupling — zero when IMU and LiDAR are co-located
    J.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot) = -R_world_imu * eigen_utils::lie::skew(t_lidar_in_imu);

    return J;
}

/// @brief Transform IMU preintegration covariance (15×15) from IMU to LiDAR error-state.
///
///   P_lidar = J · P_imu · Jᵀ
///
/// Call this once per frame, before compute_imu_hessian_gradient(), to convert
/// the covariance produced by IMUPreintegration::get_raw().covariance into the
/// LiDAR-frame prior that the LIO optimiser expects.
///
/// The Jacobian J is evaluated at x_pred.rotation (the IMU-propagated pose),
/// which is constant across Gauss-Newton iterations within a single frame.
///
/// @param P_imu           15×15 covariance from IMUPreintegration::get_raw().covariance.
/// @param T_imu_to_lidar  Extrinsic (fixed or current state estimate).
/// @param R_world_lidar   LiDAR-to-world rotation of the predicted state (x_pred.rotation).
/// @return 15×15 covariance expressed in the LiDAR error-state frame.
inline Eigen::Matrix<float, 15, 15> transform_covariance_imu_to_lidar(const Eigen::Matrix<float, 15, 15>& P_imu,
                                                                      const Eigen::Isometry3f& T_imu_to_lidar,
                                                                      const Eigen::Matrix3f& R_world_lidar) {
    const Eigen::Matrix<float, 15, 15> J = imu_to_lidar_jacobian(T_imu_to_lidar, R_world_lidar);
    return J * P_imu * J.transpose();
}

/// @brief Transform LiDAR error-state covariance (15×15) back to IMU error-state.
///
///   P_imu = J⁻¹ · P_lidar · J⁻ᵀ
///
/// Call this after solve_ldlt() to convert P_post (LiDAR frame) back to the
/// IMU body frame before passing it as initial_covariance to
/// IMUPreintegration::reset() for the next frame window.
///
/// J has the block structure  [I  A  0; 0  C  0; 0  0  I₉]
/// (only the position-rotation and rotation-rotation blocks are non-identity),
/// so its analytical inverse is  [I  -A·C⁻¹  0; 0  C⁻¹  0; 0  0  I₉]:
///
///   J⁻¹[φ, φ] = R_imu_lidar  (= R_lidar_imu^T)
///   J⁻¹[p, φ] = R_world_imu · skew(t_lidar_in_imu) · R_imu_lidar
///
/// @param P_lidar         15×15 posterior covariance in the LiDAR error-state frame.
/// @param T_imu_to_lidar  Extrinsic (fixed or current state estimate).
/// @param R_world_lidar   Current LiDAR-to-world rotation from the updated state (x_.rotation).
/// @return 15×15 covariance expressed in the IMU error-state frame.
inline Eigen::Matrix<float, 15, 15> transform_covariance_lidar_to_imu(const Eigen::Matrix<float, 15, 15>& P_lidar,
                                                                      const Eigen::Isometry3f& T_imu_to_lidar,
                                                                      const Eigen::Matrix3f& R_world_lidar) {
    Eigen::Matrix<float, 15, 15> Jinv = Eigen::Matrix<float, 15, 15>::Identity();

    const Eigen::Matrix3f R_lidar_imu = T_imu_to_lidar.rotation();
    const Eigen::Matrix3f R_imu_lidar = R_lidar_imu.transpose();
    const Eigen::Vector3f t_lidar_in_imu = T_imu_to_lidar.inverse().translation();
    const Eigen::Matrix3f R_world_imu = R_world_lidar * R_lidar_imu;

    Jinv.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot) = R_imu_lidar;
    Jinv.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot) =
        R_world_imu * eigen_utils::lie::skew(t_lidar_in_imu) * R_imu_lidar;

    return Jinv * P_lidar * Jinv.transpose();
}

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
