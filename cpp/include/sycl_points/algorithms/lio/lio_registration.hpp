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
/// Perturbation-convention mismatch and how it is resolved here:
///
///   GICP uses SE(3) right-perturbation  T_new = T * Exp([δω; δt])
///     δω  — rotation increment in the LiDAR body frame
///     δt  — translation increment in the LiDAR body frame
///             (p_new = p + R * δt,  NOT  p + δt)
///
///   LIO state uses:
///     δφ  — rotation increment in the LiDAR body frame  (same as δω) ✓
///     δp  — position increment in the WORLD frame
///             (p_new = p + δp)
///
///   For rotation the conventions match directly.
///   For translation they differ by R_world_lidar:
///     δp_world = R * δt_body
///
///   Therefore the ICP translation gradient / Hessian must be rotated
///   into the world frame before accumulation:
///     b_lio[p]   = R * b_icp[t]
///     H_lio[p,p] = R * H_icp[t,t] * Rᵀ
///     H_lio[p,φ] = R * H_icp[t,ω]   (φ stays in body frame)
///     H_lio[φ,p] = H_icp[ω,t] * Rᵀ
///
/// @param result          LIO system to accumulate into.
/// @param icp             ICP linearized result from registration::Registration.
/// @param R_world_lidar   Current LiDAR-to-world rotation (x_op.rotation).
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
/// @param[in]  H      Combined LIO Hessian.
/// @param[in]  b      Combined LIO gradient.
/// @param[out] delta  15-D state update δx on success; zero on failure.
/// @param[out] P_post Posterior covariance H⁻¹ (optional, pass nullptr to skip).
/// @return true on success; false if H is ill-conditioned.
inline bool solve_ldlt(const Eigen::Matrix<float, 15, 15>& H, const Eigen::Vector<float, 15>& b,
                       Eigen::Matrix<float, 15, 1>& delta, Eigen::Matrix<float, 15, 15>* P_post = nullptr) {
    Eigen::LDLT<Eigen::Matrix<float, 15, 15>> ldlt(H);
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

// ---------------------------------------------------------------------------
// imu_to_lidar_jacobian / transform_covariance_imu_to_lidar
// ---------------------------------------------------------------------------

/// @brief Jacobian that maps the 15-D IMU error-state to the 15-D LiDAR error-state.
///
/// The IMU preintegration covariance P_pred is expressed in the IMU body error-state
/// [δp_imu, δφ_imu, δv_imu, δb_a, δb_g].  When the LIO optimisation maintains the
/// state in the LiDAR body convention [δp_lidar, δφ_lidar, …] the two frames are
/// related by the rigid extrinsic T_imu_to_lidar.
///
/// First-order relationships (right-perturbation convention):
///   δφ_lidar = R_lidar_imu · δφ_imu          (rotation of the tangent vector)
///   δp_lidar = δp_imu − R_world_lidar · skew(t_lidar_in_imu) · δφ_imu   (lever-arm)
///   δv_lidar = δv_imu,  δb_a, δb_g unchanged
///
/// where:
///   R_lidar_imu    = T_imu_to_lidar.rotation()
///   t_lidar_in_imu = T_imu_to_lidar.inverse().translation()
///
/// @param T_imu_to_lidar  Extrinsic: pose of IMU body expressed in LiDAR body frame.
/// @param R_world_lidar   Current LiDAR-to-world rotation from the LIO state.
/// @return 15×15 Jacobian  J  such that  δx_lidar = J · δx_imu.
inline Eigen::Matrix<float, 15, 15> imu_to_lidar_jacobian(const Eigen::Isometry3f& T_imu_to_lidar,
                                                          const Eigen::Matrix3f& R_world_lidar) {
    Eigen::Matrix<float, 15, 15> J = Eigen::Matrix<float, 15, 15>::Identity();

    const Eigen::Matrix3f R_lidar_imu = T_imu_to_lidar.rotation();
    const Eigen::Vector3f t_lidar_in_imu = T_imu_to_lidar.inverse().translation();
    // R_world_imu = R_world_lidar * R_lidar_imu  (existing convention in the pipeline)
    const Eigen::Matrix3f R_world_imu = R_world_lidar * R_lidar_imu;

    // J[φ, φ]: rotation error transforms by R_lidar_imu
    J.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot) = R_lidar_imu;

    // J[p, φ]: lever-arm coupling (zero when IMU and LiDAR are co-located)
    J.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot) = -R_world_imu * eigen_utils::lie::skew(t_lidar_in_imu);

    return J;
}

/// @brief Transform IMU preintegration covariance from IMU error-state to LiDAR error-state.
///
///   P_lidar = J · P_imu · Jᵀ
///
/// Pass the result as P_pred to compute_imu_hessian_gradient() together with
/// LiDAR-frame states (x_pred_lidar, x_op_lidar) so that the information matrix
/// and gradient are consistent with the LiDAR-frame 15-DOF LIO optimisation.
///
/// @param P_imu           15×15 covariance from IMUPreintegration::get_raw().covariance.
/// @param T_imu_to_lidar  Extrinsic.
/// @param R_world_lidar   Current LiDAR rotation from the LIO state.
/// @return 15×15 covariance expressed in the LiDAR error-state frame.
inline Eigen::Matrix<float, 15, 15> transform_covariance_imu_to_lidar(const Eigen::Matrix<float, 15, 15>& P_imu,
                                                                      const Eigen::Isometry3f& T_imu_to_lidar,
                                                                      const Eigen::Matrix3f& R_world_lidar) {
    const Eigen::Matrix<float, 15, 15> J = imu_to_lidar_jacobian(T_imu_to_lidar, R_world_lidar);
    return J * P_imu * J.transpose();
}

/// @brief Transform LiDAR error-state covariance back to IMU error-state (inverse of imu_to_lidar).
///
///   P_imu = J⁻¹ · P_lidar · J⁻ᵀ
///
/// J has the block structure  [I  A  0; 0  C  0; 0  0  I₉]  (position/rotation rows only),
/// so its analytical inverse is  [I  -A·C⁻¹  0; 0  C⁻¹  0; 0  0  I₉].
///
/// Use this to convert P_post (LiDAR frame, output of solve_ldlt) back to IMU frame
/// before passing it as initial_covariance to IMUPreintegration::reset().
///
/// @param P_lidar         15×15 covariance in the LiDAR error-state frame.
/// @param T_imu_to_lidar  Extrinsic.
/// @param R_world_lidar   Current LiDAR rotation from the LIO state.
/// @return 15×15 covariance expressed in the IMU error-state frame.
inline Eigen::Matrix<float, 15, 15> transform_covariance_lidar_to_imu(const Eigen::Matrix<float, 15, 15>& P_lidar,
                                                                      const Eigen::Isometry3f& T_imu_to_lidar,
                                                                      const Eigen::Matrix3f& R_world_lidar) {
    // J⁻¹ shares the same sparsity pattern as J with:
    //   J⁻¹[φ, φ] = R_imu_lidar  (= R_lidar_imu^T)
    //   J⁻¹[p, φ] = R_world_imu · skew(t_lidar_in_imu) · R_imu_lidar
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
