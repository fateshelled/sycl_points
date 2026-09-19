#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/lio/lio_factor.hpp"
#include "sycl_points/algorithms/lio/lio_linearized_result.hpp"
#include "sycl_points/algorithms/lio/lio_registration_params.hpp"
#include "sycl_points/algorithms/lio/lio_registration_result.hpp"
#include "sycl_points/algorithms/registration/dogleg_step.hpp"
#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

// ---------------------------------------------------------------------------
// LIO (LiDAR-Inertial Odometry) registration utilities
//
// Provides the building blocks for the LIO Gauss-Newton optimisation loop:
//
//   1. add_icp_factor()   — embed the 6×6 ICP linearization into the 15-DOF
//                           normal equation, handling the perturbation-convention
//                           mismatch between ICP (body-frame δt) and LIO
//                           (world-frame δp).
//   2. add_imu_factor()   — add the 15×15 IMU prior Hessian/gradient.
//   3. solve_ldlt()       — solve H·δx = −b and optionally compute P_post = H⁻¹.
//   4. retract()          — apply a 15-D tangent-space update to a State,
//                           respecting SO(3) for the rotation component.
//   5. imu_to_lidar_jacobian() / transform_covariance_*()
//                         — convert the 15-D IMU-frame preintegration covariance
//                           to the 15-D LiDAR-frame used by the optimiser, and
//                           back again for the next IMU reset window.
//
// State-vector ordering (15-D, same as imu_factor.hpp):
//   indices  0– 2  position           δp   (world frame)
//   indices  3– 5  rotation           δφ   (so(3) tangent, right-perturbation)
//   indices  6– 8  velocity           δv   (world frame)
//   indices  9–11  accelerometer bias δb_a (body frame)
//   indices 12–14  gyroscope bias     δb_g (body frame)
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
// solve_ldlt
// ---------------------------------------------------------------------------

inline void constrain_fixed_biases(Eigen::Matrix<float, 15, 15>& H, Eigen::Matrix<float, 15, 1>& b,
                                   const BiasUpdateMask& update_bias) {
    const auto fix_block = [&](int index) {
        H.block<3, 15>(index, 0).setZero();
        H.block<15, 3>(0, index).setZero();
        H.block<3, 3>(index, index).setIdentity();
        b.segment<3>(index).setZero();
    };
    if (!update_bias.accel) fix_block(imu::State::kIdxAccBias);
    if (!update_bias.gyro) fix_block(imu::State::kIdxGyrBias);
}

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
/// @param[in]  H      Combined LIO Hessian (15×15), may include a Levenberg–
///                    Marquardt damping term λI added by the caller.
/// @param[in]  b      Combined LIO gradient (15×1).
/// @param[out] delta  15-D state update δx on success; zero on failure.
/// @param[out] P_post Posterior covariance H⁻¹ (optional, pass nullptr to skip).
/// @return true on success; false if H is numerically ill-conditioned.
inline bool solve_ldlt(const Eigen::Matrix<float, 15, 15>& H, const Eigen::Matrix<float, 15, 1>& b,
                       Eigen::Matrix<float, 15, 1>& delta, Eigen::Matrix<float, 15, 15>* P_post = nullptr,
                       const BiasUpdateMask& update_bias = {}) {
    if (!H.allFinite() || !b.allFinite()) {
        delta.setZero();
        if (P_post) P_post->setZero();
        return false;
    }
    Eigen::Matrix<float, 15, 15> constrained_H = H;
    Eigen::Matrix<float, 15, 1> constrained_b = b;
    constrain_fixed_biases(constrained_H, constrained_b, update_bias);

    Eigen::LDLT<Eigen::Matrix<float, 15, 15>> ldlt(constrained_H);
    if (ldlt.info() != Eigen::Success || ldlt.vectorD().minCoeff() <= 0.0f) {
        delta.setZero();
        if (P_post) P_post->setZero();
        return false;
    }
    delta = ldlt.solve(-constrained_b);
    if (P_post) {
        P_post->setIdentity();
        ldlt.solveInPlace(*P_post);  // P_post = H⁻¹
    }
    if (!delta.allFinite() || (P_post && !P_post->allFinite())) {
        delta.setZero();
        if (P_post) P_post->setZero();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// retract
// ---------------------------------------------------------------------------

/// @brief Apply a 15-D tangent-space update δx to a navigation state.
///
/// Implements the manifold retraction  x_new = x ⊕ δx:
///
///   position     : p_new = p + δp                         (additive)
///   rotation     : R_new = R · Exp(δφ)                   (SO(3) right-perturbation)
///   velocity     : v_new = v + δv                         (additive)
///   accel_bias   : b_a_new = b_a + δb_a                  (additive)
///   gyro_bias    : b_g_new = b_g + δb_g                  (additive)
///
/// The right-perturbation convention is consistent throughout: the same
/// convention is used by add_icp_factor() and compute_imu_hessian_gradient().
///
/// @param state  Current navigation state (linearisation point x_op).
/// @param delta  15-D tangent-space update δx from solve_ldlt().
/// @return Updated navigation state x_op ⊕ δx.
inline imu::State retract(const imu::State& state, const Eigen::Matrix<float, 15, 1>& delta) {
    imu::State updated = state;

    updated.position += delta.segment<3>(imu::State::kIdxPos);

    const Eigen::Vector4f dq = eigen_utils::lie::so3_exp(delta.segment<3>(imu::State::kIdxRot));
    updated.rotation = state.rotation * eigen_utils::geometry::quaternion_to_rotation_matrix(dq);

    updated.velocity += delta.segment<3>(imu::State::kIdxVel);
    updated.accel_bias += delta.segment<3>(imu::State::kIdxAccBias);
    updated.gyro_bias += delta.segment<3>(imu::State::kIdxGyrBias);

    return updated;
}

// ---------------------------------------------------------------------------
// imu_to_lidar_jacobian / transform_covariance_imu_to_lidar (15-DOF)
//
// The IMU preintegration propagates a 15-D covariance with world-frame
// position/velocity errors and an IMU right-rotation error
// [δp_world, δφ_imu, δv_world, δb_a, δb_g].  The LIO optimiser uses the same
// world-frame position/velocity errors and a LiDAR right-rotation error.
//
// When T_imu_to_lidar ≠ Identity the two frames differ, and the covariance
// must be transformed before it can be used as the IMU prior.  These functions
// implement that transformation via a first-order Jacobian J.
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

/// @brief Tightly-coupled ICP/IMU registration algorithm.
class LIORegistration {
public:
    using Ptr = std::shared_ptr<LIORegistration>;

    LIORegistration(const sycl_utils::DeviceQueue& queue, const registration::RegistrationFactorParams& factor_params,
                    const LIORegistrationParams& params)
        : registration_(std::make_shared<registration::Registration>(queue, factor_params)),
          factor_params_(factor_params),
          params_(params) {
        if (!std::isfinite(params_.icp_information_scale) || params_.icp_information_scale <= 0.0f) {
            throw std::invalid_argument("LIO icp_information_scale must be finite and positive");
        }
    }

    const registration::Registration::Ptr& registration_backend() const { return this->registration_; }

    LIORegistrationResult align(const PointCloudShared& source, const PointCloudShared& target,
                                const knn::KNNBase& target_knn, const imu::State& predicted_state,
                                const Eigen::Matrix<float, 15, 15>& predicted_covariance,
                                const Eigen::Matrix<float, 15, 15>& previous_posterior_covariance,
                                const BiasUpdateMask& update_bias,
                                float dt, const TransformMatrix& previous_pose) {
        Eigen::Matrix<float, 15, 15> imu_information = Eigen::Matrix<float, 15, 15>::Zero();
        Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Zero();
        Eigen::Matrix<float, 15, 1> b_imu = Eigen::Matrix<float, 15, 1>::Zero();
        const bool imu_valid = imu::compute_imu_information(predicted_covariance, imu_information);
        if (imu_valid) imu::linearize_imu_prior(predicted_state, predicted_state, imu_information, H_imu, b_imu);

        if (!imu_valid) {
            LIORegistrationResult result;
            result.status = LIORegistrationStatus::invalid_imu;
            result.state = predicted_state;
            result.posterior_covariance = previous_posterior_covariance;
            result.registration_result.T = state_to_pose(predicted_state);
            return result;
        }

        imu::State operating_state = predicted_state;
        registration::Registration::ExecutionOptions options;
        options.dt = dt;
        options.prev_pose = previous_pose;
        const TransformMatrix initial_pose = state_to_pose(predicted_state).matrix();

        registration::LinearizedResult last_icp;
        size_t actual_iterations = 0;
        Eigen::Matrix<float, 15, 15> H_undamped = Eigen::Matrix<float, 15, 15>::Zero();
        bool has_H_undamped = false;
        bool any_step_accepted = false;
        bool numeric_failure = false;
        bool finite_rejected_trial = false;
        bool abort_optimization = false;

        const auto& optimization = this->params_.optimization;
        const auto& gn_params = optimization.gn;
        const auto& lm_params = optimization.lm;
        const auto& dl_params = optimization.dogleg;
        const auto clamp_radius = [&](float radius) {
            return std::clamp(radius, dl_params.min_trust_region_radius, dl_params.max_trust_region_radius);
        };

        const auto imu_cost = [&](const imu::State& state) -> float {
            if (!imu_valid) return 0.0f;
            const Eigen::Matrix<float, 15, 1> residual = imu::compute_manifold_residual(predicted_state, state);
            return 0.5f * residual.dot(imu_information * residual);
        };

        const auto& robust_params = this->params_.robust;
        bool enable_auto_scaling = robust_params.auto_scale && this->params_.total_iterations > 0 &&
                                   this->factor_params_.robust.type != robust::RobustLossType::NONE;
        if (enable_auto_scaling &&
            (robust_params.min_scale <= 0.0f || robust_params.min_scale >= robust_params.init_scale)) {
            std::cerr << "[LIORegistration] Invalid geometry robust scale range; disabling auto scaling." << std::endl;
            enable_auto_scaling = false;
        }
        if (enable_auto_scaling && (robust_params.rotation_min_scale <= 0.0f ||
                                    robust_params.rotation_min_scale >= robust_params.rotation_init_scale)) {
            std::cerr << "[LIORegistration] Invalid rotation robust scale range; disabling auto scaling." << std::endl;
            enable_auto_scaling = false;
        }
        if (enable_auto_scaling && robust_params.auto_scaling_iter == 0) {
            std::cerr << "[LIORegistration] auto_scaling_iter must be positive; disabling auto scaling." << std::endl;
            enable_auto_scaling = false;
        }

        const size_t robust_levels =
            enable_auto_scaling ? std::min(robust_params.auto_scaling_iter, this->params_.total_iterations) : 1;
        const size_t base_iterations = this->params_.total_iterations / robust_levels;
        const size_t extra_iterations = this->params_.total_iterations % robust_levels;
        float robust_scale = enable_auto_scaling ? robust_params.init_scale : this->factor_params_.robust.default_scale;
        float rotation_robust_scale = enable_auto_scaling
                                          ? robust_params.rotation_init_scale
                                          : this->factor_params_.rotation_constraint.robust.default_scale;
        const float robust_scale_factor = robust_levels > 1
                                              ? std::pow(robust_params.min_scale / robust_params.init_scale,
                                                         1.0f / static_cast<float>(robust_levels - 1))
                                              : 1.0f;
        const float rotation_robust_scale_factor =
            robust_levels > 1 ? std::pow(robust_params.rotation_min_scale / robust_params.rotation_init_scale,
                                         1.0f / static_cast<float>(robust_levels - 1))
                              : 1.0f;

        for (size_t robust_level = 0; robust_level < robust_levels; ++robust_level) {
            options.robust_scale = robust_scale;
            options.rotation_robust_scale = rotation_robust_scale;
            if (enable_auto_scaling && this->factor_params_.verbose) {
                std::cout << "Robust scale: " << robust_scale << std::endl;
            }
            float lm_lambda = lm_params.init_lambda;
            float trust_region_radius = dl_params.initial_trust_region_radius;
            const size_t solver_iterations = base_iterations + (robust_level < extra_iterations ? 1 : 0);

            for (size_t solver_iter = 0; solver_iter < solver_iterations; ++solver_iter) {
                ++actual_iterations;

                const TransformMatrix current_pose = state_to_pose(operating_state).matrix();
                last_icp = this->registration_->compute_linearized_result(source, target, target_knn, current_pose,
                                                                          initial_pose, options);
                if (!last_icp.H.allFinite() || !last_icp.b.allFinite() || !std::isfinite(last_icp.error)) {
                    numeric_failure = true;
                    abort_optimization = true;
                    break;
                }
                if (actual_iterations > 1 && imu_valid) {
                    imu::linearize_imu_prior(predicted_state, operating_state, imu_information, H_imu, b_imu);
                }

                const float icp_weight = this->params_.icp_information_scale;

                LIOLinearizedResult icp_lio;
                add_icp_factor(icp_lio, last_icp, operating_state.rotation, icp_weight);
                apply_directional_icp_weighting(icp_lio, this->params_.directional_icp_weighting);

                LIOLinearizedResult lio = icp_lio;
                if (imu_valid) {
                    add_imu_factor(lio, H_imu, b_imu);
                    if (this->params_.directional_icp_weighting.verbose) {
                        log_imu_effective_information(H_imu, last_icp.inlier);
                    }
                } else {
                    const float regularization = this->params_.invalid_regularization_factor;
                    lio.H.block<3, 3>(imu::State::kIdxVel, imu::State::kIdxVel) +=
                        regularization * Eigen::Matrix3f::Identity();
                    lio.H.block<3, 3>(imu::State::kIdxAccBias, imu::State::kIdxAccBias) +=
                        regularization * Eigen::Matrix3f::Identity();
                    lio.H.block<3, 3>(imu::State::kIdxGyrBias, imu::State::kIdxGyrBias) +=
                        regularization * Eigen::Matrix3f::Identity();
                }
                if (!lio.H.allFinite() || !lio.b.allFinite()) {
                    numeric_failure = true;
                    abort_optimization = true;
                    break;
                }

                const auto icp_cost = [&](const imu::State& state) -> float {
                    const auto [error, inlier] = this->registration_->compute_error_frozen(
                        source, target, state_to_pose(state).matrix(), options);
                    (void)inlier;
                    return icp_weight * error;
                };

                const Eigen::Matrix<float, 15, 15> I15 = Eigen::Matrix<float, 15, 15>::Identity();
                Eigen::Matrix<float, 15, 1> delta = Eigen::Matrix<float, 15, 1>::Zero();
                bool step_accepted = false;
                bool stop = false;

                switch (optimization.optimization_method) {
                    case registration::OptimizationMethod::GAUSS_NEWTON:
                        if (solve_ldlt(lio.H + gn_params.lambda * I15, lio.b, delta, nullptr, update_bias)) {
                            step_accepted = true;
                        } else {
                            numeric_failure = true;
                            stop = true;
                        }
                        if (this->factor_params_.verbose) {
                            const float current_cost = icp_weight * last_icp.error + imu_cost(operating_state);
                            std::cout << "iter [" << actual_iterations - 1 << "] ";
                            std::cout << "error: " << current_cost << ", ";
                            std::cout << "inlier: " << last_icp.inlier << ", ";
                            std::cout << "dt: " << delta.segment<3>(imu::State::kIdxPos).norm() << ", ";
                            std::cout << "dr: " << delta.segment<3>(imu::State::kIdxRot).norm() << std::endl;
                        }
                        break;
                    case registration::OptimizationMethod::LEVENBERG_MARQUARDT: {
                        const float current_cost = icp_cost(operating_state) + imu_cost(operating_state);
                        if (!std::isfinite(current_cost)) {
                            numeric_failure = true;
                            stop = true;
                            break;
                        }
                        bool inner_solve_succeeded = false;
                        for (size_t inner = 0; inner < lm_params.max_inner_iterations; ++inner) {
                            const float trial_lambda = lm_lambda;
                            Eigen::Matrix<float, 15, 1> trial_delta = Eigen::Matrix<float, 15, 1>::Zero();
                            if (solve_ldlt(lio.H + lm_lambda * I15, lio.b, trial_delta, nullptr, update_bias)) {
                                inner_solve_succeeded = true;
                                const imu::State trial_state = retract(operating_state, trial_delta);
                                const auto [trial_icp_error, trial_inlier] = this->registration_->compute_error_frozen(
                                    source, target, state_to_pose(trial_state).matrix(), options);
                                const float trial_cost = icp_weight * trial_icp_error + imu_cost(trial_state);
                                if (this->factor_params_.verbose) {
                                    std::cout << "iter [" << actual_iterations - 1 << "] ";
                                    std::cout << "inner: " << inner << ", ";
                                    std::cout << "lambda: " << trial_lambda << ", ";
                                    std::cout << "error: " << trial_cost << ", ";
                                    std::cout << "inlier: " << trial_inlier << ", ";
                                    std::cout << "dt: " << trial_delta.segment<3>(imu::State::kIdxPos).norm() << ", ";
                                    std::cout << "dr: " << trial_delta.segment<3>(imu::State::kIdxRot).norm()
                                              << std::endl;
                                }
                                if (std::isfinite(trial_cost) && std::isfinite(current_cost) &&
                                    trial_cost <= current_cost) {
                                    delta = trial_delta;
                                    step_accepted = true;
                                    lm_lambda = std::clamp(lm_lambda / lm_params.lambda_factor, lm_params.min_lambda,
                                                           lm_params.max_lambda);
                                    break;
                                }
                                if (std::isfinite(trial_cost) && std::isfinite(current_cost)) {
                                    finite_rejected_trial = true;
                                }
                            }
                            lm_lambda = std::clamp(lm_lambda * lm_params.lambda_factor, lm_params.min_lambda,
                                                   lm_params.max_lambda);
                        }
                        if (!inner_solve_succeeded && !step_accepted) numeric_failure = true;
                        stop = !step_accepted;
                        break;
                    }
                    case registration::OptimizationMethod::POWELL_DOGLEG: {
                        const float current_cost = icp_cost(operating_state) + imu_cost(operating_state);
                        trust_region_radius = clamp_radius(trust_region_radius);
                        Eigen::Matrix<float, 15, 15> dogleg_H = lio.H;
                        Eigen::Matrix<float, 15, 1> dogleg_b = lio.b;
                        constrain_fixed_biases(dogleg_H, dogleg_b, update_bias);
                        const registration::DoglegStep<15> dogleg =
                            registration::compute_dogleg_step<15>(dogleg_H, dogleg_b, trust_region_radius);
                        Eigen::Matrix<float, 15, 1> trial_delta = dogleg.p;
                        const float predicted_reduction =
                            -(dogleg_b.dot(trial_delta) + 0.5f * trial_delta.dot(dogleg_H * trial_delta));
                        if (!trial_delta.allFinite() || !std::isfinite(predicted_reduction)) {
                            numeric_failure = true;
                            stop = true;
                            break;
                        }
                        if (predicted_reduction <= 0.0f) {
                            finite_rejected_trial = true;
                            trust_region_radius = clamp_radius(trust_region_radius * dl_params.gamma_decrease);
                            break;
                        }
                        const imu::State trial_state = retract(operating_state, trial_delta);
                        const auto [trial_icp_error, trial_inlier] = this->registration_->compute_error_frozen(
                            source, target, state_to_pose(trial_state).matrix(), options);
                        const float trial_cost = icp_weight * trial_icp_error + imu_cost(trial_state);
                        const float rho = (current_cost - trial_cost) / predicted_reduction;
                        if (!std::isfinite(trial_cost) || !std::isfinite(rho)) {
                            numeric_failure = true;
                            stop = true;
                            break;
                        }
                        if (this->factor_params_.verbose) {
                            std::cout << "iter [" << actual_iterations - 1 << "] ";
                            std::cout << "radius: " << trust_region_radius << ", ";
                            std::cout << "rho: " << rho << ", ";
                            std::cout << "error: " << trial_cost << ", ";
                            std::cout << "inlier: " << trial_inlier << ", ";
                            std::cout << "dt: " << trial_delta.segment<3>(imu::State::kIdxPos).norm() << ", ";
                            std::cout << "dr: " << trial_delta.segment<3>(imu::State::kIdxRot).norm() << std::endl;
                        }
                        if (rho < dl_params.eta1) {
                            finite_rejected_trial = true;
                            trust_region_radius = clamp_radius(trust_region_radius * dl_params.gamma_decrease);
                            break;
                        }
                        delta = trial_delta;
                        step_accepted = true;
                        if (rho > dl_params.eta2 && dogleg.step_norm >= trust_region_radius * 0.99f) {
                            trust_region_radius = clamp_radius(trust_region_radius * dl_params.gamma_increase);
                        }
                        break;
                    }
                }

                H_undamped = lio.H;
                has_H_undamped = true;
                if (step_accepted) {
                    any_step_accepted = true;
                    operating_state = retract(operating_state, delta);
                    if (is_converged(delta)) break;
                } else if (stop) {
                    break;
                }
            }

            if (abort_optimization) break;

            robust_scale *= robust_scale_factor;
            rotation_robust_scale *= rotation_robust_scale_factor;
        }

        LIORegistrationResult result;
        result.status = any_step_accepted
                            ? LIORegistrationStatus::success
                            : ((numeric_failure && !finite_rejected_trial) ? LIORegistrationStatus::numeric_failure
                                                                           : LIORegistrationStatus::no_progress);
        result.state = operating_state;
        result.posterior_covariance =
            posterior_covariance(H_undamped, has_H_undamped, previous_posterior_covariance, update_bias);
        result.registration_result.T = state_to_pose(operating_state);
        result.registration_result.converged = result.valid();
        result.registration_result.iterations = actual_iterations;
        result.registration_result.inlier = last_icp.inlier;
        result.registration_result.error = last_icp.error;
        return result;
    }

private:
    static Eigen::Isometry3f state_to_pose(const imu::State& state) {
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
        pose.linear() = state.rotation;
        pose.translation() = state.position;
        return pose;
    }

    bool is_converged(const Eigen::Matrix<float, 15, 1>& delta) const {
        return delta.segment<3>(imu::State::kIdxRot).norm() < this->params_.criteria.rotation &&
               delta.segment<3>(imu::State::kIdxPos).norm() < this->params_.criteria.translation;
    }

    static Eigen::Matrix<float, 15, 15> posterior_covariance(
        const Eigen::Matrix<float, 15, 15>& H, bool valid_H,
        const Eigen::Matrix<float, 15, 15>& previous_covariance, const BiasUpdateMask& update_bias) {
        if (!valid_H) return previous_covariance;

        Eigen::Matrix<float, 15, 1> ignored_delta;
        Eigen::Matrix<float, 15, 15> covariance;
        Eigen::Matrix<float, 15, 15> candidate = H;
        if (!solve_ldlt(candidate, Eigen::Matrix<float, 15, 1>::Zero(), ignored_delta, &covariance, update_bias)) {
            candidate.diagonal().array() += 1e-4f;
            if (!solve_ldlt(candidate, Eigen::Matrix<float, 15, 1>::Zero(), ignored_delta, &covariance, update_bias)) {
                std::cerr << "[LIORegistration] WARNING: posterior covariance solve failed; keeping previous covariance."
                          << std::endl;
                return previous_covariance;
            }
        }
        const auto restore_fixed_block = [&](int index) {
            covariance.block<3, 15>(index, 0).setZero();
            covariance.block<15, 3>(0, index).setZero();
            covariance.block<3, 3>(index, index) = previous_covariance.block<3, 3>(index, index);
        };
        if (!update_bias.accel) restore_fixed_block(imu::State::kIdxAccBias);
        if (!update_bias.gyro) restore_fixed_block(imu::State::kIdxGyrBias);
        return covariance;
    }

    registration::Registration::Ptr registration_;
    registration::RegistrationFactorParams factor_params_;
    LIORegistrationParams params_;
};

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
