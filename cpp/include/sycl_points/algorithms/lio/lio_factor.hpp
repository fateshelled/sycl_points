#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/lio/lio_linearized_result.hpp"
#include "sycl_points/algorithms/lio/lio_registration_params.hpp"
#include "sycl_points/algorithms/registration/coupled_degeneracy.hpp"
#include "sycl_points/algorithms/registration/linearized_result.hpp"

namespace sycl_points {
namespace algorithms {
namespace lio {

/// @brief Embed an ICP 6x6 Hessian/gradient into the LIO 15x15 normal equation.
///
/// ICP uses body-frame right perturbations [rotation, translation], while the
/// LIO error state uses world-frame position followed by body-frame rotation.
inline void add_icp_factor(LIOLinearizedResult& result, const registration::LinearizedResult& icp,
                           const Eigen::Matrix3f& R_world_lidar, float weight = 1.0f) {
    result.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot) += weight * icp.H.block<3, 3>(0, 0);
    result.b.segment<3>(imu::State::kIdxRot) += weight * icp.b.segment<3>(0);

    const Eigen::Matrix3f& R = R_world_lidar;
    result.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos) +=
        weight * (R * icp.H.block<3, 3>(3, 3) * R.transpose());
    result.b.segment<3>(imu::State::kIdxPos) += weight * (R * icp.b.segment<3>(3));

    result.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot) += weight * (R * icp.H.block<3, 3>(3, 0));
    result.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxPos) +=
        weight * (icp.H.block<3, 3>(0, 3) * R.transpose());

    result.error_icp += weight * icp.error;
    result.inlier += icp.inlier;
}

/// @brief Add the IMU prior, already expressed in the LIO 15-DOF ordering.
inline void add_imu_factor(LIOLinearizedResult& result, const Eigen::Matrix<float, 15, 15>& H_imu,
                           const Eigen::Matrix<float, 15, 1>& b_imu, float error = 0.0f) {
    result.H += H_imu;
    result.b += b_imu;
    result.error_imu = error;
}

/// @brief A constant-velocity prior linearized at one LIO operating point.
///
/// Keeping the information matrix and anchor velocity lets the LM and dogleg
/// optimisers evaluate the same factor cost that was accumulated into their
/// normal equation.
struct ConstantVelocityPrior {
    bool active = false;
    Eigen::Vector3f anchor_velocity = Eigen::Vector3f::Zero();
    Eigen::Matrix3f information = Eigen::Matrix3f::Zero();

    /// @brief Scalar prior cost 0.5 * (v - v_anchor)^T Omega (v - v_anchor).
    float cost(const imu::State& state) const {
        if (!active) return 0.0f;
        const Eigen::Vector3f residual = state.velocity - anchor_velocity;
        return 0.5f * residual.dot(information * residual);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Anchor the world-frame velocity to a constant-velocity reference in
///        directions where the LiDAR position information is degenerate.
///
/// The anchor is the velocity of the previous accepted state, so the prior encodes
/// v_k = v_{k-1} exactly (constant velocity) and contributes no gradient while the
/// platform actually moves at constant speed.  Directions are classified from the
/// ICP position Hessian alone (self-referenced, so immune to the P_post -> P_pred ->
/// H_imu feedback).  A direction is degenerate only when it is weak both relatively
/// (below min_eigenvalue_ratio * lambda_max) and absolutely (below the per-inlier
/// floor); the absolute gate keeps a well-conditioned frame from being anchored
/// merely because its weakest axis is the smallest of three strong ones.
///
/// @param lio               LIO normal equation; the velocity block is updated in place.
/// @param icp_position_H    ICP translation Hessian already rotated into the world frame.
/// @param operating_velocity Current operating-point velocity (world frame), used for
///                           the gradient so the linearization matches the prior cost.
/// @param anchor_velocity   Velocity of the previous accepted state (world frame).
/// @param inlier            ICP inlier count, used by the absolute degeneracy gate.
/// @param params            Constant-velocity prior parameters.
/// @return The active factor model, or an inactive model when gated out.
inline ConstantVelocityPrior add_constant_velocity_prior(LIOLinearizedResult& lio,
                                                         const Eigen::Matrix3f& icp_position_H,
                                                         const Eigen::Vector3f& operating_velocity,
                                                         const Eigen::Vector3f& anchor_velocity, uint32_t inlier,
                                                         const ConstantVelocityPriorParams& params) {
    ConstantVelocityPrior prior;
    if (!params.enable || inlier == 0) return prior;
    if (!icp_position_H.allFinite() || !operating_velocity.allFinite() || !anchor_velocity.allFinite()) return prior;

    const Eigen::Matrix3f H_sym = 0.5f * (icp_position_H + icp_position_H.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(H_sym);
    if (solver.info() != Eigen::Success) return prior;

    const float lambda_max = solver.eigenvalues().maxCoeff();
    if (!(lambda_max > 0.0f) || !std::isfinite(lambda_max)) return prior;

    const float inlier_f = static_cast<float>(inlier);
    const float ratio = std::clamp(params.min_eigenvalue_ratio, 0.0f, 1.0f);
    const float absolute_threshold = std::max(0.0f, params.min_information_per_inlier) * inlier_f;
    const auto sigma_to_info = [](float sigma) {
        return sigma > 0.0f ? 1.0f / (sigma * sigma) : 0.0f;
    };
    const float weak_info = sigma_to_info(params.degenerate_velocity_sigma);
    const float observable_info = sigma_to_info(params.observable_velocity_sigma);

    Eigen::Matrix3f information = Eigen::Matrix3f::Zero();
    for (int i = 0; i < 3; ++i) {
        const float lambda = solver.eigenvalues()(i);
        const Eigen::Vector3f q = solver.eigenvectors().col(i);
        const bool relatively_weak = (lambda / lambda_max) < ratio;
        const bool absolutely_weak = absolute_threshold > 0.0f && lambda < absolute_threshold;
        const bool degenerate = relatively_weak && (absolute_threshold <= 0.0f || absolutely_weak);
        const float info = degenerate ? weak_info : observable_info;
        information.noalias() += info * (q * q.transpose());

        if (params.verbose && degenerate) {
            std::cout << "[ConstantVelocityPrior] degenerate eigenvector: " << q.transpose()
                      << ", icp_info/inlier: " << (lambda / inlier_f) << ", velocity_info: " << info << std::endl;
        }
    }

    if (information.isZero()) return prior;
    information = 0.5f * (information + information.transpose());

    // Gradient uses the manifold residual convention r = v_op - v_anchor, matching
    // linearize_imu_prior, so solve_ldlt()'s delta = -H^-1 b drives v_op to v_anchor.
    lio.H.block<3, 3>(imu::State::kIdxVel, imu::State::kIdxVel) += information;
    lio.b.segment<3>(imu::State::kIdxVel) += information * (operating_velocity - anchor_velocity);
    lio.H = 0.5f * (lio.H + lio.H.transpose());

    prior.active = true;
    prior.anchor_velocity = anchor_velocity;
    prior.information = information;
    return prior;
}

/// @brief Attenuate ICP pose information in directions that are weak relative to the IMU prior.
///
/// @param icp_factor  LIO factor containing only the embedded ICP contribution.
/// @param H_imu       IMU information matrix (15x15) in the LIO state ordering. Pass a
///                    zero matrix when no valid IMU prior exists; the configured floor
///                    still provides a usable comparison baseline.
/// @param params      Directional weighting parameters.
inline void apply_directional_icp_weighting(LIOLinearizedResult& icp_factor,
                                            const Eigen::Matrix<float, 15, 15>& H_imu,
                                            const DirectionalIcpWeightingParams& params) {
    if (!params.enable || icp_factor.inlier == 0) return;

    constexpr int kPoseDof = 6;
    constexpr int kBlockDof = 3;
    const float inlier_f = static_cast<float>(icp_factor.inlier);
    Eigen::Matrix<float, kPoseDof, kPoseDof> H_pose = Eigen::Matrix<float, kPoseDof, kPoseDof>::Zero();
    Eigen::Matrix<float, kPoseDof, 1> b_pose = Eigen::Matrix<float, kPoseDof, 1>::Zero();

    H_pose.block<3, 3>(0, 0) = icp_factor.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos);
    H_pose.block<3, 3>(0, 3) = icp_factor.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot);
    H_pose.block<3, 3>(3, 0) = icp_factor.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxPos);
    H_pose.block<3, 3>(3, 3) = icp_factor.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot);
    H_pose = 0.5f * (H_pose + H_pose.transpose());

    b_pose.segment<3>(0) = icp_factor.b.segment<3>(imu::State::kIdxPos);
    b_pose.segment<3>(3) = icp_factor.b.segment<3>(imu::State::kIdxRot);

    const auto compute_block_filter = [&](const Eigen::Vector3f& eigenvalues, const Eigen::Matrix3f& eigenvectors,
                                          const Eigen::Matrix3f& H_imu_block, const Eigen::Vector3f& b_block,
                                          float min_information_ratio, float imu_information_floor_per_inlier,
                                          float weak_direction_scale, const char* label) -> Eigen::Matrix3f {
        if (params.verbose) {
            std::cout << "[DirectionalIcpWeighting] " << label
                      << " eigenvalues/inlier: " << (eigenvalues / inlier_f).transpose() << std::endl;
            std::cout << "[DirectionalIcpWeighting] " << label
                      << " b/inlier: " << (b_block / inlier_f).transpose() << std::endl;
        }

        const float ratio = std::max(0.0f, min_information_ratio);
        const float floor_info = std::max(0.0f, imu_information_floor_per_inlier) * inlier_f;
        const float weak_scale = std::clamp(weak_direction_scale, 0.0f, 1.0f);
        Eigen::Matrix3f filter = Eigen::Matrix3f::Zero();
        for (int i = 0; i < kBlockDof; ++i) {
            const float lambda = std::max(0.0f, eigenvalues(i));
            const Eigen::Vector3f q = eigenvectors.col(i);

            // Baseline is the IMU prior's information along this same eigen-direction,
            // floored so the P_post -> P_pred -> H_imu degeneracy feedback cannot make
            // every direction look weak.
            const float imu_info = std::max(floor_info, q.dot(H_imu_block * q));
            const float weak_threshold = ratio * imu_info;

            float scale = 1.0f;
            if (lambda <= 0.0f || !std::isfinite(lambda)) {
                scale = 0.0f;
            } else if (weak_threshold > 0.0f && lambda < weak_threshold) {
                if (params.type == DirectionalIcpWeightingType::tsvd) {
                    scale = 0.0f;
                } else {
                    const float information_ratio = std::clamp(lambda / weak_threshold, 0.0f, 1.0f);
                    scale = std::max(weak_scale, information_ratio);
                }
                if (params.verbose) {
                    std::cout << "[DirectionalIcpWeighting] " << label << " weak eigenvector: " << q.transpose()
                              << ", scale: " << scale << ", icp_info/inlier: " << (lambda / inlier_f)
                              << ", imu_info/inlier: " << (imu_info / inlier_f) << std::endl;
                }
            }

            filter.noalias() += std::sqrt(std::clamp(scale, 0.0f, 1.0f)) * (q * q.transpose());
        }
        return filter;
    };

    const auto block_eigendata = [](const Eigen::Matrix3f& H_block, Eigen::Vector3f& eigenvalues,
                                    Eigen::Matrix3f& eigenvectors) -> bool {
        const Eigen::Matrix3f H_sym = 0.5f * (H_block + H_block.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(H_sym);
        if (solver.info() != Eigen::Success) return false;
        eigenvalues = solver.eigenvalues();
        eigenvectors = solver.eigenvectors();
        return true;
    };

    // Build the 6x6 attenuation filter. The block-diagonal path keeps the
    // existing independent translation/rotation blocks. The coupled path
    // diagonalises the unit-balanced full 6x6 pose Hessian, so each eigenvector
    // is a coupled translation/rotation motion. Each balanced eigen-direction is
    // compared against the IMU prior in the same balanced units and, when weak,
    // attenuated along its original-space direction. This removes only the
    // coupled weak subspace instead of both block axes, so an observable coupled
    // motion is not over-attenuated.
    Eigen::Matrix<float, kPoseDof, kPoseDof> filter = Eigen::Matrix<float, kPoseDof, kPoseDof>::Zero();

    if (params.use_coupled_degeneracy) {
        const registration::CoupledEigenAnalysis analysis = registration::compute_coupled_eigen_analysis(
            H_pose, registration::PoseHessianOrder::translation_first,
            static_cast<double>(params.coupled_representative_length), static_cast<double>(inlier_f));
        if (!analysis.valid) return;

        Eigen::Matrix<double, kPoseDof, kPoseDof> H_imu_pose = Eigen::Matrix<double, kPoseDof, kPoseDof>::Zero();
        H_imu_pose.block<3, 3>(0, 0) = H_imu.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos).cast<double>();
        H_imu_pose.block<3, 3>(0, 3) = H_imu.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot).cast<double>();
        H_imu_pose.block<3, 3>(3, 0) = H_imu.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxPos).cast<double>();
        H_imu_pose.block<3, 3>(3, 3) = H_imu.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot).cast<double>();
        const Eigen::Matrix<double, kPoseDof, kPoseDof> H_imu_balanced =
            analysis.balance_inverse *
            (0.5 * (H_imu_pose + H_imu_pose.transpose()) / static_cast<double>(inlier_f)) *
            analysis.balance_inverse;

        if (params.verbose) {
            std::cout << "[DirectionalIcpWeighting] coupled eigenvalues/inlier: "
                      << analysis.normalized_eigenvalues.transpose() << std::endl;
            std::cout << "[DirectionalIcpWeighting] coupled representative_length: "
                      << params.coupled_representative_length << std::endl;
        }

        const double ratio = std::max(0.0, static_cast<double>(params.coupled_min_information_ratio));
        const double floor_info = std::max(0.0, static_cast<double>(params.coupled_imu_information_floor_per_inlier));
        const double weak_scale = std::clamp(static_cast<double>(params.coupled_weak_direction_scale), 0.0, 1.0);

        std::vector<int> weak_indices;
        std::vector<double> weak_scales;
        for (int k = 0; k < kPoseDof; ++k) {
            const double lambda = analysis.normalized_eigenvalues(k);
            const Eigen::Matrix<double, kPoseDof, 1> v = analysis.normalized_eigenvectors.col(k);
            const double imu_info = std::max(floor_info, v.dot(H_imu_balanced * v));
            const double weak_threshold = ratio * imu_info;

            double scale = 1.0;
            if (lambda <= 0.0) {
                scale = 0.0;
            } else if (weak_threshold > 0.0 && lambda < weak_threshold) {
                if (params.type == DirectionalIcpWeightingType::tsvd) {
                    scale = 0.0;
                } else {
                    scale = std::max(weak_scale, std::clamp(lambda / weak_threshold, 0.0, 1.0));
                }
                if (params.verbose) {
                    std::cout << "[DirectionalIcpWeighting] coupled weak eigenvector: " << v.transpose()
                              << ", scale: " << scale << ", icp_info: " << lambda << ", imu_info: " << imu_info
                              << std::endl;
                }
            }
            if (scale < 1.0) {
                weak_indices.push_back(k);
                weak_scales.push_back(scale);
            }
        }

        // The per-mode scale is attached before orthonormalisation, so it is
        // exact for a single weak mode and approximate when several weak modes
        // are non-orthogonal (L != 1); see coupled_weak_scaled_directions.
        filter = Eigen::Matrix<float, kPoseDof, kPoseDof>::Identity();
        for (const auto& candidate :
             registration::coupled_weak_scaled_directions(analysis, weak_indices, weak_scales)) {
            const Eigen::Matrix<float, kPoseDof, 1> direction = candidate.direction.cast<float>();
            const float sqrt_scale = std::sqrt(std::clamp(static_cast<float>(candidate.scale), 0.0f, 1.0f));
            filter.noalias() -= (1.0f - sqrt_scale) * (direction * direction.transpose());
        }
        filter = 0.5f * (filter + filter.transpose());
    } else {
        Eigen::Vector3f trans_eigenvalues;
        Eigen::Matrix3f trans_eigenvectors;
        Eigen::Vector3f rot_eigenvalues;
        Eigen::Matrix3f rot_eigenvectors;
        if (!block_eigendata(H_pose.block<3, 3>(0, 0), trans_eigenvalues, trans_eigenvectors)) return;
        if (!block_eigendata(H_pose.block<3, 3>(3, 3), rot_eigenvalues, rot_eigenvectors)) return;

        filter.block<3, 3>(0, 0) = compute_block_filter(
            trans_eigenvalues, trans_eigenvectors, H_imu.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos),
            b_pose.segment<3>(0), params.trans_min_information_ratio, params.trans_imu_information_floor_per_inlier,
            params.trans_weak_direction_scale, "translation");
        filter.block<3, 3>(3, 3) = compute_block_filter(
            rot_eigenvalues, rot_eigenvectors, H_imu.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot),
            b_pose.segment<3>(3), params.rot_min_information_ratio, params.rot_imu_information_floor_per_inlier,
            params.rot_weak_direction_scale, "rotation");
    }

    const Eigen::Matrix<float, kPoseDof, kPoseDof> H_filtered = filter * H_pose * filter;
    const Eigen::Matrix<float, kPoseDof, 1> b_filtered = filter * filter * b_pose;

    icp_factor.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos) = H_filtered.block<3, 3>(0, 0);
    icp_factor.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot) = H_filtered.block<3, 3>(0, 3);
    icp_factor.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxPos) = H_filtered.block<3, 3>(3, 0);
    icp_factor.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot) = H_filtered.block<3, 3>(3, 3);
    icp_factor.b.segment<3>(imu::State::kIdxPos) = b_filtered.segment<3>(0);
    icp_factor.b.segment<3>(imu::State::kIdxRot) = b_filtered.segment<3>(3);
}

/// @brief Log the IMU prior's effective information and gradient per inlier in the
///        same units as apply_directional_icp_weighting()'s ICP logs, so the
///        weak-direction decision and the resulting update direction can be read
///        directly against the IMU terms they are being balanced against.
///
/// For a diagonal P_pred the information print is 1 / (sigma_imu^2 * inlier) per
/// direction.  The gradient (b) sign shows whether the IMU prior itself is pulling
/// the solve forward or backward along each axis.
inline void log_imu_effective_information(const Eigen::Matrix<float, 15, 15>& H_imu,
                                          const Eigen::Matrix<float, 15, 1>& b_imu, uint32_t inlier) {
    if (inlier == 0) return;

    const float inlier_f = static_cast<float>(inlier);
    const auto log_block = [&](const Eigen::Matrix3f& H_block, const Eigen::Vector3f& b_block, const char* label) {
        const Eigen::Matrix3f H_sym = 0.5f * (H_block + H_block.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(H_sym);
        if (solver.info() != Eigen::Success) return;
        std::cout << "[DirectionalIcpWeighting] imu " << label
                  << " eigenvalues/inlier: " << (solver.eigenvalues() / inlier_f).transpose() << std::endl;
        std::cout << "[DirectionalIcpWeighting] imu " << label
                  << " b/inlier: " << (b_block / inlier_f).transpose() << std::endl;
    };
    log_block(H_imu.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos), b_imu.segment<3>(imu::State::kIdxPos),
              "translation");
    log_block(H_imu.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot), b_imu.segment<3>(imu::State::kIdxRot),
              "rotation");
}

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
