#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/lio/lio_linearized_result.hpp"
#include "sycl_points/algorithms/lio/lio_registration_params.hpp"
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

/// @brief Attenuate ICP pose information in weak or over-confident directions.
inline void apply_directional_icp_weighting(LIOLinearizedResult& icp_factor,
                                            const DirectionalIcpWeightingParams& params) {
    if (!params.enable || icp_factor.inlier == 0) return;

    constexpr int kPoseDof = 6;
    constexpr int kBlockDof = 3;
    Eigen::Matrix<float, kPoseDof, kPoseDof> H_pose = Eigen::Matrix<float, kPoseDof, kPoseDof>::Zero();
    Eigen::Matrix<float, kPoseDof, 1> b_pose = Eigen::Matrix<float, kPoseDof, 1>::Zero();

    H_pose.block<3, 3>(0, 0) = icp_factor.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos);
    H_pose.block<3, 3>(0, 3) = icp_factor.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot);
    H_pose.block<3, 3>(3, 0) = icp_factor.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxPos);
    H_pose.block<3, 3>(3, 3) = icp_factor.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot);
    H_pose = 0.5f * (H_pose + H_pose.transpose());

    b_pose.segment<3>(0) = icp_factor.b.segment<3>(imu::State::kIdxPos);
    b_pose.segment<3>(3) = icp_factor.b.segment<3>(imu::State::kIdxRot);

    const auto compute_block_filter = [&](const Eigen::Matrix3f& H_block, float min_eigenvalue_per_inlier,
                                          float weak_direction_scale, const char* label) -> Eigen::Matrix3f {
        const Eigen::Matrix3f H_sym = 0.5f * (H_block + H_block.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(H_sym);
        if (solver.info() != Eigen::Success) return Eigen::Matrix3f::Identity();

        if (params.verbose) {
            std::cout << "[DirectionalIcpWeighting] " << label << " eigenvalues/inlier: "
                      << (solver.eigenvalues() / static_cast<float>(icp_factor.inlier)).transpose() << std::endl;
        }

        const float min_info = std::max(0.0f, min_eigenvalue_per_inlier) * static_cast<float>(icp_factor.inlier);
        const float weak_scale = std::clamp(weak_direction_scale, 0.0f, 1.0f);
        Eigen::Matrix3f filter = Eigen::Matrix3f::Zero();
        for (int i = 0; i < kBlockDof; ++i) {
            const float lambda = std::max(0.0f, solver.eigenvalues()(i));
            float scale = 1.0f;
            if (lambda <= 0.0f || !std::isfinite(lambda)) {
                scale = 0.0f;
            } else if (min_info > 0.0f && lambda < min_info) {
                if (params.type == DirectionalIcpWeightingType::tsvd) {
                    scale = 0.0f;
                } else {
                    const float information_ratio = std::clamp(lambda / min_info, 0.0f, 1.0f);
                    scale = std::max(weak_scale, information_ratio);
                }
            }

            const Eigen::Vector3f q = solver.eigenvectors().col(i);
            filter.noalias() += std::sqrt(std::clamp(scale, 0.0f, 1.0f)) * (q * q.transpose());
        }
        return filter;
    };

    Eigen::Matrix<float, kPoseDof, kPoseDof> filter = Eigen::Matrix<float, kPoseDof, kPoseDof>::Zero();
    filter.block<3, 3>(0, 0) = compute_block_filter(H_pose.block<3, 3>(0, 0), params.trans_min_eigenvalue_per_inlier,
                                                    params.trans_weak_direction_scale, "translation");
    filter.block<3, 3>(3, 3) = compute_block_filter(H_pose.block<3, 3>(3, 3), params.rot_min_eigenvalue_per_inlier,
                                                    params.rot_weak_direction_scale, "rotation");

    const Eigen::Matrix<float, kPoseDof, kPoseDof> H_filtered = filter * H_pose * filter;
    const Eigen::Matrix<float, kPoseDof, 1> b_filtered = filter * filter * b_pose;

    icp_factor.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos) = H_filtered.block<3, 3>(0, 0);
    icp_factor.H.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxRot) = H_filtered.block<3, 3>(0, 3);
    icp_factor.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxPos) = H_filtered.block<3, 3>(3, 0);
    icp_factor.H.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot) = H_filtered.block<3, 3>(3, 3);
    icp_factor.b.segment<3>(imu::State::kIdxPos) = b_filtered.segment<3>(0);
    icp_factor.b.segment<3>(imu::State::kIdxRot) = b_filtered.segment<3>(3);
}

/// @brief Log the IMU prior's effective information per inlier in the same units
///        as apply_directional_icp_weighting()'s ICP eigenvalues/inlier, so the
///        weak-direction thresholds can be compared directly against the IMU
///        information they are delegating to.
///
/// For a diagonal P_pred this prints 1 / (sigma_imu^2 * inlier) per direction.
inline void log_imu_effective_information(const Eigen::Matrix<float, 15, 15>& H_imu, uint32_t inlier) {
    if (inlier == 0) return;

    const float inlier_f = static_cast<float>(inlier);
    const auto log_block = [&](const Eigen::Matrix3f& block, const char* label) {
        const Eigen::Matrix3f H_sym = 0.5f * (block + block.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(H_sym);
        if (solver.info() != Eigen::Success) return;
        std::cout << "[DirectionalIcpWeighting] imu " << label << " eigenvalues/inlier: "
                  << (solver.eigenvalues() / inlier_f).transpose() << std::endl;
    };
    log_block(H_imu.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos), "translation");
    log_block(H_imu.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot), "rotation");
}

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
