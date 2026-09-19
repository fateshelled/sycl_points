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

    const auto compute_block_filter = [&](const Eigen::Matrix3f& H_block, const Eigen::Matrix3f& H_imu_block,
                                          const Eigen::Vector3f& b_block, float min_information_ratio,
                                          float imu_information_floor_per_inlier, float weak_direction_scale,
                                          const char* label) -> Eigen::Matrix3f {
        const Eigen::Matrix3f H_sym = 0.5f * (H_block + H_block.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(H_sym);
        if (solver.info() != Eigen::Success) return Eigen::Matrix3f::Identity();

        if (params.verbose) {
            std::cout << "[DirectionalIcpWeighting] " << label
                      << " eigenvalues/inlier: " << (solver.eigenvalues() / inlier_f).transpose() << std::endl;
            std::cout << "[DirectionalIcpWeighting] " << label
                      << " b/inlier: " << (b_block / inlier_f).transpose() << std::endl;
        }

        const float ratio = std::max(0.0f, min_information_ratio);
        const float floor_info = std::max(0.0f, imu_information_floor_per_inlier) * inlier_f;
        const float weak_scale = std::clamp(weak_direction_scale, 0.0f, 1.0f);
        Eigen::Matrix3f filter = Eigen::Matrix3f::Zero();
        for (int i = 0; i < kBlockDof; ++i) {
            const float lambda = std::max(0.0f, solver.eigenvalues()(i));
            const Eigen::Vector3f q = solver.eigenvectors().col(i);

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

    Eigen::Matrix<float, kPoseDof, kPoseDof> filter = Eigen::Matrix<float, kPoseDof, kPoseDof>::Zero();
    filter.block<3, 3>(0, 0) = compute_block_filter(
        H_pose.block<3, 3>(0, 0), H_imu.block<3, 3>(imu::State::kIdxPos, imu::State::kIdxPos), b_pose.segment<3>(0),
        params.trans_min_information_ratio, params.trans_imu_information_floor_per_inlier,
        params.trans_weak_direction_scale, "translation");
    filter.block<3, 3>(3, 3) = compute_block_filter(
        H_pose.block<3, 3>(3, 3), H_imu.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot), b_pose.segment<3>(3),
        params.rot_min_information_ratio, params.rot_imu_information_floor_per_inlier,
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
