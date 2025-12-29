#pragma once

#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/types.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace registration {

namespace kernel {

/// @brief Compute Bhattacharyya log-det term for GICP
/// @param source_cov Source covariance matrix (4x4)
/// @param target_cov Target covariance matrix (4x4)
/// @param T transform matrix
/// @return log(det(R * C_s * R^T + C_t)) with a safety clamp
SYCL_EXTERNAL inline float compute_bhattacharyya_logdet(const Covariance& source_cov, const Covariance& target_cov,
                                                        const std::array<sycl::float4, 4>& T) {
    Covariance transform_source_cov;
    transform::kernel::transform_covs(source_cov, transform_source_cov, T);
    const Eigen::Matrix3f RCR =
        eigen_utils::add<3, 3>(transform_source_cov.block<3, 3>(0, 0), target_cov.block<3, 3>(0, 0));
    const float det = eigen_utils::determinant(RCR);
    const float safe_det = sycl::fmax(det, 1e-12f);
    return sycl::log(safe_det);
}

/// @brief Compute numerical derivatives for Bhattacharyya log-det term w.r.t. rotation
/// @param source_cov Source covariance matrix (4x4)
/// @param target_cov Target covariance matrix (4x4)
/// @param T transform matrix
/// @param coeff Weight for log-det term
/// @param H Hessian matrix to update (rotation block only)
/// @param b Gradient vector to update (rotation block only)
/// @return Weighted Bhattacharyya log-det cost
SYCL_EXTERNAL inline float accumulate_bhattacharyya_rotation_terms(const Covariance& source_cov,
                                                                   const Covariance& target_cov,
                                                                   const std::array<sycl::float4, 4>& T, float coeff,
                                                                   Eigen::Matrix<float, 6, 6>& H,
                                                                   Eigen::Vector<float, 6>& b) {
    constexpr float kRotationStep = 1e-3f;  // 0.0573 degrees
    const Eigen::Matrix4f T_mat = eigen_utils::from_sycl_vec(T);

    auto logdet_with_delta = [&](const Eigen::Vector3f& rot_delta) -> float {
        Eigen::Vector<float, 6> delta = Eigen::Vector<float, 6>::Zero();
        delta[0] = rot_delta[0];
        delta[1] = rot_delta[1];
        delta[2] = rot_delta[2];
        const Eigen::Matrix4f delta_T = eigen_utils::lie::se3_exp(delta);
        const Eigen::Matrix4f T_pert = eigen_utils::multiply<4, 4, 4>(T_mat, delta_T);
        const auto T_pert_vec = eigen_utils::to_sycl_vec(T_pert);
        return compute_bhattacharyya_logdet(source_cov, target_cov, T_pert_vec);
    };

    const float base = logdet_with_delta(Eigen::Vector3f::Zero());

    Eigen::Vector3f grad = Eigen::Vector3f::Zero();
    Eigen::Matrix3f hess = Eigen::Matrix3f::Zero();

    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3f delta = Eigen::Vector3f::Zero();
        delta[i] = kRotationStep;
        const float f_plus = logdet_with_delta(delta);
        const float f_minus = logdet_with_delta(-delta);
        grad[i] = (f_plus - f_minus) / (2.0f * kRotationStep);

        Eigen::Vector3f delta_i = Eigen::Vector3f::Zero();
        delta_i[i] = kRotationStep;

        for (int j = i; j < 3; ++j) {
            Eigen::Vector3f delta_j = Eigen::Vector3f::Zero();
            delta_j[j] = kRotationStep;

            float f_pm, f_mp;
            if (i == j) {
                f_pm = base;
                f_mp = base;
            } else {
                f_pm = logdet_with_delta(delta_i - delta_j);
                f_mp = logdet_with_delta(-delta_i + delta_j);
            }

            const float f_pp = logdet_with_delta(delta_i + delta_j);
            const float f_mm = logdet_with_delta(-delta_i - delta_j);

            const float value = (f_pp - f_pm - f_mp + f_mm) / (4.0f * kRotationStep * kRotationStep);
            hess(i, j) = value;
            hess(j, i) = value;
        }
    }

    // Apply rotation-only gradient and Hessian updates.
    for (int i = 0; i < 3; ++i) {
        b[i] += coeff * grad[i];
        for (int j = 0; j < 3; ++j) {
            H(i, j) += coeff * hess(i, j);
        }
    }
    return coeff * base;
}

}  // namespace kernel

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
