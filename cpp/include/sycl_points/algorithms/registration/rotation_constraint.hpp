#pragma once

#include "sycl_points/algorithms/common/transform.hpp"
#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/points/types.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace registration {

namespace kernel {

SYCL_EXTERNAL inline float calculate_logdet_divergence_squared(const Covariance& source_cov,
                                                               const Covariance& target_cov,
                                                               const std::array<sycl::float4, 4>& T) {
    const Eigen::Matrix3f R = eigen_utils::from_sycl_vec(T).block<3, 3>(0, 0);
    const Eigen::Matrix3f Cs = source_cov.block<3, 3>(0, 0);
    const Eigen::Matrix3f Ct = target_cov.block<3, 3>(0, 0);

    // transform target covariance to source coordinate
    // Ct' = R.T * Ct * R
    const Eigen::Matrix3f Ct_prime =
        eigen_utils::multiply<3, 3, 3>(eigen_utils::transpose<3, 3>(R), eigen_utils::multiply<3, 3, 3>(Ct, R));

    // M = 0.5 * (Cs + Ct')
    const Eigen::Matrix3f M = eigen_utils::multiply<3, 3>(eigen_utils::add<3, 3>(Cs, Ct_prime), 0.5f);

    // Jensen-Bregman LogDet Divergence: D(Cs, Ct') = log(det(0.5 * (Cs + Ct'))) - 0.5 * (log(det(Cs)) + log(det(Ct')))
    const float det_M = eigen_utils::determinant(M);
    const float det_Cs = eigen_utils::determinant(Cs);
    const float det_Ct = eigen_utils::determinant(Ct);

    auto logdet = [](const Eigen::Matrix3f& mat) {
        return sycl::log(sycl::fmax(eigen_utils::determinant(mat), 1e-10f));
    };
    const float log_det_M = logdet(M);
    const float log_det_ref = 0.5f * (logdet(Cs) + logdet(Ct));
    const float D = sycl::fmax(log_det_M - log_det_ref, 0.0f);

    return 0.5f * D * D;
}

SYCL_EXTERNAL inline float calculate_logdet_divergence(const Covariance& source_cov, const Covariance& target_cov,
                                                       const std::array<sycl::float4, 4>& T, Eigen::Vector3f& grad) {
    const Eigen::Matrix3f R = eigen_utils::from_sycl_vec(T).block<3, 3>(0, 0);
    const Eigen::Matrix3f Cs = source_cov.block<3, 3>(0, 0);
    const Eigen::Matrix3f Ct = target_cov.block<3, 3>(0, 0);

    // transform target covariance to source coordinate
    // Ct' = R.T * Ct * R
    const Eigen::Matrix3f Ct_prime =
        eigen_utils::multiply<3, 3, 3>(eigen_utils::transpose<3, 3>(R), eigen_utils::multiply<3, 3, 3>(Ct, R));

    // M = 0.5 * (Cs + Ct')
    const Eigen::Matrix3f M = eigen_utils::multiply<3, 3>(eigen_utils::add<3, 3>(Cs, Ct_prime), 0.5f);

    // Jensen-Bregman LogDet Divergence: D(Cs, Ct') = log(det(0.5 * (Cs + Ct'))) - 0.5 * (log(det(Cs)) + log(det(Ct')))
    auto logdet = [](const Eigen::Matrix3f& mat) {
        return sycl::log(sycl::fmax(eigen_utils::determinant(mat), 1e-10f));
    };
    const float log_det_M = logdet(M);
    const float log_det_ref = 0.5f * (logdet(Cs) + logdet(Ct));
    const float D = sycl::fmax(log_det_M - log_det_ref, 0.0f);

    // Gradient: g = -vex([M^{-1}, Ct')
    const Eigen::Matrix3f M_inv = eigen_utils::inverse(M);
    // commutator
    const Eigen::Matrix3f comm = eigen_utils::subtract<3, 3>(eigen_utils::multiply<3, 3, 3>(M_inv, Ct_prime),
                                                             eigen_utils::multiply<3, 3, 3>(Ct_prime, M_inv));

    // vex
    grad[0] = -0.5f * (comm(2, 1) - comm(1, 2));
    grad[1] = -0.5f * (comm(0, 2) - comm(2, 0));
    grad[2] = -0.5f * (comm(1, 0) - comm(0, 1));

    return D;
}

SYCL_EXTERNAL inline LinearizedKernelResult linearize_rotation_constraint_logdet(const Covariance& source_cov,
                                                                                 const Covariance& target_cov,
                                                                                 const std::array<sycl::float4, 4>& T) {
    // Compute jacobian
    Eigen::Vector3f J;
    const float D = calculate_logdet_divergence(source_cov, target_cov, T, J);
    const float r = D;

    LinearizedKernelResult result;
    result.b.setZero();
    result.H.setZero();

    for (size_t i = 0; i < 3; ++i) {
        result.b[i] = r * J[i];

        for (size_t j = 0; j < 3; ++j) {
            result.H(i, j) = J[i] * J[j];
        }
    }

    result.squared_error = 0.5f * r * r;
    return result;
}

SYCL_EXTERNAL inline float calculate_rotation_constraint_error(const Covariance& source_cov,
                                                               const Covariance& target_cov,
                                                               const std::array<sycl::float4, 4>& T) {
    return calculate_logdet_divergence_squared(source_cov, target_cov, T);
}

SYCL_EXTERNAL inline LinearizedKernelResult linearize_rotation_constraint(const Covariance& source_cov,
                                                                          const Covariance& target_cov,
                                                                          const std::array<sycl::float4, 4>& T) {
    return linearize_rotation_constraint_logdet(source_cov, target_cov, T);
}
}  // namespace kernel

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
