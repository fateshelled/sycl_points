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
    const float safe_det = sycl::fmax(det, 1e-6f);
    return sycl::log(safe_det);
}

/// @brief Compute analytical gradient for Bhattacharyya log-det term
/// @param source_cov Source covariance matrix (4x4)
/// @param target_cov Target covariance matrix (4x4)
/// @param T transform matrix
/// @param grad Output gradient vector (rotation part only, size 3)
/// @return log(det(M)) value
SYCL_EXTERNAL inline float compute_bhattacharyya_gradient_analytical(const Covariance& source_cov,
                                                                     const Covariance& target_cov,
                                                                     const std::array<sycl::float4, 4>& T,
                                                                     Eigen::Vector3f& grad) {
    // 1. Extract rotation matrix R from T
    const Eigen::Matrix3f R = eigen_utils::from_sycl_vec(T).block<3, 3>(0, 0);

    // 2. Extract 3x3 covariances
    const Eigen::Matrix3f Cs = source_cov.block<3, 3>(0, 0);
    const Eigen::Matrix3f Ct = target_cov.block<3, 3>(0, 0);

    // 3. Compute RCR = R * Cs * R^T + Ct
    Covariance transform_source_cov;
    transform::kernel::transform_covs(source_cov, transform_source_cov, T);
    const Eigen::Matrix3f RCR =
        eigen_utils::add<3, 3>(transform_source_cov.block<3, 3>(0, 0), target_cov.block<3, 3>(0, 0));

    // 4. Check determinant for numerical stability
    const float det = eigen_utils::determinant(RCR);
    if (det < 1e-6f) {
        grad.setZero();
        return sycl::log(1e-6f);
    }

    // 5. Compute RCR^{-1} using Cholesky or direct inverse
    const Eigen::Matrix3f RCR_inv = eigen_utils::inverse(RCR);

    // 6. Compute S = R^T * RCR^{-1} * R
    const Eigen::Matrix3f RT_RCRinv = eigen_utils::multiply<3, 3, 3>(eigen_utils::transpose<3, 3>(R), RCR_inv);
    const Eigen::Matrix3f S = eigen_utils::multiply<3, 3, 3>(RT_RCRinv, R);

    // 7. Compute commutator [Cs, S] = Cs * S - S * Cs
    const Eigen::Matrix3f CsS = eigen_utils::multiply<3, 3, 3>(Cs, S);
    const Eigen::Matrix3f SCs = eigen_utils::multiply<3, 3, 3>(S, Cs);
    const Eigen::Matrix3f commutator = eigen_utils::subtract<3, 3>(CsS, SCs);

    // 8. Extract gradient using vex operator: g = 2 * vex(commutator)
    // vex extracts the axial vector from skew-symmetric part
    grad[0] = commutator(1, 2) - commutator(2, 1);  // 2 * (A_23 - A_32) / 2
    grad[1] = commutator(2, 0) - commutator(0, 2);  // 2 * (A_31 - A_13) / 2
    grad[2] = commutator(0, 1) - commutator(1, 0);  // 2 * (A_12 - A_21) / 2

    // 9. Compute log(det(M))
    return sycl::log(det);
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
    constexpr float kRotationStep = 5e-3f;  // 0.286 degrees
    const Eigen::Matrix4f T_mat = eigen_utils::from_sycl_vec(T);

    // Lambda to compute gradient at perturbed pose
    auto gradient_at_delta = [&](const Eigen::Vector3f& rot_delta) -> Eigen::Vector3f {
        Eigen::Vector<float, 6> delta = Eigen::Vector<float, 6>::Zero();
        delta.head<3>() = rot_delta;
        const Eigen::Matrix4f delta_T = eigen_utils::lie::se3_exp(delta);
        const Eigen::Matrix4f T_pert = eigen_utils::multiply<4, 4, 4>(T_mat, delta_T);
        const auto T_pert_vec = eigen_utils::to_sycl_vec(T_pert);

        Eigen::Vector3f grad;
        compute_bhattacharyya_gradient_analytical(source_cov, target_cov, T_pert_vec, grad);
        return grad;
    };

    // Compute gradient at current pose (analytical)
    Eigen::Vector3f grad;
    const float base = compute_bhattacharyya_gradient_analytical(source_cov, target_cov, T, grad);

    // Compute Hessian by central difference of gradient
    Eigen::Matrix3f hess = Eigen::Matrix3f::Zero();
    for (size_t j = 0; j < 3; ++j) {
        Eigen::Vector3f delta_j = Eigen::Vector3f::Zero();
        delta_j[j] = kRotationStep;

        const Eigen::Vector3f grad_plus = gradient_at_delta(delta_j);
        const Eigen::Vector3f grad_minus = gradient_at_delta(-delta_j);

        // H[:, j] = (grad(+h) - grad(-h)) / (2h)
        for (size_t i = 0; i < 3; ++i) {
            hess(i, j) = (grad_plus[i] - grad_minus[i]) / (2.0f * kRotationStep);
        }
    }

    // Ensure symmetry (numerical errors may break it slightly)
    hess = eigen_utils::ensure_symmetric<3>(hess);

    // Accumulate to output
    for (size_t i = 0; i < 3; ++i) {
        b[i] += coeff * grad[i];
        for (size_t j = 0; j < 3; ++j) {
            H(i, j) += coeff * hess(i, j);
        }
    }

    return coeff * base;
}

}  // namespace kernel

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
