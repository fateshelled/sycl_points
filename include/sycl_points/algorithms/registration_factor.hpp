#pragma once

#include <sycl_points/algorithms/registration_result.hpp>
#include <sycl_points/algorithms/registration_robust.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/types.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace registration {

enum class ICPType { POINT_TO_POINT, POINT_TO_PLANE, GICP };

/// @brief Registration Linearlized Result
struct LinearlizedResult {
    /// @brief Hessian, Information Matrix
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    /// @brief Gradient, Information Vector
    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
    /// @brief Error value
    float error = std::numeric_limits<float>::max();
    /// @brief inlier point num
    uint32_t inlier = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

namespace kernel {

/// @brief Compute basic SE(3) Jacobian matrix for ICP registration
/// @param T transform matrix
/// @param source_pt Source Point
/// @return Jacobian matrix (4x6) [residual_dim x se3_param_dim]
SYCL_EXTERNAL inline Eigen::Matrix<float, 4, 6> compute_se3_jacobian(const std::array<sycl::float4, 4>& T,
                                                                     const PointType& source_pt) {
    Eigen::Matrix<float, 4, 6> J = Eigen::Matrix<float, 4, 6>::Zero();

    // Compute rotation part: J_rot = T * skew(source_pt)
    const Eigen::Matrix3f skewed = eigen_utils::lie::skew(source_pt);
    const Eigen::Matrix3f T_3x3 = eigen_utils::block3x3(eigen_utils::from_sycl_vec(T));
    const Eigen::Matrix3f T_skewed = eigen_utils::multiply<3, 3, 3>(T_3x3, skewed);

    // Fill rotation part (columns 0-2)
    J(0, 0) = T_skewed(0, 0);
    J(0, 1) = T_skewed(0, 1);
    J(0, 2) = T_skewed(0, 2);
    J(1, 0) = T_skewed(1, 0);
    J(1, 1) = T_skewed(1, 1);
    J(1, 2) = T_skewed(1, 2);
    J(2, 0) = T_skewed(2, 0);
    J(2, 1) = T_skewed(2, 1);
    J(2, 2) = T_skewed(2, 2);

    // Compute translation part: J_trans = -T_rot
    J(0, 3) = -T[0][0];
    J(0, 4) = -T[0][1];
    J(0, 5) = -T[0][2];
    J(1, 3) = -T[1][0];
    J(1, 4) = -T[1][1];
    J(1, 5) = -T[1][2];
    J(2, 3) = -T[2][0];
    J(2, 4) = -T[2][1];
    J(2, 5) = -T[2][2];

    return J;
}

/// @brief Apply weight matrix to Jacobian (left multiplication)
/// @param J_basic Basic Jacobian matrix (4x6)
/// @param weight_matrix Weight matrix (4x4)
/// @return Weighted Jacobian matrix (4x6)
SYCL_EXTERNAL inline Eigen::Matrix<float, 4, 6> apply_weight_to_jacobian(
    const Eigen::Matrix<float, 4, 6>& J_basic, const Eigen::Matrix<float, 4, 4>& weight_matrix) {
    return eigen_utils::multiply<4, 4, 6>(weight_matrix, J_basic);
}

/// @brief Compute weighted SE(3) Jacobian matrix
/// @param T transform matrix
/// @param source_pt Source Point
/// @param weight_matrix Weight matrix for error term
/// @return Weighted Jacobian matrix (4x6)
SYCL_EXTERNAL inline Eigen::Matrix<float, 4, 6> compute_weighted_se3_jacobian(
    const std::array<sycl::float4, 4>& T, const PointType& source_pt, const Eigen::Matrix<float, 4, 4>& weight_matrix) {
    const auto J_basic = compute_se3_jacobian(T, source_pt);
    return apply_weight_to_jacobian(J_basic, weight_matrix);
}

/// @brief Compute Mahalanobis covariance matrix for GICP registration
/// @param source_cov Source covariance matrix (4x4)
/// @param target_cov Target covariance matrix (4x4)
/// @param T transform matrix
/// @return Mahalanobis covariance matrix (4x4)
SYCL_EXTERNAL inline Covariance compute_mahalanobis_covariance(const Covariance& source_cov,
                                                               const Covariance& target_cov,
                                                               const std::array<sycl::float4, 4>& T) {
    Covariance mahalanobis = Covariance::Zero();

    Covariance transform_source_cov;
    transform::kernel::transform_covs(source_cov, transform_source_cov, T.data());
    const Eigen::Matrix3f RCR =
        eigen_utils::add<3, 3>(eigen_utils::block3x3(transform_source_cov), eigen_utils::block3x3(target_cov));
    const Eigen::Matrix3f RCR_inv = eigen_utils::inverse(RCR);

    mahalanobis(0, 0) = RCR_inv(0, 0);
    mahalanobis(0, 1) = RCR_inv(0, 1);
    mahalanobis(0, 2) = RCR_inv(0, 2);
    mahalanobis(1, 0) = RCR_inv(1, 0);
    mahalanobis(1, 1) = RCR_inv(1, 1);
    mahalanobis(1, 2) = RCR_inv(1, 2);
    mahalanobis(2, 0) = RCR_inv(2, 0);
    mahalanobis(2, 1) = RCR_inv(2, 1);
    mahalanobis(2, 2) = RCR_inv(2, 2);

    return mahalanobis;
}

/// @brief Iterative Closest Point (ICP Point to Point)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param target_pt Target point
/// @return linearlized result
SYCL_EXTERNAL inline LinearlizedResult linearlize_point_to_point(const std::array<sycl::float4, 4>& T,
                                                                 const PointType& source_pt, const PointType& target_pt,
                                                                 float& residual_norm) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T.data());

    const PointType residual(target_pt.x() - transform_source.x(), target_pt.y() - transform_source.y(),
                             target_pt.z() - transform_source.z(), 0.0f);
    const Eigen::Matrix<float, 4, 6> J = compute_weighted_se3_jacobian(T, source_pt, Eigen::Matrix4f::Identity());

    LinearlizedResult ret;
    const auto J_T = eigen_utils::transpose<4, 6>(J);
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T, J));
    ret.b = eigen_utils::multiply<6, 4>(J_T, residual);
    const float squared_norm = eigen_utils::frobenius_norm_squared<4>(residual);
    residual_norm = sycl::sqrt(squared_norm);
    ret.error = 0.5f * squared_norm;
    ret.inlier = 1;
    return ret;
}

/// @brief Error for Iterative Closest Point (ICP Point to Point)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param target_pt Target point
/// @return error
SYCL_EXTERNAL inline float calculate_error_point_to_point(const std::array<sycl::float4, 4>& T,
                                                          const PointType& source_pt, const PointType& target_pt) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T.data());

    const PointType residual(target_pt.x() - transform_source.x(), target_pt.y() - transform_source.y(),
                             target_pt.z() - transform_source.z(), 0.0f);
    return 0.5f * eigen_utils::frobenius_norm_squared<4>(residual);
}

/// @brief Iterative Closest Point (ICP Point to Plane)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param target_pt Target point
/// @param target_normal Target normal
/// @return linearlized result
SYCL_EXTERNAL inline LinearlizedResult linearlize_point_to_plane(const std::array<sycl::float4, 4>& T,
                                                                 const PointType& source_pt, const PointType& target_pt,
                                                                 const Normal& target_normal, float& residual_norm) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T.data());

    const PointType residual(target_pt.x() - transform_source.x(), target_pt.y() - transform_source.y(),
                             target_pt.z() - transform_source.z(), 0.0f);
    const PointType plane_error = eigen_utils::element_wise_multiply<4, 1>(target_normal, residual);

    const Eigen::Matrix4f weight_matrix =
        eigen_utils::as_diagonal<4>({target_normal.x(), target_normal.y(), target_normal.z(), 0.0f});
    const Eigen::Matrix<float, 4, 6> J = compute_weighted_se3_jacobian(T, source_pt, weight_matrix);

    LinearlizedResult ret;
    const auto J_T = eigen_utils::transpose<4, 6>(J);
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T, J));
    ret.b = eigen_utils::multiply<6, 4>(J_T, plane_error);
    const float squared_norm = eigen_utils::frobenius_norm_squared<4>(plane_error);
    residual_norm = sycl::sqrt(squared_norm);
    ret.error = 0.5f * squared_norm;
    ret.inlier = 1;
    return ret;
}

/// @brief Error for Iterative Closest Point (ICP Point to Plane)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param target_pt Target point
/// @param target_normal Target normal
/// @return error
SYCL_EXTERNAL inline float calculate_error_point_to_plane(const std::array<sycl::float4, 4>& T,
                                                          const PointType& source_pt, const PointType& target_pt,
                                                          const Normal& target_normal) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T.data());

    const PointType residual(target_pt.x() - transform_source.x(), target_pt.y() - transform_source.y(),
                             target_pt.z() - transform_source.z(), 0.0f);
    const PointType plane_error = eigen_utils::element_wise_multiply<4, 1>(target_normal, residual);
    return 0.5f * eigen_utils::frobenius_norm_squared<4>(plane_error);
}

/// @brief Generalized Iterative Closest Point (GICP)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return linearlized result
SYCL_EXTERNAL inline LinearlizedResult linearlize_gicp(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                                       const Covariance& source_cov, const PointType& target_pt,
                                                       const Covariance& target_cov, float& residual_norm) {
    PointType transform_source_pt;
    transform::kernel::transform_point(source_pt, transform_source_pt, T.data());

    const PointType residual(target_pt.x() - transform_source_pt.x(), target_pt.y() - transform_source_pt.y(),
                             target_pt.z() - transform_source_pt.z(), 0.0f);
    const Eigen::Matrix<float, 4, 6> J = compute_weighted_se3_jacobian(T, source_pt, Eigen::Matrix4f::Identity());
    const Covariance mahalanobis = compute_mahalanobis_covariance(source_cov, target_cov, T);

    const Eigen::Matrix<float, 6, 4> J_T_mah =
        eigen_utils::multiply<6, 4, 4>(eigen_utils::transpose<4, 6>(J), mahalanobis);

    LinearlizedResult ret;
    // J.transpose() * mahalanobis * J;
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T_mah, J));
    // J.transpose() * mahalanobis * residual;
    ret.b = eigen_utils::multiply<6, 4>(J_T_mah, residual);

    const float squared_norm = eigen_utils::dot<4>(residual, eigen_utils::multiply<4, 4>(mahalanobis, residual));
    residual_norm = sycl::sqrt(squared_norm);

    // 0.5 * residual.transpose() * mahalanobis * residual;
    ret.error = 0.5f * squared_norm;
    ret.inlier = 1;
    return ret;
}

/// @brief Error for Generalized Iterative Closest Point (GICP)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return error
SYCL_EXTERNAL inline float calculate_error_gicp(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                                const Covariance& source_cov, const PointType& target_pt,
                                                const Covariance& target_cov) {
    PointType transform_source_pt;
    Covariance transform_source_cov;
    transform::kernel::transform_point(source_pt, transform_source_pt, T.data());
    transform::kernel::transform_covs(source_cov, transform_source_cov, T.data());

    Covariance mahalanobis = Covariance::Zero();
    {
        const Eigen::Matrix3f RCR =
            eigen_utils::add<3, 3>(eigen_utils::block3x3(transform_source_cov), eigen_utils::block3x3(target_cov));
        const Eigen::Matrix3f RCR_inv = eigen_utils::inverse(RCR);

        mahalanobis(0, 0) = RCR_inv(0, 0);
        mahalanobis(0, 1) = RCR_inv(0, 1);
        mahalanobis(0, 2) = RCR_inv(0, 2);
        mahalanobis(1, 0) = RCR_inv(1, 0);
        mahalanobis(1, 1) = RCR_inv(1, 1);
        mahalanobis(1, 2) = RCR_inv(1, 2);
        mahalanobis(2, 0) = RCR_inv(2, 0);
        mahalanobis(2, 1) = RCR_inv(2, 1);
        mahalanobis(2, 2) = RCR_inv(2, 2);
    }

    const PointType residual(target_pt.x() - transform_source_pt.x(), target_pt.y() - transform_source_pt.y(),
                             target_pt.z() - transform_source_pt.z(), 0.0f);
    return 0.5f * (eigen_utils::dot<4>(residual, eigen_utils::multiply<4, 4>(mahalanobis, residual)));
}

/// @brief Linearlization
/// @tparam icp icp type
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @param residual_norm L2 norm of residual vector
/// @return linearlized result
template <ICPType icp = ICPType::GICP>
SYCL_EXTERNAL inline LinearlizedResult linearlize(const std::array<sycl::float4, 4>& T,  //
                                                  const PointType& source_pt, const Covariance& source_cov,
                                                  const PointType& target_pt, const Covariance& target_cov,
                                                  const Normal& target_normal, float& residual_norm) {
    if constexpr (icp == ICPType::POINT_TO_POINT) {
        return linearlize_point_to_point(T, source_pt, target_pt, residual_norm);
    } else if constexpr (icp == ICPType::POINT_TO_PLANE) {
        return linearlize_point_to_plane(T, source_pt, target_pt, target_normal, residual_norm);
    } else if constexpr (icp == ICPType::GICP) {
        return linearlize_gicp(T, source_pt, source_cov, target_pt, target_cov, residual_norm);
    } else {
        static_assert("not support type");
    }
}

/// @brief Linearlization
/// @tparam icp icp type
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return linearlized result
template <ICPType icp = ICPType::GICP>
SYCL_EXTERNAL inline LinearlizedResult linearlize(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                                  const Covariance& source_cov, const PointType& target_pt,
                                                  const Covariance& target_cov, const Normal& target_normal) {
    float residual_norm = 0.0f;
    return linearlize<icp>(T, source_pt, source_cov, target_pt, target_cov, target_normal, residual_norm);
}

/// @brief Robust Linearlization
/// @tparam icp icp type
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @param robust_scale Robust estimation scale parameter
/// @return linearlized result
template <ICPType icp = ICPType::GICP, RobustLossType LossType = RobustLossType::NONE>
SYCL_EXTERNAL inline LinearlizedResult linearlize_robust(const std::array<sycl::float4, 4>& T,
                                                         const PointType& source_pt, const Covariance& source_cov,
                                                         const PointType& target_pt, const Covariance& target_cov,
                                                         const Normal& target_normal, float robust_scale) {
    float residual_norm = 0.0f;
    auto result = linearlize<icp>(T, source_pt, source_cov, target_pt, target_cov, target_normal, residual_norm);
    const auto weight = kernel::compute_robust_weight<LossType>(residual_norm, robust_scale);
    result.error = kernel::compute_robust_error<LossType>(residual_norm, robust_scale);

    eigen_utils::multiply_inplace<6, 1>(result.b, weight);
    eigen_utils::multiply_inplace<6, 6>(result.H, weight);

    return result;
}

/// @brief Compute Error
/// @tparam icp icp type
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return error
template <ICPType icp = ICPType::GICP, RobustLossType LossType = RobustLossType::NONE>
SYCL_EXTERNAL inline float calculate_error(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                           const Covariance& source_cov, const Normal& source_normal,
                                           const PointType& target_pt, const Covariance& target_cov,
                                           const Normal& target_normal, float robust_scale) {
    float residual_norm = 0.0f;

    if constexpr (icp == ICPType::POINT_TO_POINT) {
        residual_norm = calculate_error_point_to_point(T, source_pt, target_pt);
    } else if constexpr (icp == ICPType::POINT_TO_PLANE) {
        residual_norm = calculate_error_point_to_plane(T, source_pt, target_pt, target_normal);
    } else if constexpr (icp == ICPType::GICP) {
        residual_norm = calculate_error_gicp(T, source_pt, source_cov, target_pt, target_cov);
    } else {
        static_assert("not support type");
    }
    return kernel::compute_robust_error<LossType>(residual_norm, robust_scale);
}

}  // namespace kernel

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
