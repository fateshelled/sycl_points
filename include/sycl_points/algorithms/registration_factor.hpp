#pragma once

#include <sycl_points/algorithms/registration_result.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/types.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace registration {

enum class ICPType { POINT_TO_POINT, GICP };

/// @brief Registration Linearlized Result
struct LinearlizedResult {
    /// @brief Hessian, Information Matrix
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    /// @brief Gradient, Information Vector
    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
    /// @brief Error value
    float error = std::numeric_limits<float>::max();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

namespace kernel {
/// @brief Iterative Closest Point (ICP Point to Point)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param target_pt Target point
/// @return linearlized result
SYCL_EXTERNAL inline LinearlizedResult linearlize_point_to_point(const std::array<sycl::float4, 4>& T,
                                                                 const PointType& source_pt,
                                                                 const PointType& target_pt) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T.data());

    const PointType residual(target_pt.x() - transform_source.x(), target_pt.y() - transform_source.y(),
                             target_pt.z() - transform_source.z(), 0.0f);
    Eigen::Matrix<float, 4, 6> J = Eigen::Matrix<float, 4, 6>::Zero();
    {
        const Eigen::Matrix3f skewed = eigen_utils::lie::skew(source_pt);
        const Eigen::Matrix3f T_3x3 = eigen_utils::block3x3(eigen_utils::from_sycl_vec(T));
        const Eigen::Matrix3f T_skewed = eigen_utils::multiply<3, 3, 3>(T_3x3, skewed);
        J(0, 0) = T_skewed(0, 0);
        J(0, 1) = T_skewed(0, 1);
        J(0, 2) = T_skewed(0, 2);
        J(1, 0) = T_skewed(1, 0);
        J(1, 1) = T_skewed(1, 1);
        J(1, 2) = T_skewed(1, 2);
        J(2, 0) = T_skewed(2, 0);
        J(2, 1) = T_skewed(2, 1);
        J(2, 2) = T_skewed(2, 2);

        J(0, 3) = -T[0][0];
        J(0, 4) = -T[0][1];
        J(0, 5) = -T[0][2];
        J(1, 3) = -T[1][0];
        J(1, 4) = -T[1][1];
        J(1, 5) = -T[1][2];
        J(2, 3) = -T[2][0];
        J(2, 4) = -T[2][1];
        J(2, 5) = -T[2][2];
    }
    LinearlizedResult ret;
    const auto J_T = eigen_utils::transpose<4, 6>(J);
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T, J));
    ret.b = eigen_utils::multiply<6, 4>(J_T, residual);
    ret.error = 0.5f * eigen_utils::frobenius_norm<4>(residual);
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
    return 0.5f * eigen_utils::frobenius_norm<4>(residual);
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

    Eigen::Matrix<float, 4, 6> J = Eigen::Matrix<float, 4, 6>::Zero();
    {
        const Eigen::Matrix3f skewed = eigen_utils::lie::skew(source_pt);
        const Eigen::Matrix3f T_3x3 = eigen_utils::block3x3(eigen_utils::from_sycl_vec(T));
        const Eigen::Matrix3f T_skewed = eigen_utils::multiply<3, 3, 3>(T_3x3, skewed);
        J(0, 0) = T_skewed(0, 0);
        J(0, 1) = T_skewed(0, 1);
        J(0, 2) = T_skewed(0, 2);
        J(1, 0) = T_skewed(1, 0);
        J(1, 1) = T_skewed(1, 1);
        J(1, 2) = T_skewed(1, 2);
        J(2, 0) = T_skewed(2, 0);
        J(2, 1) = T_skewed(2, 1);
        J(2, 2) = T_skewed(2, 2);

        J(0, 3) = -T[0][0];
        J(0, 4) = -T[0][1];
        J(0, 5) = -T[0][2];
        J(1, 3) = -T[1][0];
        J(1, 4) = -T[1][1];
        J(1, 5) = -T[1][2];
        J(2, 3) = -T[2][0];
        J(2, 4) = -T[2][1];
        J(2, 5) = -T[2][2];
    }

    const Eigen::Matrix<float, 6, 4> J_T_mah =
        eigen_utils::multiply<6, 4, 4>(eigen_utils::transpose<4, 6>(J), mahalanobis);

    LinearlizedResult ret;
    // J.transpose() * mahalanobis * J;
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T_mah, J));
    // J.transpose() * mahalanobis * residual;
    ret.b = eigen_utils::multiply<6, 4>(J_T_mah, residual);
    // 0.5 * residual.transpose() * mahalanobis * residual;
    ret.error = 0.5f * (eigen_utils::dot<4>(residual, eigen_utils::multiply<4, 4>(mahalanobis, residual)));
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
/// @return linearlized result
template <ICPType icp = ICPType::GICP>
SYCL_EXTERNAL inline LinearlizedResult linearlize(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                                  const Covariance& source_cov, const PointType& target_pt,
                                                  const Covariance& target_cov) {
    if constexpr (icp == ICPType::POINT_TO_POINT) {
        return linearlize_point_to_point(T, source_pt, target_pt);
    } else if constexpr (icp == ICPType::GICP) {
        return linearlize_gicp(T, source_pt, source_cov, target_pt, target_cov);
    } else {
        static_assert("not support type");
    }
}

/// @brief Compute Error
/// @tparam icp icp type
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return error
template <ICPType icp = ICPType::GICP>
SYCL_EXTERNAL inline float calculate_error(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                           const Covariance& source_cov, const PointType& target_pt,
                                           const Covariance& target_cov) {
    if constexpr (icp == ICPType::POINT_TO_POINT) {
        return calculate_error_point_to_point(T, source_pt, target_pt);
    } else if constexpr (icp == ICPType::GICP) {
        return calculate_error_gicp(T, source_pt, source_cov, target_pt, target_cov);
    } else {
        static_assert("not support type");
    }
}

}  // namespace kernel

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
