#pragma once

#include <sycl_points/points/container.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace factor {

struct LinearlizedResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
    float error = 0.0f;
};

enum class ICPType { POINT_TO_POINT, GICP };

SYCL_EXTERNAL inline LinearlizedResult linearlize_point_to_point(const TransformMatrix& T, const PointType& source,
                                                                 const PointType& transform_source,
                                                                 const PointType& target, const Covariance& source_cov,
                                                                 const Covariance& target_cov) {
    const PointType residual(target.x() - transform_source.x(), target.y() - transform_source.y(),
                             target.z() - transform_source.z(), 0.0f);
    Eigen::Matrix<float, 4, 6> J = Eigen::Matrix<float, 4, 6>::Zero();
    {
        const Eigen::Matrix3f skewed = eigen_utils::lie::skew(source);
        const Eigen::Matrix3f T_3x3 = eigen_utils::block3x3(T);
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

        J(0, 3) = -T(0, 0);
        J(0, 4) = -T(0, 1);
        J(0, 5) = -T(0, 2);
        J(1, 3) = -T(1, 0);
        J(1, 4) = -T(1, 1);
        J(1, 5) = -T(1, 2);
        J(2, 3) = -T(2, 0);
        J(2, 4) = -T(2, 1);
        J(2, 5) = -T(2, 2);
    }
    LinearlizedResult ret;
    const auto J_T = eigen_utils::transpose<4, 6>(J);
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T, J));
    ret.b = eigen_utils::multiply<6, 4>(J_T, residual);
    ret.error = 0.5f * eigen_utils::frobenius_norm<4>(residual);
    return ret;
}

SYCL_EXTERNAL inline LinearlizedResult linearlize_gicp(const TransformMatrix& T, const PointType& source,
                                                       const PointType& transform_source, const PointType& target,
                                                       const Covariance& source_cov, const Covariance& target_cov) {
    Covariance mahalanobis = Covariance::Zero();
    {
        const Eigen::Matrix3f RCR =
            eigen_utils::add<3, 3>(eigen_utils::block3x3(source_cov), eigen_utils::block3x3(target_cov));
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

    const PointType residual(target.x() - transform_source.x(), target.y() - transform_source.y(),
                             target.z() - transform_source.z(), 0.0f);

    Eigen::Matrix<float, 4, 6> J = Eigen::Matrix<float, 4, 6>::Zero();
    {
        const Eigen::Matrix3f skewed = eigen_utils::lie::skew(source);
        const Eigen::Matrix3f T_3x3 = eigen_utils::block3x3(T);
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

        J(0, 3) = -T(0, 0);
        J(0, 4) = -T(0, 1);
        J(0, 5) = -T(0, 2);
        J(1, 3) = -T(1, 0);
        J(1, 4) = -T(1, 1);
        J(1, 5) = -T(1, 2);
        J(2, 3) = -T(2, 0);
        J(2, 4) = -T(2, 1);
        J(2, 5) = -T(2, 2);
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

}  // namespace factor

}  // namespace algorithms

}  // namespace sycl_points
