#pragma once

#include <tuple>
#include <type_traits>

#include "sycl_points/algorithms/common/transform.hpp"
#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/algorithms/registration/result.hpp"
#include "sycl_points/points/types.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace registration {

enum class RegType {
    POINT_TO_POINT = 0,
    POINT_TO_PLANE,
    /// @brief Point-to-Distribution ICP
    /// @note Uses Mahalanobis distance with target covariance only (source points have no covariance)
    POINT_TO_DISTRIBUTION,
    /// @brief Generalized-ICP
    /// @authors Aleksandr V. Segal, Dirk Haehnel, Sebastian Thrun
    GICP,
    /// @brief GenZ-ICP: Generalizable and Degeneracy-Robust LiDAR Odometry Using an Adaptive Weighting
    /// @authors Daehan Lee, Hyungtae Lim, Soohee Han
    /// @date 2024
    /// @cite https://arxiv.org/abs/2411.06766
    GENZ
};

/// @brief Registration Type tags
using RegTypeTags = std::tuple<                                       //
    std::integral_constant<RegType, RegType::POINT_TO_POINT>,         //
    std::integral_constant<RegType, RegType::POINT_TO_PLANE>,         //
    std::integral_constant<RegType, RegType::POINT_TO_DISTRIBUTION>,  //
    std::integral_constant<RegType, RegType::GICP>,                   //
    std::integral_constant<RegType, RegType::GENZ>                    //
    >;                                                                //

RegType RegType_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(str.begin(), str.end(), upper.begin(), [](u_char c) { return std::toupper(c); });
    if (upper == "POINT_TO_POINT") {
        return RegType::POINT_TO_POINT;
    } else if (upper == "POINT_TO_PLANE") {
        return RegType::POINT_TO_PLANE;
    } else if (upper == "GICP") {
        return RegType::GICP;
    } else if (upper == "GENZ") {
        return RegType::GENZ;
    } else if (upper == "POINT_TO_DISTRIBUTION" || upper == "P2D") {
        return RegType::POINT_TO_DISTRIBUTION;
    }
    std::string error_str = "[RegType_from_string] Invalid RegType str '";
    error_str += str;
    error_str += "'";
    throw std::runtime_error(error_str);
}

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
    const Eigen::Matrix3f R = eigen_utils::from_sycl_vec(T).block<3, 3>(0, 0);
    const Eigen::Matrix3f T_skewed = eigen_utils::multiply<3, 3, 3>(R, skewed);

    // rotation part
    J.block<3, 3>(0, 0) = T_skewed;
    // translation part: J_trans = -R
    J.block<3, 3>(0, 3) = -R;

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
    transform::kernel::transform_covs(source_cov, transform_source_cov, T);
    const Eigen::Matrix3f RCR =
        eigen_utils::add<3, 3>(transform_source_cov.block<3, 3>(0, 0), target_cov.block<3, 3>(0, 0));
    const Eigen::Matrix3f RCR_inv = eigen_utils::inverse(RCR);

    mahalanobis.block<3, 3>(0, 0) = RCR_inv;

    return mahalanobis;
}

/// @brief Iterative Closest Point (ICP Point to Point)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param target_pt Target point
/// @return linearized result
SYCL_EXTERNAL inline LinearizedKernelResult linearize_point_to_point(const std::array<sycl::float4, 4>& T,
                                                                     const PointType& source_pt,
                                                                     const PointType& target_pt, float& residual_norm) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);

    const PointType residual(target_pt.x() - transform_source.x(), target_pt.y() - transform_source.y(),
                             target_pt.z() - transform_source.z(), 0.0f);
    const Eigen::Matrix<float, 4, 6> J = compute_weighted_se3_jacobian(T, source_pt, Eigen::Matrix4f::Identity());

    LinearizedKernelResult ret;
    const auto J_T = eigen_utils::transpose<4, 6>(J);
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T, J));
    ret.b = eigen_utils::multiply<6, 4>(J_T, residual);
    const float squared_norm = eigen_utils::frobenius_norm_squared<4>(residual);
    residual_norm = sycl::sqrt(squared_norm);
    ret.squared_error = squared_norm;
    ret.inlier = 1;
    return ret;
}

/// @brief Error for Iterative Closest Point (ICP Point to Point)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param target_pt Target point
/// @return error
SYCL_EXTERNAL inline float calculate_point_to_point_error(const std::array<sycl::float4, 4>& T,
                                                          const PointType& source_pt, const PointType& target_pt) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);

    const PointType residual(target_pt.x() - transform_source.x(), target_pt.y() - transform_source.y(),
                             target_pt.z() - transform_source.z(), 0.0f);
    return eigen_utils::frobenius_norm_squared<4>(residual);
}

/// @brief Iterative Closest Point (ICP Point to Plane)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param target_pt Target point
/// @param target_normal Target normal
/// @return linearized result
SYCL_EXTERNAL inline LinearizedKernelResult linearize_point_to_plane(const std::array<sycl::float4, 4>& T,
                                                                     const PointType& source_pt,
                                                                     const PointType& target_pt,
                                                                     const Normal& target_normal,
                                                                     float& residual_norm) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);

    const PointType residual(target_pt.x() - transform_source.x(), target_pt.y() - transform_source.y(),
                             target_pt.z() - transform_source.z(), 0.0f);

    // Project the residual onto the target normal using SYCL-friendly helpers.
    const Eigen::Vector3f normal = target_normal.head<3>();
    const float projected_residual = eigen_utils::dot<3>(normal, residual.head<3>());

    // Weighted residual aligns with the target normal (n * (n^T * r)).
    PointType plane_error = PointType::Zero();
    plane_error.head<3>() = normal * projected_residual;

    // Apply the same projection to the Jacobian to match the scalar point-to-plane constraint.
    // This computes n * (n^T * J_se3) without forming n*n^T.
    const auto se3_jacobian = compute_se3_jacobian(T, source_pt);
    const auto projected_jacobian_row =
        eigen_utils::multiply<1, 3, 6>(normal.transpose(), se3_jacobian.block<3, 6>(0, 0));
    Eigen::Matrix<float, 4, 6> J = Eigen::Matrix<float, 4, 6>::Zero();
    J.block<3, 6>(0, 0) = eigen_utils::multiply<3, 1, 6>(normal, projected_jacobian_row);

    LinearizedKernelResult ret;
    const auto J_T = eigen_utils::transpose<4, 6>(J);
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T, J));
    ret.b = eigen_utils::multiply<6, 4>(J_T, plane_error);

    // Scalar point-to-plane error uses the squared projection length.
    const float squared_norm = projected_residual * projected_residual;
    residual_norm = sycl::fabs(projected_residual);
    ret.squared_error = squared_norm;
    ret.inlier = 1;
    return ret;
}

/// @brief Error for Iterative Closest Point (ICP Point to Plane)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param target_pt Target point
/// @param target_normal Target normal
/// @return Squared error
SYCL_EXTERNAL inline float calculate_point_to_plane_error(const std::array<sycl::float4, 4>& T,
                                                          const PointType& source_pt, const PointType& target_pt,
                                                          const Normal& target_normal) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);

    const PointType residual(target_pt.x() - transform_source.x(), target_pt.y() - transform_source.y(),
                             target_pt.z() - transform_source.z(), 0.0f);

    // Error is the squared projection of the residual onto the target normal.
    const float projected_residual = eigen_utils::dot<3>(target_normal.head<3>(), residual.head<3>());
    return projected_residual * projected_residual;
}

/// @brief Generalized Iterative Closest Point (GICP)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return linearized result
SYCL_EXTERNAL inline LinearizedKernelResult linearize_gicp(const std::array<sycl::float4, 4>& T,
                                                           const PointType& source_pt, const Covariance& source_cov,
                                                           const PointType& target_pt, const Covariance& target_cov,
                                                           float& residual_norm) {
    PointType transform_source_pt;
    transform::kernel::transform_point(source_pt, transform_source_pt, T);

    const PointType residual(target_pt.x() - transform_source_pt.x(), target_pt.y() - transform_source_pt.y(),
                             target_pt.z() - transform_source_pt.z(), 0.0f);

    Covariance normalized_source_cov = source_cov;
    covariance::kernel::update_covariance_plane(normalized_source_cov);

    Covariance normalized_target_cov = target_cov;
    covariance::kernel::update_covariance_plane(normalized_target_cov);

    const Covariance mahalanobis = compute_mahalanobis_covariance(normalized_source_cov, normalized_target_cov, T);

    const Eigen::Matrix<float, 4, 6> J = compute_weighted_se3_jacobian(T, source_pt, Eigen::Matrix4f::Identity());

    const Eigen::Matrix<float, 6, 4> J_T_mah =
        eigen_utils::multiply<6, 4, 4>(eigen_utils::transpose<4, 6>(J), mahalanobis);

    LinearizedKernelResult ret;
    // H = J.transpose() * mahalanobis * J;
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T_mah, J));
    // b = J.transpose() * mahalanobis * residual;
    ret.b = eigen_utils::multiply<6, 4>(J_T_mah, residual);

    const float squared_norm = eigen_utils::dot<4>(residual, eigen_utils::multiply<4, 4>(mahalanobis, residual));
    residual_norm = sycl::sqrt(squared_norm);

    ret.squared_error = squared_norm;
    ret.inlier = 1;
    return ret;
}

/// @brief Error for Generalized Iterative Closest Point (GICP)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return Squared error
SYCL_EXTERNAL inline float calculate_gicp_error(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                                const Covariance& source_cov, const PointType& target_pt,
                                                const Covariance& target_cov) {
    PointType transform_source_pt;
    transform::kernel::transform_point(source_pt, transform_source_pt, T);

    const PointType residual(target_pt.x() - transform_source_pt.x(), target_pt.y() - transform_source_pt.y(),
                             target_pt.z() - transform_source_pt.z(), 0.0f);

    Covariance normalized_source_cov = source_cov;
    covariance::kernel::update_covariance_plane(normalized_source_cov);

    Covariance normalized_target_cov = target_cov;
    covariance::kernel::update_covariance_plane(normalized_target_cov);

    const Covariance mahalanobis = compute_mahalanobis_covariance(normalized_source_cov, normalized_target_cov, T);

    return (eigen_utils::dot<4>(residual, eigen_utils::multiply<4, 4>(mahalanobis, residual)));
}

/// @brief Compute inverse covariance matrix for Point-to-Distribution ICP
/// @param target_cov Target covariance matrix (4x4)
/// @return Inverse covariance matrix (4x4) for Mahalanobis distance
SYCL_EXTERNAL inline Covariance compute_target_mahalanobis(const Covariance& target_cov) {
    Covariance mahalanobis = Covariance::Zero();
    const Eigen::Matrix3f cov_inv = eigen_utils::inverse(target_cov.block<3, 3>(0, 0));
    mahalanobis.block<3, 3>(0, 0) = cov_inv;
    return mahalanobis;
}

/// @brief Point-to-Distribution ICP
/// @note Uses Mahalanobis distance with target covariance only.
///       Source points are treated as having no uncertainty (delta distribution).
/// @param T transform matrix
/// @param source_pt Source Point (coordinate only, no covariance)
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return linearized result
SYCL_EXTERNAL inline LinearizedKernelResult linearize_point_to_distribution(const std::array<sycl::float4, 4>& T,
                                                                            const PointType& source_pt,
                                                                            const PointType& target_pt,
                                                                            const Covariance& target_cov,
                                                                            float& residual_norm) {
    PointType transform_source_pt;
    transform::kernel::transform_point(source_pt, transform_source_pt, T);

    const PointType residual(target_pt.x() - transform_source_pt.x(), target_pt.y() - transform_source_pt.y(),
                             target_pt.z() - transform_source_pt.z(), 0.0f);
    const Eigen::Matrix<float, 4, 6> J = compute_weighted_se3_jacobian(T, source_pt, Eigen::Matrix4f::Identity());
    const Covariance mahalanobis = compute_target_mahalanobis(target_cov);

    const Eigen::Matrix<float, 6, 4> J_T_mah =
        eigen_utils::multiply<6, 4, 4>(eigen_utils::transpose<4, 6>(J), mahalanobis);

    LinearizedKernelResult ret;
    // H = J.transpose() * mahalanobis * J;
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T_mah, J));
    // b = J.transpose() * mahalanobis * residual;
    ret.b = eigen_utils::multiply<6, 4>(J_T_mah, residual);

    const float squared_norm = eigen_utils::dot<4>(residual, eigen_utils::multiply<4, 4>(mahalanobis, residual));
    residual_norm = sycl::sqrt(squared_norm);

    ret.squared_error = squared_norm;
    ret.inlier = 1;
    return ret;
}

/// @brief Error for Point-to-Distribution ICP
/// @param T transform matrix
/// @param source_pt Source Point (coordinate only, no covariance)
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return Squared error
SYCL_EXTERNAL inline float calculate_point_to_distribution_error(const std::array<sycl::float4, 4>& T,
                                                                 const PointType& source_pt, const PointType& target_pt,
                                                                 const Covariance& target_cov) {
    PointType transform_source_pt;
    transform::kernel::transform_point(source_pt, transform_source_pt, T);

    const Covariance mahalanobis = compute_target_mahalanobis(target_cov);

    const PointType residual(target_pt.x() - transform_source_pt.x(), target_pt.y() - transform_source_pt.y(),
                             target_pt.z() - transform_source_pt.z(), 0.0f);
    return (eigen_utils::dot<4>(residual, eigen_utils::multiply<4, 4>(mahalanobis, residual)));
}

/// @brief Linearization
/// @tparam reg registration type
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @param residual_norm L2 norm of residual vector
/// @return linearized result
template <RegType reg = RegType::GICP>
SYCL_EXTERNAL inline LinearizedKernelResult linearize_geometry(const std::array<sycl::float4, 4>& T,  //
                                                               const PointType& source_pt, const Covariance& source_cov,
                                                               const PointType& target_pt, const Covariance& target_cov,
                                                               const Normal& target_normal, float& residual_norm,
                                                               float genz_alpha) {
    if constexpr (reg == RegType::POINT_TO_POINT) {
        return linearize_point_to_point(T, source_pt, target_pt, residual_norm);
    } else if constexpr (reg == RegType::POINT_TO_PLANE) {
        return linearize_point_to_plane(T, source_pt, target_pt, target_normal, residual_norm);
    } else if constexpr (reg == RegType::GICP) {
        return linearize_gicp(T, source_pt, source_cov, target_pt, target_cov, residual_norm);
    } else if constexpr (reg == RegType::GENZ) {
        float pt2pt_residual_norm = 0.0f;
        float pt2pl_residual_norm = 0.0f;
        const auto pt2pt_result = linearize_point_to_point(T, source_pt, target_pt, pt2pt_residual_norm);
        const auto pt2pl_result = linearize_point_to_plane(T, source_pt, target_pt, target_normal, pt2pl_residual_norm);

        residual_norm = pt2pt_residual_norm * (1.0f - genz_alpha) + pt2pl_residual_norm * genz_alpha;

        LinearizedKernelResult result;
        result.H = eigen_utils::add<6, 6>(eigen_utils::multiply<6, 6>(pt2pt_result.H, 1.0f - genz_alpha),
                                          eigen_utils::multiply<6, 6>(pt2pl_result.H, genz_alpha));
        result.b = eigen_utils::add<6, 1>(eigen_utils::multiply<6, 1>(pt2pt_result.b, 1.0f - genz_alpha),
                                          eigen_utils::multiply<6, 1>(pt2pl_result.b, genz_alpha));
        result.squared_error =
            pt2pt_result.squared_error * (1.0f - genz_alpha) + pt2pl_result.squared_error * genz_alpha;
        result.inlier = 1;
        return result;
    } else if constexpr (reg == RegType::POINT_TO_DISTRIBUTION) {
        return linearize_point_to_distribution(T, source_pt, target_pt, target_cov, residual_norm);
    } else {
        static_assert("not support type");
    }
}

/// @brief Compute Error
/// @tparam reg registration type
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return error
template <RegType reg = RegType::GICP>
SYCL_EXTERNAL inline float calculate_geometry_error(const std::array<sycl::float4, 4>& T,                      //
                                                    const PointType& source_pt, const Covariance& source_cov,  //
                                                    const PointType& target_pt, const Covariance& target_cov,  //
                                                    const Normal& target_normal, const float genz_alpha = 1.0f) {
    if constexpr (reg == RegType::POINT_TO_POINT) {
        return calculate_point_to_point_error(T, source_pt, target_pt);
    } else if constexpr (reg == RegType::POINT_TO_PLANE) {
        return calculate_point_to_plane_error(T, source_pt, target_pt, target_normal);
    } else if constexpr (reg == RegType::GICP) {
        return calculate_gicp_error(T, source_pt, source_cov, target_pt, target_cov);
    } else if constexpr (reg == RegType::GENZ) {
        const float pt2pt_err = calculate_point_to_point_error(T, source_pt, target_pt);
        const float pt2pl_err = calculate_point_to_plane_error(T, source_pt, target_pt, target_normal);
        return pt2pt_err * (1.0f - genz_alpha) + pt2pl_err * genz_alpha;
    } else if constexpr (reg == RegType::POINT_TO_DISTRIBUTION) {
        return calculate_point_to_distribution_error(T, source_pt, target_pt, target_cov);
    } else {
        static_assert("not support type");
    }
}

}  // namespace kernel
}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
