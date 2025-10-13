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
    Eigen::Vector<float, 6> b = Eigen::Vector<float, 6>::Zero();
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
    const Eigen::Matrix3f R = eigen_utils::from_sycl_vec(T).block<3, 3>(0, 0);
    const Eigen::Matrix3f T_skewed = eigen_utils::multiply<3, 3, 3>(R, skewed);

    // rotation part
    J.block<3, 3>(0, 0) = R * skewed;
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
/// @return linearlized result
SYCL_EXTERNAL inline LinearlizedResult linearlize_point_to_point(const std::array<sycl::float4, 4>& T,
                                                                 const PointType& source_pt, const PointType& target_pt,
                                                                 float& residual_norm) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);

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
SYCL_EXTERNAL inline float calculate_point_to_point_error(const std::array<sycl::float4, 4>& T,
                                                          const PointType& source_pt, const PointType& target_pt) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);

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
    transform::kernel::transform_point(source_pt, transform_source, T);

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
SYCL_EXTERNAL inline float calculate_point_to_plane_error(const std::array<sycl::float4, 4>& T,
                                                          const PointType& source_pt, const PointType& target_pt,
                                                          const Normal& target_normal) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);

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
    transform::kernel::transform_point(source_pt, transform_source_pt, T);

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
SYCL_EXTERNAL inline float calculate_gicp_error(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                                const Covariance& source_cov, const PointType& target_pt,
                                                const Covariance& target_cov) {
    PointType transform_source_pt;
    Covariance transform_source_cov;
    transform::kernel::transform_point(source_pt, transform_source_pt, T);
    transform::kernel::transform_covs(source_cov, transform_source_cov, T);

    Covariance mahalanobis = Covariance::Zero();
    {
        const Eigen::Matrix3f RCR =
            eigen_utils::add<3, 3>(transform_source_cov.block<3, 3>(0, 0), target_cov.block<3, 3>(0, 0));
        const Eigen::Matrix3f RCR_inv = eigen_utils::inverse(RCR);

        mahalanobis.block<3, 3>(0, 0) = RCR_inv;
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
SYCL_EXTERNAL inline LinearlizedResult linearlize_geometry(const std::array<sycl::float4, 4>& T,  //
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

/// @brief Compute Error
/// @tparam icp icp type
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return error
template <ICPType icp = ICPType::GICP>
SYCL_EXTERNAL inline float calculate_geometry_error(const std::array<sycl::float4, 4>& T,                      //
                                                    const PointType& source_pt, const Covariance& source_cov,  //
                                                    const PointType& target_pt, const Covariance& target_cov,  //
                                                    const Normal& target_normal) {
    if constexpr (icp == ICPType::POINT_TO_POINT) {
        return calculate_point_to_point_error(T, source_pt, target_pt);
    } else if constexpr (icp == ICPType::POINT_TO_PLANE) {
        return calculate_point_to_plane_error(T, source_pt, target_pt, target_normal);
    } else if constexpr (icp == ICPType::GICP) {
        return calculate_gicp_error(T, source_pt, source_cov, target_pt, target_cov);
    } else {
        static_assert("not support type");
    }
}

/// @brief Linearize color residual using RGB gradient
/// @note reference:
/// https://github.com/koide3/gtsam_points/blob/master/include/gtsam_points/factors/impl/integrated_colored_gicp_factor_impl.hpp
/// (MIT License)
/// @param T SE(3) transform applied to the source point
/// @param source_pt Source point before transformation
/// @param target_pt Target point associated with the source point
/// @param source_rgb Color observed at the source point
/// @param target_rgb Color observed at the target point
/// @param target_rgb_grad Spatial gradient of the target color
SYCL_EXTERNAL inline LinearlizedResult linearlize_color(
    const std::array<sycl::float4, 4>& T,    ///< SE(3) transform
    const PointType& source_pt,              ///< Source point
    const PointType& target_pt,              ///< Target point
    const RGBType& source_rgb,               ///< Source color
    const RGBType& target_rgb,               ///< Target color
    const Normal& target_normal,             ///< Target normal
    const ColorGradient& target_rgb_grad) {  ///< Target RGB gradient

    // Transform source point to compute geometric residual
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);
    const Eigen::Vector3f geometric_residual = (transform_source - target_pt).template head<3>();

    // Project the transformed source point onto the plane defined by the target normal
    // to decouple the color error from the geometric error along the normal.
    const Eigen::Vector3f projected =
        transform_source.template head<3>() -
        target_normal.template head<3>() * eigen_utils::dot<3>(geometric_residual, target_normal.template head<3>());

    // The offset is the in-plane difference between the projected source and the target.
    const Eigen::Vector3f offset = projected - target_pt.template head<3>();

    // The color residual is the difference between the source and target colors,
    // compensated by the geometric misalignment in the tangent plane.
    // residual = (c_s - c_t) + G_t * (p_proj - p_t)
    const Eigen::Vector3f color_diff = (source_rgb - target_rgb).template head<3>();
    const Eigen::Vector3f residual_color =
        color_diff + eigen_utils::multiply<3, 3, 1>(target_rgb_grad, offset);  // G_t * offset

    // SE(3) Jacobian of the source point
    const Eigen::Matrix<float, 3, 6> J_geo = compute_se3_jacobian(T, source_pt).template block<3, 6>(0, 0);

    // Color Jacobian: gradient * J_geo
    const Eigen::Matrix<float, 3, 6> J_color = eigen_utils::multiply<3, 3, 6>(target_rgb_grad, J_geo);

    LinearlizedResult ret;
    // H = J.T * J
    // b = J.T * residual
    // error = 0.5 * norm(residual)
    ret.H = eigen_utils::multiply<6, 3, 6>(eigen_utils::transpose<3, 6>(J_color), J_color);
    ret.b = eigen_utils::multiply<6, 3, 1>(eigen_utils::transpose<3, 6>(J_color), residual_color);
    ret.error = 0.5f * eigen_utils::frobenius_norm_squared<3>(residual_color);
    ret.inlier = 1;

    return ret;
}

/// @brief Evaluate photometric error including geometric correction
/// @param T SE(3) transform applied to the source point
/// @param source_pt Source point before transformation
/// @param target_pt Target point associated with the source point
/// @param source_rgb Color observed at the source point
/// @param target_rgb Color observed at the target point
/// @param target_rgb_grad Spatial gradient of the target color
SYCL_EXTERNAL inline float calculate_color_error(const std::array<sycl::float4, 4>& T,    ///< SE(3) transform
                                                 const PointType& source_pt,              ///< Source point
                                                 const PointType& target_pt,              ///< Target point
                                                 const RGBType& source_rgb,               ///< Source color
                                                 const RGBType& target_rgb,               ///< Target color
                                                 const Normal& target_normal,             ///< Target normal
                                                 const ColorGradient& target_rgb_grad) {  ///< Target RGB gradient
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);

    const Eigen::Vector3f geometric_residual = (transform_source - target_pt).template head<3>();

    const Eigen::Vector3f projected =
        transform_source.template head<3>() -
        target_normal.template head<3>() * eigen_utils::dot<3>(geometric_residual, target_normal.template head<3>());
    const Eigen::Vector3f offset = projected - target_pt.template head<3>();
    const Eigen::Vector3f color_diff = (source_rgb - target_rgb).template head<3>();
    const Eigen::Vector3f residual_color = color_diff + eigen_utils::multiply<3, 3, 1>(target_rgb_grad, offset);

    return 0.5f * eigen_utils::frobenius_norm_squared<3>(residual_color);
}

/// @brief Robust Linearlization
/// @tparam icp icp type
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @param target_normal Target normal
/// @param source_rgb Source RGB
/// @param target_rgb Target RGB
/// @param target_rgb_grad Target RGB gradient
/// @param photometric_weight Photometric term weight (0.0 ~ 1.0). 0.0 is geometric term only
/// @return linearlized result
template <ICPType icp = ICPType::GICP>
SYCL_EXTERNAL inline LinearlizedResult linearlize(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                                  const Covariance& source_cov, const PointType& target_pt,
                                                  const Covariance& target_cov, const Normal& target_normal,
                                                  const RGBType& source_rgb, const RGBType& target_rgb,
                                                  const ColorGradient& target_grad, float photometric_weight) {
    float geo_residual_norm = 0.0f;
    LinearlizedResult result =
        linearlize_geometry<icp>(T, source_pt, source_cov, target_pt, target_cov, target_normal, geo_residual_norm);

    float total_error = result.error;

    if (0.0f < photometric_weight && photometric_weight <= 1.0f) {
        const float geo_weight = 1.0f - photometric_weight;
        auto color_result =
            linearlize_color(T, source_pt, target_pt, source_rgb, target_rgb, target_normal, target_grad);

        eigen_utils::multiply_inplace<6, 6>(result.H, geo_weight);
        eigen_utils::multiply_inplace<6, 6>(color_result.H, photometric_weight);
        eigen_utils::add_inplace<6, 6>(result.H, color_result.H);

        eigen_utils::multiply_inplace<6, 1>(result.b, geo_weight);
        eigen_utils::multiply_inplace<6, 1>(color_result.b, photometric_weight);
        eigen_utils::add_inplace<6, 1>(result.b, color_result.b);

        total_error = sycl::fma(geo_weight, result.error, photometric_weight * color_result.error);
    }

    result.error = sycl::sqrt(2.0f * total_error);

    return result;
}

/// @brief Compute Error
/// @tparam icp icp type
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @param target_normal Target normal
/// @param source_rgb Source RGB
/// @param target_rgb Target RGB
/// @param target_rgb_grad Target RGB gradient
/// @param photometric_weight Photometric term weight (0.0 ~ 1.0). 0.0 is geometric term only
/// @return error
template <ICPType icp = ICPType::GICP>
SYCL_EXTERNAL inline float calculate_error(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                           const Covariance& source_cov, const PointType& target_pt,
                                           const Covariance& target_cov, const Normal& target_normal,
                                           const RGBType& source_rgb, const RGBType& target_rgb,
                                           const ColorGradient& target_rgb_grad, float photometric_weight) {
    const float geo_error =
        calculate_geometry_error<icp>(T, source_pt, source_cov, target_pt, target_cov, target_normal);

    float total_error = geo_error;
    if (0.0f < photometric_weight && photometric_weight <= 1.0f) {
        const float geo_weight = 1.0f - photometric_weight;
        const float color_error =
            calculate_color_error(T, source_pt, target_pt, source_rgb, target_rgb, target_normal, target_rgb_grad);
        total_error = sycl::fma(geo_weight, geo_error, photometric_weight * color_error);
    }
    return sycl::sqrt(2.0f * total_error);
}

}  // namespace kernel

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
