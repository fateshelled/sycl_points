#pragma once

#include "sycl_points/algorithms/registration/factor.hpp"
#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/algorithms/registration/result.hpp"
#include "sycl_points/algorithms/transform.hpp"
#include "sycl_points/points/types.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {
namespace kernel {

/// @brief Compute the tangent plane offset between the transformed source and target points.
/// @param T SE(3) transform applied to the source point
/// @param source_pt Source point before transformation
/// @param target_pt Target point associated with the source point
/// @param target_normal Target surface normal defining the tangent plane
/// @return Offset vector lying on the tangent plane from target to projected source
SYCL_EXTERNAL inline Eigen::Vector3f compute_tangent_plane_offset(const std::array<sycl::float4, 4>& T,
                                                                  const PointType& source_pt,
                                                                  const PointType& target_pt,
                                                                  const Normal& target_normal) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);
    const Eigen::Vector3f geometric_residual = (transform_source - target_pt).template head<3>();

    const Eigen::Vector3f projected =
        transform_source.template head<3>() -
        target_normal.template head<3>() * eigen_utils::dot<3>(geometric_residual, target_normal.template head<3>());

    return projected - target_pt.template head<3>();
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
SYCL_EXTERNAL inline LinearizedKernelResult linearize_color(
    const std::array<sycl::float4, 4>& T,    ///< SE(3) transform
    const PointType& source_pt,              ///< Source point
    const PointType& target_pt,              ///< Target point
    const RGBType& source_rgb,               ///< Source color
    const RGBType& target_rgb,               ///< Target color
    const Normal& target_normal,             ///< Target normal
    const ColorGradient& target_rgb_grad) {  ///< Target RGB gradient

    // Offset between the projected point and the target point on the tangent plane.
    const Eigen::Vector3f offset = compute_tangent_plane_offset(T, source_pt, target_pt, target_normal);

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

    LinearizedKernelResult ret;
    // H = J.T * J
    // b = J.T * residual
    ret.H = eigen_utils::multiply<6, 3, 6>(eigen_utils::transpose<3, 6>(J_color), J_color);
    ret.b = eigen_utils::multiply<6, 3, 1>(eigen_utils::transpose<3, 6>(J_color), residual_color);
    ret.squared_error = eigen_utils::frobenius_norm_squared<3>(residual_color);
    ret.inlier = 1;

    return ret;
}

/// @brief Linearize intensity residual using intensity gradient
/// @note This follows the same geometric decoupling strategy used for RGB photometric residuals.
/// @param T SE(3) transform applied to the source point
/// @param source_pt Source point before transformation
/// @param target_pt Target point associated with the source point
/// @param source_intensity Intensity observed at the source point
/// @param target_intensity Intensity observed at the target point
/// @param target_normal Target surface normal
/// @param target_intensity_grad Spatial gradient of the target intensity
SYCL_EXTERNAL inline LinearizedKernelResult linearize_intensity(
    const std::array<sycl::float4, 4>& T,           ///< SE(3) transform
    const PointType& source_pt,                     ///< Source point
    const PointType& target_pt,                     ///< Target point
    float source_intensity,                         ///< Source intensity
    float target_intensity,                         ///< Target intensity
    const Normal& target_normal,                    ///< Target normal
    const IntensityGradient& target_intensity_grad  ///< Target intensity gradient
) {
    // Offset between the projected point and the target point on the tangent plane
    const Eigen::Vector3f offset = compute_tangent_plane_offset(T, source_pt, target_pt, target_normal);

    // Intensity residual compensated by the spatial gradient on the tangent plane
    const float intensity_diff = source_intensity - target_intensity;
    const float residual_intensity = intensity_diff + eigen_utils::dot<3>(target_intensity_grad, offset);

    // SE(3) Jacobian of the source point
    const Eigen::Matrix<float, 3, 6> J_geo = compute_se3_jacobian(T, source_pt).template block<3, 6>(0, 0);

    // Intensity Jacobian: gradient (row vector) * J_geo
    const Eigen::Matrix<float, 1, 3> grad_row = target_intensity_grad.transpose();
    const Eigen::Matrix<float, 1, 6> J_intensity = eigen_utils::multiply<1, 3, 6>(grad_row, J_geo);
    const Eigen::Matrix<float, 6, 1> J_intensity_T = eigen_utils::transpose<1, 6>(J_intensity);

    LinearizedKernelResult ret;
    // H = J.T * J
    ret.H = eigen_utils::multiply<6, 1, 6>(J_intensity_T, J_intensity);
    // b = J.T * residual
    ret.b = eigen_utils::multiply<6>(J_intensity_T, residual_intensity);
    ret.squared_error = residual_intensity * residual_intensity;
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
/// @return Squared error value
SYCL_EXTERNAL inline float calculate_color_error(const std::array<sycl::float4, 4>& T,    ///< SE(3) transform
                                                 const PointType& source_pt,              ///< Source point
                                                 const PointType& target_pt,              ///< Target point
                                                 const RGBType& source_rgb,               ///< Source color
                                                 const RGBType& target_rgb,               ///< Target color
                                                 const Normal& target_normal,             ///< Target normal
                                                 const ColorGradient& target_rgb_grad) {  ///< Target RGB gradient
    const Eigen::Vector3f offset = compute_tangent_plane_offset(T, source_pt, target_pt, target_normal);
    const Eigen::Vector3f color_diff = (source_rgb - target_rgb).template head<3>();
    const Eigen::Vector3f residual_color = color_diff + eigen_utils::multiply<3, 3, 1>(target_rgb_grad, offset);

    return 0.5f * eigen_utils::frobenius_norm_squared<3>(residual_color);
}

/// @brief Evaluate intensity photometric error including geometric correction
/// @param T SE(3) transform applied to the source point
/// @param source_pt Source point before transformation
/// @param target_pt Target point associated with the source point
/// @param source_intensity Intensity observed at the source point
/// @param target_intensity Intensity observed at the target point
/// @param target_normal Target surface normal
/// @param target_intensity_grad Spatial gradient of the target intensity
/// @return Squared error value
SYCL_EXTERNAL inline float calculate_intensity_error(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                                     const PointType& target_pt, float source_intensity,
                                                     float target_intensity, const Normal& target_normal,
                                                     const IntensityGradient& target_intensity_grad) {
    const Eigen::Vector3f offset = compute_tangent_plane_offset(T, source_pt, target_pt, target_normal);
    const float intensity_diff = source_intensity - target_intensity;
    const float residual_intensity = intensity_diff + eigen_utils::dot<3>(target_intensity_grad, offset);

    return residual_intensity * residual_intensity;
}

}  // namespace kernel

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
