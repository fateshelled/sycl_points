#pragma once

#include <sycl_points/algorithms/registration/linearized_result.hpp>
#include <sycl_points/algorithms/registration/result.hpp>
#include <sycl_points/algorithms/registration/robust.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/types.hpp>
#include <sycl_points/utils/eigen_utils.hpp>
#include <tuple>
#include <type_traits>

namespace sycl_points {

namespace algorithms {

namespace registration {

enum class RegType {
    POINT_TO_POINT = 0,  //
    POINT_TO_PLANE,
    /// @brief Generalized-ICP
    /// @authors Aleksandr V. Segal, Dirk Haehnel, Sebastian Thrun
    GICP,
    /// @brief GenZ-ICP: Generalizable and Degeneracy-Robust LiDAR Odometry Using an Adaptive Weighting (2024)
    /// @authors Daehan Lee, Hyungtae Lim, Soohee Han
    /// @cite https://arxiv.org/abs/2411.06766
    GENZ,
    /// @brief Point-to-Distribution ICP
    /// @note Uses Mahalanobis distance with target covariance only (source points have no covariance)
    POINT_TO_DISTRIBUTION
};

/// @brief Registration Type tags
using RegTypeTags = std::tuple<                                        //
    std::integral_constant<RegType, RegType::POINT_TO_POINT>,          //
    std::integral_constant<RegType, RegType::POINT_TO_PLANE>,          //
    std::integral_constant<RegType, RegType::GICP>,                    //
    std::integral_constant<RegType, RegType::GENZ>,                    //
    std::integral_constant<RegType, RegType::POINT_TO_DISTRIBUTION>>;  //

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

/// @brief Weighting coefficients used to mix geometric and photometric modalities.
struct PhotometricWeights {
    float geometry = 1.0f;   ///< Weight applied to the geometric term
    float color = 0.0f;      ///< Weight applied to the RGB photometric term
    float intensity = 0.0f;  ///< Weight applied to the intensity photometric term
};

/// @brief Compute weighting coefficients for the active photometric modalities.
/// @param photometric_weight Total weight allocated to photometric terms.
/// @param use_color True when RGB photometric residuals are available.
/// @param use_intensity True when intensity photometric residuals are available.
/// @return Normalized weights for geometry, color, and intensity contributions.
inline PhotometricWeights compute_photometric_weights(float photometric_weight, bool use_color, bool use_intensity) {
    PhotometricWeights weights{};

    if (0.0f < photometric_weight && photometric_weight <= 1.0f) {
        const uint32_t modality_count = static_cast<uint32_t>(use_color) + static_cast<uint32_t>(use_intensity);

        if (modality_count > 0U) {
            const float per_modality_weight = photometric_weight / static_cast<float>(modality_count);
            weights.geometry = 1.0f - photometric_weight;
            weights.color = use_color ? per_modality_weight : 0.0f;
            weights.intensity = use_intensity ? per_modality_weight : 0.0f;
        }
    }

    return weights;
}

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
SYCL_EXTERNAL inline LinearizedResult linearize_point_to_point(const std::array<sycl::float4, 4>& T,
                                                               const PointType& source_pt, const PointType& target_pt,
                                                               float& residual_norm) {
    PointType transform_source;
    transform::kernel::transform_point(source_pt, transform_source, T);

    const PointType residual(target_pt.x() - transform_source.x(), target_pt.y() - transform_source.y(),
                             target_pt.z() - transform_source.z(), 0.0f);
    const Eigen::Matrix<float, 4, 6> J = compute_weighted_se3_jacobian(T, source_pt, Eigen::Matrix4f::Identity());

    LinearizedResult ret;
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
/// @return linearized result
SYCL_EXTERNAL inline LinearizedResult linearize_point_to_plane(const std::array<sycl::float4, 4>& T,
                                                               const PointType& source_pt, const PointType& target_pt,
                                                               const Normal& target_normal, float& residual_norm) {
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

    LinearizedResult ret;
    const auto J_T = eigen_utils::transpose<4, 6>(J);
    ret.H = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(J_T, J));
    ret.b = eigen_utils::multiply<6, 4>(J_T, plane_error);

    // Scalar point-to-plane error uses the squared projection length.
    const float squared_norm = projected_residual * projected_residual;
    residual_norm = sycl::fabs(projected_residual);
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

    // Error is the squared projection of the residual onto the target normal.
    const float projected_residual = eigen_utils::dot<3>(target_normal.head<3>(), residual.head<3>());
    return 0.5f * projected_residual * projected_residual;
}

/// @brief Generalized Iterative Closest Point (GICP)
/// @param T transform matrix
/// @param source_pt Source Point
/// @param source_cov Source covariance
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return linearized result
SYCL_EXTERNAL inline LinearizedResult linearize_gicp(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
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

    LinearizedResult ret;
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
SYCL_EXTERNAL inline LinearizedResult linearize_point_to_distribution(const std::array<sycl::float4, 4>& T,
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

    LinearizedResult ret;
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

/// @brief Error for Point-to-Distribution ICP
/// @param T transform matrix
/// @param source_pt Source Point (coordinate only, no covariance)
/// @param target_pt Target point
/// @param target_cov Target covariance
/// @return error
SYCL_EXTERNAL inline float calculate_point_to_distribution_error(const std::array<sycl::float4, 4>& T,
                                                                  const PointType& source_pt,
                                                                  const PointType& target_pt,
                                                                  const Covariance& target_cov) {
    PointType transform_source_pt;
    transform::kernel::transform_point(source_pt, transform_source_pt, T);

    const Covariance mahalanobis = compute_target_mahalanobis(target_cov);

    const PointType residual(target_pt.x() - transform_source_pt.x(), target_pt.y() - transform_source_pt.y(),
                             target_pt.z() - transform_source_pt.z(), 0.0f);
    return 0.5f * (eigen_utils::dot<4>(residual, eigen_utils::multiply<4, 4>(mahalanobis, residual)));
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
SYCL_EXTERNAL inline LinearizedResult linearize_geometry(const std::array<sycl::float4, 4>& T,  //
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

        LinearizedResult result;
        result.H = eigen_utils::add<6, 6>(eigen_utils::multiply<6, 6>(pt2pt_result.H, 1.0f - genz_alpha),
                                          eigen_utils::multiply<6, 6>(pt2pl_result.H, genz_alpha));
        result.b = eigen_utils::add<6, 1>(eigen_utils::multiply<6, 1>(pt2pt_result.b, 1.0f - genz_alpha),
                                          eigen_utils::multiply<6, 1>(pt2pl_result.b, genz_alpha));
        result.error = pt2pt_result.error * (1.0f - genz_alpha) + pt2pl_result.error * genz_alpha;
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
SYCL_EXTERNAL inline LinearizedResult linearize_color(const std::array<sycl::float4, 4>& T,    ///< SE(3) transform
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

    LinearizedResult ret;
    // H = J.T * J
    // b = J.T * residual
    // error = 0.5 * norm(residual)
    ret.H = eigen_utils::multiply<6, 3, 6>(eigen_utils::transpose<3, 6>(J_color), J_color);
    ret.b = eigen_utils::multiply<6, 3, 1>(eigen_utils::transpose<3, 6>(J_color), residual_color);
    ret.error = 0.5f * eigen_utils::frobenius_norm_squared<3>(residual_color);
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
SYCL_EXTERNAL inline LinearizedResult linearize_intensity(
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

    LinearizedResult ret;
    // H = J.T * J
    ret.H = eigen_utils::multiply<6, 1, 6>(J_intensity_T, J_intensity);
    // b = J.T * residual
    ret.b = eigen_utils::multiply<6>(J_intensity_T, residual_intensity);
    // error = 0.5 * residual^2
    ret.error = 0.5f * residual_intensity * residual_intensity;
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
SYCL_EXTERNAL inline float calculate_intensity_error(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                                     const PointType& target_pt, float source_intensity,
                                                     float target_intensity, const Normal& target_normal,
                                                     const IntensityGradient& target_intensity_grad) {
    const Eigen::Vector3f offset = compute_tangent_plane_offset(T, source_pt, target_pt, target_normal);
    const float intensity_diff = source_intensity - target_intensity;
    const float residual_intensity = intensity_diff + eigen_utils::dot<3>(target_intensity_grad, offset);

    return 0.5f * residual_intensity * residual_intensity;
}

/// @brief Robust Linearization
/// @tparam reg registration type
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
/// @return linearized result
template <RegType reg = RegType::GICP>
SYCL_EXTERNAL inline LinearizedResult linearize(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                                const Covariance& source_cov, const PointType& target_pt,
                                                const Covariance& target_cov, const Normal& target_normal,
                                                const RGBType& source_rgb, const RGBType& target_rgb,
                                                const ColorGradient& target_grad, bool use_color,
                                                float source_intensity, float target_intensity,
                                                const IntensityGradient& target_intensity_grad, bool use_intensity,
                                                float photometric_weight, float genz_alpha) {
    float geo_residual_norm = 0.0f;
    LinearizedResult result = linearize_geometry<reg>(T, source_pt, source_cov, target_pt, target_cov, target_normal,
                                                      geo_residual_norm, genz_alpha);

    float total_error = result.error;
    const PhotometricWeights weights = compute_photometric_weights(photometric_weight, use_color, use_intensity);
    const bool photometric_active = (weights.color > 0.0f) || (weights.intensity > 0.0f);

    if (photometric_active) {
        // Blend geometric and photometric Hessians/gradients with the assigned weights.
        eigen_utils::multiply_inplace<6, 6>(result.H, weights.geometry);
        eigen_utils::multiply_inplace<6, 1>(result.b, weights.geometry);
        total_error = weights.geometry * result.error;

        if (weights.color > 0.0f) {
            auto color_result =
                linearize_color(T, source_pt, target_pt, source_rgb, target_rgb, target_normal, target_grad);

            eigen_utils::multiply_inplace<6, 6>(color_result.H, weights.color);
            eigen_utils::add_inplace<6, 6>(result.H, color_result.H);

            eigen_utils::multiply_inplace<6, 1>(color_result.b, weights.color);
            eigen_utils::add_inplace<6, 1>(result.b, color_result.b);

            total_error = sycl::fma(weights.color, color_result.error, total_error);
        }

        if (weights.intensity > 0.0f) {
            auto intensity_result = linearize_intensity(T, source_pt, target_pt, source_intensity, target_intensity,
                                                        target_normal, target_intensity_grad);

            eigen_utils::multiply_inplace<6, 6>(intensity_result.H, weights.intensity);
            eigen_utils::add_inplace<6, 6>(result.H, intensity_result.H);

            eigen_utils::multiply_inplace<6, 1>(intensity_result.b, weights.intensity);
            eigen_utils::add_inplace<6, 1>(result.b, intensity_result.b);

            total_error = sycl::fma(weights.intensity, intensity_result.error, total_error);
        }
    }

    result.error = sycl::sqrt(2.0f * total_error);

    return result;
}

/// @brief Compute Error
/// @tparam reg registration type
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
template <RegType reg = RegType::GICP>
SYCL_EXTERNAL inline float calculate_error(const std::array<sycl::float4, 4>& T, const PointType& source_pt,
                                           const Covariance& source_cov, const PointType& target_pt,
                                           const Covariance& target_cov, const Normal& target_normal,
                                           const RGBType& source_rgb, const RGBType& target_rgb,
                                           const ColorGradient& target_rgb_grad, bool use_color, float source_intensity,
                                           float target_intensity, const IntensityGradient& target_intensity_grad,
                                           bool use_intensity, float photometric_weight, float genz_alpha) {
    const float geo_error =
        calculate_geometry_error<reg>(T, source_pt, source_cov, target_pt, target_cov, target_normal, genz_alpha);

    float total_error = geo_error;
    const PhotometricWeights weights = compute_photometric_weights(photometric_weight, use_color, use_intensity);
    const bool photometric_active = (weights.color > 0.0f) || (weights.intensity > 0.0f);

    if (photometric_active) {
        total_error = weights.geometry * geo_error;

        if (weights.color > 0.0f) {
            const float color_error =
                calculate_color_error(T, source_pt, target_pt, source_rgb, target_rgb, target_normal, target_rgb_grad);
            total_error = sycl::fma(weights.color, color_error, total_error);
        }

        if (weights.intensity > 0.0f) {
            const float intensity_error = calculate_intensity_error(
                T, source_pt, target_pt, source_intensity, target_intensity, target_normal, target_intensity_grad);
            total_error = sycl::fma(weights.intensity, intensity_error, total_error);
        }
    }
    return sycl::sqrt(2.0f * total_error);
}

}  // namespace kernel

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
