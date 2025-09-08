#pragma once

#include <array>

#include <sycl_points/algorithms/registration_factor.hpp>
#include <sycl_points/points/types.hpp>

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Linearized result of color term
struct ColorLinearized {
    /// @brief Hessian, Information Matrix
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    /// @brief Gradient, Information Vector
    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
    /// @brief Error value
    float error = 0.f;
    /// @brief Inlier point count
    uint32_t inlier = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

namespace kernel {

/// @brief Linearize color residual using RGB gradient
SYCL_EXTERNAL inline ColorLinearized linearlize_color(
    const std::array<sycl::float4, 4>& T,      ///< SE(3) transform
    const PointType& source_pt,                ///< Source point
    const RGBType& source_rgb,                 ///< Source color
    const RGBType& target_rgb,                 ///< Target color
    const ColorGradient& target_rgb_grad,      ///< Target RGB gradient
    float& residual_norm)                      ///< Output residual norm
{
    // Compute color residual (target - source)
    float residual[3] = {
        target_rgb.x() - source_rgb.x(),
        target_rgb.y() - source_rgb.y(),
        target_rgb.z() - source_rgb.z()};

    // SE(3) Jacobian of the source point (first 3 rows)
    const Eigen::Matrix<float, 4, 6> J_basic =
        compute_se3_jacobian(T, source_pt);
    float J_geo[3][6];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 6; ++c) {
            J_geo[r][c] = J_basic(r, c);
        }
    }

    // Color Jacobian: gradient * J_geo
    float J_color[3][6] = {};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 6; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                sum = sycl::fma(target_rgb_grad(r, k), J_geo[k][c], sum);
            }
            J_color[r][c] = sum;
        }
    }

    // Assemble Hessian and gradient
    float H[6][6];
    float b[6];
    for (int i = 0; i < 6; ++i) {
        b[i] = 0.0f;
        for (int j = i; j < 6; ++j) {
            float h = 0.0f;
            for (int k = 0; k < 3; ++k) {
                h = sycl::fma(J_color[k][i], J_color[k][j], h);
            }
            H[i][j] = h;
            H[j][i] = h;
        }
        for (int k = 0; k < 3; ++k) {
            b[i] = sycl::fma(J_color[k][i], residual[k], b[i]);
        }
    }

    // Compute residual norm and error
    const float squared_norm = residual[0] * residual[0] +
                               residual[1] * residual[1] +
                               residual[2] * residual[2];
    residual_norm = sycl::sqrt(squared_norm);

    // Fill return struct
    ColorLinearized ret;
    for (int r = 0; r < 6; ++r) {
        ret.b(r, 0) = b[r];
        for (int c = 0; c < 6; ++c) {
            ret.H(r, c) = H[r][c];
        }
    }
    ret.error = 0.5f * squared_norm;
    ret.inlier = 1;

    return ret;
}

}  // namespace kernel

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points

