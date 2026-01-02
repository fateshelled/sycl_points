#pragma once

#include <Eigen/Dense>
#include <limits>

namespace sycl_points {

namespace algorithms {

namespace registration {
/// @brief Registration Linearized Result
struct LinearizedResult {
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

/// @brief Registration Linearized Result on device kernel
struct LinearizedKernelResult{
    /// @brief Hessian, Information Matrix
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    /// @brief Gradient, Information Vector
    Eigen::Vector<float, 6> b = Eigen::Vector<float, 6>::Zero();
    /// @brief Squared error value
    float squared_error = std::numeric_limits<float>::max();
    /// @brief inlier point num
    uint32_t inlier = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
