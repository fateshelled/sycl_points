#pragma once

#include <Eigen/Dense>
#include <cstdint>

namespace sycl_points {
namespace algorithms {
namespace lio {

/// @brief Accumulated Gauss-Newton normal equation for one LIO iteration.
///
/// Holds the combined 15×15 Hessian H and 15×1 gradient b that result from
/// accumulating one or more factors (ICP geometry, IMU prior). After all
/// factors have been added, pass H and b to solve_ldlt() to obtain the state
/// update δx and the posterior covariance P_post = H⁻¹.
///
/// The normal equation is formulated as:
///
///   H · δx = −b
///
/// where:
///   H = Σ Jᵢᵀ · Ωᵢ · Jᵢ   (sum of information-weighted outer products)
///   b = Σ Jᵢᵀ · Ωᵢ · rᵢ   (sum of information-weighted residuals)
///
/// Diagnostic fields (error_icp, error_imu, inlier) are for logging only.
struct LIOLinearizedResult {
    Eigen::Matrix<float, 15, 15> H = Eigen::Matrix<float, 15, 15>::Zero();  ///< Combined Hessian
    Eigen::Matrix<float, 15, 1> b = Eigen::Matrix<float, 15, 1>::Zero();    ///< Combined gradient
    float error_icp = 0.0f;                                                 ///< Accumulated ICP cost (for diagnostics)
    float error_imu = 0.0f;                                                 ///< IMU prior cost (for diagnostics)
    uint32_t inlier = 0;                                                    ///< Number of valid ICP correspondences

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
