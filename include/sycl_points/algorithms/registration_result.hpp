#pragma once

#include <Eigen/Dense>

namespace sycl_points {

namespace algorithms {

namespace registration {

/// @brief Registration Result
struct RegistrationResult {
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();                // Estimated transformation
    bool converged = false;                                             // Optimization converged or not
    size_t iterations = 0;                                              // Number of opmitization iterations
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();  // Final hessian
    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();  // Final gradient
    float error = std::numeric_limits<float>::max();                    // Final error

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
