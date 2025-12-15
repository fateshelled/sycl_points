#pragma once

#include <Eigen/Dense>

namespace sycl_points {

namespace algorithms {

namespace registration {

/// @brief Registration Result
struct RegistrationResult {
    using Ptr = std::shared_ptr<RegistrationResult>;

    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();                // Estimated transformation
    bool converged = false;                                             // Optimization converged or not
    size_t iterations = 0;                                              // Number of optimization iterations
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();  // Final hessian
    Eigen::Vector<float, 6> b = Eigen::Vector<float, 6>::Zero();        // Final gradient
    float error = std::numeric_limits<float>::max();                    // Final error
    uint32_t inlier = 0;                                                // Inlier point num;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
