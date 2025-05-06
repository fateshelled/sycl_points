#pragma once

#include <Eigen/Dense>

namespace sycl_points {

namespace algorithms {

namespace registration {

struct LinearlizedResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
    float error = std::numeric_limits<float>::max();
};

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
