#pragma once

#include <Eigen/Dense>

#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/registration/result.hpp"

namespace sycl_points {
namespace algorithms {
namespace lio {

struct LIORegistrationResult {
    registration::RegistrationResult registration_result;
    imu::State state;
    Eigen::Matrix<float, 15, 15> posterior_covariance = Eigen::Matrix<float, 15, 15>::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
