#pragma once

#include <Eigen/Dense>

#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/registration/result.hpp"

namespace sycl_points {
namespace algorithms {
namespace lio {

enum class LIORegistrationStatus {
    success,
    no_progress,
    numeric_failure,
    invalid_imu,
};

struct LIORegistrationResult {
    LIORegistrationStatus status = LIORegistrationStatus::numeric_failure;
    registration::RegistrationResult registration_result;
    imu::State state;
    Eigen::Matrix<float, 15, 15> posterior_covariance = Eigen::Matrix<float, 15, 15>::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool valid() const {
        return status == LIORegistrationStatus::success || status == LIORegistrationStatus::no_progress;
    }
};

}  // namespace lio
}  // namespace algorithms
}  // namespace sycl_points
