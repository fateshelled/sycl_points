#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "sycl_points/algorithms/lio/lio_registration.hpp"

namespace lio = sycl_points::algorithms::lio;
namespace imu = sycl_points::imu;

static constexpr float kEps = 1e-5f;

TEST(LioRegistration, DirectionalIcpWeightingAttenuatesWeakDirections) {
    lio::LIOLinearizedResult factor;
    factor.inlier = 100;
    factor.H(imu::State::kIdxPos, imu::State::kIdxPos) = 1.0f;
    factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1) = 20.0f;
    factor.b(imu::State::kIdxPos) = 10.0f;
    factor.b(imu::State::kIdxPos + 1) = 20.0f;

    lio::DirectionalIcpWeightingParams params;
    params.trans_min_eigenvalue_per_inlier = 0.05f;  // threshold = 5 for 100 inliers
    params.rot_min_eigenvalue_per_inlier = 0.0f;
    params.weak_direction_scale = 0.1f;

    lio::apply_directional_icp_weighting(factor, params);

    EXPECT_NEAR(factor.H(imu::State::kIdxPos, imu::State::kIdxPos), 0.1f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos), 1.0f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1), 20.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos + 1), 20.0f, kEps);
}
