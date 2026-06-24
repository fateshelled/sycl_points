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

    Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Identity();

    lio::DirectionalIcpWeightingParams params;
    params.trans_min_eigenvalue_per_inlier = 0.05f;  // threshold = 5 for 100 inliers
    params.rot_min_eigenvalue_per_inlier = 0.0f;
    params.weak_direction_scale = 0.1f;
    params.max_icp_to_imu_ratio = 0.0f;

    lio::apply_directional_icp_weighting(factor, H_imu, params);

    EXPECT_NEAR(factor.H(imu::State::kIdxPos, imu::State::kIdxPos), 0.1f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos), 1.0f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1), 20.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos + 1), 20.0f, kEps);
}

TEST(LioRegistration, DirectionalIcpWeightingCapsInformationRelativeToImu) {
    lio::LIOLinearizedResult factor;
    factor.inlier = 100;
    factor.H(imu::State::kIdxRot, imu::State::kIdxRot) = 1000.0f;
    factor.b(imu::State::kIdxRot) = 100.0f;

    Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Zero();
    H_imu(imu::State::kIdxRot, imu::State::kIdxRot) = 10.0f;

    lio::DirectionalIcpWeightingParams params;
    params.trans_min_eigenvalue_per_inlier = 0.0f;
    params.rot_min_eigenvalue_per_inlier = 0.0f;
    params.max_icp_to_imu_ratio = 2.0f;
    params.imu_information_floor = 0.0f;

    lio::apply_directional_icp_weighting(factor, H_imu, params);

    EXPECT_NEAR(factor.H(imu::State::kIdxRot, imu::State::kIdxRot), 20.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxRot), 2.0f, kEps);
}
