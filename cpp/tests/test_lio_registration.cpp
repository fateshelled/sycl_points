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
    factor.H(imu::State::kIdxRot, imu::State::kIdxRot) = 1.0f;
    factor.b(imu::State::kIdxPos) = 10.0f;
    factor.b(imu::State::kIdxPos + 1) = 20.0f;
    factor.b(imu::State::kIdxRot) = 10.0f;

    lio::DirectionalIcpWeightingParams params;
    params.trans_min_eigenvalue_per_inlier = 0.05f;  // threshold = 5 for 100 inliers
    params.rot_min_eigenvalue_per_inlier = 0.05f;
    params.trans_weak_direction_scale = 0.1f;
    params.rot_weak_direction_scale = 0.25f;

    lio::apply_directional_icp_weighting(factor, params);

    // Translation uses the linear information ratio (1 / 5 = 0.2), while
    // rotation is clamped to its larger minimum scale (0.25).
    EXPECT_NEAR(factor.H(imu::State::kIdxPos, imu::State::kIdxPos), 0.2f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos), 2.0f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1), 20.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos + 1), 20.0f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxRot, imu::State::kIdxRot), 0.25f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxRot), 2.5f, kEps);
}

TEST(LioRegistration, DirectionalIcpWeightingPreservesCoupledFactorStructure) {
    lio::LIOLinearizedResult factor;
    factor.inlier = 100;
    factor.H(imu::State::kIdxPos, imu::State::kIdxPos) = 1.0f;
    factor.H(imu::State::kIdxRot, imu::State::kIdxRot) = 1.0f;
    factor.H(imu::State::kIdxPos, imu::State::kIdxRot) = 0.5f;
    factor.H(imu::State::kIdxRot, imu::State::kIdxPos) = 0.5f;

    lio::DirectionalIcpWeightingParams params;
    params.trans_min_eigenvalue_per_inlier = 0.1f;
    params.rot_min_eigenvalue_per_inlier = 0.1f;
    params.trans_weak_direction_scale = 0.1f;
    params.rot_weak_direction_scale = 0.4f;

    lio::apply_directional_icp_weighting(factor, params);

    EXPECT_NEAR(factor.H(imu::State::kIdxPos, imu::State::kIdxPos), 0.1f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxRot, imu::State::kIdxRot), 0.4f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxPos, imu::State::kIdxRot), 0.1f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxRot, imu::State::kIdxPos), 0.1f, kEps);
    const Eigen::Matrix<float, 6, 6> H_pose = factor.H.block<6, 6>(0, 0);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> solver(H_pose);
    EXPECT_TRUE(H_pose.isApprox(H_pose.transpose(), kEps));
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GE(solver.eigenvalues().minCoeff(), -kEps);
}
