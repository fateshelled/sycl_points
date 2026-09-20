#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <limits>

#include "sycl_points/algorithms/lio/lio_registration.hpp"

namespace lio = sycl_points::algorithms::lio;
namespace imu = sycl_points::imu;

static constexpr float kEps = 1e-5f;

TEST(LioRegistration, DirectionalIcpWeightingTypeConversion) {
    EXPECT_EQ(lio::DirectionalIcpWeightingType_from_string("scale"), lio::DirectionalIcpWeightingType::scale);
    EXPECT_EQ(lio::DirectionalIcpWeightingType_from_string("TSVD"), lio::DirectionalIcpWeightingType::tsvd);
    EXPECT_THROW(lio::DirectionalIcpWeightingType_from_string("unknown"), std::runtime_error);
}

TEST(LioRegistration, DirectionalIcpWeightingAttenuatesWeakDirections) {
    lio::LIOLinearizedResult factor;
    factor.inlier = 100;
    factor.H(imu::State::kIdxPos, imu::State::kIdxPos) = 1.0f;
    factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1) = 20.0f;
    factor.H(imu::State::kIdxRot, imu::State::kIdxRot) = 1.0f;
    factor.b(imu::State::kIdxPos) = 10.0f;
    factor.b(imu::State::kIdxPos + 1) = 20.0f;
    factor.b(imu::State::kIdxRot) = 10.0f;

    // IMU prior supplies 5.0 per inlier along translation x, so the threshold for
    // ratio 1.0 is 500; the ICP eigenvalue of 1.0 (per inlier 0.01) is weak.
    // Rotation has no IMU information and no floor, so the zero threshold keeps it.
    Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Zero();
    H_imu(imu::State::kIdxPos, imu::State::kIdxPos) = 500.0f;

    lio::DirectionalIcpWeightingParams params;
    params.trans_min_information_ratio = 1.0f;
    params.rot_min_information_ratio = 1.0f;
    params.trans_imu_information_floor_per_inlier = 0.0f;
    params.rot_imu_information_floor_per_inlier = 0.0f;
    params.trans_weak_direction_scale = 0.1f;
    params.rot_weak_direction_scale = 0.25f;

    lio::apply_directional_icp_weighting(factor, H_imu, params);

    // Translation uses the linear information ratio (1 / 500), clamped to the 0.1 floor.
    EXPECT_NEAR(factor.H(imu::State::kIdxPos, imu::State::kIdxPos), 0.1f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos), 1.0f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1), 20.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos + 1), 20.0f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxRot, imu::State::kIdxRot), 1.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxRot), 10.0f, kEps);
}

TEST(LioRegistration, DirectionalIcpWeightingUsesImuRelativeThreshold) {
    // Identical ICP information, different IMU prior: only the weak-IMU case is
    // attenuated, proving the decision is relative to the IMU information.
    Eigen::Matrix<float, 15, 15> H_imu_strong = Eigen::Matrix<float, 15, 15>::Zero();
    H_imu_strong(imu::State::kIdxPos, imu::State::kIdxPos) = 500.0f;
    Eigen::Matrix<float, 15, 15> H_imu_weak = Eigen::Matrix<float, 15, 15>::Zero();
    H_imu_weak(imu::State::kIdxPos, imu::State::kIdxPos) = 10.0f;

    lio::DirectionalIcpWeightingParams params;
    params.trans_min_information_ratio = 1.0f;
    params.rot_min_information_ratio = 0.0f;
    params.trans_imu_information_floor_per_inlier = 0.0f;
    params.rot_imu_information_floor_per_inlier = 0.0f;
    params.trans_weak_direction_scale = 0.1f;

    const auto weighted = [&](const Eigen::Matrix<float, 15, 15>& H_imu) {
        lio::LIOLinearizedResult factor;
        factor.inlier = 100;
        factor.H(imu::State::kIdxPos, imu::State::kIdxPos) = 100.0f;
        factor.b(imu::State::kIdxPos) = 10.0f;
        lio::apply_directional_icp_weighting(factor, H_imu, params);
        return factor.H(imu::State::kIdxPos, imu::State::kIdxPos);
    };

    // Strong IMU information raises the bar, so the same ICP information is
    // treated as weak; weak IMU information keeps the ICP contribution.
    EXPECT_NEAR(weighted(H_imu_strong), 20.0f, kEps);
    EXPECT_NEAR(weighted(H_imu_weak), 100.0f, kEps);
}

TEST(LioRegistration, DirectionalIcpWeightingFloorBlocksDegeneracyFeedback) {
    // With the measured IMU information collapsed (as it is when the previous
    // frame was degenerate), the per-inlier floor still provides a baseline.
    Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Zero();
    H_imu(imu::State::kIdxPos, imu::State::kIdxPos) = 0.5f;  // 0.005 per inlier, below the floor

    lio::LIOLinearizedResult factor;
    factor.inlier = 100;
    factor.H(imu::State::kIdxPos, imu::State::kIdxPos) = 100.0f;  // 1.0 per inlier, below 0.5 * 5.0
    factor.b(imu::State::kIdxPos) = 10.0f;

    lio::DirectionalIcpWeightingParams params;
    params.trans_min_information_ratio = 0.5f;
    params.rot_min_information_ratio = 0.0f;
    params.trans_imu_information_floor_per_inlier = 5.0f;
    params.trans_weak_direction_scale = 0.1f;

    lio::apply_directional_icp_weighting(factor, H_imu, params);

    // threshold = 0.5 * (5.0 * 100) = 250; scale = max(0.1, 100 / 250) = 0.4.
    EXPECT_NEAR(factor.H(imu::State::kIdxPos, imu::State::kIdxPos), 40.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos), 4.0f, kEps);
}

TEST(LioRegistration, DirectionalIcpWeightingVerboseLogsEigenvaluesPerInlier) {
    lio::LIOLinearizedResult factor;
    factor.inlier = 10;
    factor.H(imu::State::kIdxPos, imu::State::kIdxPos) = 10.0f;
    factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1) = 20.0f;
    factor.H(imu::State::kIdxPos + 2, imu::State::kIdxPos + 2) = 30.0f;
    factor.H(imu::State::kIdxRot, imu::State::kIdxRot) = 40.0f;
    factor.H(imu::State::kIdxRot + 1, imu::State::kIdxRot + 1) = 50.0f;
    factor.H(imu::State::kIdxRot + 2, imu::State::kIdxRot + 2) = 60.0f;
    factor.b(imu::State::kIdxPos) = 7.0f;

    lio::DirectionalIcpWeightingParams params;
    params.verbose = true;
    testing::internal::CaptureStdout();
    lio::apply_directional_icp_weighting(factor, Eigen::Matrix<float, 15, 15>::Zero(), params);
    const std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("[DirectionalIcpWeighting] translation eigenvalues/inlier:"), std::string::npos);
    EXPECT_NE(output.find("[DirectionalIcpWeighting] rotation eigenvalues/inlier:"), std::string::npos);
    EXPECT_NE(output.find("[DirectionalIcpWeighting] translation b/inlier:"), std::string::npos);
    EXPECT_NE(output.find("1 2 3"), std::string::npos);
    EXPECT_NE(output.find("4 5 6"), std::string::npos);
    EXPECT_NE(output.find("0.7"), std::string::npos);
}

TEST(LioRegistration, DirectionalIcpWeightingPreservesCoupledFactorStructure) {
    lio::LIOLinearizedResult factor;
    factor.inlier = 100;
    factor.H(imu::State::kIdxPos, imu::State::kIdxPos) = 1.0f;
    factor.H(imu::State::kIdxRot, imu::State::kIdxRot) = 1.0f;
    factor.H(imu::State::kIdxPos, imu::State::kIdxRot) = 0.5f;
    factor.H(imu::State::kIdxRot, imu::State::kIdxPos) = 0.5f;

    Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Zero();
    H_imu(imu::State::kIdxPos, imu::State::kIdxPos) = 500.0f;
    H_imu(imu::State::kIdxRot, imu::State::kIdxRot) = 500.0f;

    lio::DirectionalIcpWeightingParams params;
    params.trans_min_information_ratio = 1.0f;
    params.rot_min_information_ratio = 1.0f;
    params.trans_imu_information_floor_per_inlier = 0.0f;
    params.rot_imu_information_floor_per_inlier = 0.0f;
    params.trans_weak_direction_scale = 0.1f;
    params.rot_weak_direction_scale = 0.4f;

    lio::apply_directional_icp_weighting(factor, H_imu, params);

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

TEST(LioRegistration, DirectionalIcpWeightingSchurDetectsCoupledDegeneracy) {
    // Translation along x and rotation about x are each individually strong
    // (100), but the coupled mode t_x = theta_x is flat. The block-diagonal
    // analysis sees only the 100s and keeps the factor untouched; the Schur
    // analysis sees the coupled null mode and attenuates both x directions.
    const auto weighted = [](bool use_schur) {
        lio::LIOLinearizedResult factor;
        factor.inlier = 10;
        factor.H(imu::State::kIdxPos, imu::State::kIdxPos) = 100.0f;
        factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1) = 100.0f;
        factor.H(imu::State::kIdxPos + 2, imu::State::kIdxPos + 2) = 100.0f;
        factor.H(imu::State::kIdxRot, imu::State::kIdxRot) = 100.0f;
        factor.H(imu::State::kIdxRot + 1, imu::State::kIdxRot + 1) = 100.0f;
        factor.H(imu::State::kIdxRot + 2, imu::State::kIdxRot + 2) = 100.0f;
        factor.H(imu::State::kIdxPos, imu::State::kIdxRot) = -100.0f;
        factor.H(imu::State::kIdxRot, imu::State::kIdxPos) = -100.0f;

        lio::DirectionalIcpWeightingParams params;
        params.use_schur_complement = use_schur;
        params.trans_min_information_ratio = 0.5f;
        params.rot_min_information_ratio = 0.5f;
        params.trans_imu_information_floor_per_inlier = 5.0f;
        params.rot_imu_information_floor_per_inlier = 5.0f;
        params.trans_weak_direction_scale = 0.1f;
        params.rot_weak_direction_scale = 0.1f;
        lio::apply_directional_icp_weighting(factor, Eigen::Matrix<float, 15, 15>::Zero(), params);
        return factor;
    };

    const lio::LIOLinearizedResult block = weighted(false);
    EXPECT_NEAR(block.H(imu::State::kIdxPos, imu::State::kIdxPos), 100.0f, kEps);
    EXPECT_NEAR(block.H(imu::State::kIdxRot, imu::State::kIdxRot), 100.0f, kEps);

    // The coupled x mode has zero marginal information. The existing weak
    // handling maps an exactly zero eigenvalue to full removal, so both x
    // directions are zeroed while the uncoupled y/z axes keep their 100.
    const lio::LIOLinearizedResult schur = weighted(true);
    EXPECT_NEAR(schur.H(imu::State::kIdxPos, imu::State::kIdxPos), 0.0f, kEps);
    EXPECT_NEAR(schur.H(imu::State::kIdxRot, imu::State::kIdxRot), 0.0f, kEps);
    EXPECT_NEAR(schur.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1), 100.0f, kEps);
    EXPECT_NEAR(schur.H(imu::State::kIdxRot + 1, imu::State::kIdxRot + 1), 100.0f, kEps);
}

TEST(LioRegistration, ConstantVelocityPriorAnchorsOnlyDegenerateAxis) {
    // ICP position information is strong on x/z and weak on y (ratio 0.002 < 0.05).
    Eigen::Matrix3f icp_position_H = Eigen::Matrix3f::Zero();
    icp_position_H(0, 0) = 1000.0f;
    icp_position_H(1, 1) = 2.0f;
    icp_position_H(2, 2) = 1000.0f;

    lio::LIOLinearizedResult lio;
    const Eigen::Vector3f operating_velocity(1.0f, 2.0f, 0.0f);
    const Eigen::Vector3f anchor_velocity(1.0f, 0.0f, 0.0f);

    lio::ConstantVelocityPriorParams params;
    params.enable = true;
    params.min_eigenvalue_ratio = 0.05f;
    params.min_information_per_inlier = 0.5f;
    params.degenerate_velocity_sigma = 0.1f;  // information 100, absolute
    params.observable_velocity_sigma = 0.0f;

    const lio::ConstantVelocityPrior prior =
        lio::add_constant_velocity_prior(lio, icp_position_H, operating_velocity, anchor_velocity, 100, params);

    EXPECT_TRUE(prior.active);
    // Only the y axis is anchored: information 1 / 0.1^2 = 100.
    EXPECT_NEAR(lio.H(imu::State::kIdxVel, imu::State::kIdxVel), 0.0f, kEps);
    EXPECT_NEAR(lio.H(imu::State::kIdxVel + 1, imu::State::kIdxVel + 1), 100.0f, kEps);
    EXPECT_NEAR(lio.H(imu::State::kIdxVel + 2, imu::State::kIdxVel + 2), 0.0f, kEps);
    // Gradient uses r = v_op - v_anchor, so only the y component is non-zero.
    // Tolerance accounts for eigenvector rotation in single precision.
    EXPECT_NEAR(lio.b(imu::State::kIdxVel + 1), 100.0f * 2.0f, 1e-3f);

    const imu::State at_anchor = [&] {
        imu::State s;
        s.velocity = anchor_velocity;
        return s;
    }();
    const imu::State at_operating = [&] {
        imu::State s;
        s.velocity = operating_velocity;
        return s;
    }();
    EXPECT_NEAR(prior.cost(at_anchor), 0.0f, kEps);
    EXPECT_GT(prior.cost(at_operating), 0.0f);
}

TEST(LioRegistration, ConstantVelocityPriorDisabledLeavesVelocityUntouched) {
    Eigen::Matrix3f icp_position_H = Eigen::Matrix3f::Zero();
    icp_position_H(0, 0) = 1000.0f;
    icp_position_H(1, 1) = 2.0f;

    lio::LIOLinearizedResult lio;
    lio::ConstantVelocityPriorParams params;
    params.enable = false;

    const lio::ConstantVelocityPrior prior = lio::add_constant_velocity_prior(
        lio, icp_position_H, Eigen::Vector3f(1.0f, 2.0f, 0.0f), Eigen::Vector3f::Zero(), 100, params);

    EXPECT_FALSE(prior.active);
    EXPECT_TRUE(lio.H.isZero());
    EXPECT_TRUE(lio.b.isZero());
}

TEST(LioRegistration, ConstantVelocityPriorObservableInformationAppliesEverywhere) {
    Eigen::Matrix3f icp_position_H = Eigen::Matrix3f::Identity();
    lio::LIOLinearizedResult lio;

    lio::ConstantVelocityPriorParams params;
    params.enable = true;
    params.min_eigenvalue_ratio = 0.05f;
    params.degenerate_velocity_sigma = 0.1f;
    params.observable_velocity_sigma = 1.0f;  // information 1, absolute

    const lio::ConstantVelocityPrior prior = lio::add_constant_velocity_prior(
        lio, icp_position_H, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), 100, params);

    EXPECT_TRUE(prior.active);
    EXPECT_NEAR(lio.H(imu::State::kIdxVel, imu::State::kIdxVel), 1.0f, kEps);
    EXPECT_NEAR(lio.H(imu::State::kIdxVel + 1, imu::State::kIdxVel + 1), 1.0f, kEps);
    EXPECT_NEAR(lio.H(imu::State::kIdxVel + 2, imu::State::kIdxVel + 2), 1.0f, kEps);
}

TEST(LioRegistration, ConstantVelocityPriorAbsoluteGateSkipsWellConditionedFrame) {
    // z is relatively weak (40 < 0.05 * 1000 = 50) but still strong in absolute
    // terms (40 > 0.5 * 50 = 25), so the frame is well-conditioned and must not be
    // anchored.  The ratio gate alone would fire here.
    Eigen::Matrix3f icp_position_H = Eigen::Matrix3f::Zero();
    icp_position_H(0, 0) = 1000.0f;
    icp_position_H(1, 1) = 800.0f;
    icp_position_H(2, 2) = 40.0f;

    lio::LIOLinearizedResult lio;
    lio::ConstantVelocityPriorParams params;
    params.enable = true;
    params.min_eigenvalue_ratio = 0.05f;
    params.min_information_per_inlier = 0.5f;
    params.degenerate_velocity_sigma = 0.1f;
    params.observable_velocity_sigma = 0.0f;

    const lio::ConstantVelocityPrior prior = lio::add_constant_velocity_prior(
        lio, icp_position_H, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), 50, params);

    EXPECT_FALSE(prior.active);
    EXPECT_TRUE(lio.H.isZero());
}

TEST(LioRegistration, ConstantVelocityPriorAbsoluteGateStillCatchesCorridorAxis) {
    // z is relatively weak (20 < 0.05 * 1000 = 50) and also absolutely weak
    // (20 < 0.5 * 50 = 25), so the corridor axis is anchored.
    Eigen::Matrix3f icp_position_H = Eigen::Matrix3f::Zero();
    icp_position_H(0, 0) = 1000.0f;
    icp_position_H(1, 1) = 800.0f;
    icp_position_H(2, 2) = 20.0f;  // 0.4 per inlier at 50 inliers, below 0.5

    lio::LIOLinearizedResult lio;
    lio::ConstantVelocityPriorParams params;
    params.enable = true;
    params.min_eigenvalue_ratio = 0.05f;
    params.min_information_per_inlier = 0.5f;
    params.degenerate_velocity_sigma = 0.1f;
    params.observable_velocity_sigma = 0.0f;

    const lio::ConstantVelocityPrior prior = lio::add_constant_velocity_prior(
        lio, icp_position_H, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), 50, params);

    EXPECT_TRUE(prior.active);
    EXPECT_NEAR(lio.H(imu::State::kIdxVel + 2, imu::State::kIdxVel + 2), 100.0f, kEps);
}

TEST(LioRegistration, TsvdRemovesOnlyWeakIcpDirections) {
    lio::LIOLinearizedResult factor;
    factor.inlier = 100;
    factor.H(imu::State::kIdxPos, imu::State::kIdxPos) = 1.0f;
    factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1) = 20.0f;
    factor.H(imu::State::kIdxRot, imu::State::kIdxRot) = 2.0f;
    factor.H(imu::State::kIdxRot + 1, imu::State::kIdxRot + 1) = 30.0f;
    factor.H(imu::State::kIdxPos, imu::State::kIdxRot) = 0.5f;
    factor.H(imu::State::kIdxRot, imu::State::kIdxPos) = 0.5f;
    factor.b(imu::State::kIdxPos) = 10.0f;
    factor.b(imu::State::kIdxPos + 1) = 20.0f;
    factor.b(imu::State::kIdxRot) = 5.0f;
    factor.b(imu::State::kIdxRot + 1) = 30.0f;

    // ratio 0.01 with IMU information 500 gives the same threshold of 5 as the
    // old absolute per-inlier value, so axis 0 is weak and axis 1 is strong.
    Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Zero();
    H_imu(imu::State::kIdxPos, imu::State::kIdxPos) = 500.0f;
    H_imu(imu::State::kIdxRot, imu::State::kIdxRot) = 500.0f;

    lio::DirectionalIcpWeightingParams params;
    params.type = lio::DirectionalIcpWeightingType::tsvd;
    params.trans_min_information_ratio = 0.01f;
    params.rot_min_information_ratio = 0.01f;
    params.trans_imu_information_floor_per_inlier = 0.0f;
    params.rot_imu_information_floor_per_inlier = 0.0f;

    lio::apply_directional_icp_weighting(factor, H_imu, params);

    EXPECT_NEAR(factor.H(imu::State::kIdxPos, imu::State::kIdxPos), 0.0f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxRot, imu::State::kIdxRot), 0.0f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxPos, imu::State::kIdxRot), 0.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos), 0.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxRot), 0.0f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1), 20.0f, kEps);
    EXPECT_NEAR(factor.H(imu::State::kIdxRot + 1, imu::State::kIdxRot + 1), 30.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxPos + 1), 20.0f, kEps);
    EXPECT_NEAR(factor.b(imu::State::kIdxRot + 1), 30.0f, kEps);
}

TEST(LioRegistration, TsvdLeavesImuPriorInTruncatedDirection) {
    lio::LIOLinearizedResult icp_factor;
    icp_factor.inlier = 10;
    icp_factor.H(imu::State::kIdxPos, imu::State::kIdxPos) = 1.0f;
    icp_factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1) = 20.0f;
    icp_factor.b(imu::State::kIdxPos) = 3.0f;
    icp_factor.b(imu::State::kIdxPos + 1) = 4.0f;

    // ICP x information 1.0 with IMU information 0.2 and ratio 10 -> threshold 2.0,
    // so x is weak (removed) while y at 20.0 stays.
    Eigen::Matrix<float, 15, 15> H_prior = Eigen::Matrix<float, 15, 15>::Zero();
    H_prior(imu::State::kIdxPos, imu::State::kIdxPos) = 0.2f;

    lio::DirectionalIcpWeightingParams params;
    params.type = lio::DirectionalIcpWeightingType::tsvd;
    params.trans_min_information_ratio = 10.0f;
    params.rot_min_information_ratio = 0.0f;
    params.trans_imu_information_floor_per_inlier = 0.0f;
    params.rot_imu_information_floor_per_inlier = 0.0f;
    lio::apply_directional_icp_weighting(icp_factor, H_prior, params);

    Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Identity() * 2.0f;
    Eigen::Matrix<float, 15, 1> b_imu = Eigen::Matrix<float, 15, 1>::Ones();
    lio::add_imu_factor(icp_factor, H_imu, b_imu);

    EXPECT_NEAR(icp_factor.H(imu::State::kIdxPos, imu::State::kIdxPos), 2.0f, kEps);
    EXPECT_NEAR(icp_factor.b(imu::State::kIdxPos), 1.0f, kEps);
    EXPECT_NEAR(icp_factor.H(imu::State::kIdxPos + 1, imu::State::kIdxPos + 1), 22.0f, kEps);
    EXPECT_NEAR(icp_factor.b(imu::State::kIdxPos + 1), 5.0f, kEps);
}

TEST(LioRegistration, FixedBiasIsRemovedFromCoupledSolve) {
    Eigen::Matrix<float, 15, 15> H = Eigen::Matrix<float, 15, 15>::Identity();
    Eigen::Matrix<float, 15, 1> b = Eigen::Matrix<float, 15, 1>::Zero();
    H(imu::State::kIdxPos, imu::State::kIdxAccBias) = 0.5f;
    H(imu::State::kIdxAccBias, imu::State::kIdxPos) = 0.5f;
    b(imu::State::kIdxPos) = 1.0f;
    b(imu::State::kIdxAccBias) = 10.0f;

    Eigen::Matrix<float, 15, 1> delta;
    ASSERT_TRUE(lio::solve_ldlt(H, b, delta, nullptr, {false, true}));
    EXPECT_NEAR(delta(imu::State::kIdxPos), -1.0f, kEps);
    EXPECT_TRUE(delta.segment<3>(imu::State::kIdxAccBias).isZero(kEps));
}

TEST(LioRegistration, BiasMasksAreIndependent) {
    const Eigen::Matrix<float, 15, 15> H = Eigen::Matrix<float, 15, 15>::Identity();
    Eigen::Matrix<float, 15, 1> b = Eigen::Matrix<float, 15, 1>::Ones();
    Eigen::Matrix<float, 15, 1> delta;

    ASSERT_TRUE(lio::solve_ldlt(H, b, delta, nullptr, {true, false}));
    EXPECT_FALSE(delta.segment<3>(imu::State::kIdxAccBias).isZero(kEps));
    EXPECT_TRUE(delta.segment<3>(imu::State::kIdxGyrBias).isZero(kEps));
}

TEST(LioRegistration, SolveRejectsNonFiniteSystem) {
    Eigen::Matrix<float, 15, 15> H = Eigen::Matrix<float, 15, 15>::Identity();
    Eigen::Matrix<float, 15, 1> b = Eigen::Matrix<float, 15, 1>::Zero();
    H(0, 0) = std::numeric_limits<float>::quiet_NaN();
    Eigen::Matrix<float, 15, 1> delta = Eigen::Matrix<float, 15, 1>::Ones();
    EXPECT_FALSE(lio::solve_ldlt(H, b, delta));
    EXPECT_TRUE(delta.isZero());
}

TEST(LioRegistration, StatusSeparatesNoProgressFromInvalidResults) {
    lio::LIORegistrationResult result;
    result.status = lio::LIORegistrationStatus::no_progress;
    EXPECT_TRUE(result.valid());
    result.status = lio::LIORegistrationStatus::numeric_failure;
    EXPECT_FALSE(result.valid());
    result.status = lio::LIORegistrationStatus::invalid_imu;
    EXPECT_FALSE(result.valid());
}
