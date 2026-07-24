#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "sycl_points/algorithms/registration/map_prior.hpp"

namespace registration = sycl_points::algorithms::registration;

namespace {

registration::RegistrationResult make_valid_result() {
    registration::RegistrationResult result;
    result.T = Eigen::Isometry3f::Identity();
    result.H_raw = registration::MapPriorMatrix::Identity() * 100.0f;
    result.error_raw = 10.0f;
    result.inlier = 100;
    return result;
}

registration::MapPriorParams make_params() {
    registration::MapPriorParams params;
    params.enabled = true;
    params.rot_vel_sigma = 0.3f;
    params.trans_vel_sigma = 0.2f;
    params.rot_base_sigma = 0.01f;
    params.trans_base_sigma = 0.02f;
    return params;
}

}  // namespace

TEST(MapPrior, ExplicitSingleStepCovarianceMatchesDefaultUpdate) {
    const auto params = make_params();
    const auto result = make_valid_result();

    Eigen::Isometry3f predicted = Eigen::Isometry3f::Identity();
    predicted.translation() = Eigen::Vector3f(0.5f, -0.1f, 0.2f);
    predicted.linear() = Eigen::AngleAxisf(0.2f, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    registration::MapPrior default_prior;
    default_prior.set_params(params);
    default_prior.update(result, predicted);

    registration::MapPrior explicit_prior;
    explicit_prior.set_params(params);
    explicit_prior.update(result, predicted,
                          registration::make_map_prior_process_covariance(params, result.T, predicted));

    ASSERT_TRUE(default_prior.is_active());
    ASSERT_TRUE(explicit_prior.is_active());
    EXPECT_TRUE(default_prior.information_matrix().isApprox(explicit_prior.information_matrix(), 1e-4f));
}

TEST(MapPrior, AccumulatedPredictionCovarianceWeakensPrior) {
    const auto params = make_params();
    const auto result = make_valid_result();

    Eigen::Isometry3f predicted = Eigen::Isometry3f::Identity();
    predicted.translation().x() = 0.5f;
    const auto one_step = registration::make_map_prior_process_covariance(params, result.T, predicted);

    registration::MapPrior one_step_prior;
    one_step_prior.set_params(params);
    one_step_prior.update(result, predicted, one_step);

    registration::MapPrior three_step_prior;
    three_step_prior.set_params(params);
    three_step_prior.update(result, predicted, 3.0f * one_step);

    ASSERT_TRUE(one_step_prior.is_active());
    ASSERT_TRUE(three_step_prior.is_active());
    EXPECT_LT(three_step_prior.information_matrix().trace(), one_step_prior.information_matrix().trace());

    Eigen::SelfAdjointEigenSolver<registration::MapPriorMatrix> solver(one_step_prior.information_matrix() -
                                                                       three_step_prior.information_matrix());
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-3f);
}

TEST(MapPrior, RotatedAccumulatedCovarianceRemainsFiniteAndSymmetric) {
    const auto params = make_params();

    Eigen::Isometry3f first = Eigen::Isometry3f::Identity();
    first.translation().x() = 0.2f;
    first.linear() = Eigen::AngleAxisf(0.3f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    auto accumulated = registration::make_map_prior_process_covariance(params, Eigen::Isometry3f::Identity(), first);

    Eigen::Isometry3f second = first;
    second.translate(Eigen::Vector3f(0.1f, 0.2f, 0.0f));
    second.rotate(Eigen::AngleAxisf(0.4f, Eigen::Vector3f::UnitY()));

    accumulated = registration::accumulate_map_prior_process_covariance(params, accumulated, first, second);

    EXPECT_TRUE(accumulated.allFinite());
    EXPECT_TRUE(accumulated.isApprox(accumulated.transpose(), 1e-6f));
    Eigen::LDLT<registration::MapPriorMatrix> ldlt(accumulated);
    EXPECT_EQ(ldlt.info(), Eigen::Success);
    EXPECT_TRUE(ldlt.isPositive());
}
