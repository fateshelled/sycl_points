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

/// @brief Mirror of MapPrior::prepare_for_align()'s closed-form computation,
///        re-implemented here so the event-driven path can be checked against
///        an independent reference rather than against a second MapPrior
///        instance sharing the same code path.  Returns a zero matrix when
///        the inputs are unusable (matching prepare_for_align's no-op).
registration::MapPriorMatrix compute_omega_closed_form(const registration::MapPriorParams& /*params*/,
                                                       const registration::RegistrationResult& prev,
                                                       const Eigen::Isometry3f& T_pred,
                                                       const registration::MapPriorMatrix& Q) {
    const float dof = 3.0f * static_cast<float>(prev.inlier) - 6.0f;
    if (dof <= 0.0f) return registration::MapPriorMatrix::Zero();
    if (!std::isfinite(prev.error_raw) || prev.error_raw < 0.0f) return registration::MapPriorMatrix::Zero();
    const float s_sq = std::max(1.0f, 2.0f * prev.error_raw / dof);
    const registration::MapPriorMatrix H_cal = prev.H_raw / s_sq;

    const Eigen::Matrix3f R_rel = prev.T.rotation().transpose() * T_pred.rotation();
    registration::MapPriorMatrix Ad = registration::MapPriorMatrix::Zero();
    Ad.block<3, 3>(0, 0) = R_rel;
    Ad.block<3, 3>(3, 3) = R_rel;
    const registration::MapPriorMatrix H_curr = Ad.transpose() * H_cal * Ad;

    const Eigen::Vector<float, 6> q_diag = Q.diagonal();
    Eigen::Vector<float, 6> R_diag;
    R_diag.head<3>() = q_diag.head<3>().cwiseInverse();
    R_diag.tail<3>() = q_diag.tail<3>().cwiseInverse();
    const registration::MapPriorMatrix R = R_diag.asDiagonal();

    Eigen::LDLT<registration::MapPriorMatrix> ldlt(H_curr + R);
    if (ldlt.info() != Eigen::Success) return registration::MapPriorMatrix::Zero();
    registration::MapPriorMatrix Omega = R - R * ldlt.solve(R);
    Omega = 0.5f * (Omega + Omega.transpose());
    return Omega;
}

}  // namespace

// ---------------------------------------------------------------------------
// Free-function math tests
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Event-driven API tests (accumulate_process_covariance /
// notify_registration_success / notify_prediction_only / prepare_for_align).
// ---------------------------------------------------------------------------

TEST(MapPrior, PrepareForAlignWithoutPriorSourceIsNoOp) {
    const auto params = make_params();
    registration::MapPrior prior;
    prior.set_params(params);

    Eigen::Isometry3f predicted = Eigen::Isometry3f::Identity();
    predicted.translation().x() = 0.3f;
    prior.prepare_for_align(predicted);

    EXPECT_FALSE(prior.is_active());
    EXPECT_FALSE(prior.has_valid_prior_source());
    EXPECT_FALSE(prior.last_frame_registered());
}

TEST(MapPrior, NonFallbackPathMatchesClosedFormOmega) {
    // Regression guard for the non-fallback path: notify_registration_success
    // -> accumulate_process_covariance (one step) -> prepare_for_align must
    // produce the closed-form Omega_prior derived from H_raw and the
    // single-step process covariance.  This is the path taken on every frame
    // with a valid registration and no prior fallback.
    const auto params = make_params();
    const auto result = make_valid_result();

    Eigen::Isometry3f predicted = Eigen::Isometry3f::Identity();
    predicted.translation() = Eigen::Vector3f(0.5f, -0.1f, 0.2f);
    predicted.linear() = Eigen::AngleAxisf(0.2f, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    registration::MapPrior prior;
    prior.set_params(params);
    prior.notify_registration_success(result, result.T);
    ASSERT_TRUE(prior.has_valid_prior_source());
    EXPECT_TRUE(prior.last_frame_registered());
    prior.accumulate_process_covariance(result.T, predicted);
    prior.prepare_for_align(predicted);

    ASSERT_TRUE(prior.is_active());
    const auto expected_Q = registration::make_map_prior_process_covariance(params, result.T, predicted);
    const auto expected_omega = compute_omega_closed_form(params, result, predicted, expected_Q);
    EXPECT_TRUE(prior.information_matrix().isApprox(expected_omega, 1e-5f));
}

TEST(MapPrior, NotifyRegistrationSuccessResetsAccumulator) {
    // After a valid registration, the accumulator should be reset so the
    // subsequent prepare_for_align matches the single-step covariance, not the
    // previously accumulated covariance.
    const auto params = make_params();
    const auto result = make_valid_result();

    Eigen::Isometry3f predicted = Eigen::Isometry3f::Identity();
    predicted.translation().x() = 0.4f;
    predicted.linear() = Eigen::AngleAxisf(0.1f, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    registration::MapPrior prior;
    prior.set_params(params);
    prior.notify_registration_success(result, result.T);

    // Pre-warm the accumulator with extra steps; these should be discarded by
    // the next valid notify_registration_success.
    Eigen::Isometry3f intermediate = predicted;
    intermediate.translate(Eigen::Vector3f(0.2f, 0.0f, 0.0f));
    prior.accumulate_process_covariance(result.T, intermediate);
    ASSERT_TRUE(prior.has_valid_prior_source());

    // A new valid commit resets the accumulator.
    registration::RegistrationResult new_result = result;
    new_result.T = intermediate;
    prior.notify_registration_success(new_result, new_result.T);

    prior.accumulate_process_covariance(new_result.T, predicted);
    prior.prepare_for_align(predicted);

    // Compare against the closed-form Omega keyed off the newly committed
    // result with a single-step process covariance.
    ASSERT_TRUE(prior.is_active());
    const auto expected_Q = registration::make_map_prior_process_covariance(params, new_result.T, predicted);
    const auto expected_omega = compute_omega_closed_form(params, new_result, predicted, expected_Q);
    EXPECT_TRUE(prior.information_matrix().isApprox(expected_omega, 1e-5f));
}

TEST(MapPrior, AccumulatedPredictionCovarianceWeakensPrior) {
    // One valid registration followed by a single accumulate step should be
    // tighter than one valid registration followed by three accumulate steps
    // simulating prediction-only fallback frames.
    const auto params = make_params();
    const auto result = make_valid_result();

    Eigen::Isometry3f predicted = Eigen::Isometry3f::Identity();
    predicted.translation().x() = 0.5f;

    registration::MapPrior one_step_prior;
    one_step_prior.set_params(params);
    one_step_prior.notify_registration_success(result, result.T);
    one_step_prior.accumulate_process_covariance(result.T, predicted);
    one_step_prior.prepare_for_align(predicted);

    // Three accumulate steps simulating two extra prediction-only frames.
    Eigen::Isometry3f mid1 = predicted;
    mid1.translate(Eigen::Vector3f(0.1f, 0.0f, 0.0f));
    Eigen::Isometry3f mid2 = mid1;
    mid2.translate(Eigen::Vector3f(0.1f, 0.0f, 0.0f));

    registration::MapPrior three_step_prior;
    three_step_prior.set_params(params);
    three_step_prior.notify_registration_success(result, result.T);
    three_step_prior.accumulate_process_covariance(result.T, mid1);
    three_step_prior.notify_prediction_only();
    three_step_prior.accumulate_process_covariance(mid1, mid2);
    three_step_prior.notify_prediction_only();
    three_step_prior.accumulate_process_covariance(mid2, predicted);
    three_step_prior.prepare_for_align(predicted);

    ASSERT_TRUE(one_step_prior.is_active());
    ASSERT_TRUE(three_step_prior.is_active());
    EXPECT_LT(three_step_prior.information_matrix().trace(), one_step_prior.information_matrix().trace());

    Eigen::SelfAdjointEigenSolver<registration::MapPriorMatrix> solver(one_step_prior.information_matrix() -
                                                                       three_step_prior.information_matrix());
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-3f);
}

TEST(MapPrior, NotifyPredictionOnlySetsLastFrameRegisteredFalse) {
    const auto params = make_params();
    const auto result = make_valid_result();

    registration::MapPrior prior;
    prior.set_params(params);
    prior.notify_registration_success(result, result.T);
    ASSERT_TRUE(prior.last_frame_registered());

    prior.notify_prediction_only();
    EXPECT_FALSE(prior.last_frame_registered());
    // Prior source itself remains available for motion prediction weighting.
    EXPECT_TRUE(prior.has_valid_prior_source());
}

TEST(MapPrior, InvalidResultContinuesAccumulation) {
    // notify_registration_success with an invalid result (too few inliers)
    // must NOT replace the prior source and must continue accumulating so the
    // prior weakens across successive unusable frames.
    const auto params = make_params();
    const auto result = make_valid_result();

    Eigen::Isometry3f predicted = Eigen::Isometry3f::Identity();
    predicted.translation().x() = 0.3f;

    registration::MapPrior prior;
    prior.set_params(params);
    prior.notify_registration_success(result, result.T);
    prior.accumulate_process_covariance(result.T, predicted);

    // Stash the Ω produced by one valid accumulation step.
    prior.prepare_for_align(predicted);
    const auto one_step_omega = prior.information_matrix();
    ASSERT_TRUE(prior.is_active());

    // Submit an invalid result (inlier == 0); the accumulator should advance
    // rather than reset.
    registration::RegistrationResult invalid_result;
    invalid_result.T = predicted;
    invalid_result.H_raw = registration::MapPriorMatrix::Zero();
    invalid_result.error_raw = 0.0f;
    invalid_result.inlier = 0;

    Eigen::Isometry3f next_predicted = predicted;
    next_predicted.translate(Eigen::Vector3f(0.3f, 0.0f, 0.0f));

    prior.notify_registration_success(invalid_result, next_predicted);
    EXPECT_FALSE(prior.last_frame_registered());
    // Prior source identity is preserved.
    ASSERT_TRUE(prior.has_valid_prior_source());
    ASSERT_NE(prior.last_valid_result(), nullptr);
    EXPECT_TRUE(prior.last_valid_result()->T.matrix().isApprox(result.T.matrix(), 1e-6f));

    prior.prepare_for_align(next_predicted);
    ASSERT_TRUE(prior.is_active());
    // Two-step accumulated prior should be weaker (smaller trace) than the
    // single-step prior observed earlier.
    EXPECT_LT(prior.information_matrix().trace(), one_step_omega.trace());
}
