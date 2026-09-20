#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "sycl_points/algorithms/registration/degenerate_regularization.hpp"

namespace registration = sycl_points::algorithms::registration;

TEST(DegenerateRegularization, ParsesNlRegType) {
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("NL_REG"),
              registration::DegenerateRegularizationType::nl_reg);
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("NL-REG"),
              registration::DegenerateRegularizationType::nl_reg);
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("nl_reg"),
              registration::DegenerateRegularizationType::nl_reg);
}

TEST(DegenerateRegularization, NlRegPenalizesOnlyDegenerateDirections) {
    registration::LinearizedResult input;
    input.inlier = 10;
    input.H.diagonal() << 5.0f, 20.0f, 30.0f, 2.0f, 30.0f, 40.0f;
    input.b << 10.0f, 20.0f, 30.0f, 4.0f, 30.0f, 40.0f;

    registration::DegenerateRegularizationParams params;
    params.type = registration::DegenerateRegularizationType::nl_reg;
    params.rot_eigenvalue_threshold = 1.0f;
    params.trans_eigenvalue_threshold = 1.0f;
    params.base_factor = 0.2f;

    registration::DegenerateRegularization regularization;
    regularization.set_params(params);
    const auto output = regularization.regularize(input, Eigen::Isometry3f::Identity(),
                                                   Eigen::Isometry3f::Identity());

    // lambda = base_factor * inlier = 2.0. Only the first rotation and translation
    // eigenvalues are below their per-inlier threshold.
    EXPECT_NEAR(output.H(0, 0), 7.0f, 1e-6f);
    EXPECT_NEAR(output.H(3, 3), 4.0f, 1e-6f);
    EXPECT_NEAR(output.H(1, 1), input.H(1, 1), 1e-6f);
    EXPECT_NEAR(output.H(2, 2), input.H(2, 2), 1e-6f);
    EXPECT_NEAR(output.H(4, 4), input.H(4, 4), 1e-6f);
    EXPECT_NEAR(output.H(5, 5), input.H(5, 5), 1e-6f);
    EXPECT_TRUE(output.b.isApprox(input.b, 1e-6f));
    EXPECT_EQ(output.inlier, input.inlier);
    EXPECT_FLOAT_EQ(output.error, input.error);
}

TEST(DegenerateRegularization, VerboseLogsEigenvaluesPerInlier) {
    registration::LinearizedResult input;
    input.inlier = 10;
    input.H.diagonal() << 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f;

    registration::DegenerateRegularizationParams params;
    params.type = registration::DegenerateRegularizationType::nl_reg;
    params.verbose = true;

    registration::DegenerateRegularization regularization;
    regularization.set_params(params);
    testing::internal::CaptureStdout();
    regularization.regularize(input, Eigen::Isometry3f::Identity(), Eigen::Isometry3f::Identity());
    const std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("[DegenerateRegularization] rotation eigenvalues/inlier:"), std::string::npos);
    EXPECT_NE(output.find("[DegenerateRegularization] translation eigenvalues/inlier:"), std::string::npos);
    EXPECT_NE(output.find("1 2 3"), std::string::npos);
    EXPECT_NE(output.find("4 5 6"), std::string::npos);
}

TEST(DegenerateRegularization, ParsesTsvdType) {
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("TSVD"),
              registration::DegenerateRegularizationType::tsvd);
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("tsvd"),
              registration::DegenerateRegularizationType::tsvd);
}

TEST(DegenerateRegularization, ParsesAdditionalPaperMethods) {
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("L_REG"),
              registration::DegenerateRegularizationType::l_reg);
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("L-REG"),
              registration::DegenerateRegularizationType::l_reg);
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("SOLUTION_REMAP"),
              registration::DegenerateRegularizationType::solution_remap);
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("SOLUTION-REMAP"),
              registration::DegenerateRegularizationType::solution_remap);
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("EQ_CONSTRAINT"),
              registration::DegenerateRegularizationType::eq_constraint);
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("EQ-CONSTRAINT"),
              registration::DegenerateRegularizationType::eq_constraint);
}

TEST(DegenerateRegularization, LRegAddsLinearTikhonovPenalty) {
    registration::LinearizedResult input;
    input.inlier = 10;
    input.H.diagonal() << 5.0f, 20.0f, 30.0f, 2.0f, 30.0f, 40.0f;
    input.b.setOnes();

    registration::DegenerateRegularizationParams params;
    params.type = registration::DegenerateRegularizationType::l_reg;
    params.rot_eigenvalue_threshold = 1.0f;
    params.trans_eigenvalue_threshold = 1.0f;
    params.linear_factor = 440.0f;

    registration::DegenerateRegularization regularization;
    regularization.set_params(params);
    const auto output = regularization.regularize(input, Eigen::Isometry3f::Identity(),
                                                   Eigen::Isometry3f::Identity());

    EXPECT_NEAR(output.H(0, 0), 445.0f, 1e-5f);
    EXPECT_NEAR(output.H(3, 3), 442.0f, 1e-5f);
    EXPECT_NEAR(output.H(1, 1), input.H(1, 1), 1e-6f);
    EXPECT_NEAR(output.H(4, 4), input.H(4, 4), 1e-6f);
    EXPECT_TRUE(output.b.isApprox(input.b, 1e-6f));
}

TEST(DegenerateRegularization, SolutionRemapProjectsOnlyTheSolvedUpdate) {
    registration::LinearizedResult input;
    input.inlier = 10;
    input.H.diagonal() << 5.0f, 20.0f, 30.0f, 2.0f, 30.0f, 40.0f;
    input.b.setOnes();

    registration::DegenerateRegularizationParams params;
    params.type = registration::DegenerateRegularizationType::solution_remap;
    params.rot_eigenvalue_threshold = 1.0f;
    params.trans_eigenvalue_threshold = 1.0f;

    registration::DegenerateRegularization regularization;
    regularization.set_params(params);
    const auto output = regularization.regularize(input, Eigen::Isometry3f::Identity(),
                                                   Eigen::Isometry3f::Identity());

    EXPECT_TRUE(output.H.isApprox(input.H, 1e-6f));
    EXPECT_TRUE(output.b.isApprox(input.b, 1e-6f));
    const Eigen::Vector<float, 6> update = output.solution_projector * output.H.ldlt().solve(-output.b);
    EXPECT_NEAR(update(0), 0.0f, 1e-6f);
    EXPECT_NEAR(update(3), 0.0f, 1e-6f);
    EXPECT_NE(update(1), 0.0f);
    EXPECT_NE(update(4), 0.0f);
}

TEST(DegenerateRegularization, EqualityConstraintEnforcesZeroDegenerateUpdate) {
    registration::LinearizedResult input;
    input.inlier = 10;
    input.H.diagonal() << 5.0f, 20.0f, 30.0f, 2.0f, 30.0f, 40.0f;
    input.b.setOnes();

    registration::DegenerateRegularizationParams params;
    params.type = registration::DegenerateRegularizationType::eq_constraint;
    params.rot_eigenvalue_threshold = 1.0f;
    params.trans_eigenvalue_threshold = 1.0f;

    registration::DegenerateRegularization regularization;
    regularization.set_params(params);
    const auto output = regularization.regularize(input, Eigen::Isometry3f::Identity(),
                                                   Eigen::Isometry3f::Identity());

    const Eigen::Vector<float, 6> update = output.H.ldlt().solve(-output.b);
    EXPECT_NEAR(update(0), 0.0f, 1e-6f);
    EXPECT_NEAR(update(3), 0.0f, 1e-6f);
    EXPECT_NEAR(update(1), -1.0f / 20.0f, 1e-6f);
    EXPECT_NEAR(update(4), -1.0f / 30.0f, 1e-6f);
}

TEST(DegenerateRegularization, TsvdRemovesDegeneratePoseUpdates) {
    registration::LinearizedResult input;
    input.inlier = 10;
    input.H.diagonal() << 5.0f, 20.0f, 30.0f, 2.0f, 30.0f, 40.0f;
    input.b << 10.0f, 20.0f, 30.0f, 4.0f, 30.0f, 40.0f;

    registration::DegenerateRegularizationParams params;
    params.type = registration::DegenerateRegularizationType::tsvd;
    params.rot_eigenvalue_threshold = 1.0f;
    params.trans_eigenvalue_threshold = 1.0f;

    registration::DegenerateRegularization regularization;
    regularization.set_params(params);
    const auto output = regularization.regularize(input, Eigen::Isometry3f::Identity(),
                                                   Eigen::Isometry3f::Identity());

    Eigen::Vector<float, 6> update = output.H.ldlt().solve(-output.b);
    EXPECT_NEAR(update(0), 0.0f, 1e-6f);
    EXPECT_NEAR(update(3), 0.0f, 1e-6f);
    EXPECT_NEAR(update(1), -1.0f, 1e-6f);
    EXPECT_NEAR(update(2), -1.0f, 1e-6f);
    EXPECT_NEAR(update(4), -1.0f, 1e-6f);
    EXPECT_NEAR(update(5), -1.0f, 1e-6f);
    EXPECT_EQ(output.inlier, input.inlier);
    EXPECT_FLOAT_EQ(output.error, input.error);
}

TEST(DegenerateRegularization, TsvdPreservesCoupledObservableSubspace) {
    registration::LinearizedResult input;
    input.inlier = 10;
    input.H.diagonal() << 5.0f, 20.0f, 30.0f, 2.0f, 30.0f, 40.0f;
    input.H(1, 4) = 3.0f;
    input.H(4, 1) = 3.0f;
    input.b << 7.0f, 8.0f, 9.0f, 6.0f, 5.0f, 4.0f;

    registration::DegenerateRegularizationParams params;
    params.type = registration::DegenerateRegularizationType::tsvd;
    params.rot_eigenvalue_threshold = 1.0f;
    params.trans_eigenvalue_threshold = 1.0f;

    registration::DegenerateRegularization regularization;
    regularization.set_params(params);
    const auto output = regularization.regularize(input, Eigen::Isometry3f::Identity(),
                                                   Eigen::Isometry3f::Identity());

    EXPECT_NEAR(output.H(1, 4), input.H(1, 4), 1e-6f);
    EXPECT_NEAR(output.H(4, 1), input.H(4, 1), 1e-6f);
    EXPECT_NEAR(output.H(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(output.H(3, 3), 1.0f, 1e-6f);
    EXPECT_NEAR(output.b(0), 0.0f, 1e-6f);
    EXPECT_NEAR(output.b(3), 0.0f, 1e-6f);
    EXPECT_TRUE(output.H.isApprox(output.H.transpose(), 1e-6f));
}

TEST(DegenerateRegularization, SchurDirectionsExposeCoupledNullMode) {
    // Translation along x and rotation about x are each individually strong (100),
    // but the coupled mode t_x = theta_x is exactly flat. The independent 3x3
    // blocks only see the 100s; the Schur complement sees the 0.
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    H.block<3, 3>(0, 0) = 100.0f * Eigen::Matrix3f::Identity();  // rotation
    H.block<3, 3>(3, 3) = 100.0f * Eigen::Matrix3f::Identity();  // translation
    H(0, 3) = -100.0f;
    H(3, 0) = -100.0f;

    const auto schur =
        registration::compute_schur_directions(H, registration::PoseHessianOrder::rotation_first);

    ASSERT_TRUE(schur.valid);
    EXPECT_NEAR(schur.rot_eigenvalues(0), 0.0f, 1e-5);
    EXPECT_NEAR(schur.trans_eigenvalues(0), 0.0f, 1e-5);
    EXPECT_GT(schur.rot_eigenvalues(1), 99.0);
    EXPECT_GT(schur.trans_eigenvalues(1), 99.0);
    EXPECT_EQ(schur.rotation_block_rank, 3);
    EXPECT_EQ(schur.translation_block_rank, 3);
}

TEST(DegenerateRegularization, SchurDirectionsHandleRankDeficientBlock) {
    // A rank-deficient marginalized block must yield a finite, rank-aware
    // pseudo-inverse instead of a NaN Schur complement.
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    H(0, 0) = 50.0f;
    H(3, 3) = 50.0f;
    H(0, 3) = -50.0f;
    H(3, 0) = -50.0f;

    const auto schur =
        registration::compute_schur_directions(H, registration::PoseHessianOrder::rotation_first);

    ASSERT_TRUE(schur.valid);
    EXPECT_TRUE(schur.S_translation.allFinite());
    EXPECT_TRUE(schur.S_rotation.allFinite());
    EXPECT_EQ(schur.rotation_block_rank, 1);
    EXPECT_EQ(schur.translation_block_rank, 1);
}

TEST(DegenerateRegularization, SchurComplementDetectsCoupledDegeneracyMissedByBlocks) {
    // Same coupled x-axis degeneracy as above, end to end through TSVD.
    registration::LinearizedResult input;
    input.inlier = 1;
    input.H.block<3, 3>(0, 0) = 100.0f * Eigen::Matrix3f::Identity();
    input.H.block<3, 3>(3, 3) = 100.0f * Eigen::Matrix3f::Identity();
    input.H(0, 3) = -100.0f;
    input.H(3, 0) = -100.0f;
    input.b.setOnes();

    registration::DegenerateRegularizationParams params;
    params.type = registration::DegenerateRegularizationType::tsvd;
    params.rot_eigenvalue_threshold = 1.0f;
    params.trans_eigenvalue_threshold = 1.0f;

    registration::DegenerateRegularization regularization;

    // Block-diagonal analysis: every diagonal eigenvalue is 100, so nothing is
    // truncated and the coupled degeneracy is missed.
    params.use_schur_complement = false;
    regularization.set_params(params);
    const auto block_output =
        regularization.regularize(input, Eigen::Isometry3f::Identity(), Eigen::Isometry3f::Identity());
    EXPECT_NEAR(block_output.H(0, 0), 100.0f, 1e-4f);
    EXPECT_NEAR(block_output.H(3, 3), 100.0f, 1e-4f);

    // Schur analysis: the coupled x mode is singular (0), so the x rotation and
    // translation directions are truncated while the uncoupled y/z axes keep
    // their information.
    params.use_schur_complement = true;
    regularization.set_params(params);
    const auto schur_output =
        regularization.regularize(input, Eigen::Isometry3f::Identity(), Eigen::Isometry3f::Identity());
    EXPECT_NEAR(schur_output.H(0, 0), 1.0f, 1e-4f);
    EXPECT_NEAR(schur_output.H(3, 3), 1.0f, 1e-4f);
    EXPECT_NEAR(schur_output.H(1, 1), 100.0f, 1e-4f);
    EXPECT_NEAR(schur_output.H(2, 2), 100.0f, 1e-4f);
    EXPECT_NEAR(schur_output.H(4, 4), 100.0f, 1e-4f);
    EXPECT_NEAR(schur_output.H(5, 5), 100.0f, 1e-4f);
}

TEST(DegenerateRegularization, SchurComplementMatchesBlocksForDiagonalHessian) {
    // With no translation/rotation coupling the Schur complements reduce to the
    // diagonal blocks, so both analyses select the same degenerate directions.
    registration::LinearizedResult input;
    input.inlier = 10;
    input.H.diagonal() << 5.0f, 20.0f, 30.0f, 2.0f, 30.0f, 40.0f;
    input.b << 10.0f, 20.0f, 30.0f, 4.0f, 30.0f, 40.0f;

    registration::DegenerateRegularizationParams params;
    params.type = registration::DegenerateRegularizationType::tsvd;
    params.rot_eigenvalue_threshold = 1.0f;
    params.trans_eigenvalue_threshold = 1.0f;

    registration::DegenerateRegularization regularization;

    params.use_schur_complement = false;
    regularization.set_params(params);
    const auto block_output =
        regularization.regularize(input, Eigen::Isometry3f::Identity(), Eigen::Isometry3f::Identity());

    params.use_schur_complement = true;
    regularization.set_params(params);
    const auto schur_output =
        regularization.regularize(input, Eigen::Isometry3f::Identity(), Eigen::Isometry3f::Identity());

    EXPECT_TRUE(schur_output.H.isApprox(block_output.H, 1e-4f));
    EXPECT_TRUE(schur_output.b.isApprox(block_output.b, 1e-4f));
}
