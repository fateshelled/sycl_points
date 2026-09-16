#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "sycl_points/algorithms/registration/degenerate_regularization.hpp"

namespace registration = sycl_points::algorithms::registration;

TEST(DegenerateRegularization, ParsesTsvdType) {
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("TSVD"),
              registration::DegenerateRegularizationType::tsvd);
    EXPECT_EQ(registration::DegenerateRegularizationType_from_string("tsvd"),
              registration::DegenerateRegularizationType::tsvd);
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
