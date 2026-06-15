#include <gtest/gtest.h>

#include <cmath>

#include "sycl_points/pipeline/range_bias_estimator.hpp"

using sycl_points::pipeline::lidar_odometry::RangeBiasEstimator;

namespace {

RangeBiasEstimator::Params make_enabled_params() {
    RangeBiasEstimator::Params p;
    p.enable = true;
    p.learning_rate = 1.0f;  // undamped for deterministic checks
    p.max_abs_k = 1.0f;
    p.max_step = 1.0f;
    p.min_inlier = 10;
    p.min_sxx = 1.0f;
    p.min_condition = 1.0f;
    return p;
}

}  // namespace

TEST(RangeBiasEstimator, DefaultIsZeroAndDisabled) {
    RangeBiasEstimator est;
    EXPECT_EQ(est.k(), 0.0f);
    // Disabled by default: update is a no-op.
    EXPECT_FALSE(est.update(10.0, 10.0, 1000, 100.0f));
    EXPECT_EQ(est.k(), 0.0f);
}

TEST(RangeBiasEstimator, StepSignAndMagnitude) {
    RangeBiasEstimator est(make_enabled_params());
    // dk = -learning_rate * s_xy / s_xx = -(2/4) = -0.5
    EXPECT_TRUE(est.update(2.0, 4.0, 100, 5.0f));
    EXPECT_NEAR(est.k(), -0.5f, 1e-6f);

    // Positive correlation drives k negative; negative correlation drives it positive.
    RangeBiasEstimator est2(make_enabled_params());
    EXPECT_TRUE(est2.update(-3.0, 6.0, 100, 5.0f));
    EXPECT_NEAR(est2.k(), 0.5f, 1e-6f);
}

TEST(RangeBiasEstimator, GatesOnInlierCountSxxAndConditioning) {
    RangeBiasEstimator est(make_enabled_params());

    // Too few inliers.
    EXPECT_FALSE(est.update(10.0, 10.0, 5, 100.0f));
    EXPECT_EQ(est.k(), 0.0f);

    // Regressor energy below observability floor.
    EXPECT_FALSE(est.update(10.0, 0.5, 100, 100.0f));
    EXPECT_EQ(est.k(), 0.0f);

    // Degenerate scene: conditioning below floor -> frozen.
    EXPECT_FALSE(est.update(10.0, 10.0, 100, 0.1f));
    EXPECT_EQ(est.k(), 0.0f);

    // All gates satisfied.
    EXPECT_TRUE(est.update(10.0, 10.0, 100, 100.0f));
    EXPECT_NE(est.k(), 0.0f);
}

TEST(RangeBiasEstimator, ClampsStepAndMagnitude) {
    auto params = make_enabled_params();
    params.max_step = 0.01f;
    params.max_abs_k = 0.05f;
    RangeBiasEstimator est(params);

    // Raw step would be -1.0 but is clamped to -max_step per call.
    EXPECT_TRUE(est.update(10.0, 10.0, 100, 5.0f));
    EXPECT_NEAR(est.k(), -0.01f, 1e-6f);

    // Repeated updates accumulate but saturate at -max_abs_k.
    for (int i = 0; i < 100; ++i) {
        est.update(10.0, 10.0, 100, 5.0f);
    }
    EXPECT_NEAR(est.k(), -0.05f, 1e-6f);
}

TEST(RangeBiasEstimator, FixedPointReducesResidualCorrelation) {
    // Synthetic model matching the real residual sensitivity dy/dk = +x: the corrected residual is
    // y_i(k) = (k - b_true) * x_i, which vanishes at the fixed point k = b_true. The estimator should
    // drive the residual/regressor correlation to zero, converging k -> b_true.
    auto params = make_enabled_params();
    params.learning_rate = 0.5f;
    params.max_step = 1.0f;
    params.max_abs_k = 1.0f;
    RangeBiasEstimator est(params);

    const double b_true = 0.03;
    const double x_vals[5] = {1.0, 2.0, 3.0, 5.0, 8.0};
    for (int iter = 0; iter < 200; ++iter) {
        double s_xy = 0.0;
        double s_xx = 0.0;
        const double k = static_cast<double>(est.k());
        for (double x : x_vals) {
            const double y = (k - b_true) * x;  // residual after applying current k
            s_xy += x * y;
            s_xx += x * x;
        }
        est.update(s_xy, s_xx, 100, 100.0f);
    }
    EXPECT_NEAR(static_cast<double>(est.k()), b_true, 1e-4);
}

TEST(RangeBiasEstimator, ResetClearsState) {
    RangeBiasEstimator est(make_enabled_params());
    est.update(2.0, 4.0, 100, 5.0f);
    EXPECT_NE(est.k(), 0.0f);
    est.reset();
    EXPECT_EQ(est.k(), 0.0f);
}
