#include <gtest/gtest.h>

#include <deque>
#include <vector>

#include "sycl_points/pipeline/detail/keyframe_imu_history.hpp"

namespace {

using History = sycl_points::pipeline::graph_odometry::detail::KeyframeImuHistory;

sycl_points::imu::IMUMeasurement measurement(double timestamp) {
    sycl_points::imu::IMUMeasurement result;
    result.timestamp = timestamp;
    return result;
}

std::deque<sycl_points::imu::IMUMeasurement> measurements(double start, double end,
                                                          double step) {
    std::deque<sycl_points::imu::IMUMeasurement> result;
    for (double t = start; t <= end + 1e-9; t += step) result.push_back(measurement(t));
    return result;
}

TEST(KeyframeImuHistory, IgnoresMeasurementsBeforeSourceInitialization) {
    History history({1.0, 10});
    history.append(measurement(1.0));
    EXPECT_EQ(history.size(), 0u);
}

TEST(KeyframeImuHistory, ReconstructsInterpolatedBoundaryAndRepeatableWindow) {
    History history({2.0, 20});
    auto raw = measurements(0.0, 1.0, 0.2);
    ASSERT_TRUE(history.reset(0.3, raw));
    EXPECT_DOUBLE_EQ(history.source_timestamp(), 0.3);
    std::vector<sycl_points::imu::IMUMeasurement> first;
    std::vector<sycl_points::imu::IMUMeasurement> second;
    ASSERT_TRUE(history.build_window(0.9, first));
    ASSERT_TRUE(history.build_window(0.9, second));
    ASSERT_FALSE(first.empty());
    EXPECT_DOUBLE_EQ(first.front().timestamp, 0.3);
    EXPECT_EQ(first.size(), second.size());
}

TEST(KeyframeImuHistory, AcceptsSourceAtLatestAvailableSample) {
    History history({2.0, 20});
    const auto raw = measurements(0.0, 0.5, 0.1);
    EXPECT_TRUE(history.reset(0.5, raw));
    EXPECT_DOUBLE_EQ(history.source_timestamp(), 0.5);
    EXPECT_EQ(history.size(), 1u);
}

TEST(KeyframeImuHistory, SurvivesIndependentGlobalBufferTrimming) {
    History history({3.0, 50});
    auto raw = measurements(0.0, 1.0, 0.1);
    ASSERT_TRUE(history.reset(0.2, raw));
    while (!raw.empty() && raw.front().timestamp < 0.8) raw.pop_front();
    history.append(measurement(1.1));
    std::vector<sycl_points::imu::IMUMeasurement> window;
    EXPECT_TRUE(history.build_window(1.1, window));
    EXPECT_DOUBLE_EQ(window.front().timestamp, 0.2);
}

TEST(KeyframeImuHistory, HardLimitsRequireRecoveryAndKeepBoundedStorage) {
    History by_samples({10.0, 4});
    auto raw = measurements(0.0, 0.2, 0.1);
    ASSERT_TRUE(by_samples.reset(0.0, raw));
    by_samples.append(measurement(0.3));
    by_samples.append(measurement(0.4));
    EXPECT_TRUE(by_samples.overflowed());
    EXPECT_LE(by_samples.size(), 4u);
    EXPECT_EQ(by_samples.coverage(0.4), History::Coverage::recovery_required);

    History by_duration({0.3, 20});
    ASSERT_TRUE(by_duration.reset(0.0, raw));
    by_duration.append(measurement(0.4));
    EXPECT_TRUE(by_duration.overflowed());
    EXPECT_LT(by_duration.size(), raw.size() + 1);
    EXPECT_EQ(by_duration.coverage(0.4), History::Coverage::recovery_required);
}

TEST(KeyframeImuHistory, ForceWatermarkTriggersBeforeHardLimit) {
    History history({1.0, 100});
    auto raw = measurements(0.0, 0.9, 0.1);
    ASSERT_TRUE(history.reset(0.0, raw));
    EXPECT_FALSE(history.should_force_keyframe(0.7));
    EXPECT_TRUE(history.should_force_keyframe(0.8));
}

TEST(KeyframeImuHistory, ResetAfterRecoveryClearsOverflow) {
    History history({0.5, 10});
    auto raw = measurements(0.0, 0.4, 0.1);
    ASSERT_TRUE(history.reset(0.0, raw));
    history.append(measurement(0.6));
    ASSERT_TRUE(history.overflowed());

    raw = measurements(0.5, 0.8, 0.1);
    ASSERT_TRUE(history.reset(0.6, raw));
    EXPECT_FALSE(history.overflowed());
    EXPECT_EQ(history.coverage(0.8), History::Coverage::ready);
}

TEST(KeyframeImuHistory, FutureOverflowDoesNotRejectInterpolatedRebase) {
    History history({1.0, 2});
    std::deque<sycl_points::imu::IMUMeasurement> raw = {
        measurement(0.0), measurement(0.1), measurement(0.2), measurement(0.3)};
    ASSERT_TRUE(history.reset_allowing_future_overflow(0.05, raw));
    EXPECT_TRUE(history.overflowed());
    EXPECT_EQ(history.coverage(0.2), History::Coverage::recovery_required);

    raw = {measurement(0.15), measurement(0.2)};
    ASSERT_TRUE(history.reset_allowing_future_overflow(0.175, raw));
    EXPECT_FALSE(history.overflowed());
    EXPECT_EQ(history.coverage(0.2), History::Coverage::ready);
}

TEST(KeyframeImuHistory, DistinguishesFutureWaitFromRecovery) {
    History history({2.0, 20});
    auto raw = measurements(0.0, 0.5, 0.1);
    ASSERT_TRUE(history.reset(0.0, raw));
    EXPECT_EQ(history.coverage(0.6), History::Coverage::waiting_for_future);
}

}  // namespace
