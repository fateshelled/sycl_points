#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <sycl_points/deskew/imu_preintegration.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {
namespace {

/// @brief Helper to construct IMU sample timestamps from floating seconds.
IMUData CreateIMUSample(double time_seconds, const Eigen::Vector3f &angular_velocity,
                        const Eigen::Vector3f &linear_acceleration = Eigen::Vector3f::Zero()) {
    const int32_t sec = static_cast<int32_t>(std::floor(time_seconds));
    const uint32_t nanosec = static_cast<uint32_t>(std::round((time_seconds - static_cast<double>(sec)) * 1e9));
    return IMUData(sec, nanosec, angular_velocity, linear_acceleration);
}

double ClampRatio(double value, double min_value, double max_value) {
    return std::max(min_value, std::min(value, max_value));
}

IMUMotionState InterpolateMotionState(const IMUDataContainerCPU &imu_samples,
                                     const std::vector<IMUMotionState> &motion_states,
                                     double timestamp) {
    const auto comparator = [](const IMUData &sample, double t) { return sample.timestamp_seconds() < t; };

    auto upper = std::lower_bound(imu_samples.begin(), imu_samples.end(), timestamp, comparator);
    if (upper == imu_samples.begin()) {
        return motion_states.front();
    }
    if (upper == imu_samples.end()) {
        return motion_states.back();
    }

    const size_t next_idx = static_cast<size_t>(std::distance(imu_samples.begin(), upper));
    const size_t prev_idx = next_idx - 1;

    const double t0 = imu_samples[prev_idx].timestamp_seconds();
    const double t1 = imu_samples[next_idx].timestamp_seconds();
    const double ratio = (t1 - t0) <= 0.0 ? 0.0
                                          : ClampRatio((timestamp - t0) / (t1 - t0), 0.0, 1.0);

    IMUMotionState interpolated_state;
    interpolated_state.orientation = motion_states[prev_idx].orientation.slerp(
        static_cast<float>(ratio), motion_states[next_idx].orientation);
    interpolated_state.position = motion_states[prev_idx].position +
                                  static_cast<float>(ratio) *
                                      (motion_states[next_idx].position - motion_states[prev_idx].position);
    interpolated_state.velocity = motion_states[prev_idx].velocity +
                                  static_cast<float>(ratio) *
                                      (motion_states[next_idx].velocity - motion_states[prev_idx].velocity);
    return interpolated_state;
}

TEST(DeskewIMUTest, PreintegratesConstantRotation) {
    constexpr double kDt = 0.1;
    constexpr double kTotalTime = 1.0;
    const Eigen::Vector3f kAngularVel(0.0f, 0.0f, static_cast<float>(M_PI));

    IMUDataContainerCPU imu_samples;
    for (size_t i = 0; i <= static_cast<size_t>(std::round(kTotalTime / kDt)); ++i) {
        const double timestamp = i * kDt;
        imu_samples.push_back(CreateIMUSample(timestamp, kAngularVel));
    }

    const std::vector<Eigen::Quaternionf> orientations = PreintegrateIMURotations(imu_samples);
    ASSERT_EQ(orientations.size(), imu_samples.size());

    EXPECT_NEAR(orientations.front().w(), 1.0f, 1e-6f);
    EXPECT_NEAR(orientations.front().vec().norm(), 0.0f, 1e-6f);

    const float expected_angle = static_cast<float>(kAngularVel.z() * kTotalTime);
    const Eigen::Quaternionf expected_orientation(Eigen::AngleAxisf(expected_angle, Eigen::Vector3f::UnitZ()));
    const float quaternion_dot = std::abs(expected_orientation.dot(orientations.back()));
    EXPECT_NEAR(quaternion_dot, 1.0f, 1e-5f);
}

TEST(DeskewIMUTest, DeskewsPointCloudWithRotationalMotion) {
    IMUDataContainerCPU imu_samples;
    constexpr double kDt = 0.025;
    constexpr double kTotalTime = 0.1;
    const Eigen::Vector3f kAngularVel(0.0f, 0.0f, static_cast<float>(M_PI));

    for (size_t i = 0; i <= static_cast<size_t>(std::round(kTotalTime / kDt)); ++i) {
        const double timestamp = i * kDt;
        imu_samples.push_back(CreateIMUSample(timestamp, kAngularVel));
    }

    PointCloudCPU cloud;
    cloud.timestamp_base_ns = 0;

    const std::vector<Eigen::Vector3f> reference_points = {
        Eigen::Vector3f(1.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 1.0f, 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, 1.0f),
    };

    const std::vector<double> point_times = {0.0, 0.05, 0.1};

    for (size_t i = 0; i < reference_points.size(); ++i) {
        const double timestamp = point_times[i];
        cloud.timestamp_offsets->push_back(static_cast<TimestampOffset>(timestamp * 1e9));

        const float angle = static_cast<float>(kAngularVel.z() * timestamp);
        const Eigen::Quaternionf orientation(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));
        Eigen::Vector3f rotated_point =
            orientation.conjugate() * reference_points[i];

        PointType point;
        point << rotated_point.x(), rotated_point.y(), rotated_point.z(), 1.0f;
        cloud.points->push_back(point);
    }

    ASSERT_TRUE(cloud.has_timestamps());
    ASSERT_EQ(cloud.size(), reference_points.size());

    const bool success = DeskewPointCloud(cloud, imu_samples);
    ASSERT_TRUE(success);

    for (size_t i = 0; i < reference_points.size(); ++i) {
        const Eigen::Vector3f corrected_point = (*cloud.points)[i].head<3>();
        const Eigen::Vector3f expected_point = reference_points[i];
        EXPECT_NEAR((corrected_point - expected_point).norm(), 0.0f, 1e-5f);
    }
}

TEST(DeskewIMUTest, DeskewsPointCloudWithTranslation) {
    IMUDataContainerCPU imu_samples;
    constexpr double kDt = 0.1;
    constexpr double kTotalTime = 1.0;
    const Eigen::Vector3f kAcceleration(1.0f, 0.0f, 0.0f);

    for (size_t i = 0; i <= static_cast<size_t>(std::round(kTotalTime / kDt)); ++i) {
        const double timestamp = i * kDt;
        imu_samples.push_back(CreateIMUSample(timestamp, Eigen::Vector3f::Zero(), kAcceleration));
    }

    PointCloudCPU cloud;
    cloud.timestamp_base_ns = 0;

    const Eigen::Vector3f world_point(5.0f, 0.0f, 0.0f);
    const std::vector<double> point_times = {0.0, 0.5, 1.0};

    for (double timestamp : point_times) {
        const double translation = 0.5 * static_cast<double>(kAcceleration.x()) * timestamp * timestamp;
        const Eigen::Vector3f sensor_position(static_cast<float>(translation), 0.0f, 0.0f);
        const Eigen::Vector3f observed_point = world_point - sensor_position;

        cloud.timestamp_offsets->push_back(static_cast<TimestampOffset>(timestamp * 1e9));

        PointType point;
        point << observed_point.x(), observed_point.y(), observed_point.z(), 1.0f;
        cloud.points->push_back(point);
    }

    ASSERT_TRUE(cloud.has_timestamps());
    ASSERT_EQ(cloud.size(), point_times.size());

    const bool success = DeskewPointCloud(cloud, imu_samples);
    ASSERT_TRUE(success);

    for (size_t i = 0; i < point_times.size(); ++i) {
        const Eigen::Vector3f corrected_point = (*cloud.points)[i].head<3>();
        EXPECT_NEAR((corrected_point - world_point).norm(), 0.0f, 1e-5f);
    }
}

TEST(DeskewIMUTest, DeskewsPointCloudWithCombinedMotion) {
    IMUDataContainerCPU imu_samples;
    constexpr double kDt = 0.05;
    constexpr double kTotalTime = 0.5;
    const Eigen::Vector3f kAngularVel(0.0f, 0.0f, static_cast<float>(M_PI));
    const Eigen::Vector3f kAcceleration(1.0f, 0.5f, 0.0f);

    for (size_t i = 0; i <= static_cast<size_t>(std::round(kTotalTime / kDt)); ++i) {
        const double timestamp = i * kDt;
        imu_samples.push_back(
            CreateIMUSample(timestamp, kAngularVel, kAcceleration));
    }

    const std::vector<IMUMotionState> motion_states =
        PreintegrateIMUMotion(imu_samples);
    ASSERT_EQ(motion_states.size(), imu_samples.size());

    PointCloudCPU cloud;
    cloud.timestamp_base_ns = 0;

    const Eigen::Vector3f world_point(2.0f, -1.0f, 0.5f);
    const std::vector<double> point_times = {0.0, 0.25, 0.5};

    for (double timestamp : point_times) {
        const IMUMotionState motion_state =
            InterpolateMotionState(imu_samples, motion_states, timestamp);

        // Observation in the sensor frame: undo translation then rotation.
        const Eigen::Vector3f observed_point =
            motion_state.orientation.conjugate() *
            (world_point - motion_state.position);

        cloud.timestamp_offsets->push_back(
            static_cast<TimestampOffset>(timestamp * 1e9));

        PointType point;
        point << observed_point.x(), observed_point.y(), observed_point.z(), 1.0f;
        cloud.points->push_back(point);
    }

    ASSERT_TRUE(cloud.has_timestamps());
    ASSERT_EQ(cloud.size(), point_times.size());

    const bool success = DeskewPointCloud(cloud, imu_samples);
    ASSERT_TRUE(success);

    for (size_t i = 0; i < point_times.size(); ++i) {
        const Eigen::Vector3f corrected_point = (*cloud.points)[i].head<3>();
        EXPECT_NEAR((corrected_point - world_point).norm(), 0.0f, 1e-5f);
    }
}

}  // namespace
}  // namespace sycl_points

