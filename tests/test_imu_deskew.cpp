#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <sycl_points/deskew/imu_preintegration.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <vector>

namespace sycl_points {
namespace {

/// @brief Helper to construct IMU sample timestamps from floating seconds.
IMUData CreateIMUSample(double time_seconds, const Eigen::Vector3f& angular_velocity,
                        const Eigen::Vector3f& linear_acceleration = Eigen::Vector3f::Zero()) {
    const int32_t sec = static_cast<int32_t>(std::floor(time_seconds));
    const uint32_t nanosec = static_cast<uint32_t>(std::round((time_seconds - static_cast<double>(sec)) * 1e9));
    return IMUData(sec, nanosec, angular_velocity, linear_acceleration);
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
    cloud.start_time_ms = 0.0;

    const std::vector<Eigen::Vector3f> reference_points = {
        Eigen::Vector3f(1.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 1.0f, 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, 1.0f),
    };

    const std::vector<double> point_times = {0.0, 0.05, 0.1};

    for (size_t i = 0; i < reference_points.size(); ++i) {
        const double timestamp = point_times[i];
        cloud.timestamp_offsets->push_back(static_cast<TimestampOffset>(timestamp * 1e3));

        const float angle = static_cast<float>(kAngularVel.z() * timestamp);
        const Eigen::Quaternionf orientation(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));
        Eigen::Vector3f rotated_point = orientation.conjugate() * reference_points[i];

        PointType point;
        point << rotated_point.x(), rotated_point.y(), rotated_point.z(), 1.0f;
        cloud.points->push_back(point);
    }

    cloud.update_end_time();

    ASSERT_TRUE(cloud.has_timestamps());
    ASSERT_EQ(cloud.size(), reference_points.size());

    const bool success = deskew_point_cloud(cloud, imu_samples);
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
    cloud.start_time_ms = 0.0;

    const Eigen::Vector3f world_point(5.0f, 0.0f, 0.0f);
    const std::vector<double> point_times = {0.0, 0.5, 1.0};

    for (double timestamp : point_times) {
        const double translation = 0.5 * static_cast<double>(kAcceleration.x()) * timestamp * timestamp;
        const Eigen::Vector3f sensor_position(static_cast<float>(translation), 0.0f, 0.0f);
        const Eigen::Vector3f observed_point = world_point - sensor_position;

        cloud.timestamp_offsets->push_back(static_cast<TimestampOffset>(timestamp * 1e3));

        PointType point;
        point << observed_point.x(), observed_point.y(), observed_point.z(), 1.0f;
        cloud.points->push_back(point);
    }

    cloud.update_end_time();

    ASSERT_TRUE(cloud.has_timestamps());
    ASSERT_EQ(cloud.size(), point_times.size());

    const bool success = deskew_point_cloud(cloud, imu_samples);
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
        imu_samples.push_back(CreateIMUSample(timestamp, kAngularVel, kAcceleration));
    }

    const std::vector<IMUMotionState> motion_states = preintegrate_imu_motion(imu_samples);
    ASSERT_EQ(motion_states.size(), imu_samples.size());

    PointCloudCPU cloud;
    cloud.start_time_ms = 0.0;

    const Eigen::Vector3f world_point(2.0f, -1.0f, 0.5f);
    const std::vector<double> point_times = {0.0, 0.25, 0.5};

    for (double timestamp : point_times) {
        const IMUMotionState motion_state = interpolate_motion_state(imu_samples, motion_states, timestamp);

        // Observation in the sensor frame: undo translation then rotation.
        const Eigen::Vector3f observed_point =
            motion_state.orientation.conjugate() * (world_point - motion_state.position);

        cloud.timestamp_offsets->push_back(static_cast<TimestampOffset>(timestamp * 1e3));

        PointType point;
        point << observed_point.x(), observed_point.y(), observed_point.z(), 1.0f;
        cloud.points->push_back(point);
    }

    cloud.update_end_time();

    ASSERT_TRUE(cloud.has_timestamps());
    ASSERT_EQ(cloud.size(), point_times.size());

    const bool success = deskew_point_cloud(cloud, imu_samples);
    ASSERT_TRUE(success);

    for (size_t i = 0; i < point_times.size(); ++i) {
        const Eigen::Vector3f corrected_point = (*cloud.points)[i].head<3>();
        EXPECT_NEAR((corrected_point - world_point).norm(), 0.0f, 1e-5f);
    }
}

TEST(DeskewIMUTest, DeskewsPointCloudWithBiasAndGravity) {
    IMUDataContainerCPU imu_samples;
    constexpr double kDt = 0.01;
    constexpr double kTotalTime = 0.1;
    const Eigen::Vector3f kGyroBias(0.01f, -0.02f, 0.015f);
    const Eigen::Vector3f kAccelBias(0.2f, -0.1f, 0.05f);
    const Eigen::Vector3f kGravity(0.0f, 0.0f, -9.81f);

    for (size_t i = 0; i <= static_cast<size_t>(std::round(kTotalTime / kDt)); ++i) {
        const double timestamp = i * kDt;
        imu_samples.push_back(CreateIMUSample(timestamp, kGyroBias, kGravity + kAccelBias));
    }

    PointCloudCPU cloud;
    cloud.start_time_ms = 0.0;

    const std::vector<double> point_times = {0.0, 0.05, 0.1};
    const Eigen::Vector3f fixed_point(1.0f, -2.0f, 0.5f);

    for (double timestamp : point_times) {
        cloud.timestamp_offsets->push_back(static_cast<TimestampOffset>(timestamp * 1e3));

        PointType point;
        point << fixed_point.x(), fixed_point.y(), fixed_point.z(), 1.0f;
        cloud.points->push_back(point);
    }

    cloud.update_end_time();

    ASSERT_TRUE(cloud.has_timestamps());
    ASSERT_EQ(cloud.size(), point_times.size());

    const bool success = deskew_point_cloud(cloud, imu_samples, kGyroBias, kAccelBias, kGravity);
    ASSERT_TRUE(success);

    for (size_t i = 0; i < point_times.size(); ++i) {
        const Eigen::Vector3f corrected_point = (*cloud.points)[i].head<3>();
        EXPECT_NEAR((corrected_point - fixed_point).norm(), 0.0f, 1e-5f);
    }
}

}  // namespace
}  // namespace sycl_points
