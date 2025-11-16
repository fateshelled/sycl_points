#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <sycl_points/deskew/imu_preintegration.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {
namespace {

/// @brief Helper to construct IMU sample timestamps from floating seconds.
IMUData CreateIMUSample(double time_seconds, const Eigen::Vector3f &angular_velocity) {
    const int32_t sec = static_cast<int32_t>(std::floor(time_seconds));
    const uint32_t nanosec = static_cast<uint32_t>(std::round((time_seconds - static_cast<double>(sec)) * 1e9));
    return IMUData(sec, nanosec, angular_velocity, Eigen::Vector3f::Zero());
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

    for (size_t i = 0; i <= static_cast<size_t>(kTotalTime / kDt); ++i) {
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
        Eigen::Vector3f rotated_point = orientation * reference_points[i];

        PointType point;
        point << rotated_point.x(), rotated_point.y(), rotated_point.z(), 1.0f;
        cloud.points->push_back(point);
    }

    ASSERT_TRUE(cloud.has_timestamps());
    ASSERT_EQ(cloud.size(), reference_points.size());

    const bool success = DeskewPointCloudRotations(cloud, imu_samples);
    ASSERT_TRUE(success);

    for (size_t i = 0; i < reference_points.size(); ++i) {
        const Eigen::Vector3f corrected_point = (*cloud.points)[i].head<3>();
        const Eigen::Vector3f expected_point = reference_points[i];
        EXPECT_NEAR((corrected_point - expected_point).norm(), 0.0f, 1e-5f);
    }
}

}  // namespace
}  // namespace sycl_points

