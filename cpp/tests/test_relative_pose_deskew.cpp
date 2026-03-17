#include <gtest/gtest.h>

#include "sycl_points/algorithms/deskew/relative_pose_deskew.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points::algorithms::deskew {
namespace {

TEST(RelativePoseDeskewTest, DeskewsPointsWithConstantVelocity) {
    const Eigen::Transform<float, 3, 1> start_pose = Eigen::Transform<float, 3, 1>::Identity();
    Eigen::Transform<float, 3, 1> end_pose = Eigen::Transform<float, 3, 1>::Identity();
    end_pose.translation() = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    end_pose.linear() = Eigen::AngleAxisf(static_cast<float>(M_PI_2), Eigen::Vector3f::UnitZ()).toRotationMatrix();

    const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(start_pose.inverse() * end_pose);

    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    PointCloudShared cloud(queue);
    cloud.start_time_ms = 0.0;
    const std::vector<double> timestamps = {0.0, 0.5, 1.0};
    const Eigen::Vector3f world_point(1.0f, 1.0f, 0.0f);
    const Eigen::Vector3f world_normal = Eigen::Vector3f::UnitZ();
    Eigen::Matrix3f world_covariance = Eigen::Matrix3f::Zero();
    world_covariance.diagonal() = Eigen::Vector3f(0.01f, 0.02f, 0.03f);

    for (double timestamp : timestamps) {
        const double ratio = timestamp / 1.0;
        const Eigen::Isometry3f pose(start_pose *
                                     eigen_utils::lie::se3_exp((delta_twist * static_cast<float>(ratio)).eval()));
        const Eigen::Vector3f observed_point = pose.inverse() * world_point;
        const Eigen::Vector3f observed_normal = pose.linear().transpose() * world_normal;
        const Eigen::Matrix3f observed_covariance = pose.linear().transpose() * world_covariance * pose.linear();

        cloud.timestamp_offsets->push_back(static_cast<TimestampOffset>(timestamp * 1e3));
        PointType point;
        point << observed_point.x(), observed_point.y(), observed_point.z(), 1.0f;
        cloud.points->push_back(point);

        Normal normal;
        normal.head<3>() = observed_normal;
        normal.w() = 0.0f;
        cloud.normals->push_back(normal);

        Covariance covariance = Covariance::Zero();
        covariance.topLeftCorner<3, 3>() = observed_covariance;
        cloud.covs->push_back(covariance);
    }

    cloud.end_time_ms = cloud.start_time_ms + static_cast<double>(timestamps.back()) * 1e3;
    ASSERT_TRUE(cloud.has_timestamps());
    ASSERT_TRUE(cloud.has_normal());
    ASSERT_TRUE(cloud.has_cov());

    PointCloudShared deskewed_cloud(queue);
    const bool success = deskew_point_cloud_constant_velocity(cloud, deskewed_cloud, start_pose, end_pose);
    ASSERT_TRUE(success);

    for (size_t idx = 0; idx < deskewed_cloud.size(); ++idx) {
        const Eigen::Vector3f corrected_point = (*deskewed_cloud.points)[idx].head<3>();
        EXPECT_NEAR((corrected_point - world_point).norm(), 0.0f, 1e-5f);

        const Eigen::Vector3f corrected_normal = (*deskewed_cloud.normals)[idx].head<3>();
        EXPECT_NEAR((corrected_normal - world_normal).norm(), 0.0f, 1e-5f);

        const Eigen::Matrix3f corrected_covariance = (*deskewed_cloud.covs)[idx].topLeftCorner<3, 3>();
        const Eigen::Matrix3f diff = corrected_covariance - world_covariance;
        EXPECT_NEAR(diff.norm(), 0.0f, 1e-6f);
    }
}

TEST(RelativePoseDeskewTest, PreservesAndRotatesPhotometricGradients) {
    const Eigen::Transform<float, 3, 1> start_pose = Eigen::Transform<float, 3, 1>::Identity();
    Eigen::Transform<float, 3, 1> end_pose = Eigen::Transform<float, 3, 1>::Identity();
    end_pose.translation() = Eigen::Vector3f(0.5f, -0.25f, 0.1f);
    end_pose.linear() = Eigen::AngleAxisf(static_cast<float>(M_PI_2), Eigen::Vector3f::UnitZ()).toRotationMatrix();

    const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(start_pose.inverse() * end_pose);

    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    PointCloudShared cloud(queue);
    cloud.start_time_ms = 0.0;

    const std::vector<double> timestamps = {0.0, 0.5, 1.0};
    const Eigen::Vector3f world_point(1.5f, -0.3f, 0.7f);
    const ColorGradient world_color_gradient =
        (ColorGradient() << 1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 4.0f, 0.25f, -2.0f, 1.5f).finished();
    const IntensityGradient world_intensity_gradient(0.2f, -0.4f, 0.8f);

    for (double timestamp : timestamps) {
        const double ratio = timestamp / 1.0;
        const Eigen::Isometry3f pose(start_pose *
                                     eigen_utils::lie::se3_exp((delta_twist * static_cast<float>(ratio)).eval()));
        const Eigen::Matrix3f rotation_sensor_from_world = pose.linear().transpose();

        cloud.timestamp_offsets->push_back(static_cast<TimestampOffset>(timestamp * 1e3));

        PointType point;
        point.head<3>() = pose.inverse() * world_point;
        point.w() = 1.0f;
        cloud.points->push_back(point);

        cloud.color_gradients->push_back(world_color_gradient * pose.linear());
        cloud.intensity_gradients->push_back(rotation_sensor_from_world * world_intensity_gradient);
    }

    cloud.end_time_ms = cloud.start_time_ms + static_cast<double>(timestamps.back()) * 1e3;
    ASSERT_TRUE(cloud.has_timestamps());
    ASSERT_TRUE(cloud.has_color_gradient());
    ASSERT_TRUE(cloud.has_intensity_gradient());

    PointCloudShared deskewed_cloud(queue);
    const bool success = deskew_point_cloud_constant_velocity(cloud, deskewed_cloud, start_pose, end_pose);
    ASSERT_TRUE(success);
    ASSERT_TRUE(deskewed_cloud.has_color_gradient());
    ASSERT_TRUE(deskewed_cloud.has_intensity_gradient());

    for (size_t idx = 0; idx < deskewed_cloud.size(); ++idx) {
        const ColorGradient corrected_color_gradient = (*deskewed_cloud.color_gradients)[idx];
        EXPECT_NEAR((corrected_color_gradient - world_color_gradient).norm(), 0.0f, 1e-5f);

        const IntensityGradient corrected_intensity_gradient = (*deskewed_cloud.intensity_gradients)[idx];
        EXPECT_NEAR((corrected_intensity_gradient - world_intensity_gradient).norm(), 0.0f, 1e-5f);
    }
}

TEST(RelativePoseDeskewTest, HandlesNonPositiveScanDuration) {
    const Eigen::Transform<float, 3, 1> start_pose = Eigen::Transform<float, 3, 1>::Identity();
    const Eigen::Transform<float, 3, 1> end_pose = Eigen::Transform<float, 3, 1>::Identity();

    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    PointCloudShared cloud(queue);
    cloud.start_time_ms = 0.0;
    cloud.timestamp_offsets->push_back(0);
    PointType point;
    point << 0.0f, 0.0f, 0.0f, 1.0f;
    cloud.points->push_back(point);
    cloud.end_time_ms = cloud.start_time_ms;

    PointCloudShared deskewed_cloud(queue);
    EXPECT_FALSE(deskew_point_cloud_constant_velocity(cloud, deskewed_cloud, start_pose, end_pose));
}

}  // namespace
}  // namespace sycl_points::algorithms::deskew
