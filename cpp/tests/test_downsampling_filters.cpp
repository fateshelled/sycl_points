#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "sycl_points/algorithms/filter/polar_downsampling.hpp"
#include "sycl_points/algorithms/filter/voxel_downsampling.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace {

class DownsamplingFilterTest : public ::testing::Test {
  protected:
    void SetUp() override {
        device_ = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        queue_ = std::make_unique<sycl_points::sycl_utils::DeviceQueue>(device_);
    }

    sycl::device device_;
    std::unique_ptr<sycl_points::sycl_utils::DeviceQueue> queue_;
};

TEST_F(DownsamplingFilterTest, VoxelGridUsesMedianIntensityWithSortAggregation) {
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(5);
    cpu_cloud.rgb->resize(5);
    cpu_cloud.intensities->resize(5);
    cpu_cloud.timestamp_offsets->resize(5);

    (*cpu_cloud.points)[0] = PointType(0.10f, 0.00f, 0.00f, 1.0f);
    (*cpu_cloud.points)[1] = PointType(0.40f, 0.00f, 0.00f, 1.0f);
    (*cpu_cloud.points)[2] = PointType(1.10f, 0.00f, 0.00f, 1.0f);
    (*cpu_cloud.points)[3] = PointType(1.40f, 0.00f, 0.00f, 1.0f);
    (*cpu_cloud.points)[4] = PointType(0.20f, 0.00f, 0.00f, 1.0f);

    (*cpu_cloud.rgb)[0] = RGBType(10.0f, 20.0f, 30.0f, 1.0f);
    (*cpu_cloud.rgb)[1] = RGBType(20.0f, 40.0f, 60.0f, 1.0f);
    (*cpu_cloud.rgb)[2] = RGBType(30.0f, 60.0f, 90.0f, 1.0f);
    (*cpu_cloud.rgb)[3] = RGBType(50.0f, 70.0f, 90.0f, 1.0f);
    (*cpu_cloud.rgb)[4] = RGBType(70.0f, 80.0f, 90.0f, 1.0f);

    (*cpu_cloud.intensities)[0] = 1.0f;
    (*cpu_cloud.intensities)[1] = 3.0f;
    (*cpu_cloud.intensities)[2] = 5.0f;
    (*cpu_cloud.intensities)[3] = 7.0f;
    (*cpu_cloud.intensities)[4] = 100.0f;


    (*cpu_cloud.timestamp_offsets)[0] = 0.0f;
    (*cpu_cloud.timestamp_offsets)[1] = 2.0f;
    (*cpu_cloud.timestamp_offsets)[2] = 4.0f;
    (*cpu_cloud.timestamp_offsets)[3] = 6.0f;
    (*cpu_cloud.timestamp_offsets)[4] = 8.0f;

    PointCloudShared cloud(*queue_, cpu_cloud);
    PointCloudShared result(*queue_);
    algorithms::filter::VoxelGrid voxel_filter(*queue_, 1.0f);
    voxel_filter.set_min_voxel_count(2);

    voxel_filter.downsampling(cloud, result);

    ASSERT_EQ(result.size(), 1U);
    ASSERT_TRUE(result.has_rgb());
    ASSERT_TRUE(result.has_intensity());
    ASSERT_TRUE(result.has_timestamps());

    const size_t first = 0;

    EXPECT_NEAR((*result.points)[first].x(), 0.233333f, 1e-5f);

    EXPECT_NEAR((*result.intensities)[first], 3.0f, 1e-5f);

    EXPECT_NEAR((*result.timestamp_offsets)[first], 3.333333f, 1e-5f);

    EXPECT_NEAR((*result.rgb)[first].x(), 33.333333f, 1e-5f);
    EXPECT_NEAR((*result.rgb)[first].y(), 46.666667f, 1e-5f);
    EXPECT_NEAR((*result.rgb)[first].z(), 60.0f, 1e-5f);
}

TEST_F(DownsamplingFilterTest, PolarGridUsesMedianIntensityWithSortAggregation) {
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(5);
    cpu_cloud.intensities->resize(5);

    // Two groups in distance bins [1,2) and [2,3), and one sparse group to be removed.
    (*cpu_cloud.points)[0] = PointType(1.10f, 0.00f, 0.00f, 1.0f);
    (*cpu_cloud.points)[1] = PointType(1.40f, 0.00f, 0.00f, 1.0f);
    (*cpu_cloud.points)[2] = PointType(2.10f, 0.00f, 0.00f, 1.0f);
    (*cpu_cloud.points)[3] = PointType(2.30f, 0.00f, 0.00f, 1.0f);
    (*cpu_cloud.points)[4] = PointType(2.40f, 0.00f, 0.00f, 1.0f);

    (*cpu_cloud.intensities)[0] = 2.0f;
    (*cpu_cloud.intensities)[1] = 4.0f;
    (*cpu_cloud.intensities)[2] = 6.0f;
    (*cpu_cloud.intensities)[3] = 10.0f;
    (*cpu_cloud.intensities)[4] = 100.0f;


    PointCloudShared cloud(*queue_, cpu_cloud);
    PointCloudShared result(*queue_);

    algorithms::filter::PolarGrid polar_filter(*queue_, 1.0f, 3.14159265f, 3.14159265f,
                                               algorithms::CoordinateSystem::LIDAR);
    polar_filter.set_min_voxel_count(2);
    polar_filter.downsampling(cloud, result);

    ASSERT_EQ(result.size(), 1U);
    ASSERT_TRUE(result.has_intensity());

    const size_t first = 0;

    EXPECT_NEAR((*result.points)[first].x(), 2.20f, 1e-5f);

    EXPECT_NEAR((*result.intensities)[first], 10.0f, 1e-5f);
}

}  // namespace
}  // namespace sycl_points
