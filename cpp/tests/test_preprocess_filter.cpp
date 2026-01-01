#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>

#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace {

TEST(PreprocessFilterTest, BoxFilterRemovesOutOfRangeAndKeepsAttributes) {
    sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(4);
    cpu_cloud.intensities->resize(4);

    (*cpu_cloud.points)[0] = PointType(0.5f, 0.0f, 0.0f, 1.0f);  // too close
    (*cpu_cloud.points)[1] = PointType(2.0f, 0.0f, 0.0f, 1.0f);  // kept
    (*cpu_cloud.points)[2] = PointType(0.0f, 0.0f, 4.0f, 1.0f);  // too far
    (*cpu_cloud.points)[3] = PointType(std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.0f, 1.0f);  // invalid

    (*cpu_cloud.intensities)[0] = 1.0f;
    (*cpu_cloud.intensities)[1] = 2.0f;
    (*cpu_cloud.intensities)[2] = 3.0f;
    (*cpu_cloud.intensities)[3] = 4.0f;

    PointCloudShared shared_cloud(queue, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(queue);

    filter.box_filter(shared_cloud, 1.0f, 3.0f);

    ASSERT_EQ(shared_cloud.size(), 1U);
    ASSERT_TRUE(shared_cloud.has_intensity());
    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].x(), 2.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.intensities)[0], 2.0f);
}

TEST(PreprocessFilterTest, RandomSamplingIsDeterministicWithSeed) {
    sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    PointCloudCPU cpu_cloud;
    const size_t num_points = 5;
    cpu_cloud.points->resize(num_points);
    cpu_cloud.intensities->resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        (*cpu_cloud.intensities)[i] = static_cast<float>(i);
    }

    PointCloudShared shared_cloud(queue, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(queue);
    filter.set_random_seed(42);

    filter.random_sampling(shared_cloud, 2);

    ASSERT_EQ(shared_cloud.size(), 2U);
    ASSERT_TRUE(shared_cloud.has_intensity());

    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].x(), 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[1].x(), 4.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.intensities)[0], 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.intensities)[1], 4.0f);
}

TEST(PreprocessFilterTest, RandomSamplingNoOpWhenSamplingCountTooLarge) {
    sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    (*cpu_cloud.points)[0] = PointType(0.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[1] = PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[2] = PointType(2.0f, 0.0f, 0.0f, 1.0f);

    PointCloudShared shared_cloud(queue, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(queue);

    filter.random_sampling(shared_cloud, 10);

    ASSERT_EQ(shared_cloud.size(), 3U);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].x(), 0.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[1].x(), 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[2].x(), 2.0f);
}

TEST(PreprocessFilterTest, FarthestPointSamplingSelectsSpreadPoints) {
    sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(4);
    (*cpu_cloud.points)[0] = PointType(0.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[1] = PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[2] = PointType(0.0f, 1.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[3] = PointType(1.0f, 1.0f, 0.0f, 1.0f);

    PointCloudShared shared_cloud(queue, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(queue);
    filter.set_random_seed(1234);

    filter.farthest_point_sampling(shared_cloud, 3);

    ASSERT_EQ(shared_cloud.size(), 3U);

    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].x(), 0.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].y(), 0.0f);

    EXPECT_FLOAT_EQ((*shared_cloud.points)[1].x(), 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[1].y(), 0.0f);

    EXPECT_FLOAT_EQ((*shared_cloud.points)[2].x(), 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[2].y(), 1.0f);
}

TEST(PreprocessFilterTest, AngleIncidenceFilterKeepsPointsWithinRange) {
    sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    cpu_cloud.normals->resize(3);

    (*cpu_cloud.points)[0] = PointType(1.0f, 0.0f, 0.0f, 1.0f);  // angle 0 -> remove
    (*cpu_cloud.points)[1] = PointType(1.0f, 1.0f, 0.0f, 1.0f);  // 45 degrees -> keep
    (*cpu_cloud.points)[2] = PointType(0.0f, 0.0f, 1.0f, 1.0f);  // 90 degrees -> remove

    (*cpu_cloud.normals)[0] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
    (*cpu_cloud.normals)[1] = Normal(0.0f, 1.0f, 0.0f, 0.0f);
    (*cpu_cloud.normals)[2] = Normal(0.0f, 1.0f, 0.0f, 0.0f);

    PointCloudShared shared_cloud(queue, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(queue);

    filter.angle_incidence_filter(shared_cloud, 0.2f, 1.2f);

    ASSERT_EQ(shared_cloud.size(), 1U);
    ASSERT_TRUE(shared_cloud.has_normal());

    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].x(), 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].y(), 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.normals)[0].x(), 0.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.normals)[0].y(), 1.0f);
}

TEST(PreprocessFilterTest, AngleIncidenceFilterThrowsWithoutNormalsOrCovs) {
    sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    (*cpu_cloud.points)[0] = PointType(1.0f, 0.0f, 0.0f, 1.0f);

    PointCloudShared shared_cloud(queue, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(queue);

    EXPECT_THROW(filter.angle_incidence_filter(shared_cloud, 0.1f, 1.0f), std::runtime_error);
}

TEST(PreprocessFilterTest, AngleIncidenceFilterValidatesAngles) {
    sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.normals->resize(1);
    (*cpu_cloud.points)[0] = PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.normals)[0] = Normal(1.0f, 0.0f, 0.0f, 0.0f);

    PointCloudShared shared_cloud(queue, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(queue);

    EXPECT_THROW(filter.angle_incidence_filter(shared_cloud, -0.1f, 0.5f), std::invalid_argument);
    EXPECT_THROW(filter.angle_incidence_filter(shared_cloud, 0.5f, 0.4f), std::invalid_argument);
    EXPECT_THROW(filter.angle_incidence_filter(shared_cloud, 0.1f, 2.0f), std::invalid_argument);
}

}  // namespace
}  // namespace sycl_points
