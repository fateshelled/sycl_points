#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace {

class PreprocessFilterTest : public ::testing::Test {
  protected:
    void SetUp() override {
        device_ = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        queue_ = std::make_unique<sycl_points::sycl_utils::DeviceQueue>(device_);
    }

    sycl::device device_;
    std::unique_ptr<sycl_points::sycl_utils::DeviceQueue> queue_;
};

TEST_F(PreprocessFilterTest, BoxFilterRemovesOutOfRangeAndKeepsAttributes) {

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

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.box_filter(shared_cloud, 1.0f, 3.0f);

    ASSERT_EQ(shared_cloud.size(), 1U);
    ASSERT_TRUE(shared_cloud.has_intensity());
    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].x(), 2.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.intensities)[0], 2.0f);
}

TEST_F(PreprocessFilterTest, RandomSamplingIsDeterministicWithSeed) {

    PointCloudCPU cpu_cloud;
    const size_t num_points = 5;
    cpu_cloud.points->resize(num_points);
    cpu_cloud.intensities->resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        (*cpu_cloud.intensities)[i] = static_cast<float>(i);
    }

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    PointCloudShared shared_cloud_repeat(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);
    algorithms::filter::PreprocessFilter filter_repeat(*queue_);
    filter.set_random_seed(42);
    filter_repeat.set_random_seed(42);

    filter.random_sampling(shared_cloud, 2);
    filter_repeat.random_sampling(shared_cloud_repeat, 2);

    ASSERT_EQ(shared_cloud.size(), 2U);
    ASSERT_EQ(shared_cloud_repeat.size(), 2U);
    ASSERT_TRUE(shared_cloud.has_intensity());
    ASSERT_TRUE(shared_cloud_repeat.has_intensity());

    auto build_sorted_pairs = [](const PointCloudShared& cloud) {
        std::vector<std::pair<PointType, float>> paired_data;
        paired_data.reserve(cloud.size());
        for (size_t i = 0; i < cloud.size(); ++i) {
            paired_data.emplace_back((*cloud.points)[i], (*cloud.intensities)[i]);
        }
        std::sort(paired_data.begin(), paired_data.end(), [](const auto& a, const auto& b) {
            return a.first.x() < b.first.x();
        });
        return paired_data;
    };

    const auto paired_data = build_sorted_pairs(shared_cloud);
    const auto paired_data_repeat = build_sorted_pairs(shared_cloud_repeat);

    ASSERT_EQ(paired_data.size(), paired_data_repeat.size());
    for (size_t i = 0; i < paired_data.size(); ++i) {
        EXPECT_FLOAT_EQ(paired_data[i].first.x(), paired_data_repeat[i].first.x());
        EXPECT_FLOAT_EQ(paired_data[i].second, paired_data_repeat[i].second);
    }
}

TEST_F(PreprocessFilterTest, RandomSamplingNoOpWhenSamplingCountTooLarge) {

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    (*cpu_cloud.points)[0] = PointType(0.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[1] = PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[2] = PointType(2.0f, 0.0f, 0.0f, 1.0f);

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.random_sampling(shared_cloud, 10);

    ASSERT_EQ(shared_cloud.size(), 3U);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].x(), 0.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[1].x(), 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[2].x(), 2.0f);
}

TEST_F(PreprocessFilterTest, RandomSamplingNoOpWhenSamplingCountEqualsSize) {

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    cpu_cloud.intensities->resize(3);
    (*cpu_cloud.points)[0] = PointType(0.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[1] = PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[2] = PointType(2.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.intensities)[0] = 0.5f;
    (*cpu_cloud.intensities)[1] = 1.5f;
    (*cpu_cloud.intensities)[2] = 2.5f;

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.random_sampling(shared_cloud, 3);

    ASSERT_EQ(shared_cloud.size(), 3U);
    ASSERT_TRUE(shared_cloud.has_intensity());
    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].x(), 0.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[1].x(), 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[2].x(), 2.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.intensities)[0], 0.5f);
    EXPECT_FLOAT_EQ((*shared_cloud.intensities)[1], 1.5f);
    EXPECT_FLOAT_EQ((*shared_cloud.intensities)[2], 2.5f);
}

TEST_F(PreprocessFilterTest, EmptyPointCloudIsNoOpForAllFilters) {

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->clear();
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.box_filter(shared_cloud, 1.0f, 3.0f);
    ASSERT_EQ(shared_cloud.size(), 0U);

    filter.random_sampling(shared_cloud, 2);
    ASSERT_EQ(shared_cloud.size(), 0U);

    filter.farthest_point_sampling(shared_cloud, 2);
    ASSERT_EQ(shared_cloud.size(), 0U);

    EXPECT_NO_THROW(filter.angle_incidence_filter(shared_cloud, 0.1f, 1.0f));
    ASSERT_EQ(shared_cloud.size(), 0U);
}

TEST_F(PreprocessFilterTest, FarthestPointSamplingSelectsSpreadPoints) {

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(4);
    (*cpu_cloud.points)[0] = PointType(0.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[1] = PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[2] = PointType(0.0f, 1.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[3] = PointType(1.0f, 1.0f, 0.0f, 1.0f);

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);
    filter.set_random_seed(1234);

    filter.farthest_point_sampling(shared_cloud, 3);

    ASSERT_EQ(shared_cloud.size(), 3U);

    const auto& points = *shared_cloud.points;
    const std::vector<PointType> input_points = {
        PointType(0.0f, 0.0f, 0.0f, 1.0f),
        PointType(1.0f, 0.0f, 0.0f, 1.0f),
        PointType(0.0f, 1.0f, 0.0f, 1.0f),
        PointType(1.0f, 1.0f, 0.0f, 1.0f),
    };

    auto is_input_point = [&input_points](const PointType& point) {
        return std::any_of(input_points.begin(), input_points.end(), [&point](const PointType& candidate) {
            return candidate.x() == point.x() && candidate.y() == point.y() && candidate.z() == point.z();
        });
    };

    for (const auto& point : points) {
        EXPECT_TRUE(is_input_point(point));
    }

    float max_distance = 0.0f;
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            const float dx = points[i].x() - points[j].x();
            const float dy = points[i].y() - points[j].y();
            const float distance = std::sqrt(dx * dx + dy * dy);
            max_distance = std::max(max_distance, distance);
        }
    }

    EXPECT_FLOAT_EQ(max_distance, std::sqrt(2.0f));
}

TEST_F(PreprocessFilterTest, AngleIncidenceFilterKeepsPointsWithinRange) {

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    cpu_cloud.normals->resize(3);

    (*cpu_cloud.points)[0] = PointType(1.0f, 0.0f, 0.0f, 1.0f);  // angle 0 -> remove
    (*cpu_cloud.points)[1] = PointType(1.0f, 1.0f, 0.0f, 1.0f);  // 45 degrees -> keep
    (*cpu_cloud.points)[2] = PointType(0.0f, 0.0f, 1.0f, 1.0f);  // 90 degrees -> remove

    (*cpu_cloud.normals)[0] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
    (*cpu_cloud.normals)[1] = Normal(0.0f, 1.0f, 0.0f, 0.0f);
    (*cpu_cloud.normals)[2] = Normal(0.0f, 1.0f, 0.0f, 0.0f);

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.angle_incidence_filter(shared_cloud, 0.2f, 1.2f);

    ASSERT_EQ(shared_cloud.size(), 1U);
    ASSERT_TRUE(shared_cloud.has_normal());

    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].x(), 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.points)[0].y(), 1.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.normals)[0].x(), 0.0f);
    EXPECT_FLOAT_EQ((*shared_cloud.normals)[0].y(), 1.0f);
}

TEST_F(PreprocessFilterTest, AngleIncidenceFilterThrowsWithoutNormalsOrCovs) {

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    (*cpu_cloud.points)[0] = PointType(1.0f, 0.0f, 0.0f, 1.0f);

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    EXPECT_THROW(filter.angle_incidence_filter(shared_cloud, 0.1f, 1.0f), std::runtime_error);
}

TEST_F(PreprocessFilterTest, AngleIncidenceFilterValidatesAngles) {

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.normals->resize(1);
    (*cpu_cloud.points)[0] = PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.normals)[0] = Normal(1.0f, 0.0f, 0.0f, 0.0f);

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    EXPECT_THROW(filter.angle_incidence_filter(shared_cloud, -0.1f, 0.5f), std::invalid_argument);
    EXPECT_THROW(filter.angle_incidence_filter(shared_cloud, 0.5f, 0.4f), std::invalid_argument);
    EXPECT_THROW(filter.angle_incidence_filter(shared_cloud, 0.1f, 2.0f), std::invalid_argument);
}

}  // namespace
}  // namespace sycl_points
