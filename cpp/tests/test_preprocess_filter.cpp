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

TEST_F(PreprocessFilterTest, NormalHistogramSamplingNoOpWhenSamplingCountExceedsSize) {
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    cpu_cloud.normals->resize(3);
    for (size_t i = 0; i < 3; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        (*cpu_cloud.normals)[i] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.normal_histogram_sampling(shared_cloud, 10);

    ASSERT_EQ(shared_cloud.size(), 3U);
}

TEST_F(PreprocessFilterTest, NormalHistogramSamplingNoOpWhenSamplingCountEqualsSize) {
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    cpu_cloud.normals->resize(3);
    for (size_t i = 0; i < 3; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        (*cpu_cloud.normals)[i] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.normal_histogram_sampling(shared_cloud, 3);

    ASSERT_EQ(shared_cloud.size(), 3U);
}

TEST_F(PreprocessFilterTest, NormalHistogramSamplingCopiesSourceWhenSamplingCountExceedsSize) {
    PointCloudCPU src_cpu;
    src_cpu.points->resize(3);
    src_cpu.normals->resize(3);
    src_cpu.intensities->resize(3);
    for (size_t i = 0; i < 3; ++i) {
        (*src_cpu.points)[i] = PointType(static_cast<float>(i + 1), static_cast<float>(i + 2), 0.0f, 1.0f);
        (*src_cpu.normals)[i] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
        (*src_cpu.intensities)[i] = static_cast<float>(i) * 0.5f;
    }

    PointCloudShared source(*queue_, src_cpu);
    PointCloudCPU dummy_cpu;
    dummy_cpu.points->resize(1);
    (*dummy_cpu.points)[0] = PointType(999.0f, 999.0f, 999.0f, 1.0f);
    PointCloudShared output(*queue_, dummy_cpu);

    algorithms::filter::PreprocessFilter filter(*queue_);
    filter.normal_histogram_sampling(source, output, 10);

    ASSERT_EQ(output.size(), source.size());
    ASSERT_TRUE(output.has_normal());
    ASSERT_TRUE(output.has_intensity());
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_TRUE((*output.points)[i].isApprox((*source.points)[i], 1e-6f));
        EXPECT_TRUE((*output.normals)[i].isApprox((*source.normals)[i], 1e-6f));
        EXPECT_NEAR((*output.intensities)[i], (*source.intensities)[i], 1e-6f);
    }
}

TEST_F(PreprocessFilterTest, NormalHistogramSamplingThrowsOnZeroBinSize) {
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(8);
    cpu_cloud.normals->resize(8);
    for (size_t i = 0; i < 8; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        (*cpu_cloud.normals)[i] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
    }

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    EXPECT_THROW(filter.normal_histogram_sampling(shared_cloud, 4, 0, 4), std::invalid_argument);
    EXPECT_THROW(filter.normal_histogram_sampling(shared_cloud, 4, 4, 0), std::invalid_argument);
}

TEST_F(PreprocessFilterTest, NormalHistogramSamplingThrowsOnBinCountOverflow) {
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(8);
    cpu_cloud.normals->resize(8);
    for (size_t i = 0; i < 8; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        (*cpu_cloud.normals)[i] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
    }

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    EXPECT_THROW(
        filter.normal_histogram_sampling(shared_cloud, 4, std::numeric_limits<size_t>::max(), 2), std::overflow_error);
}

TEST_F(PreprocessFilterTest, NormalHistogramSamplingThrowsWithoutNormalsOrCovs) {
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(5);
    for (size_t i = 0; i < 5; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    EXPECT_THROW(filter.normal_histogram_sampling(shared_cloud, 3), std::runtime_error);
}

TEST_F(PreprocessFilterTest, NormalHistogramSamplingRetainsCorrectCount) {
    const size_t N = 100;
    const size_t sampling_num = 20;
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(N);
    cpu_cloud.normals->resize(N);
    for (size_t i = 0; i < N; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        // Spread normals across 4 cardinal equatorial directions
        const float angle = static_cast<float>(i) / N * 2.0f * eigen_utils::PI;
        (*cpu_cloud.normals)[i] = Normal(std::cos(angle), std::sin(angle), 0.0f, 0.0f);
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.normal_histogram_sampling(shared_cloud, sampling_num);

    ASSERT_EQ(shared_cloud.size(), sampling_num);
}

TEST_F(PreprocessFilterTest, NormalHistogramSamplingPreservesAttributes) {
    const size_t N = 40;
    const size_t sampling_num = 10;
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(N);
    cpu_cloud.normals->resize(N);
    cpu_cloud.intensities->resize(N);
    for (size_t i = 0; i < N; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        (*cpu_cloud.normals)[i] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
        (*cpu_cloud.intensities)[i] = static_cast<float>(i) * 0.1f;
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.normal_histogram_sampling(shared_cloud, sampling_num);

    ASSERT_EQ(shared_cloud.size(), sampling_num);
    ASSERT_TRUE(shared_cloud.has_normal());
    ASSERT_TRUE(shared_cloud.has_intensity());
    // Each selected point's intensity must match a valid intensity from the original cloud
    for (size_t i = 0; i < shared_cloud.size(); ++i) {
        const float x = (*shared_cloud.points)[i].x();
        const float expected_intensity = x * 0.1f;
        EXPECT_NEAR((*shared_cloud.intensities)[i], expected_intensity, 1e-5f);
    }
}

TEST_F(PreprocessFilterTest, NormalHistogramSamplingCoversAllBins) {
    // 4 groups of 20 points, each with normals in one of the 4 cardinal equatorial directions.
    // With 8 longitude bins and 1 latitude bin, the 4 groups map to bins 0, 2, 4, 6.
    // Requesting sampling_num=4 should yield exactly 1 point from each group via round-robin.
    const size_t group_size = 20;
    const size_t N = 4 * group_size;
    const size_t sampling_num = 4;

    const std::array<Normal, 4> group_normals = {
        Normal(1.0f, 0.0f, 0.0f, 0.0f),   // lon_bin=0 with 8 bins
        Normal(0.0f, 1.0f, 0.0f, 0.0f),   // lon_bin=2
        Normal(-1.0f, 0.0f, 0.0f, 0.0f),  // lon_bin=4
        Normal(0.0f, -1.0f, 0.0f, 0.0f),  // lon_bin=6
    };

    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(N);
    cpu_cloud.normals->resize(N);
    for (size_t g = 0; g < 4; ++g) {
        for (size_t j = 0; j < group_size; ++j) {
            const size_t i = g * group_size + j;
            (*cpu_cloud.points)[i] = PointType(static_cast<float>(g * 100 + j), 0.0f, 0.0f, 1.0f);
            (*cpu_cloud.normals)[i] = group_normals[g];
        }
    }

    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.normal_histogram_sampling(shared_cloud, sampling_num, /*longitude_bins=*/8, /*latitude_bins=*/1);

    ASSERT_EQ(shared_cloud.size(), sampling_num);

    // Each of the 4 direction groups must be represented in the output.
    std::array<bool, 4> group_covered = {false, false, false, false};
    for (size_t i = 0; i < shared_cloud.size(); ++i) {
        const auto& n = (*shared_cloud.normals)[i];
        bool matched = false;
        for (size_t g = 0; g < 4; ++g) {
            if (n.isApprox(group_normals[g], 1e-3f)) {
                group_covered[g] = true;
                matched = true;
                break;
            }
        }
        EXPECT_TRUE(matched) << "Sampled normal does not match any expected group normal.";
    }
    for (size_t g = 0; g < 4; ++g) {
        EXPECT_TRUE(group_covered[g]) << "Direction group " << g << " not covered in output";
    }
}

TEST_F(PreprocessFilterTest, NormalHistogramSamplingWorksWithCovariancesOnly) {
    // Covariance matrix C = diag(0, 1, 1) → smallest eigenvalue eigenvector = (1, 0, 0) → normal along +x
    const size_t N = 40;
    const size_t sampling_num = 10;
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(N);
    cpu_cloud.covs->resize(N);
    for (size_t i = 0; i < N; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        Covariance cov = Covariance::Zero();
        // Small eigenvalue in x → normal along +x
        cov(0, 0) = 1e-4f;
        cov(1, 1) = 1.0f;
        cov(2, 2) = 1.0f;
        (*cpu_cloud.covs)[i] = cov;
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    EXPECT_NO_THROW(filter.normal_histogram_sampling(shared_cloud, sampling_num));
    ASSERT_EQ(shared_cloud.size(), sampling_num);
}

TEST_F(PreprocessFilterTest, SphericalFibonacciSamplingNoOpWhenSamplingCountExceedsSize) {
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    cpu_cloud.normals->resize(3);
    for (size_t i = 0; i < 3; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        (*cpu_cloud.normals)[i] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.spherical_fibonacci_sampling(shared_cloud, 10);

    ASSERT_EQ(shared_cloud.size(), 3U);
}

TEST_F(PreprocessFilterTest, SphericalFibonacciSamplingNoOpWhenSamplingCountEqualsSize) {
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    cpu_cloud.normals->resize(3);
    for (size_t i = 0; i < 3; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        (*cpu_cloud.normals)[i] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.spherical_fibonacci_sampling(shared_cloud, 3);

    ASSERT_EQ(shared_cloud.size(), 3U);
}

TEST_F(PreprocessFilterTest, SphericalFibonacciSamplingCopiesSourceWhenSamplingCountExceedsSize) {
    PointCloudCPU src_cpu;
    src_cpu.points->resize(3);
    src_cpu.normals->resize(3);
    src_cpu.intensities->resize(3);
    for (size_t i = 0; i < 3; ++i) {
        (*src_cpu.points)[i] = PointType(static_cast<float>(i + 1), static_cast<float>(i + 2), 0.0f, 1.0f);
        (*src_cpu.normals)[i] = Normal(1.0f, 0.0f, 0.0f, 0.0f);
        (*src_cpu.intensities)[i] = static_cast<float>(i) * 2.0f;
    }

    PointCloudShared source(*queue_, src_cpu);
    PointCloudCPU dummy_cpu;
    dummy_cpu.points->resize(1);
    (*dummy_cpu.points)[0] = PointType(999.0f, 999.0f, 999.0f, 1.0f);
    PointCloudShared output(*queue_, dummy_cpu);

    algorithms::filter::PreprocessFilter filter(*queue_);
    filter.spherical_fibonacci_sampling(source, output, 10);

    ASSERT_EQ(output.size(), source.size());
    ASSERT_TRUE(output.has_normal());
    ASSERT_TRUE(output.has_intensity());
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_TRUE((*output.points)[i].isApprox((*source.points)[i], 1e-6f));
        EXPECT_TRUE((*output.normals)[i].isApprox((*source.normals)[i], 1e-6f));
        EXPECT_NEAR((*output.intensities)[i], (*source.intensities)[i], 1e-6f);
    }
}

TEST_F(PreprocessFilterTest, SphericalFibonacciSamplingReturnsEmptyOutputForEmptyInputOutOfPlace) {
    PointCloudCPU empty_cpu;
    PointCloudShared source(*queue_, empty_cpu);

    PointCloudCPU dummy_cpu;
    dummy_cpu.points->resize(1);
    (*dummy_cpu.points)[0] = PointType(123.0f, 456.0f, 789.0f, 1.0f);
    PointCloudShared output(*queue_, dummy_cpu);

    algorithms::filter::PreprocessFilter filter(*queue_);
    filter.spherical_fibonacci_sampling(source, output, 4);

    ASSERT_EQ(output.size(), 0U);
}

TEST_F(PreprocessFilterTest, SphericalFibonacciSamplingThrowsWithoutNormalsOrCovs) {
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(5);
    for (size_t i = 0; i < 5; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    EXPECT_THROW(filter.spherical_fibonacci_sampling(shared_cloud, 3), std::runtime_error);
}

TEST_F(PreprocessFilterTest, SphericalFibonacciSamplingRetainsCorrectCount) {
    const size_t N = 100;
    const size_t sampling_num = 20;
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(N);
    cpu_cloud.normals->resize(N);
    for (size_t i = 0; i < N; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        const float angle = static_cast<float>(i) / N * 2.0f * eigen_utils::PI;
        (*cpu_cloud.normals)[i] = Normal(std::cos(angle), std::sin(angle), 0.0f, 0.0f);
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.spherical_fibonacci_sampling(shared_cloud, sampling_num);

    ASSERT_EQ(shared_cloud.size(), sampling_num);
}

TEST_F(PreprocessFilterTest, SphericalFibonacciSamplingSelectsUniquePoints) {
    // Each point has a unique x-coordinate; after sampling, all x-coordinates must be distinct.
    const size_t N = 50;
    const size_t sampling_num = 10;
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(N);
    cpu_cloud.normals->resize(N);
    for (size_t i = 0; i < N; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        // Spread normals uniformly on the equator
        const float angle = static_cast<float>(i) / N * 2.0f * eigen_utils::PI;
        (*cpu_cloud.normals)[i] = Normal(std::cos(angle), std::sin(angle), 0.0f, 0.0f);
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.spherical_fibonacci_sampling(shared_cloud, sampling_num);

    ASSERT_EQ(shared_cloud.size(), sampling_num);

    // Verify all selected points are distinct (unique x-coordinates)
    std::vector<float> xs;
    xs.reserve(sampling_num);
    for (size_t i = 0; i < shared_cloud.size(); ++i) {
        xs.push_back((*shared_cloud.points)[i].x());
    }
    std::sort(xs.begin(), xs.end());
    const bool all_unique = std::adjacent_find(xs.begin(), xs.end()) == xs.end();
    EXPECT_TRUE(all_unique) << "Duplicate points found in Fibonacci sampling output";
}

TEST_F(PreprocessFilterTest, SphericalFibonacciSamplingPreservesAttributes) {
    const size_t N = 50;
    const size_t sampling_num = 10;
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(N);
    cpu_cloud.normals->resize(N);
    cpu_cloud.intensities->resize(N);
    for (size_t i = 0; i < N; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        const float angle = static_cast<float>(i) / N * 2.0f * eigen_utils::PI;
        (*cpu_cloud.normals)[i] = Normal(std::cos(angle), std::sin(angle), 0.0f, 0.0f);
        (*cpu_cloud.intensities)[i] = static_cast<float>(i) * 2.0f;
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    filter.spherical_fibonacci_sampling(shared_cloud, sampling_num);

    ASSERT_EQ(shared_cloud.size(), sampling_num);
    ASSERT_TRUE(shared_cloud.has_normal());
    ASSERT_TRUE(shared_cloud.has_intensity());
    // Each selected point's intensity must match its original value (intensity = x * 2)
    for (size_t i = 0; i < shared_cloud.size(); ++i) {
        const float x = (*shared_cloud.points)[i].x();
        EXPECT_NEAR((*shared_cloud.intensities)[i], x * 2.0f, 1e-5f);
    }
}

TEST_F(PreprocessFilterTest, SphericalFibonacciSamplingWorksWithCovariancesOnly) {
    const size_t N = 50;
    const size_t sampling_num = 10;
    PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(N);
    cpu_cloud.covs->resize(N);
    for (size_t i = 0; i < N; ++i) {
        (*cpu_cloud.points)[i] = PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        // Covariance with small eigenvalue in x → normal along +x
        Covariance cov = Covariance::Zero();
        cov(0, 0) = 1e-4f;
        cov(1, 1) = 1.0f;
        cov(2, 2) = 1.0f;
        (*cpu_cloud.covs)[i] = cov;
    }
    PointCloudShared shared_cloud(*queue_, cpu_cloud);
    algorithms::filter::PreprocessFilter filter(*queue_);

    EXPECT_NO_THROW(filter.spherical_fibonacci_sampling(shared_cloud, sampling_num));
    ASSERT_EQ(shared_cloud.size(), sampling_num);
}

}  // namespace
}  // namespace sycl_points
