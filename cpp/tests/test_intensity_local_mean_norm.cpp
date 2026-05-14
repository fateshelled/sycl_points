#include <gtest/gtest.h>

#include <cmath>

#include "sycl_points/algorithms/filter/intensity_local_mean_norm.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace {

sycl_points::PointCloudShared make_cloud(const sycl_points::sycl_utils::DeviceQueue& queue,
                                         const std::vector<sycl_points::PointType>& pts,
                                         const std::vector<float>& intensities) {
    sycl_points::PointCloudShared cloud(queue);
    cloud.points->resize(pts.size());
    cloud.intensities->resize(intensities.size());
    for (size_t i = 0; i < pts.size(); ++i) cloud.points->at(i) = pts[i];
    for (size_t i = 0; i < intensities.size(); ++i) cloud.intensities->at(i) = intensities[i];
    return cloud;
}

}  // namespace

TEST(IntensityLocalMeanNormTest, FlatIntensityYieldsUnity) {
    // Uniform intensity field: I[i] / local_mean(I) = I / I = 1 for every point.
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    std::vector<sycl_points::PointType> pts = {
        {3.0f, -0.2f, 0.0f, 1.0f}, {3.0f, -0.1f, 0.0f, 1.0f}, {3.0f, 0.0f, 0.0f, 1.0f},
        {3.0f, 0.1f, 0.0f, 1.0f},  {3.0f, 0.2f, 0.0f, 1.0f},
    };
    std::vector<float> intensities(pts.size(), 0.5f);

    auto cloud = make_cloud(queue, pts, intensities);
    auto kdtree = sycl_points::algorithms::knn::KDTree::build(queue, cloud);
    auto neighbors = kdtree->knn_search(cloud, 5);

    sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud, neighbors, 0.3f, 0.3f, 0.3f);

    for (size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR((*cloud.intensities)[i], 1.0f, 1e-4f) << "index " << i;
    }
}

TEST(IntensityLocalMeanNormTest, BrightPointExceedsUnity) {
    // Center is bright, neighbors are dark → center's local-mean is dominated by self bias plus dark
    // neighbors, so normalized center > 1 and normalized dark neighbors < 1.
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    std::vector<sycl_points::PointType> pts = {
        {3.0f, -0.2f, 0.0f, 1.0f}, {3.0f, -0.1f, 0.0f, 1.0f}, {3.0f, 0.0f, 0.0f, 1.0f},
        {3.0f, 0.1f, 0.0f, 1.0f},  {3.0f, 0.2f, 0.0f, 1.0f},
    };
    std::vector<float> intensities = {0.1f, 0.1f, 1.0f, 0.1f, 0.1f};

    auto cloud = make_cloud(queue, pts, intensities);
    auto kdtree = sycl_points::algorithms::knn::KDTree::build(queue, cloud);
    auto neighbors = kdtree->knn_search(cloud, 5);

    sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud, neighbors, 0.3f, 0.3f, 0.3f);

    EXPECT_GT((*cloud.intensities)[2], 1.0f);
    EXPECT_LT((*cloud.intensities)[1], 1.0f);
    EXPECT_LT((*cloud.intensities)[3], 1.0f);
}

TEST(IntensityLocalMeanNormTest, MeanMinClampPreventsExplosion) {
    // All-zero intensities: 0 / max(0, mean_min) = 0; no NaN / Inf.
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    std::vector<sycl_points::PointType> pts = {
        {3.0f, -0.1f, 0.0f, 1.0f}, {3.0f, 0.0f, 0.0f, 1.0f}, {3.0f, 0.1f, 0.0f, 1.0f},
    };
    std::vector<float> intensities(pts.size(), 0.0f);

    auto cloud = make_cloud(queue, pts, intensities);
    auto kdtree = sycl_points::algorithms::knn::KDTree::build(queue, cloud);
    auto neighbors = kdtree->knn_search(cloud, 3);

    sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud, neighbors, 0.3f, 0.3f, 0.3f, 1e-3f);

    for (size_t i = 0; i < pts.size(); ++i) {
        EXPECT_TRUE(std::isfinite((*cloud.intensities)[i])) << "non-finite at index " << i;
        EXPECT_FLOAT_EQ((*cloud.intensities)[i], 0.0f);
    }
}

TEST(IntensityLocalMeanNormTest, ContinuousAcrossMeanMinBoundary) {
    // Verify that fmax-based clamping leaves no discontinuity at local_mean == mean_min.
    // Compare two clouds with intensities just below and just above mean_min — outputs must agree
    // to high precision (they are evaluated at the same divisor mean_min).
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    std::vector<sycl_points::PointType> pts = {
        {3.0f, -0.1f, 0.0f, 1.0f}, {3.0f, 0.0f, 0.0f, 1.0f}, {3.0f, 0.1f, 0.0f, 1.0f},
    };

    const float mean_min = 1e-3f;
    const float delta = 1e-6f;  // <<< mean_min, so local_mean stays just below mean_min in case A

    std::vector<float> low(pts.size(), mean_min - delta);
    auto cloud_low = make_cloud(queue, pts, low);
    auto kd_low = sycl_points::algorithms::knn::KDTree::build(queue, cloud_low);
    auto nb_low = kd_low->knn_search(cloud_low, 3);
    sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud_low, nb_low, 0.3f, 0.3f, 0.3f, mean_min);

    std::vector<float> high(pts.size(), mean_min + delta);
    auto cloud_high = make_cloud(queue, pts, high);
    auto kd_high = sycl_points::algorithms::knn::KDTree::build(queue, cloud_high);
    auto nb_high = kd_high->knn_search(cloud_high, 3);
    sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud_high, nb_high, 0.3f, 0.3f, 0.3f, mean_min);

    // Expected outputs: low ≈ (mean_min - delta) / mean_min   (clamp active)
    //                   high ≈ (mean_min + delta) / (mean_min + delta) = 1
    // Both must be near 1 with the gap proportional to delta/mean_min — no large jump.
    for (size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR((*cloud_low.intensities)[i], (*cloud_high.intensities)[i], 1e-2f) << "index " << i;
    }
}

TEST(IntensityLocalMeanNormTest, ThrowsWithoutIntensityField) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu;
    cpu.points->resize(2);
    (*cpu.points)[0] = {1.0f, 0.0f, 0.0f, 1.0f};
    (*cpu.points)[1] = {1.1f, 0.0f, 0.0f, 1.0f};
    cpu.intensities->clear();
    sycl_points::PointCloudShared cloud(queue, cpu);

    auto kdtree = sycl_points::algorithms::knn::KDTree::build(queue, cloud);
    auto neighbors = kdtree->knn_search(cloud, 2);

    EXPECT_THROW(sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud, neighbors, 0.3f, 0.3f, 0.3f),
                 std::runtime_error);
}

TEST(IntensityLocalMeanNormTest, ThrowsNonPositiveSigma) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    std::vector<sycl_points::PointType> pts = {{1.0f, 0.0f, 0.0f, 1.0f}, {1.1f, 0.0f, 0.0f, 1.0f}};
    auto cloud = make_cloud(queue, pts, {1.0f, 0.5f});
    auto kdtree = sycl_points::algorithms::knn::KDTree::build(queue, cloud);
    auto neighbors = kdtree->knn_search(cloud, 2);

    EXPECT_THROW(sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud, neighbors, 0.0f, 0.1f, 0.1f),
                 std::runtime_error);
    EXPECT_THROW(sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud, neighbors, 0.1f, -1.0f, 0.1f),
                 std::runtime_error);
    EXPECT_THROW(sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud, neighbors, 0.1f, 0.1f, 0.0f),
                 std::runtime_error);
}

TEST(IntensityLocalMeanNormTest, ThrowsNegativeMeanMin) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    std::vector<sycl_points::PointType> pts = {{1.0f, 0.0f, 0.0f, 1.0f}, {1.1f, 0.0f, 0.0f, 1.0f}};
    auto cloud = make_cloud(queue, pts, {1.0f, 0.5f});
    auto kdtree = sycl_points::algorithms::knn::KDTree::build(queue, cloud);
    auto neighbors = kdtree->knn_search(cloud, 2);

    EXPECT_THROW(
        sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud, neighbors, 0.1f, 0.1f, 0.1f, -1.0f),
        std::runtime_error);
}

TEST(IntensityLocalMeanNormTest, EmptyCloudNoThrow) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu;
    sycl_points::PointCloudShared cloud(queue, cpu);
    sycl_points::algorithms::knn::KNNResult empty_neighbors;
    empty_neighbors.allocate(queue, 0, 1);

    EXPECT_NO_THROW(
        sycl_points::algorithms::intensity_local_mean_norm::normalize(cloud, empty_neighbors, 0.1f, 0.1f, 0.1f));
}
