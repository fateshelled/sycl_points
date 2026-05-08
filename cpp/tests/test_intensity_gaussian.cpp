#include <gtest/gtest.h>

#include "sycl_points/algorithms/filter/intensity_gaussian.hpp"
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

TEST(IntensityGaussianTest, IsotropicSmoothingReducesVariance) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    // Points on a horizontal arc at range=3m; center point has intensity 1, rest 0
    std::vector<sycl_points::PointType> pts = {
        {3.0f, -0.2f, 0.0f, 1.0f}, {3.0f, -0.1f, 0.0f, 1.0f}, {3.0f, 0.0f, 0.0f, 1.0f},  // center
        {3.0f, 0.1f, 0.0f, 1.0f},  {3.0f, 0.2f, 0.0f, 1.0f},
    };
    std::vector<float> intensities = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f};

    auto cloud = make_cloud(queue, pts, intensities);
    auto kdtree = sycl_points::algorithms::knn::KDTree::build(queue, cloud);
    auto neighbors = kdtree->knn_search(cloud, 5);

    const float before_center = (*cloud.intensities)[2];
    sycl_points::algorithms::intensity_gaussian::smooth_intensity(cloud, neighbors, 0.3f, 0.3f, 0.3f);

    // Spike at center should decrease; adjacent points should gain intensity
    EXPECT_LT((*cloud.intensities)[2], before_center);
    EXPECT_GT((*cloud.intensities)[1], 0.0f);
    EXPECT_GT((*cloud.intensities)[3], 0.0f);
}

TEST(IntensityGaussianTest, AzimuthSmoothsMoreThanElevation) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    // Point at (5, 0, 0): r_hat=(1,0,0), az_hat=(0,1,0), el_hat=(0,0,1)
    // Config A: neighbor offset d in azimuth direction (0, d, 0)
    // Config B: neighbor offset d in elevation direction (0, 0, d)
    // Wide sigma_azimuth → az neighbor pulls center down more than el neighbor
    const float d = 0.3f;

    {
        std::vector<sycl_points::PointType> pts = {{5.0f, 0.0f, 0.0f, 1.0f}, {5.0f, d, 0.0f, 1.0f}};
        auto cloud = make_cloud(queue, pts, {1.0f, 0.0f});
        auto kd = sycl_points::algorithms::knn::KDTree::build(queue, cloud);
        auto nb = kd->knn_search(cloud, 2);
        sycl_points::algorithms::intensity_gaussian::smooth_intensity(cloud, nb, 0.5f, 0.1f, 10.0f);

        std::vector<sycl_points::PointType> pts2 = {{5.0f, 0.0f, 0.0f, 1.0f}, {5.0f, 0.0f, d, 1.0f}};
        auto cloud2 = make_cloud(queue, pts2, {1.0f, 0.0f});
        auto kd2 = sycl_points::algorithms::knn::KDTree::build(queue, cloud2);
        auto nb2 = kd2->knn_search(cloud2, 2);
        sycl_points::algorithms::intensity_gaussian::smooth_intensity(cloud2, nb2, 0.5f, 0.1f, 10.0f);

        // Wide azimuth sigma → center blended more toward az neighbor → lower intensity
        EXPECT_LT((*cloud.intensities)[0], (*cloud2.intensities)[0]);
    }
}

TEST(IntensityGaussianTest, NarrowRangeSigmaPreservesDepthEdge) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    // Foreground at 2m and background at 5m on the same ray; 3m depth gap >> sigma_range=0.05m
    std::vector<sycl_points::PointType> pts = {
        {2.0f, 0.0f, 0.0f, 1.0f},
        {5.0f, 0.0f, 0.0f, 1.0f},
    };
    auto cloud = make_cloud(queue, pts, {1.0f, 0.0f});
    auto kdtree = sycl_points::algorithms::knn::KDTree::build(queue, cloud);
    auto neighbors = kdtree->knn_search(cloud, 2);
    sycl_points::algorithms::intensity_gaussian::smooth_intensity(cloud, neighbors, 1.0f, 1.0f, 0.05f);

    // 3m gap >> sigma_range(0.05m): both points should be nearly unchanged
    EXPECT_NEAR((*cloud.intensities)[0], 1.0f, 0.01f);
    EXPECT_NEAR((*cloud.intensities)[1], 0.0f, 0.01f);
}

TEST(IntensityGaussianTest, ThrowsWithoutIntensityField) {
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

    EXPECT_THROW(sycl_points::algorithms::intensity_gaussian::smooth_intensity(cloud, neighbors, 0.1f, 0.1f),
                 std::runtime_error);
}

TEST(IntensityGaussianTest, ThrowsNonPositiveSigma) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    std::vector<sycl_points::PointType> pts = {{1.0f, 0.0f, 0.0f, 1.0f}, {1.1f, 0.0f, 0.0f, 1.0f}};
    auto cloud = make_cloud(queue, pts, {1.0f, 0.0f});
    auto kdtree = sycl_points::algorithms::knn::KDTree::build(queue, cloud);
    auto neighbors = kdtree->knn_search(cloud, 2);

    EXPECT_THROW(sycl_points::algorithms::intensity_gaussian::smooth_intensity(cloud, neighbors, 0.0f, 0.1f, 0.1f),
                 std::runtime_error);
    EXPECT_THROW(sycl_points::algorithms::intensity_gaussian::smooth_intensity(cloud, neighbors, 0.1f, -1.0f, 0.1f),
                 std::runtime_error);
}

TEST(IntensityGaussianTest, EmptyCloudNoThrow) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu;
    sycl_points::PointCloudShared cloud(queue, cpu);
    sycl_points::algorithms::knn::KNNResult empty_neighbors;
    empty_neighbors.allocate(queue, 0, 1);

    EXPECT_NO_THROW(sycl_points::algorithms::intensity_gaussian::smooth_intensity(cloud, empty_neighbors, 0.1f, 0.1f));
}
