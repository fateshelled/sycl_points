#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/localization/voxel_occupancy_map.hpp"

namespace {
using namespace sycl_points;
using namespace sycl_points::algorithms::localization;

PointCloudShared make_cloud(const sycl_utils::DeviceQueue& queue, const std::vector<PointType>& points) {
    PointCloudShared result(queue);
    result.points->assign(points.begin(), points.end());
    return result;
}

class VoxelOccupancyMapTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue{sycl::device(sycl_utils::device_selector::default_selector_v)};
};

TEST_F(VoxelOccupancyMapTest, CountsOccupiedVoxels) {
    auto target = make_cloud(this->queue, {PointType(0.1f, 0, 0, 1), PointType(5.5f, 0, 0, 1)});
    auto scan =
        make_cloud(this->queue, {PointType(0.2f, 0, 0, 1), PointType(5.4f, 0, 0, 1), PointType(10.0f, 0, 0, 1)});
    VoxelOccupancyMap map(this->queue, {1.0f});
    map.set_target(target);

    shared_vector<TransformMatrix> transforms(1, *this->queue.ptr);
    transforms[0] = TransformMatrix::Identity();
    shared_vector<uint32_t> counts(1, 0, *this->queue.ptr);
    map.score_async(scan, transforms, 0, counts).wait_and_throw();
    EXPECT_EQ(counts[0], 2U);
}

TEST_F(VoxelOccupancyMapTest, AppliesTransform) {
    auto target = make_cloud(this->queue, {PointType(0.1f, 0, 0, 1), PointType(5.5f, 0, 0, 1)});
    auto scan =
        make_cloud(this->queue, {PointType(0.2f, 0, 0, 1), PointType(5.4f, 0, 0, 1), PointType(10.0f, 0, 0, 1)});
    VoxelOccupancyMap map(this->queue, {1.0f});
    map.set_target(target);

    shared_vector<TransformMatrix> transforms(1, *this->queue.ptr);
    transforms[0] = TransformMatrix::Identity();
    transforms[0](0, 3) = 5.0f;
    shared_vector<uint32_t> counts(1, 0, *this->queue.ptr);
    map.score_async(scan, transforms, 0, counts).wait_and_throw();
    EXPECT_EQ(counts[0], 1U);
}

TEST_F(VoxelOccupancyMapTest, InflatesOccupiedVoxelsTowardNegativeNeighbors) {
    // The target is in voxel 1. The negative inflation marks voxel 0, so a query in voxel 0 hits.
    auto target = make_cloud(this->queue, {PointType(2.2f, 0, 0, 1)});
    auto scan = make_cloud(this->queue, {PointType(0.0f, 0, 0, 1)});
    VoxelOccupancyMap map(this->queue, {2.0f});
    map.set_target(target);

    shared_vector<TransformMatrix> transforms(1, *this->queue.ptr);
    transforms[0] = TransformMatrix::Identity();
    shared_vector<uint32_t> counts(1, 0, *this->queue.ptr);
    map.score_async(scan, transforms, 0, counts).wait_and_throw();
    EXPECT_EQ(counts[0], 1U);
}

TEST_F(VoxelOccupancyMapTest, ReportsSizeAndVoxelSize) {
    VoxelOccupancyMap map(this->queue, {1.0f, 2.0f});
    EXPECT_EQ(map.level_count(), 2U);
    EXPECT_FLOAT_EQ(map.voxel_size(0), 1.0f);
    EXPECT_FLOAT_EQ(map.voxel_size(1), 2.0f);
    EXPECT_EQ(map.target_size(), 0U);
}

TEST_F(VoxelOccupancyMapTest, RejectsInvalidInput) {
    PointCloudShared empty(this->queue);
    VoxelOccupancyMap map(this->queue, {1.0f});
    EXPECT_THROW(map.set_target(empty), std::invalid_argument);
    EXPECT_THROW(VoxelOccupancyMap(this->queue, {}), std::invalid_argument);
    EXPECT_THROW(VoxelOccupancyMap(this->queue, {0.0f}), std::invalid_argument);
}

// Compares the per-candidate cost of the exact radius count and the voxel occupancy score.
// The map has sparse surface structure, and most candidate poses are wrong.
TEST_F(VoxelOccupancyMapTest, BenchmarkScoreMethods) {
    constexpr size_t target_count = 34000;
    constexpr size_t scan_count = 4000;
    constexpr size_t candidate_count = 200;

    std::mt19937 gen(7);
    std::uniform_real_distribution<float> long_axis(-50.0f, 50.0f);
    std::uniform_real_distribution<float> short_axis(-20.0f, 20.0f);
    std::uniform_real_distribution<float> height(0.0f, 10.0f);
    std::uniform_real_distribution<float> shift(-30.0f, 30.0f);
    std::uniform_real_distribution<float> yaw(-3.14159265f, 3.14159265f);

    std::vector<PointType> target_points;
    target_points.reserve(target_count);
    for (size_t i = 0; i < target_count; ++i) {
        if (i % 3 == 0) {
            target_points.emplace_back(long_axis(gen), short_axis(gen), 0.0f, 1.0f);
        } else {
            const float x = i % 3 == 1 ? 20.0f : -20.0f;
            target_points.emplace_back(x, short_axis(gen), height(gen), 1.0f);
        }
    }
    std::vector<PointType> scan_points;
    scan_points.reserve(scan_count);
    const size_t stride = std::max<size_t>(1, target_points.size() / scan_count);
    for (size_t i = 0; i < target_points.size() && scan_points.size() < scan_count; i += stride) {
        scan_points.push_back(target_points[i]);
    }

    auto target = make_cloud(this->queue, target_points);
    auto scan = make_cloud(this->queue, scan_points);
    const auto tree = algorithms::knn::KDTree::build(this->queue, target);
    VoxelOccupancyMap map(this->queue, {1.0f, 2.0f, 4.0f, 8.0f, 16.0f});
    map.set_target(target);

    shared_vector<TransformMatrix> transforms(candidate_count, *this->queue.ptr);
    for (size_t i = 0; i < candidate_count; ++i) {
        TransformMatrix transform = TransformMatrix::Identity();
        transform(0, 3) = shift(gen);
        transform(1, 3) = shift(gen);
        transform(2, 3) = shift(gen);
        transform(0, 0) = std::cos(yaw(gen));
        transform(0, 1) = -std::sin(yaw(gen));
        transform(1, 0) = std::sin(yaw(gen));
        transform(1, 1) = std::cos(yaw(gen));
        transforms[i] = transform;
    }

    shared_vector<uint32_t> kdtree_counts(candidate_count, 0, *this->queue.ptr);
    shared_vector<uint32_t> voxel_counts(candidate_count, 0, *this->queue.ptr);
    std::cout << "score benchmark: map=" << target.size() << " scan=" << scan.size()
              << " candidates=" << candidate_count << "\n";
    for (size_t level = 0; level < map.level_count(); ++level) {
        const float radius = map.voxel_size(level);
        const auto kdtree_begin = std::chrono::steady_clock::now();
        tree->count_neighbors_within_radius_async(scan, transforms, radius, kdtree_counts).wait_and_throw();
        const auto kdtree_end = std::chrono::steady_clock::now();
        const auto voxel_begin = std::chrono::steady_clock::now();
        map.score_async(scan, transforms, level, voxel_counts).wait_and_throw();
        const auto voxel_end = std::chrono::steady_clock::now();
        const double kdtree_ms = std::chrono::duration<double, std::milli>(kdtree_end - kdtree_begin).count();
        const double voxel_ms = std::chrono::duration<double, std::milli>(voxel_end - voxel_begin).count();
        std::cout << "level " << level << " radius=" << radius << " kdtree=" << kdtree_ms / candidate_count
                  << " ms/candidate voxel=" << voxel_ms / candidate_count << " ms/candidate\n";
    }
}
}  // namespace
