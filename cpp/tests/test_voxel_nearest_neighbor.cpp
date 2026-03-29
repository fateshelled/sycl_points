#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "sycl_points/algorithms/knn/result.hpp"
#include "sycl_points/algorithms/mapping/occupancy_grid_map.hpp"
#include "sycl_points/algorithms/mapping/voxel_hash_map.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/points/types.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace {

/// @brief Build a PointCloudShared from a list of 3-D positions.
sycl_points::PointCloudShared MakeCloud(const sycl_points::sycl_utils::DeviceQueue& queue,
                                        const std::vector<Eigen::Vector3f>& positions) {
    sycl_points::PointCloudCPU cpu;
    cpu.points->resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        (*cpu.points)[i] = sycl_points::PointType(positions[i].x(), positions[i].y(), positions[i].z(), 1.0f);
    }
    return sycl_points::PointCloudShared(queue, cpu);
}

/// @brief Identity pose used for add_point_cloud calls.
Eigen::Isometry3f IdentityPose() { return Eigen::Isometry3f::Identity(); }

}  // namespace

// =============================================================================
// VoxelHashMap tests
// =============================================================================

class VoxelHashMapNNTest : public ::testing::Test {
protected:
    sycl_points::sycl_utils::DeviceQueue queue;
    const float voxel_size = 1.0f;

    VoxelHashMapNNTest()
        : queue(sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v)) {}
};

/// Basic case: insert one point at the origin, query near it.
TEST_F(VoxelHashMapNNTest, BasicNearestNeighbor) {
    sycl_points::algorithms::mapping::VoxelHashMap map(queue, voxel_size);

    // Insert a single point at the origin.
    auto cloud = MakeCloud(queue, {{0.1f, 0.1f, 0.1f}});
    map.add_point_cloud(cloud, IdentityPose());

    // Query at the origin itself.
    auto query = MakeCloud(queue, {{0.0f, 0.0f, 0.0f}});
    sycl_points::algorithms::knn::KNNResult result;
    map.nearest_neighbor_search(query, result);

    ASSERT_EQ(result.query_size, 1u);
    ASSERT_EQ(result.k, 1u);

    const int32_t idx = result.indices->at(0);
    const float dist  = result.distances->at(0);

    EXPECT_GE(idx, 0) << "Expected a valid slot index for the inserted voxel";
    EXPECT_LT(dist, voxel_size * voxel_size) << "Distance should be within one voxel";
}

/// Empty map: all queries should return index=-1 and distance=FLT_MAX.
TEST_F(VoxelHashMapNNTest, EmptyMapReturnsNoResult) {
    sycl_points::algorithms::mapping::VoxelHashMap map(queue, voxel_size);

    auto query = MakeCloud(queue, {{0.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f}});
    sycl_points::algorithms::knn::KNNResult result;
    map.nearest_neighbor_search(query, result);

    ASSERT_EQ(result.query_size, 2u);
    for (size_t i = 0; i < result.query_size; ++i) {
        EXPECT_EQ(result.indices->at(i), -1) << "Empty map must return -1 for query " << i;
        EXPECT_EQ(result.distances->at(i), std::numeric_limits<float>::max());
    }
}

/// 7-neighbor vs 27-neighbor: a point at a corner voxel offset (1,1,1)
/// should be missed by the 7-neighbor search from the origin but found with 27.
TEST_F(VoxelHashMapNNTest, NeighborPattern7vs27) {
    const float vs = 1.0f;
    sycl_points::algorithms::mapping::VoxelHashMap map(queue, vs);

    // Place a point inside the voxel at (1.5, 1.5, 1.5) — corner neighbor of origin voxel.
    auto cloud = MakeCloud(queue, {{1.5f, 1.5f, 1.5f}});
    map.add_point_cloud(cloud, IdentityPose());

    // Query from origin (voxel 0,0,0). The target is in voxel (1,1,1).
    auto query = MakeCloud(queue, {{0.1f, 0.1f, 0.1f}});

    sycl_points::algorithms::knn::KNNResult result7, result27;
    map.nearest_neighbor_search<7>(query, result7);
    map.nearest_neighbor_search<27>(query, result27);

    // 7-neighbor only covers face neighbors: (1,1,1) is a corner, so it must NOT be found.
    EXPECT_EQ(result7.indices->at(0), -1) << "7-neighbor search should not find a corner voxel";

    // 27-neighbor covers all 3x3x3: must find the corner voxel.
    EXPECT_GE(result27.indices->at(0), 0) << "27-neighbor search must find the corner voxel";
}

/// 19-neighbor: edge neighbors should be found, corners should not.
TEST_F(VoxelHashMapNNTest, NeighborPattern19) {
    const float vs = 1.0f;
    sycl_points::algorithms::mapping::VoxelHashMap map(queue, vs);

    // Place points in an edge-neighbor voxel (1,1,0) and a corner-neighbor voxel (1,1,1).
    auto cloud = MakeCloud(queue, {{1.5f, 1.5f, 0.5f}, {1.5f, 1.5f, 1.5f}});
    map.add_point_cloud(cloud, IdentityPose());

    auto query = MakeCloud(queue, {{0.1f, 0.1f, 0.1f}});

    sycl_points::algorithms::knn::KNNResult result19;
    map.nearest_neighbor_search<19>(query, result19);

    // 19-neighbor covers edges (sum_abs <= 2): voxel (1,1,0) has sum_abs=2, should be found.
    // If the corner (1,1,1) is also present but has sum_abs=3 it should be excluded.
    // The edge voxel centroid (1.5, 1.5, 0.5) is closer to the query than the corner (1.5, 1.5, 1.5).
    EXPECT_GE(result19.indices->at(0), 0) << "19-neighbor search must find the edge voxel";
    // Verify it found the edge voxel, not the corner (smaller distance expected).
    const float dist_to_edge   = (Eigen::Vector3f(0.1f, 0.1f, 0.1f) - Eigen::Vector3f(1.5f, 1.5f, 0.5f)).squaredNorm();
    const float dist_to_corner = (Eigen::Vector3f(0.1f, 0.1f, 0.1f) - Eigen::Vector3f(1.5f, 1.5f, 1.5f)).squaredNorm();
    EXPECT_LT(dist_to_edge, dist_to_corner);
    EXPECT_NEAR(result19.distances->at(0), dist_to_edge, 1e-4f);
}

/// Multiple queries: each query should match the closest voxel independently.
TEST_F(VoxelHashMapNNTest, MultipleQueries) {
    sycl_points::algorithms::mapping::VoxelHashMap map(queue, voxel_size);

    // Two clearly separated voxels.
    auto cloud = MakeCloud(queue, {{0.5f, 0.5f, 0.5f}, {10.5f, 10.5f, 10.5f}});
    map.add_point_cloud(cloud, IdentityPose());

    // Two queries — one near each voxel.
    auto query = MakeCloud(queue, {{0.6f, 0.6f, 0.6f}, {10.4f, 10.4f, 10.4f}});
    sycl_points::algorithms::knn::KNNResult result;
    map.nearest_neighbor_search(query, result);

    ASSERT_EQ(result.query_size, 2u);
    EXPECT_GE(result.indices->at(0), 0);
    EXPECT_GE(result.indices->at(1), 0);

    // The two queries should match different slots.
    EXPECT_NE(result.indices->at(0), result.indices->at(1));

    // First query should be closer to the first voxel centroid.
    EXPECT_LT(result.distances->at(0), result.distances->at(1));
}

/// Async variant: event-based wait should yield the same result as the sync variant.
TEST_F(VoxelHashMapNNTest, AsyncVariant) {
    sycl_points::algorithms::mapping::VoxelHashMap map(queue, voxel_size);

    auto cloud = MakeCloud(queue, {{0.5f, 0.5f, 0.5f}});
    map.add_point_cloud(cloud, IdentityPose());

    auto query = MakeCloud(queue, {{0.6f, 0.6f, 0.6f}});

    sycl_points::algorithms::knn::KNNResult result_async, result_sync;
    auto events = map.nearest_neighbor_search_async(query, result_async);
    events.wait_and_throw();

    map.nearest_neighbor_search(query, result_sync);

    EXPECT_EQ(result_async.indices->at(0), result_sync.indices->at(0));
    EXPECT_NEAR(result_async.distances->at(0), result_sync.distances->at(0), 1e-6f);
}

// =============================================================================
// OccupancyGridMap tests
// =============================================================================

class OccupancyGridMapNNTest : public ::testing::Test {
protected:
    sycl_points::sycl_utils::DeviceQueue queue;
    const float voxel_size = 1.0f;

    OccupancyGridMapNNTest()
        : queue(sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v)) {}
};

/// Basic case: insert a point with enough log-odds, query near it.
TEST_F(OccupancyGridMapNNTest, BasicNearestNeighbor) {
    sycl_points::algorithms::mapping::OccupancyGridMap map(queue, voxel_size);
    map.set_free_space_updates_enabled(false);
    map.set_occupancy_threshold(0.5f);

    // Insert the same point multiple times to accumulate log-odds above threshold.
    auto cloud = MakeCloud(queue, {{0.5f, 0.5f, 0.5f}});
    for (int i = 0; i < 5; ++i) {
        map.add_point_cloud(cloud, IdentityPose());
    }

    auto query = MakeCloud(queue, {{0.0f, 0.0f, 0.0f}});
    sycl_points::algorithms::knn::KNNResult result;
    map.nearest_neighbor_search(query, result);

    ASSERT_EQ(result.query_size, 1u);
    const int32_t idx = result.indices->at(0);
    const float dist  = result.distances->at(0);

    EXPECT_GE(idx, 0) << "Expected occupied voxel to be found";
    EXPECT_LT(dist, voxel_size * voxel_size);
}

/// Empty map: all results should be -1.
TEST_F(OccupancyGridMapNNTest, EmptyMapReturnsNoResult) {
    sycl_points::algorithms::mapping::OccupancyGridMap map(queue, voxel_size);

    auto query = MakeCloud(queue, {{1.0f, 1.0f, 1.0f}});
    sycl_points::algorithms::knn::KNNResult result;
    map.nearest_neighbor_search(query, result);

    EXPECT_EQ(result.indices->at(0), -1);
    EXPECT_EQ(result.distances->at(0), std::numeric_limits<float>::max());
}

/// Voxel below occupancy threshold must not be returned.
TEST_F(OccupancyGridMapNNTest, BelowThresholdNotReturned) {
    sycl_points::algorithms::mapping::OccupancyGridMap map(queue, voxel_size);
    map.set_free_space_updates_enabled(false);
    // Set a very high threshold so a single insertion cannot satisfy it.
    map.set_occupancy_threshold(0.99f);

    auto cloud = MakeCloud(queue, {{0.5f, 0.5f, 0.5f}});
    map.add_point_cloud(cloud, IdentityPose());  // only one hit → log-odds too low

    auto query = MakeCloud(queue, {{0.5f, 0.5f, 0.5f}});
    sycl_points::algorithms::knn::KNNResult result;
    map.nearest_neighbor_search(query, result);

    EXPECT_EQ(result.indices->at(0), -1) << "Voxel below occupancy threshold must not be returned";
}

/// Neighbor pattern 7 vs 27 for OccupancyGridMap.
TEST_F(OccupancyGridMapNNTest, NeighborPattern7vs27) {
    sycl_points::algorithms::mapping::OccupancyGridMap map(queue, voxel_size);
    map.set_free_space_updates_enabled(false);
    map.set_occupancy_threshold(0.5f);

    // Place a point at a corner neighbor voxel (1,1,1) relative to query voxel (0,0,0).
    auto cloud = MakeCloud(queue, {{1.5f, 1.5f, 1.5f}});
    for (int i = 0; i < 5; ++i) {
        map.add_point_cloud(cloud, IdentityPose());
    }

    auto query = MakeCloud(queue, {{0.1f, 0.1f, 0.1f}});

    sycl_points::algorithms::knn::KNNResult result7, result27;
    map.nearest_neighbor_search<7>(query, result7);
    map.nearest_neighbor_search<27>(query, result27);

    EXPECT_EQ(result7.indices->at(0), -1) << "7-neighbor must not find a corner voxel";
    EXPECT_GE(result27.indices->at(0), 0) << "27-neighbor must find the corner voxel";
}
