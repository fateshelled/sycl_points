#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include <sycl_points/algorithms/experimental/voxel_hash_map.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/points/types.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace {

sycl_points::PointCloudShared MakePointCloud(
    const sycl_points::sycl_utils::DeviceQueue& queue,
    const std::vector<Eigen::Vector3f>& positions) {
    // Prepare a CPU point cloud to populate deterministic test data.
    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        const Eigen::Vector3f& pos = positions[i];
        (*cpu_cloud.points)[i] = sycl_points::PointType(pos.x(), pos.y(), pos.z(), 1.0f);
    }
    // Move the CPU data to shared memory for the voxel hash map.
    return sycl_points::PointCloudShared(queue, cpu_cloud);
}

std::vector<Eigen::Vector3f> ExtractPositions(const sycl_points::PointContainerShared& points) {
    // Convert the shared vector into a regular container that is easy to inspect in the test.
    std::vector<Eigen::Vector3f> positions;
    positions.reserve(points.size());
    for (const auto& point : points) {
        positions.emplace_back(point.x(), point.y(), point.z());
    }
    return positions;
}

}  // namespace

TEST(VoxelHashMapTest, ConstructorRejectsNonPositiveVoxelSize) {
    sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    EXPECT_THROW(sycl_points::algorithms::filter::VoxelHashMap(queue, 0.0f), std::invalid_argument);
    EXPECT_THROW(sycl_points::algorithms::filter::VoxelHashMap(queue, -0.1f), std::invalid_argument);
}

TEST(VoxelHashMapTest, AggregatesPointsWithinSameVoxel) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        const float voxel_size = 0.1f;
        sycl_points::algorithms::filter::VoxelHashMap voxel_map(queue, voxel_size);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.02f, 0.02f, 0.00f},
            {0.03f, 0.04f, 0.00f},
            {0.11f, 0.02f, 0.00f},
            {0.12f, 0.03f, 0.00f},
        };

        auto cloud = MakePointCloud(queue, input_positions);
        voxel_map.add_point_cloud(cloud);

        sycl_points::PointContainerShared result(
            sycl_points::shared_allocator<sycl_points::PointType, sycl_points::PointAlignment>(*queue.ptr));
        voxel_map.downsampling(result);

        ASSERT_EQ(result.size(), 2U);

        auto averaged_positions = ExtractPositions(result);
        std::sort(averaged_positions.begin(), averaged_positions.end(), [](const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
            if (lhs.x() != rhs.x()) {
                return lhs.x() < rhs.x();
            }
            if (lhs.y() != rhs.y()) {
                return lhs.y() < rhs.y();
            }
            return lhs.z() < rhs.z();
        });

        const std::vector<Eigen::Vector3f> expected_positions = {
            {0.025f, 0.03f, 0.0f},
            {0.115f, 0.025f, 0.0f},
        };

        ASSERT_EQ(averaged_positions.size(), expected_positions.size());
        for (size_t i = 0; i < expected_positions.size(); ++i) {
            EXPECT_NEAR(averaged_positions[i].x(), expected_positions[i].x(), 1e-5f);
            EXPECT_NEAR(averaged_positions[i].y(), expected_positions[i].y(), 1e-5f);
            EXPECT_NEAR(averaged_positions[i].z(), expected_positions[i].z(), 1e-5f);
        }
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(VoxelHashMapTest, RemovesStaleVoxelsAfterConfiguredCycles) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::filter::VoxelHashMap voxel_map(queue, 0.1f);
        voxel_map.set_max_staleness(1);
        voxel_map.set_remove_old_data_cycle(1);

        auto old_cloud = MakePointCloud(queue, {{0.0f, 0.0f, 0.0f}});
        voxel_map.add_point_cloud(old_cloud);

        sycl_points::PointContainerShared result(
            sycl_points::shared_allocator<sycl_points::PointType, sycl_points::PointAlignment>(*queue.ptr));
        voxel_map.downsampling(result);
        ASSERT_EQ(result.size(), 1U);

        auto recent_cloud = MakePointCloud(queue, {{1.0f, 0.0f, 0.0f}});
        voxel_map.add_point_cloud(recent_cloud);
        voxel_map.downsampling(result);
        ASSERT_EQ(result.size(), 2U);

        sycl_points::PointCloudShared empty_cloud(queue);
        voxel_map.add_point_cloud(empty_cloud);
        voxel_map.downsampling(result);
        ASSERT_EQ(result.size(), 1U);

        auto remaining_positions = ExtractPositions(result);
        ASSERT_EQ(remaining_positions.size(), 1U);
        EXPECT_NEAR(remaining_positions[0].x(), 1.0f, 1e-5f);
        EXPECT_NEAR(remaining_positions[0].y(), 0.0f, 1e-5f);
        EXPECT_NEAR(remaining_positions[0].z(), 0.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}
