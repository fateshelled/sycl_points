#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "sycl_points/algorithms/common/voxel_constants.hpp"
#include "sycl_points/algorithms/mapping/voxel_hash_map.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/points/types.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace {

sycl_points::PointCloudShared MakePointCloud(const sycl_points::sycl_utils::DeviceQueue& queue,
                                             const std::vector<Eigen::Vector3f>& positions,
                                             const std::vector<sycl_points::RGBType>* colors = nullptr,
                                             const std::vector<float>* intensities = nullptr) {
    // Prepare a CPU point cloud to populate deterministic test data.
    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        const Eigen::Vector3f& pos = positions[i];
        (*cpu_cloud.points)[i] = sycl_points::PointType(pos.x(), pos.y(), pos.z(), 1.0f);
    }

    if (colors) {
        EXPECT_EQ(colors->size(), positions.size());
        cpu_cloud.rgb->resize(positions.size());
        std::copy(colors->begin(), colors->end(), cpu_cloud.rgb->begin());
    }

    if (intensities) {
        EXPECT_EQ(intensities->size(), positions.size());
        cpu_cloud.intensities->resize(positions.size());
        std::copy(intensities->begin(), intensities->end(), cpu_cloud.intensities->begin());
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

    EXPECT_THROW(sycl_points::algorithms::mapping::VoxelHashMap(queue, 0.0f), std::invalid_argument);
    EXPECT_THROW(sycl_points::algorithms::mapping::VoxelHashMap(queue, -0.1f), std::invalid_argument);
}

TEST(VoxelHashMapTest, AggregatesPointsWithinSameVoxel) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        const float voxel_size = 0.1f;
        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, voxel_size);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.02f, 0.02f, 0.00f},
            {0.03f, 0.04f, 0.00f},
            {0.11f, 0.02f, 0.00f},
            {0.12f, 0.03f, 0.00f},
        };

        auto cloud = MakePointCloud(queue, input_positions);
        voxel_map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());

        ASSERT_EQ(result.size(), 2U);

        auto averaged_positions = ExtractPositions(*result.points);
        std::sort(averaged_positions.begin(), averaged_positions.end(),
                  [](const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
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

TEST(VoxelHashMapTest, AggregatesRgbAndIntensityWithinVoxel) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        const float voxel_size = 0.5f;
        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, voxel_size);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.0f, 0.0f, 0.0f},
            {0.1f, 0.0f, 0.0f},
        };
        const std::vector<sycl_points::RGBType> colors = {
            sycl_points::RGBType(0.2f, 0.4f, 0.6f, 1.0f),
            sycl_points::RGBType(0.6f, 0.2f, 0.0f, 1.0f),
        };
        const std::vector<float> intensities = {10.0f, 20.0f};

        auto cloud = MakePointCloud(queue, input_positions, &colors, &intensities);
        voxel_map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());

        ASSERT_EQ(result.size(), 1U);
        ASSERT_TRUE(result.has_rgb());
        ASSERT_TRUE(result.has_intensity());

        const auto point = (*result.points)[0];
        EXPECT_NEAR(point.x(), 0.05f, 1e-5f);
        EXPECT_NEAR(point.y(), 0.0f, 1e-5f);
        EXPECT_NEAR(point.z(), 0.0f, 1e-5f);

        const auto color = (*result.rgb)[0];
        EXPECT_NEAR(color.x(), 0.4f, 1e-5f);
        EXPECT_NEAR(color.y(), 0.3f, 1e-5f);
        EXPECT_NEAR(color.z(), 0.3f, 1e-5f);
        EXPECT_NEAR(color.w(), 1.0f, 1e-5f);

        const float intensity = (*result.intensities)[0];
        EXPECT_NEAR(intensity, 15.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(VoxelHashMapTest, AppliesMinimumPointThresholdPerVoxel) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, 0.2f);
        voxel_map.set_min_num_point(2);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.01f, 0.01f, 0.0f},  // First voxel, point 1
            {0.02f, 0.01f, 0.0f},  // First voxel, point 2
            {0.30f, 0.30f, 0.0f},  // Second voxel, only point
        };

        auto cloud = MakePointCloud(queue, input_positions);
        voxel_map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());

        ASSERT_EQ(result.size(), 1U);
        auto remaining_positions = ExtractPositions(*result.points);
        ASSERT_EQ(remaining_positions.size(), 1U);
        EXPECT_NEAR(remaining_positions[0].x(), 0.015f, 1e-5f);
        EXPECT_NEAR(remaining_positions[0].y(), 0.01f, 1e-5f);
        EXPECT_NEAR(remaining_positions[0].z(), 0.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(VoxelHashMapTest, PreservesAttributesWhenBelowThresholdCountsExist) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, 0.2f);
        voxel_map.set_min_num_point(3);

        const std::vector<Eigen::Vector3f> colored_positions = {
            {0.01f, 0.01f, 0.0f},
            {0.02f, 0.01f, 0.0f},
        };
        const std::vector<sycl_points::RGBType> colors = {
            sycl_points::RGBType(0.2f, 0.4f, 0.6f, 1.0f),
            sycl_points::RGBType(0.6f, 0.2f, 0.0f, 1.0f),
        };
        const std::vector<float> intensities = {10.0f, 20.0f};

        auto colored_cloud = MakePointCloud(queue, colored_positions, &colors, &intensities);
        voxel_map.add_point_cloud(colored_cloud, Eigen::Isometry3f::Identity());

        const std::vector<Eigen::Vector3f> colorless_positions = {
            {0.03f, 0.01f, 0.0f},
        };
        auto colorless_cloud = MakePointCloud(queue, colorless_positions);
        voxel_map.add_point_cloud(colorless_cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());

        ASSERT_EQ(result.size(), 1U);
        ASSERT_TRUE(result.has_rgb());
        ASSERT_TRUE(result.has_intensity());

        const auto color = (*result.rgb)[0];
        EXPECT_NEAR(color.x(), 0.4f, 1e-5f);
        EXPECT_NEAR(color.y(), 0.3f, 1e-5f);
        EXPECT_NEAR(color.z(), 0.3f, 1e-5f);
        EXPECT_NEAR(color.w(), 1.0f, 1e-5f);

        const float intensity = (*result.intensities)[0];
        EXPECT_NEAR(intensity, 15.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(VoxelHashMapTest, DownsamplingRespectsAxisAlignedBoundingBox) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, 0.2f);

        const std::vector<Eigen::Vector3f> input_positions = {
            {1.05f, 0.00f, 0.00f},  // Inside the bounding box after aggregation
            {1.12f, 0.00f, 0.00f},  // Aggregated with the first point
            {1.35f, 0.00f, 0.00f},  // Outside the queried bounding box
            {1.00f, 0.25f, 0.00f},  // Outside along the Y axis
        };

        auto cloud = MakePointCloud(queue, input_positions);
        voxel_map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f(1.0f, 0.0f, 0.0f), 0.2f);

        ASSERT_EQ(result.size(), 1U);

        const auto filtered_positions = ExtractPositions(*result.points);
        ASSERT_EQ(filtered_positions.size(), 1U);

        // The centroid of the first two points should remain inside the bounding box.
        EXPECT_NEAR(filtered_positions[0].x(), 1.085f, 1e-5f);
        EXPECT_NEAR(filtered_positions[0].y(), 0.0f, 1e-5f);
        EXPECT_NEAR(filtered_positions[0].z(), 0.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(VoxelHashMapTest, ComputesOverlapRatioFromPointCloud) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, 0.5f);

        const std::vector<Eigen::Vector3f> map_positions = {
            {0.1f, 0.1f, 0.0f},
            {1.1f, 0.0f, 0.0f},
        };

        auto map_cloud = MakePointCloud(queue, map_positions);
        voxel_map.add_point_cloud(map_cloud, Eigen::Isometry3f::Identity());

        const std::vector<Eigen::Vector3f> query_positions = {
            {-0.9f, 0.1f, 0.0f},
            {0.1f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
        };

        auto query_cloud = MakePointCloud(queue, query_positions);

        Eigen::Isometry3f sensor_pose = Eigen::Isometry3f::Identity();
        sensor_pose.translation() = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

        float overlap_ratio = voxel_map.compute_overlap_ratio(query_cloud, sensor_pose);
        EXPECT_NEAR(overlap_ratio, 2.0f / 3.0f, 1e-5f);

        voxel_map.set_min_num_point(2);
        overlap_ratio = voxel_map.compute_overlap_ratio(query_cloud, sensor_pose);
        EXPECT_NEAR(overlap_ratio, 0.0f, 1e-5f);

        voxel_map.add_point_cloud(map_cloud, Eigen::Isometry3f::Identity());
        overlap_ratio = voxel_map.compute_overlap_ratio(query_cloud, sensor_pose);
        EXPECT_NEAR(overlap_ratio, 2.0f / 3.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(VoxelHashMapTest, RemovesStaleVoxelsAfterConfiguredCycles) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, 0.1f);
        voxel_map.set_max_staleness(1);
        voxel_map.set_remove_old_data_cycle(1);

        auto old_cloud = MakePointCloud(queue, {{0.0f, 0.0f, 0.0f}});
        voxel_map.add_point_cloud(old_cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());
        ASSERT_EQ(result.size(), 1U);

        auto recent_cloud = MakePointCloud(queue, {{1.0f, 0.0f, 0.0f}});
        voxel_map.add_point_cloud(recent_cloud, Eigen::Isometry3f::Identity());
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());
        ASSERT_EQ(result.size(), 2U);

        sycl_points::PointCloudShared empty_cloud(queue);
        voxel_map.add_point_cloud(empty_cloud, Eigen::Isometry3f::Identity());
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());
        ASSERT_EQ(result.size(), 1U);

        auto remaining_positions = ExtractPositions(*result.points);
        ASSERT_EQ(remaining_positions.size(), 1U);
        EXPECT_NEAR(remaining_positions[0].x(), 1.0f, 1e-5f);
        EXPECT_NEAR(remaining_positions[0].y(), 0.0f, 1e-5f);
        EXPECT_NEAR(remaining_positions[0].z(), 0.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

// Morton code (Z-order curve) encoding tests
TEST(MortonCodeTest, ExpandBits21ProducesCorrectPattern) {
    using sycl_points::algorithms::filter::kernel::expand_bits_21;

    // Test single bits - each bit should be separated by 2 zero bits
    // Input bit 0 should go to output bit 0
    EXPECT_EQ(expand_bits_21(0b1), 0b1ULL);

    // Input bit 1 should go to output bit 3
    EXPECT_EQ(expand_bits_21(0b10), 0b1000ULL);

    // Input bit 2 should go to output bit 6
    EXPECT_EQ(expand_bits_21(0b100), 0b1000000ULL);

    // Input bits 0 and 1 should go to output bits 0 and 3
    EXPECT_EQ(expand_bits_21(0b11), 0b1001ULL);

    // Test all 21 bits set
    // Expected: bits at positions 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60
    EXPECT_EQ(expand_bits_21(0x1FFFFF), 0x1249249249249249ULL);

    // Test masking - input values larger than 21 bits should be masked
    EXPECT_EQ(expand_bits_21(0xFFFFFFFF), expand_bits_21(0x1FFFFF));
}

TEST(MortonCodeTest, MortonEncode3DInterleavesCorrectly) {
    using sycl_points::algorithms::filter::kernel::morton_encode_3d;

    // Test origin
    EXPECT_EQ(morton_encode_3d(0, 0, 0), 0ULL);

    // Test single coordinate values
    // X=1: bit 0 set -> output bit 0
    EXPECT_EQ(morton_encode_3d(1, 0, 0), 0b001ULL);

    // Y=1: bit 0 set -> output bit 1
    EXPECT_EQ(morton_encode_3d(0, 1, 0), 0b010ULL);

    // Z=1: bit 0 set -> output bit 2
    EXPECT_EQ(morton_encode_3d(0, 0, 1), 0b100ULL);

    // All coordinates = 1
    EXPECT_EQ(morton_encode_3d(1, 1, 1), 0b111ULL);

    // X=2 (bit 1 set): output bit 3
    EXPECT_EQ(morton_encode_3d(2, 0, 0), 0b001000ULL);

    // Y=2 (bit 1 set): output bit 4
    EXPECT_EQ(morton_encode_3d(0, 2, 0), 0b010000ULL);

    // Z=2 (bit 1 set): output bit 5
    EXPECT_EQ(morton_encode_3d(0, 0, 2), 0b100000ULL);

    // Combined test: X=3, Y=5, Z=7
    // X=3 = 0b11 -> bits 0,3 of output
    // Y=5 = 0b101 -> bits 1,4,7 of output (shifted by 1)
    // Z=7 = 0b111 -> bits 2,5,8 of output (shifted by 2)
    // Output: z2y2x2 z1y1x1 z0y0x0 = 111 101 011 = 0b111101011
    EXPECT_EQ(morton_encode_3d(3, 5, 7), 0b111101011ULL);
}

TEST(MortonCodeTest, MortonCodePreservesSpatialLocality) {
    using sycl_points::algorithms::filter::kernel::morton_encode_3d;

    // Adjacent voxels should have close Morton codes
    uint64_t origin = morton_encode_3d(100, 100, 100);
    uint64_t neighbor_x = morton_encode_3d(101, 100, 100);
    uint64_t neighbor_y = morton_encode_3d(100, 101, 100);
    uint64_t neighbor_z = morton_encode_3d(100, 100, 101);
    uint64_t far_point = morton_encode_3d(200, 200, 200);

    // Neighbors should differ by small amounts (within some bits)
    // The difference between adjacent cells is bounded by the Morton curve structure
    uint64_t diff_x = (origin > neighbor_x) ? (origin - neighbor_x) : (neighbor_x - origin);
    uint64_t diff_y = (origin > neighbor_y) ? (origin - neighbor_y) : (neighbor_y - origin);
    uint64_t diff_z = (origin > neighbor_z) ? (origin - neighbor_z) : (neighbor_z - origin);
    uint64_t diff_far = (origin > far_point) ? (origin - far_point) : (far_point - origin);

    // Neighbors should be closer than far points
    EXPECT_LT(diff_x, diff_far);
    EXPECT_LT(diff_y, diff_far);
    EXPECT_LT(diff_z, diff_far);
}

TEST(MortonCodeTest, ComputeVoxelBitUsesMortonEncoding) {
    using sycl_points::algorithms::filter::kernel::compute_voxel_bit;
    using sycl_points::algorithms::filter::kernel::morton_encode_3d;
    using sycl_points::algorithms::VoxelConstants;

    const float voxel_size = 1.0f;
    const float voxel_size_inv = 1.0f / voxel_size;

    // Test point at origin
    sycl_points::PointType origin_point(0.5f, 0.5f, 0.5f, 1.0f);
    uint64_t origin_hash = compute_voxel_bit(origin_point, voxel_size_inv);

    // Expected: coordinates (0,0,0) + offset
    uint64_t expected_x = 0 + VoxelConstants::coord_offset;
    uint64_t expected_y = 0 + VoxelConstants::coord_offset;
    uint64_t expected_z = 0 + VoxelConstants::coord_offset;
    uint64_t expected_hash = morton_encode_3d(expected_x, expected_y, expected_z);

    EXPECT_EQ(origin_hash, expected_hash);

    // Test point at (1.5, 2.5, 3.5) with voxel_size=1.0
    sycl_points::PointType test_point(1.5f, 2.5f, 3.5f, 1.0f);
    uint64_t test_hash = compute_voxel_bit(test_point, voxel_size_inv);

    expected_x = 1 + VoxelConstants::coord_offset;
    expected_y = 2 + VoxelConstants::coord_offset;
    expected_z = 3 + VoxelConstants::coord_offset;
    expected_hash = morton_encode_3d(expected_x, expected_y, expected_z);

    EXPECT_EQ(test_hash, expected_hash);

    // Test negative coordinates
    sycl_points::PointType neg_point(-0.5f, -1.5f, -2.5f, 1.0f);
    uint64_t neg_hash = compute_voxel_bit(neg_point, voxel_size_inv);

    expected_x = static_cast<uint64_t>(-1 + static_cast<int64_t>(VoxelConstants::coord_offset));
    expected_y = static_cast<uint64_t>(-2 + static_cast<int64_t>(VoxelConstants::coord_offset));
    expected_z = static_cast<uint64_t>(-3 + static_cast<int64_t>(VoxelConstants::coord_offset));
    expected_hash = morton_encode_3d(expected_x, expected_y, expected_z);

    EXPECT_EQ(neg_hash, expected_hash);
}
