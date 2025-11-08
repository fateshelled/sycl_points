#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Geometry>

#include <sycl_points/algorithms/experimental/occupancy_grid_map.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/points/types.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace {

sycl_points::PointCloudShared MakePointCloud(
    const sycl_points::sycl_utils::DeviceQueue& queue,
    const std::vector<Eigen::Vector3f>& positions,
    const std::vector<sycl_points::RGBType>* colors = nullptr,
    const std::vector<float>* intensities = nullptr) {
    // Populate a CPU point cloud to create deterministic test data.
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

    return sycl_points::PointCloudShared(queue, cpu_cloud);
}

std::vector<Eigen::Vector3f> ExtractPositions(const sycl_points::PointContainerShared& points) {
    // Convert shared memory container into an STL vector for assertions.
    std::vector<Eigen::Vector3f> positions;
    positions.reserve(points.size());
    for (const auto& point : points) {
        positions.emplace_back(point.x(), point.y(), point.z());
    }
    return positions;
}

}  // namespace

TEST(OccupancyGridMapTest, ConstructorRejectsNonPositiveVoxelSize) {
    sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    EXPECT_THROW(sycl_points::algorithms::mapping::OccupancyGridMap(queue, 0.0f), std::invalid_argument);
    EXPECT_THROW(sycl_points::algorithms::mapping::OccupancyGridMap(queue, -0.1f), std::invalid_argument);
}

TEST(OccupancyGridMapTest, IntegratesPointsAndReturnsVisibleVoxels) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::OccupancyGridMap map(queue, 0.2f);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.05f, 0.05f, 0.0f},
            {0.07f, 0.05f, 0.0f},
            {0.35f, 0.05f, 0.0f},
        };

        auto cloud = MakePointCloud(queue, input_positions);
        map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        map.downsampling(result, Eigen::Isometry3f::Identity(), 1.0f);

        ASSERT_EQ(result.size(), 2U);
        auto positions = ExtractPositions(*result.points);
        std::sort(positions.begin(), positions.end(), [](const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
            if (lhs.x() != rhs.x()) {
                return lhs.x() < rhs.x();
            }
            if (lhs.y() != rhs.y()) {
                return lhs.y() < rhs.y();
            }
            return lhs.z() < rhs.z();
        });

        ASSERT_EQ(positions.size(), 2U);
        EXPECT_NEAR(positions[0].x(), 0.06f, 1e-5f);
        EXPECT_NEAR(positions[0].y(), 0.05f, 1e-5f);
        EXPECT_NEAR(positions[0].z(), 0.0f, 1e-5f);
        EXPECT_NEAR(positions[1].x(), 0.35f, 1e-5f);
        EXPECT_NEAR(positions[1].y(), 0.05f, 1e-5f);
        EXPECT_NEAR(positions[1].z(), 0.0f, 1e-5f);

        sycl_points::PointCloudShared raycast_result(queue);
        map.extract_visible_points(raycast_result, Eigen::Isometry3f::Identity(), 1.0f);

        ASSERT_EQ(raycast_result.size(), 2U);
        auto raycast_positions = ExtractPositions(*raycast_result.points);
        std::sort(raycast_positions.begin(), raycast_positions.end(),
                  [](const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
                      if (lhs.x() != rhs.x()) {
                          return lhs.x() < rhs.x();
                      }
                      if (lhs.y() != rhs.y()) {
                          return lhs.y() < rhs.y();
                      }
                      return lhs.z() < rhs.z();
                  });

        ASSERT_EQ(raycast_positions.size(), 2U);
        EXPECT_NEAR(raycast_positions[0].x(), 0.06f, 1e-5f);
        EXPECT_NEAR(raycast_positions[0].y(), 0.05f, 1e-5f);
        EXPECT_NEAR(raycast_positions[0].z(), 0.0f, 1e-5f);
        EXPECT_NEAR(raycast_positions[1].x(), 0.35f, 1e-5f);
        EXPECT_NEAR(raycast_positions[1].y(), 0.05f, 1e-5f);
        EXPECT_NEAR(raycast_positions[1].z(), 0.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(OccupancyGridMapTest, DownsamplingSkipsFarVoxels) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::OccupancyGridMap map(queue, 0.2f);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.0f, 0.0f, 0.0f},
            {5.0f, 0.0f, 0.0f},
        };

        auto cloud = MakePointCloud(queue, input_positions);
        map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        map.downsampling(result, Eigen::Isometry3f::Identity(), 1.0f);

        ASSERT_EQ(result.size(), 1U);
        const auto point = (*result.points)[0];
        EXPECT_NEAR(point.x(), 0.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(OccupancyGridMapTest, AggregatesColorAndIntensity) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::OccupancyGridMap map(queue, 0.1f);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.0f, 0.0f, 0.0f},
            {0.05f, 0.0f, 0.0f},
        };
        const std::vector<sycl_points::RGBType> colors = {
            sycl_points::RGBType(0.0f, 0.2f, 0.4f, 1.0f),
            sycl_points::RGBType(0.2f, 0.4f, 0.6f, 1.0f),
        };
        const std::vector<float> intensities = {10.0f, 30.0f};

        auto cloud = MakePointCloud(queue, input_positions, &colors, &intensities);
        map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        map.downsampling(result, Eigen::Isometry3f::Identity(), 1.0f);

        ASSERT_EQ(result.size(), 1U);
        ASSERT_TRUE(result.has_rgb());
        ASSERT_TRUE(result.has_intensity());

        const auto point = (*result.points)[0];
        EXPECT_NEAR(point.x(), 0.025f, 1e-5f);
        EXPECT_NEAR(point.y(), 0.0f, 1e-5f);
        EXPECT_NEAR(point.z(), 0.0f, 1e-5f);

        const auto color = (*result.rgb)[0];
        EXPECT_NEAR(color.x(), 0.1f, 1e-5f);
        EXPECT_NEAR(color.y(), 0.3f, 1e-5f);
        EXPECT_NEAR(color.z(), 0.5f, 1e-5f);
        EXPECT_NEAR(color.w(), 1.0f, 1e-5f);

        const float intensity = (*result.intensities)[0];
        EXPECT_NEAR(intensity, 20.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(OccupancyGridMapTest, VisibilityDecayReducesUnobservedVoxels) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::OccupancyGridMap map(queue, 0.1f);
        map.set_log_odds_hit(1.0f);
        map.set_log_odds_miss(-0.5f);

        const std::vector<Eigen::Vector3f> first_scan = {
            {0.0f, 0.0f, 0.0f},
            {0.2f, 0.0f, 0.0f},
        };
        auto first_cloud = MakePointCloud(queue, first_scan);
        map.add_point_cloud(first_cloud, Eigen::Isometry3f::Identity());

        const std::vector<Eigen::Vector3f> second_scan = {
            {0.0f, 0.0f, 0.0f},
        };
        auto second_cloud = MakePointCloud(queue, second_scan);
        map.add_point_cloud(second_cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        map.downsampling(result, Eigen::Isometry3f::Identity(), 1.0f);

        ASSERT_EQ(result.size(), 2U);
        auto positions = ExtractPositions(*result.points);
        std::sort(positions.begin(), positions.end(), [](const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
            return lhs.x() < rhs.x();
        });

        // The voxel that was not observed on the second scan should have lower confidence.
        EXPECT_LT(map.voxel_probability(positions[1]), map.voxel_probability(positions[0]));
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(OccupancyGridMapTest, ExtractVisiblePointsRemovesOccludedVoxels) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::OccupancyGridMap map(queue, 0.2f);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.2f, 0.0f, 0.0f},
            {0.6f, 0.0f, 0.0f},
            {0.0f, 0.2f, 0.0f},
        };

        auto cloud = MakePointCloud(queue, input_positions);
        map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared downsampled(queue);
        map.downsampling(downsampled, Eigen::Isometry3f::Identity(), 2.0f);
        ASSERT_EQ(downsampled.size(), 3U);

        sycl_points::PointCloudShared visible(queue);
        map.extract_visible_points(visible, Eigen::Isometry3f::Identity(), 2.0f);

        ASSERT_EQ(visible.size(), 2U);
        auto positions = ExtractPositions(*visible.points);
        std::sort(positions.begin(), positions.end(),
                  [](const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
                      if (lhs.x() != rhs.x()) {
                          return lhs.x() < rhs.x();
                      }
                      if (lhs.y() != rhs.y()) {
                          return lhs.y() < rhs.y();
                      }
                      return lhs.z() < rhs.z();
                  });

        ASSERT_EQ(positions.size(), 2U);
        EXPECT_NEAR(positions[0].x(), 0.0f, 1e-5f);
        EXPECT_NEAR(positions[0].y(), 0.2f, 1e-5f);
        EXPECT_NEAR(positions[0].z(), 0.0f, 1e-5f);
        EXPECT_NEAR(positions[1].x(), 0.2f, 1e-5f);
        EXPECT_NEAR(positions[1].y(), 0.0f, 1e-5f);
        EXPECT_NEAR(positions[1].z(), 0.0f, 1e-5f);

        Eigen::Isometry3f shifted_pose = Eigen::Isometry3f::Identity();
        shifted_pose.translation() = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        map.extract_visible_points(visible, shifted_pose, 2.0f);

        auto shifted_positions = ExtractPositions(*visible.points);
        const bool has_far_voxel = std::any_of(shifted_positions.begin(), shifted_positions.end(),
                                               [](const Eigen::Vector3f& position) {
                                                   return std::fabs(position.x() - 0.6f) < 1e-4f;
                                               });
        EXPECT_TRUE(has_far_voxel);

        const bool has_occluder = std::any_of(shifted_positions.begin(), shifted_positions.end(),
                                              [](const Eigen::Vector3f& position) {
                                                  return std::fabs(position.x() - 0.2f) < 1e-4f;
                                              });
        EXPECT_FALSE(has_occluder);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

