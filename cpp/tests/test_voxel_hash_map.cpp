#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "sycl_points/algorithms/mapping/voxel_hash_map.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/points/types.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace {

sycl_points::PointCloudShared MakePointCloud(const sycl_points::sycl_utils::DeviceQueue& queue,
                                             const std::vector<Eigen::Vector3f>& positions,
                                             const std::vector<sycl_points::Covariance>* covariances = nullptr,
                                             const std::vector<sycl_points::RGBType>* colors = nullptr,
                                             const std::vector<float>* intensities = nullptr) {
    // Prepare a CPU point cloud to populate deterministic test data.
    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        const Eigen::Vector3f& pos = positions[i];
        (*cpu_cloud.points)[i] = sycl_points::PointType(pos.x(), pos.y(), pos.z(), 1.0f);
    }

    if (covariances) {
        EXPECT_EQ(covariances->size(), positions.size());
        cpu_cloud.covs->resize(positions.size());
        std::copy(covariances->begin(), covariances->end(), cpu_cloud.covs->begin());
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

sycl_points::Covariance MakeCovariance(float xx, float xy, float xz, float yy, float yz, float zz) {
    sycl_points::Covariance cov = sycl_points::Covariance::Zero();
    cov(0, 0) = xx;
    cov(0, 1) = xy;
    cov(1, 0) = xy;
    cov(0, 2) = xz;
    cov(2, 0) = xz;
    cov(1, 1) = yy;
    cov(1, 2) = yz;
    cov(2, 1) = yz;
    cov(2, 2) = zz;
    return cov;
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

        auto cloud = MakePointCloud(queue, input_positions, nullptr, &colors, &intensities);
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

TEST(VoxelHashMapTest, AggregatesCovariancesWithinVoxel) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, 0.5f);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.0f, 0.0f, 0.0f},
            {0.1f, 0.0f, 0.0f},
        };
        const std::vector<sycl_points::Covariance> covariances = {
            MakeCovariance(1.0f, 0.2f, 0.3f, 2.0f, 0.4f, 3.0f),
            MakeCovariance(3.0f, 0.6f, 0.9f, 4.0f, 0.8f, 5.0f),
        };
        const std::vector<sycl_points::RGBType> colors = {
            sycl_points::RGBType(0.2f, 0.4f, 0.6f, 1.0f),
            sycl_points::RGBType(0.6f, 0.2f, 0.0f, 1.0f),
        };
        const std::vector<float> intensities = {10.0f, 20.0f};

        auto cloud = MakePointCloud(queue, input_positions, &covariances, &colors, &intensities);
        voxel_map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());

        ASSERT_EQ(result.size(), 1U);
        ASSERT_TRUE(result.has_cov());
        ASSERT_TRUE(result.has_rgb());
        ASSERT_TRUE(result.has_intensity());

        const auto& cov = (*result.covs)[0];
        EXPECT_NEAR(cov(0, 0), 2.0f, 1e-5f);
        EXPECT_NEAR(cov(0, 1), 0.4f, 1e-5f);
        EXPECT_NEAR(cov(1, 0), 0.4f, 1e-5f);
        EXPECT_NEAR(cov(0, 2), 0.6f, 1e-5f);
        EXPECT_NEAR(cov(2, 0), 0.6f, 1e-5f);
        EXPECT_NEAR(cov(1, 1), 3.0f, 1e-5f);
        EXPECT_NEAR(cov(1, 2), 0.6f, 1e-5f);
        EXPECT_NEAR(cov(2, 1), 0.6f, 1e-5f);
        EXPECT_NEAR(cov(2, 2), 4.0f, 1e-5f);
        EXPECT_NEAR(cov.row(3).norm(), 0.0f, 1e-5f);
        EXPECT_NEAR(cov.col(3).norm(), 0.0f, 1e-5f);

        const auto color = (*result.rgb)[0];
        EXPECT_NEAR(color.x(), 0.4f, 1e-5f);
        EXPECT_NEAR(color.y(), 0.3f, 1e-5f);
        EXPECT_NEAR(color.z(), 0.3f, 1e-5f);
        EXPECT_NEAR(color.w(), 1.0f, 1e-5f);
        EXPECT_NEAR((*result.intensities)[0], 15.0f, 1e-5f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(VoxelHashMapTest, RotatesCovariancesIntoMapFrame) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, 0.5f);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.0f, 0.0f, 0.0f},
            {0.1f, 0.0f, 0.0f},
        };
        const std::vector<sycl_points::Covariance> covariances = {
            MakeCovariance(1.0f, 0.0f, 0.0f, 4.0f, 0.0f, 9.0f),
            MakeCovariance(9.0f, 0.0f, 0.0f, 16.0f, 0.0f, 25.0f),
        };

        auto cloud = MakePointCloud(queue, input_positions, &covariances);

        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
        pose.linear() = Eigen::AngleAxisf(static_cast<float>(M_PI_2), Eigen::Vector3f::UnitZ()).toRotationMatrix();
        pose.translation() = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

        voxel_map.add_point_cloud(cloud, pose);

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());

        ASSERT_EQ(result.size(), 1U);
        ASSERT_TRUE(result.has_cov());

        const auto& cov = (*result.covs)[0];
        EXPECT_NEAR(cov(0, 0), 10.0f, 1e-4f);
        EXPECT_NEAR(cov(1, 1), 5.0f, 1e-4f);
        EXPECT_NEAR(cov(2, 2), 17.0f, 1e-4f);
        EXPECT_NEAR(cov(0, 1), 0.0f, 1e-4f);
        EXPECT_NEAR(cov(0, 2), 0.0f, 1e-4f);
        EXPECT_NEAR(cov(1, 2), 0.0f, 1e-4f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(VoxelHashMapTest, SupportsLogEuclideanCovarianceAggregation) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, 0.5f);
        voxel_map.set_covariance_aggregation_mode(
            sycl_points::algorithms::mapping::CovarianceAggregationMode::LOG_EUCLIDEAN);

        const std::vector<Eigen::Vector3f> input_positions = {
            {0.0f, 0.0f, 0.0f},
            {0.1f, 0.0f, 0.0f},
        };
        const std::vector<sycl_points::Covariance> covariances = {
            MakeCovariance(1.0f, 0.0f, 0.0f, 4.0f, 0.0f, 9.0f),
            MakeCovariance(9.0f, 0.0f, 0.0f, 16.0f, 0.0f, 25.0f),
        };

        auto cloud = MakePointCloud(queue, input_positions, &covariances);
        voxel_map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());

        ASSERT_EQ(result.size(), 1U);
        ASSERT_TRUE(result.has_cov());

        const auto& cov = (*result.covs)[0];
        EXPECT_NEAR(cov(0, 0), 3.0f, 1e-4f);
        EXPECT_NEAR(cov(1, 1), 8.0f, 1e-4f);
        EXPECT_NEAR(cov(2, 2), 15.0f, 1e-4f);
        EXPECT_NEAR(cov(0, 1), 0.0f, 1e-4f);
        EXPECT_NEAR(cov(0, 2), 0.0f, 1e-4f);
        EXPECT_NEAR(cov(1, 2), 0.0f, 1e-4f);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(VoxelHashMapTest, KeepsCovarianceOutputDisabledWithoutInputCovariances) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, 0.5f);
        auto cloud = MakePointCloud(queue, {{0.0f, 0.0f, 0.0f}, {0.1f, 0.0f, 0.0f}});
        voxel_map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());

        ASSERT_EQ(result.size(), 1U);
        EXPECT_FALSE(result.has_cov());
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

TEST(VoxelHashMapTest, CountsVoxelsCorrectlyForLargeBatch) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        const float voxel_size = 1.0f;
        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, voxel_size);

        // Generate 100 points, each in a distinct voxel (spacing > voxel_size).
        const size_t num_points = 100;
        std::vector<Eigen::Vector3f> positions;
        positions.reserve(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            positions.emplace_back(static_cast<float>(i) * 2.0f + 0.5f, 0.5f, 0.5f);
        }

        auto cloud = MakePointCloud(queue, positions);
        voxel_map.add_point_cloud(cloud, Eigen::Isometry3f::Identity());

        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero(), 1000.0f);
        ASSERT_EQ(result.size(), num_points);

        auto out_positions = ExtractPositions(*result.points);
        std::sort(out_positions.begin(), out_positions.end(),
                  [](const Eigen::Vector3f& a, const Eigen::Vector3f& b) { return a.x() < b.x(); });
        for (size_t i = 0; i < num_points; ++i) {
            EXPECT_NEAR(out_positions[i].x(), static_cast<float>(i) * 2.0f + 0.5f, 1e-5f);
        }
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(VoxelHashMapTest, PreservesDataAfterRehash) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        const float voxel_size = 1.0f;
        sycl_points::algorithms::mapping::VoxelHashMap voxel_map(queue, voxel_size);
        // Set an extremely low threshold so that rehash is triggered after very few voxels.
        voxel_map.set_rehash_threshold(0.0f);

        // Insert a first batch of points. Each point lands in a separate voxel because
        // voxel_size is 1.0 and coordinates are spaced by 10.
        const std::vector<Eigen::Vector3f> batch1_positions = {
            {0.5f, 0.5f, 0.5f},
            {10.5f, 0.5f, 0.5f},
            {20.5f, 0.5f, 0.5f},
        };
        auto cloud1 = MakePointCloud(queue, batch1_positions);
        // This call should trigger rehash (threshold 0.0 < 0/30029 is false at first,
        // but after the first add the voxel count becomes 3, so next add triggers rehash).
        voxel_map.add_point_cloud(cloud1, Eigen::Isometry3f::Identity());

        // Insert a second batch to trigger rehash (voxel_num=3, 3/30029 > 0.0).
        const std::vector<Eigen::Vector3f> batch2_positions = {
            {30.5f, 0.5f, 0.5f},
            {40.5f, 0.5f, 0.5f},
        };
        auto cloud2 = MakePointCloud(queue, batch2_positions);
        voxel_map.add_point_cloud(cloud2, Eigen::Isometry3f::Identity());

        // After rehash, all 5 voxels should still be present.
        sycl_points::PointCloudShared result(queue);
        voxel_map.downsampling(result, Eigen::Vector3f::Zero());
        ASSERT_EQ(result.size(), 5U);

        auto positions = ExtractPositions(*result.points);
        std::sort(positions.begin(), positions.end(),
                  [](const Eigen::Vector3f& a, const Eigen::Vector3f& b) { return a.x() < b.x(); });

        EXPECT_NEAR(positions[0].x(), 0.5f, 1e-5f);
        EXPECT_NEAR(positions[1].x(), 10.5f, 1e-5f);
        EXPECT_NEAR(positions[2].x(), 20.5f, 1e-5f);
        EXPECT_NEAR(positions[3].x(), 30.5f, 1e-5f);
        EXPECT_NEAR(positions[4].x(), 40.5f, 1e-5f);
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
