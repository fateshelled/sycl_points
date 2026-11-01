#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <utility>

#include "sycl_points/io/point_cloud_reader.hpp"
#include "sycl_points/io/point_cloud_writer.hpp"
#include "sycl_points/pipeline/submapping.hpp"

namespace {

namespace sp = sycl_points;

sp::PointCloudShared MakeCloud(const sp::sycl_utils::DeviceQueue& queue, float y = 0.0f) {
    sp::PointCloudCPU cpu_cloud;
    *cpu_cloud.points = {
        sp::PointType(1.1f, y, 0.0f, 1.0f),
        sp::PointType(2.1f, y, 0.0f, 1.0f),
        sp::PointType(3.1f, y, 0.0f, 1.0f),
    };
    return sp::PointCloudShared(queue, cpu_cloud);
}

sp::pipeline::odometry::CommonParameters MakeParams(sp::pipeline::odometry::SubmapMapType map_type) {
    sp::pipeline::odometry::CommonParameters params;
    params.submap.map_type = map_type;
    params.submap.voxel_size = 0.5f;
    params.submap.point_random_sampling_num = 3;
    params.submap.max_distance_range = 100.0f;
    params.submap.occupancy_grid_map.enable_free_space_updates = false;
    params.submap.occupancy_grid_map.enable_pruning = false;
    params.registration.min_num_points = 1;
    params.registration.factor.reg_type = sp::algorithms::registration::RegType::POINT_TO_POINT;
    return params;
}

const void* AsVoid(const void* p) { return p; }

class StagedSubmapTest : public testing::TestWithParam<sp::pipeline::odometry::SubmapMapType> {};

TEST_P(StagedSubmapTest, EvictedKeyframesUpdateMapInPlaceAndRetarget) {
    sp::sycl_utils::DeviceQueue queue{sycl::device(sp::sycl_utils::device_selector::default_selector_v)};
    sp::pipeline::submapping::Submap submap(queue, MakeParams(GetParam()));
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 0U);

    submap.insert_evicted_keyframe(MakeCloud(queue), Eigen::Isometry3f::Identity());
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 3U);
    EXPECT_EQ(submap.get_last_keyframe_point_cloud().size(), 3U);

    // The target handle is the owned current generation: cloud == the published
    // submap cloud and kNN == the search structure built on exactly that cloud.
    const auto first = submap.get_target();
    ASSERT_NE(first.cloud, nullptr);
    ASSERT_NE(first.knn, nullptr);
    EXPECT_EQ(AsVoid(first.cloud.get()), AsVoid(&submap.get_submap_point_cloud()));
    EXPECT_EQ(AsVoid(first.knn.get()), AsVoid(&submap.get_submap_kdtree()));

    submap.insert_evicted_keyframe(MakeCloud(queue, 1.0f), Eigen::Isometry3f::Identity());
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 6U);

    // Insertion advances the generation: both handles are replaced together and
    // stay paired, and the kNN is rebuilt (not reused across generations).
    const auto second = submap.get_target();
    ASSERT_NE(second.cloud, nullptr);
    ASSERT_NE(second.knn, nullptr);
    EXPECT_EQ(AsVoid(second.cloud.get()), AsVoid(&submap.get_submap_point_cloud()));
    EXPECT_EQ(AsVoid(second.knn.get()), AsVoid(&submap.get_submap_kdtree()));
    EXPECT_NE(AsVoid(second.cloud.get()), AsVoid(first.cloud.get()));
    EXPECT_NE(AsVoid(second.knn.get()), AsVoid(first.knn.get()));
}

TEST_P(StagedSubmapTest, FullExportIncludesDistantPointsWithoutChangingTarget) {
    sp::sycl_utils::DeviceQueue queue{sycl::device(sp::sycl_utils::device_selector::default_selector_v)};
    auto params = MakeParams(GetParam());
    params.submap.max_distance_range = 5.0f;
    sp::pipeline::submapping::Submap submap(queue, params);
    auto cloud = MakeCloud(queue);
    cloud.resize_rgb(cloud.size());
    cloud.resize_intensities(cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i) {
        (*cloud.rgb)[i] = sp::RGBType(1.0f, 0.0f, 0.0f, 1.0f);
        (*cloud.intensities)[i] = 42.0f;
    }
    submap.add_first_frame(cloud, 0.0, Eigen::Isometry3f::Identity());
    Eigen::Isometry3f distant_pose = Eigen::Isometry3f::Identity();
    distant_pose.translation().x() = 100.0f;
    submap.insert_evicted_keyframe(cloud, distant_pose);
    const auto target = submap.get_target();
    ASSERT_EQ(target.cloud->size(), 3U);

    sp::PointCloudShared full_cloud(queue);
    submap.extract_full_point_cloud(full_cloud);
    ASSERT_EQ(full_cloud.size(), 6U);
    ASSERT_TRUE(full_cloud.has_rgb());
    ASSERT_TRUE(full_cloud.has_intensity());
    size_t distant_count = 0;
    for (size_t i = 0; i < full_cloud.size(); ++i) {
        distant_count += (*full_cloud.points)[i].x() > 100.0f;
        EXPECT_FLOAT_EQ((*full_cloud.intensities)[i], 42.0f);
        EXPECT_FLOAT_EQ((*full_cloud.rgb)[i].x(), 1.0f);
    }
    EXPECT_EQ(distant_count, 3U);
    EXPECT_EQ(submap.get_target().cloud, target.cloud);
    EXPECT_EQ(submap.get_target().knn, target.knn);
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 3U);

    // Exercise both file formats and encodings with the full exported map.
    for (const auto* extension : {".pcd", ".ply"}) {
        for (bool binary : {false, true}) {
            const auto path = std::filesystem::temp_directory_path() /
                              ("sycl_points_full_submap_" + std::to_string(static_cast<int>(GetParam())) +
                               (binary ? "_binary" : "_ascii") + extension);
            sp::PointCloudWriter::writeFile(path.string(), full_cloud, binary);
            if (std::string(extension) == ".pcd") {
                // The existing reader does not decode the writer's packed PCD RGB field.
                std::ifstream file(path, std::ios::binary);
                std::string line;
                std::getline(file, line);
                std::getline(file, line);
                std::getline(file, line);
                EXPECT_EQ(line, "FIELDS x y z rgb");
            }
            auto loaded = sp::PointCloudReader::readFile(path.string());
            std::filesystem::remove(path);
            ASSERT_EQ(loaded.size(), full_cloud.size());
            if (std::string(extension) == ".ply") {
                ASSERT_TRUE(loaded.has_rgb());
                ASSERT_TRUE(loaded.has_intensity());
            }
            for (size_t i = 0; i < loaded.size(); ++i) {
                EXPECT_TRUE((*loaded.points)[i].isApprox((*full_cloud.points)[i]));
                if (loaded.has_intensity()) {
                    EXPECT_FLOAT_EQ((*loaded.intensities)[i], 42.0f);
                }
            }
        }
    }
}

TEST_P(StagedSubmapTest, FullExportHandlesEmptyAndFirstFrameMaps) {
    sp::sycl_utils::DeviceQueue queue{sycl::device(sp::sycl_utils::device_selector::default_selector_v)};
    sp::pipeline::submapping::Submap submap(queue, MakeParams(GetParam()));
    auto result = MakeCloud(queue);
    submap.extract_full_point_cloud(result);
    EXPECT_EQ(result.size(), 0U);
    submap.add_first_frame(MakeCloud(queue), 0.0, Eigen::Isometry3f::Identity());
    submap.extract_full_point_cloud(result);
    EXPECT_EQ(result.size(), 3U);
}

INSTANTIATE_TEST_SUITE_P(AllMapBackends, StagedSubmapTest,
                         testing::Values(sp::pipeline::odometry::SubmapMapType::VOXEL_HASH_MAP,
                                         sp::pipeline::odometry::SubmapMapType::OCCUPANCY_GRID_MAP));

TEST(SubmapTest, PreparedFirstFramePublishesCloudAndMetadataOnlyOnCommit) {
    sp::sycl_utils::DeviceQueue queue{sycl::device(sp::sycl_utils::device_selector::default_selector_v)};
    auto params = MakeParams(sp::pipeline::odometry::SubmapMapType::VOXEL_HASH_MAP);
    sp::pipeline::submapping::Submap submap(queue, params);
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.translation() = Eigen::Vector3f(1.0f, 2.0f, 3.0f);

    auto prepared = submap.prepare_first_frame(MakeCloud(queue), 2.5, pose);
    using PreparedFirstFrame = sp::pipeline::submapping::Submap::PreparedFirstFrame;
    static_assert(noexcept(submap.commit_first_frame(std::declval<PreparedFirstFrame&&>())));
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 0u);
    EXPECT_TRUE(submap.get_last_keyframe_pose().isApprox(params.pose.initial));
    // The candidate is not published until commit: there is no target kNN yet.
    EXPECT_EQ(submap.get_target().knn, nullptr);

    submap.commit_first_frame(std::move(prepared));
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 3u);
    EXPECT_TRUE(submap.get_last_keyframe_pose().isApprox(pose));

    // First-frame target is the full transformed scan with its own kNN.
    const auto target = submap.get_target();
    ASSERT_NE(target.cloud, nullptr);
    ASSERT_NE(target.knn, nullptr);
    EXPECT_EQ(target.cloud->size(), 3u);
    EXPECT_EQ(AsVoid(target.cloud.get()), AsVoid(&submap.get_submap_point_cloud()));
    EXPECT_EQ(AsVoid(target.knn.get()), AsVoid(&submap.get_submap_kdtree()));
}

TEST(SubmapTest, AddFirstFramePublishesMapAndTarget) {
    sp::sycl_utils::DeviceQueue queue{sycl::device(sp::sycl_utils::device_selector::default_selector_v)};
    sp::pipeline::submapping::Submap submap(
        queue, MakeParams(sp::pipeline::odometry::SubmapMapType::VOXEL_HASH_MAP));
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.translation() = Eigen::Vector3f(0.5f, -1.0f, 2.0f);

    submap.add_first_frame(MakeCloud(queue), 1.0, pose);
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 3u);
    EXPECT_TRUE(submap.get_last_keyframe_pose().isApprox(pose));
    EXPECT_NE(submap.get_target().cloud, nullptr);
    EXPECT_NE(submap.get_target().knn, nullptr);
}

}  // namespace
