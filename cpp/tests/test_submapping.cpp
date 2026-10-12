#include <gtest/gtest.h>

#include <utility>

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
