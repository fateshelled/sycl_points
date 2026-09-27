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

class StagedSubmapTest : public testing::TestWithParam<sp::pipeline::odometry::SubmapMapType> {};

TEST_P(StagedSubmapTest, EvictedKeyframesUpdateMapInPlace) {
    sp::sycl_utils::DeviceQueue queue{sycl::device(sp::sycl_utils::device_selector::default_selector_v)};
    sp::pipeline::submapping::Submap submap(queue, MakeParams(GetParam()));
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 0U);

    submap.freeze_keyframe_to_submap(MakeCloud(queue), Eigen::Isometry3f::Identity());
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 3U);
    EXPECT_EQ(submap.get_last_keyframe_point_cloud().size(), 3U);

    submap.freeze_keyframe_to_submap(MakeCloud(queue, 1.0f), Eigen::Isometry3f::Identity());
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 6U);
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
    using PreparedFreeze = sp::pipeline::submapping::Submap::PreparedFreeze;
    static_assert(noexcept(submap.commit_freeze_keyframe_to_submap(std::declval<PreparedFreeze&&>())));
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 0u);
    EXPECT_TRUE(submap.get_last_keyframe_pose().isApprox(params.pose.initial));

    submap.commit_freeze_keyframe_to_submap(std::move(prepared));
    EXPECT_EQ(submap.get_submap_point_cloud().size(), 3u);
    EXPECT_TRUE(submap.get_last_keyframe_pose().isApprox(pose));
}

}  // namespace
