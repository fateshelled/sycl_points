#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "sycl_points/pipeline/prior_map.hpp"

namespace {

TEST(PriorMapTest, RotatesCovariancesAndReusesOnlyContainedTargets) {
    const auto path = std::filesystem::temp_directory_path() / "sycl_points_prior_map_test.ply";
    {
        std::ofstream file(path);
        file << "ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\n"
                "property float z\nproperty float cov_xx\nproperty float cov_xy\nproperty float cov_xz\n"
                "property float cov_yy\nproperty float cov_yz\nproperty float cov_zz\nend_header\n";
        for (int i = 0; i < 8; ++i) file << i << " 0 0 4 0 0 1 0 1\n";
    }
    sycl_points::sycl_utils::DeviceQueue queue{sycl::device{sycl_points::sycl_utils::device_selector::default_selector_v}};
    sycl_points::pipeline::PriorMapOptions options;
    options.downsample = false;
    options.local_radius = 3.0f;
    options.prediction_margin = 0.2f;
    options.local_min_points = 1;
    options.file_pose.rotate(Eigen::AngleAxisf(1.57079632679f, Eigen::Vector3f::UnitZ()));
    sycl_points::pipeline::PriorMap map(queue, path.string(), options);
    EXPECT_EQ(map.size(), 8U);
    auto first = map.target(Eigen::Vector3f(0, 1, 0), 1.0f);
    ASSERT_NE(first, nullptr);
    ASSERT_TRUE(first->cloud->has_cov());
    EXPECT_NEAR((*first->cloud->covs)[0](1, 1), 4.0f, 1e-4f);
    EXPECT_EQ(map.target(Eigen::Vector3f(0, 1.5f, 0), 1.0f), first);
    auto second = map.target(Eigen::Vector3f(0, 3, 0), 1.0f);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(second, first);
    EXPECT_EQ(map.target(Eigen::Vector3f(0, 3, 0), 3.0f), nullptr);
    std::filesystem::remove(path);
}

TEST(PriorMapTest, EstimatesMissingCovariancesAfterDownsampling) {
    const auto path = std::filesystem::temp_directory_path() / "sycl_points_prior_map_estimate_test.ply";
    {
        std::ofstream file(path);
        file << "ply\nformat ascii 1.0\nelement vertex 64\nproperty float x\nproperty float y\n"
                "property float z\nend_header\n";
        for (int i = 0; i < 64; ++i)
            file << (i % 4) << ' ' << (i / 4 % 4) << ' ' << (i / 16) << '\n';
    }
    sycl_points::sycl_utils::DeviceQueue queue{sycl::device{sycl_points::sycl_utils::device_selector::default_selector_v}};
    sycl_points::pipeline::PriorMapOptions options;
    options.voxel_size = 0.5f;
    options.covariance_neighbors = 8;
    options.local_min_points = 4;
    options.local_radius = 5;
    for (const char* mode : {"AUTO", "REESTIMATE"}) {
        options.covariance_mode = mode;
        sycl_points::pipeline::PriorMap map(queue, path.string(), options);
        EXPECT_EQ(map.size(), 64U);
        const auto target = map.target(Eigen::Vector3f(2, 2, 2), 2);
        ASSERT_NE(target, nullptr);
        ASSERT_TRUE(target->cloud->has_cov());
        for (const auto& cov : *target->cloud->covs) {
            EXPECT_TRUE((cov.topLeftCorner<3, 3>().allFinite()));
            EXPECT_GT(cov(0, 0), 0.0f);
        }
    }
    std::filesystem::remove(path);
}

}  // namespace
