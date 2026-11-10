#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <unordered_set>

#include "sycl_points/algorithms/localization/branch_and_bound.hpp"

namespace {
using namespace sycl_points;
using namespace sycl_points::algorithms::localization;

PointCloudShared make_cloud(const sycl_utils::DeviceQueue& queue, const std::vector<PointType>& points) {
    PointCloudShared result(queue);
    result.points->assign(points.begin(), points.end());
    return result;
}

std::vector<PointType> make_scene(size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> x(-7.0f, 5.0f), y(-4.0f, 9.0f), z(-2.0f, 3.0f);
    std::vector<PointType> points;
    for (size_t i = 0; i < n; ++i) {
        points.emplace_back(x(gen), y(gen), z(gen), 1.0f);
    }
    return points;
}

BranchAndBoundParams small_params() {
    BranchAndBoundParams p;
    p.auto_translation_range = false;
    p.min_resolution = 1.5f;
    p.max_level = 1;
    p.min_translation = Eigen::Vector3f::Zero();
    p.max_translation = Eigen::Vector3f::Zero();
    p.max_rpy = Eigen::Vector3f(0, 0, 0.4f);
    p.score_batch_size = 64;
    return p;
}

class LocalizationTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue{sycl::device(sycl_utils::device_selector::default_selector_v)};
};

TEST_F(LocalizationTest, RecoversKnownTransform) {
    auto p = small_params();
    p.max_level = 2;
    p.max_rpy.z() = 2.0f * std::numbers::pi_v<float>;
    p.min_translation = Eigen::Vector3f(3, -3, 0);
    p.max_translation = p.min_translation;
    const auto source_points = make_scene(180);
    const auto truth = (Eigen::Translation3f(3, -3, 0) * Eigen::AngleAxisf(0.5f, Eigen::Vector3f::UnitZ())).matrix();
    std::vector<PointType> target_points;
    for (const auto& point : source_points) {
        target_points.push_back(truth * point);
    }
    auto source = make_cloud(this->queue, source_points);
    auto target = make_cloud(this->queue, target_points);
    BranchAndBoundLocalizer localizer(this->queue, p);
    localizer.set_target(target);
    const auto result = localizer.localize(source);
    ASSERT_TRUE(result.converged);
    EXPECT_GT(result.inlier, 0U);
    EXPECT_LE((result.T.translation() - truth.block<3, 1>(0, 3)).norm(), 1.5f * p.min_resolution);
    const float yaw = std::atan2(result.T.rotation()(1, 0), result.T.rotation()(0, 0));
    EXPECT_LE(std::abs(std::remainder(yaw - 0.5f, 2.0f * std::numbers::pi_v<float>)), 0.3f);
}

TEST_F(LocalizationTest, AppliesScanToReferenceTransform) {
    auto p = small_params();
    p.max_level = 2;
    p.max_rpy.z() = 2.0f * std::numbers::pi_v<float>;
    p.min_translation = Eigen::Vector3f(3, -3, 0);
    p.max_translation = p.min_translation;
    const auto scene = make_scene(180);

    // The scan frame is offset from the reference frame whose pose is estimated.
    Eigen::Isometry3f reference_from_scan = Eigen::Isometry3f::Identity();
    reference_from_scan.translation() = Eigen::Vector3f(5.0f, -2.0f, 1.0f);
    reference_from_scan.linear() = Eigen::AngleAxisf(-0.7f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    Eigen::Isometry3f truth = Eigen::Isometry3f::Identity();
    truth.translation() = Eigen::Vector3f(3.0f, -3.0f, 0.0f);
    truth.linear() = Eigen::AngleAxisf(0.5f, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    std::vector<PointType> scan_points;
    std::vector<PointType> target_points;
    for (const auto& point : scene) {
        scan_points.push_back(reference_from_scan.inverse() * point);
        target_points.push_back(truth * point);
    }
    auto scan = make_cloud(this->queue, scan_points);
    auto target = make_cloud(this->queue, target_points);
    BranchAndBoundLocalizer localizer(this->queue, p);
    localizer.set_target(target);
    const auto result = localizer.localize(scan, reference_from_scan);
    ASSERT_TRUE(result.converged);
    EXPECT_LE((result.T.translation() - truth.translation()).norm(), 1.5f * p.min_resolution);
    const float yaw = std::atan2(result.T.rotation()(1, 0), result.T.rotation()(0, 0));
    EXPECT_LE(std::abs(std::remainder(yaw - 0.5f, 2.0f * std::numbers::pi_v<float>)), 0.3f);
}

TEST_F(LocalizationTest, AgreesWithExhaustiveLeafEnumeration) {
    auto p = small_params();
    p.max_level = 2;
    p.score_batch_size = 3;  // Use small batches for candidates at different levels.
    p.min_translation = Eigen::Vector3f::Constant(-0.1f);
    p.max_translation = Eigen::Vector3f::Constant(0.1f);
    const auto points = make_scene(35);
    auto source = make_cloud(this->queue, points);
    auto target = make_cloud(this->queue, points);
    BranchAndBoundLocalizer localizer(this->queue, p);
    localizer.set_target(target);
    const auto result = localizer.localize(source);

    const auto resolutions = detail::level_resolutions(p);
    float max_norm = 0;
    for (const auto& point : points) {
        max_norm = std::max(max_norm, point.head<3>().norm());
    }
    const auto angular = detail::angular_info(p, resolutions, max_norm);

    // Build the leaf-level occupancy set with the same negative dilation as the scorer.
    std::unordered_set<uint64_t> occupied;
    const float leaf_inv = 1.0f / resolutions[0];
    for (const auto& point : points) {
        voxel_detail::HostVoxelCoord coord;
        if (!voxel_detail::compute_voxel_coord_host(point, leaf_inv, coord)) {
            continue;
        }
        for (int dx = 0; dx >= -1; --dx) {
            for (int dy = 0; dy >= -1; --dy) {
                for (int dz = 0; dz >= -1; --dz) {
                    if (coord.x + dx < 0 || coord.y + dy < 0 || coord.z + dz < 0) {
                        continue;
                    }
                    occupied.insert(voxel_detail::encode_voxel_key(coord.x + dx, coord.y + dy, coord.z + dz));
                }
            }
        }
    }

    const int top_low[3] = {
        detail::checked_int(std::floor(static_cast<double>(p.min_translation.x()) / resolutions.back())),
        detail::checked_int(std::floor(static_cast<double>(p.min_translation.y()) / resolutions.back())),
        detail::checked_int(std::floor(static_cast<double>(p.min_translation.z()) / resolutions.back()))};
    const int top_high[3] = {
        detail::checked_int(std::ceil(static_cast<double>(p.max_translation.x()) / resolutions.back())),
        detail::checked_int(std::ceil(static_cast<double>(p.max_translation.y()) / resolutions.back())),
        detail::checked_int(std::ceil(static_cast<double>(p.max_translation.z()) / resolutions.back()))};
    uint32_t expected = 0;
    for (int x = top_low[0]; x <= top_high[0]; ++x) {
        for (int y = top_low[1]; y <= top_high[1]; ++y) {
            for (int z = top_low[2]; z <= top_high[2]; ++z) {
                for (int roll = 0; roll < angular.back().num_division.x(); ++roll) {
                    for (int pitch = 0; pitch < angular.back().num_division.y(); ++pitch) {
                        for (int yaw = 0; yaw < angular.back().num_division.z(); ++yaw) {
                            detail::DiscreteTransformation root{0, p.max_level, x, y, z, roll, pitch, yaw};
                            std::vector<detail::DiscreteTransformation> upper;
                            root.branch(upper, p.max_level - 1, p.branch_factor, angular[p.max_level - 1].num_division);
                            for (const auto& node : upper) {
                                std::vector<detail::DiscreteTransformation> leaves;
                                node.branch(leaves, 0, p.branch_factor, angular[0].num_division);
                                for (const auto& leaf : leaves) {
                                    const Eigen::Vector3f lower(
                                        p.min_resolution * static_cast<float>(leaf.x),
                                        p.min_resolution * static_cast<float>(leaf.y),
                                        p.min_resolution * static_cast<float>(leaf.z));
                                    const Eigen::Array3f upper = lower.array() + p.min_resolution;
                                    if (!((upper > p.min_translation.array()).all() &&
                                          (lower.array() <= p.max_translation.array()).all())) {
                                        continue;
                                    }
                                    const auto matrix =
                                        leaf.create_matrix(p.min_resolution, angular[0].rpy_res, angular[0].min_rpy);
                                    uint32_t count = 0;
                                    for (const auto& query : points) {
                                        const PointType transformed = matrix * query;
                                        voxel_detail::HostVoxelCoord coord;
                                        if (!voxel_detail::compute_voxel_coord_host(transformed, leaf_inv, coord)) {
                                            continue;
                                        }
                                        if (occupied.count(voxel_detail::encode_voxel_key(coord.x, coord.y, coord.z)) !=
                                            0) {
                                            ++count;
                                        }
                                    }
                                    expected = std::max(expected, count);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ASSERT_TRUE(result.converged);
    EXPECT_EQ(result.inlier, expected);
}

TEST_F(LocalizationTest, ReportsProfile) {
    auto p = small_params();
    p.max_level = 2;
    p.min_translation = Eigen::Vector3f::Constant(-0.1f);
    p.max_translation = Eigen::Vector3f::Constant(0.1f);
    const auto points = make_scene(35);
    auto source = make_cloud(this->queue, points);
    auto target = make_cloud(this->queue, points);
    BranchAndBoundLocalizer localizer(this->queue, p);
    localizer.set_target(target);
    const auto result = localizer.localize(source);

    const auto& profile = result.profile;
    EXPECT_EQ(profile.levels.size(), static_cast<size_t>(p.max_level) + 1);
    EXPECT_GT(profile.initial_candidates, 0U);
    EXPECT_GT(profile.flush_calls, 0U);
    EXPECT_GT(profile.score_batches, 0U);
    EXPECT_EQ(profile.initial_candidates, profile.levels.back().evaluated);
    EXPECT_GE(profile.total_ms, result.elapsed_ms);
    EXPECT_GT(profile.total_ms, 0.0);
}

TEST_F(LocalizationTest, ThresholdAndTimeout) {
    auto p = small_params();
    p.max_level = 0;
    p.max_rpy.setZero();
    p.score_threshold_ratio = 1.0f;
    auto scan = make_cloud(this->queue, {PointType(100, 0, 0, 1)});
    auto target = make_cloud(this->queue, {PointType(0, 0, 0, 1)});
    BranchAndBoundLocalizer localizer(this->queue, p);
    localizer.set_target(target);
    EXPECT_FALSE(localizer.localize(scan).converged);
    p.enable_timeout = true;
    p.timeout = std::chrono::milliseconds(0);
    BranchAndBoundLocalizer timed(this->queue, p);
    timed.set_target(target);
    const auto result = timed.localize(scan);
    EXPECT_TRUE(result.timed_out);
    EXPECT_FALSE(result.converged);
}

TEST(LocalizationDetailTest, MatchesReferenceAngularSchedule) {
    BranchAndBoundParams p;
    p.min_resolution = 1.5f;
    p.max_level = 1;
    p.max_rpy = Eigen::Vector3f(0.0f, 0.0f, 0.4f);
    const auto resolutions = detail::level_resolutions(p);
    const auto angular = detail::angular_info(p, resolutions, 5.0f);

    // 3D-BBS keeps an undivided level at the zero angle and bisects a divided level from min_rpy.
    ASSERT_EQ(angular[1].num_division.z(), 1);
    EXPECT_FLOAT_EQ(angular[1].rpy_res.z(), 0.0f);
    EXPECT_FLOAT_EQ(angular[1].min_rpy.z(), 0.0f);

    ASSERT_EQ(angular[0].num_division.z(), 2);
    EXPECT_NEAR(angular[0].rpy_res.z(), 0.2f, 1e-6f);
    EXPECT_FLOAT_EQ(angular[0].min_rpy.z(), 0.0f);

    // Zero-width roll and pitch ranges stay undivided.
    EXPECT_EQ(angular[0].num_division.x(), 1);
    EXPECT_EQ(angular[0].num_division.y(), 1);
}

TEST_F(LocalizationTest, AcceptsLeavesThatOverlapTranslationRange) {
    auto p = small_params();
    p.max_level = 2;
    p.max_rpy.setZero();
    p.min_translation = Eigen::Vector3f::Constant(-0.1f);
    p.max_translation = Eigen::Vector3f::Constant(-0.1f);
    const auto points = make_scene(35);
    auto source = make_cloud(this->queue, points);
    auto target = make_cloud(this->queue, points);
    BranchAndBoundLocalizer localizer(this->queue, p);
    localizer.set_target(target);

    const auto result = localizer.localize(source);
    ASSERT_TRUE(result.converged);
    // The leaf cell has one minimum-resolution voxel as the rounding margin.
    const Eigen::Vector3f translation = result.T.translation();
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_GE(translation[axis], p.min_translation[axis] - p.min_resolution);
        EXPECT_LE(translation[axis], p.max_translation[axis]);
    }
}

TEST_F(LocalizationTest, RejectsLeavesOutsideTranslationRange) {
    auto p = small_params();
    p.max_level = 2;
    p.max_rpy.setZero();
    p.min_translation = Eigen::Vector3f::Zero();
    p.max_translation = Eigen::Vector3f::Zero();
    auto source = make_cloud(this->queue, {PointType(0, 0, 0, 1)});
    auto target = make_cloud(this->queue, {PointType(3, 0, 0, 1)});
    BranchAndBoundLocalizer localizer(this->queue, p);
    localizer.set_target(target);

    // The only point pair is 3 m apart, but the requested range is zero.
    const auto result = localizer.localize(source);
    EXPECT_FALSE(result.converged);
    EXPECT_GT(result.profile.out_of_range_candidates, 0U);
}

TEST_F(LocalizationTest, TimesOutDuringInitialEnumeration) {
    auto p = small_params();
    p.max_level = 3;
    p.max_rpy.setZero();
    p.min_translation = Eigen::Vector3f::Constant(-50.0f);
    p.max_translation = Eigen::Vector3f::Constant(50.0f);
    p.score_batch_size = 8;
    p.enable_timeout = true;
    p.timeout = std::chrono::milliseconds(0);
    auto scan = make_cloud(this->queue, {PointType(0, 0, 0, 1)});
    auto target = make_cloud(this->queue, {PointType(0, 0, 0, 1)});
    BranchAndBoundLocalizer localizer(this->queue, p);
    localizer.set_target(target);

    const auto result = localizer.localize(scan);
    EXPECT_TRUE(result.timed_out);
    EXPECT_FALSE(result.converged);
    // The deadline stops the initial grid before all candidates get a score.
    EXPECT_LT(result.evaluated_candidates, result.profile.initial_candidates);
}

TEST_F(LocalizationTest, ValidatesInputs) {
    auto p = small_params();
    auto valid = make_cloud(this->queue, {PointType(0, 0, 0, 1)});
    PointCloudShared empty(this->queue);
    BranchAndBoundLocalizer localizer(this->queue, p);
    EXPECT_THROW(localizer.set_target(empty), std::invalid_argument);
    auto non_finite = make_cloud(this->queue, {PointType(std::numeric_limits<float>::quiet_NaN(), 0, 0, 1)});
    EXPECT_THROW(localizer.set_target(non_finite), std::invalid_argument);
    localizer.set_target(valid);
    EXPECT_THROW(localizer.localize(empty), std::invalid_argument);
    EXPECT_THROW(localizer.localize(non_finite), std::invalid_argument);
    Eigen::Isometry3f invalid_transform = Eigen::Isometry3f::Identity();
    invalid_transform.translation().x() = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(localizer.localize(valid, invalid_transform), std::invalid_argument);
    auto invalid = [&](BranchAndBoundParams params) {
        EXPECT_THROW(BranchAndBoundLocalizer(this->queue, params), std::invalid_argument);
    };
    p.max_level = -1;
    invalid(p);
    p.max_level = 1;
    p.branch_factor = 1;
    invalid(p);
    p.branch_factor = 2;
    p.min_resolution = 0;
    invalid(p);
    p.min_resolution = 1.5f;
    p.min_translation.x() = 1;
    invalid(p);
    p.min_translation.x() = 0;
    p.score_threshold_ratio = -0.1f;
    invalid(p);
    p.score_threshold_ratio = 1.1f;
    invalid(p);
}
}  // namespace
