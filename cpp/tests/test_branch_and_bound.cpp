#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>

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
    for (size_t i = 0; i < n; ++i) points.emplace_back(x(gen), y(gen), z(gen), 1.0f);
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
    p.radius_scale = 0.6f;
    const auto source_points = make_scene(180);
    const auto truth = (Eigen::Translation3f(3, -3, 0) * Eigen::AngleAxisf(0.5f, Eigen::Vector3f::UnitZ())).matrix();
    std::vector<PointType> target_points;
    for (const auto& point : source_points) target_points.push_back(truth * point);
    auto source = make_cloud(queue, source_points);
    auto target = make_cloud(queue, target_points);
    BranchAndBoundLocalizer localizer(queue, p);
    localizer.set_target(target);
    const auto result = localizer.localize(source);
    ASSERT_TRUE(result.converged);
    EXPECT_GT(result.inlier, 0U);
    EXPECT_LE((result.T.translation() - truth.block<3, 1>(0, 3)).norm(), 1.5f * p.min_resolution);
    const float yaw = std::atan2(result.T.rotation()(1, 0), result.T.rotation()(0, 0));
    EXPECT_LE(std::abs(std::remainder(yaw - 0.5f, 2.0f * std::numbers::pi_v<float>)), 0.3f);
}

TEST_F(LocalizationTest, AgreesWithExhaustiveLeafEnumeration) {
    auto p = small_params();
    p.max_level = 2;
    p.score_batch_size = 3;  // Force frequent flushes with mixed levels in the stock.
    p.min_translation = Eigen::Vector3f::Constant(-0.1f);
    p.max_translation = Eigen::Vector3f::Constant(0.1f);
    const auto points = make_scene(35);
    auto source = make_cloud(queue, points);
    auto target = make_cloud(queue, points);
    BranchAndBoundLocalizer localizer(queue, p);
    localizer.set_target(target);
    const auto result = localizer.localize(source);

    const auto resolutions = detail::level_resolutions(p);
    float max_norm = 0;
    for (const auto& point : points) max_norm = std::max(max_norm, point.head<3>().norm());
    const auto angular = detail::angular_info(p, resolutions, max_norm);
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
                                    const auto matrix =
                                        leaf.create_matrix(p.min_resolution, angular[0].rpy_res, angular[0].min_rpy);
                                    uint32_t count = 0;
                                    const float radius_sq = p.min_resolution * p.min_resolution;
                                    for (const auto& query : points) {
                                        const PointType transformed = matrix * query;
                                        for (const auto& point : points) {
                                            if ((transformed.head<3>() - point.head<3>()).squaredNorm() <= radius_sq) {
                                                ++count;
                                                break;
                                            }
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

TEST_F(LocalizationTest, ThresholdAndTimeout) {
    auto p = small_params();
    p.max_level = 0;
    p.max_rpy.setZero();
    p.score_threshold_ratio = 1.0f;
    auto scan = make_cloud(queue, {PointType(100, 0, 0, 1)});
    auto target = make_cloud(queue, {PointType(0, 0, 0, 1)});
    BranchAndBoundLocalizer localizer(queue, p);
    localizer.set_target(target);
    EXPECT_FALSE(localizer.localize(scan).converged);
    p.enable_timeout = true;
    p.timeout = std::chrono::milliseconds(0);
    BranchAndBoundLocalizer timed(queue, p);
    timed.set_target(target);
    const auto result = timed.localize(scan);
    EXPECT_TRUE(result.timed_out);
    EXPECT_FALSE(result.converged);
}

TEST_F(LocalizationTest, ValidatesInputs) {
    auto p = small_params();
    auto valid = make_cloud(queue, {PointType(0, 0, 0, 1)});
    PointCloudShared empty(queue);
    BranchAndBoundLocalizer localizer(queue, p);
    EXPECT_THROW(localizer.set_target(empty), std::invalid_argument);
    auto non_finite = make_cloud(queue, {PointType(std::numeric_limits<float>::quiet_NaN(), 0, 0, 1)});
    EXPECT_THROW(localizer.set_target(non_finite), std::invalid_argument);
    localizer.set_target(valid);
    EXPECT_THROW(localizer.localize(empty), std::invalid_argument);
    EXPECT_THROW(localizer.localize(non_finite), std::invalid_argument);
    auto invalid = [&](BranchAndBoundParams params) {
        EXPECT_THROW(BranchAndBoundLocalizer(queue, params), std::invalid_argument);
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
