#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/localization/branch_and_bound_detail.hpp"
#include "sycl_points/algorithms/localization/branch_and_bound_params.hpp"
#include "sycl_points/algorithms/localization/branch_and_bound_result.hpp"

namespace sycl_points::algorithms::localization {

/// @brief Best-first multiresolution global localization against a KDTree map.
class BranchAndBoundLocalizer {
public:
    /// @brief Construct and validate a localizer using the given device queue.
    BranchAndBoundLocalizer(const sycl_utils::DeviceQueue& queue, const BranchAndBoundParams& params)
        : queue_(queue), params_(params) {
        const auto& p = params_;
        if (!std::isfinite(p.min_resolution) || p.min_resolution <= 0 || p.max_level < 0 || p.branch_factor < 2 ||
            !p.min_translation.allFinite() || !p.max_translation.allFinite() ||
            !(p.min_translation.array() <= p.max_translation.array()).all() || !p.min_rpy.allFinite() ||
            !p.max_rpy.allFinite() || !(p.min_rpy.array() <= p.max_rpy.array()).all() ||
            !std::isfinite(p.radius_scale) || p.radius_scale <= 0 || !std::isfinite(p.correspondence_distance) ||
            p.correspondence_distance < 0 || !std::isfinite(p.score_threshold_ratio) || p.score_threshold_ratio < 0 ||
            p.score_threshold_ratio > 1 || p.score_batch_size == 0 || (p.enable_timeout && p.timeout.count() < 0)) {
            throw std::invalid_argument("[BranchAndBoundLocalizer] invalid parameters");
        }
        resolutions_ = detail::level_resolutions(params_);
        for (float res : resolutions_) {
            if (!std::isfinite(p.correspondence_distance + p.radius_scale * res))
                throw std::invalid_argument("[BranchAndBoundLocalizer] radius overflow");
        }
    }

    /// @brief Build the target KDTree and optionally adopt the map's bounding box.
    /// @throws std::invalid_argument if the map is empty or contains non-finite coordinates.
    void set_target(const PointCloudShared& map) {
        validate_cloud(map);
        Eigen::Vector3f low = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
        Eigen::Vector3f high = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
        for (const auto& point : *map.points) {
            low = low.cwiseMin(point.head<3>());
            high = high.cwiseMax(point.head<3>());
        }
        auto tree = knn::KDTree::build(queue_, map);
        tree_ = std::move(tree);
        target_size_ = map.size();
        if (params_.auto_translation_range) {
            min_translation_ = low;
            max_translation_ = high;
        } else {
            min_translation_ = params_.min_translation;
            max_translation_ = params_.max_translation;
        }
    }

    /// @brief Locate a gravity-aligned scan in the target map.
    /// @throws std::invalid_argument if the scan is empty or contains non-finite coordinates.
    /// @throws std::runtime_error if no target was set.
    BranchAndBoundResult localize(const PointCloudShared& scan) {
        validate_cloud(scan);
        if (!tree_) throw std::runtime_error("[BranchAndBoundLocalizer] target not set");
        if (scan.size() > std::numeric_limits<uint32_t>::max())
            throw std::overflow_error("[BranchAndBoundLocalizer] scan exceeds score range");

        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        BranchAndBoundResult result;
        float max_norm = 0.0f;
        for (const auto& point : *scan.points) max_norm = std::max(max_norm, point.head<3>().norm());
        const auto angular = detail::angular_info(params_, resolutions_, max_norm);
        const float top_res = resolutions_.back();
        int low[3], high[3];
        for (int axis = 0; axis < 3; ++axis) {
            low[axis] = detail::checked_int(std::floor(static_cast<double>(min_translation_[axis]) / top_res));
            high[axis] = detail::checked_int(std::ceil(static_cast<double>(max_translation_[axis]) / top_res));
        }
        size_t initial_count = 1;
        auto multiply_checked = [&](size_t count) {
            if (count && initial_count > std::numeric_limits<size_t>::max() / count)
                throw std::overflow_error("[BranchAndBoundLocalizer] candidate count overflow");
            initial_count *= count;
        };
        for (int axis = 0; axis < 3; ++axis) {
            multiply_checked(static_cast<size_t>(static_cast<int64_t>(high[axis]) - low[axis] + 1));
            multiply_checked(static_cast<size_t>(angular.back().num_division[axis]));
        }

        const uint32_t threshold =
            static_cast<uint32_t>(std::floor(static_cast<double>(scan.size()) * params_.score_threshold_ratio));
        uint32_t best_score = threshold;
        detail::DiscreteTransformation best;
        bool have_leaf = false;
        std::priority_queue<detail::DiscreteTransformation> queue;
        std::vector<detail::DiscreteTransformation> stock;
        stock.reserve(std::min(initial_count, params_.score_batch_size));

        auto flush = [&]() {
            if (stock.empty()) return;
            // Branch stock can mix levels. Bucket the indices per level so that
            // every candidate is scored exactly once with its level's radius.
            std::vector<size_t> indices;
            for (int level = 0; level <= params_.max_level; ++level) {
                indices.clear();
                for (size_t i = 0; i < stock.size(); ++i) {
                    if (stock[i].level == level) indices.push_back(i);
                }
                for (size_t begin = 0; begin < indices.size(); begin += params_.score_batch_size) {
                    const size_t count = std::min(params_.score_batch_size, indices.size() - begin);
                    shared_vector<TransformMatrix> matrices(count, *queue_.ptr);
                    shared_vector<uint32_t> counts(count, 0U, *queue_.ptr);
                    for (size_t i = 0; i < count; ++i) {
                        matrices[i] = stock[indices[begin + i]].create_matrix(
                            resolutions_[level], angular[level].rpy_res, angular[level].min_rpy);
                    }
                    const float radius = params_.correspondence_distance + params_.radius_scale * resolutions_[level];
                    tree_->count_neighbors_within_radius_async(scan, matrices, radius, counts).wait_and_throw();
                    for (size_t i = 0; i < count; ++i) {
                        auto& candidate = stock[indices[begin + i]];
                        candidate.score = counts[i];
                        if (candidate.score >= best_score) queue.push(candidate);
                    }
                    result.evaluated_candidates += count;
                }
            }
            stock.clear();
        };

        // Stream the initial grid in batches instead of allocating all candidates at once.
        for (int64_t x = low[0]; x <= high[0]; ++x)
            for (int64_t y = low[1]; y <= high[1]; ++y)
                for (int64_t z = low[2]; z <= high[2]; ++z)
                    for (int roll = 0; roll < angular.back().num_division.x(); ++roll)
                        for (int pitch = 0; pitch < angular.back().num_division.y(); ++pitch)
                            for (int yaw = 0; yaw < angular.back().num_division.z(); ++yaw) {
                                stock.push_back({0, params_.max_level, static_cast<int>(x), static_cast<int>(y),
                                                 static_cast<int>(z), roll, pitch, yaw});
                                if (stock.size() >= params_.score_batch_size) flush();
                            }
        flush();

        // Initial scoring is included in the deadline, even when all initial
        // candidates were pruned and the priority queue is already empty.
        if (params_.enable_timeout && Clock::now() - start >= params_.timeout) result.timed_out = true;

        while (!result.timed_out && (!queue.empty() || !stock.empty())) {
            if (params_.enable_timeout && Clock::now() - start >= params_.timeout) {
                result.timed_out = true;
                break;
            }
            if (queue.empty()) {
                flush();
                continue;
            }
            auto candidate = queue.top();
            queue.pop();
            if (candidate.score < best_score) continue;
            if (candidate.is_leaf()) {
                best_score = candidate.score;
                best = candidate;
                have_leaf = true;
            } else {
                candidate.branch(stock, candidate.level - 1, params_.branch_factor,
                                 angular[candidate.level - 1].num_division);
            }
            if (stock.size() >= params_.score_batch_size || (queue.empty() && !stock.empty())) flush();
        }

        result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (have_leaf) {
            result.inlier = best_score;
            result.score_ratio = static_cast<float>(best_score) / static_cast<float>(scan.size());
        }
        result.converged = have_leaf && best_score > threshold && !result.timed_out;
        if (result.converged)
            result.T.matrix() = best.create_matrix(params_.min_resolution, angular[0].rpy_res, angular[0].min_rpy);
        return result;
    }

    /// @brief Current configuration.
    const BranchAndBoundParams& params() const { return params_; }
    /// @brief Number of points in the most recently set target.
    size_t target_size() const { return target_size_; }

private:
    static void validate_cloud(const PointCloudShared& cloud) {
        if (!cloud.points || cloud.size() == 0) throw std::invalid_argument("[BranchAndBoundLocalizer] empty cloud");
        for (const auto& point : *cloud.points)
            if (!point.allFinite()) throw std::invalid_argument("[BranchAndBoundLocalizer] non-finite point");
    }

    sycl_utils::DeviceQueue queue_;
    BranchAndBoundParams params_;
    std::vector<float> resolutions_;
    knn::KDTree::Ptr tree_;
    size_t target_size_ = 0;
    Eigen::Vector3f min_translation_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f max_translation_ = Eigen::Vector3f::Zero();
};

}  // namespace sycl_points::algorithms::localization
