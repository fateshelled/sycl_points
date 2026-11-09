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

#include "sycl_points/algorithms/localization/branch_and_bound_detail.hpp"
#include "sycl_points/algorithms/localization/branch_and_bound_params.hpp"
#include "sycl_points/algorithms/localization/branch_and_bound_result.hpp"
#include "sycl_points/algorithms/localization/voxel_occupancy_map.hpp"

namespace sycl_points::algorithms::localization {

/// @brief Find a global pose in a voxel occupancy map with a best-first search at multiple resolutions.
class BranchAndBoundLocalizer {
public:
    /// @brief Construct a localizer with the device queue and validate the parameters.
    BranchAndBoundLocalizer(const sycl_utils::DeviceQueue& queue, const BranchAndBoundParams& params)
        : queue_(queue), params_(params), score_matrices_(0, *queue.ptr), score_counts_(0, *queue.ptr) {
        const auto& p = this->params_;
        if (!std::isfinite(p.min_resolution) || p.min_resolution <= 0 || p.max_level < 0 || p.branch_factor < 2 ||
            !p.min_translation.allFinite() || !p.max_translation.allFinite() ||
            !(p.min_translation.array() <= p.max_translation.array()).all() || !p.min_rpy.allFinite() ||
            !p.max_rpy.allFinite() || !(p.min_rpy.array() <= p.max_rpy.array()).all() ||
            !std::isfinite(p.score_threshold_ratio) || p.score_threshold_ratio < 0 || p.score_threshold_ratio > 1 ||
            p.score_batch_size == 0 || (p.enable_timeout && p.timeout.count() < 0)) {
            throw std::invalid_argument("[BranchAndBoundLocalizer] invalid parameters");
        }
        this->resolutions_ = detail::level_resolutions(this->params_);
    }

    /// @brief Build the target score structure that the configured score method uses.
    ///        Use the map bounds if auto_translation_range is true.
    /// @throws std::invalid_argument if the map is empty or contains non-finite coordinates.
    void set_target(const PointCloudShared& map) {
        this->validate_cloud(map);
        Eigen::Vector3f low = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
        Eigen::Vector3f high = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
        for (const auto& point : *map.points) {
            low = low.cwiseMin(point.head<3>());
            high = high.cwiseMax(point.head<3>());
        }
        this->score_map_.reset();
        this->score_map_ = std::make_shared<VoxelOccupancyMap>(this->queue_, this->resolutions_);
        this->score_map_->set_target(map);
        this->target_size_ = map.size();
        if (this->params_.auto_translation_range) {
            this->min_translation_ = low;
            this->max_translation_ = high;
        } else {
            this->min_translation_ = this->params_.min_translation;
            this->max_translation_ = this->params_.max_translation;
        }
    }

    /// @brief Find the pose of a scan in the target map.
    /// @param scan Scan points expressed in the scan frame.
    /// @param scan_to_reference Transform from the scan frame to the reference frame.
    /// @return Best pose from the map frame to the reference frame.
    /// @note The reference frame is the frame whose pose is estimated. The default identity estimates the map frame
    ///       pose of the scan frame.
    /// @throws std::invalid_argument if the scan is empty, contains non-finite coordinates, or the transform is
    /// invalid.
    /// @throws std::runtime_error if no target was set.
    BranchAndBoundResult localize(const PointCloudShared& scan,
                                  const Eigen::Isometry3f& scan_to_reference = Eigen::Isometry3f::Identity()) {
        using Clock = std::chrono::steady_clock;
        const auto entry = Clock::now();
        this->validate_cloud(scan);
        if (!this->score_map_) {
            throw std::runtime_error("[BranchAndBoundLocalizer] target not set");
        }
        if (scan.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("[BranchAndBoundLocalizer] scan exceeds score range");
        }
        if (!scan_to_reference.matrix().allFinite()) {
            throw std::invalid_argument("[BranchAndBoundLocalizer] invalid scan transform");
        }

        const auto start = Clock::now();
        BranchAndBoundResult result;
        BranchAndBoundProfile& profile = result.profile;
        profile.levels.resize(static_cast<size_t>(this->params_.max_level) + 1);
        const auto elapsed_ms = [](Clock::time_point begin, Clock::time_point end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        const TransformMatrix reference_from_scan = scan_to_reference.matrix();
        float max_norm = 0.0f;
        for (const auto& point : *scan.points) {
            max_norm = std::max(max_norm, (reference_from_scan * point).head<3>().norm());
        }
        const auto angular = detail::angular_info(this->params_, this->resolutions_, max_norm);
        const float top_res = this->resolutions_.back();
        int low[3], high[3];
        for (int axis = 0; axis < 3; ++axis) {
            low[axis] = detail::checked_int(std::floor(static_cast<double>(this->min_translation_[axis]) / top_res));
            high[axis] = detail::checked_int(std::ceil(static_cast<double>(this->max_translation_[axis]) / top_res));
        }
        size_t initial_count = 1;
        auto multiply_checked = [&](size_t count) {
            if (count && initial_count > std::numeric_limits<size_t>::max() / count) {
                throw std::overflow_error("[BranchAndBoundLocalizer] candidate count overflow");
            }
            initial_count *= count;
        };
        for (int axis = 0; axis < 3; ++axis) {
            multiply_checked(static_cast<size_t>(static_cast<int64_t>(high[axis]) - low[axis] + 1));
            multiply_checked(static_cast<size_t>(angular.back().num_division[axis]));
        }

        const uint32_t threshold =
            static_cast<uint32_t>(std::floor(static_cast<double>(scan.size()) * this->params_.score_threshold_ratio));
        uint32_t best_score = threshold;
        detail::DiscreteTransformation best;
        bool have_leaf = false;
        std::priority_queue<detail::DiscreteTransformation> queue;
        std::vector<detail::DiscreteTransformation> stock;
        stock.reserve(std::min(initial_count, this->params_.score_batch_size));
        profile.initial_candidates = initial_count;
        profile.prepare_ms = elapsed_ms(entry, Clock::now());

        auto flush = [&]() {
            if (stock.empty()) {
                return;
            }
            const auto flush_begin = Clock::now();
            profile.flush_calls += 1;
            // Stock can contain candidates at different levels.
            // Group indices by level to score each candidate with the correct radius.
            std::vector<size_t> indices;
            for (int level = 0; level <= this->params_.max_level; ++level) {
                const auto partition_begin = Clock::now();
                indices.clear();
                for (size_t i = 0; i < stock.size(); ++i) {
                    if (stock[i].level == level) {
                        indices.push_back(i);
                    }
                }
                profile.flush_partition_ms += elapsed_ms(partition_begin, Clock::now());
                auto& level_profile = profile.levels[static_cast<size_t>(level)];
                for (size_t begin = 0; begin < indices.size(); begin += this->params_.score_batch_size) {
                    const size_t count = std::min(this->params_.score_batch_size, indices.size() - begin);
                    this->score_matrices_.resize(count);
                    this->score_counts_.resize(count);
                    const auto matrix_begin = Clock::now();
                    for (size_t i = 0; i < count; ++i) {
                        this->score_matrices_[i] =
                            stock[indices[begin + i]].create_matrix(this->resolutions_[level], angular[level].rpy_res,
                                                                    angular[level].min_rpy) *
                            reference_from_scan;
                    }
                    const auto matrix_end = Clock::now();
                    const double matrix_ms = elapsed_ms(matrix_begin, matrix_end);
                    profile.flush_matrix_build_ms += matrix_ms;
                    level_profile.matrix_build_ms += matrix_ms;
                    const auto score_begin = Clock::now();
                    this->score_map_
                        ->score_async(scan, this->score_matrices_, static_cast<size_t>(level), this->score_counts_)
                        .wait_and_throw();
                    const auto score_end = Clock::now();
                    const double score_ms = elapsed_ms(score_begin, score_end);
                    profile.flush_score_wait_ms += score_ms;
                    level_profile.score_ms += score_ms;
                    const auto queue_begin = Clock::now();
                    for (size_t i = 0; i < count; ++i) {
                        auto& candidate = stock[indices[begin + i]];
                        candidate.score = this->score_counts_[i];
                        if (candidate.score >= best_score) {
                            queue.push(candidate);
                            level_profile.queued += 1;
                            profile.queued_candidates += 1;
                        }
                    }
                    profile.flush_queue_ms += elapsed_ms(queue_begin, Clock::now());
                    profile.max_queue_size = std::max(profile.max_queue_size, queue.size());
                    level_profile.evaluated += count;
                    profile.score_batches += 1;
                    result.evaluated_candidates += count;
                }
            }
            stock.clear();
            profile.flush_total_ms += elapsed_ms(flush_begin, Clock::now());
        };

        // Score the initial grid in batches. Do not store all candidates at the same time.
        const auto initial_begin = Clock::now();
        const double flush_before_initial = profile.flush_total_ms;
        for (int64_t x = low[0]; x <= high[0]; ++x) {
            for (int64_t y = low[1]; y <= high[1]; ++y) {
                for (int64_t z = low[2]; z <= high[2]; ++z) {
                    for (int roll = 0; roll < angular.back().num_division.x(); ++roll) {
                        for (int pitch = 0; pitch < angular.back().num_division.y(); ++pitch) {
                            for (int yaw = 0; yaw < angular.back().num_division.z(); ++yaw) {
                                stock.push_back({0, this->params_.max_level, static_cast<int>(x), static_cast<int>(y),
                                                 static_cast<int>(z), roll, pitch, yaw});
                                if (stock.size() >= this->params_.score_batch_size) {
                                    flush();
                                }
                            }
                        }
                    }
                }
            }
        }
        flush();
        const auto initial_end = Clock::now();
        profile.initial_flush_ms = profile.flush_total_ms - flush_before_initial;
        profile.initial_enumerate_ms = elapsed_ms(initial_begin, initial_end) - profile.initial_flush_ms;

        // The deadline includes initial scores, even if no candidate remains in the queue.
        if (this->params_.enable_timeout && Clock::now() - start >= this->params_.timeout) {
            result.timed_out = true;
        }

        const auto search_begin = Clock::now();
        const double flush_before_search = profile.flush_total_ms;
        while (!result.timed_out && (!queue.empty() || !stock.empty())) {
            if (this->params_.enable_timeout && Clock::now() - start >= this->params_.timeout) {
                result.timed_out = true;
                break;
            }
            if (queue.empty()) {
                flush();
                continue;
            }
            auto candidate = queue.top();
            queue.pop();
            if (candidate.score < best_score) {
                profile.pruned_candidates += 1;
                profile.levels[static_cast<size_t>(candidate.level)].pruned += 1;
                continue;
            }
            if (candidate.is_leaf()) {
                best_score = candidate.score;
                best = candidate;
                have_leaf = true;
                profile.leaf_candidates += 1;
                profile.levels[0].leaves += 1;
            } else {
                const auto branch_begin = Clock::now();
                candidate.branch(stock, candidate.level - 1, this->params_.branch_factor,
                                 angular[candidate.level - 1].num_division);
                const auto branch_end = Clock::now();
                const double branch_ms = elapsed_ms(branch_begin, branch_end);
                profile.search_branch_ms += branch_ms;
                auto& level_profile = profile.levels[static_cast<size_t>(candidate.level)];
                level_profile.branch_ms += branch_ms;
                level_profile.branched += 1;
                profile.branch_calls += 1;
            }
            if (stock.size() >= this->params_.score_batch_size || (queue.empty() && !stock.empty())) {
                flush();
            }
        }
        const auto search_end = Clock::now();
        profile.search_flush_ms = profile.flush_total_ms - flush_before_search;
        profile.search_bookkeeping_ms =
            std::max(0.0, elapsed_ms(search_begin, search_end) - profile.search_flush_ms - profile.search_branch_ms);

        const auto finalize_begin = Clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (have_leaf) {
            result.inlier = best_score;
            result.score_ratio = static_cast<float>(best_score) / static_cast<float>(scan.size());
        }
        result.converged = have_leaf && best_score > threshold && !result.timed_out;
        if (result.converged) {
            result.T.matrix() =
                best.create_matrix(this->params_.min_resolution, angular[0].rpy_res, angular[0].min_rpy);
        }
        profile.finalize_ms = elapsed_ms(finalize_begin, Clock::now());
        profile.total_ms = elapsed_ms(entry, Clock::now());
        return result;
    }

    /// @brief Get the current parameters.
    const BranchAndBoundParams& params() const { return this->params_; }
    /// @brief Get the number of points in the current target.
    size_t target_size() const { return this->target_size_; }

private:
    static void validate_cloud(const PointCloudShared& cloud) {
        if (!cloud.points || cloud.size() == 0) {
            throw std::invalid_argument("[BranchAndBoundLocalizer] empty cloud");
        }
        for (const auto& point : *cloud.points) {
            if (!point.allFinite()) {
                throw std::invalid_argument("[BranchAndBoundLocalizer] non-finite point");
            }
        }
    }

    sycl_utils::DeviceQueue queue_;
    BranchAndBoundParams params_;
    std::vector<float> resolutions_;
    VoxelOccupancyMap::Ptr score_map_;
    shared_vector<TransformMatrix> score_matrices_;
    shared_vector<uint32_t> score_counts_;
    size_t target_size_ = 0;
    Eigen::Vector3f min_translation_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f max_translation_ = Eigen::Vector3f::Zero();
};

}  // namespace sycl_points::algorithms::localization
