#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "sycl_points/algorithms/common/voxel_constants.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace mapping {

/// @brief Host-side occupancy grid map implementation using PointCloudShared I/O.
/// @note This class is intentionally independent from OccupancyGridMap and does not submit SYCL kernels.
class OccupancyGridMapCPU {
public:
    /// @brief Construct the occupancy grid map.
    /// @param queue Device queue used by PointCloudShared allocations.
    /// @param voxel_size Edge length of a voxel in meters.
    OccupancyGridMapCPU(const sycl_utils::DeviceQueue& queue, const float voxel_size) : queue_(queue) {
        this->set_voxel_size(voxel_size);
        this->clear();
    }

    void clear() {
        this->capacity_ = kCapacityCandidates[0];
        this->voxel_num_ = 0;
        this->has_rgb_data_ = false;
        this->has_intensity_data_ = false;
        this->frame_index_ = 0;

        this->keys_.assign(this->capacity_, VoxelConstants::invalid_coord);
        this->core_data_.assign(this->capacity_, VoxelCoreData{});
        this->color_data_.assign(this->capacity_, VoxelColorData{});
        this->intensity_data_.assign(this->capacity_, VoxelIntensityData{});
    }

    void set_voxel_size(const float voxel_size) {
        if (!(voxel_size > 0.0f)) {
            throw std::invalid_argument("voxel_size must be positive.");
        }
        this->voxel_size_ = voxel_size;
        this->inv_voxel_size_ = 1.0f / voxel_size;
    }

    float voxel_size() const { return this->voxel_size_; }

    float voxel_probability(const Eigen::Vector3f& position) const {
        const uint64_t key = this->compute_key(position);
        const VoxelCoreData* core = this->find_voxel(key);
        if (!core) {
            return 0.5f;
        }
        return this->log_odds_to_probability(core->log_odds);
    }

    void set_log_odds_hit(const float value) { this->log_odds_hit_ = value; }
    void set_log_odds_miss(const float value) { this->log_odds_miss_ = value; }
    void set_free_space_updates_enabled(const bool enabled) { this->free_space_updates_enabled_ = enabled; }
    void set_voxel_pruning_enabled(const bool enabled) { this->voxel_pruning_enabled_ = enabled; }

    void set_log_odds_limits(const float minimum, const float maximum) {
        if (minimum > maximum) {
            throw std::invalid_argument("minimum must not exceed maximum.");
        }
        this->min_log_odds_ = minimum;
        this->max_log_odds_ = maximum;
    }

    void set_occupancy_threshold(const float probability) {
        if (!(probability > 0.0f) || !(probability < 1.0f)) {
            throw std::invalid_argument("probability must be between 0 and 1.");
        }
        this->occupancy_threshold_log_odds_ = this->probability_to_log_odds(probability);
    }

    void set_stale_frame_threshold(const uint32_t threshold) { this->stale_frame_threshold_ = threshold; }

    void set_intensity_ema_alpha(const float alpha) {
        if (alpha < 0.0f || alpha > 1.0f) {
            throw std::invalid_argument("alpha must be within [0, 1].");
        }
        this->intensity_ema_alpha_ = alpha;
    }

    float intensity_ema_alpha() const { return this->intensity_ema_alpha_; }

    void set_intensity_integration_enabled(const bool enabled) {
        this->intensity_integration_enabled_ = enabled;
        if (!enabled) {
            this->has_intensity_data_ = false;
        }
    }

    bool intensity_integration_enabled() const { return this->intensity_integration_enabled_; }

    void add_point_cloud(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) {
        if (!cloud.points || cloud.points->empty()) {
            return;
        }

        this->ensure_rehash();

        const bool has_rgb = cloud.has_rgb();
        const bool has_intensity = this->intensity_integration_enabled_ && cloud.has_intensity();
        this->has_rgb_data_ = this->has_rgb_data_ || has_rgb;
        this->has_intensity_data_ = this->has_intensity_data_ || has_intensity;

        const Eigen::Vector3f sensor_origin = sensor_pose.translation();
        for (size_t i = 0; i < cloud.size(); ++i) {
            const Eigen::Vector3f world_point = sensor_pose * cloud.points->at(i).head<3>();
            const uint64_t voxel_key = this->compute_key(world_point);
            if (voxel_key == VoxelConstants::invalid_coord) {
                continue;
            }

            size_t slot = this->capacity_;
            this->ensure_slot_capacity(voxel_key, slot);

            auto& core = this->core_data_.at(slot);
            core.sum_x += world_point.x();
            core.sum_y += world_point.y();
            core.sum_z += world_point.z();
            core.pending_log_odds += this->log_odds_hit_;
            core.hit_count += 1U;
            core.last_updated = this->frame_index_;

            if (has_rgb) {
                const RGBType& color = cloud.rgb->at(i);
                auto& voxel_color = this->color_data_.at(slot);
                voxel_color.sum_r += color.x();
                voxel_color.sum_g += color.y();
                voxel_color.sum_b += color.z();
                voxel_color.sum_a += color.w();
            }

            if (has_intensity) {
                auto& voxel_intensity = this->intensity_data_.at(slot);
                voxel_intensity.pending_sum_intensity += cloud.intensities->at(i);
                voxel_intensity.pending_count += 1U;
            }

            if (!this->free_space_updates_enabled_ || this->log_odds_miss_ == 0.0f) {
                continue;
            }

            this->traverse_ray(sensor_origin, world_point, [&](const int64_t ix, const int64_t iy, const int64_t iz) {
                uint64_t free_key = VoxelConstants::invalid_coord;
                if (!this->grid_to_key(ix, iy, iz, free_key)) {
                    return true;
                }

                size_t free_slot = this->capacity_;
                this->ensure_slot_capacity(free_key, free_slot);
                auto& free_core = this->core_data_.at(free_slot);
                free_core.pending_log_odds += this->log_odds_miss_;
                free_core.last_updated = this->frame_index_;
                return true;
            });
        }

        this->apply_pending_log_odds();
        if (this->voxel_pruning_enabled_) {
            this->prune_stale_voxels();
        }

        ++this->frame_index_;
    }

    void extract_occupied_points(PointCloudShared& result, const Eigen::Isometry3f& sensor_pose,
                                 const float max_distance = 100.0f) const {
        result.resize_points(0);
        result.resize_rgb(0);
        result.resize_intensities(0);

        if (this->voxel_num_ == 0) {
            return;
        }

        const Eigen::Vector3f sensor_position = sensor_pose.translation();
        result.resize_points(this->voxel_num_);
        if (this->has_rgb_data_) {
            result.resize_rgb(this->voxel_num_);
        }
        if (this->has_intensity_data_ && this->intensity_integration_enabled_) {
            result.resize_intensities(this->voxel_num_);
        }

        size_t out_idx = 0;
        for (size_t i = 0; i < this->capacity_; ++i) {
            const uint64_t key = this->keys_.at(i);
            if (key == VoxelConstants::invalid_coord || key == VoxelConstants::deleted_coord) {
                continue;
            }

            const auto& core = this->core_data_.at(i);
            if (core.hit_count == 0U || core.log_odds < this->occupancy_threshold_log_odds_) {
                continue;
            }

            const float inv_count = 1.0f / static_cast<float>(core.hit_count);
            const float cx = core.sum_x * inv_count;
            const float cy = core.sum_y * inv_count;
            const float cz = core.sum_z * inv_count;
            const float dist_inf = std::max({std::fabs(cx - sensor_position.x()), std::fabs(cy - sensor_position.y()),
                                             std::fabs(cz - sensor_position.z())});
            if (dist_inf > max_distance) {
                continue;
            }

            result.points->at(out_idx) = PointType(cx, cy, cz, 1.0f);
            if (this->has_rgb_data_) {
                const auto& color = this->color_data_.at(i);
                result.rgb->at(out_idx) =
                    RGBType(color.sum_r * inv_count, color.sum_g * inv_count, color.sum_b * inv_count, color.sum_a * inv_count);
            }
            if (this->has_intensity_data_ && this->intensity_integration_enabled_) {
                const auto& intensity = this->intensity_data_.at(i);
                result.intensities->at(out_idx) = intensity.update_count > 0U ? intensity.ema_intensity : 0.0f;
            }
            ++out_idx;
        }

        result.resize_points(out_idx);
        if (this->has_rgb_data_) {
            result.resize_rgb(out_idx);
        }
        if (this->has_intensity_data_ && this->intensity_integration_enabled_) {
            result.resize_intensities(out_idx);
        }
    }

    void extract_visible_points(PointCloudShared& result, const Eigen::Isometry3f& sensor_pose, float max_distance,
                                float horizontal_fov, float vertical_fov) const {
        result.resize_points(0);
        result.resize_rgb(0);
        result.resize_intensities(0);

        if (this->voxel_num_ == 0) {
            return;
        }

        horizontal_fov = std::clamp(horizontal_fov, kFovTolerance, kPi - kFovTolerance);
        vertical_fov = std::clamp(vertical_fov, kFovTolerance, 2.0f * kPi - kFovTolerance);

        result.resize_points(this->voxel_num_);
        if (this->has_rgb_data_) {
            result.resize_rgb(this->voxel_num_);
        }
        if (this->has_intensity_data_ && this->intensity_integration_enabled_) {
            result.resize_intensities(this->voxel_num_);
        }

        const Eigen::Matrix3f world_to_sensor_R = sensor_pose.inverse().linear();
        const Eigen::Vector3f sensor_t = sensor_pose.translation();
        const float max_dist_sq = max_distance * max_distance;
        const float cos_limit_horizontal = std::cos(horizontal_fov * 0.5f);
        const float cos_limit_vertical = std::cos(vertical_fov * 0.5f);
        const bool include_backward = horizontal_fov >= (kPi - kFovTolerance);

        size_t out_idx = 0;
        for (size_t i = 0; i < this->capacity_; ++i) {
            const uint64_t key = this->keys_.at(i);
            if (key == VoxelConstants::invalid_coord || key == VoxelConstants::deleted_coord) {
                continue;
            }

            const auto& core = this->core_data_.at(i);
            if (core.hit_count == 0U || core.log_odds < this->occupancy_threshold_log_odds_) {
                continue;
            }

            const float inv_count = 1.0f / static_cast<float>(core.hit_count);
            const Eigen::Vector3f center(core.sum_x * inv_count, core.sum_y * inv_count, core.sum_z * inv_count);
            const Eigen::Vector3f delta = center - sensor_t;
            if (delta.squaredNorm() > max_dist_sq) {
                continue;
            }

            const Eigen::Vector3f local_pt = world_to_sensor_R * delta;
            if (!include_backward && local_pt.x() <= 0.0f) {
                continue;
            }

            const float forward_projection = include_backward ? std::fabs(local_pt.x()) : local_pt.x();
            const float horizontal_norm_sq = forward_projection * forward_projection + local_pt.y() * local_pt.y();
            float cos_horizontal = 1.0f;
            if (horizontal_norm_sq > 0.0f) {
                cos_horizontal = std::clamp(forward_projection / std::sqrt(horizontal_norm_sq), -1.0f, 1.0f);
            }
            if (cos_horizontal < cos_limit_horizontal) {
                continue;
            }

            const float vertical_norm_sq = forward_projection * forward_projection + local_pt.z() * local_pt.z();
            float cos_vertical = 1.0f;
            if (vertical_norm_sq > 0.0f) {
                cos_vertical = std::clamp(forward_projection / std::sqrt(vertical_norm_sq), -1.0f, 1.0f);
            }
            if (cos_vertical < cos_limit_vertical) {
                continue;
            }

            result.points->at(out_idx) = PointType(center.x(), center.y(), center.z(), 1.0f);
            if (this->has_rgb_data_) {
                const auto& color = this->color_data_.at(i);
                result.rgb->at(out_idx) =
                    RGBType(color.sum_r * inv_count, color.sum_g * inv_count, color.sum_b * inv_count, color.sum_a * inv_count);
            }
            if (this->has_intensity_data_ && this->intensity_integration_enabled_) {
                const auto& intensity = this->intensity_data_.at(i);
                result.intensities->at(out_idx) = intensity.update_count > 0U ? intensity.ema_intensity : 0.0f;
            }
            ++out_idx;
        }

        result.resize_points(out_idx);
        if (this->has_rgb_data_) {
            result.resize_rgb(out_idx);
        }
        if (this->has_intensity_data_ && this->intensity_integration_enabled_) {
            result.resize_intensities(out_idx);
        }
    }

    float compute_overlap_ratio(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) const {
        if (!cloud.points || cloud.points->empty() || this->voxel_num_ == 0) {
            return 0.0f;
        }

        uint32_t overlap_count = 0U;
        for (size_t i = 0; i < cloud.size(); ++i) {
            const Eigen::Vector3f world_point = sensor_pose * cloud.points->at(i).head<3>();
            const uint64_t key = this->compute_key(world_point);
            if (key == VoxelConstants::invalid_coord) {
                continue;
            }

            const VoxelCoreData* core = this->find_voxel(key);
            if (!core) {
                continue;
            }

            if (core->hit_count > 0U && core->log_odds >= this->occupancy_threshold_log_odds_) {
                ++overlap_count;
            }
        }

        return static_cast<float>(overlap_count) / static_cast<float>(cloud.size());
    }

private:
    inline static constexpr float kPi = 3.1415927f;
    inline static constexpr float kFovTolerance = 1e-6f;

    struct VoxelCoreData {
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        float log_odds = 0.0f;
        float pending_log_odds = 0.0f;
        uint32_t hit_count = 0U;
        uint32_t last_updated = 0U;
        uint32_t padding = 0U;
    };

    struct VoxelColorData {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_a = 0.0f;
    };

    struct VoxelIntensityData {
        float ema_intensity = 0.0f;
        float pending_sum_intensity = 0.0f;
        uint32_t pending_count = 0U;
        uint32_t update_count = 0U;
    };

    static float probability_to_log_odds(const float probability) {
        return std::log(probability / (1.0f - probability));
    }

    static float log_odds_to_probability(const float log_odds) { return 1.0f / (1.0f + std::exp(-log_odds)); }

    float clamp_log_odds(const float value) const {
        return std::min(std::max(value, this->min_log_odds_), this->max_log_odds_);
    }

    uint64_t compute_key(const Eigen::Vector3f& point) const {
        const float scaled_x = point.x() * this->inv_voxel_size_;
        const float scaled_y = point.y() * this->inv_voxel_size_;
        const float scaled_z = point.z() * this->inv_voxel_size_;

        const int64_t coord_x = static_cast<int64_t>(std::floor(scaled_x)) + VoxelConstants::coord_offset;
        const int64_t coord_y = static_cast<int64_t>(std::floor(scaled_y)) + VoxelConstants::coord_offset;
        const int64_t coord_z = static_cast<int64_t>(std::floor(scaled_z)) + VoxelConstants::coord_offset;

        if (coord_x < 0 || coord_x > VoxelConstants::coord_bit_mask || coord_y < 0 ||
            coord_y > VoxelConstants::coord_bit_mask || coord_z < 0 || coord_z > VoxelConstants::coord_bit_mask) {
            return VoxelConstants::invalid_coord;
        }

        return (static_cast<uint64_t>(coord_x & VoxelConstants::coord_bit_mask)
                << (VoxelConstants::coord_bit_size * CartesianCoordComponent::X)) |
               (static_cast<uint64_t>(coord_y & VoxelConstants::coord_bit_mask)
                << (VoxelConstants::coord_bit_size * CartesianCoordComponent::Y)) |
               (static_cast<uint64_t>(coord_z & VoxelConstants::coord_bit_mask)
                << (VoxelConstants::coord_bit_size * CartesianCoordComponent::Z));
    }

    const VoxelCoreData* find_voxel(const uint64_t key) const {
        if (key == VoxelConstants::invalid_coord) {
            return nullptr;
        }
        for (size_t probe = 0; probe < this->max_probe_length_; ++probe) {
            const size_t slot = this->compute_slot_id(key, probe, this->capacity_);
            const uint64_t stored_key = this->keys_.at(slot);
            if (stored_key == key) {
                return &this->core_data_.at(slot);
            }
            if (stored_key == VoxelConstants::invalid_coord) {
                return nullptr;
            }
        }
        return nullptr;
    }

    static uint64_t hash2(const uint64_t voxel_hash, const size_t capacity) {
        return (capacity - 2) - (voxel_hash % (capacity - 2));
    }

    static size_t compute_slot_id(const uint64_t voxel_hash, const size_t probe, const size_t capacity) {
        return (voxel_hash + probe * hash2(voxel_hash, capacity)) % capacity;
    }

    void ensure_rehash() {
        if (this->rehash_threshold_ < static_cast<float>(this->voxel_num_) / static_cast<float>(this->capacity_)) {
            const size_t next_capacity = this->get_next_capacity_value();
            if (next_capacity > this->capacity_) {
                this->rehash(next_capacity);
            }
        }
    }

    size_t get_next_capacity_value() const {
        for (const auto candidate : this->kCapacityCandidates) {
            if (candidate > this->capacity_) {
                return candidate;
            }
        }
        return this->capacity_;
    }

    void rehash(const size_t new_capacity) {
        const auto old_keys = this->keys_;
        const auto old_core = this->core_data_;
        const auto old_color = this->color_data_;
        const auto old_intensity = this->intensity_data_;

        this->capacity_ = new_capacity;
        this->keys_.assign(this->capacity_, VoxelConstants::invalid_coord);
        this->core_data_.assign(this->capacity_, VoxelCoreData{});
        this->color_data_.assign(this->capacity_, VoxelColorData{});
        this->intensity_data_.assign(this->capacity_, VoxelIntensityData{});

        size_t inserted_count = 0;
        for (size_t i = 0; i < old_keys.size(); ++i) {
            const uint64_t key = old_keys.at(i);
            if (key == VoxelConstants::invalid_coord || key == VoxelConstants::deleted_coord) {
                continue;
            }

            for (size_t probe = 0; probe < this->max_probe_length_; ++probe) {
                const size_t slot = this->compute_slot_id(key, probe, this->capacity_);
                if (this->keys_.at(slot) == VoxelConstants::invalid_coord) {
                    this->keys_.at(slot) = key;
                    this->core_data_.at(slot) = old_core.at(i);
                    this->color_data_.at(slot) = old_color.at(i);
                    this->intensity_data_.at(slot) = old_intensity.at(i);
                    ++inserted_count;
                    break;
                }
            }
        }
        this->voxel_num_ = inserted_count;
    }

    bool find_or_insert_slot(const uint64_t key, size_t& slot) {
        for (size_t probe = 0; probe < this->max_probe_length_; ++probe) {
            const size_t candidate = this->compute_slot_id(key, probe, this->capacity_);
            const uint64_t stored_key = this->keys_.at(candidate);
            if (stored_key == key) {
                slot = candidate;
                return true;
            }
            if (stored_key == VoxelConstants::invalid_coord || stored_key == VoxelConstants::deleted_coord) {
                this->keys_.at(candidate) = key;
                this->core_data_.at(candidate) = VoxelCoreData{};
                this->color_data_.at(candidate) = VoxelColorData{};
                this->intensity_data_.at(candidate) = VoxelIntensityData{};
                ++this->voxel_num_;
                slot = candidate;
                return true;
            }
        }
        return false;
    }

    void ensure_slot_capacity(const uint64_t key, size_t& slot) {
        while (!this->find_or_insert_slot(key, slot)) {
            const size_t next_capacity = this->get_next_capacity_value();
            if (next_capacity <= this->capacity_) {
                throw std::runtime_error("Failed to insert voxel: map capacity exhausted.");
            }
            this->rehash(next_capacity);
        }
    }

    void apply_pending_log_odds() {
        for (size_t i = 0; i < this->capacity_; ++i) {
            const uint64_t key = this->keys_.at(i);
            if (key == VoxelConstants::invalid_coord || key == VoxelConstants::deleted_coord) {
                continue;
            }

            auto& core = this->core_data_.at(i);
            if (core.pending_log_odds != 0.0f) {
                core.log_odds = this->clamp_log_odds(core.log_odds + core.pending_log_odds);
                core.pending_log_odds = 0.0f;
            }

            if (!(this->has_intensity_data_ && this->intensity_integration_enabled_)) {
                continue;
            }

            auto& intensity = this->intensity_data_.at(i);
            if (intensity.pending_count == 0U) {
                continue;
            }

            const float frame_mean = intensity.pending_sum_intensity / static_cast<float>(intensity.pending_count);
            if (intensity.update_count == 0U) {
                intensity.ema_intensity = frame_mean;
            } else {
                intensity.ema_intensity = (1.0f - this->intensity_ema_alpha_) * intensity.ema_intensity +
                                          this->intensity_ema_alpha_ * frame_mean;
            }
            intensity.update_count += 1U;
            intensity.pending_sum_intensity = 0.0f;
            intensity.pending_count = 0U;
        }
    }

    void prune_stale_voxels() {
        size_t active_count = 0;
        for (size_t i = 0; i < this->capacity_; ++i) {
            const uint64_t key = this->keys_.at(i);
            if (key == VoxelConstants::invalid_coord || key == VoxelConstants::deleted_coord) {
                continue;
            }

            const uint32_t age = this->frame_index_ - this->core_data_.at(i).last_updated;
            if (age > this->stale_frame_threshold_) {
                this->keys_.at(i) = VoxelConstants::deleted_coord;
                this->core_data_.at(i) = VoxelCoreData{};
                this->color_data_.at(i) = VoxelColorData{};
                this->intensity_data_.at(i) = VoxelIntensityData{};
                continue;
            }
            ++active_count;
        }
        this->voxel_num_ = active_count;
    }

    template <typename Visitor>
    static inline void traverse_ray_exclusive_impl(const float origin_x, const float origin_y, const float origin_z,
                                                   const float target_x, const float target_y, const float target_z,
                                                   const float inv_voxel, Visitor&& visitor) {
        const float scaled_origin_x = origin_x * inv_voxel;
        const float scaled_origin_y = origin_y * inv_voxel;
        const float scaled_origin_z = origin_z * inv_voxel;
        const float scaled_target_x = target_x * inv_voxel;
        const float scaled_target_y = target_y * inv_voxel;
        const float scaled_target_z = target_z * inv_voxel;

        int64_t ix = static_cast<int64_t>(std::floor(scaled_origin_x));
        int64_t iy = static_cast<int64_t>(std::floor(scaled_origin_y));
        int64_t iz = static_cast<int64_t>(std::floor(scaled_origin_z));

        const int64_t target_ix = static_cast<int64_t>(std::floor(scaled_target_x));
        const int64_t target_iy = static_cast<int64_t>(std::floor(scaled_target_y));
        const int64_t target_iz = static_cast<int64_t>(std::floor(scaled_target_z));

        if (ix == target_ix && iy == target_iy && iz == target_iz) {
            return;
        }

        const float dir_x = scaled_target_x - scaled_origin_x;
        const float dir_y = scaled_target_y - scaled_origin_y;
        const float dir_z = scaled_target_z - scaled_origin_z;

        const float abs_dir_x = std::fabs(dir_x);
        const float abs_dir_y = std::fabs(dir_y);
        const float abs_dir_z = std::fabs(dir_z);

        const int step_x = (dir_x > 0.0f) ? 1 : ((dir_x < 0.0f) ? -1 : 0);
        const int step_y = (dir_y > 0.0f) ? 1 : ((dir_y < 0.0f) ? -1 : 0);
        const int step_z = (dir_z > 0.0f) ? 1 : ((dir_z < 0.0f) ? -1 : 0);

        const float start_frac_x = scaled_origin_x - std::floor(scaled_origin_x);
        const float start_frac_y = scaled_origin_y - std::floor(scaled_origin_y);
        const float start_frac_z = scaled_origin_z - std::floor(scaled_origin_z);

        const float inv_dir_mag_x = (abs_dir_x > std::numeric_limits<float>::epsilon())
                                        ? (1.0f / abs_dir_x)
                                        : std::numeric_limits<float>::infinity();
        const float inv_dir_mag_y = (abs_dir_y > std::numeric_limits<float>::epsilon())
                                        ? (1.0f / abs_dir_y)
                                        : std::numeric_limits<float>::infinity();
        const float inv_dir_mag_z = (abs_dir_z > std::numeric_limits<float>::epsilon())
                                        ? (1.0f / abs_dir_z)
                                        : std::numeric_limits<float>::infinity();

        const float inf = std::numeric_limits<float>::infinity();
        float t_max_x = (step_x != 0) ? ((step_x > 0 ? (1.0f - start_frac_x) : start_frac_x) * inv_dir_mag_x) : inf;
        float t_max_y = (step_y != 0) ? ((step_y > 0 ? (1.0f - start_frac_y) : start_frac_y) * inv_dir_mag_y) : inf;
        float t_max_z = (step_z != 0) ? ((step_z > 0 ? (1.0f - start_frac_z) : start_frac_z) * inv_dir_mag_z) : inf;

        const float t_delta_x = (step_x != 0) ? inv_dir_mag_x : inf;
        const float t_delta_y = (step_y != 0) ? inv_dir_mag_y : inf;
        const float t_delta_z = (step_z != 0) ? inv_dir_mag_z : inf;

        while (true) {
            if (t_max_x <= t_max_y && t_max_x <= t_max_z) {
                ix += step_x;
                t_max_x += t_delta_x;
            } else if (t_max_y <= t_max_z) {
                iy += step_y;
                t_max_y += t_delta_y;
            } else {
                iz += step_z;
                t_max_z += t_delta_z;
            }

            if (ix == target_ix && iy == target_iy && iz == target_iz) {
                break;
            }

            if (!visitor(ix, iy, iz)) {
                break;
            }
        }
    }

    static inline bool grid_to_key_device(const int64_t x, const int64_t y, const int64_t z, uint64_t& key) {
        const int64_t shift_x = x + VoxelConstants::coord_offset;
        const int64_t shift_y = y + VoxelConstants::coord_offset;
        const int64_t shift_z = z + VoxelConstants::coord_offset;

        if (shift_x < 0 || shift_x > VoxelConstants::coord_bit_mask || shift_y < 0 ||
            shift_y > VoxelConstants::coord_bit_mask || shift_z < 0 || shift_z > VoxelConstants::coord_bit_mask) {
            return false;
        }

        key = (static_cast<uint64_t>(shift_x & VoxelConstants::coord_bit_mask)
               << (VoxelConstants::coord_bit_size * CartesianCoordComponent::X)) |
              (static_cast<uint64_t>(shift_y & VoxelConstants::coord_bit_mask)
               << (VoxelConstants::coord_bit_size * CartesianCoordComponent::Y)) |
              (static_cast<uint64_t>(shift_z & VoxelConstants::coord_bit_mask)
               << (VoxelConstants::coord_bit_size * CartesianCoordComponent::Z));
        return true;
    }

    template <typename Visitor>
    void traverse_ray(const Eigen::Vector3f& origin, const Eigen::Vector3f& target, Visitor&& visitor) const {
        const float inv_voxel = this->inv_voxel_size_;

        int64_t ix = static_cast<int64_t>(std::floor(origin.x() * inv_voxel));
        int64_t iy = static_cast<int64_t>(std::floor(origin.y() * inv_voxel));
        int64_t iz = static_cast<int64_t>(std::floor(origin.z() * inv_voxel));

        const int64_t target_ix = static_cast<int64_t>(std::floor(target.x() * inv_voxel));
        const int64_t target_iy = static_cast<int64_t>(std::floor(target.y() * inv_voxel));
        const int64_t target_iz = static_cast<int64_t>(std::floor(target.z() * inv_voxel));

        if (ix == target_ix && iy == target_iy && iz == target_iz) {
            return;
        }

        if (!visitor(ix, iy, iz)) {
            return;
        }

        traverse_ray_exclusive_impl(origin.x(), origin.y(), origin.z(), target.x(), target.y(), target.z(), inv_voxel,
                                    std::forward<Visitor>(visitor));
    }

    bool grid_to_key(const int64_t x, const int64_t y, const int64_t z, uint64_t& key) const {
        return grid_to_key_device(x, y, z, key);
    }

    sycl_utils::DeviceQueue queue_;
    float voxel_size_ = 0.1f;
    float inv_voxel_size_ = 10.0f;
    float log_odds_hit_ = 0.85f;
    float log_odds_miss_ = -0.4f;
    float min_log_odds_ = -4.0f;
    float max_log_odds_ = 4.0f;
    float occupancy_threshold_log_odds_ = probability_to_log_odds(0.5f);
    bool free_space_updates_enabled_ = true;
    bool voxel_pruning_enabled_ = true;
    float intensity_ema_alpha_ = 0.2f;
    bool intensity_integration_enabled_ = true;

    bool has_rgb_data_ = false;
    bool has_intensity_data_ = false;
    uint32_t frame_index_ = 0U;
    uint32_t stale_frame_threshold_ = 100U;

    inline static constexpr std::array<size_t, 11> kCapacityCandidates = {
        30029, 60013, 120011, 240007, 480013, 960017, 1920001, 3840007, 7680017, 15360013, 30720007};
    size_t capacity_ = kCapacityCandidates[0];
    size_t voxel_num_ = 0;

    const size_t max_probe_length_ = 128;
    const float rehash_threshold_ = 0.7f;

    std::vector<uint64_t> keys_;
    std::vector<VoxelCoreData> core_data_;
    std::vector<VoxelColorData> color_data_;
    std::vector<VoxelIntensityData> intensity_data_;
};

}  // namespace mapping
}  // namespace algorithms
}  // namespace sycl_points
