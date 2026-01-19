#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "sycl_points/algorithms/common/prefix_sum.hpp"
#include "sycl_points/algorithms/common/transform.hpp"
#include "sycl_points/algorithms/common/voxel_constants.hpp"
#include "sycl_points/algorithms/common/workgroup_utils.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace mapping {

/// @brief Occupancy grid map that performs voxel integration on the device.
/// @note The class mirrors the hashing infrastructure used by VoxelHashMap so that
///       log-odds accumulation and visibility updates run in parallel on the device.
class OccupancyGridMap {
public:
    using Ptr = std::shared_ptr<OccupancyGridMap>;

    /// @brief Construct the occupancy grid map.
    /// @param queue Device queue used for all kernels.
    /// @param voxel_size Edge length of a voxel in meters.
    OccupancyGridMap(const sycl_utils::DeviceQueue& queue, const float voxel_size) : queue_(queue) {
        this->set_voxel_size(voxel_size);
        this->allocate_storage(this->capacity_);

        this->prefix_sum_ = std::make_shared<common::PrefixSum>(this->queue_);
        this->valid_flags_ptr_ = std::make_shared<shared_vector<uint8_t>>(*this->queue_.ptr);

        this->clear();
    }

    /// @brief Reset the map data.
    void clear() {
        this->capacity_ = kCapacityCandidates[0];
        this->voxel_num_ = 0;
        this->has_rgb_data_ = false;
        this->has_intensity_data_ = false;
        this->frame_index_ = 0;

        this->key_ptr_->resize(this->capacity_);
        this->core_data_ptr_->resize(this->capacity_);
        this->color_data_ptr_->resize(this->capacity_);
        this->intensity_data_ptr_->resize(this->capacity_);

        // Reset the hash table content before the next integration round.
        sycl_utils::events evs;
        evs += this->queue_.ptr->fill<uint64_t>(this->key_ptr_->data(), VoxelConstants::invalid_coord,
                                                this->key_ptr_->size());
        evs += this->queue_.ptr->fill<VoxelCoreData>(this->core_data_ptr_->data(), VoxelCoreData{},
                                                     this->core_data_ptr_->size());
        evs += this->queue_.ptr->fill<VoxelColorData>(this->color_data_ptr_->data(), VoxelColorData{},
                                                      this->color_data_ptr_->size());
        evs += this->queue_.ptr->fill<VoxelIntensityData>(this->intensity_data_ptr_->data(), VoxelIntensityData{},
                                                          this->intensity_data_ptr_->size());
        evs.wait_and_throw();
    }

    /// @brief Set the voxel size.
    /// @param voxel_size Edge length in meters.
    void set_voxel_size(const float voxel_size) {
        if (!(voxel_size > 0.0f)) {
            throw std::invalid_argument("voxel_size must be positive.");
        }
        this->voxel_size_ = voxel_size;
        this->inv_voxel_size_ = 1.0f / voxel_size;
    }

    /// @brief Get the voxel size.
    float voxel_size() const { return this->voxel_size_; }

    /// @brief Query the occupancy probability at the specified position.
    float voxel_probability(const Eigen::Vector3f& position) const {
        const uint64_t key = this->compute_key(position);
        const VoxelCoreData* core = this->find_voxel(key);
        if (!core) {
            return 0.5f;
        }
        return this->log_odds_to_probability(core->log_odds);
    }

    /// @brief Configure the log-odds increment applied on hits.
    void set_log_odds_hit(const float value) { this->log_odds_hit_ = value; }

    /// @brief Configure the log-odds decrement applied on misses.
    void set_log_odds_miss(const float value) { this->log_odds_miss_ = value; }

    /// @brief Enable or disable free-space carving along measurement rays.
    void set_free_space_updates_enabled(const bool enabled) { this->free_space_updates_enabled_ = enabled; }

    /// @brief Enable or disable pruning of stale voxels.
    void set_voxel_pruning_enabled(const bool enabled) { this->voxel_pruning_enabled_ = enabled; }

    /// @brief Configure the minimum and maximum allowed log-odds.
    void set_log_odds_limits(const float minimum, const float maximum) {
        if (minimum > maximum) {
            throw std::invalid_argument("minimum must not exceed maximum.");
        }
        this->min_log_odds_ = minimum;
        this->max_log_odds_ = maximum;
    }

    /// @brief Configure the occupancy probability threshold.
    void set_occupancy_threshold(const float probability) {
        if (!(probability > 0.0f) || !(probability < 1.0f)) {
            throw std::invalid_argument("probability must be between 0 and 1.");
        }
        this->occupancy_threshold_log_odds_ = this->probability_to_log_odds(probability);
    }

    /// @brief Set the frame-age threshold used to prune stale voxels.
    void set_stale_frame_threshold(const uint32_t threshold) { this->stale_frame_threshold_ = threshold; }

    /// @brief Insert a point cloud captured at the given pose.
    /// @param cloud Point cloud in the sensor frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    void add_point_cloud(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) {
        if (!cloud.points || cloud.points->empty()) {
            return;
        }

        const size_t N = cloud.size();

        // Prepare hashing buffers for the expected number of insertions in this frame.
        this->ensure_rehash();

        const bool has_rgb = cloud.has_rgb();
        const bool has_intensity = cloud.has_intensity();
        this->has_rgb_data_ = this->has_rgb_data_ || has_rgb;
        this->has_intensity_data_ = this->has_intensity_data_ || has_intensity;

        // Integrate hits: transform to world frame, hash, and accumulate statistics.
        this->integrate_points(cloud, sensor_pose, has_rgb, has_intensity);

        if (this->free_space_updates_enabled_ && this->log_odds_miss_ != 0.0f) {
            // Traverse rays and record free-space updates before applying log-odds.
            this->update_free_space(cloud, sensor_pose);
        }

        // Apply pending log-odds changes from hits and misses to the main storage.
        this->apply_pending_log_odds();

        if (this->voxel_pruning_enabled_) {
            // Remove voxels that have not been updated recently to keep the map fresh.
            this->prune_stale_voxels();
        }

        ++this->frame_index_;
    }

    /// @brief Extract occupied voxels from the sensor pose to L∞ distance.
    /// @param result Output point cloud in the map frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    /// @param max_distance Maximum extract L-infinity distance in meters.
    void extract_occupied_points(PointCloudShared& result, const Eigen::Isometry3f& sensor_pose,
                                 const float max_distance = 100.0f) {
        result.resize_points(0);
        result.resize_rgb(0);
        result.resize_intensities(0);

        if (this->voxel_num_ == 0) {
            return;
        }

        this->extract_occupied_points_impl(result, sensor_pose.translation(), max_distance);
    }

    /// @brief [Experimental] Extract the visible subset of the map as a new point cloud.
    /// @param result Output point cloud in the map frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    /// @param max_distance Maximum visibility distance in meters.
    /// @param horizontal_fov Horizontal field of view in radians.
    /// @param vertical_fov Vertical field of view in radians.
    void extract_visible_points(PointCloudShared& result, const Eigen::Isometry3f& sensor_pose, float max_distance,
                                float horizontal_fov, float vertical_fov) {
        result.resize_points(0);
        if (this->voxel_num_ == 0) {
            return;
        }

        horizontal_fov = std::clamp(horizontal_fov, kFovTolerance, kPi - kFovTolerance);
        vertical_fov = std::clamp(vertical_fov, kFovTolerance, 2.0f * kPi - kFovTolerance);

        const size_t N = this->capacity_;
        const bool is_nvidia = this->queue_.is_nvidia();
        size_t filtered_voxel_count = 0;

        // Allocate the worst-case storage before the filtering pass and shrink once visibility is known.
        result.resize_points(this->voxel_num_);
        if (this->has_rgb_data_) {
            result.resize_rgb(this->voxel_num_);
        }
        if (this->has_intensity_data_) {
            result.resize_intensities(this->voxel_num_);
        }

        if (is_nvidia) {
            // NVIDIA GPU: Use prefix sum approach
            // Step 1: Compute valid flags
            if (this->valid_flags_ptr_->size() < N) {
                this->valid_flags_ptr_->resize(N);
            }

            this->queue_.ptr
                ->submit([&](sycl::handler& h) {
                    const size_t work_group_size = this->queue_.get_work_group_size();
                    const size_t global_size = this->queue_.get_global_size(N);

                    auto valid_flags = this->valid_flags_ptr_->data();
                    auto key_ptr = this->key_ptr_->data();
                    auto core_ptr = this->core_data_ptr_->data();

                    const auto world_to_sensor_T = eigen_utils::to_sycl_vec(sensor_pose.inverse().matrix());
                    const float occupancy_threshold = this->occupancy_threshold_log_odds_;
                    const float voxel_size = this->voxel_size_;
                    const float inv_voxel_size = this->inv_voxel_size_;
                    const size_t max_probe = this->max_probe_length_;
                    const size_t capacity = this->capacity_;

                    const float sensor_x = sensor_pose.translation().x();
                    const float sensor_y = sensor_pose.translation().y();
                    const float sensor_z = sensor_pose.translation().z();

                    const float max_dist_sq = max_distance * max_distance;
                    const float cos_limit_horizontal = sycl::cos(horizontal_fov * 0.5f);
                    const float cos_limit_vertical = sycl::cos(vertical_fov * 0.5f);
                    const bool include_backward = horizontal_fov >= (kPi - kFovTolerance);

                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t i = item.get_global_id(0);
                        if (i >= N) {
                            return;
                        }

                        const uint64_t current_key = key_ptr[i];
                        if (current_key == VoxelConstants::invalid_coord || current_key == VoxelConstants::deleted_coord) {
                            valid_flags[i] = 0U;
                            return;
                        }

                        const VoxelCoreData& core = core_ptr[i];
                        if (core.hit_count == 0U || core.log_odds < occupancy_threshold) {
                            valid_flags[i] = 0U;
                            return;
                        }

                        const float inv_count = 1.0f / static_cast<float>(core.hit_count);
                        const float cx = core.sum_x * inv_count;
                        const float cy = core.sum_y * inv_count;
                        const float cz = core.sum_z * inv_count;

                        const float dx = cx - sensor_x;
                        const float dy = cy - sensor_y;
                        const float dz = cz - sensor_z;
                        const float dist_sq = dx * dx + dy * dy + dz * dz;
                        if (dist_sq > max_dist_sq) {
                            valid_flags[i] = 0U;
                            return;
                        }

                        const auto world_to_sensor_Mat = eigen_utils::from_sycl_vec(world_to_sensor_T);
                        const Eigen::Vector3f local_pt =
                            eigen_utils::multiply<3, 3>(world_to_sensor_Mat.block<3, 3>(0, 0), Eigen::Vector3f{dx, dy, dz});

                        if (!include_backward && local_pt.x() <= 0.0f) {
                            valid_flags[i] = 0U;
                            return;
                        }

                        const float forward_projection = include_backward ? sycl::fabs(local_pt.x()) : local_pt.x();

                        const float horizontal_norm_sq = forward_projection * forward_projection + local_pt.y() * local_pt.y();
                        float cos_horizontal = 1.0f;
                        if (horizontal_norm_sq > 0.0f) {
                            const float inv_horizontal_norm = sycl::rsqrt(horizontal_norm_sq);
                            cos_horizontal = forward_projection * inv_horizontal_norm;
                            cos_horizontal = sycl::clamp(cos_horizontal, -1.0f, 1.0f);
                        }
                        if (cos_horizontal < cos_limit_horizontal) {
                            valid_flags[i] = 0U;
                            return;
                        }

                        const float vertical_norm_sq = forward_projection * forward_projection + local_pt.z() * local_pt.z();
                        float cos_vertical = 1.0f;
                        if (vertical_norm_sq > 0.0f) {
                            const float inv_vertical_norm = sycl::rsqrt(vertical_norm_sq);
                            cos_vertical = forward_projection * inv_vertical_norm;
                            cos_vertical = sycl::clamp(cos_vertical, -1.0f, 1.0f);
                        }
                        if (cos_vertical < cos_limit_vertical) {
                            valid_flags[i] = 0U;
                            return;
                        }

                        const float distance = sycl::sqrt(dist_sq);
                        bool occluded = false;

                        if (distance > voxel_size) {
                            traverse_ray_exclusive_impl(
                                sensor_x, sensor_y, sensor_z, cx, cy, cz, inv_voxel_size,
                                [&](int64_t ix, int64_t iy, int64_t iz) {
                                    uint64_t sample_key = VoxelConstants::invalid_coord;
                                    if (!grid_to_key_device(ix, iy, iz, sample_key) || sample_key == current_key) {
                                        return true;
                                    }

                                    for (size_t probe = 0; probe < max_probe; ++probe) {
                                        const size_t slot = compute_slot_id(sample_key, probe, capacity);
                                        const uint64_t stored_key = key_ptr[slot];
                                        if (stored_key == sample_key) {
                                            const VoxelCoreData& occ = core_ptr[slot];
                                            if (occ.hit_count > 0U && occ.log_odds >= occupancy_threshold) {
                                                const float inv_occ = 1.0f / static_cast<float>(occ.hit_count);
                                                const float occ_cx = occ.sum_x * inv_occ;
                                                const float occ_cy = occ.sum_y * inv_occ;
                                                const float occ_cz = occ.sum_z * inv_occ;
                                                const float occ_dx = occ_cx - sensor_x;
                                                const float occ_dy = occ_cy - sensor_y;
                                                const float occ_dz = occ_cz - sensor_z;
                                                const float occ_dist_sq = occ_dx * occ_dx + occ_dy * occ_dy + occ_dz * occ_dz;
                                                if (occ_dist_sq + kOcclusionEpsilon < dist_sq) {
                                                    occluded = true;
                                                    return false;
                                                }
                                            }
                                            return true;
                                        }
                                        if (stored_key == VoxelConstants::invalid_coord) {
                                            return true;
                                        }
                                        if (stored_key == VoxelConstants::deleted_coord) {
                                            continue;
                                        }
                                    }
                                    return true;
                                });
                        }

                        if (occluded) {
                            valid_flags[i] = 0U;
                            return;
                        }

                        valid_flags[i] = 1U;
                    });
                })
                .wait_and_throw();

            // Step 2: Compute prefix sum
            filtered_voxel_count = this->prefix_sum_->compute(*this->valid_flags_ptr_);

            // Step 3: Write output using prefix sum indices
            this->queue_.ptr
                ->submit([&](sycl::handler& h) {
                    const size_t work_group_size = this->queue_.get_work_group_size();
                    const size_t global_size = this->queue_.get_global_size(N);

                    auto valid_flags = this->valid_flags_ptr_->data();
                    auto prefix_sum_ptr = this->prefix_sum_->get_prefix_sum().data();
                    auto core_ptr = this->core_data_ptr_->data();
                    auto color_ptr = this->color_data_ptr_->data();
                    auto intensity_data_ptr = this->intensity_data_ptr_->data();

                    auto points_ptr = result.points_ptr();
                    auto rgb_ptr = this->has_rgb_data_ ? result.rgb_ptr() : static_cast<RGBType*>(nullptr);
                    auto intensity_ptr =
                        this->has_intensity_data_ ? result.intensities_ptr() : static_cast<float*>(nullptr);

                    const bool has_rgb = this->has_rgb_data_;
                    const bool has_intensity = this->has_intensity_data_;

                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t i = item.get_global_id(0);
                        if (i >= N || valid_flags[i] == 0U) {
                            return;
                        }

                        const size_t output_idx = prefix_sum_ptr[i] - 1;
                        const VoxelCoreData& core = core_ptr[i];
                        const float inv_count = 1.0f / static_cast<float>(core.hit_count);
                        const float cx = core.sum_x * inv_count;
                        const float cy = core.sum_y * inv_count;
                        const float cz = core.sum_z * inv_count;

                        points_ptr[output_idx].x() = cx;
                        points_ptr[output_idx].y() = cy;
                        points_ptr[output_idx].z() = cz;
                        points_ptr[output_idx].w() = 1.0f;

                        if (has_rgb && rgb_ptr) {
                            const VoxelColorData& color = color_ptr[i];
                            if (core.hit_count > 0U) {
                                rgb_ptr[output_idx].x() = color.sum_r * inv_count;
                                rgb_ptr[output_idx].y() = color.sum_g * inv_count;
                                rgb_ptr[output_idx].z() = color.sum_b * inv_count;
                                rgb_ptr[output_idx].w() = color.sum_a * inv_count;
                            } else {
                                rgb_ptr[output_idx].setZero();
                            }
                        }

                        if (has_intensity && intensity_ptr) {
                            const VoxelIntensityData& intensity_data = intensity_data_ptr[i];
                            if (core.hit_count > 0U) {
                                intensity_ptr[output_idx] = intensity_data.sum_intensity * inv_count;
                            } else {
                                intensity_ptr[output_idx] = 0.0f;
                            }
                        }
                    });
                })
                .wait_and_throw();
        } else {
            // Non-NVIDIA: Use fetch_add approach (original implementation)
            shared_vector<uint32_t> counter(1, 0U, *this->queue_.ptr);

            auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
                auto key_ptr = this->key_ptr_->data();
                auto core_ptr = this->core_data_ptr_->data();
                auto color_ptr = this->color_data_ptr_->data();
                auto intensity_data_ptr = this->intensity_data_ptr_->data();

                auto points_ptr = result.points_ptr();
                auto rgb_ptr = this->has_rgb_data_ ? result.rgb_ptr() : static_cast<RGBType*>(nullptr);
                auto intensity_ptr = this->has_intensity_data_ ? result.intensities_ptr() : static_cast<float*>(nullptr);

                const auto world_to_sensor_T = eigen_utils::to_sycl_vec(sensor_pose.inverse().matrix());

                const float occupancy_threshold = this->occupancy_threshold_log_odds_;
                const float voxel_size = this->voxel_size_;
                const float inv_voxel_size = this->inv_voxel_size_;

                const bool has_rgb = this->has_rgb_data_;
                const bool has_intensity = this->has_intensity_data_;

                const size_t max_probe = this->max_probe_length_;
                const size_t capacity = this->capacity_;

                auto counter_ptr = counter.data();
                const float sensor_x = sensor_pose.translation().x();
                const float sensor_y = sensor_pose.translation().y();
                const float sensor_z = sensor_pose.translation().z();

                const float max_dist_sq = max_distance * max_distance;
                const float cos_limit_horizontal = sycl::cos(horizontal_fov * 0.5f);
                const float cos_limit_vertical = sycl::cos(vertical_fov * 0.5f);
                const bool include_backward = horizontal_fov >= (kPi - kFovTolerance);

                h.parallel_for(sycl::range<1>(this->capacity_), [=](sycl::id<1> idx) {
                    const size_t i = idx[0];
                    const uint64_t current_key = key_ptr[i];
                    if (current_key == VoxelConstants::invalid_coord || current_key == VoxelConstants::deleted_coord) {
                        return;
                    }

                    const VoxelCoreData& core = core_ptr[i];
                    if (core.hit_count == 0U || core.log_odds < occupancy_threshold) {
                        return;
                    }

                    const float inv_count = 1.0f / static_cast<float>(core.hit_count);
                    const float cx = core.sum_x * inv_count;
                    const float cy = core.sum_y * inv_count;
                    const float cz = core.sum_z * inv_count;

                    const float dx = cx - sensor_x;
                    const float dy = cy - sensor_y;
                    const float dz = cz - sensor_z;
                    const float dist_sq = dx * dx + dy * dy + dz * dz;
                    if (dist_sq > max_dist_sq) {
                        return;
                    }
                    const auto world_to_sensor_Mat = eigen_utils::from_sycl_vec(world_to_sensor_T);
                    const Eigen::Vector3f local_pt =
                        eigen_utils::multiply<3, 3>(world_to_sensor_Mat.block<3, 3>(0, 0), Eigen::Vector3f{dx, dy, dz});

                    if (!include_backward && local_pt.x() <= 0.0f) {
                        return;
                    }

                    const float forward_projection = include_backward ? sycl::fabs(local_pt.x()) : local_pt.x();

                    const float horizontal_norm_sq = forward_projection * forward_projection + local_pt.y() * local_pt.y();
                    float cos_horizontal = 1.0f;
                    if (horizontal_norm_sq > 0.0f) {
                        const float inv_horizontal_norm = sycl::rsqrt(horizontal_norm_sq);
                        cos_horizontal = forward_projection * inv_horizontal_norm;
                        cos_horizontal = sycl::clamp(cos_horizontal, -1.0f, 1.0f);
                    }
                    if (cos_horizontal < cos_limit_horizontal) {
                        return;
                    }

                    const float vertical_norm_sq = forward_projection * forward_projection + local_pt.z() * local_pt.z();
                    float cos_vertical = 1.0f;
                    if (vertical_norm_sq > 0.0f) {
                        const float inv_vertical_norm = sycl::rsqrt(vertical_norm_sq);
                        cos_vertical = forward_projection * inv_vertical_norm;
                        cos_vertical = sycl::clamp(cos_vertical, -1.0f, 1.0f);
                    }
                    if (cos_vertical < cos_limit_vertical) {
                        return;
                    }

                    const float distance = sycl::sqrt(dist_sq);
                    bool occluded = false;

                    if (distance > voxel_size) {
                        traverse_ray_exclusive_impl(
                            sensor_x, sensor_y, sensor_z, cx, cy, cz, inv_voxel_size,
                            [&](int64_t ix, int64_t iy, int64_t iz) {
                                uint64_t sample_key = VoxelConstants::invalid_coord;
                                if (!grid_to_key_device(ix, iy, iz, sample_key) || sample_key == current_key) {
                                    return true;
                                }

                                for (size_t probe = 0; probe < max_probe; ++probe) {
                                    const size_t slot = compute_slot_id(sample_key, probe, capacity);
                                    const uint64_t stored_key = key_ptr[slot];
                                    if (stored_key == sample_key) {
                                        const VoxelCoreData& occ = core_ptr[slot];
                                        if (occ.hit_count > 0U && occ.log_odds >= occupancy_threshold) {
                                            const float inv_occ = 1.0f / static_cast<float>(occ.hit_count);
                                            const float occ_cx = occ.sum_x * inv_occ;
                                            const float occ_cy = occ.sum_y * inv_occ;
                                            const float occ_cz = occ.sum_z * inv_occ;
                                            const float occ_dx = occ_cx - sensor_x;
                                            const float occ_dy = occ_cy - sensor_y;
                                            const float occ_dz = occ_cz - sensor_z;
                                            const float occ_dist_sq = occ_dx * occ_dx + occ_dy * occ_dy + occ_dz * occ_dz;
                                            if (occ_dist_sq + kOcclusionEpsilon < dist_sq) {
                                                occluded = true;
                                                return false;
                                            }
                                        }
                                        return true;
                                    }
                                    if (stored_key == VoxelConstants::invalid_coord) {
                                        return true;
                                    }
                                    if (stored_key == VoxelConstants::deleted_coord) {
                                        continue;
                                    }
                                }
                                return true;
                            });
                    }

                    if (occluded) {
                        return;
                    }

                    const uint32_t index = atomic_ref_uint32_t(counter_ptr[0]).fetch_add(1U);
                    points_ptr[index].x() = cx;
                    points_ptr[index].y() = cy;
                    points_ptr[index].z() = cz;
                    points_ptr[index].w() = 1.0f;

                    if (has_rgb && rgb_ptr) {
                        const VoxelColorData& color = color_ptr[i];
                        if (core.hit_count > 0U) {
                            rgb_ptr[index].x() = color.sum_r * inv_count;
                            rgb_ptr[index].y() = color.sum_g * inv_count;
                            rgb_ptr[index].z() = color.sum_b * inv_count;
                            rgb_ptr[index].w() = color.sum_a * inv_count;
                        } else {
                            rgb_ptr[index].setZero();
                        }
                    }

                    if (has_intensity && intensity_ptr) {
                        const VoxelIntensityData& intensity_data = intensity_data_ptr[i];
                        if (core.hit_count > 0U) {
                            intensity_ptr[index] = intensity_data.sum_intensity * inv_count;
                        } else {
                            intensity_ptr[index] = 0.0f;
                        }
                    }
                });
            });
            event.wait_and_throw();
            filtered_voxel_count = static_cast<size_t>(counter.at(0));
        }

        // Resize to actual count
        result.resize_points(filtered_voxel_count);
        if (this->has_rgb_data_) {
            result.resize_rgb(filtered_voxel_count);
        }
        if (this->has_intensity_data_) {
            result.resize_intensities(filtered_voxel_count);
        }
    }

    /// @brief Compute the overlap ratio between the map and an input point cloud.
    /// @param cloud Point cloud in the sensor frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    /// @return Ratio of points that overlap with existing voxels in the map.
    float compute_overlap_ratio(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) const {
        if (!cloud.points || cloud.points->empty() || this->voxel_num_ == 0) {
            return 0.0f;
        }

        const size_t N = cloud.size();
        shared_vector<uint32_t> overlap_counter(1, 0U, *this->queue_.ptr);

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            auto overlap_reduction = sycl::reduction(overlap_counter.data(), sycl::plus<uint32_t>());
            const auto trans = eigen_utils::to_sycl_vec(sensor_pose.matrix());

            const auto point_ptr = cloud.points_ptr();
            auto key_ptr = this->key_ptr_->data();
            auto core_ptr = this->core_data_ptr_->data();
            const float voxel_size_inv = this->inv_voxel_size_;
            const float occupancy_threshold = this->occupancy_threshold_log_odds_;
            const size_t max_probe = this->max_probe_length_;
            const size_t capacity = this->capacity_;

            h.parallel_for(sycl::range<1>(N), overlap_reduction, [=](sycl::id<1> idx, auto& overlap_sum) {
                const size_t i = idx[0];
                const PointType local_point = point_ptr[i];
                PointType world_point;
                // Transform the input point into the map frame before voxel hashing.
                transform::kernel::transform_point(local_point, world_point, trans);
                const uint64_t voxel_hash = filter::kernel::compute_voxel_bit(world_point, voxel_size_inv);
                if (voxel_hash == VoxelConstants::invalid_coord) {
                    return;
                }

                for (size_t probe = 0; probe < max_probe; ++probe) {
                    const size_t slot = compute_slot_id(voxel_hash, probe, capacity);
                    const uint64_t stored_key = key_ptr[slot];
                    if (stored_key == voxel_hash) {
                        // Count as overlap only when the voxel is confidently occupied.
                        const VoxelCoreData& core = core_ptr[slot];
                        if (core.hit_count > 0U && core.log_odds >= occupancy_threshold) {
                            overlap_sum += 1U;
                        }
                        return;
                    }
                    if (stored_key == VoxelConstants::invalid_coord) {
                        return;
                    }
                    if (stored_key == VoxelConstants::deleted_coord) {
                        continue;
                    }
                }
            });
        });

        event.wait_and_throw();

        return static_cast<float>(overlap_counter.at(0)) / static_cast<float>(N);
    }

private:
    inline static constexpr float kPi = 3.1415927f;
    inline static constexpr float kFovTolerance = 1e-6f;
    inline static constexpr float kOcclusionEpsilon = 1e-6f;
    using atomic_ref_float = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint32_t = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint64_t = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;

    /// @brief Core voxel data containing position and occupancy information (32 bytes)
    struct VoxelCoreData {
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        float log_odds = 0.0f;
        float pending_log_odds = 0.0f;
        uint32_t hit_count = 0U;
        uint32_t last_updated = 0U;  // Frame index when the voxel was last modified.
        uint32_t padding = 0U;       // Padding for 32-byte alignment
    };

    /// @brief Color data for RGB information (16 bytes)
    struct VoxelColorData {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_a = 0.0f;
    };

    /// @brief Intensity data for reflectivity information (4 bytes)
    struct VoxelIntensityData {
        float sum_intensity = 0.0f;
    };

    /// @brief Core accumulator for position and occupancy
    struct VoxelCoreAccumulator {
        uint64_t voxel_hash = VoxelConstants::invalid_coord;
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        float log_odds_delta = 0.0f;
        uint32_t hit_increment = 0U;
    };

    /// @brief Color accumulator for RGB information
    struct VoxelColorAccumulator {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_a = 0.0f;
    };

    /// @brief Intensity accumulator for reflectivity information
    struct VoxelIntensityAccumulator {
        float sum_intensity = 0.0f;
    };

    /// @brief Voxel local accumulator (64 bytes)
    struct VoxelLocalData {
        uint64_t voxel_idx = VoxelConstants::invalid_coord;
        VoxelCoreAccumulator core_acc;
        VoxelColorAccumulator color_acc;
        VoxelIntensityAccumulator intensity_acc;
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
        if (!this->key_ptr_) {
            return nullptr;
        }
        for (size_t j = 0; j < this->max_probe_length_; ++j) {
            const size_t slot = this->compute_slot_id(key, j, this->capacity_);
            const uint64_t stored_key = this->key_ptr_->at(slot);
            if (stored_key == key) {
                return &this->core_data_ptr_->at(slot);
            }
            if (stored_key == VoxelConstants::invalid_coord) {
                break;
            }
            if (stored_key == VoxelConstants::deleted_coord) {
                continue;
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

    void allocate_storage(size_t new_capacity) {
        this->key_ptr_ =
            std::make_shared<shared_vector<uint64_t>>(new_capacity, VoxelConstants::invalid_coord, *this->queue_.ptr);
        this->core_data_ptr_ =
            std::make_shared<shared_vector<VoxelCoreData>>(new_capacity, VoxelCoreData{}, *this->queue_.ptr);
        this->color_data_ptr_ =
            std::make_shared<shared_vector<VoxelColorData>>(new_capacity, VoxelColorData{}, *this->queue_.ptr);
        this->intensity_data_ptr_ =
            std::make_shared<shared_vector<VoxelIntensityData>>(new_capacity, VoxelIntensityData{}, *this->queue_.ptr);

        this->capacity_ = new_capacity;
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
        const auto old_capacity = this->capacity_;
        auto old_keys = this->key_ptr_;
        auto old_core_data = this->core_data_ptr_;
        auto old_color_data = this->color_data_ptr_;
        auto old_intensity_data = this->intensity_data_ptr_;

        this->allocate_storage(new_capacity);

        shared_vector<uint32_t> voxel_counter(1, 0U, *this->queue_.ptr);
        shared_vector<uint32_t> failure_flag(1, 0U, *this->queue_.ptr);

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t N = old_capacity;
            const size_t work_group_size = this->queue_.get_work_group_size();
            const size_t global_size = this->queue_.get_global_size(N);

            const auto old_key_ptr = old_keys->data();
            const auto old_core_ptr = old_core_data->data();
            const auto old_color_ptr = old_color_data->data();
            const auto old_intensity_ptr = old_intensity_data->data();
            auto new_key_ptr = this->key_ptr_->data();
            auto new_core_ptr = this->core_data_ptr_->data();
            auto new_color_ptr = this->color_data_ptr_->data();
            auto new_intensity_ptr = this->intensity_data_ptr_->data();
            const size_t new_capacity_local = this->capacity_;
            const size_t max_probe = this->max_probe_length_;
            auto voxel_num_ptr = voxel_counter.data();
            auto failure_ptr = failure_flag.data();
            auto range = sycl::nd_range<1>(global_size, work_group_size);

            const auto has_rgb = this->has_rgb_data_;
            const auto has_intensity = this->has_intensity_data_;

            if (this->queue_.is_nvidia()) {
                // Count inserted voxels via reduction when running on NVIDIA GPUs.
                auto voxel_num = sycl::reduction(voxel_num_ptr, sycl::plus<uint32_t>());

                h.parallel_for(range, voxel_num, [=](sycl::nd_item<1> item, auto& voxel_num_arg) {
                    const uint32_t i = item.get_global_id(0);
                    if (i >= N) return;

                    const uint64_t key = old_key_ptr[i];
                    if (key == VoxelConstants::invalid_coord || key == VoxelConstants::deleted_coord) return;

                    const VoxelCoreData core_data = old_core_ptr[i];
                    const VoxelColorData color_data = old_color_ptr[i];
                    const VoxelIntensityData intensity_data = old_intensity_ptr[i];
                    bool inserted = false;

                    for (size_t probe = 0; probe < max_probe; ++probe) {
                        const size_t slot = compute_slot_id(key, probe, new_capacity_local);
                        uint64_t expected = VoxelConstants::invalid_coord;

                        if (atomic_ref_uint64_t(new_key_ptr[slot]).compare_exchange_strong(expected, key)) {
                            new_core_ptr[slot] = core_data;
                            if (has_rgb) {
                                new_color_ptr[slot] = color_data;
                            }
                            if (has_intensity) {
                                new_intensity_ptr[slot] = intensity_data;
                            }
                            voxel_num_arg += 1U;
                            inserted = true;
                            break;
                        }
                    }

                    if (!inserted) {
                        atomic_ref_uint32_t(failure_ptr[0]).store(1U);
                    }
                });
            } else {
                h.parallel_for(range, [=](sycl::nd_item<1> item) {
                    const uint32_t i = item.get_global_id(0);
                    if (i >= N) return;

                    const uint64_t key = old_key_ptr[i];
                    if (key == VoxelConstants::invalid_coord || key == VoxelConstants::deleted_coord) return;

                    const VoxelCoreData core_data = old_core_ptr[i];
                    const VoxelColorData color_data = old_color_ptr[i];
                    const VoxelIntensityData intensity_data = old_intensity_ptr[i];
                    bool inserted = false;

                    for (size_t probe = 0; probe < max_probe; ++probe) {
                        const size_t slot = compute_slot_id(key, probe, new_capacity_local);
                        uint64_t expected = VoxelConstants::invalid_coord;

                        if (atomic_ref_uint64_t(new_key_ptr[slot]).compare_exchange_strong(expected, key)) {
                            new_core_ptr[slot] = core_data;
                            if (has_rgb) {
                                new_color_ptr[slot] = color_data;
                            }
                            if (has_intensity) {
                                new_intensity_ptr[slot] = intensity_data;
                            }
                            atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(1U);
                            inserted = true;
                            break;
                        }
                    }

                    if (!inserted) {
                        atomic_ref_uint32_t(failure_ptr[0]).store(1U);
                    }
                });
            }
        });

        auto host_event = this->queue_.ptr->submit([&](sycl::handler& h) {
            h.depends_on(event);
            h.host_task([&]() {
                if (failure_flag.at(0) != 0U) {
                    std::cerr << "Could not find slot for " << failure_flag.at(0) << " voxel" << std::endl;
                    // throw std::runtime_error("Rehash failed: could not find a slot for a voxel.");
                }
                this->voxel_num_ = static_cast<size_t>(voxel_counter.at(0));
            });
        });
        host_event.wait_and_throw();
    }

    template <typename CounterFunc>
    static void global_reduction(const VoxelLocalData& data, uint64_t* key_ptr, VoxelCoreData* core_ptr,
                                 VoxelColorData* color_ptr, VoxelIntensityData* intensity_ptr,
                                 const uint32_t current_frame, const size_t max_probe, const size_t capacity,
                                 CounterFunc counter, bool has_rgb, bool has_intensity) {
        const uint64_t voxel_hash = data.voxel_idx;
        if (voxel_hash == VoxelConstants::invalid_coord) {
            return;
        }

        for (size_t probe = 0; probe < max_probe; ++probe) {
            const size_t slot_idx = compute_slot_id(voxel_hash, probe, capacity);
            auto key_ref = atomic_ref_uint64_t(key_ptr[slot_idx]);
            uint64_t expected = key_ref.load();
            if (expected == VoxelConstants::invalid_coord || expected == VoxelConstants::deleted_coord) {
                // Attempt to insert. On CAS failure, `expected` is updated, and we fall through.
                if (key_ref.compare_exchange_strong(expected, voxel_hash)) {
                    counter(1U);
                    atomic_add_voxel_data(data.core_acc, data.color_acc, data.intensity_acc, core_ptr[slot_idx],
                                          color_ptr[slot_idx], intensity_ptr[slot_idx], has_rgb, has_intensity);
                    atomic_ref_uint32_t(core_ptr[slot_idx].last_updated).store(current_frame);
                    break;
                }
            }
            // If the slot was already occupied, or if another thread just inserted our key, update it.
            if (expected == voxel_hash) {
                atomic_add_voxel_data(data.core_acc, data.color_acc, data.intensity_acc, core_ptr[slot_idx],
                                      color_ptr[slot_idx], intensity_ptr[slot_idx], has_rgb, has_intensity);
                atomic_ref_uint32_t(core_ptr[slot_idx].last_updated).store(current_frame);
                break;
            }
        }
    }

    /// @brief Traverse voxels between the origin and target in grid coordinates, excluding both endpoints.
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

        int64_t ix = static_cast<int64_t>(sycl::floor(scaled_origin_x));
        int64_t iy = static_cast<int64_t>(sycl::floor(scaled_origin_y));
        int64_t iz = static_cast<int64_t>(sycl::floor(scaled_origin_z));

        const int64_t target_ix = static_cast<int64_t>(sycl::floor(scaled_target_x));
        const int64_t target_iy = static_cast<int64_t>(sycl::floor(scaled_target_y));
        const int64_t target_iz = static_cast<int64_t>(sycl::floor(scaled_target_z));

        if (ix == target_ix && iy == target_iy && iz == target_iz) {
            return;
        }

        const float dir_x = scaled_target_x - scaled_origin_x;
        const float dir_y = scaled_target_y - scaled_origin_y;
        const float dir_z = scaled_target_z - scaled_origin_z;

        const float abs_dir_x = sycl::fabs(dir_x);
        const float abs_dir_y = sycl::fabs(dir_y);
        const float abs_dir_z = sycl::fabs(dir_z);

        const int step_x = (dir_x > 0.0f) ? 1 : ((dir_x < 0.0f) ? -1 : 0);
        const int step_y = (dir_y > 0.0f) ? 1 : ((dir_y < 0.0f) ? -1 : 0);
        const int step_z = (dir_z > 0.0f) ? 1 : ((dir_z < 0.0f) ? -1 : 0);

        const float start_frac_x = scaled_origin_x - sycl::floor(scaled_origin_x);
        const float start_frac_y = scaled_origin_y - sycl::floor(scaled_origin_y);
        const float start_frac_z = scaled_origin_z - sycl::floor(scaled_origin_z);

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

        float t_delta_x = (step_x != 0) ? inv_dir_mag_x : inf;
        float t_delta_y = (step_y != 0) ? inv_dir_mag_y : inf;
        float t_delta_z = (step_z != 0) ? inv_dir_mag_z : inf;

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

    /// @brief Convert integer grid coordinates into the packed voxel key representation.
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

    /// @brief Visit voxels from origin to target, including the origin cell on the host.
    template <typename Visitor>
    void traverse_ray(const Eigen::Vector3f& origin, const Eigen::Vector3f& target, Visitor&& visitor) const {
        const float inv_voxel = this->inv_voxel_size_;
        const float scaled_origin_x = origin.x() * inv_voxel;
        const float scaled_origin_y = origin.y() * inv_voxel;
        const float scaled_origin_z = origin.z() * inv_voxel;
        const float scaled_target_x = target.x() * inv_voxel;
        const float scaled_target_y = target.y() * inv_voxel;
        const float scaled_target_z = target.z() * inv_voxel;

        int64_t ix = static_cast<int64_t>(std::floor(scaled_origin_x));
        int64_t iy = static_cast<int64_t>(std::floor(scaled_origin_y));
        int64_t iz = static_cast<int64_t>(std::floor(scaled_origin_z));

        const int64_t target_ix = static_cast<int64_t>(std::floor(scaled_target_x));
        const int64_t target_iy = static_cast<int64_t>(std::floor(scaled_target_y));
        const int64_t target_iz = static_cast<int64_t>(std::floor(scaled_target_z));

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

    static void atomic_add_voxel_data(const VoxelCoreAccumulator& core_src, const VoxelColorAccumulator& color_src,
                                      const VoxelIntensityAccumulator& intensity_src, VoxelCoreData& core_dst,
                                      VoxelColorData& color_dst, VoxelIntensityData& intensity_dst, bool has_rgb,
                                      bool has_intensity) {
        // Core data
        atomic_ref_float(core_dst.sum_x).fetch_add(core_src.sum_x);
        atomic_ref_float(core_dst.sum_y).fetch_add(core_src.sum_y);
        atomic_ref_float(core_dst.sum_z).fetch_add(core_src.sum_z);
        atomic_ref_uint32_t(core_dst.hit_count).fetch_add(core_src.hit_increment);
        atomic_ref_float(core_dst.pending_log_odds).fetch_add(core_src.log_odds_delta);

        // Color data (only if present)
        if (has_rgb) {
            atomic_ref_float(color_dst.sum_r).fetch_add(color_src.sum_r);
            atomic_ref_float(color_dst.sum_g).fetch_add(color_src.sum_g);
            atomic_ref_float(color_dst.sum_b).fetch_add(color_src.sum_b);
            atomic_ref_float(color_dst.sum_a).fetch_add(color_src.sum_a);
        }

        // Intensity data (only if present)
        if (has_intensity) {
            atomic_ref_float(intensity_dst.sum_intensity).fetch_add(intensity_src.sum_intensity);
        }
    }

    void integrate_points(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose, const bool has_rgb,
                          const bool has_intensity) {
        const size_t N = cloud.size();
        shared_vector<uint32_t> voxel_counter(1, this->voxel_num_, *this->queue_.ptr);

        // Parallel reduction merges per-point contributions into hashed voxels.
        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t local_size = this->compute_work_group_size();
            const size_t num_work_groups = (N + local_size - 1) / local_size;
            const size_t global_size = num_work_groups * local_size;

            auto local_voxel_data = sycl::local_accessor<VoxelLocalData>(local_size, h);
            const auto trans = eigen_utils::to_sycl_vec(sensor_pose.matrix());

            size_t power_of_2 = 1;
            while (power_of_2 < local_size) {
                power_of_2 <<= 1;
            }

            const auto point_ptr = cloud.points_ptr();
            const auto rgb_ptr = has_rgb ? cloud.rgb_ptr() : static_cast<RGBType*>(nullptr);
            const auto intensity_ptr = has_intensity ? cloud.intensities_ptr() : static_cast<float*>(nullptr);
            auto key_ptr = this->key_ptr_->data();
            auto core_ptr = this->core_data_ptr_->data();
            auto color_ptr = this->color_data_ptr_->data();
            auto intensity_data_ptr = this->intensity_data_ptr_->data();
            const auto voxel_size_inv = this->inv_voxel_size_;
            const auto current_frame = this->frame_index_;
            const auto max_probe = this->max_probe_length_;
            const auto capacity = this->capacity_;
            const float log_odds_hit = this->log_odds_hit_;

            auto load_entry = [=](VoxelLocalData& entry, const size_t idx) {
                const PointType local_point = point_ptr[idx];
                PointType world_point;
                transform::kernel::transform_point(local_point, world_point, trans);
                const uint64_t voxel_hash = filter::kernel::compute_voxel_bit(world_point, voxel_size_inv);

                entry.voxel_idx = voxel_hash;
                entry.core_acc.voxel_hash = voxel_hash;
                entry.core_acc.sum_x = world_point.x();
                entry.core_acc.sum_y = world_point.y();
                entry.core_acc.sum_z = world_point.z();
                entry.core_acc.hit_increment = 1U;
                entry.core_acc.log_odds_delta = log_odds_hit;

                if (has_rgb && rgb_ptr) {
                    const auto color = rgb_ptr[idx];
                    entry.color_acc.sum_r = color.x();
                    entry.color_acc.sum_g = color.y();
                    entry.color_acc.sum_b = color.z();
                    entry.color_acc.sum_a = color.w();
                } else {
                    entry.color_acc.sum_r = 0.0f;
                    entry.color_acc.sum_g = 0.0f;
                    entry.color_acc.sum_b = 0.0f;
                    entry.color_acc.sum_a = 0.0f;
                }

                if (has_intensity && intensity_ptr) {
                    entry.intensity_acc.sum_intensity = intensity_ptr[idx];
                } else {
                    entry.intensity_acc.sum_intensity = 0.0f;
                }
            };

            auto combine_entry = [has_rgb, has_intensity](VoxelLocalData& dst, const VoxelLocalData& src) {
                dst.core_acc.sum_x += src.core_acc.sum_x;
                dst.core_acc.sum_y += src.core_acc.sum_y;
                dst.core_acc.sum_z += src.core_acc.sum_z;
                dst.core_acc.hit_increment += src.core_acc.hit_increment;
                dst.core_acc.log_odds_delta += src.core_acc.log_odds_delta;
                if (has_rgb) {
                    dst.color_acc.sum_r += src.color_acc.sum_r;
                    dst.color_acc.sum_g += src.color_acc.sum_g;
                    dst.color_acc.sum_b += src.color_acc.sum_b;
                    dst.color_acc.sum_a += src.color_acc.sum_a;
                }
                if (has_intensity) {
                    dst.intensity_acc.sum_intensity += src.intensity_acc.sum_intensity;
                }
            };

            auto reset_entry = [](VoxelLocalData& entry) {
                entry.voxel_idx = VoxelConstants::invalid_coord;
                entry.core_acc = VoxelCoreAccumulator{};
                entry.color_acc = VoxelColorAccumulator{};
                entry.intensity_acc = VoxelIntensityAccumulator{};
            };

            // Configure key accessors and comparators for the shared reduction helpers.
            auto key_of_entry = [](const VoxelLocalData& entry) { return entry.voxel_idx; };
            auto compare_keys = [](uint64_t lhs, uint64_t rhs) { return lhs < rhs; };
            auto equal_keys = [](uint64_t lhs, uint64_t rhs) { return lhs == rhs; };

            if (this->queue_.is_nvidia()) {
                auto reduction = sycl::reduction(voxel_counter.data(), sycl::plus<uint32_t>());
                h.parallel_for(  //
                    sycl::nd_range<1>(global_size, local_size), reduction,
                    [=](sycl::nd_item<1> item, auto& voxel_num_arg) {
                        common::local_reduction<true, VoxelLocalData>(
                            local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(), N, local_size,
                            power_of_2, item, load_entry, combine_entry, reset_entry, VoxelConstants::invalid_coord,
                            key_of_entry, compare_keys, equal_keys);

                        const size_t lid = item.get_local_id(0);
                        if (item.get_global_id(0) >= N) {
                            return;
                        }

                        const VoxelLocalData local = local_voxel_data[lid];
                        global_reduction(
                            local, key_ptr, core_ptr, color_ptr, intensity_data_ptr, current_frame, max_probe, capacity,
                            [&](uint32_t add) { voxel_num_arg += add; }, has_rgb, has_intensity);
                    });
            } else {
                auto voxel_ptr_counter = voxel_counter.data();
                h.parallel_for(  //
                    sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
                        common::local_reduction<false, VoxelLocalData>(
                            local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(), N, local_size,
                            power_of_2, item, load_entry, combine_entry, reset_entry, VoxelConstants::invalid_coord,
                            key_of_entry, compare_keys, equal_keys);

                        const size_t lid = item.get_local_id(0);
                        if (item.get_global_id(0) >= N) {
                            return;
                        }

                        const VoxelLocalData local = local_voxel_data[lid];
                        global_reduction(
                            local, key_ptr, core_ptr, color_ptr, intensity_data_ptr, current_frame, max_probe, capacity,
                            [&](uint32_t add) { atomic_ref_uint32_t(voxel_ptr_counter[0]).fetch_add(add); }, has_rgb,
                            has_intensity);
                    });
            }
        });

        auto host_event = this->queue_.ptr->submit([&](sycl::handler& h) {
            h.depends_on(event);
            h.host_task([&]() { this->voxel_num_ = static_cast<size_t>(voxel_counter.at(0)); });
        });
        host_event.wait_and_throw();
    }

    void update_free_space(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) {
        const size_t point_count = cloud.size();
        if (point_count == 0U) {
            return;
        }

        const float sensor_x = sensor_pose.translation().x();
        const float sensor_y = sensor_pose.translation().y();
        const float sensor_z = sensor_pose.translation().z();
        const float scaled_origin_x = sensor_x * this->inv_voxel_size_;
        const float scaled_origin_y = sensor_y * this->inv_voxel_size_;
        const float scaled_origin_z = sensor_z * this->inv_voxel_size_;

        const int64_t origin_ix_host = static_cast<int64_t>(std::floor(scaled_origin_x));
        const int64_t origin_iy_host = static_cast<int64_t>(std::floor(scaled_origin_y));
        const int64_t origin_iz_host = static_cast<int64_t>(std::floor(scaled_origin_z));

        uint64_t origin_key = VoxelConstants::invalid_coord;
        const bool has_origin_key = this->grid_to_key(origin_ix_host, origin_iy_host, origin_iz_host, origin_key);

        sycl::event hit_event;
        shared_vector<uint32_t> origin_hit_flag(1, 0U, *this->queue_.ptr);

        if (has_origin_key) {
            // Detect if any points fall into the sensor-origin voxel to avoid clearing true hits.
            hit_event = this->queue_.ptr->submit([&](sycl::handler& h) {
                const auto trans = eigen_utils::to_sycl_vec(sensor_pose.matrix());
                auto hit_ptr = origin_hit_flag.data();
                auto hit_reduction = sycl::reduction(hit_ptr, sycl::maximum<uint32_t>());
                auto local_points_ptr = cloud.points_ptr();
                const float inv_voxel_size = this->inv_voxel_size_;
                const uint64_t origin_key_device = origin_key;

                h.parallel_for(sycl::range<1>(point_count), hit_reduction, [=](sycl::id<1> idx, auto& max_hit) {
                    const size_t i = idx[0];

                    const PointType local_point = local_points_ptr[i];
                    PointType world_point{};
                    transform::kernel::transform_point(local_point, world_point, trans);
                    const uint64_t voxel_hash = filter::kernel::compute_voxel_bit(world_point, inv_voxel_size);
                    if (voxel_hash == origin_key_device) {
                        max_hit.combine(1U);
                    }
                });
            });
        }

        shared_vector<uint32_t> expected_visit_counter(1, 0U, *this->queue_.ptr);
        auto estimate_event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const auto trans = eigen_utils::to_sycl_vec(sensor_pose.matrix());

            auto counter_ptr = expected_visit_counter.data();
            auto local_points_ptr = cloud.points_ptr();
            const float inv_voxel_size = this->inv_voxel_size_;

            auto reduction = sycl::reduction(counter_ptr, sycl::plus<uint32_t>());
            h.parallel_for(sycl::range<1>(point_count), reduction, [=](sycl::id<1> idx, auto& visit_acc) {
                const size_t i = idx[0];

                const PointType local_point = local_points_ptr[i];
                PointType world_point{};
                transform::kernel::transform_point(local_point, world_point, trans);

                const float world_x = world_point.x();
                const float world_y = world_point.y();
                const float world_z = world_point.z();

                const float diff_x = world_x - sensor_x;
                const float diff_y = world_y - sensor_y;
                const float diff_z = world_z - sensor_z;
                const float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                if (dist_sq <= std::numeric_limits<float>::epsilon()) {
                    return;
                }

                const float scaled_target_x = world_x * inv_voxel_size;
                const float scaled_target_y = world_y * inv_voxel_size;
                const float scaled_target_z = world_z * inv_voxel_size;

                const int64_t origin_ix = static_cast<int64_t>(sycl::floor(scaled_origin_x));
                const int64_t origin_iy = static_cast<int64_t>(sycl::floor(scaled_origin_y));
                const int64_t origin_iz = static_cast<int64_t>(sycl::floor(scaled_origin_z));
                const int64_t target_ix = static_cast<int64_t>(sycl::floor(scaled_target_x));
                const int64_t target_iy = static_cast<int64_t>(sycl::floor(scaled_target_y));
                const int64_t target_iz = static_cast<int64_t>(sycl::floor(scaled_target_z));

                // Calculate the number of traversed voxels with the sum of axis steps
                // between the origin and target grid coordinates. This mirrors the
                // traversal behavior in traverse_ray_exclusive_impl without performing
                // the full ray walk during estimation.
                const int64_t diff_ix = sycl::abs(target_ix - origin_ix);
                const int64_t diff_iy = sycl::abs(target_iy - origin_iy);
                const int64_t diff_iz = sycl::abs(target_iz - origin_iz);

                const int64_t traversal_count = diff_ix + diff_iy + diff_iz;
                // The total count includes the origin voxel plus the traversed voxels.
                const uint32_t local_count = (traversal_count > 0) ? static_cast<uint32_t>(traversal_count + 1) : 0U;

                visit_acc += local_count;
            });
        });

        size_t expected_voxel_visits = 0;
        auto expected_voxel_event = this->queue_.ptr->submit([&](sycl::handler& h) {
            h.depends_on(estimate_event);
            h.host_task([&]() { expected_voxel_visits = static_cast<size_t>(expected_visit_counter.at(0)); });
        });
        expected_voxel_event.wait_and_throw();

        if (expected_voxel_visits == 0U) {
            hit_event.wait_and_throw();
            return;
        }

        const size_t minimum_required = this->voxel_num_ + expected_voxel_visits;
        while (this->rehash_threshold_ < static_cast<float>(minimum_required) / static_cast<float>(this->capacity_)) {
            const size_t next_capacity = this->get_next_capacity_value();
            if (next_capacity <= this->capacity_) {
                break;
            }
            this->rehash(next_capacity);
        }

        shared_vector<uint32_t> voxel_counter(1, static_cast<uint32_t>(this->voxel_num_), *this->queue_.ptr);

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            h.depends_on(hit_event);

            const auto trans = eigen_utils::to_sycl_vec(sensor_pose.matrix());

            const float inv_voxel_size = this->inv_voxel_size_;

            auto key_ptr = this->key_ptr_->data();
            auto core_ptr = this->core_data_ptr_->data();
            auto color_ptr = this->color_data_ptr_->data();
            auto intensity_ptr = this->intensity_data_ptr_->data();
            auto counter_ptr = voxel_counter.data();
            auto origin_hit_ptr = origin_hit_flag.data();

            const auto local_points_ptr = cloud.points_ptr();
            const float log_miss = this->log_odds_miss_;

            const size_t max_probe = this->max_probe_length_;
            const size_t capacity = this->capacity_;
            const uint32_t current_frame = this->frame_index_;

            const auto has_rgb = this->has_rgb_data_;
            const auto has_intensity = this->has_intensity_data_;

            h.parallel_for(sycl::range<1>(point_count), [=](sycl::id<1> idx) {
                const size_t i = idx[0];

                const PointType local_point = local_points_ptr[i];
                PointType world_point;
                transform::kernel::transform_point(local_point, world_point, trans);

                const float world_x = world_point.x();
                const float world_y = world_point.y();
                const float world_z = world_point.z();

                const float diff_x = world_x - sensor_x;
                const float diff_y = world_y - sensor_y;
                const float diff_z = world_z - sensor_z;
                const float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                if (dist_sq <= std::numeric_limits<float>::epsilon()) {
                    return;
                }

                const float scaled_target_x = world_x * inv_voxel_size;
                const float scaled_target_y = world_y * inv_voxel_size;
                const float scaled_target_z = world_z * inv_voxel_size;

                const int64_t origin_ix = static_cast<int64_t>(sycl::floor(scaled_origin_x));
                const int64_t origin_iy = static_cast<int64_t>(sycl::floor(scaled_origin_y));
                const int64_t origin_iz = static_cast<int64_t>(sycl::floor(scaled_origin_z));
                const int64_t target_ix = static_cast<int64_t>(sycl::floor(scaled_target_x));
                const int64_t target_iy = static_cast<int64_t>(sycl::floor(scaled_target_y));
                const int64_t target_iz = static_cast<int64_t>(sycl::floor(scaled_target_z));

                auto accumulate_miss = [=](uint64_t key) {
                    VoxelLocalData local{};
                    local.voxel_idx = key;
                    local.core_acc.voxel_hash = key;
                    local.core_acc.log_odds_delta = log_miss;

                    global_reduction(
                        local, key_ptr, core_ptr, color_ptr, intensity_ptr, current_frame, max_probe, capacity,
                        [=](uint32_t add) { atomic_ref_uint32_t(counter_ptr[0]).fetch_add(add); }, has_rgb,
                        has_intensity);
                };

                const bool skip_origin_miss = has_origin_key ? origin_hit_ptr[0] != 0U : false;
                if (!skip_origin_miss && (origin_ix != target_ix || origin_iy != target_iy || origin_iz != target_iz)) {
                    uint64_t origin_key = VoxelConstants::invalid_coord;
                    if (grid_to_key_device(origin_ix, origin_iy, origin_iz, origin_key)) {
                        accumulate_miss(origin_key);
                    }
                }

                // Mirror the visibility traversal so that free-space updates exclude the hit voxel.
                traverse_ray_exclusive_impl(sensor_x, sensor_y, sensor_z, world_x, world_y, world_z, inv_voxel_size,
                                            [=](int64_t grid_x, int64_t grid_y, int64_t grid_z) {
                                                uint64_t key = VoxelConstants::invalid_coord;
                                                if (!grid_to_key_device(grid_x, grid_y, grid_z, key)) {
                                                    return true;
                                                }

                                                // Accumulate a miss observation for the current voxel on the device.
                                                accumulate_miss(key);
                                                return true;
                                            });
            });
        });

        auto host_event = this->queue_.ptr->submit([&](sycl::handler& h) {
            h.depends_on(event);
            h.host_task([&]() { this->voxel_num_ = static_cast<size_t>(voxel_counter.at(0)); });
        });
        host_event.wait_and_throw();
    }

    void apply_pending_log_odds() {
        const size_t N = this->capacity_;

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            auto key_ptr = this->key_ptr_->data();
            auto core_ptr = this->core_data_ptr_->data();
            const float min_log_odds = this->min_log_odds_;
            const float max_log_odds = this->max_log_odds_;
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                const size_t i = idx[0];
                if (key_ptr[i] == VoxelConstants::invalid_coord || key_ptr[i] == VoxelConstants::deleted_coord) {
                    return;
                }

                const float delta = core_ptr[i].pending_log_odds;
                if (delta == 0.0f) {
                    return;
                }

                float current_log_odds = core_ptr[i].log_odds;
                current_log_odds = sycl::fmax(min_log_odds, sycl::fmin(max_log_odds, current_log_odds + delta));
                core_ptr[i].log_odds = current_log_odds;
                core_ptr[i].pending_log_odds = 0.0f;
            });
        });
        event.wait_and_throw();
    }

    void prune_stale_voxels() {
        const size_t N = this->capacity_;
        const uint32_t current_frame = this->frame_index_;
        const uint32_t stale_threshold = this->stale_frame_threshold_;

        if (current_frame < stale_threshold) {
            return;
        }

        shared_vector<uint32_t> voxel_counter(1, 0U, *this->queue_.ptr);

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            auto key_ptr = this->key_ptr_->data();
            auto core_ptr = this->core_data_ptr_->data();
            auto color_ptr = this->color_data_ptr_->data();
            auto intensity_ptr = this->intensity_data_ptr_->data();

            auto counter_reduction = sycl::reduction(voxel_counter.data(), sycl::plus<uint32_t>());

            h.parallel_for(sycl::range<1>(N), counter_reduction, [=](sycl::id<1> idx, auto& counter) {
                const size_t i = idx[0];
                if (key_ptr[i] == VoxelConstants::invalid_coord || key_ptr[i] == VoxelConstants::deleted_coord) {
                    return;
                }

                const auto last_updated = core_ptr[i].last_updated;
                // Evict voxels whose last update frame is older than the allowed threshold.
                const bool is_stale = (current_frame - last_updated) > stale_threshold;
                if (is_stale) {
                    key_ptr[i] = VoxelConstants::deleted_coord;
                    core_ptr[i] = VoxelCoreData{};
                    color_ptr[i] = VoxelColorData{};
                    intensity_ptr[i] = VoxelIntensityData{};
                    return;
                }

                counter += 1U;
            });
        });
        auto host_event = this->queue_.ptr->submit([&](sycl::handler& h) {
            h.depends_on(event);
            h.host_task([&]() { this->voxel_num_ = static_cast<size_t>(voxel_counter.at(0)); });
        });
        host_event.wait_and_throw();
    }

    /// @brief Template helper for extracting points with prefix sum or fetch_add
    /// @tparam GenerateFlagsFunc Functor for generating valid flags (NVIDIA path)
    /// @tparam WriteOutputFunc Functor for writing output data (NVIDIA path)
    /// @tparam FetchAddFunc Functor for fetch_add path (non-NVIDIA)
    template<typename GenerateFlagsFunc, typename WriteOutputFunc, typename FetchAddFunc>
    void extract_points_with_prefix_sum(PointCloudShared& result, size_t estimated_size,
                                        GenerateFlagsFunc&& generate_flags_func,
                                        WriteOutputFunc&& write_output_func, FetchAddFunc&& fetch_add_func) {
        const size_t N = this->capacity_;
        const bool is_nvidia = this->queue_.is_nvidia();
        size_t filtered_voxel_count = 0;

        result.resize_points(estimated_size);
        if (this->has_rgb_data_) {
            result.resize_rgb(estimated_size);
        }
        if (this->has_intensity_data_) {
            result.resize_intensities(estimated_size);
        }

        if (is_nvidia) {
            // NVIDIA GPU: Use prefix sum approach
            // Step 1: Compute valid flags
            if (this->valid_flags_ptr_->size() < N) {
                this->valid_flags_ptr_->resize(N);
            }

            this->queue_.ptr->submit(generate_flags_func).wait_and_throw();

            // Step 2: Compute prefix sum
            // Prefix sum guarantees: when valid_flags[i] == 1, prefix_sum_ptr[i] >= 1
            // The output index is calculated as prefix_sum_ptr[i] - 1 for inclusive scan results
            filtered_voxel_count = this->prefix_sum_->compute(*this->valid_flags_ptr_);

            // Step 3: Write output using prefix sum indices
            this->queue_.ptr->submit(write_output_func).wait_and_throw();
        } else {
            // Non-NVIDIA: Use fetch_add approach
            shared_vector<uint32_t> counter(1, 0U, *this->queue_.ptr);
            auto event = this->queue_.ptr->submit([&, fetch_add_func](sycl::handler& h) {
                fetch_add_func(h, counter.data());
            });
            event.wait_and_throw();
            filtered_voxel_count = static_cast<size_t>(counter.at(0));
        }

        // Resize to actual count
        result.resize_points(filtered_voxel_count);
        if (this->has_rgb_data_) {
            result.resize_rgb(filtered_voxel_count);
        }
        if (this->has_intensity_data_) {
            result.resize_intensities(filtered_voxel_count);
        }
    }

    void extract_occupied_points_impl(PointCloudShared& result, const Eigen::Vector3f& sensor_position,
                                      const float max_distance) {
        const size_t N = this->capacity_;

        // Lambda for generating valid flags (NVIDIA path)
        auto generate_flags = [&](sycl::handler& h) {
            const size_t work_group_size = this->queue_.get_work_group_size();
            const size_t global_size = this->queue_.get_global_size(N);

            auto valid_flags = this->valid_flags_ptr_->data();
            auto key_ptr = this->key_ptr_->data();
            auto core_ptr = this->core_data_ptr_->data();

            const float threshold = this->occupancy_threshold_log_odds_;
            const float sensor_x = sensor_position.x();
            const float sensor_y = sensor_position.y();
            const float sensor_z = sensor_position.z();
            const float max_dist = max_distance;

            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) {
                    return;
                }

                const uint64_t key = key_ptr[i];
                if (key == VoxelConstants::invalid_coord || key == VoxelConstants::deleted_coord) {
                    valid_flags[i] = 0U;
                    return;
                }

                const VoxelCoreData& core = core_ptr[i];
                if (core.hit_count == 0U || core.log_odds < threshold) {
                    valid_flags[i] = 0U;
                    return;
                }

                const float inv_count = 1.0f / static_cast<float>(core.hit_count);
                const float cx = core.sum_x * inv_count;
                const float cy = core.sum_y * inv_count;
                const float cz = core.sum_z * inv_count;

                const float dx = sycl::fabs(cx - sensor_x);
                const float dy = sycl::fabs(cy - sensor_y);
                const float dz = sycl::fabs(cz - sensor_z);
                if (sycl::fmax(sycl::fmax(dx, dy), dz) > max_dist) {
                    valid_flags[i] = 0U;
                    return;
                }

                valid_flags[i] = 1U;
            });
        };

        // Lambda for writing output (NVIDIA path)
        auto write_output = [&](sycl::handler& h) {
            const size_t work_group_size = this->queue_.get_work_group_size();
            const size_t global_size = this->queue_.get_global_size(N);

            auto valid_flags = this->valid_flags_ptr_->data();
            auto prefix_sum_ptr = this->prefix_sum_->get_prefix_sum().data();
            auto core_ptr = this->core_data_ptr_->data();
            auto color_ptr = this->color_data_ptr_->data();
            auto intensity_data_ptr = this->intensity_data_ptr_->data();

            auto points_ptr = result.points_ptr();
            auto rgb_ptr = this->has_rgb_data_ ? result.rgb_ptr() : static_cast<RGBType*>(nullptr);
            auto intensity_ptr =
                this->has_intensity_data_ ? result.intensities_ptr() : static_cast<float*>(nullptr);

            const bool has_rgb = this->has_rgb_data_;
            const bool has_intensity = this->has_intensity_data_;

            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N || valid_flags[i] == 0U) {
                    return;
                }

                const size_t output_idx = prefix_sum_ptr[i] - 1;
                const VoxelCoreData& core = core_ptr[i];
                const float inv_count = 1.0f / static_cast<float>(core.hit_count);
                const float cx = core.sum_x * inv_count;
                const float cy = core.sum_y * inv_count;
                const float cz = core.sum_z * inv_count;

                points_ptr[output_idx].x() = cx;
                points_ptr[output_idx].y() = cy;
                points_ptr[output_idx].z() = cz;
                points_ptr[output_idx].w() = 1.0f;

                if (has_rgb && rgb_ptr) {
                    const VoxelColorData& color = color_ptr[i];
                    if (core.hit_count > 0U) {
                        rgb_ptr[output_idx].x() = color.sum_r * inv_count;
                        rgb_ptr[output_idx].y() = color.sum_g * inv_count;
                        rgb_ptr[output_idx].z() = color.sum_b * inv_count;
                        rgb_ptr[output_idx].w() = color.sum_a * inv_count;
                    } else {
                        rgb_ptr[output_idx].setZero();
                    }
                }

                if (has_intensity && intensity_ptr) {
                    const VoxelIntensityData& intensity_data = intensity_data_ptr[i];
                    if (core.hit_count > 0U) {
                        intensity_ptr[output_idx] = intensity_data.sum_intensity * inv_count;
                    } else {
                        intensity_ptr[output_idx] = 0.0f;
                    }
                }
            });
        };

        // Lambda for fetch_add path (non-NVIDIA)
        auto fetch_add_kernel = [&](sycl::handler& h, uint32_t* counter_ptr) {
            auto key_ptr = this->key_ptr_->data();
            auto core_ptr = this->core_data_ptr_->data();
            auto color_ptr = this->color_data_ptr_->data();
            auto intensity_data_ptr = this->intensity_data_ptr_->data();

            auto points_ptr = result.points_ptr();
            auto rgb_ptr = this->has_rgb_data_ ? result.rgb_ptr() : static_cast<RGBType*>(nullptr);
            auto intensity_ptr = this->has_intensity_data_ ? result.intensities_ptr() : static_cast<float*>(nullptr);

            const float threshold = this->occupancy_threshold_log_odds_;
            const float sensor_x = sensor_position.x();
            const float sensor_y = sensor_position.y();
            const float sensor_z = sensor_position.z();
            const float max_dist = max_distance;
            const bool has_rgb = this->has_rgb_data_;
            const bool has_intensity = this->has_intensity_data_;

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                const size_t i = idx[0];
                if (key_ptr[i] == VoxelConstants::invalid_coord || key_ptr[i] == VoxelConstants::deleted_coord) {
                    return;
                }

                const VoxelCoreData& core = core_ptr[i];
                if (core.hit_count == 0U || core.log_odds < threshold) {
                    return;
                }

                const float inv_count = 1.0f / static_cast<float>(core.hit_count);
                const float cx = core.sum_x * inv_count;
                const float cy = core.sum_y * inv_count;
                const float cz = core.sum_z * inv_count;

                const float dx = sycl::fabs(cx - sensor_x);
                const float dy = sycl::fabs(cy - sensor_y);
                const float dz = sycl::fabs(cz - sensor_z);
                if (sycl::fmax(sycl::fmax(dx, dy), dz) > max_dist) {
                    return;
                }

                const uint32_t index = atomic_ref_uint32_t(counter_ptr[0]).fetch_add(1U);
                points_ptr[index].x() = cx;
                points_ptr[index].y() = cy;
                points_ptr[index].z() = cz;
                points_ptr[index].w() = 1.0f;

                if (has_rgb && rgb_ptr) {
                    const VoxelColorData& color = color_ptr[i];
                    if (core.hit_count > 0U) {
                        rgb_ptr[index].x() = color.sum_r * inv_count;
                        rgb_ptr[index].y() = color.sum_g * inv_count;
                        rgb_ptr[index].z() = color.sum_b * inv_count;
                        rgb_ptr[index].w() = color.sum_a * inv_count;
                    } else {
                        rgb_ptr[index].setZero();
                    }
                }

                if (has_intensity && intensity_ptr) {
                    const VoxelIntensityData& intensity_data = intensity_data_ptr[i];
                    if (core.hit_count > 0U) {
                        intensity_ptr[index] = intensity_data.sum_intensity * inv_count;
                    } else {
                        intensity_ptr[index] = 0.0f;
                    }
                }
            });
        };

        this->extract_points_with_prefix_sum(result, this->voxel_num_, generate_flags, write_output,
                                             fetch_add_kernel);
    }

    size_t compute_work_group_size() const {
        const size_t max_work_group_size =
            this->queue_.get_device().get_info<sycl::info::device::max_work_group_size>();
        const size_t compute_units = this->queue_.get_device().get_info<sycl::info::device::max_compute_units>();
        if (this->queue_.is_nvidia()) {
            return std::min(max_work_group_size, size_t{64});
        }
        if (this->queue_.is_intel() && this->queue_.is_gpu()) {
            return std::min(max_work_group_size, compute_units * size_t{8});
        }
        if (this->queue_.is_cpu()) {
            return std::min(max_work_group_size, compute_units * size_t{100});
        }
        return std::min<size_t>(128, max_work_group_size);
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

    shared_vector_ptr<uint64_t> key_ptr_ = nullptr;
    shared_vector_ptr<VoxelCoreData> core_data_ptr_ = nullptr;
    shared_vector_ptr<VoxelColorData> color_data_ptr_ = nullptr;
    shared_vector_ptr<VoxelIntensityData> intensity_data_ptr_ = nullptr;

    shared_vector_ptr<uint8_t> valid_flags_ptr_ = nullptr;
    common::PrefixSum::Ptr prefix_sum_ = nullptr;
};

}  // namespace mapping
}  // namespace algorithms
}  // namespace sycl_points
