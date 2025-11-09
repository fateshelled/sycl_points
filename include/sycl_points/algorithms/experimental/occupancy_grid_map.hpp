#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <sycl_points/algorithms/common/voxel_constants.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/algorithms/voxel_downsampling.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>
#include <vector>

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
        this->allocate_storage();
        // Preallocate the buffer used to store world-frame points before hashing.
        this->clear();
    }

    /// @brief Reset the map data.
    void clear() {
        this->initialize_storage();
        this->voxel_num_ = 0;
        this->has_rgb_data_ = false;
        this->has_intensity_data_ = false;
        this->frame_index_ = 0;
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
        const VoxelData* voxel = this->find_voxel(key);
        if (!voxel) {
            return 0.5f;
        }
        return this->log_odds_to_probability(voxel->log_odds);
    }

    /// @brief Configure the log-odds increment applied on hits.
    void set_log_odds_hit(const float value) { this->log_odds_hit_ = value; }

    /// @brief Configure the log-odds decrement applied on misses.
    void set_log_odds_miss(const float value) { this->log_odds_miss_ = value; }

    /// @brief Set the visibility decay range in meters.
    void set_visibility_decay_range(const float distance) {
        if (!(distance >= 0.0f)) {
            throw std::invalid_argument("distance must be non-negative.");
        }
        this->visibility_decay_range_ = distance;
    }

    /// @brief Get the visibility decay range in meters.
    float visibility_decay_range() const { return this->visibility_decay_range_; }

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

    /// @brief Insert a point cloud captured at the given pose.
    /// @param cloud Point cloud in the sensor frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    void add_point_cloud(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) {
        if (!cloud.points || cloud.points->empty()) {
            return;
        }

        const size_t N = cloud.size();

        this->ensure_rehash();

        const bool has_rgb = cloud.has_rgb();
        const bool has_intensity = cloud.has_intensity();
        this->has_rgb_data_ = this->has_rgb_data_ || has_rgb;
        this->has_intensity_data_ = this->has_intensity_data_ || has_intensity;

        this->integrate_points(cloud, sensor_pose, has_rgb, has_intensity);

        this->apply_pending_log_odds();

        if (this->log_odds_miss_ != 0.0f) {
            this->apply_visibility_decay(sensor_pose.translation());
        }

        ++this->frame_index_;
    }

    /// @brief Extract occupied voxels that are visible from the sensor pose.
    /// @param result Output point cloud in the map frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    /// @param max_distance Maximum visibility distance in meters.
    void downsampling(PointCloudShared& result, const Eigen::Isometry3f& sensor_pose,
                      const float max_distance = 100.0f) const {
        result.resize_points(0);
        result.resize_rgb(0);
        result.resize_intensities(0);

        if (this->voxel_num_ == 0) {
            return;
        }

        this->export_visible_voxels(result, sensor_pose.translation(), max_distance);
    }

private:
    using atomic_ref_float = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint32_t = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint64_t = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;

    struct VoxelData {
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_a = 0.0f;
        float sum_intensity = 0.0f;
        float log_odds = 0.0f;
        float pending_log_odds = 0.0f;
        uint32_t hit_count = 0U;
        uint32_t color_count = 0U;
        uint32_t intensity_count = 0U;
        uint32_t last_observed = 0U;
    };

    struct VoxelAccumulator {
        uint64_t voxel_hash = VoxelConstants::invalid_coord;
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_a = 0.0f;
        float sum_intensity = 0.0f;
        float log_odds_delta = 0.0f;
        uint32_t hit_increment = 0U;
        uint32_t color_count = 0U;
        uint32_t intensity_count = 0U;
    };

    struct VoxelLocalData {
        uint64_t voxel_idx = VoxelConstants::invalid_coord;
        VoxelAccumulator acc;
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

    const VoxelData* find_voxel(const uint64_t key) const {
        if (!this->key_ptr_) {
            return nullptr;
        }
        for (size_t j = 0; j < this->max_probe_length_; ++j) {
            const size_t slot = this->compute_slot_id(key, j, this->capacity_);
            const uint64_t stored_key = this->key_ptr_.get()[slot];
            if (stored_key == key) {
                return &this->data_ptr_.get()[slot];
            }
            if (stored_key == VoxelConstants::invalid_coord) {
                break;
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

    void allocate_storage() {
        this->key_ptr_ = std::shared_ptr<uint64_t>(sycl::malloc_shared<uint64_t>(this->capacity_, *this->queue_.ptr),
                                                   [&](uint64_t* ptr) { sycl::free(ptr, *this->queue_.ptr); });
        this->data_ptr_ = std::shared_ptr<VoxelData>(sycl::malloc_shared<VoxelData>(this->capacity_, *this->queue_.ptr),
                                                     [&](VoxelData* ptr) { sycl::free(ptr, *this->queue_.ptr); });

        this->queue_.set_accessed_by_device(this->key_ptr_.get(), this->capacity_);
        this->queue_.set_accessed_by_device(this->data_ptr_.get(), this->capacity_);
    }

    void initialize_storage() {
        // Reset the hash table content before the next integration round.
        sycl_utils::events evs;
        evs += this->queue_.ptr->fill<uint64_t>(this->key_ptr_.get(), VoxelConstants::invalid_coord, this->capacity_);
        evs += this->queue_.ptr->fill<VoxelData>(this->data_ptr_.get(), VoxelData{}, this->capacity_);
        evs.wait();
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
        auto old_data = this->data_ptr_;

        this->capacity_ = new_capacity;
        this->allocate_storage();
        this->initialize_storage();

        size_t voxel_count = 0;
        for (size_t i = 0; i < old_capacity; ++i) {
            const uint64_t key = old_keys.get()[i];
            if (key == VoxelConstants::invalid_coord) {
                continue;
            }

            const VoxelData& src = old_data.get()[i];
            for (size_t probe = 0; probe < this->max_probe_length_; ++probe) {
                const size_t slot = this->compute_slot_id(key, probe, this->capacity_);
                uint64_t& dst_key = this->key_ptr_.get()[slot];
                if (dst_key == VoxelConstants::invalid_coord) {
                    dst_key = key;
                    this->data_ptr_.get()[slot] = src;
                    ++voxel_count;
                    break;
                }
            }
        }

        this->voxel_num_ = voxel_count;
    }

    template <bool Aggregate>
    static void local_reduction(VoxelLocalData* local_data, const PointType* local_points, const RGBType* rgb_ptr,
                                const float* intensity_ptr, const bool has_rgb, const bool has_intensity,
                                const size_t point_num, const size_t wg_size, const size_t wg_size_power_of_2,
                                const std::array<sycl::vec<float, 4>, 4>& trans, const float voxel_size_inv,
                                const float log_odds_hit, const sycl::nd_item<1>& item) {
        const size_t local_id = item.get_local_id(0);
        const size_t global_id = item.get_global_id(0);

        if (global_id < point_num) {
            const PointType local_point = local_points[global_id];
            PointType world_point;
            transform::kernel::transform_point(local_point, world_point, trans);
            const uint64_t voxel_hash = filter::kernel::compute_voxel_bit(world_point, voxel_size_inv);

            local_data[local_id].voxel_idx = voxel_hash;
            local_data[local_id].acc.voxel_hash = voxel_hash;
            local_data[local_id].acc.sum_x = world_point.x();
            local_data[local_id].acc.sum_y = world_point.y();
            local_data[local_id].acc.sum_z = world_point.z();
            local_data[local_id].acc.hit_increment = 1U;
            local_data[local_id].acc.log_odds_delta = log_odds_hit;
            local_data[local_id].acc.sum_r = 0.0f;
            local_data[local_id].acc.sum_g = 0.0f;
            local_data[local_id].acc.sum_b = 0.0f;
            local_data[local_id].acc.sum_a = 0.0f;
            local_data[local_id].acc.sum_intensity = 0.0f;
            local_data[local_id].acc.color_count = 0U;
            local_data[local_id].acc.intensity_count = 0U;

            if (has_rgb && rgb_ptr) {
                const auto color = rgb_ptr[global_id];
                local_data[local_id].acc.sum_r = color.x();
                local_data[local_id].acc.sum_g = color.y();
                local_data[local_id].acc.sum_b = color.z();
                local_data[local_id].acc.sum_a = color.w();
                local_data[local_id].acc.color_count = 1U;
            }

            if (has_intensity && intensity_ptr) {
                local_data[local_id].acc.sum_intensity = intensity_ptr[global_id];
                local_data[local_id].acc.intensity_count = 1U;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        if constexpr (Aggregate) {
            const size_t group_id = item.get_group(0);
            const size_t group_offset = group_id * wg_size;
            const size_t remaining_points = (point_num > group_offset) ? point_num - group_offset : 0;
            // The active size tells the sorter how many valid elements are present in the current work-group.
            const size_t active_size = std::min(wg_size, remaining_points);

            // sort within work group by voxel index
            bitonic_sort_local_data(local_data, active_size, wg_size_power_of_2, item);
            // reduction
            reduction_sorted_local_data(local_data, active_size, item);
        }
    }

    /// @brief Bitonic sort that works correctly with any work group size
    /// @details Uses virtual infinity padding to handle non-power-of-2 sizes
    SYCL_EXTERNAL static void bitonic_sort_local_data(VoxelLocalData* data, size_t size, size_t size_power_of_2,
                                                      const sycl::nd_item<1>& item) {
        const size_t local_id = item.get_local_id(0);

        if (size <= 1) return;

        // Bitonic sort with virtual infinity padding
        for (size_t k = 2; k <= size_power_of_2; k *= 2) {
            for (size_t j = k / 2; j > 0; j /= 2) {
                const size_t i = local_id;
                const size_t ixj = i ^ j;

                if (ixj > i && i < size_power_of_2) {
                    // Determine if we're in ascending or descending phase
                    const bool ascending = ((i & k) == 0);

                    // Get values (use infinity for out-of-bounds elements)
                    const uint64_t val_i = (i < size) ? data[i].voxel_idx : VoxelConstants::invalid_coord;
                    const uint64_t val_ixj = (ixj < size) ? data[ixj].voxel_idx : VoxelConstants::invalid_coord;

                    // Determine if swap is needed based on virtual values
                    const bool should_swap = (val_i > val_ixj) == ascending;

                    // Perform actual swap only if both indices are within real data
                    if (should_swap && i < size && ixj < size) {
                        std::swap(data[i], data[ixj]);
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    static void reduction_sorted_local_data(VoxelLocalData* data, const size_t size, const sycl::nd_item<1>& item) {
        const size_t local_id = item.get_local_id(0);
        const uint64_t current_voxel = (local_id < size) ? data[local_id].voxel_idx : VoxelConstants::invalid_coord;

        const bool is_segment_start = (current_voxel != VoxelConstants::invalid_coord) &&
                                      ((local_id == 0) || (data[local_id - 1].voxel_idx != current_voxel));

        if (is_segment_start) {
            for (size_t i = local_id + 1; i < size && data[i].voxel_idx == current_voxel; ++i) {
                data[i].voxel_idx = VoxelConstants::invalid_coord;
                data[local_id].acc.sum_x += data[i].acc.sum_x;
                data[local_id].acc.sum_y += data[i].acc.sum_y;
                data[local_id].acc.sum_z += data[i].acc.sum_z;
                data[local_id].acc.hit_increment += data[i].acc.hit_increment;
                data[local_id].acc.log_odds_delta += data[i].acc.log_odds_delta;
                data[local_id].acc.sum_r += data[i].acc.sum_r;
                data[local_id].acc.sum_g += data[i].acc.sum_g;
                data[local_id].acc.sum_b += data[i].acc.sum_b;
                data[local_id].acc.sum_a += data[i].acc.sum_a;
                data[local_id].acc.color_count += data[i].acc.color_count;
                data[local_id].acc.sum_intensity += data[i].acc.sum_intensity;
                data[local_id].acc.intensity_count += data[i].acc.intensity_count;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    template <typename CounterFunc>
    static void global_reduction(const VoxelLocalData& data, uint64_t* key_ptr, VoxelData* voxel_ptr,
                                 const uint32_t current_frame, const size_t max_probe, const size_t capacity,
                                 CounterFunc counter) {
        const uint64_t voxel_hash = data.voxel_idx;
        if (voxel_hash == VoxelConstants::invalid_coord) {
            return;
        }

        for (size_t probe = 0; probe < max_probe; ++probe) {
            const size_t slot_idx = compute_slot_id(voxel_hash, probe, capacity);
            uint64_t expected = VoxelConstants::invalid_coord;
            if (atomic_ref_uint64_t(key_ptr[slot_idx]).compare_exchange_strong(expected, voxel_hash)) {
                counter(1U);
                atomic_add_voxel_data(data.acc, voxel_ptr[slot_idx]);
                atomic_ref_uint32_t(voxel_ptr[slot_idx].last_observed).store(current_frame);
                break;
            }
            if (expected == voxel_hash) {
                atomic_add_voxel_data(data.acc, voxel_ptr[slot_idx]);
                atomic_ref_uint32_t(voxel_ptr[slot_idx].last_observed).store(current_frame);
                break;
            }
        }
    }

    static void atomic_add_voxel_data(const VoxelAccumulator& src, VoxelData& dst) {
        atomic_ref_float(dst.sum_x).fetch_add(src.sum_x);
        atomic_ref_float(dst.sum_y).fetch_add(src.sum_y);
        atomic_ref_float(dst.sum_z).fetch_add(src.sum_z);

        atomic_ref_uint32_t(dst.hit_count).fetch_add(src.hit_increment);
        atomic_ref_float(dst.pending_log_odds).fetch_add(src.log_odds_delta);

        if (src.color_count > 0U) {
            atomic_ref_float(dst.sum_r).fetch_add(src.sum_r);
            atomic_ref_float(dst.sum_g).fetch_add(src.sum_g);
            atomic_ref_float(dst.sum_b).fetch_add(src.sum_b);
            atomic_ref_float(dst.sum_a).fetch_add(src.sum_a);
            atomic_ref_uint32_t(dst.color_count).fetch_add(src.color_count);
        }

        if (src.intensity_count > 0U) {
            atomic_ref_float(dst.sum_intensity).fetch_add(src.sum_intensity);
            atomic_ref_uint32_t(dst.intensity_count).fetch_add(src.intensity_count);
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
            auto key_ptr = this->key_ptr_.get();
            auto voxel_ptr = this->data_ptr_.get();
            const auto voxel_size_inv = this->inv_voxel_size_;
            const auto current_frame = this->frame_index_;
            const auto max_probe = this->max_probe_length_;
            const auto capacity = this->capacity_;
            const float log_odds_hit = this->log_odds_hit_;

            if (this->queue_.is_nvidia()) {
                auto reduction = sycl::reduction(voxel_counter.data(), sycl::plus<uint32_t>());
                h.parallel_for(  //
                    sycl::nd_range<1>(global_size, local_size), reduction,
                    [=](sycl::nd_item<1> item, auto& voxel_num_arg) {
                        local_reduction<true>(local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(),
                                              point_ptr, rgb_ptr, intensity_ptr, has_rgb, has_intensity, N, local_size,
                                              power_of_2, trans, voxel_size_inv, log_odds_hit, item);

                        const size_t lid = item.get_local_id(0);
                        if (item.get_global_id(0) >= N) {
                            return;
                        }

                        const VoxelLocalData local = local_voxel_data[lid];
                        global_reduction(local, key_ptr, voxel_ptr, current_frame, max_probe, capacity,
                                         [&](uint32_t add) { voxel_num_arg += add; });
                    });
            } else {
                auto voxel_ptr_counter = voxel_counter.data();
                h.parallel_for(  //
                    sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
                        local_reduction<true>(local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(),
                                              point_ptr, rgb_ptr, intensity_ptr, has_rgb, has_intensity, N, local_size,
                                              power_of_2, trans, voxel_size_inv, log_odds_hit, item);

                        const size_t lid = item.get_local_id(0);
                        if (item.get_global_id(0) >= N) {
                            return;
                        }

                        const VoxelLocalData local = local_voxel_data[lid];
                        global_reduction(
                            local, key_ptr, voxel_ptr, current_frame, max_probe, capacity,
                            [&](uint32_t add) { atomic_ref_uint32_t(voxel_ptr_counter[0]).fetch_add(add); });
                    });
            }
        });

        event.wait();
        this->voxel_num_ = static_cast<size_t>(voxel_counter.at(0));
    }

    void apply_pending_log_odds() {
        const size_t N = this->capacity_;

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            auto key_ptr = this->key_ptr_.get();
            auto voxel_ptr = this->data_ptr_.get();
            const float min_log_odds = this->min_log_odds_;
            const float max_log_odds = this->max_log_odds_;
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                const size_t i = idx[0];
                if (key_ptr[i] == VoxelConstants::invalid_coord) {
                    return;
                }

                const float delta = voxel_ptr[i].pending_log_odds;
                if (delta == 0.0f) {
                    return;
                }

                float current_log_odds = voxel_ptr[i].log_odds;
                current_log_odds = sycl::fmax(min_log_odds, sycl::fmin(max_log_odds, current_log_odds + delta));
                voxel_ptr[i].log_odds = current_log_odds;
                voxel_ptr[i].pending_log_odds = 0.0f;
            });
        });
        event.wait();
    }

    void apply_visibility_decay(const Eigen::Vector3f& sensor_position) {
        const size_t N = this->capacity_;
        const float max_distance_sq = this->visibility_decay_range_ * this->visibility_decay_range_;
        const uint32_t current_frame = this->frame_index_;
        const float log_odds_miss = this->log_odds_miss_;
        const float min_log_odds = this->min_log_odds_;

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            auto key_ptr = this->key_ptr_.get();
            auto voxel_ptr = this->data_ptr_.get();
            const float sensor_x = sensor_position.x();
            const float sensor_y = sensor_position.y();
            const float sensor_z = sensor_position.z();
            const float max_dist_sq = max_distance_sq;
            const float log_miss = log_odds_miss;
            const float min_log = min_log_odds;
            const float max_log = this->max_log_odds_;

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                const size_t i = idx[0];
                if (key_ptr[i] == VoxelConstants::invalid_coord) {
                    return;
                }

                const VoxelData& data = voxel_ptr[i];
                if (data.last_observed == current_frame) {
                    return;
                }

                if (data.hit_count == 0U) {
                    return;
                }

                const float inv_count = 1.0f / static_cast<float>(data.hit_count);
                const float cx = data.sum_x * inv_count;
                const float cy = data.sum_y * inv_count;
                const float cz = data.sum_z * inv_count;

                const float dx = cx - sensor_x;
                const float dy = cy - sensor_y;
                const float dz = cz - sensor_z;
                const float dist_sq = dx * dx + dy * dy + dz * dz;
                if (dist_sq > max_dist_sq) {
                    return;
                }

                float updated = data.log_odds + log_miss;
                updated = sycl::fmax(min_log, sycl::fmin(max_log, updated));
                voxel_ptr[i].log_odds = updated;
            });
        });
        event.wait();
    }

    void export_visible_voxels(PointCloudShared& result, const Eigen::Vector3f& sensor_position,
                               const float max_distance) const {
        const size_t N = this->capacity_;
        const float max_distance_sq = max_distance * max_distance;
        shared_vector<uint32_t> counter(1, 0U, *this->queue_.ptr);

        result.resize_points(this->voxel_num_);
        if (this->has_rgb_data_) {
            result.resize_rgb(this->voxel_num_);
        }
        if (this->has_intensity_data_) {
            result.resize_intensities(this->voxel_num_);
        }

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            auto key_ptr = this->key_ptr_.get();
            auto voxel_ptr = this->data_ptr_.get();

            auto points_ptr = result.points_ptr();
            auto rgb_ptr = this->has_rgb_data_ ? result.rgb_ptr() : static_cast<RGBType*>(nullptr);
            auto intensity_ptr = this->has_intensity_data_ ? result.intensities_ptr() : static_cast<float*>(nullptr);

            const float threshold = this->occupancy_threshold_log_odds_;
            const float sensor_x = sensor_position.x();
            const float sensor_y = sensor_position.y();
            const float sensor_z = sensor_position.z();
            const float max_dist_sq = max_distance_sq;
            const bool has_rgb = this->has_rgb_data_;
            const bool has_intensity = this->has_intensity_data_;
            auto counter_ptr = counter.data();

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                const size_t i = idx[0];
                if (key_ptr[i] == VoxelConstants::invalid_coord) {
                    return;
                }

                const VoxelData& data = voxel_ptr[i];
                if (data.hit_count == 0U || data.log_odds < threshold) {
                    return;
                }

                const float inv_count = 1.0f / static_cast<float>(data.hit_count);
                const float cx = data.sum_x * inv_count;
                const float cy = data.sum_y * inv_count;
                const float cz = data.sum_z * inv_count;

                const float dx = cx - sensor_x;
                const float dy = cy - sensor_y;
                const float dz = cz - sensor_z;
                const float dist_sq = dx * dx + dy * dy + dz * dz;
                if (dist_sq > max_dist_sq) {
                    return;
                }

                const uint32_t index = atomic_ref_uint32_t(counter_ptr[0]).fetch_add(1U);
                points_ptr[index].x() = cx;
                points_ptr[index].y() = cy;
                points_ptr[index].z() = cz;
                points_ptr[index].w() = 1.0f;

                if (has_rgb && rgb_ptr) {
                    if (data.color_count > 0U) {
                        const float inv_color = 1.0f / static_cast<float>(data.color_count);
                        rgb_ptr[index].x() = data.sum_r * inv_color;
                        rgb_ptr[index].y() = data.sum_g * inv_color;
                        rgb_ptr[index].z() = data.sum_b * inv_color;
                        rgb_ptr[index].w() = data.sum_a * inv_color;
                    } else {
                        rgb_ptr[index].setZero();
                    }
                }

                if (has_intensity && intensity_ptr) {
                    if (data.intensity_count > 0U) {
                        const float inv_intensity = 1.0f / static_cast<float>(data.intensity_count);
                        intensity_ptr[index] = data.sum_intensity * inv_intensity;
                    } else {
                        intensity_ptr[index] = 0.0f;
                    }
                }
            });
        });

        event.wait();

        const uint32_t final_count = counter.at(0);
        result.resize_points(final_count);
        if (this->has_rgb_data_) {
            result.resize_rgb(final_count);
        }
        if (this->has_intensity_data_) {
            result.resize_intensities(final_count);
        }
    }

    size_t compute_work_group_size() const {
        const size_t max_work_group_size =
            this->queue_.get_device().get_info<sycl::info::device::max_work_group_size>();
        const size_t compute_units = this->queue_.get_device().get_info<sycl::info::device::max_compute_units>();
        if (this->queue_.is_nvidia()) {
            return std::min(max_work_group_size, size_t{64});
        }
        if (this->queue_.is_intel() && this->queue_.is_gpu()) {
            return std::min(max_work_group_size, compute_units * size_t{16});
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
    float visibility_decay_range_ = 30.0f;
    bool has_rgb_data_ = false;
    bool has_intensity_data_ = false;
    uint32_t frame_index_ = 0U;

    inline static constexpr std::array<size_t, 11> kCapacityCandidates = {
        30029, 60013, 120011, 240007, 480013, 960017, 1920001, 3840007, 7680017, 15360013, 30720007};
    size_t capacity_ = kCapacityCandidates[0];
    size_t voxel_num_ = 0;
    const size_t max_probe_length_ = 100;
    float rehash_threshold_ = 0.7f;

    std::shared_ptr<uint64_t> key_ptr_ = nullptr;
    std::shared_ptr<VoxelData> data_ptr_ = nullptr;
};

}  // namespace mapping
}  // namespace algorithms
}  // namespace sycl_points
