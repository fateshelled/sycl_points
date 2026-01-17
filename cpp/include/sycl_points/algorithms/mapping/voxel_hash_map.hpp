#pragma once

#include <Eigen/Core>
#include <array>
#include <iostream>
#include <stdexcept>

#include "sycl_points/algorithms/common/prefix_sum.hpp"
#include "sycl_points/algorithms/common/transform.hpp"
#include "sycl_points/algorithms/common/voxel_constants.hpp"
#include "sycl_points/algorithms/common/workgroup_utils.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace mapping {

// Reuse the voxel hashing utilities defined for filtering algorithms.
namespace kernel = sycl_points::algorithms::filter::kernel;

class VoxelHashMap {
public:
    using Ptr = std::shared_ptr<VoxelHashMap>;

    /// @brief Constructor
    /// @param queue SYCL queue
    /// @param voxel_size voxel size
    VoxelHashMap(const sycl_utils::DeviceQueue& queue, const float voxel_size) : queue_(queue) {
        this->set_voxel_size(voxel_size);
        this->allocate_storage(this->capacity_);
        this->prefix_sum_ = std::make_shared<common::PrefixSum>(this->queue_);
        this->valid_flags_ptr_ = std::make_shared<shared_vector<uint8_t>>(*this->queue_.ptr);
        this->clear();
        this->wg_size_add_point_cloud_ = this->compute_wg_size_add_point_cloud();
    }

    /// @brief Set voxel size
    /// @param size voxel size
    void set_voxel_size(const float voxel_size) {
        if (voxel_size <= 0.0f) {
            throw std::invalid_argument("voxel_size must be positive.");
        }
        this->voxel_size_ = voxel_size;
        // Keep the cached reciprocal consistent for hashing operations.
        this->voxel_size_inv_ = 1.0f / voxel_size;
    }
    /// @brief Get voxel size
    /// @param voxel_size voxel size
    float get_voxel_size() const { return this->voxel_size_; }

    /// @brief
    /// @param max_staleness
    void set_max_staleness(const uint32_t max_staleness) { this->max_staleness_ = max_staleness; }
    /// @brief
    /// @return
    uint32_t get_max_staleness() const { return this->max_staleness_; }

    /// @brief
    /// @param remove_old_data_cycle
    void set_remove_old_data_cycle(const uint32_t remove_old_data_cycle) {
        this->remove_old_data_cycle_ = remove_old_data_cycle;
    }
    /// @brief
    /// @return
    uint32_t get_remove_old_data_cycle() const { return this->remove_old_data_cycle_; }

    /// @brief
    /// @param rehash_threshold
    void set_rehash_threshold(const float rehash_threshold) { this->rehash_threshold_ = rehash_threshold; }
    /// @brief
    /// @return
    float get_rehash_threshold() const { return this->rehash_threshold_; }

    /// @brief Set minimum number of points required to keep a voxel in the output
    /// @param min_num_point minimum number of accumulated points
    void set_min_num_point(const uint32_t min_num_point) { this->min_num_point_ = min_num_point; }
    /// @brief Get minimum number of points required to keep a voxel in the output
    /// @return minimum number of accumulated points
    uint32_t get_min_num_point() const { return this->min_num_point_; }

    /// @brief Reset the map data.
    void clear() {
        this->capacity_ = kCapacityCandidates[0];
        this->voxel_num_ = 0;
        this->staleness_counter_ = 0;
        this->has_rgb_data_ = false;
        this->has_intensity_data_ = false;

        this->key_ptr_->resize(this->capacity_);
        this->core_data_ptr_->resize(this->capacity_);
        this->color_data_ptr_->resize(this->capacity_);
        this->intensity_data_ptr_->resize(this->capacity_);
        this->last_update_ptr_->resize(this->capacity_);

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
        evs += this->queue_.ptr->fill<uint32_t>(this->last_update_ptr_->data(), 0U, this->last_update_ptr_->size());
        evs.wait_and_throw();
    }

    /// @brief add PointCloud to voxel map
    /// @param cloud Point cloud in the sensor frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    void add_point_cloud(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) {
        const size_t N = cloud.size();

        // rehash
        if (this->rehash_threshold_ < (float)this->voxel_num_ / (float)this->capacity_) {
            const size_t next_capacity = this->get_next_capacity_value();
            if (next_capacity > this->capacity_) {
                this->rehash(next_capacity);
            }
        }

        if (N > 0) {
            // add PointCloud to voxel map
            this->add_point_cloud_impl(cloud, sensor_pose);
        }

        // remove old data
        if (this->remove_old_data_cycle_ > 0 && (this->staleness_counter_ % this->remove_old_data_cycle_) == 0) {
            this->remove_old_data();
        }

        // increment counter
        ++this->staleness_counter_;
    }

    /// @brief Export the aggregated voxels within the provided bounding range.
    /// @param result Point cloud container storing the filtered voxels.
    /// @param center Center of the query bounding box in meters.
    /// @param distance Half-extent of the axis-aligned bounding box in meters.
    void downsampling(PointCloudShared& result, const Eigen::Vector3f& center, const float distance = 100.0f) {
        if (this->voxel_num_ == 0) {
            result.clear();
            return;
        }

        const size_t allocation_size = this->voxel_num_;

        result.resize_points(allocation_size);

        RGBType* rgb_output = nullptr;
        if (this->has_rgb_data_) {
            // Allocate RGB container when aggregated color data is available.
            result.resize_rgb(allocation_size);
            rgb_output = result.rgb_ptr();
        } else {
            result.resize_rgb(0);
        }

        float* intensity_output = nullptr;
        if (this->has_intensity_data_) {
            // Allocate intensity container when aggregated intensity data is available.
            result.resize_intensities(allocation_size);
            intensity_output = result.intensities_ptr();
        } else {
            result.resize_intensities(0);
        }

        const size_t final_voxel_num =
            this->downsampling_impl(*result.points, center, distance, rgb_output, intensity_output);
        result.resize_points(final_voxel_num);
        result.resize_rgb(this->has_rgb_data_ ? final_voxel_num : 0);
        result.resize_intensities(this->has_intensity_data_ ? final_voxel_num : 0);
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
            const auto key_ptr = this->key_ptr_->data();
            const auto core_ptr = this->core_data_ptr_->data();
            const float voxel_size_inv = this->voxel_size_inv_;
            const size_t max_probe = this->max_probe_length_;
            const size_t capacity = this->capacity_;
            const uint32_t min_num_point = this->min_num_point_;

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
                        // Count as overlap only when enough samples were accumulated in the voxel.
                        const VoxelCoreData& voxel_core = core_ptr[slot];
                        if (voxel_core.count >= min_num_point) {
                            overlap_sum += 1U;
                        }
                        return;
                    }
                    if (stored_key == VoxelConstants::invalid_coord) {
                        return;
                    }
                }
            });
        });

        event.wait_and_throw();

        return static_cast<float>(overlap_counter.at(0)) / static_cast<float>(N);
    }

    void remove_old_data() { this->remove_old_data_impl(); }

private:
    using atomic_ref_float = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint32_t = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint64_t = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;

    /// @brief Core voxel data containing position information (16 bytes)
    struct VoxelCoreData {
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        uint32_t count = 0U;
    };
    static_assert(sizeof(VoxelCoreData) == 16, "VoxelCoreData must be 16 bytes for optimal memory layout");

    /// @brief Color data for RGB information (20 bytes)
    struct VoxelColorData {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_a = 0.0f;
        uint32_t color_count = 0U;
    };
    static_assert(sizeof(VoxelColorData) == 20, "VoxelColorData must be 20 bytes for optimal memory layout");

    /// @brief Intensity data for reflectivity information (8 bytes)
    struct VoxelIntensityData {
        float sum_intensity = 0.0f;
        uint32_t intensity_count = 0U;
    };
    static_assert(sizeof(VoxelIntensityData) == 8, "VoxelIntensityData must be 8 bytes for optimal memory layout");

    /// @brief Accumulator types for local workgroup reduction.
    /// @note These are type aliases to the corresponding Data structs since they share
    ///       identical field layouts. The semantic difference (persistent storage vs
    ///       temporary accumulation) is preserved through naming and usage context.
    using VoxelCoreAccumulator = VoxelCoreData;
    using VoxelColorAccumulator = VoxelColorData;
    using VoxelIntensityAccumulator = VoxelIntensityData;

    struct VoxelLocalData {
        uint64_t voxel_idx = VoxelConstants::invalid_coord;
        VoxelCoreAccumulator core_acc;
        VoxelColorAccumulator color_acc;
        VoxelIntensityAccumulator intensity_acc;
    };

    SYCL_EXTERNAL static void atomic_add_voxel_data(const VoxelCoreAccumulator& core_src,
                                                     const VoxelColorAccumulator& color_src,
                                                     const VoxelIntensityAccumulator& intensity_src,
                                                     VoxelCoreData& core_dst, VoxelColorData& color_dst,
                                                     VoxelIntensityData& intensity_dst, bool has_rgb, bool has_intensity) {
        // Core data - position accumulation
        atomic_ref_float(core_dst.sum_x).fetch_add(core_src.sum_x);
        atomic_ref_float(core_dst.sum_y).fetch_add(core_src.sum_y);
        atomic_ref_float(core_dst.sum_z).fetch_add(core_src.sum_z);
        atomic_ref_uint32_t(core_dst.count).fetch_add(core_src.count);

        // Color data (only if present)
        if (has_rgb && color_src.color_count > 0U) {
            atomic_ref_float(color_dst.sum_r).fetch_add(color_src.sum_r);
            atomic_ref_float(color_dst.sum_g).fetch_add(color_src.sum_g);
            atomic_ref_float(color_dst.sum_b).fetch_add(color_src.sum_b);
            atomic_ref_float(color_dst.sum_a).fetch_add(color_src.sum_a);
            atomic_ref_uint32_t(color_dst.color_count).fetch_add(color_src.color_count);
        }

        // Intensity data (only if present)
        if (has_intensity && intensity_src.intensity_count > 0U) {
            atomic_ref_float(intensity_dst.sum_intensity).fetch_add(intensity_src.sum_intensity);
            atomic_ref_uint32_t(intensity_dst.intensity_count).fetch_add(intensity_src.intensity_count);
        }
    }

    SYCL_EXTERNAL static void atomic_store_timestamp(uint32_t old_timestamp, uint32_t& new_timestamp) {
        // update
        atomic_ref_uint32_t(new_timestamp).store(old_timestamp);
    }

    SYCL_EXTERNAL static void compute_averaged_attributes(const VoxelCoreData& core, const VoxelColorData& color,
                                                          const VoxelIntensityData& intensity, size_t output_idx,
                                                          PointType* pt_output, RGBType* rgb_output,
                                                          float* intensity_output, uint32_t min_num_point = 1) {
        if (core.count >= min_num_point) {
            const float inv_count = 1.0f / static_cast<float>(core.count);
            pt_output[output_idx].x() = core.sum_x * inv_count;
            pt_output[output_idx].y() = core.sum_y * inv_count;
            pt_output[output_idx].z() = core.sum_z * inv_count;
            pt_output[output_idx].w() = 1.0f;
        } else {
            pt_output[output_idx].setZero();
        }
        if (rgb_output) {
            if (color.color_count > 0U) {
                const float inv_color_count = 1.0f / static_cast<float>(color.color_count);
                rgb_output[output_idx].x() = color.sum_r * inv_color_count;
                rgb_output[output_idx].y() = color.sum_g * inv_color_count;
                rgb_output[output_idx].z() = color.sum_b * inv_color_count;
                rgb_output[output_idx].w() = color.sum_a * inv_color_count;
            } else {
                rgb_output[output_idx].setZero();
            }
        }

        if (intensity_output) {
            if (intensity.intensity_count > 0U) {
                const float inv_intensity_count = 1.0f / static_cast<float>(intensity.intensity_count);
                intensity_output[output_idx] = intensity.sum_intensity * inv_intensity_count;
            } else {
                intensity_output[output_idx] = 0.0f;
            }
        }
    }

    SYCL_EXTERNAL static bool centroid_inside_bbox(const VoxelCoreData& core, float min_x, float min_y, float min_z,
                                                   float max_x, float max_y, float max_z) {
        if (core.count == 0U) {
            return false;
        }

        const float inv_count = 1.0f / static_cast<float>(core.count);
        const float centroid_x = core.sum_x * inv_count;
        const float centroid_y = core.sum_y * inv_count;
        const float centroid_z = core.sum_z * inv_count;

        return (centroid_x >= min_x && centroid_x <= max_x) && (centroid_y >= min_y && centroid_y <= max_y) &&
               (centroid_z >= min_z && centroid_z <= max_z);
    }

    SYCL_EXTERNAL static bool should_include_voxel(uint64_t key, const VoxelCoreData& core, uint32_t min_num_point,
                                                   float min_x, float min_y, float min_z, float max_x, float max_y,
                                                   float max_z) {
        if (key == VoxelConstants::invalid_coord || core.count < min_num_point) {
            return false;
        }

        return centroid_inside_bbox(core, min_x, min_y, min_z, max_x, max_y, max_z);
    }

    sycl_utils::DeviceQueue queue_;
    float voxel_size_ = 0.0f;
    float voxel_size_inv_ = 0.0f;
    inline static constexpr std::array<size_t, 11> kCapacityCandidates = {
        30029, 60013, 120011, 240007, 480013, 960017, 1920001, 3840007, 7680017, 15360013, 30720007};  // prime number
    size_t capacity_ = kCapacityCandidates[0];

    shared_vector_ptr<uint64_t> key_ptr_ = nullptr;
    shared_vector_ptr<VoxelCoreData> core_data_ptr_ = nullptr;
    shared_vector_ptr<VoxelColorData> color_data_ptr_ = nullptr;
    shared_vector_ptr<VoxelIntensityData> intensity_data_ptr_ = nullptr;
    shared_vector_ptr<uint32_t> last_update_ptr_ = nullptr;
    shared_vector_ptr<uint8_t> valid_flags_ptr_ = nullptr;
    common::PrefixSum::Ptr prefix_sum_ = nullptr;

    uint32_t staleness_counter_ = 0;
    uint32_t max_staleness_ = 100;
    uint32_t remove_old_data_cycle_ = 10;

    const size_t max_probe_length_ = 100;

    float rehash_threshold_ = 0.7f;

    size_t wg_size_add_point_cloud_ = 128UL;

    size_t voxel_num_ = 0;
    bool has_rgb_data_ = false;
    bool has_intensity_data_ = false;
    uint32_t min_num_point_ = 1U;

    void update_voxel_num_and_flags(size_t new_voxel_num) {
        this->voxel_num_ = new_voxel_num;
        if (this->voxel_num_ == 0) {
            this->has_rgb_data_ = false;
            this->has_intensity_data_ = false;
        }
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
        this->last_update_ptr_ = std::make_shared<shared_vector<uint32_t>>(new_capacity, 0U, *this->queue_.ptr);

        this->capacity_ = new_capacity;
    }

    size_t get_next_capacity_value() const {
        // Select the next pre-defined capacity to keep probing statistics stable.
        for (const auto candidate : kCapacityCandidates) {
            if (candidate > this->capacity_) {
                return candidate;
            }
        }
        std::cout << "[Caution] VoxelHashMap reached the maximum predefined capacity (" << this->capacity_
                  << "). Further growth is not available." << std::endl;
        return this->capacity_;
    }

    size_t compute_wg_size_add_point_cloud() const {
        const size_t max_work_group_size =
            this->queue_.get_device().get_info<sycl::info::device::max_work_group_size>();
        const size_t compute_units = this->queue_.get_device().get_info<sycl::info::device::max_compute_units>();
        if (this->queue_.is_nvidia()) {
            // NVIDIA:
            return std::min(max_work_group_size, 64UL);
        } else if (this->queue_.is_intel() && this->queue_.is_gpu()) {
            // Intel iGPU:
            return std::min(max_work_group_size, compute_units * 16UL);
        } else if (this->queue_.is_cpu()) {
            // CPU:
            return std::min(max_work_group_size, compute_units * 100UL);
        }
        return 128UL;
    }

    template <typename Func>
    SYCL_EXTERNAL static void global_reduction(const VoxelLocalData& data, uint64_t* key_ptr, VoxelCoreData* core_ptr,
                                               VoxelColorData* color_ptr, VoxelIntensityData* intensity_ptr,
                                               uint32_t current, uint32_t* last_update_ptr, size_t max_probe,
                                               size_t capacity, Func voxel_num_counter, bool has_rgb, bool has_intensity) {
        const uint64_t voxel_hash = data.voxel_idx;
        if (voxel_hash == VoxelConstants::invalid_coord) return;

        for (size_t j = 0; j < max_probe; ++j) {
            const size_t slot_idx = compute_slot_id(voxel_hash, j, capacity);

            uint64_t expected = VoxelConstants::invalid_coord;
            if (atomic_ref_uint64_t(key_ptr[slot_idx]).compare_exchange_strong(expected, voxel_hash)) {
                // count up num of voxel
                voxel_num_counter(1U);

                atomic_add_voxel_data(data.core_acc, data.color_acc, data.intensity_acc, core_ptr[slot_idx],
                                      color_ptr[slot_idx], intensity_ptr[slot_idx], has_rgb, has_intensity);
                atomic_store_timestamp(current, last_update_ptr[slot_idx]);
                break;

            } else if (expected == voxel_hash) {
                atomic_add_voxel_data(data.core_acc, data.color_acc, data.intensity_acc, core_ptr[slot_idx],
                                      color_ptr[slot_idx], intensity_ptr[slot_idx], has_rgb, has_intensity);
                atomic_store_timestamp(current, last_update_ptr[slot_idx]);
                break;
            }
        }
    }

    SYCL_EXTERNAL static uint64_t hash2(uint64_t voxel_hash, size_t capacity) {
        return (capacity - 2) - (voxel_hash % (capacity - 2));
    }
    SYCL_EXTERNAL static size_t compute_slot_id(uint64_t voxel_hash, size_t probe, size_t capacity) {
        return (voxel_hash + probe * hash2(voxel_hash, capacity)) % capacity;
    }

    void add_point_cloud_impl(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) {
        const size_t N = cloud.size();
        if (N == 0) return;

        const bool has_rgb = cloud.has_rgb();
        const bool has_intensity = cloud.has_intensity();
        this->has_rgb_data_ |= has_rgb;
        this->has_intensity_data_ |= has_intensity;

        // add to voxel hash map
        shared_vector<uint32_t> voxel_num_vec(1, this->voxel_num_, *this->queue_.ptr);

        auto reduction_event = this->queue_.ptr->submit([&](sycl::handler& h) {
            // Use the configured work-group size as the kernel's local size.
            const size_t local_size = this->wg_size_add_point_cloud_;
            const size_t num_work_groups = (N + local_size - 1) / local_size;
            const size_t global_size = num_work_groups * local_size;

            // Allocate local memory for work group operations
            const auto local_voxel_data = sycl::local_accessor<VoxelLocalData>(local_size, h);
            const auto trans = eigen_utils::to_sycl_vec(sensor_pose.matrix());

            size_t power_of_2 = 1;
            while (power_of_2 < local_size) {
                power_of_2 *= 2;
            }

            // memory ptr
            const auto key_ptr = this->key_ptr_->data();
            const auto core_ptr = this->core_data_ptr_->data();
            const auto color_ptr = this->color_data_ptr_->data();
            const auto intensity_data_ptr = this->intensity_data_ptr_->data();
            const auto last_update_ptr = this->last_update_ptr_->data();

            const auto point_ptr = cloud.points_ptr();
            const auto rgb_ptr = has_rgb ? cloud.rgb_ptr() : static_cast<RGBType*>(nullptr);
            const auto intensity_ptr = has_intensity ? cloud.intensities_ptr() : static_cast<float*>(nullptr);

            const auto vs_inv = this->voxel_size_inv_;
            const auto cp = this->capacity_;
            const auto current = this->staleness_counter_;
            const auto max_probe = this->max_probe_length_;

            auto load_entry = [=](VoxelLocalData& entry, const size_t idx) {
                const PointType local_point = point_ptr[idx];
                PointType world_point;
                transform::kernel::transform_point(local_point, world_point, trans);

                const auto voxel_hash = kernel::compute_voxel_bit(world_point, vs_inv);

                entry.voxel_idx = voxel_hash;
                entry.core_acc.sum_x = world_point.x();
                entry.core_acc.sum_y = world_point.y();
                entry.core_acc.sum_z = world_point.z();
                entry.core_acc.count = 1U;

                entry.color_acc.sum_r = 0.0f;
                entry.color_acc.sum_g = 0.0f;
                entry.color_acc.sum_b = 0.0f;
                entry.color_acc.sum_a = 0.0f;
                entry.color_acc.color_count = 0U;

                entry.intensity_acc.sum_intensity = 0.0f;
                entry.intensity_acc.intensity_count = 0U;

                if (has_rgb && rgb_ptr) {
                    const auto color = rgb_ptr[idx];
                    entry.color_acc.sum_r = color.x();
                    entry.color_acc.sum_g = color.y();
                    entry.color_acc.sum_b = color.z();
                    entry.color_acc.sum_a = color.w();
                    entry.color_acc.color_count = 1U;
                }

                if (has_intensity && intensity_ptr) {
                    entry.intensity_acc.sum_intensity = intensity_ptr[idx];
                    entry.intensity_acc.intensity_count = 1U;
                }
            };

            auto combine_entry = [](VoxelLocalData& dst, const VoxelLocalData& src) {
                dst.core_acc.sum_x += src.core_acc.sum_x;
                dst.core_acc.sum_y += src.core_acc.sum_y;
                dst.core_acc.sum_z += src.core_acc.sum_z;
                dst.core_acc.count += src.core_acc.count;
                dst.color_acc.sum_r += src.color_acc.sum_r;
                dst.color_acc.sum_g += src.color_acc.sum_g;
                dst.color_acc.sum_b += src.color_acc.sum_b;
                dst.color_acc.sum_a += src.color_acc.sum_a;
                dst.color_acc.color_count += src.color_acc.color_count;
                dst.intensity_acc.sum_intensity += src.intensity_acc.sum_intensity;
                dst.intensity_acc.intensity_count += src.intensity_acc.intensity_count;
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

            auto range = sycl::nd_range<1>(global_size, local_size);

            if (this->queue_.is_nvidia()) {
                auto voxel_num = sycl::reduction(voxel_num_vec.data(), sycl::plus<uint32_t>());

                h.parallel_for(range, voxel_num, [=](sycl::nd_item<1> item, auto& voxel_num_arg) {
                    const size_t global_id = item.get_global_id(0);
                    const size_t local_id = item.get_local_id(0);

                    // Reduction on workgroup
                    common::local_reduction<true, VoxelLocalData>(
                        local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(), N, local_size, power_of_2,
                        item, load_entry, combine_entry, reset_entry, VoxelConstants::invalid_coord, key_of_entry,
                        compare_keys, equal_keys);

                    if (global_id >= N) return;

                    // Reduction on global memory
                    global_reduction(local_voxel_data[local_id], key_ptr, core_ptr, color_ptr, intensity_data_ptr, current,
                                     last_update_ptr, max_probe, cp, [&](uint32_t num) { voxel_num_arg += num; }, has_rgb,
                                     has_intensity);
                });
            } else {
                auto voxel_num_ptr = voxel_num_vec.data();

                h.parallel_for(range, [=](sycl::nd_item<1> item) {
                    const size_t global_id = item.get_global_id(0);
                    const size_t local_id = item.get_local_id(0);

                    // Reduction on workgroup
                    common::local_reduction<false, VoxelLocalData>(
                        local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(), N, local_size, power_of_2,
                        item, load_entry, combine_entry, reset_entry, VoxelConstants::invalid_coord, key_of_entry,
                        compare_keys, equal_keys);

                    if (global_id >= N) return;

                    // Reduction on global memory
                    global_reduction(local_voxel_data[local_id], key_ptr, core_ptr, color_ptr, intensity_data_ptr, current,
                                     last_update_ptr, max_probe, cp,
                                     [&](uint32_t num) { atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(num); }, has_rgb,
                                     has_intensity);
                });
            }
        });
        reduction_event.wait_and_throw();
        this->voxel_num_ = static_cast<size_t>(voxel_num_vec.at(0));
    }

    void remove_old_data_impl() {
        if (this->staleness_counter_ <= this->max_staleness_) return;

        shared_vector<uint32_t> voxel_num_vec(1, 0, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t N = this->capacity_;
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                // memory ptr
                const auto key_ptr = this->key_ptr_->data();
                const auto core_ptr = this->core_data_ptr_->data();
                const auto color_ptr = this->color_data_ptr_->data();
                const auto intensity_ptr = this->intensity_data_ptr_->data();
                const auto last_update_ptr = this->last_update_ptr_->data();
                auto clear_function = [&](uint64_t& key, VoxelCoreData& core, VoxelColorData& color,
                                          VoxelIntensityData& intensity, uint32_t& last_update) {
                    key = VoxelConstants::invalid_coord;
                    core = VoxelCoreData{};
                    color = VoxelColorData{};
                    intensity = VoxelIntensityData{};
                    last_update = 0;
                };

                auto voxel_num = sycl::reduction(voxel_num_vec.data(), sycl::plus<uint32_t>());

                const auto remove_staleness = (int64_t)this->staleness_counter_ - this->max_staleness_;
                auto range = sycl::nd_range<1>(global_size, work_group_size);

                if (this->queue_.is_nvidia()) {
                    h.parallel_for(range, voxel_num, [=](sycl::nd_item<1> item, auto& voxel_num_arg) {
                        const uint32_t i = item.get_global_id(0);
                        if (i >= N) return;

                        const auto voxel_hash = key_ptr[i];
                        if (voxel_hash == VoxelConstants::invalid_coord) return;

                        const auto last_update = last_update_ptr[i];
                        if (last_update >= remove_staleness) {
                            // count up num of voxel
                            voxel_num_arg += 1U;
                            return;
                        }
                        clear_function(key_ptr[i], core_ptr[i], color_ptr[i], intensity_ptr[i], last_update_ptr[i]);
                    });
                } else {
                    const auto voxel_num_ptr = voxel_num_vec.data();

                    h.parallel_for(range, [=](sycl::nd_item<1> item) {
                        const uint32_t i = item.get_global_id(0);
                        if (i >= N) return;

                        const auto voxel_hash = key_ptr[i];
                        if (voxel_hash == VoxelConstants::invalid_coord) return;

                        const auto last_update = last_update_ptr[i];
                        if (last_update >= remove_staleness) {
                            // count up num of voxel
                            atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(1U);
                            return;
                        }
                        clear_function(key_ptr[i], core_ptr[i], color_ptr[i], intensity_ptr[i], last_update_ptr[i]);
                    });
                }
            })
            .wait_and_throw();
        this->update_voxel_num_and_flags(static_cast<size_t>(voxel_num_vec.at(0)));
    }

    void rehash(size_t new_capacity) {
        if (this->capacity_ >= new_capacity) return;

        const auto old_capacity = this->capacity_;

        // old pointer
        auto old_key_ptr = this->key_ptr_;
        auto old_core_ptr = this->core_data_ptr_;
        auto old_color_ptr = this->color_data_ptr_;
        auto old_intensity_ptr = this->intensity_data_ptr_;
        auto old_last_update_ptr = this->last_update_ptr_;

        // make new
        this->allocate_storage(new_capacity);

        shared_vector<uint32_t> voxel_num_vec(1, 0, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t N = old_capacity;
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                // memory ptr
                const auto old_key = old_key_ptr->data();
                const auto old_core = old_core_ptr->data();
                const auto old_color = old_color_ptr->data();
                const auto old_intensity = old_intensity_ptr->data();
                const auto old_last_update = old_last_update_ptr->data();
                const auto new_key = this->key_ptr_->data();
                const auto new_core = this->core_data_ptr_->data();
                const auto new_color = this->color_data_ptr_->data();
                const auto new_intensity = this->intensity_data_ptr_->data();
                const auto new_last_update = this->last_update_ptr_->data();

                const auto new_cp = new_capacity;
                const auto max_probe = this->max_probe_length_;
                const auto has_rgb = this->has_rgb_data_;
                const auto has_intensity = this->has_intensity_data_;
                auto range = sycl::nd_range<1>(global_size, work_group_size);

                if (this->queue_.is_nvidia()) {
                    auto voxel_num = sycl::reduction(voxel_num_vec.data(), sycl::plus<uint32_t>());

                    h.parallel_for(range, voxel_num, [=](sycl::nd_item<1> item, auto& voxel_num_arg) {
                        const uint32_t i = item.get_global_id(0);
                        if (i >= N) return;

                        const uint64_t key = old_key[i];
                        if (key == VoxelConstants::invalid_coord) return;

                        VoxelLocalData data;
                        data.voxel_idx = key;
                        data.core_acc.sum_x = old_core[i].sum_x;
                        data.core_acc.sum_y = old_core[i].sum_y;
                        data.core_acc.sum_z = old_core[i].sum_z;
                        data.core_acc.count = old_core[i].count;
                        data.color_acc.sum_r = old_color[i].sum_r;
                        data.color_acc.sum_g = old_color[i].sum_g;
                        data.color_acc.sum_b = old_color[i].sum_b;
                        data.color_acc.sum_a = old_color[i].sum_a;
                        data.color_acc.color_count = old_color[i].color_count;
                        data.intensity_acc.sum_intensity = old_intensity[i].sum_intensity;
                        data.intensity_acc.intensity_count = old_intensity[i].intensity_count;

                        global_reduction(data, new_key, new_core, new_color, new_intensity, old_last_update[i],
                                         new_last_update, max_probe, new_cp, [&](uint32_t num) { voxel_num_arg += num; },
                                         has_rgb, has_intensity);
                    });
                } else {
                    auto voxel_num_ptr = voxel_num_vec.data();

                    h.parallel_for(range, [=](sycl::nd_item<1> item) {
                        const uint32_t i = item.get_global_id(0);
                        if (i >= N) return;

                        const uint64_t key = old_key[i];
                        if (key == VoxelConstants::invalid_coord) return;

                        VoxelLocalData data;
                        data.voxel_idx = key;
                        data.core_acc.sum_x = old_core[i].sum_x;
                        data.core_acc.sum_y = old_core[i].sum_y;
                        data.core_acc.sum_z = old_core[i].sum_z;
                        data.core_acc.count = old_core[i].count;
                        data.color_acc.sum_r = old_color[i].sum_r;
                        data.color_acc.sum_g = old_color[i].sum_g;
                        data.color_acc.sum_b = old_color[i].sum_b;
                        data.color_acc.sum_a = old_color[i].sum_a;
                        data.color_acc.color_count = old_color[i].color_count;
                        data.intensity_acc.sum_intensity = old_intensity[i].sum_intensity;
                        data.intensity_acc.intensity_count = old_intensity[i].intensity_count;

                        global_reduction(data, new_key, new_core, new_color, new_intensity, old_last_update[i],
                                         new_last_update, max_probe, new_cp,
                                         [&](uint32_t num) { atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(num); }, has_rgb,
                                         has_intensity);
                    });
                }
            })
            .wait_and_throw();
        this->update_voxel_num_and_flags(static_cast<size_t>(voxel_num_vec.at(0)));
    }

    size_t downsampling_impl(PointContainerShared& result, const Eigen::Vector3f& center, const float distance,
                             RGBType* rgb_output_ptr = nullptr, float* intensity_output_ptr = nullptr) {
        // Compute the axis-aligned bounding box around the requested query center.
        const float bbox_min_x = center.x() - distance;
        const float bbox_min_y = center.y() - distance;
        const float bbox_min_z = center.z() - distance;
        const float bbox_max_x = center.x() + distance;
        const float bbox_max_y = center.y() + distance;
        const float bbox_max_z = center.z() + distance;

        const bool is_nvidia = this->queue_.is_nvidia();
        size_t filtered_voxel_count = 0;

        if (is_nvidia) {
            // compute valid flags
            if (this->valid_flags_ptr_->size() < this->capacity_) {
                this->valid_flags_ptr_->resize(this->capacity_);
            }
            this->queue_.ptr
                ->submit([&](sycl::handler& h) {
                    const size_t cp = this->capacity_;
                    const size_t work_group_size = this->queue_.get_work_group_size();
                    const size_t global_size = this->queue_.get_global_size(cp);

                    // memory ptr
                    const auto valid_flags = this->valid_flags_ptr_->data();
                    const auto key_ptr = this->key_ptr_->data();
                    const auto core_ptr = this->core_data_ptr_->data();
                    const auto min_num_point = this->min_num_point_;

                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t global_id = item.get_global_id(0);
                        if (global_id >= cp) return;

                        const auto key = key_ptr[global_id];
                        const auto core = core_ptr[global_id];

                        if (!should_include_voxel(key, core, min_num_point, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x,
                                                  bbox_max_y, bbox_max_z)) {
                            valid_flags[global_id] = 0U;
                            return;
                        }

                        valid_flags[global_id] = 1U;
                    });
                })
                .wait_and_throw();
            // compute prefix sum
            filtered_voxel_count = this->prefix_sum_->compute(*this->valid_flags_ptr_);
        }

        // voxel hash map to point cloud
        result.resize(this->voxel_num_);
        shared_vector<uint32_t> point_num_vec(1, 0, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const auto cp = this->capacity_;
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(cp);

                // memory ptr
                const auto core_ptr = this->core_data_ptr_->data();
                const auto color_ptr = this->color_data_ptr_->data();
                const auto intensity_data_ptr = this->intensity_data_ptr_->data();
                const auto result_ptr = result.data();
                // Optional output arrays for aggregated RGB colors and intensity values.
                const auto rgb_output = rgb_output_ptr;
                const auto intensity_output = intensity_output_ptr;

                if (is_nvidia) {
                    const auto flag_ptr = this->valid_flags_ptr_->data();
                    const auto prefix_sum_ptr = this->prefix_sum_->get_prefix_sum().data();
                    const auto min_num_point = this->min_num_point_;

                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t i = item.get_global_id(0);
                        if (i >= cp) return;

                        if (flag_ptr[i] == 1) {
                            const size_t output_idx = prefix_sum_ptr[i] - 1;
                            const auto core = core_ptr[i];
                            const auto color = color_ptr[i];
                            const auto intensity = intensity_data_ptr[i];

                            compute_averaged_attributes(core, color, intensity, output_idx, result_ptr, rgb_output,
                                                        intensity_output, min_num_point);
                        }
                    });

                } else {
                    const auto key_ptr = this->key_ptr_->data();

                    const auto point_num_ptr = point_num_vec.data();
                    const auto min_num_point = this->min_num_point_;
                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const auto i = item.get_global_id(0);
                        if (i >= cp) return;

                        const auto key = key_ptr[i];
                        const auto core = core_ptr[i];

                        if (!should_include_voxel(key, core, min_num_point, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x,
                                                  bbox_max_y, bbox_max_z)) {
                            return;
                        }

                        const auto output_idx = atomic_ref_uint32_t(point_num_ptr[0]).fetch_add(1U);

                        const auto color = color_ptr[i];
                        const auto intensity = intensity_data_ptr[i];

                        compute_averaged_attributes(core, color, intensity, output_idx, result_ptr, rgb_output,
                                                    intensity_output, min_num_point);
                    });
                }
            })
            .wait_and_throw();

        if (!is_nvidia) {
            filtered_voxel_count = static_cast<size_t>(point_num_vec.at(0));
        }

        return filtered_voxel_count;
    }
};

}  // namespace mapping
}  // namespace algorithms
}  // namespace sycl_points
