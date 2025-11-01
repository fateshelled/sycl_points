#pragma once

#include <stdexcept>

#include <sycl_points/algorithms/common/prefix_sum.hpp>
#include <sycl_points/algorithms/common/voxel_constants.hpp>
#include <sycl_points/algorithms/voxel_downsampling.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {
namespace algorithms {
namespace filter {

class VoxelHashMap {
public:
    using Ptr = std::shared_ptr<VoxelHashMap>;

    /// @brief Constructor
    /// @param queue SYCL queue
    /// @param voxel_size voxel size
    VoxelHashMap(const sycl_utils::DeviceQueue& queue, const float voxel_size) : queue_(queue) {
        this->set_voxel_size(voxel_size);
        this->malloc_data();
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

    /// @brief
    void clear() {
        this->initialize_device_ptr();
        this->voxel_num_ = 0;
        this->staleness_counter_ = 0;
        this->has_rgb_data_ = false;
        this->has_intensity_data_ = false;
    }

    /// @brief add PointCloud to voxel map
    /// @param pc Point Cloud
    void add_point_cloud(const PointCloudShared& pc) {
        const size_t N = pc.size();

        // rehash
        if (this->rehash_threshold_ < (float)this->voxel_num_ / (float)this->capacity_) {
            this->rehash(this->capacity_ * 2);
        }

        if (N > 0) {
            // add PointCloud to voxel map
            this->add_point_cloud_(pc);
        }

        // remove old data
        if (this->remove_old_data_cycle_ > 0 && (this->staleness_counter_ % this->remove_old_data_cycle_) == 0) {
            this->remove_old_data();
        }

        // increment counter
        ++this->staleness_counter_;
    }

    void downsampling(PointContainerShared& result) { this->downsampling_(result); }

    void downsampling(PointCloudShared& result) {
        if (this->voxel_num_ == 0) {
            result.clear();
            return;
        }

        const size_t allocation_size = this->capacity_;

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

        this->downsampling_(*result.points, rgb_output, intensity_output);

        const size_t final_voxel_num = this->voxel_num_;
        result.resize_points(final_voxel_num);

        if (this->has_rgb_data_) {
            result.resize_rgb(final_voxel_num);
        } else {
            result.resize_rgb(0);
        }

        if (this->has_intensity_data_) {
            result.resize_intensities(final_voxel_num);
        } else {
            result.resize_intensities(0);
        }
    }

    void remove_old_data() { this->remove_old_data_(); }

private:
    using atomic_ref_float = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint32_t = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint64_t = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;

    struct VoxelPoint {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        float a = 0.0f;
        float intensity = 0.0f;
        uint32_t count = 0;
        uint32_t color_count = 0;
        uint32_t intensity_count = 0;
    };

    struct VoxelLocalData {
        uint64_t voxel_idx;
        VoxelPoint pt;
    };

    SYCL_EXTERNAL static void atomic_add_voxel_point(const VoxelPoint& src, VoxelPoint& dst) {
        // add point coord
        atomic_ref_float(dst.x).fetch_add(src.x);
        atomic_ref_float(dst.y).fetch_add(src.y);
        atomic_ref_float(dst.z).fetch_add(src.z);

        // count up num of points in voxel
        atomic_ref_uint32_t(dst.count).fetch_add(src.count);

        if (src.color_count > 0U) {
            // Aggregate RGB components when available.
            atomic_ref_float(dst.r).fetch_add(src.r);
            atomic_ref_float(dst.g).fetch_add(src.g);
            atomic_ref_float(dst.b).fetch_add(src.b);
            atomic_ref_float(dst.a).fetch_add(src.a);
            atomic_ref_uint32_t(dst.color_count).fetch_add(src.color_count);
        }

        if (src.intensity_count > 0U) {
            // Aggregate intensity when available.
            atomic_ref_float(dst.intensity).fetch_add(src.intensity);
            atomic_ref_uint32_t(dst.intensity_count).fetch_add(src.intensity_count);
        }
    }

    SYCL_EXTERNAL static void atomic_store_timestamp(uint32_t old_timestamp, uint32_t& new_timestamp) {
        // update
        atomic_ref_uint32_t(new_timestamp).store(old_timestamp);
    }

    SYCL_EXTERNAL static void compute_averaged_attributes(const VoxelPoint& sum, size_t output_idx,
                                                          RGBType* rgb_output, float* intensity_output) {
        if (rgb_output) {
            if (sum.color_count > 0U) {
                const float inv_color_count = 1.0f / static_cast<float>(sum.color_count);
                rgb_output[output_idx].x() = sum.r * inv_color_count;
                rgb_output[output_idx].y() = sum.g * inv_color_count;
                rgb_output[output_idx].z() = sum.b * inv_color_count;
                rgb_output[output_idx].w() = sum.a * inv_color_count;
            } else {
                rgb_output[output_idx] = RGBType::Zero();
            }
        }

        if (intensity_output) {
            if (sum.intensity_count > 0U) {
                const float inv_intensity_count = 1.0f / static_cast<float>(sum.intensity_count);
                intensity_output[output_idx] = sum.intensity * inv_intensity_count;
            } else {
                intensity_output[output_idx] = 0.0f;
            }
        }
    }

    sycl_utils::DeviceQueue queue_;
    float voxel_size_ = 0.0f;
    float voxel_size_inv_ = 0.0f;
    size_t capacity_ = 30029;  // prime number recommended

    std::shared_ptr<uint64_t> key_ptr_ = nullptr;
    std::shared_ptr<VoxelPoint> sum_ptr_ = nullptr;
    std::shared_ptr<uint32_t> last_update_ptr_ = nullptr;
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

    void update_voxel_num_and_flags(size_t new_voxel_num) {
        this->voxel_num_ = new_voxel_num;
        if (this->voxel_num_ == 0) {
            this->has_rgb_data_ = false;
            this->has_intensity_data_ = false;
        }
    }

    void malloc_data() {
        this->key_ptr_ = std::shared_ptr<uint64_t>(sycl::malloc_shared<uint64_t>(this->capacity_, *this->queue_.ptr),
                                                   [&](uint64_t* ptr) { sycl::free(ptr, *this->queue_.ptr); });
        this->sum_ptr_ =
            std::shared_ptr<VoxelPoint>(sycl::malloc_shared<VoxelPoint>(this->capacity_, *this->queue_.ptr),
                                        [&](VoxelPoint* ptr) { sycl::free(ptr, *this->queue_.ptr); });
        this->last_update_ptr_ =
            std::shared_ptr<uint32_t>(sycl::malloc_shared<uint32_t>(this->capacity_, *this->queue_.ptr),
                                      [&](uint32_t* ptr) { sycl::free(ptr, *this->queue_.ptr); });
        this->queue_.set_accessed_by_device(this->key_ptr_.get(), this->capacity_);
        this->queue_.set_accessed_by_device(this->sum_ptr_.get(), this->capacity_);
        this->queue_.set_accessed_by_device(this->last_update_ptr_.get(), this->capacity_);
    }

    void initialize_device_ptr() {
        sycl_utils::events evs;
        evs += this->queue_.ptr->fill<VoxelPoint>(this->sum_ptr_.get(), VoxelPoint{}, this->capacity_);
        evs += this->queue_.ptr->fill<uint64_t>(this->key_ptr_.get(), VoxelConstants::invalid_coord, this->capacity_);
        evs += this->queue_.ptr->fill<uint32_t>(this->last_update_ptr_.get(), 0, this->capacity_);
        evs.wait();
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

    /// @brief Bitonic sort that works correctly with any work group size
    /// @details Uses virtual infinity padding to handle non-power-of-2 sizes
    SYCL_EXTERNAL static void bitonic_sort_local_data(VoxelLocalData* data, size_t size, size_t size_power_of_2,
                                                      sycl::nd_item<1> item) {
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

    /// @brief Reduce consecutive same voxel indices and output results
    SYCL_EXTERNAL static void reduction_sorted_local_data(VoxelLocalData* sorted_data, size_t wg_size,
                                                          sycl::nd_item<1> item) {
        const size_t local_id = item.get_local_id(0);

        // Find segments of same voxel indices and reduce them
        const auto current_voxel = local_id < wg_size ? sorted_data[local_id].voxel_idx : VoxelConstants::invalid_coord;
        // Check if this is the start of a new voxel segment
        const bool is_segment_start = (current_voxel != VoxelConstants::invalid_coord) &&
                                      ((local_id == 0) || (sorted_data[local_id - 1].voxel_idx != current_voxel));

        if (is_segment_start) {
            // Accumulate points to segment start
            for (size_t i = local_id + 1; i < wg_size && sorted_data[i].voxel_idx == current_voxel; ++i) {
                // change to invalid
                sorted_data[i].voxel_idx = VoxelConstants::invalid_coord;

                // accumulate
                sorted_data[local_id].pt.x += sorted_data[i].pt.x;
                sorted_data[local_id].pt.y += sorted_data[i].pt.y;
                sorted_data[local_id].pt.z += sorted_data[i].pt.z;
                sorted_data[local_id].pt.count += sorted_data[i].pt.count;
                sorted_data[local_id].pt.r += sorted_data[i].pt.r;
                sorted_data[local_id].pt.g += sorted_data[i].pt.g;
                sorted_data[local_id].pt.b += sorted_data[i].pt.b;
                sorted_data[local_id].pt.a += sorted_data[i].pt.a;
                sorted_data[local_id].pt.color_count += sorted_data[i].pt.color_count;
                sorted_data[local_id].pt.intensity += sorted_data[i].pt.intensity;
                sorted_data[local_id].pt.intensity_count += sorted_data[i].pt.intensity_count;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    template <bool AGGREGATE = true>
    SYCL_EXTERNAL static void local_reduction(VoxelLocalData* local_voxel_data, PointType* point_ptr,
                                              RGBType* rgb_ptr, float* intensity_ptr, bool has_rgb,
                                              bool has_intensity, size_t point_num, size_t wg_size,
                                              size_t wg_size_power_of_2, float voxel_size_inv, sycl::nd_item<1> item) {
        // Reduction on workgroup
        const size_t local_id = item.get_local_id(0);
        const size_t global_id = item.get_global_id(0);

        if (global_id < point_num && local_id < wg_size) {
            const auto voxel_hash = kernel::compute_voxel_bit(point_ptr[global_id], voxel_size_inv);

            // set local data
            local_voxel_data[local_id].voxel_idx = voxel_hash;
            local_voxel_data[local_id].pt.x = point_ptr[global_id].x();
            local_voxel_data[local_id].pt.y = point_ptr[global_id].y();
            local_voxel_data[local_id].pt.z = point_ptr[global_id].z();
            local_voxel_data[local_id].pt.count = 1;
            local_voxel_data[local_id].pt.r = 0.0f;
            local_voxel_data[local_id].pt.g = 0.0f;
            local_voxel_data[local_id].pt.b = 0.0f;
            local_voxel_data[local_id].pt.a = 0.0f;
            local_voxel_data[local_id].pt.color_count = 0;
            local_voxel_data[local_id].pt.intensity = 0.0f;
            local_voxel_data[local_id].pt.intensity_count = 0;

            if (has_rgb && rgb_ptr) {
                const auto color = rgb_ptr[global_id];
                local_voxel_data[local_id].pt.r = color.x();
                local_voxel_data[local_id].pt.g = color.y();
                local_voxel_data[local_id].pt.b = color.z();
                local_voxel_data[local_id].pt.a = color.w();
                local_voxel_data[local_id].pt.color_count = 1;
            }

            if (has_intensity && intensity_ptr) {
                local_voxel_data[local_id].pt.intensity = intensity_ptr[global_id];
                local_voxel_data[local_id].pt.intensity_count = 1;
            }
        }
        // wait local
        item.barrier(sycl::access::fence_space::local_space);

        if constexpr (AGGREGATE) {
            // sort within work group by voxel index
            bitonic_sort_local_data(local_voxel_data, wg_size, wg_size_power_of_2, item);
            // reduction
            reduction_sorted_local_data(local_voxel_data, wg_size, item);
        }
    }

    template <typename Func>
    SYCL_EXTERNAL static void global_reduction(uint64_t voxel_hash, const VoxelPoint& point, uint64_t* key_ptr,
                                               VoxelPoint* sum_ptr, uint32_t current, uint32_t* last_update_ptr,
                                               size_t max_probe, size_t capacity, Func voxel_num_counter) {
        if (voxel_hash == VoxelConstants::invalid_coord) return;

        for (size_t j = 0; j < max_probe; ++j) {
            const size_t slot_idx = compute_slot_id(voxel_hash, j, capacity);

            uint64_t expected = VoxelConstants::invalid_coord;
            if (atomic_ref_uint64_t(key_ptr[slot_idx]).compare_exchange_strong(expected, voxel_hash)) {
                // count up num of voxel
                voxel_num_counter(1U);

                atomic_add_voxel_point(point, sum_ptr[slot_idx]);
                atomic_store_timestamp(current, last_update_ptr[slot_idx]);
                break;

            } else if (expected == voxel_hash) {
                atomic_add_voxel_point(point, sum_ptr[slot_idx]);
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

    void add_point_cloud_(const PointCloudShared& pc) {
        const size_t N = pc.size();
        if (N == 0) return;

        const bool has_rgb = pc.has_rgb();
        const bool has_intensity = pc.has_intensity();
        this->has_rgb_data_ |= has_rgb;
        this->has_intensity_data_ |= has_intensity;

        // using `mem_advise to device` is slow
        // add to voxel hash map
        shared_vector<uint32_t> voxel_num_vec(1, this->voxel_num_, *this->queue_.ptr);

        auto reduction_event = this->queue_.ptr->submit([&](sycl::handler& h) {
            // Use the configured work-group size as the kernel's local size.
            const size_t local_size = this->wg_size_add_point_cloud_;
            const size_t num_work_groups = (N + local_size - 1) / local_size;
            const size_t global_size = num_work_groups * local_size;

            // Allocate local memory for work group operations
            const auto local_voxel_data = sycl::local_accessor<VoxelLocalData>(local_size, h);

            size_t power_of_2 = 1;
            while (power_of_2 < local_size) {
                power_of_2 *= 2;
            }

            // memory ptr
            const auto key_ptr = this->key_ptr_.get();
            const auto sum_ptr = this->sum_ptr_.get();
            const auto last_update_ptr = this->last_update_ptr_.get();

            const auto point_ptr = pc.points_ptr();
            const auto rgb_ptr = has_rgb ? pc.rgb_ptr() : static_cast<RGBType*>(nullptr);
            const auto intensity_ptr = has_intensity ? pc.intensities_ptr() : static_cast<float*>(nullptr);

            const auto vs_inv = this->voxel_size_inv_;
            const auto cp = this->capacity_;
            const auto current = this->staleness_counter_;
            const auto max_probe = this->max_probe_length_;

            auto range = sycl::nd_range<1>(global_size, local_size);

            if (this->queue_.is_nvidia()) {
                auto voxel_num = sycl::reduction(voxel_num_vec.data(), sycl::plus<uint32_t>());

                h.parallel_for(range, voxel_num, [=](sycl::nd_item<1> item, auto& voxel_num_arg) {
                    const size_t global_id = item.get_global_id(0);
                    const size_t local_id = item.get_local_id(0);

                    // Reduction on workgroup
                    local_reduction<true>(local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(),
                                          point_ptr, rgb_ptr, intensity_ptr, has_rgb, has_intensity, N, local_size,
                                          power_of_2, vs_inv, item);

                    if (global_id >= N) return;

                    // Reduction on global memory
                    global_reduction(local_voxel_data[local_id].voxel_idx, local_voxel_data[local_id].pt, key_ptr,
                                     sum_ptr, current, last_update_ptr, max_probe, cp,
                                     [&](uint32_t num) { voxel_num_arg += num; });
                });
            } else {
                auto voxel_num_ptr = voxel_num_vec.data();

                h.parallel_for(range, [=](sycl::nd_item<1> item) {
                    const size_t global_id = item.get_global_id(0);
                    const size_t local_id = item.get_local_id(0);

                    // Reduction on workgroup
                    local_reduction<false>(local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(),
                                           point_ptr, rgb_ptr, intensity_ptr, has_rgb, has_intensity, N, local_size,
                                           power_of_2, vs_inv, item);

                    if (global_id >= N) return;

                    // Reduction on global memory
                    global_reduction(local_voxel_data[local_id].voxel_idx, local_voxel_data[local_id].pt, key_ptr,
                                     sum_ptr, current, last_update_ptr, max_probe, cp,
                                     [&](uint32_t num) { atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(num); });
                });
            }
        });
        reduction_event.wait();
        this->voxel_num_ = static_cast<size_t>(voxel_num_vec.at(0));
    }

    void remove_old_data_() {
        if (this->staleness_counter_ <= this->max_staleness_) return;

        shared_vector<uint32_t> voxel_num_vec(1, 0, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t N = this->capacity_;
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                // memory ptr
                const auto key_ptr = this->key_ptr_.get();
                const auto sum_ptr = this->sum_ptr_.get();
                const auto last_update_ptr = this->last_update_ptr_.get();
                auto clear_function = [&](uint64_t& key, VoxelPoint& pt, uint32_t& last_update) {
                    key = VoxelConstants::invalid_coord;
                    pt = VoxelPoint{};
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
                        clear_function(key_ptr[i], sum_ptr[i], last_update_ptr[i]);
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
                        clear_function(key_ptr[i], sum_ptr[i], last_update_ptr[i]);
                    });
                }
            })
            .wait();
        this->update_voxel_num_and_flags(static_cast<size_t>(voxel_num_vec.at(0)));
    }

    void rehash(size_t new_capacity) {
        if (this->capacity_ >= new_capacity) return;

        const auto old_capacity = this->capacity_;
        this->capacity_ = new_capacity;

        // old pointer
        auto old_key_ptr = this->key_ptr_;
        auto old_sum_ptr = this->sum_ptr_;
        auto old_last_update_ptr = this->last_update_ptr_;

        // make new
        this->malloc_data();
        this->initialize_device_ptr();

        shared_vector<uint32_t> voxel_num_vec(1, 0, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t N = old_capacity;
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                // memory ptr
                const auto old_key = old_key_ptr.get();
                const auto old_sum = old_sum_ptr.get();
                const auto old_last_update = old_last_update_ptr.get();
                const auto new_key = this->key_ptr_.get();
                const auto new_sum = this->sum_ptr_.get();
                const auto new_last_update = this->last_update_ptr_.get();

                const auto new_cp = new_capacity;
                const auto max_probe = this->max_probe_length_;
                auto range = sycl::nd_range<1>(global_size, work_group_size);

                if (this->queue_.is_nvidia()) {
                    auto voxel_num = sycl::reduction(voxel_num_vec.data(), sycl::plus<uint32_t>());

                    h.parallel_for(range, voxel_num, [=](sycl::nd_item<1> item, auto& voxel_num_arg) {
                        const uint32_t i = item.get_global_id(0);
                        if (i >= N) return;

                        global_reduction(old_key[i], old_sum[i], new_key, new_sum, old_last_update[i], new_last_update,
                                         max_probe, new_cp,  //
                                         [&](uint32_t num) { voxel_num_arg += num; });
                    });
                } else {
                    auto voxel_num_ptr = voxel_num_vec.data();

                    h.parallel_for(range, [=](sycl::nd_item<1> item) {
                        const uint32_t i = item.get_global_id(0);
                        if (i >= N) return;

                        global_reduction(old_key[i], old_sum[i], new_key, new_sum, old_last_update[i], new_last_update,
                                         max_probe, new_cp,  //
                                         [&](uint32_t num) { atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(num); });
                    });
                }
            })
            .wait();
        this->update_voxel_num_and_flags(static_cast<size_t>(voxel_num_vec.at(0)));
    }

    void downsampling_(PointContainerShared& result, RGBType* rgb_output_ptr = nullptr,
                       float* intensity_output_ptr = nullptr) {
        // NVIDIA device
        if (this->queue_.is_nvidia()) {
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
                    const auto key_ptr = this->key_ptr_.get();

                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t global_id = item.get_global_id(0);
                        if (global_id >= cp) return;

                        valid_flags[global_id] =
                            static_cast<uint8_t>(key_ptr[global_id] != VoxelConstants::invalid_coord);
                    });
                })
                .wait();
            // compute prefix sum
            this->voxel_num_ = this->prefix_sum_->compute(*this->valid_flags_ptr_);
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
                const auto sum_ptr = this->sum_ptr_.get();
                const auto result_ptr = result.data();
                // Optional output arrays for aggregated RGB colors and intensity values.
                const auto rgb_output = rgb_output_ptr;
                const auto intensity_output = intensity_output_ptr;

                if (this->queue_.is_nvidia()) {
                    const auto flag_ptr = this->valid_flags_ptr_->data();
                    const auto prefix_sum_ptr = this->prefix_sum_->get_prefix_sum().data();

                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t i = item.get_global_id(0);
                        if (i >= cp) return;

                        if (flag_ptr[i] == 1) {
                            const size_t output_idx = prefix_sum_ptr[i] - 1;

                            const auto sum = sum_ptr[i];
                            result_ptr[output_idx].x() = sum.x / sum.count;
                            result_ptr[output_idx].y() = sum.y / sum.count;
                            result_ptr[output_idx].z() = sum.z / sum.count;
                            result_ptr[output_idx].w() = 1.0f;

                            compute_averaged_attributes(sum, output_idx, rgb_output, intensity_output);
                        }
                    });

                } else {
                    const auto key_ptr = this->key_ptr_.get();

                    const auto point_num_ptr = point_num_vec.data();
                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const auto i = item.get_global_id(0);
                        if (i >= cp) return;

                        if (key_ptr[i] == VoxelConstants::invalid_coord) return;

                        const auto output_idx = atomic_ref_uint32_t(point_num_ptr[0]).fetch_add(1U);

                        const auto sum = sum_ptr[i];
                        result_ptr[output_idx].x() = sum.x / sum.count;
                        result_ptr[output_idx].y() = sum.y / sum.count;
                        result_ptr[output_idx].z() = sum.z / sum.count;
                        result_ptr[output_idx].w() = 1.0f;

                        compute_averaged_attributes(sum, output_idx, rgb_output, intensity_output);
                    });
                }
            })
            .wait();

        if (!this->queue_.is_nvidia()) {
            this->update_voxel_num_and_flags(static_cast<size_t>(point_num_vec.at(0)));
        } else {
            this->update_voxel_num_and_flags(this->voxel_num_);
        }
    }
};

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
