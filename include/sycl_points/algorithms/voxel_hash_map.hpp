#pragma once

#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {
namespace algorithms {
namespace voxel_hash_map {

struct VoxelConstants {
    static constexpr uint64_t invalid_coord = std::numeric_limits<uint64_t>::max();
    static constexpr uint8_t coord_bit_size = 21;                       // Bits to represent each voxel coordinate
    static constexpr int64_t coord_bit_mask = (1 << 21) - 1;            // Bit mask
    static constexpr int64_t coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive
};

namespace kernel {

SYCL_EXTERNAL inline uint64_t compute_voxel_bit(const PointType& point, const float voxel_size_inv) {
    // Ref: https://github.com/koide3/gtsam_points/blob/master/src/gtsam_points/types/point_cloud_cpu_funcs.cpp
    // function: voxelgrid_sampling
    // MIT License

    if (!sycl::isfinite(point.x()) || !sycl::isfinite(point.y()) || !sycl::isfinite(point.z())) {
        return VoxelConstants::invalid_coord;
    }

    const auto coord0 = static_cast<int64_t>(std::floor(point.x() * voxel_size_inv)) + VoxelConstants::coord_offset;
    const auto coord1 = static_cast<int64_t>(std::floor(point.y() * voxel_size_inv)) + VoxelConstants::coord_offset;
    const auto coord2 = static_cast<int64_t>(std::floor(point.z() * voxel_size_inv)) + VoxelConstants::coord_offset;

    if (coord0 < 0 || VoxelConstants::coord_bit_mask < coord0 || coord1 < 0 ||
        VoxelConstants::coord_bit_mask < coord1 || coord2 < 0 || VoxelConstants::coord_bit_mask < coord2) {
        return VoxelConstants::invalid_coord;
    }

    // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
    return (static_cast<uint64_t>(coord0 & VoxelConstants::coord_bit_mask) << (VoxelConstants::coord_bit_size * 0)) |
           (static_cast<uint64_t>(coord1 & VoxelConstants::coord_bit_mask) << (VoxelConstants::coord_bit_size * 1)) |
           (static_cast<uint64_t>(coord2 & VoxelConstants::coord_bit_mask) << (VoxelConstants::coord_bit_size * 2));
}

}  // namespace kernel

class VoxelHashMap {
public:
    using Ptr = std::shared_ptr<VoxelHashMap>;

    /// @brief Constructor
    /// @param queue SYCL queue
    /// @param voxel_size voxel size
    VoxelHashMap(const sycl_utils::DeviceQueue& queue, const float voxel_size)
        : queue_(queue), voxel_size_(voxel_size), voxel_size_inv_(1.0f / voxel_size) {
        if (queue.is_cpu()) {
            throw std::runtime_error("VoxelHashMap does not support CPU");
        }
        this->make_device_ptr();
        this->clear();
        this->wg_size_add_point_cloud_ = this->compute_wg_size_add_point_cloud();
    }

    /// @brief Set voxel size
    /// @param size voxel size
    void set_voxel_size(const float voxel_size) { this->voxel_size_ = voxel_size; }
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

    void downsampling(PointContainerShared& result) {
        // voxel hash map to point cloud
        result.resize(this->voxel_num_);
        shared_vector<uint32_t> point_num_vec(1, 0, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(this->capacity_);

                // memory ptr
                const auto key_ptr = this->key_ptr_.get();
                const auto sum_ptr = this->sum_ptr_.get();

                const auto result_ptr = result.data();

                const auto point_num_ptr = point_num_vec.data();
                const auto cp = this->capacity_;
                const uint64_t vx_num = this->voxel_num_;
                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const auto i = item.get_global_id(0);
                    if (i >= cp) return;

                    if (key_ptr[i] == VoxelConstants::invalid_coord) return;

                    const auto sum = sum_ptr[i];
                    if (sum.count == 0) return;
                    if (!std::isfinite(sum.x) || !std::isfinite(sum.y) || !std::isfinite(sum.z)) return;

                    const auto output_idx = atomic_ref_uint32_t(point_num_ptr[0]).fetch_add(1U);
                    if (output_idx >= vx_num) return;

                    result_ptr[output_idx].x() = sum.x / sum.count;
                    result_ptr[output_idx].y() = sum.y / sum.count;
                    result_ptr[output_idx].z() = sum.z / sum.count;
                    result_ptr[output_idx].w() = 1.0f;
                });
            })
            .wait();
    }

    void downsampling(PointCloudShared& result) {
        if (this->voxel_num_ == 0) {
            result.clear();
            return;
        }
        this->downsampling(*result.points);
    }

    void remove_old_data() {
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

                const auto voxel_num_ptr = voxel_num_vec.data();

                const auto remove_staleness = (int64_t)this->staleness_counter_ - this->max_staleness_;

                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
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

                    key_ptr[i] = VoxelConstants::invalid_coord;
                    sum_ptr[i].x = 0.0f;
                    sum_ptr[i].y = 0.0f;
                    sum_ptr[i].z = 0.0f;
                    sum_ptr[i].count = 0;
                    last_update_ptr[i] = 0;
                });
            })
            .wait();
        this->voxel_num_ = static_cast<size_t>(voxel_num_vec.at(0));
    }

private:
    using atomic_ref_float = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint32_t = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint64_t = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;

    struct VoxelPoint {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        uint32_t count = 0;
    };

    sycl_utils::DeviceQueue queue_;
    float voxel_size_;
    float voxel_size_inv_;
    size_t capacity_ = 30029;  // prime number recommended

    std::shared_ptr<uint64_t> key_ptr_ = nullptr;
    std::shared_ptr<VoxelPoint> sum_ptr_ = nullptr;
    std::shared_ptr<uint32_t> last_update_ptr_ = nullptr;

    uint32_t staleness_counter_ = 0;
    uint32_t max_staleness_ = 100;
    uint32_t remove_old_data_cycle_ = 10;

    const size_t max_probe_length_ = 100;

    float rehash_threshold_ = 0.7f;

    size_t wg_size_add_point_cloud_ = 128UL;

    size_t voxel_num_ = 0;

    void make_device_ptr() {
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
        this->queue_.ptr->fill<uint64_t>(this->key_ptr_.get(), VoxelConstants::invalid_coord, this->capacity_);
        this->queue_.ptr->fill<VoxelPoint>(this->sum_ptr_.get(), VoxelPoint{0.0f, 0.0f, 0.0f, 0}, this->capacity_);
        this->queue_.ptr->fill<uint32_t>(this->last_update_ptr_.get(), 0, this->capacity_);
    }

    size_t compute_wg_size_add_point_cloud() const {
        const size_t max_work_group_size =
            this->queue_.get_device().get_info<sycl::info::device::max_work_group_size>();
        const size_t compute_units = this->queue_.get_device().get_info<sycl::info::device::max_compute_units>();
        if (this->queue_.is_nvidia()) {
            // NVIDIA:
            return std::min(max_work_group_size, 128UL);
        } else if (this->queue_.is_intel() && this->queue_.is_gpu()) {
            // Intel iGPU:
            return std::max(std::min(max_work_group_size, compute_units * 8UL), 64UL);
        } else if (this->queue_.is_cpu()) {
            // CPU:
            return std::min(max_work_group_size, compute_units * 50UL);
        } 
        return 128UL;
    }

    struct VoxelData {
        uint64_t voxel_idx = voxel_hash_map::VoxelConstants::invalid_coord;
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        uint32_t count = 0;
    };

    /// @brief Bitonic sort that works correctly with any work group size
    /// @details Uses virtual infinity padding to handle non-power-of-2 sizes
    SYCL_EXTERNAL static void bitonic_sort_local_data(VoxelData* data, size_t size, size_t size_power_of_2,
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
                    const uint64_t val_i =
                        (i < size) ? data[i].voxel_idx : voxel_hash_map::VoxelConstants::invalid_coord;
                    const uint64_t val_ixj =
                        (ixj < size) ? data[ixj].voxel_idx : voxel_hash_map::VoxelConstants::invalid_coord;

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
    SYCL_EXTERNAL static void reduce_local_data(VoxelData* sorted_data, size_t wg_size, sycl::nd_item<1> item) {
        const size_t local_id = item.get_local_id(0);

        // Find segments of same voxel indices and reduce them
        const auto current_voxel =
            local_id < wg_size ? sorted_data[local_id].voxel_idx : voxel_hash_map::VoxelConstants::invalid_coord;
        // Check if this is the start of a new voxel segment
        const bool is_segment_start = (current_voxel != voxel_hash_map::VoxelConstants::invalid_coord) &&
                                      ((local_id == 0) || (sorted_data[local_id - 1].voxel_idx != current_voxel));

        if (is_segment_start) {
            // Accumulate points to segment start
            for (size_t i = local_id + 1; i < wg_size && sorted_data[i].voxel_idx == current_voxel; ++i) {
                // change to invalid
                sorted_data[i].voxel_idx = voxel_hash_map::VoxelConstants::invalid_coord;

                // accumulate
                sorted_data[local_id].x += sorted_data[i].x;
                sorted_data[local_id].y += sorted_data[i].y;
                sorted_data[local_id].z += sorted_data[i].z;
                sorted_data[local_id].count += sorted_data[i].count;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    void add_point_cloud_(const PointCloudShared& pc) {
        const size_t N = pc.size();
        if (N == 0) return;

        // using `mem_advise to device` is too slow

        // add to voxel hash map
        shared_vector<uint32_t> voxel_num_vec(1, this->voxel_num_, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t work_group_size = (N + this->wg_size_add_point_cloud_ - 1) / this->wg_size_add_point_cloud_;
                const size_t global_size = work_group_size * this->wg_size_add_point_cloud_;

                // Find the next power of 2 that is >= size
                size_t power_of_2 = 1;
                while (power_of_2 < work_group_size) {
                    power_of_2 *= 2;
                }

                // Allocate local memory for work group operations
                const auto local_voxel_data = sycl::local_accessor<VoxelData>(work_group_size, h);

                // memory ptr
                const auto key_ptr = this->key_ptr_.get();
                const auto sum_ptr = this->sum_ptr_.get();
                const auto last_update_ptr = this->last_update_ptr_.get();

                const auto point_ptr = pc.points_ptr();

                const auto vs_inv = this->voxel_size_inv_;
                const auto cp = this->capacity_;
                const auto current = this->staleness_counter_;
                const auto max_probe = this->max_probe_length_;
                const auto voxel_num_ptr = voxel_num_vec.data();

                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_id(0);
                    const size_t local_id = item.get_local_id(0);

                    // Reduction on workgroup
                    if (i < N && local_id < work_group_size) {
                        const auto voxel_hash = kernel::compute_voxel_bit(point_ptr[i], vs_inv);

                        // set local data
                        local_voxel_data[local_id].voxel_idx = voxel_hash;
                        local_voxel_data[local_id].x = point_ptr[i].x();
                        local_voxel_data[local_id].y = point_ptr[i].y();
                        local_voxel_data[local_id].z = point_ptr[i].z();
                        local_voxel_data[local_id].count = 1;
                    }
                    // wait local
                    item.barrier(sycl::access::fence_space::local_space);

                    // sort within work group by voxel index
                    bitonic_sort_local_data(local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(),
                                            work_group_size, power_of_2, item);
                    // reduction
                    reduce_local_data(local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(),
                                      work_group_size, item);

                    if (i >= N || local_id >= work_group_size) return;

                    const auto local_hash = local_voxel_data[local_id].voxel_idx;
                    if (local_hash == VoxelConstants::invalid_coord) return;

                    const float local_x = local_voxel_data[local_id].x;
                    const float local_y = local_voxel_data[local_id].y;
                    const float local_z = local_voxel_data[local_id].z;
                    const float local_count = local_voxel_data[local_id].count;

                    // reduction on global memory
                    for (size_t j = 0; j < max_probe; ++j) {
                        const size_t slot_idx = (local_hash + j) % cp;

                        uint64_t expected = VoxelConstants::invalid_coord;
                        if (atomic_ref_uint64_t(key_ptr[slot_idx]).compare_exchange_strong(expected, local_hash)) {
                            // count up num of voxel
                            atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(1U);

                            // add point coord
                            atomic_ref_float(sum_ptr[slot_idx].x).fetch_add(local_x);
                            atomic_ref_float(sum_ptr[slot_idx].y).fetch_add(local_y);
                            atomic_ref_float(sum_ptr[slot_idx].z).fetch_add(local_z);

                            // count up num of points in voxel
                            atomic_ref_uint32_t(sum_ptr[slot_idx].count).fetch_add(local_count);

                            // last update
                            atomic_ref_uint32_t(last_update_ptr[slot_idx]).store(current);

                            break;
                        } else if (expected == local_hash) {
                            // add point coord
                            atomic_ref_float(sum_ptr[slot_idx].x).fetch_add(local_x);
                            atomic_ref_float(sum_ptr[slot_idx].y).fetch_add(local_y);
                            atomic_ref_float(sum_ptr[slot_idx].z).fetch_add(local_z);

                            // count up num of points in voxel
                            atomic_ref_uint32_t(sum_ptr[slot_idx].count).fetch_add(local_count);

                            // last update
                            atomic_ref_uint32_t(last_update_ptr[slot_idx]).store(current);

                            break;
                        }
                    }
                });
            })
            .wait();

        this->voxel_num_ = static_cast<size_t>(voxel_num_vec.at(0));
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
        this->make_device_ptr();
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

                const auto voxel_num_ptr = voxel_num_vec.data();

                const auto new_cp = new_capacity;
                const auto max_probe = this->max_probe_length_;

                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const uint32_t i = item.get_global_id(0);
                    if (i >= N) return;

                    const auto voxel_hash = old_key[i];
                    if (voxel_hash == VoxelConstants::invalid_coord) return;

                    for (size_t j = 0; j < max_probe; ++j) {
                        const size_t slot_idx = (voxel_hash + j) % new_cp;

                        uint64_t expected = VoxelConstants::invalid_coord;
                        if (atomic_ref_uint64_t(new_key[slot_idx]).compare_exchange_strong(expected, voxel_hash)) {
                            // count up num of voxel
                            atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(1U);

                            // copy to new container
                            // use fetch_add instead of store
                            // because the same key may be inserted at different positions
                            atomic_ref_float(new_sum[slot_idx].x).fetch_add(old_sum[i].x);
                            atomic_ref_float(new_sum[slot_idx].y).fetch_add(old_sum[i].y);
                            atomic_ref_float(new_sum[slot_idx].z).fetch_add(old_sum[i].z);
                            atomic_ref_uint32_t(new_sum[slot_idx].count).fetch_add(old_sum[i].count);

                            atomic_ref_uint32_t(new_last_update[slot_idx]).store(old_last_update[i]);
                            break;
                        } else if (expected == voxel_hash) {
                            atomic_ref_float(new_sum[slot_idx].x).fetch_add(old_sum[i].x);
                            atomic_ref_float(new_sum[slot_idx].y).fetch_add(old_sum[i].y);
                            atomic_ref_float(new_sum[slot_idx].z).fetch_add(old_sum[i].z);
                            atomic_ref_uint32_t(new_sum[slot_idx].count).fetch_add(old_sum[i].count);

                            atomic_ref_uint32_t(new_last_update[slot_idx]).store(old_last_update[i]);
                            break;
                        }
                    }
                });
            })
            .wait();
        this->voxel_num_ = static_cast<size_t>(voxel_num_vec.at(0));
    }
};

}  // namespace voxel_hash_map
}  // namespace algorithms
}  // namespace sycl_points
