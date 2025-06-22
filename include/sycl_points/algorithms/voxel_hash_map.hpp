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
    /// @param capacity hash table initial capacity
    VoxelHashMap(const sycl_utils::DeviceQueue& queue, const float voxel_size, size_t capacity = 10000)
        : queue_(queue), voxel_size_(voxel_size), voxel_size_inv_(1.0f / voxel_size), capacity_(capacity) {
        this->key_ptr_ =
            std::make_shared<shared_vector<uint64_t>>(capacity, VoxelConstants::invalid_coord, *this->queue_.ptr);
        this->sum_ptr_ =
            std::make_shared<shared_vector<VoxelPoint>>(capacity, VoxelPoint{0.0f, 0.0f, 0.0f, 0}, *this->queue_.ptr);
        this->last_update_ptr_ = std::make_shared<shared_vector<uint32_t>>(capacity, 0, *this->queue_.ptr);

        this->downsampled_ptr_ = std::make_shared<PointContainerShared>(0, *this->queue_.ptr);
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

    /// @brief add PointCloud to voxel map
    /// @param pc Point Cloud
    void add_point_cloud(const PointCloudShared& pc) {
        const size_t N = pc.size();

        if (N > 0) {
            // add PointCloud to voxel map
            this->add_point_cloud_(pc);
        }

        // remove old data
        if (this->staleness_counter_ % this->remove_old_data_cycle_ == 0) {
            this->remove_old_data();
        }

        // rehash
        if (this->rehash_threshold_ < (float)this->voxel_num_ / (float)this->capacity_) {
            this->rehash(this->capacity_ * 2);
        }

        // increment counter
        ++this->staleness_counter_;
    }

    std::shared_ptr<PointContainerShared> downsampling() {
        if (this->voxel_num_ == 0) {
            return nullptr;
        }

        // voxel hash map to point cloud
        this->downsampled_ptr_->resize(this->voxel_num_);
        shared_vector<size_t> point_num_vec(1, 0, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(this->capacity_);

                // memory ptr
                const auto key_ptr = this->key_ptr_->data();
                const auto sum_ptr = this->sum_ptr_->data();

                const auto downsampled_ptr = this->downsampled_ptr_->data();

                const auto point_num_ptr = point_num_vec.data();
                const auto cp = this->capacity_;
                const auto vx_num = this->voxel_num_;
                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const uint32_t i = item.get_global_id(0);
                    if (i >= cp) return;

                    if (key_ptr[i] == VoxelConstants::invalid_coord) return;
                    const auto sum = sum_ptr[i];
                    if (sum.count == 0) return;

                    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_num(
                        point_num_ptr[0]);

                    const auto output_idx = atomic_num.fetch_add((size_t)1);  // core dump
                    if (output_idx >= vx_num) return;

                    downsampled_ptr[output_idx].x() = sum.x / sum.count;
                    downsampled_ptr[output_idx].y() = sum.y / sum.count;
                    downsampled_ptr[output_idx].z() = sum.z / sum.count;
                    downsampled_ptr[output_idx].w() = 1.0f;
                });
            })
            .wait();
        return this->downsampled_ptr_;
    }

private:
    struct VoxelPoint {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        uint32_t count = 0;
    };
    sycl_utils::DeviceQueue queue_;
    float voxel_size_;
    float voxel_size_inv_;
    size_t capacity_;

    shared_vector_ptr<uint64_t> key_ptr_ = nullptr;
    shared_vector_ptr<VoxelPoint> sum_ptr_ = nullptr;

    shared_vector_ptr<uint32_t> last_update_ptr_ = nullptr;
    uint32_t staleness_counter_ = 0;
    uint32_t max_staleness_ = 100;
    uint32_t remove_old_data_cycle_ = 10;

    float rehash_threshold_ = 0.7f;

    std::shared_ptr<PointContainerShared> downsampled_ptr_ = nullptr;
    size_t voxel_num_ = 0;

    void add_point_cloud_(const PointCloudShared& pc) {
        const size_t N = pc.size();
        // add to voxel hash map
        shared_vector<size_t> voxel_num_vec(1, this->voxel_num_, *this->queue_.ptr);
        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                // memory ptr
                const auto key_ptr = this->key_ptr_->data();
                const auto sum_ptr = this->sum_ptr_->data();
                const auto last_update_ptr = this->last_update_ptr_->data();

                const auto point_ptr = pc.points_ptr();

                const auto vs_inv = this->voxel_size_inv_;
                const auto cp = this->capacity_;
                const auto current = this->staleness_counter_;
                const auto voxel_num_ptr = voxel_num_vec.data();
                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const uint32_t i = item.get_global_id(0);
                    if (i >= N) return;

                    const auto voxel_hash = kernel::compute_voxel_bit(point_ptr[i], vs_inv);
                    if (voxel_hash == VoxelConstants::invalid_coord) return;

                    for (size_t j = 0; j < 100; ++j) {
                        const auto slot_idx = (voxel_hash + j) % cp;
                        sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_key(
                            key_ptr[slot_idx]);

                        uint64_t expected = VoxelConstants::invalid_coord;
                        if (atomic_key.compare_exchange_weak(expected, voxel_hash)) {
                            sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>
                                atomic_num(voxel_num_ptr[0]);
                            atomic_num.fetch_add((size_t)1);
                        }
                        if (key_ptr[slot_idx] == voxel_hash) {
                            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>(
                                sum_ptr[slot_idx].x)
                                .fetch_add(point_ptr[i].x());

                            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>(
                                sum_ptr[slot_idx].y)
                                .fetch_add(point_ptr[i].y());

                            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>(
                                sum_ptr[slot_idx].z)
                                .fetch_add(point_ptr[i].z());

                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(
                                sum_ptr[slot_idx].count)
                                .fetch_add(1);

                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(
                                last_update_ptr[slot_idx])
                                .store(current);
                            break;
                        }
                    }
                });
            })
            .wait();
        this->voxel_num_ = voxel_num_vec.at(0);
    }

    void rehash(size_t new_capacity) {
        this->capacity_ = new_capacity;
        this->key_ptr_->resize(new_capacity, VoxelConstants::invalid_coord);
        this->sum_ptr_->resize(new_capacity, VoxelPoint{0.0f, 0.0f, 0.0f, 0});
        this->last_update_ptr_->resize(new_capacity, 0);
        // TODO

    }

    void remove_old_data() {
        // TODO
    }
};

}  // namespace voxel_hash_map
}  // namespace algorithms
}  // namespace sycl_points
