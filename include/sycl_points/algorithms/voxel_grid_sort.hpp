#pragma once

#include <sycl_points/algorithms/voxel_hash_map.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {
namespace algorithms {
namespace filter {

/// @brief High-performance sort-based voxel grid downsampling
class VoxelGridSort {
public:
    using Ptr = std::shared_ptr<VoxelGridSort>;

    /// @brief Constructor
    /// @param queue SYCL queue
    /// @param voxel_size voxel size
    VoxelGridSort(const sycl_points::sycl_utils::DeviceQueue& queue, const float voxel_size)
        : queue_(queue), voxel_size_(voxel_size), voxel_size_inv_(1.0f / voxel_size) {
        // Initialize shared memory buffers
        this->output_counter_ = std::make_shared<shared_vector<uint32_t>>(1, 0, *this->queue_.ptr);

        // Calculate optimal work group size for sorting
        this->optimal_wg_size_ = calculate_optimal_workgroup_size();
    }

    /// @brief Set voxel size
    /// @param voxel_size voxel size
    void set_voxel_size(const float voxel_size) {
        this->voxel_size_ = voxel_size;
        this->voxel_size_inv_ = 1.0f / voxel_size;
    }

    /// @brief Get voxel size
    float get_voxel_size() const { return this->voxel_size_; }

    /// @brief Sort-based voxel downsampling with work-group level optimization
    /// @param points Input point cloud
    /// @param result Output downsampled point cloud
    void downsampling(const PointContainerShared& points, PointContainerShared& result) {
        const size_t N = points.size();
        if (N == 0) {
            result.resize(0);
            return;
        }

        // Step 1: Estimate maximum possible output size (pessimistic)
        const size_t max_output_size = N;  // In worst case, every point creates a new voxel
        result.resize(max_output_size);

        // Step 2: Reset output counter
        (*this->output_counter_)[0] = 0;

        // Step 3: Execute sort-based downsampling kernel
        this->execute_sort_based_kernel(points, result, N);

        // Step 4: Get actual output size and resize result
        const uint32_t actual_output_size = (*this->output_counter_)[0];
        result.resize(actual_output_size);
    }

    /// @brief Sort-based voxel downsampling for point cloud
    /// @param cloud Input point cloud
    /// @param result Output downsampled point cloud
    void downsampling(const PointCloudShared& cloud, PointCloudShared& result) {
        this->downsampling(*cloud.points, *result.points);
    }

private:
    sycl_points::sycl_utils::DeviceQueue queue_;
    float voxel_size_;
    float voxel_size_inv_;
    size_t optimal_wg_size_;

    shared_vector_ptr<uint32_t> output_counter_;

    /// @brief Calculate optimal work group size based on device characteristics
    size_t calculate_optimal_workgroup_size() const {
        const size_t max_work_group_size =
            this->queue_.get_device().get_info<sycl::info::device::max_work_group_size>();
        const size_t compute_units = this->queue_.get_device().get_info<sycl::info::device::max_compute_units>();

        // Device-specific optimization
        if (this->queue_.is_nvidia()) {
            // NVIDIA:
            return std::min(max_work_group_size, 512UL);
        } else if (this->queue_.is_intel() && this->queue_.is_gpu()) {
            // Intel iGPU:
            return std::min(max_work_group_size, compute_units * 8UL);
        } else {
            // CPU:
            // return std::min(max_work_group_size, compute_units * 25UL);
            return std::min(max_work_group_size, compute_units * 50UL);
        }
        return std::min(max_work_group_size, 512UL);
    }

    /// @brief Main kernel: compute voxel indices, sort, and reduce within work groups
    void execute_sort_based_kernel(const PointContainerShared& points, PointContainerShared& result, size_t N) {
        const size_t wg_size = this->optimal_wg_size_;
        const size_t num_work_groups = (N + wg_size - 1) / wg_size;
        const size_t global_size = num_work_groups * wg_size;

        // Find the next power of 2 that is >= size
        size_t power_of_2 = 1;
        while (power_of_2 < wg_size) {
            power_of_2 *= 2;
        }

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                // Allocate local memory for work group operations
                auto local_voxel_data = sycl::local_accessor<VoxelData>(wg_size, h);

                // Capture variables for device code
                const auto input_points = points.data();
                const auto output_points = result.data();
                const auto global_counter = this->output_counter_->data();
                const auto voxel_size_inv = this->voxel_size_inv_;
                const auto total_points = N;

                h.parallel_for(sycl::nd_range<1>(global_size, wg_size), [=](sycl::nd_item<1> item) {
                    const size_t global_id = item.get_global_id(0);
                    const size_t local_id = item.get_local_id(0);
                    const size_t group_id = item.get_group(0);

                    // Step 1: Load points and compute voxel indices
                    VoxelData local_data;
                    if (global_id < total_points) {
                        const auto point = input_points[global_id];
                        const uint64_t voxel_idx = voxel_hash_map::kernel::compute_voxel_bit(point, voxel_size_inv);

                        local_data.voxel_index = voxel_idx;
                        if (voxel_idx != voxel_hash_map::VoxelConstants::invalid_coord) {
                            eigen_utils::copy<4, 1>(point, local_data.point);
                        }
                    }

                    copy_voxel_data(local_data, local_voxel_data[local_id]);

                    item.barrier(sycl::access::fence_space::local_space);

                    // Step 2: Sort within work group by voxel index
                    bitonic_sort_local_data(local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(), wg_size,
                                            power_of_2, item);

                    // Step 3: Reduce consecutive same voxel indices
                    reduce_and_output(local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(), output_points,
                                      global_counter, wg_size, item);
                });
            })
            .wait();
    }

private:
    /// @brief Structure for local work group processing
    struct VoxelData {
        uint64_t voxel_index = voxel_hash_map::VoxelConstants::invalid_coord;
        PointType point = PointType::Zero();

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    SYCL_EXTERNAL static void copy_voxel_data(const VoxelData& src, VoxelData& dst) {
        dst.voxel_index = src.voxel_index;
        eigen_utils::copy<4, 1>(src.point, dst.point);
    }

    SYCL_EXTERNAL static void swap_voxel_data(VoxelData& a, VoxelData& b) {
        std::swap(a.voxel_index, b.voxel_index);
        eigen_utils::swap<4, 1>(a.point, b.point);
    }

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
                        (i < size) ? data[i].voxel_index : voxel_hash_map::VoxelConstants::invalid_coord;
                    const uint64_t val_ixj =
                        (ixj < size) ? data[ixj].voxel_index : voxel_hash_map::VoxelConstants::invalid_coord;

                    // Determine if swap is needed based on virtual values
                    const bool should_swap = (val_i > val_ixj) == ascending;

                    // Perform actual swap only if both indices are within real data
                    if (should_swap && i < size && ixj < size) {
                        swap_voxel_data(data[i], data[ixj]);
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    /// @brief Reduce consecutive same voxel indices and output results
    SYCL_EXTERNAL static void reduce_and_output(VoxelData* sorted_data, PointType* output, uint32_t* global_counter,
                                                size_t wg_size, sycl::nd_item<1> item) {
        const size_t local_id = item.get_local_id(0);

        // Find segments of same voxel indices and reduce them
        if (local_id < wg_size) {
            const auto current_voxel = sorted_data[local_id].voxel_index;

            // Skip invalid voxels
            if (current_voxel == voxel_hash_map::VoxelConstants::invalid_coord ||
                sorted_data[local_id].point.w() != 1.0f) {
                return;
            }

            // Check if this is the start of a new voxel segment
            bool is_segment_start = (local_id == 0) || (sorted_data[local_id - 1].voxel_index != current_voxel);

            if (is_segment_start) {
                // Accumulate all points in this voxel segment
                PointType accumulated_point = PointType::Zero();

                for (size_t i = local_id; i < wg_size && sorted_data[i].voxel_index == current_voxel; ++i) {
                    if (sorted_data[i].point.w() > 0.0f) {
                        eigen_utils::add_zerocopy<4, 1>(accumulated_point, sorted_data[i].point);
                    }
                }

                if (accumulated_point.w() > 0.0f) {
                    // Compute average point
                    const PointType avg_point =
                        eigen_utils::multiply<4, 1>(accumulated_point, 1.0f / accumulated_point.w());

                    // Get global output index using atomic operation
                    const uint32_t output_idx =
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(
                            *global_counter)
                            .fetch_add(1);

                    // Write averaged point to output
                    eigen_utils::copy<4, 1>(avg_point, output[output_idx]);
                }
            }
        }
    }
};

/// @brief Factory function for backward compatibility
inline VoxelGridSort::Ptr create_sort_based_voxel_grid(const sycl_utils::DeviceQueue& queue, float voxel_size) {
    return std::make_shared<VoxelGridSort>(queue, voxel_size);
}

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points