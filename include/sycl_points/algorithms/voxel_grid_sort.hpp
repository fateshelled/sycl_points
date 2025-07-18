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

        // Step 3: Reset output counter
        (*this->output_counter_)[0] = 0;

        // Step 4: Execute sort-based downsampling kernel
        this->execute_sort_based_kernel(points, result, N);

        // Step 5: Get actual output size and resize result
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
            // NVIDIA: TODO optimize
            return std::min(max_work_group_size, 512UL);
        } else if (this->queue_.is_intel() && this->queue_.is_gpu()) {
            // Intel iGPU: TODO optimize
            return std::min(max_work_group_size, 256UL);
            // return std::min(max_work_group_size, compute_units * 10UL);
        } else {
            // CPU:
            return std::min(max_work_group_size, compute_units * 25UL);
        }
    }

    /// @brief Main kernel: compute voxel indices, sort, and reduce within work groups
    void execute_sort_based_kernel(const PointContainerShared& points, PointContainerShared& result, size_t N) {
        const size_t wg_size = this->optimal_wg_size_;
        const size_t num_work_groups = (N + wg_size - 1) / wg_size;
        const size_t global_size = num_work_groups * wg_size;

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                // Allocate local memory for work group operations
                auto local_voxel_data = sycl::local_accessor<VoxelData>(wg_size, h);
                auto local_output_count = sycl::local_accessor<uint32_t>(1, h);

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
                    auto group = item.get_group();

                    // Initialize local counter
                    if (local_id == 0) {
                        local_output_count[0] = 0;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

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

                    local_voxel_data[local_id].voxel_index = local_data.voxel_index;
                    eigen_utils::copy<4, 1>(local_data.point, local_voxel_data[local_id].point);

                    item.barrier(sycl::access::fence_space::local_space);

                    // Step 2: Sort within work group by voxel index
                    sort_local_data(local_voxel_data.get_pointer(), wg_size, item);

                    // Step 3: Reduce consecutive same voxel indices
                    reduce_and_output(local_voxel_data.get_pointer(), local_output_count.get_pointer(), output_points,
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

    /// @brief Efficient bitonic sort for power-of-2 work group sizes
    SYCL_EXTERNAL static void sort_local_data(VoxelData* data, size_t size, sycl::nd_item<1> item) {
        const size_t local_id = item.get_local_id(0);

        // Bitonic sort implementation optimized for voxel indices
        for (size_t k = 2; k <= size; k *= 2) {
            for (size_t j = k / 2; j > 0; j /= 2) {
                const size_t i = local_id;
                const size_t ixj = i ^ j;

                if (ixj > i && i < size && ixj < size) {
                    const bool should_swap = (data[i].voxel_index > data[ixj].voxel_index) == ((i & k) == 0);

                    if (should_swap) {
                        // Swap voxel data
                        const VoxelData temp = data[i];
                        std::swap(data[i].voxel_index, data[ixj].voxel_index);
                        eigen_utils::swap<4, 1>(data[i].point, data[ixj].point);
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    /// @brief Reduce consecutive same voxel indices and output results
    SYCL_EXTERNAL static void reduce_and_output(VoxelData* sorted_data, uint32_t* local_counter, PointType* output,
                                                uint32_t* global_counter, size_t wg_size, sycl::nd_item<1> item) {
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
                        // accumulated_point += sorted_data[i].point;
                        eigen_utils::add_zerocopy<4, 1>(accumulated_point, sorted_data[i].point);
                    }
                }

                if (accumulated_point.w() > 0.0f) {
                    // Compute average point
                    const PointType avg_point =
                        eigen_utils::multiply<4, 1>(accumulated_point, 1.0f / accumulated_point.w());
                    // const PointType avg_point = accumulated_point / static_cast<float>(point_count);

                    // Get global output index using atomic operation
                    const uint32_t output_idx =
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(
                            *global_counter)
                            .fetch_add(1);

                    // Write averaged point to output
                    eigen_utils::copy<4, 1>(avg_point, output[output_idx]);
                    // output[output_idx] = avg_point;

                    // Update local counter for statistics
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>(
                        *local_counter)
                        .fetch_add(1);
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