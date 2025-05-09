#pragma once

#include <mutex>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {

namespace algorithms {

namespace voxel_downsampling {

namespace {

struct VoxelConstants {
    static constexpr uint64_t invalid_coord = std::numeric_limits<uint64_t>::max();
    static constexpr uint8_t coord_bit_size = 21;                       // Bits to represent each voxel coordinate
    static constexpr int64_t coord_bit_mask = (1 << 21) - 1;            // Bit mask
    static constexpr int64_t coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive
};

}  // namespace

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

/// @brief Voxel grid downsampling with SYCL implementation
class VoxelGridSYCL {
public:
    /// @brief Constructor
    /// @param queue_ptr SYCL queue shared_ptr
    /// @param voxel_size voxel size
    VoxelGridSYCL(const std::shared_ptr<sycl::queue>& queue_ptr, const float voxel_size)
        : queue_ptr_(queue_ptr), voxel_size_(voxel_size) {
        this->bit_ptr_ = std::make_shared<shared_vector<uint64_t>>(0, VoxelConstants::invalid_coord,
                                                                   shared_allocator<uint64_t>(*queue_ptr_));
        this->voxel_size_inv_ = 1.0f / this->voxel_size_;
    }

    /// @brief Set voxel size
    /// @param voxel_size voxel size
    void set_voxel_size(const float voxel_size) { voxel_size_ = voxel_size; }
    /// @brief Get voxel size
    /// @param voxel_size voxel size
    float get_voxel_size() const { return voxel_size_; }

    /// @brief Voxel downsampling
    /// @param points Point Cloud
    /// @param result Voxel downsampled
    void downsampling(const PointContainerShared& points, PointContainerShared& result) {
        const size_t N = points.size();
        if (N == 0) {
            result.resize(0);
            return;
        }

        if (this->bit_ptr_->size() < N) {
            this->bit_ptr_->resize(N);
        }

        // compute bit on device
        this->compute_voxel_bit(points);

        // compute Voxel map on host
        const auto voxel_map = this->compute_voxel_map(points);

        // Voxel map to point cloud on host
        this->voxel_map_to_cloud(voxel_map, result);
    }

    /// @brief Voxel downsampling
    /// @param points Point Cloud
    /// @param result Voxel downsampled
    void downsampling(const PointCloudShared& cloud, PointCloudShared& result) {
        this->downsampling(*cloud.points, *result.points);
    }

private:
    std::shared_ptr<sycl::queue> queue_ptr_;
    float voxel_size_;
    float voxel_size_inv_;

    std::shared_ptr<shared_vector<uint64_t>> bit_ptr_ = nullptr;

    void compute_voxel_bit(const PointContainerShared& points) {
        const size_t N = points.size();

        // mem_advise set to device
        {
            sycl_utils::mem_advise::set_accessed_by_device(*this->queue_ptr_, this->bit_ptr_->data(), N);
            sycl_utils::mem_advise::set_accessed_by_device(*this->queue_ptr_, points.data(), N);
        }

        const size_t work_group_size = sycl_utils::get_work_group_size(*queue_ptr_);
        const size_t global_size = sycl_utils::get_global_size(N, work_group_size);
        auto event = queue_ptr_->submit([&](sycl::handler& h) {
            // memory ptr
            const auto point_ptr = points.data();
            const auto bit_ptr = this->bit_ptr_->data();
            const auto voxel_size_inv = this->voxel_size_inv_;
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const uint32_t i = item.get_global_id(0);
                if (i >= N) return;

                bit_ptr[i] = kernel::compute_voxel_bit(point_ptr[i], voxel_size_inv);
            });
        });
        event.wait();

        // mem_advise clear
        {
            sycl_utils::mem_advise::clear_accessed_by_device(*this->queue_ptr_, this->bit_ptr_->data(), N);
            sycl_utils::mem_advise::clear_accessed_by_device(*this->queue_ptr_, points.data(), N);
        }
    }

    std::unordered_map<uint64_t, PointType> compute_voxel_map(const PointContainerShared& points) const {
        const size_t N = points.size();

        // mem_advise set to host
        {
            sycl_utils::mem_advise::set_accessed_by_host(*this->queue_ptr_, this->bit_ptr_->data(), N);
            sycl_utils::mem_advise::set_accessed_by_host(*this->queue_ptr_, points.data(), N);
        }

        std::unordered_map<uint64_t, PointType> voxel_map;
        {
            for (size_t i = 0; i < N; ++i) {
                const auto voxel_bit = (*this->bit_ptr_)[i];
                if (voxel_bit == VoxelConstants::invalid_coord) continue;
                const auto it = voxel_map.find(voxel_bit);
                if (it == voxel_map.end()) {
                    voxel_map[voxel_bit] = points[i];
                } else {
                    it->second += points[i];
                }
            }
        }
        // mem_advise clear
        {
            sycl_utils::mem_advise::clear_accessed_by_host(*this->queue_ptr_, this->bit_ptr_->data(), N);
            sycl_utils::mem_advise::clear_accessed_by_host(*this->queue_ptr_, points.data(), N);
        }

        return voxel_map;
    }

    void voxel_map_to_cloud(const std::unordered_map<uint64_t, PointType>& voxel_map,
                            PointContainerShared& result) const {
        const size_t N = voxel_map.size();
        result.clear();
        result.resize(N);
        // mem_advise set to host
        sycl_utils::mem_advise::set_accessed_by_host(*this->queue_ptr_, result.data(), N);
        // to point cloud
        size_t idx = 0;
        for (const auto& [_, point] : voxel_map) {
            result[idx++] = point / point.w();
        }
        // mem_advise clear
        sycl_utils::mem_advise::clear_accessed_by_host(*this->queue_ptr_, result.data(), N);
    }
};

}  // namespace voxel_downsampling
}  // namespace algorithms
}  // namespace sycl_points
