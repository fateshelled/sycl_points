#pragma once

#include <sycl_points/points/point_cloud.hpp>

namespace {

struct VoxelConstants {
    static constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
    static constexpr int coord_bit_size = 21;                       // Bits to represent each voxel coordinate
    static constexpr size_t coord_bit_mask = (1 << 21) - 1;         // Bit mask
    static constexpr int coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive
};

}  // namespace

namespace sycl_points {

namespace algorithms {

namespace kernel {

SYCL_EXTERNAL inline uint64_t compute_voxel_bit(const PointType& point, const float inv_voxel_size) {
    if (!sycl::isfinite(point.x()) || !sycl::isfinite(point.y()) || !sycl::isfinite(point.z())) {
        return VoxelConstants::invalid_coord;
    }

    const auto coord0 = static_cast<int64_t>(sycl::floor(point.x() * inv_voxel_size)) + VoxelConstants::coord_offset;
    const auto coord1 = static_cast<int64_t>(sycl::floor(point.y() * inv_voxel_size)) + VoxelConstants::coord_offset;
    const auto coord2 = static_cast<int64_t>(sycl::floor(point.z() * inv_voxel_size)) + VoxelConstants::coord_offset;
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

class VoxelGridSYCL {
public:
    VoxelGridSYCL(const std::shared_ptr<sycl::queue>& queue_ptr, const float voxel_size)
        : queue_ptr_(queue_ptr), voxel_size_(voxel_size) {
        this->bit_ptr_ = std::make_shared<shared_vector<uint64_t>>(0, VoxelConstants::invalid_coord,
                                                                   shared_allocator<uint64_t>(*queue_ptr_));
        this->voxel_size_inv_ = 1.0f / this->voxel_size_;
    }

    void set_voxel_size(const float voxel_size) { voxel_size_ = voxel_size; }
    float get_voxel_size() const { return voxel_size_; }

    void downsampling(const PointContainerShared& points, PointContainerShared& result) {
        // Ref: https://github.com/koide3/gtsam_points/blob/master/src/gtsam_points/types/point_cloud_cpu_funcs.cpp
        // function: voxelgrid_sampling
        // MIT License
        const size_t N = points.size();
        if (N == 0) {
            result.resize(0);
            return;
        }
        const size_t work_group_size = sycl_utils::get_work_group_size(*queue_ptr_);
        const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;
        // compute bit on device
        this->bit_ptr_->resize(N, VoxelConstants::invalid_coord);
        {
            auto event = queue_ptr_->submit([&](sycl::handler& h) {
                // memory ptr
                const auto point_ptr = points.data();
                const auto bit_ptr = this->bit_ptr_->data();
                const auto voxel_size_inv = this->voxel_size_inv_;
                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)),
                               [=](sycl::nd_item<1> item) {
                                   const size_t i = item.get_global_id(0);
                                   if (i >= N) return;

                                   bit_ptr[i] = kernel::compute_voxel_bit(point_ptr[i], voxel_size_inv);
                               });
            });
            event.wait();
        }

        std::unordered_map<uint64_t, PointType> voxel_map;
        /* Kahan algorithm (high precision) */
        // {
        //   std::unordered_map<uint64_t, PointType> kahan_map;
        //   kahan_map.reserve(N / 2);

        //   for (size_t i = 0; i < N; ++i) {
        //     if (bits[i] == VoxelConstants::invalid_coord) continue;

        //     const auto it = voxel_map.find(bits[i]);
        //     if (it == voxel_map.end()) {
        //       voxel_map[bits[i]] = points[i];
        //     } else {
        //       const auto y = points[i] - kahan_map[bits[i]];
        //       const auto t = it->second + y;
        //       kahan_map[bits[i]] = (t - it->second) - y;
        //       it->second = t;
        //     }
        //   }
        // }
        /* Fast algorithm */
        {
            for (size_t i = 0; i < N; ++i) {
                if ((*this->bit_ptr_)[i] == VoxelConstants::invalid_coord) continue;
                const auto it = voxel_map.find((*this->bit_ptr_)[i]);
                if (it == voxel_map.end()) {
                    voxel_map[(*this->bit_ptr_)[i]] = points[i];
                } else {
                    it->second += points[i];
                }
            }
        }

        result.resize(voxel_map.size());
        size_t idx = 0;
        for (const auto& [_, point] : voxel_map) {
            result[idx++] = point / point.w();
        }
    }

    void downsampling(const PointCloudShared& cloud, PointCloudShared& result) {
        this->downsampling(*cloud.points, *result.points);
    }

private:
    std::shared_ptr<sycl::queue> queue_ptr_;
    float voxel_size_;
    float voxel_size_inv_;

    std::shared_ptr<shared_vector<uint64_t>> bit_ptr_ = nullptr;
};

}  // namespace algorithms
}  // namespace sycl_points
