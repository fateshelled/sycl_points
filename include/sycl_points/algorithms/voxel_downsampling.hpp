#pragma once

#include <sycl_points/algorithms/common/voxel_constants.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {
namespace algorithms {
namespace filter {

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

    if (coord0 < 0 || VoxelConstants::coord_bit_mask < coord0 ||  //
        coord1 < 0 || VoxelConstants::coord_bit_mask < coord1 ||  //
        coord2 < 0 || VoxelConstants::coord_bit_mask < coord2) {
        return VoxelConstants::invalid_coord;
    }

    // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
    return (static_cast<uint64_t>(coord0 & VoxelConstants::coord_bit_mask)
            << (VoxelConstants::coord_bit_size * CartesianCoordComponent::X)) |
           (static_cast<uint64_t>(coord1 & VoxelConstants::coord_bit_mask)
            << (VoxelConstants::coord_bit_size * CartesianCoordComponent::Y)) |
           (static_cast<uint64_t>(coord2 & VoxelConstants::coord_bit_mask)
            << (VoxelConstants::coord_bit_size * CartesianCoordComponent::Z));
}

}  // namespace kernel

/// @brief Voxel grid downsampling with SYCL implementation
class VoxelGrid {
public:
    using Ptr = std::shared_ptr<VoxelGrid>;

    /// @brief Constructor
    /// @param queue SYCL queue
    /// @param voxel_size voxel size
    VoxelGrid(const sycl_points::sycl_utils::DeviceQueue& queue, const float voxel_size)
        : queue_(queue), voxel_size_(voxel_size) {
        if (voxel_size <= 0.0f) {
            throw std::invalid_argument("voxel_size must be positive");
        }
        this->bit_ptr_ = std::make_shared<shared_vector<uint64_t>>(0, *this->queue_.ptr);
        this->voxel_size_inv_ = 1.0f / this->voxel_size_;
        this->min_voxel_count_ = 1;
    }

    /// @brief Set voxel size
    /// @param voxel_size voxel size
    void set_voxel_size(const float voxel_size) {
        if (voxel_size <= 0.0f) {
            throw std::invalid_argument("voxel_size must be positive");
        }
        this->voxel_size_ = voxel_size;
        this->voxel_size_inv_ = 1.0f / this->voxel_size_;
    }

    /// @brief Get voxel size
    /// @param voxel_size voxel size
    float get_voxel_size() const { return this->voxel_size_; }

    void set_min_voxel_count(const size_t min_voxel_count) { this->min_voxel_count_ = min_voxel_count; }

    /// @brief Voxel downsampling
    /// @param points Point Cloud
    /// @param result Voxel downsampled
    void downsampling(const PointContainerShared& points, PointContainerShared& result) {
        const size_t N = points.size();
        if (N == 0) {
            result.resize(0);
            return;
        }
        // compute Voxel map on host
        const auto voxel_map = this->compute_voxel_bit_and_voxel_map(points);

        // Voxel map to point cloud on host
        this->voxel_map_to_cloud(voxel_map, result);
    }

    /// @brief Voxel downsampling
    /// @param points Point Cloud
    /// @param result Voxel downsampled
    void downsampling(const PointCloudShared& cloud, PointCloudShared& result) {
        const size_t N = cloud.size();
        if (N == 0) {
            result.resize_points(0);
            return;
        }
        // compute Voxel map on host
        const auto voxel_map = this->compute_voxel_bit_and_voxel_map(*cloud.points);

        if (cloud.has_rgb() || cloud.has_intensity()) {
            // compute RGB map on host
            const auto rgb_map = this->compute_voxel_map<RGBType>(*cloud.rgb);
            const auto intensity_map = this->compute_voxel_map<float>(*cloud.intensities);
            // Voxel map to point cloud on host
            this->voxel_map_to_cloud(voxel_map, rgb_map, intensity_map, result);
        } else {
            // Voxel map to point cloud on host
            this->voxel_map_to_cloud(voxel_map, *result.points);
        }
    }

private:
    sycl_points::sycl_utils::DeviceQueue queue_;
    float voxel_size_;
    float voxel_size_inv_;
    size_t min_voxel_count_;

    shared_vector_ptr<uint64_t> bit_ptr_ = nullptr;

    void compute_voxel_bit(const PointContainerShared& points) {
        const size_t N = points.size();

        // mem_advise set to device
        {
            this->queue_.set_accessed_by_device(this->bit_ptr_->data(), N);
            this->queue_.set_accessed_by_device(points.data(), N);
        }

        const size_t work_group_size = this->queue_.get_work_group_size();
        const size_t global_size = this->queue_.get_global_size(N);
        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
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
            this->queue_.clear_accessed_by_device(this->bit_ptr_->data(), N);
            this->queue_.clear_accessed_by_device(points.data(), N);
        }
    }

    template <typename T>
    std::unordered_map<uint64_t, T> compute_voxel_map(const shared_vector<T>& data) const {
        const size_t N = data.size();
        if (N == 0) {
            return {};
        }

        // mem_advise set to host
        {
            this->queue_.set_accessed_by_host(this->bit_ptr_->data(), N);
            this->queue_.set_accessed_by_host(data.data(), N);
        }

        std::unordered_map<uint64_t, T> voxel_map;
        {
            for (size_t i = 0; i < N; ++i) {
                const auto voxel_bit = (*this->bit_ptr_)[i];
                if (voxel_bit == VoxelConstants::invalid_coord) continue;
                const auto it = voxel_map.find(voxel_bit);
                if (it == voxel_map.end()) {
                    voxel_map[voxel_bit] = data[i];
                } else {
                    it->second += data[i];
                }
            }
        }
        // mem_advise clear
        {
            this->queue_.clear_accessed_by_host(this->bit_ptr_->data(), N);
            this->queue_.clear_accessed_by_host(data.data(), N);
        }

        return voxel_map;
    }

    std::unordered_map<uint64_t, PointType> compute_voxel_bit_and_voxel_map(const PointContainerShared& points) {
        const size_t N = points.size();
        if (this->bit_ptr_->size() < N) {
            this->bit_ptr_->resize(N);
        }

        // compute bit on device
        this->compute_voxel_bit(points);

        // compute Voxel map on host
        return this->compute_voxel_map<PointType>(points);
    }

    void voxel_map_to_cloud(const std::unordered_map<uint64_t, PointType>& voxel_map,
                            PointContainerShared& result) const {
        const size_t N = voxel_map.size();
        result.clear();
        result.reserve(N);
        const float min_voxel_count = static_cast<float>(this->min_voxel_count_);
        for (const auto& [_, point] : voxel_map) {
            const auto point_count = point.w();
            if (point_count >= min_voxel_count) {
                result.push_back(point / point_count);
            }
        }
    }

    void voxel_map_to_cloud(const std::unordered_map<uint64_t, PointType>& voxel_map,
                            const std::unordered_map<uint64_t, RGBType>& voxel_map_rgb,
                            const std::unordered_map<uint64_t, float>& voxel_map_intensity,
                            PointCloudShared& result) const {
        const size_t N = voxel_map.size();
        const bool has_rgb = voxel_map_rgb.size() == N;
        const bool has_intensity = voxel_map_intensity.size() == N;
        result.clear();
        result.reserve_points(N);
        if (has_rgb) {
            result.reserve_rgb(voxel_map_rgb.size());
        }
        if (has_intensity) {
            result.reserve_intensities(voxel_map_intensity.size());
        }

        // to point cloud
        const float min_voxel_count = static_cast<float>(this->min_voxel_count_);
        for (const auto& [voxel_idx, point] : voxel_map) {
            const auto point_count = point.w();
            if (point_count >= min_voxel_count) {
                result.points->emplace_back(point / point_count);
                if (has_rgb) result.rgb->emplace_back(voxel_map_rgb.at(voxel_idx) / point_count);
                if (has_intensity) result.intensities->emplace_back(voxel_map_intensity.at(voxel_idx) / point_count);
            }
        }
    }
};

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
