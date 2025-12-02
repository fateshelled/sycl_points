#pragma once

#include <cstdint>
#include <stdexcept>
#include <sycl_points/algorithms/common/coordinate_system.hpp>
#include <sycl_points/algorithms/common/voxel_constants.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {
namespace algorithms {
namespace filter {

namespace kernel {

/// @brief Computes a 64-bit key for a point based on its polar coordinates.
/// @details This function converts a 3D point from Cartesian to polar coordinates (distance, elevation, azimuth)
///          and then quantizes these coordinates into a single 64-bit integer key. This key is used for voxelization.
///          The coordinate system (LIDAR or CAMERA) affects how angles are calculated.
/// @tparam coord_system The coordinate system to use for conversion (LIDAR or CAMERA).
/// @param point The input 3D point.
/// @param distance_voxel_size_inv Inverse of the voxel size for the radial distance.
/// @param elevation_voxel_size_inv Inverse of the voxel size for the elevation angle.
/// @param azimuth_voxel_size_inv Inverse of the voxel size for the azimuth angle.
/// @return A 64-bit key representing the polar voxel, or `VoxelConstants::invalid_coord` if the point is invalid.
template <CoordinateSystem coord_system = CoordinateSystem::LIDAR>
SYCL_EXTERNAL inline uint64_t compute_polar_bit(const PointType& point, const float distance_voxel_size_inv,
                                                const float elevation_voxel_size_inv,
                                                const float azimuth_voxel_size_inv) {
    // Coordinate system (REP-103)
    // LiDAR: https://ros.org/reps/rep-0103.html#axis-orientation
    // x: forward
    // y: left
    // z: up
    //        z |  / x
    //          | /
    //          |/
    // y <---------------
    //         /|
    //        / |
    //       /  |
    //
    // Camera: https://ros.org/reps/rep-0103.html#suffix-frames
    // x: right
    // y: down
    // z: forward
    //          |  / z
    //          | /
    //          |/
    //   ---------------> x
    //         /|
    //        / |
    //       /  | y

    if (!sycl::isfinite(point.x()) || !sycl::isfinite(point.y()) || !sycl::isfinite(point.z())) {
        return VoxelConstants::invalid_coord;
    }

    const float r = sycl::sqrt(point.x() * point.x() + point.y() * point.y() + point.z() * point.z());
    if (r == 0.0f) {
        return VoxelConstants::invalid_coord;
    }
    float azimuth, elevation;
    if constexpr (coord_system == CoordinateSystem::LIDAR) {
        const float x2y2 = point.x() * point.x() + point.y() * point.y();
        if (x2y2 == 0.0f) {
            return VoxelConstants::invalid_coord;
        }
        azimuth = sycl::atan2(point.y(), point.x());
        elevation = sycl::atan2(point.z(), sycl::sqrt(x2y2));
    } else if constexpr (coord_system == CoordinateSystem::CAMERA) {
        const float x2z2 = point.x() * point.x() + point.z() * point.z();
        if (x2z2 == 0.0f) {
            return VoxelConstants::invalid_coord;
        }
        azimuth = sycl::atan2(point.x(), point.z());
        elevation = sycl::atan2(-point.y(), sycl::sqrt(x2z2));
    } else {
        // not support type
        return VoxelConstants::invalid_coord;
    }

    const auto coord0 = static_cast<int64_t>(sycl::floor(r * distance_voxel_size_inv)) + VoxelConstants::coord_offset;
    const auto coord1 =
        static_cast<int64_t>(sycl::floor(elevation * elevation_voxel_size_inv)) + VoxelConstants::coord_offset;
    const auto coord2 =
        static_cast<int64_t>(sycl::floor(azimuth * azimuth_voxel_size_inv)) + VoxelConstants::coord_offset;

    if (coord0 < 0 || VoxelConstants::coord_bit_mask < coord0 ||  //
        coord1 < 0 || VoxelConstants::coord_bit_mask < coord1 ||  //
        coord2 < 0 || VoxelConstants::coord_bit_mask < coord2) {
        return VoxelConstants::invalid_coord;
    }

    // Encode range, polar angle, and azimuth angle into a 64-bit key
    return (static_cast<uint64_t>(coord0 & VoxelConstants::coord_bit_mask)
            << (VoxelConstants::coord_bit_size * PolarCoordComponent::DISTANCE)) |
           (static_cast<uint64_t>(coord1 & VoxelConstants::coord_bit_mask)
            << (VoxelConstants::coord_bit_size * PolarCoordComponent::POLAR)) |
           (static_cast<uint64_t>(coord2 & VoxelConstants::coord_bit_mask)
            << (VoxelConstants::coord_bit_size * PolarCoordComponent::AZIMUTH));
}

}  // namespace kernel

/// @brief Polar grid downsampling with SYCL implementation
class PolarGrid {
public:
    using Ptr = std::shared_ptr<PolarGrid>;

    /// @brief Constructor
    /// @param queue SYCL queue
    /// @param distance_voxel_size voxel size for radial distance
    /// @param elevation_voxel_size voxel size for elevation angle (radian)
    /// @param azimuth_voxel_size voxel size for azimuth angle (radian)
    /// @param coord The coordinate system to use for polar conversion.
    PolarGrid(const sycl_points::sycl_utils::DeviceQueue& queue, float distance_voxel_size, float elevation_voxel_size,
              float azimuth_voxel_size, CoordinateSystem coord = CoordinateSystem::LIDAR)  //
        : queue_(queue),
          distance_voxel_size_(distance_voxel_size),
          elevation_voxel_size_(elevation_voxel_size),
          azimuth_voxel_size_(azimuth_voxel_size),
          coord_(coord) {
        if (distance_voxel_size <= 0.0f || elevation_voxel_size <= 0.0f || azimuth_voxel_size <= 0.0f) {
            throw std::invalid_argument("voxel sizes must be positive");
        }
        this->bit_ptr_ = std::make_shared<shared_vector<uint64_t>>(0, *this->queue_.ptr);
        this->distance_voxel_size_inv_ = 1.0f / this->distance_voxel_size_;
        this->elevation_voxel_size_inv_ = 1.0f / this->elevation_voxel_size_;
        this->azimuth_voxel_size_inv_ = 1.0f / this->azimuth_voxel_size_;
        this->min_voxel_count_ = 1;
    }

    /// @brief Set the voxel size for the radial distance.
    /// @param size The new distance voxel size. Must be positive.
    void set_distance_voxel_size(const float size) {
        if (size <= 0.0f) {
            throw std::invalid_argument("distance_voxel_size must be positive");
        }
        this->distance_voxel_size_ = size;
        this->distance_voxel_size_inv_ = 1.0f / size;
    }
    /// @brief Get the current distance voxel size.
    /// @return The distance voxel size.
    float get_distance_voxel_size() const { return this->distance_voxel_size_; }

    /// @brief Set the voxel size for the polar (elevation) angle.
    /// @param size The new polar voxel size in radians. Must be positive.
    void set_elevation_voxel_size(const float size) {
        if (size <= 0.0f) {
            throw std::invalid_argument("elevation_voxel_size must be positive");
        }
        this->elevation_voxel_size_ = size;
        this->elevation_voxel_size_inv_ = 1.0f / size;
    }
    /// @brief Get the current polar (elevation) voxel size.
    /// @return The polar voxel size in radians.
    float get_elevation_voxel_size() const { return this->elevation_voxel_size_; }

    /// @brief Set the voxel size for the azimuth angle.
    /// @param size The new azimuth voxel size in radians. Must be positive.
    void set_azimuth_voxel_size(const float size) {
        if (size <= 0.0f) {
            throw std::invalid_argument("azimuth_voxel_size must be positive");
        }
        this->azimuth_voxel_size_ = size;
        this->azimuth_voxel_size_inv_ = 1.0f / size;
    }
    /// @brief Get the current azimuth voxel size.
    /// @return The azimuth voxel size in radians.
    float get_azimuth_voxel_size() const { return this->azimuth_voxel_size_; }

    /// @brief Set the minimum number of points required in a voxel to be kept.
    /// @param min_voxel_count The minimum number of points.
    void set_min_voxel_count(const size_t min_voxel_count) { this->min_voxel_count_ = min_voxel_count; }
    /// @brief Get the minimum number of points required in a voxel.
    /// @return The minimum number of points.
    size_t get_min_voxel_count() const { return this->min_voxel_count_; }

    /// @brief Set the coordinate system used for polar conversion.
    /// @param coord The coordinate system (LIDAR or CAMERA).
    void set_coordinate_system(const CoordinateSystem coord) { this->coord_ = coord; }
    /// @brief Get the current coordinate system.
    /// @return The current coordinate system.
    CoordinateSystem get_coordinate_system() const { return this->coord_; }

    /// @brief Downsample a point cloud using a polar grid filter.
    /// @details Points are grouped into voxels in polar coordinate space. The centroid of the points in each voxel is
    /// computed to produce the downsampled point cloud.
    /// @param points The input point container to be downsampled.
    /// @param result The output point container for the downsampled points.
    void downsampling(const PointContainerShared& points, PointContainerShared& result) {
        const size_t N = points.size();
        if (N == 0) {
            result.resize(0);
            return;
        }
        const auto voxel_map = this->compute_voxel_bit_and_voxel_map(points);
        this->voxel_map_to_cloud(voxel_map, result);
    }

    /// @brief Downsample a point cloud with attributes using a polar grid filter.
    /// @details Points and their attributes (RGB, intensity) are grouped into voxels in polar coordinate space.
    ///          The centroid of points and the average of attributes in each voxel are computed.
    /// @param cloud The input point cloud with attributes to be downsampled.
    /// @param result The output point cloud for the downsampled data.
    void downsampling(const PointCloudShared& cloud, PointCloudShared& result) {
        const size_t N = cloud.size();
        if (N == 0) {
            result.resize_points(0);
            return;
        }
        const auto voxel_map = this->compute_voxel_bit_and_voxel_map(*cloud.points);

        if (cloud.has_rgb() || cloud.has_intensity()) {
            const auto rgb_map = this->compute_voxel_map(*cloud.rgb);
            const auto intensity_map = this->compute_voxel_map(*cloud.intensities);
            this->voxel_map_to_cloud(voxel_map, rgb_map, intensity_map, result);
        } else {
            this->voxel_map_to_cloud(voxel_map, *result.points);
        }
    }

private:
    sycl_points::sycl_utils::DeviceQueue queue_;
    float distance_voxel_size_;
    float elevation_voxel_size_;
    float azimuth_voxel_size_;
    float distance_voxel_size_inv_;
    float elevation_voxel_size_inv_;
    float azimuth_voxel_size_inv_;
    size_t min_voxel_count_;
    CoordinateSystem coord_;

    shared_vector_ptr<uint64_t> bit_ptr_ = nullptr;

    /// @brief Computes the 64-bit polar key for each point in the input container on the SYCL device.
    /// @param points The input point container.
    void compute_voxel_bit(const PointContainerShared& points) {
        const size_t N = points.size();
        if (this->bit_ptr_->size() < N) {
            this->bit_ptr_->resize(N);
        }
        if (N == 0) {
            return;
        }

        // mem_advise set to device
        {
            this->queue_.set_accessed_by_device(this->bit_ptr_->data(), N);
            this->queue_.set_accessed_by_device(points.data(), N);
        }

        const size_t work_group_size = this->queue_.get_work_group_size();
        const size_t global_size = this->queue_.get_global_size(N);

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const auto point_ptr = points.data();
            const auto bit_ptr = this->bit_ptr_->data();
            const auto distance_inv = this->distance_voxel_size_inv_;
            const auto polar_inv = this->elevation_voxel_size_inv_;
            const auto azimuth_inv = this->azimuth_voxel_size_inv_;
            const auto coord = this->coord_;

            auto kernel_launch = [&](auto coord_tag) {
                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const uint32_t i = item.get_global_id(0);
                    if (i >= N) return;
                    bit_ptr[i] =
                        kernel::compute_polar_bit<coord_tag.value>(point_ptr[i], distance_inv, polar_inv, azimuth_inv);
                });
            };
            if (this->coord_ == CoordinateSystem::LIDAR) {
                kernel_launch(std::integral_constant<CoordinateSystem, CoordinateSystem::LIDAR>{});
            } else if (this->coord_ == CoordinateSystem::CAMERA) {
                kernel_launch(std::integral_constant<CoordinateSystem, CoordinateSystem::CAMERA>{});
            }
        });
        event.wait_and_throw();

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(this->bit_ptr_->data(), N);
            this->queue_.clear_accessed_by_device(points.data(), N);
        }
    }

    /// @brief Aggregates data on the host based on the pre-computed voxel keys.
    /// @tparam T The type of data to aggregate (e.g., PointType, RGBType).
    /// @param data The shared_vector containing the data to be aggregated.
    /// @return An unordered_map from voxel key to the summed data within that voxel.
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

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_host(this->bit_ptr_->data(), N);
            this->queue_.clear_accessed_by_host(data.data(), N);
        }

        return voxel_map;
    }

    /// @brief A helper function that first computes polar keys and then creates the voxel map for points.
    /// @param points The input point container.
    /// @return An unordered_map from voxel key to the summed points within that voxel.
    std::unordered_map<uint64_t, PointType> compute_voxel_bit_and_voxel_map(const PointContainerShared& points) {
        this->compute_voxel_bit(points);
        return this->compute_voxel_map(points);
    }

    /// @brief Converts a voxel map of points back into a point container.
    /// @param voxel_map The map from voxel key to summed PointType.
    /// @param result The output point container.
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

    /// @brief Converts voxel maps of points and attributes back into a PointCloudShared object.
    /// @param voxel_map The map from voxel key to summed PointType.
    /// @param voxel_map_rgb The map from voxel key to summed RGBType.
    /// @param voxel_map_intensity The map from voxel key to summed intensity.
    /// @param result The output PointCloudShared object.
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
