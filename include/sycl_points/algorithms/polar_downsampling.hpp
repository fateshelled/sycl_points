#pragma once

#include <sycl_points/algorithms/common/voxel_constants.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <stdexcept>

namespace sycl_points {
namespace algorithms {
namespace filter {

namespace kernel {

SYCL_EXTERNAL inline uint64_t compute_polar_bit(const PointType &point,
                                                const float distance_inv,
                                                const float polar_inv,
                                                const float azimuth_inv) {
  // Convert Cartesian coordinates to spherical (r, polar, azimuth)
  if (!sycl::isfinite(point.x()) || !sycl::isfinite(point.y()) ||
      !sycl::isfinite(point.z())) {
    return VoxelConstants::invalid_coord;
  }

  const float xy_dist_sq = point.x() * point.x() + point.y() * point.y();
  const float r = sycl::sqrt(xy_dist_sq + point.z() * point.z());
  const float polar = sycl::atan2(sycl::sqrt(xy_dist_sq), point.z());
  const float azimuth =
      sycl::atan2(point.y(), point.x()) + sycl::numbers::pi_v<float>;

  const auto coord0 = static_cast<int64_t>(sycl::floor(r * distance_inv)) +
                      VoxelConstants::coord_offset;
  const auto coord1 = static_cast<int64_t>(sycl::floor(polar * polar_inv)) +
                      VoxelConstants::coord_offset;
  const auto coord2 = static_cast<int64_t>(sycl::floor(azimuth * azimuth_inv)) +
                      VoxelConstants::coord_offset;

  if (coord0 < 0 || VoxelConstants::coord_bit_mask < coord0 || coord1 < 0 ||
      VoxelConstants::coord_bit_mask < coord1 || coord2 < 0 ||
      VoxelConstants::coord_bit_mask < coord2) {
    return VoxelConstants::invalid_coord;
  }

  // Encode range, polar angle, and azimuth angle into a 64-bit key
  return (static_cast<uint64_t>(coord0 & VoxelConstants::coord_bit_mask)
          << (VoxelConstants::coord_bit_size * 0)) |
         (static_cast<uint64_t>(coord1 & VoxelConstants::coord_bit_mask)
          << (VoxelConstants::coord_bit_size * 1)) |
         (static_cast<uint64_t>(coord2 & VoxelConstants::coord_bit_mask)
          << (VoxelConstants::coord_bit_size * 2));
}

} // namespace kernel

/// @brief Polar grid downsampling with SYCL implementation
class PolarGrid {
public:
  using Ptr = std::shared_ptr<PolarGrid>;

  /// @brief Constructor
  /// @param queue SYCL queue
  /// @param distance_leaf_size leaf size for radial distance
  /// @param polar_leaf_size leaf size for polar angle (radian)
  /// @param azimuth_leaf_size leaf size for azimuth angle (radian)
  PolarGrid(const sycl_points::sycl_utils::DeviceQueue &queue,
            const float distance_leaf_size, const float polar_leaf_size,
            const float azimuth_leaf_size)
      : queue_(queue), distance_leaf_size_(distance_leaf_size),
        polar_leaf_size_(polar_leaf_size),
        azimuth_leaf_size_(azimuth_leaf_size) {
    if (distance_leaf_size <= 0.0f || polar_leaf_size <= 0.0f ||
        azimuth_leaf_size <= 0.0f) {
      throw std::invalid_argument("leaf sizes must be positive");
    }
    this->bit_ptr_ = std::make_shared<shared_vector<uint64_t>>(
        0, shared_allocator<uint64_t>(*this->queue_.ptr));
    this->distance_leaf_size_inv_ = 1.0f / this->distance_leaf_size_;
    this->polar_leaf_size_inv_ = 1.0f / this->polar_leaf_size_;
    this->azimuth_leaf_size_inv_ = 1.0f / this->azimuth_leaf_size_;
    this->min_voxel_count_ = 1;
  }

  void set_distance_leaf_size(const float size) {
    if (size <= 0.0f) {
      throw std::invalid_argument("distance_leaf_size must be positive");
    }
    this->distance_leaf_size_ = size;
    this->distance_leaf_size_inv_ = 1.0f / size;
  }
  float get_distance_leaf_size() const { return this->distance_leaf_size_; }

  void set_polar_leaf_size(const float size) {
    if (size <= 0.0f) {
      throw std::invalid_argument("polar_leaf_size must be positive");
    }
    this->polar_leaf_size_ = size;
    this->polar_leaf_size_inv_ = 1.0f / size;
  }
  float get_polar_leaf_size() const { return this->polar_leaf_size_; }

  void set_azimuth_leaf_size(const float size) {
    if (size <= 0.0f) {
      throw std::invalid_argument("azimuth_leaf_size must be positive");
    }
    this->azimuth_leaf_size_ = size;
    this->azimuth_leaf_size_inv_ = 1.0f / size;
  }
  float get_azimuth_leaf_size() const { return this->azimuth_leaf_size_; }

  void set_min_voxel_count(const size_t min_voxel_count) {
    this->min_voxel_count_ = min_voxel_count;
  }

  /// @brief Downsampling based on polar grid
  /// @param points Point Cloud
  /// @param result Downsampled point cloud
  void downsampling(const PointContainerShared &points,
                    PointContainerShared &result) {
    const size_t N = points.size();
    if (N == 0) {
      result.resize(0);
      return;
    }
    const auto voxel_map = this->compute_voxel_bit_and_voxel_map(points);
    this->voxel_map_to_cloud(voxel_map, result);
  }

  /// @brief Downsampling based on polar grid
  /// @param cloud Point Cloud with attributes
  /// @param result Downsampled point cloud with attributes
  void downsampling(const PointCloudShared &cloud, PointCloudShared &result) {
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
  float distance_leaf_size_;
  float polar_leaf_size_;
  float azimuth_leaf_size_;
  float distance_leaf_size_inv_;
  float polar_leaf_size_inv_;
  float azimuth_leaf_size_inv_;
  size_t min_voxel_count_;

  shared_vector_ptr<uint64_t> bit_ptr_ = nullptr;

  void compute_voxel_bit(const PointContainerShared &points) {
    const size_t N = points.size();

    this->queue_.set_accessed_by_device(this->bit_ptr_->data(), N);
    this->queue_.set_accessed_by_device(points.data(), N);

    const size_t work_group_size = this->queue_.get_work_group_size();
    const size_t global_size = this->queue_.get_global_size(N);
    auto event = this->queue_.ptr->submit([&](sycl::handler &h) {
      const auto point_ptr = points.data();
      const auto bit_ptr = this->bit_ptr_->data();
      const auto distance_inv = this->distance_leaf_size_inv_;
      const auto polar_inv = this->polar_leaf_size_inv_;
      const auto azimuth_inv = this->azimuth_leaf_size_inv_;
      h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                     [=](sycl::nd_item<1> item) {
                       const uint32_t i = item.get_global_id(0);
                       if (i >= N)
                         return;
                       bit_ptr[i] = kernel::compute_polar_bit(
                           point_ptr[i], distance_inv, polar_inv, azimuth_inv);
                     });
    });
    event.wait();

    this->queue_.clear_accessed_by_device(this->bit_ptr_->data(), N);
    this->queue_.clear_accessed_by_device(points.data(), N);
  }

  template <typename T, size_t AllocSize = 0>
  std::unordered_map<uint64_t, T>
  compute_voxel_map(const shared_vector<T, AllocSize> &data) const {
    const size_t N = data.size();
    if (N == 0) {
      return {};
    }

    this->queue_.set_accessed_by_host(this->bit_ptr_->data(), N);
    this->queue_.set_accessed_by_host(data.data(), N);

    std::unordered_map<uint64_t, T> voxel_map;
    for (size_t i = 0; i < N; ++i) {
      const auto voxel_bit = (*this->bit_ptr_)[i];
      if (voxel_bit == VoxelConstants::invalid_coord)
        continue;
      const auto it = voxel_map.find(voxel_bit);
      if (it == voxel_map.end()) {
        voxel_map[voxel_bit] = data[i];
      } else {
        it->second += data[i];
      }
    }

    this->queue_.clear_accessed_by_host(this->bit_ptr_->data(), N);
    this->queue_.clear_accessed_by_host(data.data(), N);

    return voxel_map;
  }

  std::unordered_map<uint64_t, PointType>
  compute_voxel_bit_and_voxel_map(const PointContainerShared &points) {
    const size_t N = points.size();
    if (this->bit_ptr_->size() < N) {
      this->bit_ptr_->resize(N);
    }

    this->compute_voxel_bit(points);
    return this->compute_voxel_map(points);
  }

  void
  voxel_map_to_cloud(const std::unordered_map<uint64_t, PointType> &voxel_map,
                     PointContainerShared &result) const {
    const size_t N = voxel_map.size();
    result.clear();
    result.resize(N);
    this->queue_.set_accessed_by_host(result.data(), N);
    const float min_voxel_count = static_cast<float>(this->min_voxel_count_);
    size_t idx = 0;
    for (const auto &[_, point] : voxel_map) {
      if (point.w() >= min_voxel_count) {
        result[idx++] = point / point.w();
      }
    }
    result.resize(idx);
    this->queue_.clear_accessed_by_host(result.data(), N);
  }

  void voxel_map_to_cloud(
      const std::unordered_map<uint64_t, PointType> &voxel_map,
      const std::unordered_map<uint64_t, RGBType> &voxel_map_rgb,
      const std::unordered_map<uint64_t, float> &voxel_map_intensity,
      PointCloudShared &result) const {
    const size_t N = voxel_map.size();
    const bool has_rgb = voxel_map_rgb.size() == N;
    const bool has_intensity = voxel_map_intensity.size() == N;
    result.clear();
    result.resize_points(N);
    this->queue_.set_accessed_by_host(result.points_ptr(), N);
    if (has_rgb) {
      result.resize_rgb(voxel_map_rgb.size());
      this->queue_.set_accessed_by_host(result.rgb_ptr(), voxel_map_rgb.size());
    }
    if (has_intensity) {
      result.resize_intensities(voxel_map_intensity.size());
      this->queue_.set_accessed_by_host(result.intensities_ptr(),
                                        voxel_map_intensity.size());
    }

    const float min_voxel_count = static_cast<float>(this->min_voxel_count_);
    size_t idx = 0;
    for (const auto &[voxel_idx, point] : voxel_map) {
      if (point.w() >= min_voxel_count) {
        (*result.points)[idx] = point / point.w();
        if (has_rgb)
          (*result.rgb)[idx] = voxel_map_rgb.at(voxel_idx) / point.w();
        if (has_intensity)
          (*result.intensities)[idx] =
              voxel_map_intensity.at(voxel_idx) / point.w();
        ++idx;
      }
    }
    result.resize_points(idx);
    if (has_rgb) {
      result.resize_rgb(idx);
    }
    if (has_intensity) {
      result.resize_intensities(idx);
    }

    this->queue_.clear_accessed_by_host(result.points_ptr(), N);
    if (has_rgb) {
      this->queue_.clear_accessed_by_host(result.rgb_ptr(),
                                          voxel_map_rgb.size());
    }
    if (has_intensity) {
      this->queue_.clear_accessed_by_host(result.intensities_ptr(),
                                          voxel_map_intensity.size());
    }
  }
};

} // namespace filter
} // namespace algorithms
} // namespace sycl_points
