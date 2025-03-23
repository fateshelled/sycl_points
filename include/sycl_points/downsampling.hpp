#pragma once

#include "point_cloud.hpp"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace {

struct VoxelConstants {
  static constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
  static constexpr int coord_bit_size = 21;                       // Bits to represent each voxel coordinate (pack 21x3=63bits in 64bit int)
  static constexpr size_t coord_bit_mask = (1 << 21) - 1;         // Bit mask
  static constexpr int coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive
};

}  // namespace

namespace sycl_points {

inline PointContainerShared voxel_downsampling_sycl(sycl::queue& queue, const PointContainerShared& points, const float voxel_size) {
  // Ref: https://github.com/koide3/gtsam_points/blob/master/src/gtsam_points/types/point_cloud_cpu_funcs.cpp
  // function: voxelgrid_sampling
  // MIT License

  const size_t N = points.size();
  const shared_allocator<PointType> point_alloc(queue);
  if (N == 0) return PointContainerShared(0, point_alloc);

  const float inv_voxel_size = 1.0f / voxel_size;

  // compute bit on device
  shared_vector<uint64_t> bits(N, VoxelConstants::invalid_coord, shared_allocator<uint64_t>(queue));
  {
    // memory ptr
    const auto point_ptr = points.data();
    const auto bit_ptr = bits.data();

    auto event = queue
      .submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
          if (!sycl::isfinite(point_ptr[i].x()) || !sycl::isfinite(point_ptr[i].y()) || !sycl::isfinite(point_ptr[i].z())) {
            return;
          }
          const auto coord0 = static_cast<int64_t>(sycl::floor(point_ptr[i].x() * inv_voxel_size)) + VoxelConstants::coord_offset;
          const auto coord1 = static_cast<int64_t>(sycl::floor(point_ptr[i].y() * inv_voxel_size)) + VoxelConstants::coord_offset;
          const auto coord2 = static_cast<int64_t>(sycl::floor(point_ptr[i].z() * inv_voxel_size)) + VoxelConstants::coord_offset;
          if (coord0 < 0 || VoxelConstants::coord_bit_mask < coord0 ||
              coord1 < 0 || VoxelConstants::coord_bit_mask < coord1 ||
              coord2 < 0 || VoxelConstants::coord_bit_mask < coord2) {
            return;
          }
          // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
          bit_ptr[i] = (static_cast<uint64_t>(coord0 & VoxelConstants::coord_bit_mask) << (VoxelConstants::coord_bit_size * 0)) |
                       (static_cast<uint64_t>(coord1 & VoxelConstants::coord_bit_mask) << (VoxelConstants::coord_bit_size * 1)) |
                       (static_cast<uint64_t>(coord2 & VoxelConstants::coord_bit_mask) << (VoxelConstants::coord_bit_size * 2));
        });
      });
      event.wait();
  }

  PointContainerShared result(point_alloc);
  result.reserve(N);
  {
    // prepare indices
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), (size_t)0);

    // sort indices by bits
    std::sort(indices.begin(), indices.end(), [&bits](size_t a, size_t b) { return bits[a] < bits[b]; });

    // compute average coords with same bit
    size_t counter = 1;
    auto& current_bit = bits[indices[0]];
    if (current_bit != VoxelConstants::invalid_coord) {
      result.emplace_back(points[indices[0]]);
      for (size_t i = 1; i < N; ++i) {
        if (bits[indices[i]] == VoxelConstants::invalid_coord) break;

        if (current_bit == bits[indices[i]]) {
          result.back() += points[indices[i]];
          ++counter;
          if (i == N - 1) {
            result.back() /= counter;
          }
        } else {
          result.back() /= counter;
          counter = 1;
          current_bit = bits[indices[i]];
          result.emplace_back(points[indices[i]]);
        }
      }
    }
  }
  result.shrink_to_fit();
  return result;
}

inline PointCloudShared voxel_downsampling_sycl(sycl::queue& queue, const PointCloudShared& points, const float voxel_size) {
  PointCloudShared ret(queue);
  *ret.points = voxel_downsampling_sycl(queue, *points.points, voxel_size);
  return ret;
}

}  // namespace sycl_points
