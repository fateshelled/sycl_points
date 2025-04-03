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

  // Optimize work group size
  const size_t work_group_size = sycl_utils::get_work_group_size(queue);
  const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;

  // compute bit on device
  shared_vector<uint64_t> bits(N, VoxelConstants::invalid_coord, shared_allocator<uint64_t>(queue));
  {
    // memory ptr
    const auto point_ptr = points.data();
    const auto bit_ptr = bits.data();

    auto event = queue
      .submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)), [=](sycl::nd_item<1> item) {
          const size_t i = item.get_global_id(0);
          if (i >= N) return;

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
  /* Kahan algorithm (high precision) */
  // {
  //   std::unordered_map<uint64_t, PointType> voxel_map;
  //   std::unordered_map<uint64_t, PointType> kahan_map;
  //   voxel_map.reserve(N / 2);
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
  //   for (auto& [_, point] : voxel_map) {
  //     point /= point.w();
  //     result.emplace_back(point);
  //   }
  // }
  {
    std::unordered_map<uint64_t, PointType> voxel_map;
    voxel_map.reserve(N / 2);

    for (size_t i = 0; i < N; ++i) {
      if (bits[i] == VoxelConstants::invalid_coord) continue;

      const auto it = voxel_map.find(bits[i]);
      if (it == voxel_map.end()) {
        voxel_map[bits[i]] = points[i];
      } else {
        it->second += points[i];
      }
    }
    for (auto& [_, point] : voxel_map) {
      point /= point.w();
      result.emplace_back(point);
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
