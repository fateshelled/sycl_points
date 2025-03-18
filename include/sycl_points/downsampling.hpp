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

template <typename T = float>
inline PointContainerCPU<T> voxel_downsampling_sycl(sycl::queue& queue, const PointContainerCPU<T>& points, const T voxel_size) {
  // Ref: https://github.com/koide3/gtsam_points/blob/master/src/gtsam_points/types/point_cloud_cpu_funcs.cpp
  // function: voxelgrid_sampling
  // MIT License

  const size_t N = points.size();
  if (N == 0) return PointContainerCPU<T>{};

  const T inv_voxel_size = (T)1.0 / voxel_size;

  // compute bit on device
  std::vector<uint64_t> bits(N);
  {
    // allocate device memory
    auto* dev_points = sycl::malloc_device<PointType<T>>(N, queue);
    auto* dev_bits = sycl::malloc_device<uint64_t>(N, queue);
    queue.memcpy(dev_points, points[0].data(), N * sizeof(PointType<T>));
    queue.fill(dev_bits, VoxelConstants::invalid_coord, N);
    queue.wait();

    queue
      .submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
          if (!sycl::isfinite(dev_points[i].x()) || !sycl::isfinite(dev_points[i].y()) || !sycl::isfinite(dev_points[i].z())) {
            return;
          }
          const auto coord0 = static_cast<int64_t>(sycl::floor(dev_points[i].x() * inv_voxel_size)) + VoxelConstants::coord_offset;
          const auto coord1 = static_cast<int64_t>(sycl::floor(dev_points[i].y() * inv_voxel_size)) + VoxelConstants::coord_offset;
          const auto coord2 = static_cast<int64_t>(sycl::floor(dev_points[i].z() * inv_voxel_size)) + VoxelConstants::coord_offset;
          if (coord0 < 0 || VoxelConstants::coord_bit_mask < coord0 ||
              coord1 < 0 || VoxelConstants::coord_bit_mask < coord1 ||
              coord2 < 0 || VoxelConstants::coord_bit_mask < coord2) {
            return;
          }
          // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
          dev_bits[i] = (static_cast<uint64_t>(coord0 & VoxelConstants::coord_bit_mask) << (VoxelConstants::coord_bit_size * 0)) |
                        (static_cast<uint64_t>(coord1 & VoxelConstants::coord_bit_mask) << (VoxelConstants::coord_bit_size * 1)) |
                        (static_cast<uint64_t>(coord2 & VoxelConstants::coord_bit_mask) << (VoxelConstants::coord_bit_size * 2));
        });
      })
      .wait_and_throw();

    // copy results from device to host
    queue.memcpy(bits.data(), dev_bits, N * sizeof(int64_t)).wait();

    // free device memory
    sycl::free(dev_bits, queue);
    sycl::free(dev_points, queue);
  }

  PointContainerCPU<T> result;
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

template <typename T = float>
inline PointCloudCPU<T> voxel_downsampling_sycl(sycl::queue& queue, const PointCloudCPU<T>& points, const T voxel_size) {
  PointCloudCPU<T> ret;
  ret.points = voxel_downsampling_sycl<T>(queue, points.points, voxel_size);
  return ret;
}

}  // namespace sycl_points
