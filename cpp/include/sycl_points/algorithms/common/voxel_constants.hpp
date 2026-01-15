#pragma once

#include <cstdint>
#include <limits>

#include "sycl_points/points/types.hpp"

namespace sycl_points {
namespace algorithms {

struct VoxelConstants {
    static constexpr uint64_t invalid_coord = std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t deleted_coord = std::numeric_limits<uint64_t>::max() - 1;
    static constexpr uint8_t coord_bit_size = 21;                       // Bits to represent each voxel coordinate
    static constexpr int64_t coord_bit_mask = (1 << 21) - 1;            // Bit mask
    static constexpr int64_t coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive
};

/// @brief Constants for bit-shifting components of the cartesian coordinate key.
struct CartesianCoordComponent {
    static constexpr uint8_t X = 0;
    static constexpr uint8_t Y = 1;
    static constexpr uint8_t Z = 2;
};

/// @brief Constants for bit-shifting components of the polar coordinate key.
struct PolarCoordComponent {
    static constexpr uint8_t DISTANCE = 0;
    static constexpr uint8_t POLAR = 1;
    static constexpr uint8_t AZIMUTH = 2;
};

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
}  // namespace filter

/// @brief Hash a 64-bit voxel key to a 32-bit hash value using XOR folding.
/// @param key The 64-bit voxel key.
/// @return The 32-bit hash value.
SYCL_EXTERNAL inline uint32_t hash_voxel_key_to_32bit(const uint64_t key) {
    return static_cast<uint32_t>((key >> 32) ^ (key & 0xFFFFFFFF));
}

}  // namespace algorithms
}  // namespace sycl_points
