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

/// @brief Expand 21-bit value to 63-bit with 3-bit spacing for Morton code encoding.
/// Each bit is separated by 2 zero bits for interleaving with other coordinates.
/// @param v Input value (only lower 21 bits are used)
/// @return 63-bit expanded value with bits at positions 0, 3, 6, 9, ..., 60
SYCL_EXTERNAL inline uint64_t expand_bits_21(uint64_t v) {
    static constexpr uint64_t MASK_21_BITS = 0x1FFFFF;
    static constexpr uint64_t MASK_STEP_1 = 0x1F00000000FFFFULL;
    static constexpr uint64_t MASK_STEP_2 = 0x1F0000FF0000FFULL;
    static constexpr uint64_t MASK_STEP_3 = 0x100F00F00F00F00FULL;
    static constexpr uint64_t MASK_STEP_4 = 0x10C30C30C30C30C3ULL;
    static constexpr uint64_t MASK_STEP_5 = 0x1249249249249249ULL;

    v &= MASK_21_BITS;                           // Mask to 21 bits
    v = (v | (v << 32)) & MASK_STEP_1; // Split: upper 5 bits and lower 16 bits
    v = (v | (v << 16)) & MASK_STEP_2; // Split into 5-8-8 bit groups
    v = (v | (v << 8))  & MASK_STEP_3; // Split into smaller groups
    v = (v | (v << 4))  & MASK_STEP_4; // 2-bit groups
    v = (v | (v << 2))  & MASK_STEP_5; // Final: 1-bit with 2-bit gaps
    return v;
}

/// @brief Encode 3D coordinates into 63-bit Morton code (Z-order curve).
/// Morton code interleaves bits from x, y, z coordinates to preserve spatial locality.
/// Spatially adjacent voxels will have numerically close Morton codes.
/// @param x X coordinate (21 bits)
/// @param y Y coordinate (21 bits)
/// @param z Z coordinate (21 bits)
/// @return 63-bit Morton code: ...z2y2x2 z1y1x1 z0y0x0
SYCL_EXTERNAL inline uint64_t morton_encode_3d(uint64_t x, uint64_t y, uint64_t z) {
    return expand_bits_21(x) | (expand_bits_21(y) << 1) | (expand_bits_21(z) << 2);
}

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

    // Compute Morton code (Z-order curve) for spatial locality
    // Interleaves bits: ...z2y2x2 z1y1x1 z0y0x0
    return morton_encode_3d(
        static_cast<uint64_t>(coord0 & VoxelConstants::coord_bit_mask),
        static_cast<uint64_t>(coord1 & VoxelConstants::coord_bit_mask),
        static_cast<uint64_t>(coord2 & VoxelConstants::coord_bit_mask));
}

}  // namespace kernel
}  // namespace filter

}  // namespace algorithms
}  // namespace sycl_points
