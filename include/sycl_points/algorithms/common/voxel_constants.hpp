#pragma once

#include <cstdint>
#include <limits>

namespace sycl_points {
namespace algorithms {

struct VoxelConstants {
    static constexpr uint64_t invalid_coord = std::numeric_limits<uint64_t>::max();
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

}  // namespace algorithms
}  // namespace sycl_points
