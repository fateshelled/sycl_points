#pragma once

#include <cstdint>

namespace sycl_points {
namespace algorithms {
namespace filter {

/// @brief Defines the coordinate system for polar coordinate conversion, following REP-103.
/// @see https://ros.org/reps/rep-0103.html
enum class CoordinateSystem : std::uint8_t { LIDAR = 0, CAMERA = 1 };
}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
