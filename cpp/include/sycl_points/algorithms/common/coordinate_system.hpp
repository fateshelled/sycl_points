#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sycl_points {
namespace algorithms {

/// @brief Defines the coordinate system for polar coordinate conversion, following REP-103.
/// @see https://ros.org/reps/rep-0103.html
enum class CoordinateSystem : std::uint8_t { LIDAR = 0, CAMERA = 1 };

inline CoordinateSystem coordinate_system_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(str.begin(), str.end(), upper.begin(), [](u_char c) { return std::toupper(c); });
    if (upper == "LIDAR") {
        return CoordinateSystem::LIDAR;
    } else if (upper == "CAMERA") {
        return CoordinateSystem::CAMERA;
    } else {
        throw std::invalid_argument("Invalid coordinate system: " + str);
    }
}

}  // namespace algorithms
}  // namespace sycl_points
