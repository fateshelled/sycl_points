#pragma once

#include <cstdint>

namespace sycl_points {
namespace algorithms {

/// @brief Defines the coordinate system for polar coordinate conversion, following REP-103.
/// @see https://ros.org/reps/rep-0103.html
enum class CoordinateSystem : std::uint8_t { LIDAR = 0, CAMERA = 1 };

inline CoordinateSystem coordinate_system_from_string(const std::string &str) {
    std::string str_lower = str;
    std::transform(str_lower.begin(), str_lower.end(), str_lower.begin(), ::tolower);
    if (str_lower == "lidar") {
        return CoordinateSystem::LIDAR;
    } else if (str_lower == "camera") {
        return CoordinateSystem::CAMERA;
    } else {
        throw std::invalid_argument("Invalid coordinate system: " + str);
    }
}

}  // namespace algorithms
}  // namespace sycl_points
