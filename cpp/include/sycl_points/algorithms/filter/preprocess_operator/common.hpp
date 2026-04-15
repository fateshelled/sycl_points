#pragma once

#include <cmath>

#include "sycl_points/algorithms/common/filter_by_flags.hpp"
#include "sycl_points/points/types.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {
namespace preprocess_operator {
namespace kernel {

SYCL_EXTERNAL inline bool is_finite(const PointType& pt) {
    return std::isfinite(pt[0]) && std::isfinite(pt[1]) && std::isfinite(pt[2]) && std::isfinite(pt[3]);
}

SYCL_EXTERNAL inline void box_filter(const PointType& pt, uint8_t& flag, float min_distance, float max_distance) {
    const float linf_dist = sycl::max(sycl::fabs(pt.x()), sycl::max(sycl::fabs(pt.y()), sycl::fabs(pt.z())));

    if (linf_dist < min_distance || linf_dist > max_distance) {
        flag = REMOVE_FLAG;
    }
}

}  // namespace kernel
}  // namespace preprocess_operator
}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
