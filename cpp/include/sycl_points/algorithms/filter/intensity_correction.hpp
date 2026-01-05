#pragma once

#include <stdexcept>

#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace intensity_correction {

/// @brief Correct intensity based on distance and scaling
///        (Intensity' = clamp(scale * Intensity * dist^exponent, min_intensity, max_intensity)
/// @param cloud Point Cloud (will be updated in place)
/// @param exponent Exponent for distance correction
/// @param scale Scale factor
/// @param min_intensity minimum intensity
/// @param max_intensity maximum intensity
inline void correct_intensity(PointCloudShared& cloud, float exponent = 2.0f, float scale = 1.0f,
                              float min_intensity = 0.0f, float max_intensity = 1000.0f) {
    const size_t N = cloud.size();
    if (N == 0) {
        return;
    }

    if (exponent < 0.0f) {
        throw std::runtime_error("[correct_intensity] exponent must be non-negative");
    }
    if (!cloud.has_intensity()) {
        throw std::runtime_error("[correct_intensity] Intensity field not found");
    }

    auto event = cloud.queue.ptr->submit([&](sycl::handler& h) {
        const size_t work_group_size = cloud.queue.get_work_group_size();
        const size_t global_size = cloud.queue.get_global_size(N);

        const auto point_ptr = cloud.points_ptr();
        const auto intensity_ptr = cloud.intensities_ptr();
        const float s = scale;
        const float min = min_intensity;
        const float max = max_intensity;

        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) {
                return;
            }

            const auto point = point_ptr[i];
            const float dist_sq = point.x() * point.x() + point.y() * point.y() + point.z() * point.z();
            const float corrected_intensity = intensity_ptr[i] * sycl::pow(dist_sq, exponent * 0.5f);
            intensity_ptr[i] = sycl::clamp(corrected_intensity * s, min, max);
        });
    });

    event.wait_and_throw();
}

}  // namespace intensity_correction

}  // namespace algorithms

}  // namespace sycl_points
