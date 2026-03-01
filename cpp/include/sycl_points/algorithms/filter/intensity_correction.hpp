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

/// @brief Correct intensity based on distance and angle of incidence using surface normals
///        (I' = clamp(scale * I * dist^exponent / max(cos(theta), min_cos_theta), min_intensity, max_intensity))
///        cos(theta) = |n . p| / dist  where n is the surface normal and p is the point position
///        (beam direction = -p/dist, so cos(theta) = |n . (-p/dist)| = |n . p| / dist)
/// @param cloud Point Cloud with intensity and normals (will be updated in place)
/// @param exponent Exponent for distance correction
/// @param scale Scale factor
/// @param min_intensity Minimum intensity after correction
/// @param max_intensity Maximum intensity after correction
/// @param min_cos_theta Minimum cos(theta) to prevent division by zero at grazing angles (~10 deg = 0.17)
inline void correct_intensity_with_normal(PointCloudShared& cloud, float exponent = 2.0f, float scale = 1.0f,
                                          float min_intensity = 0.0f, float max_intensity = 1000.0f,
                                          float min_cos_theta = 0.17f) {
    const size_t N = cloud.size();
    if (N == 0) {
        return;
    }

    if (exponent < 0.0f) {
        throw std::runtime_error("[correct_intensity_with_normal] exponent must be non-negative");
    }
    if (min_cos_theta <= 0.0f) {
        throw std::runtime_error("[correct_intensity_with_normal] min_cos_theta must be positive");
    }
    if (!cloud.has_intensity()) {
        throw std::runtime_error("[correct_intensity_with_normal] Intensity field not found");
    }
    if (!cloud.has_normal()) {
        throw std::runtime_error("[correct_intensity_with_normal] Normal field not found");
    }

    auto event = cloud.queue.ptr->submit([&](sycl::handler& h) {
        const size_t work_group_size = cloud.queue.get_work_group_size();
        const size_t global_size = cloud.queue.get_global_size(N);

        const auto point_ptr = cloud.points_ptr();
        const auto intensity_ptr = cloud.intensities_ptr();
        const auto normal_ptr = cloud.normals_ptr();
        const float s = scale;
        const float min = min_intensity;
        const float max = max_intensity;
        const float min_cos = min_cos_theta;

        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) {
                return;
            }

            const auto point = point_ptr[i];
            const float px = point.x(), py = point.y(), pz = point.z();
            const float dist_sq = px * px + py * py + pz * pz;

            // Guard against points at sensor origin
            if (dist_sq < 1e-12f) {
                intensity_ptr[i] = min;
                return;
            }

            const float dist = sycl::sqrt(dist_sq);

            // cos(theta) = |n . beam_dir| = |n . p| / dist
            const auto normal = normal_ptr[i];
            const float n_dot_p = sycl::fabs(normal.x() * px + normal.y() * py + normal.z() * pz);
            const float cos_theta = sycl::max(n_dot_p / dist, min_cos);

            const float corrected_intensity = intensity_ptr[i] * sycl::pow(dist_sq, exponent * 0.5f) / cos_theta;
            intensity_ptr[i] = sycl::clamp(corrected_intensity * s, min, max);
        });
    });

    event.wait_and_throw();
}

}  // namespace intensity_correction

}  // namespace algorithms

}  // namespace sycl_points
