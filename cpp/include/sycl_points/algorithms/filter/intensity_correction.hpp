#pragma once

#include <stdexcept>

#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace intensity_correction {

namespace kernel {

/// @brief Compute incidence angle correction factor from point position and surface normal.
///        Returns pow(max(|cos(theta)|, 1e-3), -angle_exponent).
SYCL_EXTERNAL inline float compute_angle_factor(const PointType& point, const Normal& normal, float angle_exponent) {
    const float dot = eigen_utils::dot<3>(point.head<3>(), normal.head<3>());
    const float denom =
        eigen_utils::frobenius_norm<3>(point.head<3>()) * eigen_utils::frobenius_norm<3>(normal.head<3>());
    if (denom <= 1e-6f) return 1.0f;
    const float abs_cos = sycl::fabs(dot / denom);
    return sycl::pow(sycl::fmax(abs_cos, 1e-3f), -angle_exponent);
}

/// @brief Apply distance and angle correction to a single intensity value.
///        Returns clamp(intensity * (dist/ref_distance)^exponent * angle_factor * scale, min, max).
SYCL_EXTERNAL inline float correct_intensity(const PointType& point, float intensity, float exponent, float scale,
                                             float min_intensity, float max_intensity, float ref_distance,
                                             float angle_factor) {
    const float dist_sq = point.x() * point.x() + point.y() * point.y() + point.z() * point.z();
    const float dist = sycl::sqrt(dist_sq);
    const float dist_factor = sycl::pow(dist / ref_distance, exponent);
    return sycl::clamp(intensity * dist_factor * angle_factor * scale, min_intensity, max_intensity);
}

}  // namespace kernel

/// @brief Correct intensity based on distance, reference distance, and optionally incidence angle.
///        I' = clamp(scale * I * (dist/ref_distance)^exponent * (1/cos(theta))^angle_exponent, min, max)
///        When angle_exponent == 0 or no normals/covariances are available, angle correction is skipped.
/// @param cloud Point Cloud (will be updated in place)
/// @param exponent Exponent for distance correction
/// @param scale Scale factor
/// @param min_intensity minimum intensity
/// @param max_intensity maximum intensity
/// @param ref_distance Reference distance for normalization (must be > 0)
/// @param angle_exponent Exponent for incidence angle correction (0 = disabled)
inline void correct_intensity(PointCloudShared& cloud, float exponent = 2.0f, float scale = 1.0f,
                              float min_intensity = 0.0f, float max_intensity = 1000.0f, float ref_distance = 1.0f,
                              float angle_exponent = 0.0f) {
    const size_t N = cloud.size();
    if (N == 0) {
        return;
    }

    if (exponent < 0.0f) {
        throw std::runtime_error("[correct_intensity] exponent must be non-negative");
    }
    if (ref_distance <= 0.0f) {
        throw std::runtime_error("[correct_intensity] ref_distance must be positive");
    }
    if (!cloud.has_intensity()) {
        throw std::runtime_error("[correct_intensity] Intensity field not found");
    }

    auto submit_kernel = [&](auto compute_ang_factor) {
        return cloud.queue.ptr->submit([&](sycl::handler& h) {
            const size_t work_group_size = cloud.queue.get_work_group_size();
            const size_t global_size = cloud.queue.get_global_size(N);

            const auto point_ptr = cloud.points_ptr();
            const auto intensity_ptr = cloud.intensities_ptr();
            const float s = scale;
            const float min = min_intensity;
            const float max = max_intensity;
            const float ref_dist = ref_distance;

            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) {
                    return;
                }
                const float ang_factor = compute_ang_factor(i, point_ptr[i]);
                intensity_ptr[i] = kernel::correct_intensity(point_ptr[i], intensity_ptr[i], exponent, s, min, max,
                                                             ref_dist, ang_factor);
            });
        });
    };

    // mem_advise set to device
    {
        cloud.queue.set_accessed_by_device(cloud.points_ptr(), N);
        cloud.queue.set_accessed_by_device(cloud.intensities_ptr(), N);
    }

    const bool use_angle = (angle_exponent != 0.0f) && (cloud.has_normal() || cloud.has_cov());

    if (use_angle && cloud.has_normal()) {
        cloud.queue.set_accessed_by_device(cloud.normals_ptr(), N);

        const auto normal_ptr = cloud.normals_ptr();
        const float ang_exp = angle_exponent;

        submit_kernel([=](size_t i, const PointType& pt) {
            return kernel::compute_angle_factor(pt, normal_ptr[i], ang_exp);
        }).wait_and_throw();

        cloud.queue.clear_accessed_by_device(cloud.normals_ptr(), N);
    } else if (use_angle && cloud.has_cov()) {
        cloud.queue.set_accessed_by_device(cloud.covs_ptr(), N);

        const auto cov_ptr = cloud.covs_ptr();
        const float ang_exp = angle_exponent;

        submit_kernel([=](size_t i, const PointType& pt) {
            Normal n;
            algorithms::covariance::kernel::compute_normal_from_covariance(pt, cov_ptr[i], n);
            return kernel::compute_angle_factor(pt, n, ang_exp);
        }).wait_and_throw();

        cloud.queue.clear_accessed_by_device(cloud.covs_ptr(), N);
    } else {
        submit_kernel([=](size_t /*i*/, const PointType& /*pt*/) { return 1.0f; }).wait_and_throw();
    }

    // mem_advise clear
    {
        cloud.queue.clear_accessed_by_device(cloud.points_ptr(), N);
        cloud.queue.clear_accessed_by_device(cloud.intensities_ptr(), N);
    }
}

}  // namespace intensity_correction

}  // namespace algorithms

}  // namespace sycl_points
