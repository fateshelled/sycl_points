#pragma once

#include <stdexcept>

#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace intensity_correction {

/// @brief Correct intensity based on normalized distance and scaling
///        (Intensity' = clamp(scale * Intensity * (dist/reference_distance)^exponent, min_intensity, max_intensity))
/// @param cloud Point Cloud (will be updated in place)
/// @param exponent Exponent for distance correction
/// @param scale Scale factor
/// @param min_intensity minimum intensity
/// @param max_intensity maximum intensity
/// @param reference_distance Reference distance for normalization
/// @param sensor_pose Sensor pose represented in the same coordinate frame as the input cloud.
///        The correction uses point vectors relative to the sensor origin (translation part of the pose).
inline void correct_intensity(PointCloudShared& cloud, float exponent = 2.0f, float scale = 1.0f,
                              float min_intensity = 0.0f, float max_intensity = 1000.0f,
                              float reference_distance = 1.0f,
                              const Eigen::Isometry3f& sensor_pose = Eigen::Isometry3f::Identity()) {
    const size_t N = cloud.size();
    if (N == 0) {
        return;
    }

    if (exponent < 0.0f) {
        throw std::runtime_error("[correct_intensity] exponent must be non-negative");
    }
    if (reference_distance <= 0.0f) {
        throw std::runtime_error("[correct_intensity] reference_distance must be positive");
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
        const float inv_ref_dist_sq = 1.0f / (reference_distance * reference_distance);
        const float sensor_tx = sensor_pose.translation().x();
        const float sensor_ty = sensor_pose.translation().y();
        const float sensor_tz = sensor_pose.translation().z();

        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) {
                return;
            }

            const auto point = point_ptr[i];
            // Use point coordinates relative to the sensor origin so transformed clouds can be corrected as well.
            const float rel_x = point.x() - sensor_tx;
            const float rel_y = point.y() - sensor_ty;
            const float rel_z = point.z() - sensor_tz;
            const float dist_sq = rel_x * rel_x + rel_y * rel_y + rel_z * rel_z;
            const float normalized_dist_sq = dist_sq * inv_ref_dist_sq;
            const float corrected_intensity = intensity_ptr[i] * sycl::pow(normalized_dist_sq, exponent * 0.5f);
            intensity_ptr[i] = sycl::clamp(corrected_intensity * s, min, max);
        });
    });

    event.wait_and_throw();
}

/// @brief Correct intensity based on normalized distance and angle of incidence using surface normals
///        (I' = clamp(scale * I * (dist/reference_distance)^exponent / max(cos(theta), min_cos_theta),
///         min_intensity, max_intensity))
///        cos(theta) = |n . p| / dist  where n is the surface normal and p is the point position
///        (beam direction = -p/dist, so cos(theta) = |n . (-p/dist)| = |n . p| / dist)
/// @param cloud Point Cloud with intensity and normals (will be updated in place)
/// @param exponent Exponent for distance correction
/// @param scale Scale factor
/// @param min_intensity Minimum intensity after correction
/// @param max_intensity Maximum intensity after correction
/// @param min_cos_theta Minimum cos(theta) to prevent division by zero at grazing angles (~10 deg = 0.17)
/// @param reference_distance Reference distance for normalization
/// @param sensor_pose Sensor pose represented in the same coordinate frame as the input cloud.
///        The correction uses vectors from sensor origin to each point (translation part of the pose).
inline void correct_intensity_with_normal(PointCloudShared& cloud, float exponent = 2.0f, float scale = 1.0f,
                                          float min_intensity = 0.0f, float max_intensity = 1000.0f,
                                          float min_cos_theta = 0.17f, float reference_distance = 1.0f,
                                          const Eigen::Isometry3f& sensor_pose = Eigen::Isometry3f::Identity()) {
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
    if (reference_distance <= 0.0f) {
        throw std::runtime_error("[correct_intensity_with_normal] reference_distance must be positive");
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
        const float inv_ref_dist_sq = 1.0f / (reference_distance * reference_distance);
        const float sensor_tx = sensor_pose.translation().x();
        const float sensor_ty = sensor_pose.translation().y();
        const float sensor_tz = sensor_pose.translation().z();

        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) {
                return;
            }

            const auto point = point_ptr[i];
            // Use vectors from sensor origin so transformed point clouds can still use incidence-based correction.
            const float px = point.x() - sensor_tx;
            const float py = point.y() - sensor_ty;
            const float pz = point.z() - sensor_tz;
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

            const float normalized_dist_sq = dist_sq * inv_ref_dist_sq;
            const float corrected_intensity =
                intensity_ptr[i] * sycl::pow(normalized_dist_sq, exponent * 0.5f) / cos_theta;
            intensity_ptr[i] = sycl::clamp(corrected_intensity * s, min, max);
        });
    });

    event.wait_and_throw();
}

}  // namespace intensity_correction

}  // namespace algorithms

}  // namespace sycl_points
