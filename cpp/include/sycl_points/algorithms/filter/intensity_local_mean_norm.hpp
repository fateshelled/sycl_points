#pragma once

#include <stdexcept>

#include "sycl_points/algorithms/filter/intensity_gaussian.hpp"
#include "sycl_points/algorithms/knn/result.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace intensity_local_mean_norm {

namespace kernel {

/// @brief Normalize intensity by the directional Gaussian-weighted local mean.
///
/// Computes the same Gaussian-weighted mean as intensity_gaussian::kernel::compute,
/// then returns I[i] / max(local_mean, mean_min).  Clamping the divisor avoids
/// division by near-zero while keeping the output continuous across the threshold
/// (no hard discontinuity as with a conditional skip).
/// The sensor-local orthonormal basis and weight formula are identical to
/// intensity_gaussian (see that file for geometry details).
///
/// @param mean_min  Floor applied to the local mean before division.
SYCL_EXTERNAL inline float compute(const PointType* points, const float* intensities, const int32_t* index_ptr,
                                   size_t k_stride, size_t k_use, size_t i, float inv2_az, float inv2_el, float inv2_r,
                                   float mean_min) {
    const float local_mean = intensity_gaussian::kernel::compute(points, intensities, index_ptr, k_stride, k_use, i,
                                                                 inv2_az, inv2_el, inv2_r);
    return intensities[i] / sycl::fmax(local_mean, mean_min);
}

}  // namespace kernel

/// @brief Normalize cloud.intensities in-place by the Gaussian-weighted local mean.
///        Each point's intensity is divided by max(local_mean, mean_min), removing
///        spatially-varying reflectivity bias while preserving local contrast.
///        The fmax-based clamp keeps the output continuous across the mean_min
///        threshold and prevents division by near-zero in dark regions.
///        Uses a temporary buffer to avoid read/write race conditions.
///
/// @param cloud           Point cloud with intensity field (modified in-place)
/// @param neighbors       KNN result (typically k >= 5)
/// @param sigma_azimuth   Gaussian σ in the horizontal scan direction [m]
/// @param sigma_elevation Gaussian σ in the vertical elevation direction [m]
/// @param sigma_range     Gaussian σ in the radial depth direction [m].
///                        Set small (e.g. 0.05) to avoid blending across range discontinuities.
/// @param mean_min        Floor applied to the local mean before division.
///                        Prevents near-zero division while keeping output continuous.
/// @param k_limit         If > 0, use only the first k_limit neighbors from neighbors.
inline void normalize(PointCloudShared& cloud, const knn::KNNResult& neighbors, float sigma_azimuth,
                      float sigma_elevation, float sigma_range = 0.05f, float mean_min = 1e-3f, size_t k_limit = 0) {
    const size_t N = cloud.size();
    if (N == 0) return;
    if (!cloud.has_intensity()) {
        throw std::runtime_error("[intensity_local_mean_norm::normalize] Intensity field not found");
    }
    if (neighbors.k < 1) {
        throw std::runtime_error("[intensity_local_mean_norm::normalize] neighbors.k must be >= 1");
    }
    if (sigma_azimuth <= 0.0f || sigma_elevation <= 0.0f || sigma_range <= 0.0f) {
        throw std::runtime_error("[intensity_local_mean_norm::normalize] All sigma values must be positive");
    }
    if (mean_min < 0.0f) {
        throw std::runtime_error("[intensity_local_mean_norm::normalize] mean_min must be non-negative");
    }

    auto tmp = std::make_shared<shared_vector<float>>(N, 0.0f, *cloud.queue.ptr);

    auto event = cloud.queue.ptr->submit([&, tmp](sycl::handler& h) {
        const size_t work_group_size = cloud.queue.get_work_group_size();
        const size_t global_size = cloud.queue.get_global_size(N);
        const auto indices = neighbors.indices;
        const size_t k_stride = neighbors.k;
        const size_t k_use = (k_limit > 0 && k_limit < k_stride) ? k_limit : k_stride;

        const float inv2_az = 0.5f / (sigma_azimuth * sigma_azimuth);
        const float inv2_el = 0.5f / (sigma_elevation * sigma_elevation);
        const float inv2_r = 0.5f / (sigma_range * sigma_range);

        const auto pt_ptr = cloud.points_ptr();
        const auto int_ptr = cloud.intensities_ptr();
        const auto idx_ptr = indices->data();
        const auto tmp_ptr = tmp->data();
        const float m_min = mean_min;

        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;
            tmp_ptr[i] = kernel::compute(pt_ptr, int_ptr, idx_ptr, k_stride, k_use, i, inv2_az, inv2_el, inv2_r, m_min);
        });
    });
    event.wait_and_throw();

    std::swap(cloud.intensities, tmp);
}

}  // namespace intensity_local_mean_norm
}  // namespace algorithms
}  // namespace sycl_points
