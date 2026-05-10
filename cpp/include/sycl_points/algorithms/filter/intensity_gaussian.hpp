#pragma once

#include <stdexcept>

#include "sycl_points/algorithms/knn/result.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace intensity_gaussian {

namespace kernel {

/// @brief Directional Gaussian smoothing of intensity using KNN neighborhood.
///
/// Constructs a sensor-local orthonormal basis at p0 from its 3D position,
/// assuming the sensor is at the origin:
///
///   r_hat  = normalize(p0)                  — radial (range) direction
///   az_hat = normalize(-p0.y, p0.x, 0)      — azimuthal tangent (scan rotation)
///   el_hat = cross(r_hat, az_hat)            — elevation tangent (unit, no normalization needed)
///
/// The Gaussian weight for each neighbor j is:
///
///   w_j = exp(-0.5 * (dp_r²/σ_r² + dp_az²/σ_az² + dp_el²/σ_el²))
///
/// Near zenith (|p0.xy| < 1e-6), az_hat is clamped — the azimuth and elevation
/// components effectively degenerate to zero, making the filter isotropic in those
/// directions. This is acceptable since LiDAR rarely has meaningful points at zenith.
///
/// @param sigma_azimuth  σ for the horizontal (azimuthal) scan direction [same units as coordinates]
/// @param sigma_elevation σ for the vertical (elevation) direction [same units as coordinates]
/// @param sigma_range    σ for the radial (depth) direction [same units as coordinates]
/// @param k_stride  Full stride of index_ptr (== neighbors.k, the memory layout stride).
/// @param k_use     Number of neighbors to actually use (<= k_stride).
SYCL_EXTERNAL inline float compute(const PointType* points, const float* intensities, const int32_t* index_ptr,
                                   size_t k_stride, size_t k_use, size_t i, float inv2_az, float inv2_el,
                                   float inv2_r) {
    const float px = points[i].x();
    const float py = points[i].y();
    const float pz = points[i].z();

    const float r = sycl::sqrt(px * px + py * py + pz * pz);
    if (r < 1e-6f) return intensities[i];

    // Radial direction
    const float rx = px / r, ry = py / r, rz = pz / r;

    // Azimuthal tangent: cylindrical tangent in XY plane.
    // Near zenith (rxy < 1e-6), az_hat degenerates; use fallback basis (x-axis, y-axis)
    // via branchless select to avoid warp divergence on GPU.
    const float rxy = sycl::sqrt(px * px + py * py);
    const bool near_zenith = rxy < 1e-6f;
    const float inv_rxy = 1.0f / sycl::fmax(rxy, 1e-6f);
    const float ax = near_zenith ? 1.0f : (-py * inv_rxy);
    const float ay = near_zenith ? 0.0f : (px * inv_rxy);
    // az_hat.z == 0 in both normal and fallback cases

    // Elevation tangent: cross(r_hat, az_hat) in normal case, y-axis in fallback.
    // Analytically unit length when rxy > 0.
    const float ex = near_zenith ? 0.0f : (-rz * ay);
    const float ey = near_zenith ? 1.0f : (rz * ax);
    const float ez = near_zenith ? 0.0f : (rxy / r);

    float sum_w = 0.0f, sum_wI = 0.0f;

    for (size_t j = 0; j < k_use; ++j) {
        const int32_t idx = index_ptr[i * k_stride + j];
        const float dpx = points[idx].x() - px;
        const float dpy = points[idx].y() - py;
        const float dpz = points[idx].z() - pz;

        const float dp_r = dpx * rx + dpy * ry + dpz * rz;
        const float dp_az = dpx * ax + dpy * ay;  // az_hat.z == 0
        const float dp_el = dpx * ex + dpy * ey + dpz * ez;

        const float exponent = dp_r * dp_r * inv2_r + dp_az * dp_az * inv2_az + dp_el * dp_el * inv2_el;
        const float w = sycl::exp(-exponent);

        sum_w += w;
        sum_wI += w * intensities[idx];
    }

    return (sum_w > 0.0f) ? sum_wI / sum_w : intensities[i];
}

}  // namespace kernel

/// @brief Apply directional Gaussian smoothing to cloud.intensities in-place.
///        Uses a temporary buffer to avoid read/write race conditions (same pattern as
///        intensity_zscore::compute).
///
/// @param cloud          Point cloud with intensity field (modified in-place)
/// @param neighbors      KNN result (typically k >= 5 for meaningful smoothing)
/// @param sigma_azimuth  Gaussian σ in the horizontal scan direction [m]
/// @param sigma_elevation Gaussian σ in the vertical elevation direction [m]
/// @param sigma_range    Gaussian σ in the radial depth direction [m].
///                       Set small (e.g. 0.05) to preserve depth discontinuities at object
///                       boundaries. Set large to blend across range (angular-only smoothing).
/// @param k_limit        If > 0, use only the first k_limit neighbors from neighbors (must be <= neighbors.k).
///                       Allows reusing a larger KNNResult (e.g. from covariance estimation) while
///                       limiting the actual computation to the requested count.
inline void smooth_intensity(PointCloudShared& cloud, const knn::KNNResult& neighbors, float sigma_azimuth,
                             float sigma_elevation, float sigma_range = 0.05f, size_t k_limit = 0) {
    const size_t N = cloud.size();
    if (N == 0) return;
    if (!cloud.has_intensity()) {
        throw std::runtime_error("[intensity_gaussian::smooth_intensity] Intensity field not found");
    }
    if (neighbors.k < 1) {
        throw std::runtime_error("[intensity_gaussian::smooth_intensity] neighbors.k must be >= 1");
    }
    if (sigma_azimuth <= 0.0f || sigma_elevation <= 0.0f || sigma_range <= 0.0f) {
        throw std::runtime_error("[intensity_gaussian::smooth_intensity] All sigma values must be positive");
    }

    auto tmp = std::make_shared<shared_vector<float>>(N, 0.0f, *cloud.queue.ptr);

    auto event = cloud.queue.ptr->submit(  //
        [&, tmp](sycl::handler& h) {
            const size_t work_group_size = cloud.queue.get_work_group_size();
            const size_t global_size = cloud.queue.get_global_size(N);
            const auto indices = neighbors.indices;
            // k_stride: memory layout stride (always neighbors.k)
            // k_use:    number of neighbors to process in the kernel loop
            const size_t k_stride = neighbors.k;
            const size_t k_use = (k_limit > 0 && k_limit < k_stride) ? k_limit : k_stride;

            const float inv2_az = 0.5f / (sigma_azimuth * sigma_azimuth);
            const float inv2_el = 0.5f / (sigma_elevation * sigma_elevation);
            const float inv2_r = 0.5f / (sigma_range * sigma_range);

            const auto pt_ptr = cloud.points_ptr();
            const auto int_ptr = cloud.intensities_ptr();

            const auto idx_ptr = indices->data();
            const auto tmp_ptr = tmp->data();

            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                tmp_ptr[i] = kernel::compute(pt_ptr, int_ptr, idx_ptr, k_stride, k_use, i, inv2_az, inv2_el, inv2_r);
            });
        });
    event.wait_and_throw();

    std::swap(cloud.intensities, tmp);
}

}  // namespace intensity_gaussian
}  // namespace algorithms
}  // namespace sycl_points
