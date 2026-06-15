#pragma once

#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {

/// @brief Apply a beam-direction range-bias correction in the sensor frame.
///
/// For each point, with beam direction d = p / |p|, range r = |p|, and local incidence angle theta
/// between the beam and the surface normal (smallest-eigenvalue eigenvector of the covariance):
///   p' = p + clamp(k * r * tan(theta), +-max_correction) * d
///
/// This compensates the systematic range error from the elliptical laser footprint at far range /
/// grazing incidence.  @c k is the dimensionless coefficient produced by RangeBiasEstimator.
///
/// No-op when k == 0, the cloud is empty, or per-point covariances are unavailable (the normal,
/// hence the incidence angle, cannot be derived).  The normal orientation is irrelevant because the
/// correction uses |cos(theta)|.
///
/// @param queue           SYCL device queue.
/// @param cloud           Point cloud in the sensor frame (modified in place).
/// @param k               Dimensionless range-bias coefficient.
/// @param cos_min         Clamp on |cos(theta)| to bound tan() near grazing rays.
/// @param max_correction  Clamp on the per-point range shift [m].
inline void apply_range_bias_correction(const sycl_utils::DeviceQueue& queue, PointCloudShared& cloud, float k,
                                        float cos_min, float max_correction) {
    if (k == 0.0f) return;
    const size_t N = cloud.size();
    if (N == 0 || !cloud.has_cov()) return;

    auto* points = cloud.points_ptr();
    const auto* covs = cloud.covs_ptr();
    const float cos_min_c = sycl::fmax(cos_min, 1e-3f);

    queue.ptr
        ->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
                const size_t i = id[0];
                PointType& p = points[i];
                const float r = sycl::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
                if (r < 1e-6f) return;

                Normal n;
                covariance::kernel::extract_normal(p, covs[i], n);

                const float inv_r = 1.0f / r;
                const float dx = p.x() * inv_r;
                const float dy = p.y() * inv_r;
                const float dz = p.z() * inv_r;

                float cos_theta = sycl::fabs(dx * n.x() + dy * n.y() + dz * n.z());
                cos_theta = sycl::clamp(cos_theta, cos_min_c, 1.0f);
                const float sin_theta = sycl::sqrt(sycl::fmax(0.0f, 1.0f - cos_theta * cos_theta));
                const float tan_theta = sin_theta / cos_theta;

                const float dr = sycl::clamp(k * r * tan_theta, -max_correction, max_correction);
                p.x() += dr * dx;
                p.y() += dr * dy;
                p.z() += dr * dz;
            });
        })
        .wait_and_throw();
}

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
