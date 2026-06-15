#pragma once

#include <algorithm>
#include <cstdint>

namespace sycl_points {
namespace pipeline {
namespace lidar_odometry {

/// @brief Online self-calibrating estimator for the LiDAR footprint range bias.
///
/// Models the systematic range error caused by the elliptical laser footprint at far range /
/// grazing incidence as a per-point shift along the beam:
///   dr = k * range * tan(incidence)
/// applied in the sensor frame.  The coefficient @c k is dimensionless and is learned online
/// from each frame's point-to-plane residuals, so no sensor-specific magnitude (beam divergence,
/// range-noise sigma, ...) needs to be configured by hand.
///
/// Estimation is a damped fixed-point step.  Given the signed regressor x (= range * sin(theta) *
/// sign(beam . normal)) and the signed point-to-plane residual y of the already-corrected cloud, a
/// change dk shifts every residual by dk * x (the regressor is exactly the residual sensitivity
/// dy/dk).  Minimising sum(w * (y + dk * x)^2) gives dk = -sum(w*x*y) / sum(w*x*x), which drives the
/// residual/regressor correlation to zero.  At convergence k stops changing.
///
/// Updates are gated to well-conditioned frames so the bias is only learned where the pose is well
/// constrained; in degenerate scenes (narrow corridors, narrow->wide transitions) k is frozen and
/// the last good value keeps being applied.  Because the correction is a position shift it never
/// removes a constraint, so freezing/applying it does not worsen degeneracy.
class RangeBiasEstimator {
public:
    struct Params {
        bool enable = false;
        /// Damping (learning rate) applied to each fixed-point step in (0, 1].
        float learning_rate = 0.5f;
        /// Clamp on |k| (dimensionless).
        float max_abs_k = 0.05f;
        /// Clamp on the per-frame |delta k|, rate-limiting how fast k can move.
        float max_step = 5.0e-3f;
        /// Clamp on |cos(incidence)| used when applying the correction (bounds tan()).
        float cos_min = 0.1f;
        /// Clamp on the per-point range shift [m].
        float max_correction = 0.5f;
        /// Geman-McClure scale [m] for residual outlier rejection during estimation.
        float residual_robust_scale = 0.3f;
        /// Gate: minimum number of inlier correspondences required to update k.
        std::uint32_t min_inlier = 200;
        /// Gate: minimum regressor energy sum(w*x*x) required (observability of the bias).
        float min_sxx = 1.0f;
        /// Gate: minimum (smallest Hessian eigenvalue / inlier) required (scene conditioning).
        float min_condition = 1.0f;
    };

    RangeBiasEstimator() = default;
    explicit RangeBiasEstimator(const Params& params) : params_(params) {}

    void set_params(const Params& params) { this->params_ = params; }
    const Params& params() const { return this->params_; }

    /// @brief Current calibrated coefficient (0 until the first successful update).
    float k() const { return this->k_; }
    void reset() { this->k_ = 0.0f; }

    /// @brief Incremental fixed-point update of k from one frame's residual statistics.
    /// @param s_xy  sum(w * x * y) over inlier correspondences.
    /// @param s_xx  sum(w * x * x) over inlier correspondences.
    /// @param count number of inlier correspondences used.
    /// @param min_eig_per_inlier  smallest Hessian eigenvalue divided by inlier count (conditioning).
    /// @return true if k was updated, false if the frame was gated out.
    bool update(double s_xy, double s_xx, std::uint32_t count, float min_eig_per_inlier) {
        if (!this->params_.enable) return false;
        if (count < this->params_.min_inlier) return false;
        if (s_xx < static_cast<double>(this->params_.min_sxx)) return false;
        if (min_eig_per_inlier < this->params_.min_condition) return false;

        double step = -(s_xy / s_xx) * static_cast<double>(this->params_.learning_rate);
        const double max_step = static_cast<double>(this->params_.max_step);
        step = std::clamp(step, -max_step, max_step);

        float k = this->k_ + static_cast<float>(step);
        this->k_ = std::clamp(k, -this->params_.max_abs_k, this->params_.max_abs_k);
        return true;
    }

private:
    Params params_;
    float k_ = 0.0f;
};

}  // namespace lidar_odometry
}  // namespace pipeline
}  // namespace sycl_points
