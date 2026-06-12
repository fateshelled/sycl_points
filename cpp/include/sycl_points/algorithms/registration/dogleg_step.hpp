#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Result of one Powell dogleg step computation.
template <int N>
struct DoglegStep {
    Eigen::Vector<float, N> p = Eigen::Vector<float, N>::Zero();  ///< Dogleg update δx
    float step_norm = 0.0f;                                       ///< ‖p‖
    float predicted_reduction = 0.0f;                             ///< Model reduction −(gᵀp + ½·pᵀHp)
};

/// @brief Compute the Powell dogleg step for the normal equation H·δ = −g within
///        a trust region.
///
/// Dimension-generic: used by the 6-DOF registration solver
/// (Registration::optimize_powell_dogleg) and the 15-DOF LIO update loop.
/// Trust-region bookkeeping (radius clamping, gain-ratio acceptance, radius
/// update) stays with the caller; this function only computes the step and the
/// predicted model reduction.
///
/// @param H                   Normal-equation Hessian (N×N, symmetric).
/// @param g                   Gradient vector (N×1); the descent direction is −g.
/// @param trust_region_radius Current trust-region radius (must be > 0).
/// @return Dogleg step, its norm, and the predicted model cost reduction.
template <int N>
inline DoglegStep<N> compute_dogleg_step(const Eigen::Matrix<float, N, N>& H, const Eigen::Vector<float, N>& g,
                                         float trust_region_radius) {
    DoglegStep<N> result;

    // Gauss-Newton step: H·p_gn = −g
    Eigen::Vector<float, N> p_gn = Eigen::Vector<float, N>::Zero();
    float norm_p_gn = 0.0f;
    bool has_valid_gn = false;
    {
        Eigen::LDLT<Eigen::Matrix<float, N, N>> ldlt;
        ldlt.compute(H);
        if (ldlt.info() == Eigen::Success && ldlt.vectorD().minCoeff() > 0.0f) {
            p_gn = ldlt.solve(-g);
            norm_p_gn = p_gn.norm();
            has_valid_gn = std::isfinite(norm_p_gn);
        }
    }

    // Steepest-descent (Cauchy) step: p_sd = −α·g with α = ‖g‖² / gᵀHg
    const float g_norm_sq = g.squaredNorm();
    const Eigen::Vector<float, N> Hg = H * g;
    const float g_H_g = g.dot(Hg);
    Eigen::Vector<float, N> p_sd = -g;
    if (g_H_g > std::numeric_limits<float>::epsilon()) {
        const float alpha = g_norm_sq / g_H_g;
        if (std::isfinite(alpha)) {
            p_sd = -alpha * g;
        }
    }
    const float norm_p_sd = p_sd.norm();

    // Dogleg combination
    if (has_valid_gn && norm_p_gn <= trust_region_radius) {
        result.p = p_gn;
        result.step_norm = norm_p_gn;
    } else if (norm_p_sd >= trust_region_radius) {
        if (norm_p_sd > std::numeric_limits<float>::epsilon()) {
            result.p = (trust_region_radius / norm_p_sd) * p_sd;
        }
        result.step_norm = trust_region_radius;
    } else if (has_valid_gn) {
        // Walk from p_sd toward p_gn until the trust-region boundary is hit.
        const Eigen::Vector<float, N> diff = p_gn - p_sd;
        const float a = diff.squaredNorm();
        const float b = 2.0f * p_sd.dot(diff);
        const float c = p_sd.squaredNorm() - trust_region_radius * trust_region_radius;
        float discriminant = b * b - 4.0f * a * c;
        discriminant = std::max(discriminant, 0.0f);
        float tau = 0.0f;
        if (a > std::numeric_limits<float>::epsilon()) {
            tau = (-b + std::sqrt(discriminant)) / (2.0f * a);
        }
        tau = std::clamp(tau, 0.0f, 1.0f);
        result.p = p_sd + tau * diff;
        result.step_norm = result.p.norm();
    } else {
        result.p = p_sd;
        if (norm_p_sd > trust_region_radius && norm_p_sd > std::numeric_limits<float>::epsilon()) {
            const float scale = trust_region_radius / norm_p_sd;
            result.p *= scale;
            result.step_norm = trust_region_radius;
        } else {
            result.step_norm = norm_p_sd;
        }
    }

    result.predicted_reduction = -(g.dot(result.p) + 0.5f * result.p.dot(H * result.p));
    return result;
}

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
