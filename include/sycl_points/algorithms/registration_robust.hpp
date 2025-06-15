#pragma once

#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Robust loss function types for M-estimation
enum class RobustLossType {
    NONE,           // No robust loss (standard least squares)
    HUBER,          // Huber loss
    TUKEY,          // Tukey bi-weight
    CAUCHY,         // Cauchy loss
    GERMAN_MCCLURE  // German-McClure loss
};

namespace kernel {

/// @brief Compute robust weight for given residual
/// @param residual_norm Norm of residual vector
/// @param threshold Threshold parameter for robust loss
/// @return Robust weight (0.0 to 1.0)
template <RobustLossType LossType = RobustLossType::NONE>
SYCL_EXTERNAL inline float compute_robust_weight(float residual_norm, float threshold) {
    if constexpr (LossType == RobustLossType::NONE) {
        return 1.0f;
    }
    if (residual_norm <= 1e-8f) {
        return 1.0f;
    }

    const float normalized_residual = residual_norm / threshold;

    if constexpr (LossType == RobustLossType::HUBER) {
        // Huber loss: w = min(1, threshold/|r|)
        return sycl::min(1.0f, 1.0f / normalized_residual);
    } else if constexpr (LossType == RobustLossType::TUKEY) {
        // Tukey bi-weight: w = (1 - (r/threshold)^2)^2 if |r| < threshold, else 0
        if (normalized_residual >= 1.0f) {
            return 0.0f;
        }
        const float x = normalized_residual * normalized_residual;
        const float factor = 1.0f - x;
        return factor * factor;
    } else if constexpr (LossType == RobustLossType::CAUCHY) {
        // Cauchy loss: w = 1 / (1 + (r/threshold)^2)
        const float x = normalized_residual * normalized_residual;
        return 1.0f / (1.0f + x);
    } else if constexpr (LossType == RobustLossType::GERMAN_MCCLURE) {
        // German-McClure loss: w = 1 / (1 + (r/threshold)^2)^2
        const float x = normalized_residual * normalized_residual;
        const float denominator = 1.0f + x;
        return 1.0f / (denominator * denominator);
    }

    return 1.0f;
}

/// @brief Compute robust error for given residual
/// @param residual_norm Norm of residual vector
/// @param threshold Threshold parameter for robust loss
/// @return Robust weight (0.0 to 1.0)
template <RobustLossType LossType = RobustLossType::NONE>
SYCL_EXTERNAL inline float compute_robust_error(float residual_norm, float threshold) {
    if constexpr (LossType == RobustLossType::NONE) {
        return 0.5f * residual_norm * residual_norm;
    } else if constexpr (LossType == RobustLossType::HUBER) {
        return residual_norm <= threshold ? 0.5f * residual_norm * residual_norm
                                          : threshold * (residual_norm - 0.5f * threshold);
    } else if constexpr (LossType == RobustLossType::TUKEY) {
        return residual_norm <= threshold
                   ? (threshold * threshold / 6.0f) *
                         (1.0f - sycl::pow(1.0f - ((residual_norm * residual_norm) / (threshold * threshold)), 3.0f))
                   : threshold * threshold / 6.0f;
    } else if constexpr (LossType == RobustLossType::CAUCHY) {
        return 0.5f * threshold * threshold *
               sycl::log(1.0f + ((residual_norm * residual_norm) / (threshold * threshold)));
    } else if constexpr (LossType == RobustLossType::GERMAN_MCCLURE) {
        return 0.5f * (threshold * threshold * residual_norm * residual_norm) /
               (threshold * threshold + residual_norm * residual_norm);
    }

    return 0.5f * residual_norm * residual_norm;
}

}  // namespace kernel

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points