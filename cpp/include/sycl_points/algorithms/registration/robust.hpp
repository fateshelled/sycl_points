#pragma once

#include <sycl_points/utils/sycl_utils.hpp>
#include <tuple>
#include <type_traits>

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Robust loss function types for M-estimation
enum class RobustLossType {
    NONE,          // No robust loss (standard least squares)
    HUBER,         // Huber loss
    TUKEY,         // Tukey bi-weight
    CAUCHY,        // Cauchy loss
    GEMAN_MCCLURE  // Geman-McClure loss
};

/// @brief Robust loss Type tags
using RobustLossTypeTags = std::tuple<                                     //
    std::integral_constant<RobustLossType, RobustLossType::NONE>,          //
    std::integral_constant<RobustLossType, RobustLossType::HUBER>,         //
    std::integral_constant<RobustLossType, RobustLossType::TUKEY>,         //
    std::integral_constant<RobustLossType, RobustLossType::CAUCHY>,        //
    std::integral_constant<RobustLossType, RobustLossType::GEMAN_MCCLURE>  //
    >;

RobustLossType RobustLossType_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(str.begin(), str.end(), upper.begin(), [](u_char c) { return std::toupper(c); });
    if (upper == "NONE") {
        return RobustLossType::NONE;
    } else if (upper == "HUBER") {
        return RobustLossType::HUBER;
    } else if (upper == "TUKEY") {
        return RobustLossType::TUKEY;
    } else if (upper == "CAUCHY") {
        return RobustLossType::CAUCHY;
    } else if (upper == "GEMAN_MCCLURE") {
        return RobustLossType::GEMAN_MCCLURE;
    }
    std::string error_str = "[RobustLossType_from_string] Invalid RobustLossType str '";
    error_str += str;
    error_str += "'";
    throw std::runtime_error(error_str);
}

namespace kernel {

/// @brief Compute robust weight for given residual
/// @param residual_norm Norm of residual vector
/// @param scale Scale parameter for robust loss
/// @return Robust weight (0.0 to 1.0)
template <RobustLossType LossType = RobustLossType::NONE>
SYCL_EXTERNAL inline float compute_robust_weight(float residual_norm, float scale) {
    if constexpr (LossType == RobustLossType::NONE) {
        return 1.0f;
    }
    if (residual_norm <= 1e-8f) {
        return 1.0f;
    }

    const float normalized_residual = residual_norm / scale;

    if constexpr (LossType == RobustLossType::HUBER) {
        // Huber loss: w = min(1, scale/|r|)
        return sycl::min(1.0f, 1.0f / normalized_residual);
    } else if constexpr (LossType == RobustLossType::TUKEY) {
        // Tukey bi-weight: w = (1 - (r/scale)^2)^2 if |r| < scale, else 0
        if (normalized_residual >= 1.0f) {
            return 0.0f;
        }
        const float x = normalized_residual * normalized_residual;
        const float factor = 1.0f - x;
        return factor * factor;
    } else if constexpr (LossType == RobustLossType::CAUCHY) {
        // Cauchy loss: w = 1 / (1 + (r/scale)^2)
        const float x = normalized_residual * normalized_residual;
        return 1.0f / (1.0f + x);
    } else if constexpr (LossType == RobustLossType::GEMAN_MCCLURE) {
        // German-McClure loss: w = 1 / (1 + (r/scale)^2)^2
        const float x = normalized_residual * normalized_residual;
        const float denominator = 1.0f + x;
        return 1.0f / (denominator * denominator);
    }

    return 1.0f;
}

/// @brief Compute robust error for given residual
/// @param residual_norm Norm of residual vector
/// @param scale Scale parameter for robust loss
/// @return Robust weight (0.0 to 1.0)
template <RobustLossType LossType = RobustLossType::NONE>
SYCL_EXTERNAL inline float compute_robust_error(float residual_norm, float scale) {
    if constexpr (LossType == RobustLossType::NONE) {
        return 0.5f * residual_norm * residual_norm;
    } else if constexpr (LossType == RobustLossType::HUBER) {
        return residual_norm <= scale ? 0.5f * residual_norm * residual_norm : scale * (residual_norm - 0.5f * scale);
    } else if constexpr (LossType == RobustLossType::TUKEY) {
        return residual_norm <= scale
                   ? (scale * scale / 6.0f) *
                         (1.0f - sycl::pow(1.0f - ((residual_norm * residual_norm) / (scale * scale)), 3.0f))
                   : scale * scale / 6.0f;
    } else if constexpr (LossType == RobustLossType::CAUCHY) {
        return 0.5f * scale * scale * sycl::log(1.0f + ((residual_norm * residual_norm) / (scale * scale)));
    } else if constexpr (LossType == RobustLossType::GEMAN_MCCLURE) {
        return 0.5f * (scale * scale * residual_norm * residual_norm) / (scale * scale + residual_norm * residual_norm);
    }

    return 0.5f * residual_norm * residual_norm;
}

}  // namespace kernel

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
