#pragma once

#include <algorithm>
#include <cmath>

#include "sycl_points/algorithms/registration/degenerate_regularization.hpp"
#include "sycl_points/algorithms/registration/factor.hpp"
#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/algorithms/registration/photometric_factor.hpp"
#include "sycl_points/algorithms/registration/rotation_constraint.hpp"
#include "sycl_points/algorithms/robust/robust.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

enum class OptimizationMethod {
    GAUSS_NEWTON = 0,
    LEVENBERG_MARQUARDT,
    POWELL_DOGLEG,
};

OptimizationMethod OptimizationMethod_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(str.begin(), str.end(), upper.begin(), [](u_char c) { return std::toupper(c); });

    if (upper.compare("GN") == 0 || upper.compare("GAUSS_NEWTON") == 0) {
        return OptimizationMethod::GAUSS_NEWTON;
    } else if (upper.compare("LM") == 0 || upper.compare("LEVENBERG_MARQUARDT") == 0) {
        return OptimizationMethod::LEVENBERG_MARQUARDT;
    } else if (upper.compare("DOGLEG") == 0 || upper.compare("POWELL_DOGLEG") == 0) {
        return OptimizationMethod::POWELL_DOGLEG;
    }
    std::string error_str = "[OptimizationMethod_from_string] Invalid OptimizationMethod str [";
    error_str += str;
    error_str += "]";
    throw std::runtime_error(error_str);
}

struct RegistrationParams {
    struct Criteria {
        float translation = 1e-3f;  // translation tolerance
        float rotation = 1e-3f;     // rotation tolerance [rad]
    };
    struct Robust {
        robust::RobustLossType type = robust::RobustLossType::NONE;  // robust loss function type
        bool auto_scale = false;                                     // enable auto robust scale
        float init_scale = 10.0f;                                    // scale for robust loss function
        float min_scale = 0.5f;                                      // minimum scale
        size_t scaling_iter = 4;                                     // scaling iteration
    };
    struct PhotometricTerm {
        bool enable = false;  // If true, use photometric term.
        float weight = 0.2f;  // Scaling factor to balance photometric error with geometric error
        float robust_scale = 5.0f;
    };
    struct GenZ {
        float planarity_threshold = 0.2f;
    };
    struct RotationConstraint {
        bool enable = false;
        float weight = 1.0f;              // Scaling factor to balance constraint error with geometric error
        float robust_init_scale = 10.0f;  // scale for robust loss function
        float robust_min_scale = 0.5f;    // minimum scale
    };
    struct LevenbergMarquardt {
        size_t max_inner_iterations = 10;  // (for LM method)
        float lambda_factor = 2.0f;        // lambda increase factor (for LM method)
        float max_lambda = 1e3f;           // max lambda (for LM method)
        float min_lambda = 1e-6f;          // min lambda (for LM method)
    };
    struct Dogleg {
        float initial_trust_region_radius = 1.0f;  // Initial trust region radius (for Powell's dogleg method)
        float min_trust_region_radius = 1e-4f;     // Minimum trust region radius (for Powell's dogleg method)
        float max_trust_region_radius = 10.0f;     // Maximum trust region radius (for Powell's dogleg method)
        float eta1 = 0.25f;                        // Lower acceptance threshold for ratio (for Powell's dogleg method)
        float eta2 = 0.75f;                        // Upper acceptance threshold for ratio (for Powell's dogleg method)
        float gamma_decrease = 0.25f;              // Shrink factor for trust region (for Powell's dogleg method)
        float gamma_increase = 2.0f;               // Expand factor for trust region (for Powell's dogleg method)
    };

    RegType reg_type = RegType::GICP;          // Registration Type
    size_t max_iterations = 20;                // max iteration
    float lambda = 1e-6f;                      // damping factor
    float max_correspondence_distance = 2.0f;  // max correspondence distance
    float mahalanobis_distance_threshold =
        100.0f;  // Mahalanobis distance threshold (for GICP and Point to Distribution)

    Criteria criteria;
    Robust robust;
    PhotometricTerm photometric;
    RotationConstraint rotation_constraint;
    GenZ genz;
    LevenbergMarquardt lm;
    Dogleg dogleg;
    OptimizationMethod optimization_method = OptimizationMethod::GAUSS_NEWTON;  // Optimization method selector

    DegenerateRegularizationParams degenerate_reg;  // Degenerate Regularization

    bool verbose = false;  // If true, print debug messages
};
}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
