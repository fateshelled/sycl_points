#pragma once

#include <algorithm>
#include <cmath>

#include "sycl_points/algorithms/registration/degenerate_regularization.hpp"
#include "sycl_points/algorithms/registration/factor.hpp"
#include "sycl_points/algorithms/robust/robust.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

enum class OptimizationMethod {
    GAUSS_NEWTON = 0,
    LEVENBERG_MARQUARDT,
    POWELL_DOGLEG,
};

inline OptimizationMethod OptimizationMethod_from_string(const std::string& str) {
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
        float translation = 1e-3f;  // translation tolerance [m]
        float rotation = 1e-3f;     // rotation tolerance [rad]
    };
    struct Robust {
        robust::RobustLossType type = robust::RobustLossType::NONE;  // robust loss function type
        float default_scale = 10.0f;                                 // default scale for robust loss function
    };
    struct PhotometricTerm {
        bool enable = false;  // If true, use photometric term.
        float weight = 0.2f;  // Scaling factor to balance photometric error with geometric error
        float robust_scale = 5.0f;
        float zscore_sigma_min = 0.01f;  // Points with local σ below this are set to z=0 (flat region)
    };
    struct GenZ {
        float planarity_threshold = 0.2f;
    };
    struct RotationConstraint {
        struct Robust {
            float default_scale = 10.0f;  // default scale for robust loss function
        };

        bool enable = false;
        float weight = 1.0f;  // Scaling factor to balance constraint error with geometric error
        Robust robust;
    };

    struct GaussNewton {
        float lambda = 1.0f;  // damping factor
    };
    struct LevenbergMarquardt {
        size_t max_inner_iterations = 10;  // (for LM method)
        float lambda_factor = 2.0f;        // lambda increase factor (for LM method)
        float init_lambda = 1.0f;          // initial lambda (for LM method)
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

    struct AndersonAcceleration {
        bool enabled = false;     // If true, apply Anderson acceleration to the outer iteration
        size_t window_size = 5;   // History window size m (Anderson(m))
        float beta = 1.0f;        // Mixing parameter: 1.0 = pure Anderson acceleration
    };

    RegType reg_type = RegType::GICP;          // Registration Type
    size_t max_iterations = 20;                // max iteration
    float lambda = 1e-6f;                      // damping factor
    float max_correspondence_distance = 2.0f;  // max correspondence distance

    Criteria criteria;
    Robust robust;
    PhotometricTerm photometric;
    RotationConstraint rotation_constraint;
    GenZ genz;
    GaussNewton gn;
    LevenbergMarquardt lm;
    Dogleg dogleg;
    AndersonAcceleration anderson;
    OptimizationMethod optimization_method = OptimizationMethod::GAUSS_NEWTON;  // Optimization method selector

    DegenerateRegularizationParams degenerate_reg;  // Degenerate Regularization

    bool verbose = false;  // If true, print debug messages
};
}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
