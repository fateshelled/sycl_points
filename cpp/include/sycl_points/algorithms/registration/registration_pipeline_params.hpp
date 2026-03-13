#pragma once

#include <cstddef>

#include "sycl_points/algorithms/registration/registration_params.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Parameters for the registration pipeline wrappers
struct RegistrationPipelineParams {
    RegistrationParams registration;  // Base registration solver parameters

    /// @brief Parameters for multi-stage robust scale annealing
    struct Robust {
        bool auto_scale = false;  // If false, keep the robust scale fixed at RegistrationParams::robust.init_scale.
        float min_scale = 0.5f;   // Minimum scale reached by the ICP robust loss schedule
        size_t auto_scaling_iter = 4;  // Number of robust-scale refinement levels
    };

    /// @brief Parameters for repeated deskew and re-alignment
    struct VelocityUpdate {
        bool enable = false;  // Enable motion-based deskew updates between alignments
        size_t iter = 1;      // Number of deskew and re-alignment iterations
    };

    Robust robust;                   // Robust scale scheduling parameters
    VelocityUpdate velocity_update;  // Constant-velocity deskew refinement parameters
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
