#pragma once

#include <cstddef>

#include "sycl_points/algorithms/registration/registration_params.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Parameters for the registration pipeline wrappers
struct RegistrationPipelineParams {
    RegistrationParams registration;  // Base registration solver parameters

    struct RandomSampling {
        bool enable = true;
        size_t num = 1000;
        // When true, use per-point intensity values as sampling weights via mixed weighted/uniform sampling.
        // Assumes intensities have been pre-normalized (e.g. by intensity_local_mean_norm); raw 0-255 intensities
        // will skew the distribution heavily toward retro-reflectors. Silently falls back to uniform sampling
        // when the source point cloud has no intensity field.
        bool use_intensities = false;
        // Fraction of samples drawn from the weighted distribution (rest are uniform). Range: [0.0, 1.0].
        float weighted_ratio = 0.8f;
    };

    /// @brief Parameters for multi-stage robust scale annealing
    struct Robust {
        bool auto_scale = false;   // If false, keep the robust scale fixed at RegistrationParams::robust.default_scale.
        float init_scale = 10.0f;  // Initial scale used by the ICP robust loss schedule when auto scaling is enabled
        float min_scale = 0.5f;    // Minimum scale reached by the ICP robust loss schedule
        float rotation_init_scale =
            10.0f;  // Initial scale used by the rotation constraint robust loss schedule when auto scaling is enabled
        float rotation_min_scale = 0.5f;  // Minimum scale reached by the rotation constraint robust loss schedule
        size_t auto_scaling_iter = 4;     // Number of robust-scale refinement levels
    };

    /// @brief Parameters for repeated deskew and re-alignment
    struct VelocityUpdate {
        bool enable = false;  // Enable motion-based deskew updates between alignments
        size_t iter = 1;      // Number of deskew and re-alignment iterations
    };

    RandomSampling random_sampling;  // Optional source random sampling before registration
    Robust robust;                   // Robust scale scheduling parameters
    VelocityUpdate velocity_update;  // Constant-velocity deskew refinement parameters
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
