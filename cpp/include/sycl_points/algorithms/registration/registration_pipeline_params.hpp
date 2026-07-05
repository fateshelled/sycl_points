#pragma once

#include <cstddef>

#include "sycl_points/algorithms/registration/registration_params.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

struct RegistrationRandomSamplingParams {
    bool enable = true;
    size_t num = 1000;
    bool use_intensities = false;
    float weighted_ratio = 0.8f;
};

struct RegistrationRobustScheduleParams {
    bool auto_scale = false;
    float init_scale = 10.0f;
    float min_scale = 0.5f;
    float rotation_init_scale = 10.0f;
    float rotation_min_scale = 0.5f;
    size_t auto_scaling_iter = 4;
};

struct RegistrationVelocityUpdateParams {
    bool enable = false;
    size_t iter = 1;
};

/// @brief Parameters for the LiDAR-only registration pipeline wrappers.
struct RegistrationPipelineParams {
    using RandomSampling = RegistrationRandomSamplingParams;
    using Robust = RegistrationRobustScheduleParams;
    using VelocityUpdate = RegistrationVelocityUpdateParams;

    RegistrationParams registration;
    RandomSampling random_sampling;
    Robust robust;
    VelocityUpdate velocity_update;
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
