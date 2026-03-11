#pragma once

#include <cstddef>

#include "sycl_points/algorithms/registration/registration_params.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

struct RegistrationPipelineParams {
    RegistrationParams registration;

    struct Robust {
        bool auto_scale = false;  // If false, the robust scale is fixed to RegistrationParams::robust.init_scale.
        float min_scale = 0.5f;   // minimum scale for the ICP robust loss
        float rotation_min_scale = 0.5f;  // minimum scale for the rotation constraint robust loss
        size_t auto_scaling_iter = 4;     // auto scaling iterations
    };

    struct VelocityUpdate {
        bool enable = false;
        size_t iter = 1;
    };

    Robust robust;
    VelocityUpdate velocity_update;
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
