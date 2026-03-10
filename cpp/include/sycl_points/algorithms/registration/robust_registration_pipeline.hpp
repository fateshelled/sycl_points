#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

#include "sycl_points/algorithms/registration/registration_params.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

struct RobustLoopState {
    size_t level = 0;
    float robust_scale = 1.0f;
    float rotation_robust_scale = 1.0f;
};

/// @brief Pipeline for robust scale scheduling.
class RobustRegistrationPipeline {
public:
    explicit RobustRegistrationPipeline(const RegistrationParams& params) : params_(params) {
        enable_auto_scaling_ = params_.robust.type != robust::RobustLossType::NONE && params_.robust.auto_scale;
        robust_levels_ = enable_auto_scaling_ ? std::max<size_t>(1, params_.robust.auto_scaling_iter) : 1;
        robust_scaling_factor_ = robust_levels_ > 1
                                     ? std::pow(params_.robust.min_scale / params_.robust.init_scale,
                                                1.0f / static_cast<float>(robust_levels_ - 1))
                                     : 1.0f;
        rotation_robust_scaling_factor_ =
            robust_levels_ > 1
                ? std::pow(params_.rotation_constraint.robust_min_scale / params_.rotation_constraint.robust_init_scale,
                           1.0f / static_cast<float>(robust_levels_ - 1))
                : 1.0f;
    }

    template <typename Func>
    void run(Func&& func) const {
        RobustLoopState state;
        state.robust_scale = params_.robust.init_scale;
        state.rotation_robust_scale = params_.rotation_constraint.robust_init_scale;

        for (size_t robust_level = 0; robust_level < robust_levels_; ++robust_level) {
            state.level = robust_level;
            if (enable_auto_scaling_ && params_.verbose) {
                std::cout << "Robust scale: " << state.robust_scale << std::endl;
            }
            func(state);
            state.robust_scale *= robust_scaling_factor_;
            state.rotation_robust_scale *= rotation_robust_scaling_factor_;
        }
    }

private:
    const RegistrationParams& params_;
    bool enable_auto_scaling_ = false;
    size_t robust_levels_ = 1;
    float robust_scaling_factor_ = 1.0f;
    float rotation_robust_scaling_factor_ = 1.0f;
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points

