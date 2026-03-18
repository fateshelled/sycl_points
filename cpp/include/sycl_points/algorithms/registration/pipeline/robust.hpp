#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

#include "sycl_points/algorithms/registration/pipeline/aligner.hpp"
#include "sycl_points/algorithms/registration/registration_pipeline_params.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {
namespace pipeline {

/// @brief Adds robust-scale annealing around a registration aligner
class RobustAligner {
public:
    using Ptr = std::shared_ptr<RobustAligner>;

    /// @brief Constructor
    /// @param aligner Wrapped aligner to execute at each robust-scale level
    /// @param pipeline_params Registration and pipeline parameters
    RobustAligner(RegistrationAligner aligner, const RegistrationPipelineParams& pipeline_params)
        : aligner_(std::move(aligner)),
          params_(pipeline_params.registration),
          pipeline_params_(pipeline_params.robust) {}

    /// @brief Constructor
    /// @param registration Registration backend to wrap
    /// @param pipeline_params Registration and pipeline parameters
    RobustAligner(const Registration::Ptr& registration, const RegistrationPipelineParams& pipeline_params)
        : RobustAligner(make_registration_aligner(registration), pipeline_params) {}

    /// @brief Aligns point clouds while optionally shrinking the robust loss scales over multiple passes
    /// @param source Source point cloud
    /// @param target Target point cloud
    /// @param target_knn KNN structure built on the target point cloud
    /// @param initial_guess Initial transformation matrix
    /// @param options Per-call execution overrides
    /// @return Registration result from the final robust-scale level
    RegistrationResult align(const PointCloudShared& source, const PointCloudShared& target,
                             const knn::KNNBase& target_knn,
                             const TransformMatrix& initial_guess = TransformMatrix::Identity(),
                             const Registration::ExecutionOptions& options = Registration::ExecutionOptions()) const {
        RegistrationResult result;
        result.T.matrix() = initial_guess;

        if (source.size() == 0) {
            return result;
        }

        const bool use_fixed_scales = options.robust_scale > 0.0f || options.rotation_robust_scale > 0.0f;
        bool enable_auto_scaling = !use_fixed_scales && this->params_.robust.type != robust::RobustLossType::NONE &&
                                   this->pipeline_params_.auto_scale;
        const float robust_init_scale = this->pipeline_params_.init_scale;
        const float rotation_robust_init_scale = this->pipeline_params_.rotation_init_scale;
        if (enable_auto_scaling &&
            (this->pipeline_params_.min_scale <= 0.0f || this->pipeline_params_.min_scale >= robust_init_scale)) {
            std::cout << "[Caution] `pipeline.robust.min_scale` must be greater than zero and less than "
                         "`pipeline.robust.init_scale`."
                      << std::endl;
            enable_auto_scaling = false;
        }
        if (enable_auto_scaling && (this->pipeline_params_.rotation_min_scale <= 0.0f ||
                                    this->pipeline_params_.rotation_min_scale >= rotation_robust_init_scale)) {
            std::cout << "[Caution] `pipeline.robust.rotation_min_scale` must be greater than zero and less than "
                         "`pipeline.robust.rotation_init_scale`."
                      << std::endl;
            enable_auto_scaling = false;
        }
        if (enable_auto_scaling && this->pipeline_params_.auto_scaling_iter == 0) {
            std::cout
                << "[Caution] `pipeline.robust.auto_scaling_iter` must be greater than zero. Disable auto scaling."
                << std::endl;
            enable_auto_scaling = false;
        }
        const size_t robust_levels =
            enable_auto_scaling ? std::max<size_t>(1, this->pipeline_params_.auto_scaling_iter) : 1;

        float robust_scale = options.robust_scale > 0.0f
                                 ? options.robust_scale
                                 : (enable_auto_scaling ? robust_init_scale : this->params_.robust.default_scale);
        const float robust_scaling_factor = robust_levels > 1
                                                ? std::pow(this->pipeline_params_.min_scale / robust_init_scale,
                                                           1.0f / static_cast<float>(robust_levels - 1))
                                                : 1.0f;

        float rotation_robust_scale =
            options.rotation_robust_scale > 0.0f
                ? options.rotation_robust_scale
                : (enable_auto_scaling ? rotation_robust_init_scale
                                       : this->params_.rotation_constraint.robust.default_scale);
        const float rotation_robust_scaling_factor =
            robust_levels > 1 ? std::pow(this->pipeline_params_.rotation_min_scale / rotation_robust_init_scale,
                                         1.0f / static_cast<float>(robust_levels - 1))
                              : 1.0f;

        for (size_t robust_level = 0; robust_level < robust_levels; ++robust_level) {
            if (enable_auto_scaling && this->params_.verbose) {
                std::cout << "Robust scale: " << robust_scale << std::endl;
            }

            auto level_options = options;
            level_options.robust_scale = robust_scale;
            level_options.rotation_robust_scale = rotation_robust_scale;
            result = this->aligner_(source, target, target_knn, result.T.matrix(), level_options);

            robust_scale *= robust_scaling_factor;
            rotation_robust_scale *= rotation_robust_scaling_factor;
        }

        return result;
    }

    /// @brief Exposes this aligner as a RegistrationAligner
    RegistrationAligner make_aligner() const {
        return [this](const PointCloudShared& source, const PointCloudShared& target, const knn::KNNBase& target_knn,
                      const TransformMatrix& initial_guess, const Registration::ExecutionOptions& options) {
            return this->align(source, target, target_knn, initial_guess, options);
        };
    }

private:
    RegistrationAligner aligner_;
    RegistrationParams params_;
    RegistrationPipelineParams::Robust pipeline_params_;
};

}  // namespace pipeline
}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
