#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>

#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/algorithms/registration/registration_pipeline_params.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Callable registration interface used by pipeline wrappers
using RegistrationAligner =
    std::function<RegistrationResult(const PointCloudShared&, const PointCloudShared&, const knn::KNNBase&,
                                     const TransformMatrix&, const Registration::ExecutionOptions&)>;

/// @brief Creates a callable aligner from a Registration instance
inline RegistrationAligner make_registration_aligner(const Registration::Ptr& registration) {
    return
        [registration](const PointCloudShared& source, const PointCloudShared& target, const knn::KNNBase& target_knn,
                       const TransformMatrix& initial_guess, const Registration::ExecutionOptions& options) {
            return registration->align(source, target, target_knn, initial_guess, options);
        };
}

/// @brief Adds robust-scale annealing around a registration aligner
class RobustPipeline {
public:
    using Ptr = std::shared_ptr<RobustPipeline>;

    /// @brief Constructor
    /// @param aligner Wrapped aligner to execute at each robust-scale level
    /// @param pipeline_params Registration and pipeline parameters
    RobustPipeline(RegistrationAligner aligner, const RegistrationPipelineParams& pipeline_params)
        : aligner_(std::move(aligner)),
          params_(pipeline_params.registration),
          pipeline_params_(pipeline_params.robust) {}

    /// @brief Constructor
    /// @param registration Registration backend to wrap
    /// @param pipeline_params Registration and pipeline parameters
    RobustPipeline(const Registration::Ptr& registration, const RegistrationPipelineParams& pipeline_params)
        : RobustPipeline(make_registration_aligner(registration), pipeline_params) {}

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
        if (enable_auto_scaling && (this->pipeline_params_.min_scale <= 0.0f ||
                                    this->pipeline_params_.min_scale >= this->params_.robust.init_scale)) {
            std::cout
                << "[Caution] `pipeline.robust.min_scale` must be greater than zero and less than robust.init_scale."
                << std::endl;
            enable_auto_scaling = false;
        }
        if (enable_auto_scaling && (this->params_.rotation_constraint.robust.min_scale <= 0.0f ||
                                    this->params_.rotation_constraint.robust.min_scale >=
                                        this->params_.rotation_constraint.robust.init_scale)) {
            std::cout << "[Caution] `rotation_constraint.robust.min_scale` must be greater than zero and less than "
                         "rotation_constraint.robust.init_scale."
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

        float robust_scale = options.robust_scale > 0.0f ? options.robust_scale : this->params_.robust.init_scale;
        const float robust_scaling_factor =
            robust_levels > 1 ? std::pow(this->pipeline_params_.min_scale / this->params_.robust.init_scale,
                                         1.0f / static_cast<float>(robust_levels - 1))
                              : 1.0f;

        float rotation_robust_scale = options.rotation_robust_scale > 0.0f
                                          ? options.rotation_robust_scale
                                          : this->params_.rotation_constraint.robust.init_scale;
        const float rotation_robust_scaling_factor =
            robust_levels > 1 ? std::pow(this->params_.rotation_constraint.robust.min_scale /
                                             this->params_.rotation_constraint.robust.init_scale,
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

    /// @brief Exposes this pipeline as a RegistrationAligner
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

/// @brief Repeats deskew and registration assuming constant sensor velocity
class VelocityUpdatePipeline {
public:
    using Ptr = std::shared_ptr<VelocityUpdatePipeline>;

    /// @brief Constructor
    /// @param aligner Wrapped aligner to execute after each deskew update
    /// @param velocity_update_iter Number of deskew and re-alignment iterations
    /// @param verbose If true, print deskew progress
    VelocityUpdatePipeline(RegistrationAligner aligner, size_t velocity_update_iter, bool verbose = false)
        : aligner_(std::move(aligner)), velocity_update_iter_(velocity_update_iter), verbose_(verbose) {}

    /// @brief Constructor
    /// @param registration Registration backend to wrap
    /// @param velocity_update_iter Number of deskew and re-alignment iterations
    /// @param verbose If true, print deskew progress
    VelocityUpdatePipeline(const Registration::Ptr& registration, size_t velocity_update_iter, bool verbose = false)
        : VelocityUpdatePipeline(make_registration_aligner(registration), velocity_update_iter, verbose) {}

    /// @brief Aligns point clouds while iteratively updating the deskewed source cloud
    /// @param source Source point cloud
    /// @param target Target point cloud
    /// @param target_knn KNN structure built on the target point cloud
    /// @param initial_guess Initial transformation matrix
    /// @param options Per-call execution options including previous pose and time delta
    /// @return Registration result from the final deskew iteration
    RegistrationResult align(const PointCloudShared& source, const PointCloudShared& target,
                             const knn::KNNBase& target_knn,
                             const TransformMatrix& initial_guess = TransformMatrix::Identity(),
                             const Registration::ExecutionOptions& options = Registration::ExecutionOptions()) const {
        RegistrationResult result;
        result.T.matrix() = initial_guess;

        if (source.size() == 0) {
            return result;
        }

        PointCloudShared deskewed(source);
        const size_t deskew_levels = std::max<size_t>(1, this->velocity_update_iter_);

        for (size_t deskew_iter = 0; deskew_iter < deskew_levels; ++deskew_iter) {
            if (this->verbose_) {
                std::cout << "deskewed: " << deskew_iter << std::endl;
            }

            const Eigen::Isometry3f delta_pose = Eigen::Isometry3f(options.prev_pose).inverse() * result.T;
            const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(delta_pose);
            const float delta_angle = delta_twist.head<3>().norm();
            const float delta_dist = delta_twist.tail<3>().norm();
            if (this->verbose_) {
                std::cout << "deskewed[" << deskew_iter << "]: angle=" << delta_angle << ", dist=" << delta_dist
                          << std::endl;
            }

            deskew::deskew_point_cloud_constant_velocity(source, deskewed, Eigen::Isometry3f(options.prev_pose),
                                                         result.T, options.dt);
            result = this->aligner_(deskewed, target, target_knn, result.T.matrix(), options);
        }

        return result;
    }

    /// @brief Exposes this pipeline as a RegistrationAligner
    RegistrationAligner make_aligner() const {
        return [this](const PointCloudShared& source, const PointCloudShared& target, const knn::KNNBase& target_knn,
                      const TransformMatrix& initial_guess, const Registration::ExecutionOptions& options) {
            return this->align(source, target, target_knn, initial_guess, options);
        };
    }

private:
    RegistrationAligner aligner_;
    size_t velocity_update_iter_ = 1;
    bool verbose_ = false;
};

/// @brief Composes optional registration wrappers around the core Registration solver
class RegistrationPipeline {
public:
    using Ptr = std::shared_ptr<RegistrationPipeline>;

    /// @brief Constructor
    /// @param registration Registration backend
    /// @param pipeline_params Parameters for the solver and optional wrappers
    RegistrationPipeline(const Registration::Ptr& registration,
                         const RegistrationPipelineParams& pipeline_params = RegistrationPipelineParams())
        : registration_(registration) {
        this->aligner_ = make_registration_aligner(this->registration_);

        if (pipeline_params.velocity_update.enable) {
            this->velocity_update_pipeline_ = std::make_shared<VelocityUpdatePipeline>(
                this->aligner_, pipeline_params.velocity_update.iter, pipeline_params.registration.verbose);
            this->aligner_ = this->velocity_update_pipeline_->make_aligner();
        }

        if (pipeline_params.robust.auto_scale) {
            this->robust_pipeline_ = std::make_shared<RobustPipeline>(this->aligner_, pipeline_params);
            this->aligner_ = this->robust_pipeline_->make_aligner();
        }
    }

    /// @brief Constructor
    /// @param queue SYCL queue used to construct the registration backend
    /// @param pipeline_params Parameters for the solver and optional wrappers
    RegistrationPipeline(const sycl_utils::DeviceQueue& queue,
                         const RegistrationPipelineParams& pipeline_params = RegistrationPipelineParams())
        : RegistrationPipeline(std::make_shared<Registration>(queue, pipeline_params.registration), pipeline_params) {}

    /// @brief Aligns point clouds using the configured wrapper chain
    /// @param source Source point cloud
    /// @param target Target point cloud
    /// @param target_knn KNN structure built on the target point cloud
    /// @param initial_guess Initial transformation matrix
    /// @param options Per-call execution overrides
    /// @return Registration result
    RegistrationResult align(const PointCloudShared& source, const PointCloudShared& target,
                             const knn::KNNBase& target_knn,
                             const TransformMatrix& initial_guess = TransformMatrix::Identity(),
                             const Registration::ExecutionOptions& options = Registration::ExecutionOptions()) const {
        return this->aligner_(source, target, target_knn, initial_guess, options);
    }

    /// @brief Returns the underlying Registration backend
    const Registration::Ptr& registration() const { return this->registration_; }

private:
    Registration::Ptr registration_;
    RobustPipeline::Ptr robust_pipeline_ = nullptr;
    VelocityUpdatePipeline::Ptr velocity_update_pipeline_ = nullptr;
    RegistrationAligner aligner_;
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
