#pragma once

#include <memory>

#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/algorithms/registration/pipeline/aligner.hpp"
#include "sycl_points/algorithms/registration/pipeline/robust.hpp"
#include "sycl_points/algorithms/registration/pipeline/velocity_update.hpp"
#include "sycl_points/algorithms/registration/registration_pipeline_params.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Composes optional registration wrappers around the core Registration solver
class RegistrationPipeline {
public:
    using Ptr = std::shared_ptr<RegistrationPipeline>;

    /// @brief Constructor
    /// @param aligner Registration callable
    /// @param pipeline_params Parameters for the solver and optional wrappers
    RegistrationPipeline(pipeline::RegistrationAligner aligner,
                         const RegistrationPipelineParams& pipeline_params = RegistrationPipelineParams())
        : pipeline_params_(pipeline_params), aligner_(std::move(aligner)) {
        this->wrap_aligner();
    }

    /// @brief Constructor
    /// @param registration Registration backend
    /// @param pipeline_params Parameters for the solver and optional wrappers
    RegistrationPipeline(const Registration::Ptr& registration,
                         const RegistrationPipelineParams& pipeline_params = RegistrationPipelineParams())
        : RegistrationPipeline(pipeline::make_registration_aligner(registration), pipeline_params) {
        this->registration_ = registration;
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
        this->update_registration_input(source);
        return this->aligner_(*this->registration_input_pc_, target, target_knn, initial_guess, options);
    }

    /// @brief Returns the underlying Registration backend
    const Registration::Ptr& registration() const { return this->registration_; }

    /// @brief Computes geometry ICP robust weights using the latest registration input point cloud
    void compute_icp_robust_weights(const PointCloudShared& target, const knn::KNNBase& target_knn,
                                    const TransformMatrix& pose, float robust_scale, shared_vector<float>& out) const {
        if (this->registration_ == nullptr) {
            throw std::runtime_error(
                "[RegistrationPipeline::compute_icp_robust_weights] Registration backend is not available.");
        }
        const auto source = this->get_deskewed_point_cloud();
        if (source == nullptr) {
            throw std::runtime_error(
                "[RegistrationPipeline::compute_icp_robust_weights] Registration input point cloud is not available.");
        }
        this->registration_->compute_icp_robust_weights(*source, target, target_knn, pose, robust_scale, out);
    }

    /// @brief Returns the source point cloud used by the most recent align() call
    const PointCloudShared* get_registration_input_point_cloud() const { return this->registration_input_pc_.get(); }

    /// @brief Returns the deskewed source point cloud from the most recent align() call
    /// @note If velocity update is disabled, this returns the registration input point cloud.
    const PointCloudShared::Ptr get_deskewed_point_cloud() const {
        if (this->velocity_update_pipeline_ != nullptr) {
            return this->velocity_update_pipeline_->get_deskewed_point_cloud();
        }
        return this->registration_input_pc_;
    }

    float get_inlier_ratio(const RegistrationResult& result) const {
        const auto* input_cloud_ptr = this->get_registration_input_point_cloud();
        if (input_cloud_ptr && input_cloud_ptr->size() > 0) {
            return static_cast<float>(result.inlier) / static_cast<float>(input_cloud_ptr->size());
        }
        return 0.0f;
    }

private:
    void wrap_aligner() {
        // Execution order:
        //   pipeline::RobustAligner -> pipeline::VelocityUpdateAligner -> base aligner
        // Loop nesting:
        //   for each robust scale:
        //     for each deskew update:
        //       align(...)
        if (this->pipeline_params_.velocity_update.enable) {
            this->velocity_update_pipeline_ = std::make_shared<pipeline::VelocityUpdateAligner>(
                this->aligner_, this->pipeline_params_.velocity_update.iter,
                this->pipeline_params_.registration.verbose);
            this->aligner_ = this->velocity_update_pipeline_->make_aligner();
        }

        if (this->pipeline_params_.robust.auto_scale) {
            this->robust_pipeline_ = std::make_shared<pipeline::RobustAligner>(this->aligner_, this->pipeline_params_);
            this->aligner_ = this->robust_pipeline_->make_aligner();
        }
    }

    void initialize_runtime_state(const sycl_utils::DeviceQueue& queue) const {
        if (this->preprocess_filter_ != nullptr && this->registration_input_pc_ != nullptr) {
            return;
        }
        this->preprocess_filter_ = std::make_shared<filter::PreprocessFilter>(queue);
        this->registration_input_pc_ = std::make_shared<PointCloudShared>(queue);
    }

    void update_registration_input(const PointCloudShared& source) const {
        this->initialize_runtime_state(source.queue);
        if (this->pipeline_params_.random_sampling.enable &&
            source.size() > this->pipeline_params_.random_sampling.num) {
            this->preprocess_filter_->random_sampling(source, *this->registration_input_pc_,
                                                      this->pipeline_params_.random_sampling.num);
        } else {
            *this->registration_input_pc_ = source;
        }
    }

    Registration::Ptr registration_;
    RegistrationPipelineParams pipeline_params_;
    pipeline::RobustAligner::Ptr robust_pipeline_ = nullptr;
    pipeline::VelocityUpdateAligner::Ptr velocity_update_pipeline_ = nullptr;
    pipeline::RegistrationAligner aligner_;
    mutable filter::PreprocessFilter::Ptr preprocess_filter_ = nullptr;
    mutable PointCloudShared::Ptr registration_input_pc_ = nullptr;
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
