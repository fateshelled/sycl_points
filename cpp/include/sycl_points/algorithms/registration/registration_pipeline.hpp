#pragma once

#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/feature/photometric_gradient.hpp"
#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/registration/deskew_registration_pipeline.hpp"
#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/algorithms/registration/robust_registration_pipeline.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Registration pipeline with robust/deskew orchestration and feature preparation.
class RegistrationPipeline {
public:
    using Ptr = std::shared_ptr<RegistrationPipeline>;

    struct PipelineParams {
        bool random_sampling_enable = true;
        size_t random_sampling_num = 1000;

        size_t covariance_estimation_neighbor_num = 10;
        bool covariance_estimation_m_estimation_enable = true;
        robust::RobustLossType covariance_estimation_m_estimation_type = robust::RobustLossType::GEMAN_MCCLURE;
        float covariance_estimation_m_estimation_mad_scale = 1.0f;
        float covariance_estimation_m_estimation_min_robust_scale = 5.0f;
        size_t covariance_estimation_m_estimation_max_iterations = 1;

        bool deskew_recompute_features = true;
    };

    RegistrationPipeline(const sycl_utils::DeviceQueue& queue, const RegistrationParams& params = RegistrationParams(),
                         const PipelineParams& pipeline_params = PipelineParams())
        : params_(params),
          pipeline_params_(pipeline_params),
          queue_(queue),
          registration_(std::make_shared<Registration>(queue, params)),
          preprocess_filter_(std::make_shared<filter::PreprocessFilter>(queue)),
          sampled_source_(std::make_shared<PointCloudShared>(queue)),
          deskewed_source_(std::make_shared<PointCloudShared>(queue)) {}

    RegistrationResult align(const PointCloudShared& source, const PointCloudShared& target, const knn::KNNBase& target_knn,
                             const TransformMatrix& initial_guess = TransformMatrix::Identity()) {
        RegistrationResult result;
        result.T.matrix() = initial_guess;

        this->prepare_input(source, *this->sampled_source_);
        this->prepare_features(*this->sampled_source_, target, target_knn, true, false);

        const RobustRegistrationPipeline robust_pipeline(this->params_);
        robust_pipeline.run([&](const RobustLoopState& robust_state) {
            result = this->registration_->align(*this->sampled_source_, target, target_knn, result.T.matrix(),
                                                robust_state.robust_scale, robust_state.rotation_robust_scale);
        });
        return result;
    }

    RegistrationResult align_velocity_update(const PointCloudShared& source, const PointCloudShared& target,
                                             const knn::KNNBase& target_knn,
                                             const TransformMatrix& initial_guess = TransformMatrix::Identity(),
                                             float dt = 0.1f, size_t velocity_update_iter = 1,
                                             const TransformMatrix& prev_pose = TransformMatrix::Identity()) {
        RegistrationResult result;
        result.T.matrix() = initial_guess;

        this->prepare_input(source, *this->sampled_source_);

        this->prepare_features(*this->sampled_source_, target, target_knn, true, false);

        const RobustRegistrationPipeline robust_pipeline(this->params_);
        const DeskewRegistrationPipeline deskew_pipeline(velocity_update_iter);

        robust_pipeline.run([&](const RobustLoopState& robust_state) {
            deskew_pipeline.run(
                *this->sampled_source_, *this->deskewed_source_, prev_pose, dt, this->params_.verbose,
                [&]() { return result.T.matrix(); }, [&](size_t, PointCloudShared& deskewed_cloud) {
                    if (this->pipeline_params_.deskew_recompute_features) {
                        this->prepare_source_features(deskewed_cloud, true);
                    }
                    result = this->registration_->align(deskewed_cloud, target, target_knn, result.T.matrix(),
                                                        robust_state.robust_scale,
                                                        robust_state.rotation_robust_scale);
                });
        });

        return result;
    }

private:
    RegistrationParams params_;
    PipelineParams pipeline_params_;
    sycl_utils::DeviceQueue queue_;

    Registration::Ptr registration_;
    filter::PreprocessFilter::Ptr preprocess_filter_;

    PointCloudShared::Ptr sampled_source_;
    PointCloudShared::Ptr deskewed_source_;

    knn::KNNResult source_knn_result_;
    knn::KNNResult target_knn_result_;

    bool need_source_covariance() const {
        return this->params_.reg_type == RegType::GICP || this->params_.rotation_constraint.enable;
    }

    bool need_target_covariance() const {
        return this->params_.reg_type == RegType::GICP || this->params_.reg_type == RegType::GENZ ||
               this->params_.reg_type == RegType::POINT_TO_DISTRIBUTION || this->params_.rotation_constraint.enable;
    }

    bool need_target_normals() const {
        return this->params_.reg_type == RegType::POINT_TO_PLANE || this->params_.reg_type == RegType::GENZ ||
               this->params_.photometric.enable;
    }

    bool need_target_gradients() const { return this->params_.photometric.enable; }

    void prepare_input(const PointCloudShared& source, PointCloudShared& output) {
        if (this->pipeline_params_.random_sampling_enable) {
            this->preprocess_filter_->random_sampling(source, output, this->pipeline_params_.random_sampling_num);
        } else {
            output = source;
        }
    }

    void prepare_features(PointCloudShared& source, const PointCloudShared& target, const knn::KNNBase& target_knn,
                          bool force_source_recompute, bool force_target_recompute) {
        this->prepare_source_features(source, force_source_recompute);
        this->prepare_target_features(target, target_knn, force_target_recompute);
    }

    void prepare_source_features(PointCloudShared& source, bool force_recompute) {
        if (!this->need_source_covariance()) {
            return;
        }
        if (!force_recompute && source.has_cov()) {
            return;
        }

        auto src_tree = knn::KDTree::build(this->queue_, source);
        auto events = src_tree->knn_search_async(source, this->pipeline_params_.covariance_estimation_neighbor_num,
                                                 this->source_knn_result_);

        if (this->pipeline_params_.covariance_estimation_m_estimation_enable) {
            events += algorithms::covariance::compute_covariances_with_m_estimation_async(
                this->source_knn_result_, source, this->pipeline_params_.covariance_estimation_m_estimation_type,
                this->pipeline_params_.covariance_estimation_m_estimation_mad_scale,
                this->pipeline_params_.covariance_estimation_m_estimation_min_robust_scale,
                this->pipeline_params_.covariance_estimation_m_estimation_max_iterations, events.evs);
        } else {
            events += algorithms::covariance::compute_covariances_async(this->source_knn_result_, source, events.evs);
        }
        events.wait_and_throw();
    }

    void prepare_target_features(const PointCloudShared& target, const knn::KNNBase& target_knn, bool force_recompute) {
        const bool require_cov = this->need_target_covariance();
        const bool require_normal = this->need_target_normals();
        const bool require_gradient = this->need_target_gradients();

        const bool has_cov = target.has_cov();
        const bool has_normal = target.has_normal();
        const bool has_gradient = target.has_color_gradient() || target.has_intensity_gradient();

        if (!force_recompute && (!require_cov || has_cov) && (!require_normal || has_normal) &&
            (!require_gradient || has_gradient)) {
            return;
        }

        auto knn_events = target_knn.knn_search_async(target, this->pipeline_params_.covariance_estimation_neighbor_num,
                                                      this->target_knn_result_);
        sycl_utils::events cov_events;

        if (require_cov && (force_recompute || !has_cov)) {
            cov_events += algorithms::covariance::compute_covariances_async(this->target_knn_result_, target,
                                                                            knn_events.evs);
        }

        if (require_normal && (force_recompute || !has_normal)) {
            if (require_cov && (force_recompute || !has_cov)) {
                cov_events += algorithms::covariance::compute_normals_from_covariances_async(target, cov_events.evs);
            } else {
                cov_events +=
                    algorithms::covariance::compute_normals_async(this->target_knn_result_, target, knn_events.evs);
            }
        }

        if (require_gradient && (force_recompute || !has_gradient)) {
            if (target.has_rgb()) {
                cov_events +=
                    algorithms::color_gradient::compute_color_gradients_async(target, this->target_knn_result_, knn_events.evs);
            } else if (target.has_intensity()) {
                cov_events += algorithms::intensity_gradient::compute_intensity_gradients_async(
                    target, this->target_knn_result_, knn_events.evs);
            }
        }

        knn_events.wait_and_throw();
        cov_events.wait_and_throw();
    }
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
