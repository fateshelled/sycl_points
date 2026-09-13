#pragma once

#include <memory>
#include <optional>
#include <utility>

#include "sycl_points/algorithms/graph/graph_factor.hpp"
#include "sycl_points/algorithms/graph/graph_factor_kernel.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/algorithms/graph/relative_pose_factor.hpp"
#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/algorithms/registration/registration.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Unary GICP factor: a pose node against a fixed target (e.g. submap).
///
/// The connected source node is referenced so the factor linearizes at the
/// node's current linearization_pose. For Phase 1 the linearization point is
/// fixed (set once by the caller); later phases trigger relinearization.
class UnaryGicpFactor : public GraphFactorBase {
public:
    UnaryGicpFactor(const sycl_utils::DeviceQueue& queue, NodeId source_id,
                    std::shared_ptr<PoseNode> source_node, std::shared_ptr<const PointCloudShared> target,
                    std::shared_ptr<const knn::KNNBase> target_knn,
                    const registration::RegistrationParams& params)
        : source_id_(source_id),
          source_node_(std::move(source_node)),
          target_(std::move(target)),
          target_knn_(std::move(target_knn)),
          registration_(queue, params),
          error_(std::make_shared<shared_vector<float>>(1, 0.0f, *queue.ptr)),
          inlier_(std::make_shared<shared_vector<uint32_t>>(1, 0, *queue.ptr)) {
        begin_annealing();
    }

    FactorLinearization linearize(const sycl_utils::DeviceQueue&, float scale = 0.0f) override {
        source_node_->linearization_pose = source_node_->pose;
        registration::Registration::ExecutionOptions opts;
        if (scale > 0.0f) {
            opts.robust_scale = scale;
        }
        const auto result = registration_.compute_linearized_result(
            *source_node_->cloud, *target_, *target_knn_,
            source_node_->linearization_pose.matrix(), source_node_->linearization_pose.matrix(), opts);
        FactorLinearization ret;
        ret.H00 = result.H;
        ret.b0 = result.b;
        ret.error = result.error;
        ret.inlier = result.inlier;
        ret.source_linearization_pose = source_node_->linearization_pose;
        // H01 / H11 / b1 remain zero: target is fixed.
        return ret;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f& src_pose,
                                             const Eigen::Isometry3f&) const override {
        auto evaluation = compute_error_async(src_pose, Eigen::Isometry3f::Identity());
        evaluation.events.wait_and_throw();
        return evaluation.collect();
    }

    FactorErrorEvaluation compute_error_async(const Eigen::Isometry3f& src_pose,
                                               const Eigen::Isometry3f&) const override {
        if (cached_linearization() == nullptr) {
            throw std::logic_error("[UnaryGicpFactor::compute_error] linearize must be called first");
        }
        registration::Registration::ExecutionOptions opts;
        if (last_linearization_scale() > 0.0f) {
            opts.robust_scale = last_linearization_scale();
        }
        auto events = registration_.compute_error_frozen_async(
            *source_node_->cloud, *target_, src_pose.matrix(), *error_, *inlier_, opts);
        events.keep_alive.push_back(error_);
        events.keep_alive.push_back(inlier_);
        const auto error = error_;
        const auto inlier = inlier_;
        return {std::move(events), [error, inlier]() {
                    return std::pair<float, uint32_t>{(*error)[0], (*inlier)[0]};
                }};
    }

    std::pair<NodeId, NodeId> node_ids() const override { return {source_id_, INVALID_NODE_ID}; }

    bool contributes_lidar_observability() const override { return true; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float rot_th,
                               float trans_th) const override {
        // Judge against this factor's own cached linearization pose, NOT the
        // shared PoseNode::linearization_pose: an earlier factor that
        // relinearizes also refreshes the node field, and judging on it would
        // suppress relinearization for this factor even though its own cache
        // (frozen KNN / weights / Hessian) is stale. nullopt never reaches
        // here with a stale intent: the base get_linearization() relinearizes
        // whenever the cache is empty.
        const graph::FactorLinearization* lin = cached_linearization();
        if (lin == nullptr) return true;
        return relinearization_needed(source_node_->pose, lin->source_linearization_pose, rot_th,
                                      trans_th);
    }

private:
    NodeId source_id_ = INVALID_NODE_ID;
    std::shared_ptr<PoseNode> source_node_;
    std::shared_ptr<const PointCloudShared> target_;
    std::shared_ptr<const knn::KNNBase> target_knn_;
    registration::Registration registration_;
    mutable std::shared_ptr<shared_vector<float>> error_;
    mutable std::shared_ptr<shared_vector<uint32_t>> inlier_;
};

/// @brief Binary GICP factor: two pose nodes against each other (current <-> window_i).
///
/// Both endpoints are variables. The linearization is computed by the SYCL
/// two-sided Jacobian reduction in BinaryGicpLinearizer. For Phase 2 the
/// linearization point is fixed at the node's current linearization_pose.
class BinaryGicpFactor : public GraphFactorBase {
public:
    BinaryGicpFactor(const sycl_utils::DeviceQueue& queue, NodeId source_id,
                     std::shared_ptr<PoseNode> source_node, NodeId target_id,
                     std::shared_ptr<PoseNode> target_node,
                     const registration::RegistrationParams& params)
        : source_id_(source_id),
          source_node_(std::move(source_node)),
          target_id_(target_id),
          target_node_(std::move(target_node)),
          linearizer_(queue, params),
          error_(std::make_shared<shared_vector<float>>(1, 0.0f, *queue.ptr)),
          inlier_(std::make_shared<shared_vector<uint32_t>>(1, 0, *queue.ptr)) {
        begin_annealing();
    }

    FactorLinearization linearize(const sycl_utils::DeviceQueue& queue, float scale = 0.0f) override {
        source_node_->linearization_pose = source_node_->pose;
        if (target_node_) target_node_->linearization_pose = target_node_->pose;
        const Eigen::Matrix4f T_src = source_node_->linearization_pose.matrix();
        const Eigen::Matrix4f T_tgt = target_node_->linearization_pose.matrix();
        FactorLinearization lin;
        lin = linearizer_.linearize(*source_node_->cloud, *target_node_->knn, T_src,
                                    *target_node_->cloud, T_tgt, scale);
        lin.source_linearization_pose = source_node_->linearization_pose;
        lin.target_linearization_pose = target_node_->linearization_pose;
        return lin;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f& src_pose,
                                             const Eigen::Isometry3f& tgt_pose) const override {
        auto evaluation = compute_error_async(src_pose, tgt_pose);
        evaluation.events.wait_and_throw();
        return evaluation.collect();
    }

    FactorErrorEvaluation compute_error_async(const Eigen::Isometry3f& src_pose,
                                               const Eigen::Isometry3f& tgt_pose) const override {
        const Eigen::Matrix4f T_src = src_pose.matrix();
        const Eigen::Matrix4f T_tgt = tgt_pose.matrix();
        auto events = linearizer_.compute_error_frozen_async(
            *source_node_->cloud, T_src, *target_node_->cloud, T_tgt,
            *error_, *inlier_, last_linearization_scale());
        events.keep_alive.push_back(error_);
        events.keep_alive.push_back(inlier_);
        const auto error = error_;
        const auto inlier = inlier_;
        return {std::move(events), [error, inlier]() {
                    return std::pair<float, uint32_t>{(*error)[0], (*inlier)[0]};
                }};
    }

    std::pair<NodeId, NodeId> node_ids() const override { return {source_id_, target_id_}; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float rot_th,
                               float trans_th) const override {
        // Factor-local judgement (see UnaryGicpFactor): compare the current
        // node poses against the poses this factor itself linearized at.
        const graph::FactorLinearization* lin = cached_linearization();
        if (lin == nullptr) return true;
        if (relinearization_needed(source_node_->pose, lin->source_linearization_pose, rot_th,
                                   trans_th))
            return true;
        if (target_node_ &&
            relinearization_needed(target_node_->pose, lin->target_linearization_pose, rot_th,
                                   trans_th))
            return true;
        return false;
    }

    bool is_point_cloud_binary() const override { return true; }

    bool contributes_lidar_observability() const override { return true; }

    std::optional<RelativePoseMeasurement> make_relative_pose_measurement() const override {
        // Use the latest cached linearization as-is: the conversion must not
        // trigger extra GPU work. A missing cache yields nullopt and the caller
        // falls back to the sigma-based chain factor.
        const FactorLinearization* lin = cached_linearization();
        if (lin == nullptr) return std::nullopt;
        return relative_pose_measurement_from_linearization(*lin);
    }

private:
    NodeId source_id_ = INVALID_NODE_ID;
    std::shared_ptr<PoseNode> source_node_;
    NodeId target_id_ = INVALID_NODE_ID;
    std::shared_ptr<PoseNode> target_node_;
    BinaryGicpLinearizer linearizer_;
    mutable std::shared_ptr<shared_vector<float>> error_;
    mutable std::shared_ptr<shared_vector<uint32_t>> inlier_;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
