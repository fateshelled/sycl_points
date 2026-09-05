#pragma once

#include <memory>
#include <optional>
#include <utility>

#include "sycl_points/algorithms/graph/graph_factor.hpp"
#include "sycl_points/algorithms/graph/graph_factor_kernel.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"
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
class UnaryGicpFactor : public GicpFactorBase {
public:
    UnaryGicpFactor(const sycl_utils::DeviceQueue& queue, NodeId source_id,
                    std::shared_ptr<PoseNode> source_node, std::shared_ptr<const PointCloudShared> target,
                    std::shared_ptr<const knn::KNNBase> target_knn,
                    const registration::RegistrationParams& params)
        : queue_(queue),
          source_id_(source_id),
          source_node_(std::move(source_node)),
          target_(std::move(target)),
          target_knn_(std::move(target_knn)),
          params_(params) {
        begin_annealing();
    }

    FactorLinearization linearize(const sycl_utils::DeviceQueue& queue, float scale = 0.0f,
                                  bool raw = false) override {
        source_node_->linearization_pose = source_node_->pose;
        registration::RegistrationParams params = params_;
        if (raw) {
            params.robust.type = robust::RobustLossType::NONE;
        }
        registration::Registration reg(queue, params);
        registration::Registration::ExecutionOptions opts;
        if (scale > 0.0f) {
            opts.robust_scale = scale;
        }
        const auto result = reg.compute_linearized_result(
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
        registration::Registration reg(queue_, params_);
        const auto [err, inlier] = reg.compute_error_frozen(*source_node_->cloud, *target_, src_pose.matrix());
        return {err, inlier};
    }

    std::pair<NodeId, NodeId> node_ids() const override { return {source_id_, INVALID_NODE_ID}; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float rot_th,
                               float trans_th) const override {
        // Reuse is allowed only when the connected node pose is still close to the
        // pose at which this factor was last linearized.
        return relinearization_needed(source_node_->pose, source_node_->linearization_pose, rot_th,
                                      trans_th);
    }

private:
    sycl_utils::DeviceQueue queue_;
    NodeId source_id_ = INVALID_NODE_ID;
    std::shared_ptr<PoseNode> source_node_;
    std::shared_ptr<const PointCloudShared> target_;
    std::shared_ptr<const knn::KNNBase> target_knn_;
    registration::RegistrationParams params_;
};

/// @brief Binary GICP factor: two pose nodes against each other (current <-> window_i).
///
/// Both endpoints are variables. The linearization is computed by the SYCL
/// two-sided Jacobian reduction in BinaryGicpLinearizer. For Phase 2 the
/// linearization point is fixed at the node's current linearization_pose.
class BinaryGicpFactor : public GicpFactorBase {
public:
    BinaryGicpFactor(const sycl_utils::DeviceQueue& queue, NodeId source_id,
                     std::shared_ptr<PoseNode> source_node, NodeId target_id,
                     std::shared_ptr<PoseNode> target_node,
                     const registration::RegistrationParams& params)
        : queue_(queue),
          source_id_(source_id),
          source_node_(std::move(source_node)),
          target_id_(target_id),
          target_node_(std::move(target_node)),
          linearizer_(queue, params),
          params_(params) {
        begin_annealing();
    }

    FactorLinearization linearize(const sycl_utils::DeviceQueue& queue, float scale = 0.0f,
                                  bool raw = false) override {
        source_node_->linearization_pose = source_node_->pose;
        if (target_node_) target_node_->linearization_pose = target_node_->pose;
        const Eigen::Matrix4f T_src = source_node_->linearization_pose.matrix();
        const Eigen::Matrix4f T_tgt = target_node_->linearization_pose.matrix();
        FactorLinearization lin;
        if (raw) {
            // Unweighted (robust loss = NONE) linearization for the Schur-complement
            // prior: keeps the marginalized block well conditioned for strongly
            // reweighted losses (e.g. Geman-McClure). Marginalization is once per
            // frame, so the on-demand raw linearizer buffer cost is negligible.
            registration::RegistrationParams raw_params = params_;
            raw_params.robust.type = robust::RobustLossType::NONE;
            BinaryGicpLinearizer raw_linearizer(queue, raw_params);
            lin = raw_linearizer.linearize(*source_node_->cloud, *target_node_->knn, T_src,
                                           *target_node_->cloud, T_tgt, scale);
        } else {
            lin = linearizer_.linearize(*source_node_->cloud, *target_node_->knn, T_src,
                                        *target_node_->cloud, T_tgt, scale);
        }
        lin.source_linearization_pose = source_node_->linearization_pose;
        lin.target_linearization_pose = target_node_->linearization_pose;
        return lin;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f& src_pose,
                                             const Eigen::Isometry3f& tgt_pose) const override {
        const Eigen::Matrix4f T_src = src_pose.matrix();
        const Eigen::Matrix4f T_tgt = tgt_pose.matrix();
        auto lin = linearizer_.linearize(*source_node_->cloud, *target_node_->knn, T_src,
                                         *target_node_->cloud, T_tgt);
        return {lin.error, lin.inlier};
    }

    std::pair<NodeId, NodeId> node_ids() const override { return {source_id_, target_id_}; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float rot_th,
                               float trans_th) const override {
        if (relinearization_needed(source_node_->pose, source_node_->linearization_pose, rot_th,
                                  trans_th))
            return true;
        if (target_node_ &&
            relinearization_needed(target_node_->pose, target_node_->linearization_pose, rot_th,
                                  trans_th))
            return true;
        return false;
    }

    bool is_point_cloud_binary() const override { return true; }

private:
    sycl_utils::DeviceQueue queue_;
    NodeId source_id_ = INVALID_NODE_ID;
    std::shared_ptr<PoseNode> source_node_;
    NodeId target_id_ = INVALID_NODE_ID;
    std::shared_ptr<PoseNode> target_node_;
    BinaryGicpLinearizer linearizer_;
    registration::RegistrationParams params_;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
