#pragma once

#include <memory>

#include <Eigen/Dense>

#include "sycl_points/algorithms/graph/graph_factor.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Weak full-state prior on a node's navigation state (velocity and
///        biases; the pose is left entirely to the LiDAR factors).
///
/// A pure preintegration graph has a (near-)gauge in the velocity/bias block:
/// with a single edge the constant-velocity offset and bias can trade off
/// against each other, so the 15-DOF normal matrix is rank deficient and the
/// solver (correctly) refuses to trust it. This factor regularizes exactly
/// those directions without touching the geometry:
///   r = [ v - v_ref ; ba - ba_ref ; bg - bg_ref ]
///   H = diag(0_pose6, 1/σ_v² I3, 1/σ_ba² I3, 1/σ_bg² I3)
/// The sigmas are deliberately loose: the prior only fixes the gauge, it does
/// not override the preintegration measurement.
class NavStatePriorFactor : public GraphFactorBase {
public:
    NavStatePriorFactor(std::shared_ptr<PoseNode> node, const NodeState& reference,
                        float sigma_velocity, float sigma_accel_bias, float sigma_gyro_bias)
        : node_(std::move(node)), reference_(reference) {
        w_vel_ = sigma_velocity > 0.0f ? 1.0f / (sigma_velocity * sigma_velocity) : 0.0f;
        w_acc_ = sigma_accel_bias > 0.0f ? 1.0f / (sigma_accel_bias * sigma_accel_bias) : 0.0f;
        w_gyr_ = sigma_gyro_bias > 0.0f ? 1.0f / (sigma_gyro_bias * sigma_gyro_bias) : 0.0f;
    }

    FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        const NodeState s = node_->state();
        Eigen::Matrix<float, 15, 1> r = Eigen::Matrix<float, 15, 1>::Zero();
        r.segment<3>(kVelOffset) = s.velocity - reference_.velocity;
        r.segment<3>(kAccBiasOffset) = s.accel_bias - reference_.accel_bias;
        r.segment<3>(kGyrBiasOffset) = s.gyro_bias - reference_.gyro_bias;

        Eigen::Matrix<float, 15, 15> H = Eigen::Matrix<float, 15, 15>::Zero();
        H.block<3, 3>(kVelOffset, kVelOffset) = w_vel_ * Eigen::Matrix3f::Identity();
        H.block<3, 3>(kAccBiasOffset, kAccBiasOffset) = w_acc_ * Eigen::Matrix3f::Identity();
        H.block<3, 3>(kGyrBiasOffset, kGyrBiasOffset) = w_gyr_ * Eigen::Matrix3f::Identity();

        FactorLinearization lin;
        lin.full_state = true;
        lin.H00_full = H;
        lin.b0_full = H * r;
        lin.error = 0.5f * r.dot(H * r);
        lin.inlier = 1;
        lin.source_linearization_pose = s.pose;
        return lin;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&, const Eigen::Isometry3f&) const override {
        return compute_error_state(node_->state(), NodeState{});
    }

    std::pair<float, uint32_t> compute_error_state(const NodeState& s, const NodeState&) const override {
        const Eigen::Vector3f dv = s.velocity - reference_.velocity;
        const Eigen::Vector3f dba = s.accel_bias - reference_.accel_bias;
        const Eigen::Vector3f dbg = s.gyro_bias - reference_.gyro_bias;
        const float error =
            0.5f * (w_vel_ * dv.squaredNorm() + w_acc_ * dba.squaredNorm() + w_gyr_ * dbg.squaredNorm());
        return {error, 1};
    }

    FactorErrorEvaluation compute_error_state_async(const NodeState& s,
                                                    const NodeState& t) const override {
        const auto result = compute_error_state(s, t);
        return {sycl_utils::events{}, [result]() { return result; }};
    }

    std::pair<NodeId, NodeId> node_ids() const override { return {node_->id, INVALID_NODE_ID}; }

    bool uses_full_state() const override { return true; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    std::shared_ptr<PoseNode> node_;
    NodeState reference_;
    float w_vel_ = 0.0f;
    float w_acc_ = 0.0f;
    float w_gyr_ = 0.0f;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
