#pragma once

#include <memory>

#include <Eigen/Dense>

#include "sycl_points/algorithms/graph/graph_factor.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief One-time full navigation-state anchor for the graph-LIO root node.
class BootstrapStatePriorFactor : public GraphFactorBase {
public:
    BootstrapStatePriorFactor(std::shared_ptr<PoseNode> node, const NodeState& reference,
                              float sigma_pose, float sigma_velocity,
                              float sigma_accel_bias, float sigma_gyro_bias)
        : node_(std::move(node)), reference_(reference) {
        w_pose_ = inverse_variance(sigma_pose);
        w_vel_ = inverse_variance(sigma_velocity);
        w_acc_ = inverse_variance(sigma_accel_bias);
        w_gyr_ = inverse_variance(sigma_gyro_bias);
    }

    FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        const NodeState state = node_->state();
        const Eigen::Matrix<float, kNodeDof, 1> residual = make_residual(state);
        Eigen::Matrix<float, kNodeDof, kNodeDof> information =
            Eigen::Matrix<float, kNodeDof, kNodeDof>::Zero();
        information.block<kPoseDof, kPoseDof>(0, 0) =
            w_pose_ * Eigen::Matrix<float, kPoseDof, kPoseDof>::Identity();
        information.block<3, 3>(kVelOffset, kVelOffset) = w_vel_ * Eigen::Matrix3f::Identity();
        information.block<3, 3>(kAccBiasOffset, kAccBiasOffset) = w_acc_ * Eigen::Matrix3f::Identity();
        information.block<3, 3>(kGyrBiasOffset, kGyrBiasOffset) = w_gyr_ * Eigen::Matrix3f::Identity();
        Eigen::Matrix<float, kNodeDof, kNodeDof> jacobian =
            Eigen::Matrix<float, kNodeDof, kNodeDof>::Identity();
        jacobian.block<kPoseDof, kPoseDof>(0, 0) =
            eigen_utils::lie::se3_right_jacobian(residual.head<kPoseDof>());

        FactorLinearization result;
        result.full_state = true;
        result.H00_full = jacobian.transpose() * information * jacobian;
        result.b0_full = jacobian.transpose() * information * residual;
        result.error = 0.5f * residual.dot(information * residual);
        result.inlier = 1;
        result.source_linearization_pose = state.pose;
        return result;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f& source,
                                              const Eigen::Isometry3f&) const override {
        NodeState state = node_->state();
        state.pose = source;
        return compute_error_state(state, NodeState{});
    }

    std::pair<float, uint32_t> compute_error_state(const NodeState& source,
                                                   const NodeState&) const override {
        const auto residual = make_residual(source);
        const float error = 0.5f * (w_pose_ * residual.head<kPoseDof>().squaredNorm() +
                                    w_vel_ * residual.segment<3>(kVelOffset).squaredNorm() +
                                    w_acc_ * residual.segment<3>(kAccBiasOffset).squaredNorm() +
                                    w_gyr_ * residual.segment<3>(kGyrBiasOffset).squaredNorm());
        return {error, 1};
    }

    std::pair<NodeId, NodeId> node_ids() const override { return {node_->id, INVALID_NODE_ID}; }
    bool uses_full_state() const override { return true; }
    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    static float inverse_variance(float sigma) { return 1.0f / (sigma * sigma); }

    Eigen::Matrix<float, kNodeDof, 1> make_residual(const NodeState& state) const {
        Eigen::Matrix<float, kNodeDof, 1> residual;
        residual.head<kPoseDof>() = eigen_utils::lie::se3_log(reference_.pose.inverse() * state.pose);
        residual.segment<3>(kVelOffset) = state.velocity - reference_.velocity;
        residual.segment<3>(kAccBiasOffset) = state.accel_bias - reference_.accel_bias;
        residual.segment<3>(kGyrBiasOffset) = state.gyro_bias - reference_.gyro_bias;
        return residual;
    }

    std::shared_ptr<PoseNode> node_;
    NodeState reference_;
    float w_pose_ = 0.0f;
    float w_vel_ = 0.0f;
    float w_acc_ = 0.0f;
    float w_gyr_ = 0.0f;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
