#pragma once

#include <vector>

#include <Eigen/Dense>

#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Dense Schur-complement prior over the marginalized node's Markov blanket.
///
/// `dof` is 6 for the legacy pose-only layout (one se3 block per node) and 15
/// when the prior also constrains velocity and biases. Pose blocks are
/// transported with the SE(3) right Jacobian of the pose offset; velocity and
/// bias blocks are additive (identity transport).
class MarginalizationPrior {
public:
    using StateVector = std::vector<NodeState, Eigen::aligned_allocator<NodeState>>;

    struct PriorContribution {
        Eigen::MatrixXf H;
        Eigen::VectorXf b;
        float error;
    };

    int dof = kPoseDof;
    std::vector<NodeId> node_ids;
    std::vector<Eigen::Isometry3f> linearization_poses;
    /// @brief Linearization states (velocity/biases included) for dof==15.
    StateVector linearization_states;
    Eigen::MatrixXf H_prior;
    Eigen::VectorXf b_prior;
    float error_constant = 0.0f;

    /// @brief Pose-only compatibility overload (dof==6 priors). State priors
    ///        must use evaluate_states().
    PriorContribution evaluate(const std::vector<Eigen::Isometry3f>& current_poses) const {
        StateVector states;
        states.reserve(current_poses.size());
        for (const auto& p : current_poses) {
            NodeState s;
            s.pose = p;
            states.push_back(s);
        }
        return evaluate_states(states);
    }

    PriorContribution evaluate_states(const StateVector& current_states) const {
        const size_t n = node_ids.size();
        const Eigen::Index step = static_cast<Eigen::Index>(dof);
        // Each node deviates from its linearization state by e_i. Pose blocks
        // follow the solver's right perturbation T <- T Exp(delta):
        //     e_pose(delta) = Log(T_lin^-1 T Exp(delta)) = e_pose + Jr(e_pose) delta
        // (BCH), while velocity/bias are additive. Transport the cached Schur
        // model into the current tangent with the block-diagonal A.
        Eigen::VectorXf e = Eigen::VectorXf::Zero(step * static_cast<Eigen::Index>(n));
        Eigen::MatrixXf A = Eigen::MatrixXf::Zero(step * static_cast<Eigen::Index>(n),
                                                  step * static_cast<Eigen::Index>(n));
        const bool has_states = dof == kNodeDof && linearization_states.size() == n;
        for (size_t i = 0; i < n; ++i) {
            const Eigen::Index base = step * static_cast<Eigen::Index>(i);
            const Eigen::Isometry3f T_lin =
                has_states ? linearization_states[i].pose : linearization_poses[i];
            const Eigen::Isometry3f& T_cur = current_states[i].pose;
            e.segment<6>(base) = eigen_utils::lie::se3_log(T_lin.inverse() * T_cur);
            A.block<6, 6>(base, base) = eigen_utils::lie::se3_right_jacobian(e.segment<6>(base));
            if (dof == kNodeDof) {
                const Eigen::Vector3f lin_v = has_states ? linearization_states[i].velocity
                                                         : Eigen::Vector3f::Zero();
                const Eigen::Vector3f lin_ba = has_states ? linearization_states[i].accel_bias
                                                          : Eigen::Vector3f::Zero();
                const Eigen::Vector3f lin_bg = has_states ? linearization_states[i].gyro_bias
                                                          : Eigen::Vector3f::Zero();
                e.segment<3>(base + kVelOffset) = current_states[i].velocity - lin_v;
                e.segment<3>(base + kAccBiasOffset) = current_states[i].accel_bias - lin_ba;
                e.segment<3>(base + kGyrBiasOffset) = current_states[i].gyro_bias - lin_bg;
                A.block<3, 3>(base + kVelOffset, base + kVelOffset).setIdentity();
                A.block<3, 3>(base + kAccBiasOffset, base + kAccBiasOffset).setIdentity();
                A.block<3, 3>(base + kGyrBiasOffset, base + kGyrBiasOffset).setIdentity();
            }
        }
        PriorContribution ret;
        ret.H = A.transpose() * H_prior * A;
        ret.b = A.transpose() * (H_prior * e + b_prior);  // updated by deviation from linearization point
        ret.error = 0.5f * e.dot(H_prior * e) + b_prior.dot(e) + error_constant;
        return ret;
    }

    bool is_valid() const {
        const Eigen::Index step = static_cast<Eigen::Index>(dof);
        const Eigen::Index expected = step * static_cast<Eigen::Index>(node_ids.size());
        if (node_ids.empty()) return false;
        if (dof == kNodeDof) {
            if (linearization_states.size() != node_ids.size()) return false;
        } else if (linearization_poses.size() != node_ids.size()) {
            return false;
        }
        return H_prior.rows() == expected && H_prior.cols() == expected &&
               b_prior.size() == expected && H_prior.any();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
