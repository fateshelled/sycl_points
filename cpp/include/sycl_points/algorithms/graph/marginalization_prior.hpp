#pragma once

#include <vector>

#include <Eigen/Dense>

#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Dense Schur-complement prior over the marginalized node's Markov blanket.
class MarginalizationPrior {
public:
    struct PriorContribution {
        Eigen::MatrixXf H;
        Eigen::VectorXf b;
        float error;
    };

    std::vector<NodeId> node_ids;
    std::vector<Eigen::Isometry3f> linearization_poses;
    Eigen::MatrixXf H_prior;
    Eigen::VectorXf b_prior;
    float error_constant = 0.0f;

    PriorContribution evaluate(const std::vector<Eigen::Isometry3f>& current_poses) const {
        const size_t n = node_ids.size();
        // Each node deviates from its linearization pose by e_i = Log(T_lin^-1 T).
        // The solver perturbs poses on the right: T <- T Exp(delta), so
        //     e_i(delta) = Log(T_lin^-1 T Exp(delta)) = Log(Exp(e_i) Exp(delta))
        // and to first order e_i(delta) = e_i + Jr(e_i) delta with
        // Jr(e) = Jl(-e)^-1 (BCH; the plain identity Jr = I is only correct in
        // Euclidean coordinates). Transport the cached Schur model into the
        // current tangent with the block-diagonal A = diag(Jr(e_i)):
        //     H_current = A^T H_prior A,  b_current = e-independent part
        //     b_current = A^T (H_prior e + b_prior)
        Eigen::VectorXf e = Eigen::VectorXf::Zero(6 * n);
        Eigen::MatrixXf A = Eigen::MatrixXf::Zero(6 * n, 6 * n);
        for (size_t i = 0; i < n; ++i) {
            const Eigen::Isometry3f T_rel = linearization_poses[i].inverse() * current_poses[i];
            e.segment<6>(6 * i) = eigen_utils::lie::se3_log(T_rel);
            A.block<6, 6>(6 * i, 6 * i) = eigen_utils::lie::se3_right_jacobian(e.segment<6>(6 * i));
        }
        PriorContribution ret;
        ret.H = A.transpose() * H_prior * A;
        ret.b = A.transpose() * (H_prior * e + b_prior);  // updated by deviation from linearization point
        ret.error = 0.5f * e.dot(H_prior * e) + b_prior.dot(e) + error_constant;
        return ret;
    }

    bool is_valid() const {
        const Eigen::Index expected = static_cast<Eigen::Index>(6 * node_ids.size());
        return !node_ids.empty() && linearization_poses.size() == node_ids.size() &&
               H_prior.rows() == expected && H_prior.cols() == expected &&
               b_prior.size() == expected && H_prior.any();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
