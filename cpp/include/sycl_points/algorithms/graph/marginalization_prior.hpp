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
        Eigen::VectorXf e = Eigen::VectorXf::Zero(6 * node_ids.size());
        for (size_t i = 0; i < node_ids.size(); ++i) {
            const Eigen::Isometry3f T_rel = linearization_poses[i].inverse() * current_poses[i];
            e.segment<6>(6 * i) = eigen_utils::lie::se3_log(T_rel);
        }
        PriorContribution ret;
        ret.H = H_prior;
        ret.b = H_prior * e + b_prior;  // updated by deviation from linearization point
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
