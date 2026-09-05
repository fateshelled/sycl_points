#pragma once

#include <vector>

#include <Eigen/Dense>

#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Schur-complement marginalization prior applied as a unary factor
///        on a single node (the newest kept node after marginalizing the oldest).
class MarginalizationPrior {
public:
    struct PriorContribution {
        Eigen::Matrix<float, 6, 6> H;
        Eigen::Matrix<float, 6, 1> b;
        float error;
    };

    std::vector<NodeId> node_ids;  // single element for the star-shaped prior
    std::vector<Eigen::Isometry3f> linearization_poses;
    Eigen::Matrix<float, 6, 6> H_prior = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> b_prior = Eigen::Matrix<float, 6, 1>::Zero();
    float error_constant = 0.0f;

    PriorContribution evaluate(const Eigen::Isometry3f& current_pose) const {
        const Eigen::Isometry3f T_rel = linearization_poses[0].inverse() * current_pose;
        const Eigen::Matrix<float, 6, 1> e = eigen_utils::lie::se3_log(T_rel);
        PriorContribution ret;
        ret.H = H_prior;
        ret.b = H_prior * e + b_prior;  // updated by deviation from linearization point
        ret.error = 0.5f * e.dot(H_prior * e) + b_prior.dot(e) + error_constant;
        return ret;
    }

    bool is_valid() const { return !node_ids.empty() && H_prior.any(); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
