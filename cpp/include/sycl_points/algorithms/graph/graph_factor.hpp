#pragma once

#include "sycl_points/algorithms/graph/pose_node.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Abstract base for all GICP factors.
///
/// A factor connects either one node (unary, against a fixed target such as a
/// submap) or two nodes (binary, between two pose estimates). The linearization
/// point is captured at linearize() call time so the solver controls when
/// relinearization happens.
class GicpFactorBase {
public:
    using Ptr = std::shared_ptr<GicpFactorBase>;

    virtual ~GicpFactorBase() = default;

    /// @brief Linearize the factor at the current node estimates.
    virtual FactorLinearization linearize(const sycl_utils::DeviceQueue& queue) = 0;

    /// @brief Evaluate the robust error at the given source/target poses
    ///        using frozen correspondences.
    virtual std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f& src_pose,
                                                      const Eigen::Isometry3f& tgt_pose) const = 0;

    /// @brief IDs of the two connected nodes. For a unary factor the target
    ///        id is INVALID_NODE_ID (fixed target).
    virtual std::pair<NodeId, NodeId> node_ids() const = 0;

    /// @brief Whether the factor should be relinearized given the latest poses.
    virtual bool needs_relinearization(const Eigen::Isometry3f& src, const Eigen::Isometry3f& tgt,
                                       float rot_th, float trans_th) const = 0;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
