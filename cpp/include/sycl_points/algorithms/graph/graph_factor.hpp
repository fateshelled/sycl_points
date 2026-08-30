#pragma once

#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief True if the relative pose (current vs linearization point) moved beyond
///        the relinearization thresholds. Rotation/translation norms follow the
///        same SE(3) convention as the solver (tail<3>() = rotation, head<3>() = translation).
inline bool relinearization_needed(const Eigen::Isometry3f& current, const Eigen::Isometry3f& lin,
                                   float rot_th, float trans_th) {
    const Eigen::Matrix<float, 6, 1> e = eigen_utils::lie::se3_log(lin.inverse() * current);
    return e.tail<3>().norm() > rot_th || e.head<3>().norm() > trans_th;
}

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

    /// @brief Return the linearization, reusing a cached result when the connected
    ///        node poses have not moved beyond the relinearization thresholds.
    ///        The default implementation always re-linearizes (no caching).
    virtual FactorLinearization get_linearization(const sycl_utils::DeviceQueue& queue,
                                                 float relinearize_rotation_thresh,
                                                 float relinearize_translation_thresh) {
        (void)relinearize_rotation_thresh;
        (void)relinearize_translation_thresh;
        return this->linearize(queue);
    }

    /// @brief Drop any cached linearization so the next get_linearization re-computes.
    virtual void clear_cache() {}
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
