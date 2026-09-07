#pragma once

#include <memory>
#include <utility>

#include <Eigen/Geometry>

#include "sycl_points/algorithms/deskew/relative_pose_deskew.hpp"
#include "sycl_points/algorithms/graph/sliding_window.hpp"
#include "sycl_points/points/point_cloud.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Re-deskew the tip node's cloud assuming constant velocity between
///        (prev_pose, tip->pose). GO analog of the per-iteration deskew inside
///        registration::pipeline::VelocityUpdateAligner.
///
/// The tip's kNN is intentionally NOT rebuilt here: a tip's own kNN is never
/// read within the frame it is created (binary factors use the target side's
/// kNN, the unary factor uses the submap's kNN), so it is built once when the
/// tip is promoted to a persistent node (keyframe gate pass / retention).
class TipVelocityUpdater {
public:
    /// @brief Deskew the raw tip scan with the refined tip pose, swap the
    ///        node's cloud, and clear the caches of every factor touching the
    ///        tip so the next linearization uses the new cloud.
    /// @return false when deskew is not applicable (missing timestamps,
    ///         non-positive duration, or deskew failure); the caller should
    ///         stop iterating and keep the last result.
    bool redeskew(SlidingWindow& window, NodeId tip_id, const PointCloudShared& raw_source,
                  const Eigen::Isometry3f& prev_pose, float dt) const {
        const auto tip = window.get_node(tip_id);
        if (!tip || !tip->cloud || !raw_source.has_timestamps() || !(dt > 0.0f)) {
            return false;
        }
        auto deskewed = std::make_shared<PointCloudShared>(raw_source.queue);
        if (!deskew::deskew_point_cloud_constant_velocity(raw_source, *deskewed, prev_pose, tip->pose, dt)) {
            return false;
        }
        tip->cloud = std::move(deskewed);
        for (auto& factor : window.factors()) {
            const auto [sid, tid] = factor->node_ids();
            if (sid == tip_id || tid == tip_id) {
                factor->clear_cache();
            }
        }
        return true;
    }
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
