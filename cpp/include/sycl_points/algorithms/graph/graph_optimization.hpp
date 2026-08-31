#pragma once

#include <memory>

#include <Eigen/Geometry>

#include "sycl_points/algorithms/graph/gicp_factor.hpp"
#include "sycl_points/algorithms/graph/graph_solver.hpp"
#include "sycl_points/algorithms/graph/sliding_window.hpp"
#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/points/point_cloud.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Integrated entry point for local BA (sliding-window graph optimization).
///
/// Phase 1: each frame is added as a node with a unary GICP factor against a
/// fixed submap. The Gauss-Newton solver optimizes all active nodes with a
/// single fixed linearization. Binary factors and marginalization follow in
/// later phases.
class GraphOptimization {
public:
    /// @brief Graph connectivity strategy for binary factors.
    enum class BinaryTopology {
        clique,        ///< every co-existing pair keeps its BinaryGicpFactor (legacy)
        sparse_chain,  ///< point-cloud binaries touch only the current tip; older
                       ///< adjacent pairs live on as host-only RelativePoseFactors
    };

    struct Options {
        /// @brief Keyframe gating for the sliding window. When enabled, a frame
        ///        is retained as a persistent node only if it moved beyond the
        ///        thresholds relative to the last kept keyframe; otherwise the
        ///        transient tip (and its fresh factors) is dropped after the
        ///        solve. Disabled => every frame persists (legacy per-frame window).
        struct KeyframeGate {
            bool enabled = false;
            float min_translation = 0.3f;  // [m]
            float min_rotation = 0.0873f;  // [rad] (~5 deg)
            float min_time_seconds = 0.0f;  // <=0: disabled
        };

        BinaryTopology binary_topology = BinaryTopology::sparse_chain;
        RelativePoseParams relative_pose;
        KeyframeGate gate;
    };

    GraphOptimization(const sycl_utils::DeviceQueue& queue,
                      const GraphSolverParams& solver_params = GraphSolverParams(),
                      size_t max_window_size = 5)
        : GraphOptimization(queue, solver_params, max_window_size, Options{}) {}

    GraphOptimization(const sycl_utils::DeviceQueue& queue,
                      const GraphSolverParams& solver_params, size_t max_window_size,
                      const Options& options)
        : queue_(queue), solver_(queue_, solver_params), window_(max_window_size), opts_(options) {}

    struct FrameResult {
        Eigen::Isometry3f current_pose;
        bool converged = false;
        size_t iterations = 0;
        float error = 0.0f;
        bool keyframe = true;
    };

    FrameResult process_frame(std::shared_ptr<PointCloudShared> source_cloud,
                              std::shared_ptr<const PointCloudShared> submap_cloud,
                              std::shared_ptr<const knn::KNNBase> submap_knn,
                              std::shared_ptr<knn::KNNBase> source_knn,
                              const Eigen::Isometry3f& initial_pose, double timestamp,
                              const registration::RegistrationParams& reg_params) {
        submap_ = std::move(submap_cloud);
        submap_knn_ = std::move(submap_knn);

        // 1. Add the new scan as a node (with its own kNN for future binary factors).
        NodeId current_id = window_.add_node(initial_pose, timestamp, source_cloud, std::move(source_knn));

        // 1b. Sparse-chain bookkeeping: the previous tip's point-cloud star is now
        //     stale; drop it except for the adjacent pair, which is frozen into a
        //     host-only RelativePoseFactor chain edge.
        if (opts_.binary_topology == BinaryTopology::sparse_chain) {
            auto& nodes = window_.active_nodes();
            const size_t n = nodes.size();
            const NodeId convert_a = (n >= 3) ? nodes[n - 3]->id : INVALID_NODE_ID;
            const NodeId convert_b = (n >= 3) ? nodes[n - 2]->id : INVALID_NODE_ID;
            window_.prune_point_cloud_binaries(current_id, convert_a, convert_b,
                                               opts_.relative_pose);
        }

        // 2a. Unary GICP factor: current <-> fixed submap.
        auto current_node = window_.get_node(current_id);
        window_.add_factor(std::make_shared<UnaryGicpFactor>(
            queue_, current_id, current_node, submap_, submap_knn_, reg_params));

        // 2b. Binary GICP factors: current <-> each existing active window node.
        for (auto& node : window_.active_nodes()) {
            if (node->id == current_id) continue;
            if (!node->knn) continue;  // need a kNN on the target node's cloud
            window_.add_factor(std::make_shared<BinaryGicpFactor>(
                queue_, current_id, current_node, node->id, node, reg_params));
        }

        // 3. Local BA (fixed linearization, Gauss-Newton).
        auto result = solver_.optimize(window_);

        // 4. Keyframe gate: keep the solved tip pose, then decide persistence.
        auto cur = window_.get_node(current_id);
        FrameResult fr;
        fr.current_pose = cur ? cur->pose : initial_pose;
        fr.converged = result.converged;
        fr.iterations = result.iterations;
        fr.error = result.final_error;

        if (opts_.gate.enabled) {
            const Eigen::Isometry3f d = last_keyframe_pose_.inverse() * fr.current_pose;
            const bool time_hit = opts_.gate.min_time_seconds > 0.0f && last_keyframe_time_ >= 0.0 &&
                                  timestamp - last_keyframe_time_ >= opts_.gate.min_time_seconds;
            const bool is_keyframe = !has_keyframe_ ||
                                     d.translation().norm() >= opts_.gate.min_translation ||
                                     Eigen::AngleAxisf(d.rotation()).angle() >= opts_.gate.min_rotation ||
                                     time_hit;
            fr.keyframe = is_keyframe;
            if (is_keyframe) {
                last_keyframe_pose_ = fr.current_pose;
                last_keyframe_time_ = timestamp;
                has_keyframe_ = true;
            } else {
                // Transient tip: its observation lives on only in fr.current_pose;
                // the next frame re-observes the same region through its fresh tip.
                window_.remove_node(current_id);
                return fr;
            }
        }

        // 5. Marginalize the oldest node once the window exceeds its max size.
        if (window_.window_size() > window_.max_window_size()) {
            window_.marginalize_oldest(
                queue_, opts_.binary_topology == BinaryTopology::sparse_chain
                            ? SlidingWindow::MarginalizationAnchor::OldestSurviving
                            : SlidingWindow::MarginalizationAnchor::Newest);
        }

        return fr;
    }

    SlidingWindow& window() { return window_; }
    const SlidingWindow& window() const { return window_; }

private:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    sycl_utils::DeviceQueue queue_;
    GraphSolver solver_;
    SlidingWindow window_;
    Options opts_;

    std::shared_ptr<const PointCloudShared> submap_;
    std::shared_ptr<const knn::KNNBase> submap_knn_;

    Eigen::Isometry3f last_keyframe_pose_ = Eigen::Isometry3f::Identity();
    double last_keyframe_time_ = -1.0;
    bool has_keyframe_ = false;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
