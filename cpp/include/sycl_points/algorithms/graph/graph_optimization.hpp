#pragma once

#include <algorithm>
#include <cmath>
#include <memory>

#include <Eigen/Geometry>

#include "sycl_points/algorithms/graph/gicp_factor.hpp"
#include "sycl_points/algorithms/graph/graph_solver.hpp"
#include "sycl_points/algorithms/graph/sliding_window.hpp"
#include "sycl_points/algorithms/graph/velocity_update.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
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

    /// @brief Constant-velocity deskew context for the per-frame tip node.
    /// Mirrors the align path's velocity-update inputs (prev_pose / dt, see
    /// registration::pipeline::VelocityUpdateAligner) plus the raw tip scan.
    struct VelocityUpdateContext {
        bool enable = false;
        size_t iterations = 1;  ///< deskew+re-solve rounds (RegistrationVelocityUpdateParams::iter)
        /// @brief Previous frame pose: the deskew velocity basis (S of log(S^-1 E)).
        Eigen::Isometry3f prev_pose = Eigen::Isometry3f::Identity();
        float dt = 0.0f;  ///< inter-frame duration [s] (deskew duration basis)
        /// @brief Raw (pre-deskew) tip scan with timestamps. When the update is
        ///        active, each inner round re-deskews this copy with the
        ///        refined tip pose; the caller passes a nullptr tip kNN since
        ///        the tip's own kNN is only needed once it becomes a binary
        ///        target (built at promotion, see process_frame).
        std::shared_ptr<const PointCloudShared> raw_source = nullptr;
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
        /// @brief Final deskewed tip cloud when the velocity update ran,
        ///        nullptr otherwise. The pipeline feeds this exact cloud to the
        ///        submap so the map shares the tip node's reference frame.
        std::shared_ptr<PointCloudShared> tip_cloud = nullptr;
        /// @brief Inlier ratio of the tip's unary submap factor from the last
        ///        solver linearization (LO analog: RegistrationPipeline::
        ///        get_inlier_ratio, i.e. inlier count / factor input size).
        ///        Falls back to 1.0 (neutral, never blocks the submap gate)
        ///        when the ratio is unavailable.
        float inlier_ratio = 0.0f;
    };

    FrameResult process_frame(std::shared_ptr<PointCloudShared> source_cloud,
                              std::shared_ptr<const PointCloudShared> submap_cloud,
                              std::shared_ptr<const knn::KNNBase> submap_knn,
                              std::shared_ptr<knn::KNNBase> source_knn,
                              const Eigen::Isometry3f& initial_pose, double timestamp,
                              const registration::RegistrationParams& reg_params) {
        return this->process_frame(source_cloud, submap_cloud, submap_knn, source_knn, initial_pose, timestamp,
                                   reg_params, VelocityUpdateContext());
    }

    FrameResult process_frame(std::shared_ptr<PointCloudShared> source_cloud,
                              std::shared_ptr<const PointCloudShared> submap_cloud,
                              std::shared_ptr<const knn::KNNBase> submap_knn,
                              std::shared_ptr<knn::KNNBase> source_knn,
                              const Eigen::Isometry3f& initial_pose, double timestamp,
                              const registration::RegistrationParams& reg_params,
                              const VelocityUpdateContext& vu) {
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
        auto unary_factor = std::make_shared<UnaryGicpFactor>(
            queue_, current_id, current_node, submap_, submap_knn_, reg_params);
        window_.add_factor(unary_factor);

        // 2b. Binary GICP factors: current <-> each existing active window node.
        for (auto& node : window_.active_nodes()) {
            if (node->id == current_id) continue;
            if (!node->knn) continue;  // need a kNN on the target node's cloud
            window_.add_factor(std::make_shared<BinaryGicpFactor>(
                queue_, current_id, current_node, node->id, node, reg_params));
        }

        // 3. Local BA. The frame-level schedule mirrors the align path
        //    (RobustAligner -> VelocityUpdateAligner -> align): robust ladder
        //    rungs outermost (one optimize() pass per rung), deskew+re-solve
        //    rounds innermost. The very first pass consumes the pipeline's
        //    iter-0 deskew (basis: prev_pose, initial pose); every later pass
        //    re-deskews with the refined tip pose before optimizing.
        const bool vu_active =
            vu.enable && vu.raw_source != nullptr && vu.raw_source->has_timestamps() && vu.dt > 0.0f;
        const size_t vu_rounds = vu_active ? std::max<size_t>(1, vu.iterations) : 1;
        const GraphSolverParams::RobustSchedule& robust = solver_.params().robust;
        const size_t rungs = robust.enable ? std::max<size_t>(1, robust.levels) : 1;
        for (auto& f : window_.factors()) {
            f->set_robust_force_mode(robust.relinearize_per_rung);
        }
        FrameResult fr;
        {
            const TipVelocityUpdater tip_velocity_updater;
            bool first_pass = true;
            bool deskew_stopped = false;
            for (size_t rung = 0; rung < rungs && !deskew_stopped; ++rung) {
                const float rung_scale = robust.enable ? robust_ladder_scale_at_level(robust, rung) : 0.0f;
                for (size_t v = 0; v < vu_rounds; ++v) {
                    if (!first_pass && vu_active &&
                        !tip_velocity_updater.redeskew(window_, current_id, *vu.raw_source, vu.prev_pose,
                                                       vu.dt)) {
                        // Deskew no longer applicable: keep the last result.
                        deskew_stopped = true;
                        break;
                    }
                    first_pass = false;
                    if (v == 0 && robust.enable && solver_.params().verbose) {
                        std::cout << "Robust scale: " << rung_scale << std::endl;
                    }
                    const auto result = solver_.optimize(window_, rung_scale);
                    fr.converged = result.converged;
                    fr.iterations = result.iterations;
                    fr.error = result.final_error;
                }
            }
        }

        // 3b. End of the robust ladder for this frame: tip factors lock their
        //     last (floor) scale; already-frozen factors are unaffected.
        window_.finalize_robust();

        // 4. Keyframe gate: keep the solved tip pose, then decide persistence.
        auto cur = window_.get_node(current_id);
        fr.current_pose = cur ? cur->pose : initial_pose;
        fr.tip_cloud = cur ? cur->cloud : nullptr;

        // Tip inlier ratio from the last linearization of the tip's unary submap
        // factor: the same "final-iteration inlier count / factor input size"
        // statistic as the align path's get_inlier_ratio.
        if (cur && cur->cloud && cur->cloud->size() > 0) {
            if (const auto* lin = unary_factor->cached_linearization()) {
                const float ratio =
                    static_cast<float>(lin->inlier) / static_cast<float>(cur->cloud->size());
                fr.inlier_ratio = std::isfinite(ratio) ? std::clamp(ratio, 0.0f, 1.0f) : 0.0f;
            }
        }

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

        // 4b. Deferred tip kNN build: the tip's own kNN is never read within
        //     its own frame (unary uses the submap kNN, binaries use the target
        //     side's kNN), so build it exactly once here on the final cloud
        //     when the node is retained. Next frame's binary factors consume it
        //     as the target kNN.
        if (vu_active && cur && cur->knn == nullptr) {
            cur->knn = knn::KDTree::build(queue_, *cur->cloud);
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
