#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>

#include <Eigen/Geometry>

#include "sycl_points/algorithms/graph/gicp_factor.hpp"
#include "sycl_points/algorithms/graph/graph_solver.hpp"
#include "sycl_points/algorithms/graph/sliding_window.hpp"
#include "sycl_points/algorithms/graph/velocity_update.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/algorithms/registration/result.hpp"
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
            /// @brief Defer the keep/drop decision to the owning pipeline. This lets
            ///        GraphOdometry use Submap::add_frame as the single LO-compatible
            ///        keyframe gate instead of maintaining duplicate state.
            bool external_decision = false;
            float min_translation = 0.3f;  // [m]
            float min_rotation = 0.0873f;  // [rad] (~5 deg)
            float min_time_seconds = 0.0f;  // <=0: disabled
            float min_inlier_ratio = 0.0f;
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
        : queue_(queue),
          solver_(queue_, solver_params),
          window_(max_window_size, solver_params.marginalization_lambda, frozen_marg_scale(solver_params)),
          opts_(options) {}

    struct FrameResult {
        NodeId current_node_id = INVALID_NODE_ID;
        Eigen::Isometry3f current_pose;
        bool converged = false;
        size_t iterations = 0;
        float error = 0.0f;
        bool keyframe = true;
        /// @brief Solver outcome of the last optimize() pass for this frame.
        ///        MAX_ITERATIONS (default) means the loop simply ran out of
        ///        iterations; an invalid status means the estimate is unusable.
        GraphSolver::Status solver_status = GraphSolver::Status::MAX_ITERATIONS;
        /// @brief True unless the solver ended in an unrecoverable failure
        ///        state (non-finite system / decomposition failure / bad step).
        ///        The pipeline must discard the frame instead of updating the
        ///        map or odometry when this is false.
        bool solver_valid() const { return GraphSolver::valid_status(solver_status); }
        /// @brief Exact sampled/final-deskewed cloud used by the tip factor.
        std::shared_ptr<PointCloudShared> tip_cloud = nullptr;
        /// @brief Inlier ratio of the tip's unary submap factor from the last
        ///        solver linearization (LO analog: RegistrationPipeline::
        ///        get_inlier_ratio, i.e. inlier count / factor input size).
        ///        Falls back to 0.0 when the ratio is unavailable.
        float inlier_ratio = 0.0f;
        /// @brief LO-compatible statistics from the final tip unary factor.
        registration::RegistrationResult tip_registration;
        float tip_robust_scale = 0.0f;
        bool finalized = false;
        /// @brief Marginalization failure reason at finalize time. Never overwritten
        ///        by the pipeline action (see marginalization_action).
        SlidingWindow::MarginalizationStatus marginalization_status =
            SlidingWindow::MarginalizationStatus::NotRequired;
        /// @brief Pipeline action taken after the marginalization attempt: a deferred
        ///        next-frame retry, or a force-drop of the oldest node on persistent
        ///        failure. Independent of the failure reason held in marginalization_status.
        SlidingWindow::MarginalizationAction marginalization_action =
            SlidingWindow::MarginalizationAction::None;
        float marginalization_lambda = 0.0f;
    };

    /// @brief Apply the authoritative keyframe decision for the current tip.
    ///
    /// GraphOdometry calls this with Submap::add_frame's result so the graph and
    /// map retain exactly the same frames. Standalone GraphOptimization users keep
    /// the internal gate unless KeyframeGate::external_decision is enabled.
    void finalize_frame(FrameResult& frame_result, bool keep) {
        if (frame_result.finalized || frame_result.current_node_id == INVALID_NODE_ID) return;

        frame_result.keyframe = keep;
        auto current = window_.get_node(frame_result.current_node_id);
        if (!keep) {
            window_.remove_node(frame_result.current_node_id);
            frame_result.finalized = true;
            return;
        }

        if (current && current->cloud && current->knn == nullptr) {
            current->knn = knn::KDTree::build(queue_, *current->cloud);
        }
        if (window_.window_size() > window_.max_window_size()) {
            const auto m = window_.marginalize_oldest(queue_);
            frame_result.marginalization_status = m.status;
            frame_result.marginalization_lambda = m.lambda_used;
            if (m.status == SlidingWindow::MarginalizationStatus::Success ||
                m.status == SlidingWindow::MarginalizationStatus::NotRequired) {
                frame_result.marginalization_action = SlidingWindow::MarginalizationAction::None;
            } else {
                // Persistent failure must not grow the window without bound: defer
                // for a next-frame retry first, then force-drop the oldest node (no
                // prior) once the cap is exceeded. The failure reason (m.status) is
                // preserved separately from this action.
                frame_result.marginalization_action = SlidingWindow::MarginalizationAction::Deferred;
                // Always log the failure (not gated on SYCL_POINTS_VERBOSE): a
                // persistently failing marginalization is operationally relevant.
                std::cerr << "[GraphOptimization] marginalization failed (status="
                          << static_cast<int>(frame_result.marginalization_status)
                          << ", lambda=" << frame_result.marginalization_lambda
                          << "); deferred to next frame" << std::endl;
                if (window_.window_size() > window_.max_window_size() + 2) {
                    const NodeId dropped = window_.force_drop_oldest();
                    if (dropped != INVALID_NODE_ID) {
                        frame_result.marginalization_action =
                            SlidingWindow::MarginalizationAction::ForceDropped;
                        std::cerr << "[GraphOptimization] marginalization force-dropped node "
                                  << dropped << " (window growth cap exceeded)" << std::endl;
                    }
                }
            }
        }
        frame_result.finalized = true;
    }

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
        fr.current_node_id = current_id;
        {
            const TipVelocityUpdater tip_velocity_updater;
            bool first_pass = true;
            bool deskew_stopped = false;
            bool frame_valid = true;
            // Snapshots for failure recovery: a failed system may have already
            // applied partial Gauss-Newton updates to every node pose.
            const std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>>
                pre_solve_poses = [&] {
                    std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> poses;
                    poses.reserve(window_.active_nodes().size());
                    for (const auto& n : window_.active_nodes()) poses.push_back(n->pose);
                    return poses;
                }();
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
                    const auto result = solver_.optimize(
                        window_, rung_scale,
                        robust.enable
                            ? std::optional<size_t>(std::max<size_t>(1, robust.iters_per_level))
                            : std::nullopt);
                    fr.solver_status = result.status;
                    fr.converged = result.converged;
                    fr.iterations += result.iterations;
                    fr.error = result.final_error;
                    if (!result.valid()) {
                        // A failed system leaves the estimate unreliable: stop the
                        // remaining ladder / velocity rounds. The pipeline must
                        // discard this frame (see FrameResult::solver_valid).
                        frame_valid = false;
                        break;
                    }
                }
                if (!frame_valid) break;
            }

            if (!frame_valid) {
                // End-of-frame robust bookkeeping still applies to the surviving
                // factors (they lock their last used scale either way).
                window_.finalize_robust();
                // Frame-local rollback contract on solver failure:
                // - restored: poses of the surviving nodes (the failed solve may
                //   have applied partial Gauss-Newton updates to them)
                // - discarded: the failed tip node and every factor incident to it
                // - retained: sparse-chain bookkeeping already committed for older
                //   nodes (prune_point_cloud_binaries ran before the solve; the
                //   chain conversion reflects the committed pre-solve estimates)
                const auto& nodes = window_.active_nodes();
                for (size_t i = 0; i < nodes.size() && i < pre_solve_poses.size(); ++i) {
                    nodes[i]->pose = pre_solve_poses[i];
                }
                window_.remove_node(current_id);
                fr.current_node_id = INVALID_NODE_ID;
                fr.current_pose = initial_pose;
                fr.tip_cloud = nullptr;
                fr.converged = false;
                fr.error = 0.0f;
                fr.tip_registration = registration::RegistrationResult{};
                fr.tip_robust_scale = 0.0f;
                fr.inlier_ratio = 0.0f;
                return fr;
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
                fr.tip_registration.T = fr.current_pose;
                fr.tip_registration.converged = fr.converged;
                fr.tip_registration.iterations = fr.iterations;
                fr.tip_registration.H = lin->H00;
                fr.tip_registration.b = lin->b0;
                fr.tip_registration.error = lin->error;
                // Graph mode does not compute a separate linearization pass for
                // *_raw: these fields mirror the final robust-weighted
                // linearization, which is exactly the LO path's H_raw semantics
                // (robust-weighted, before degenerate regularization / MAP
                // prior; the graph factors run with degenerate regularization
                // disabled). Consumers such as AdaptiveMotionPredictor can
                // therefore treat them like the LO H_raw. A truly unweighted
                // Hessian (robust loss fully disabled) is NOT provided here;
                // compute it on demand from the tip's unary factor with the
                // robust loss disabled if ever needed.
                fr.tip_registration.H_raw = lin->H00;
                fr.tip_registration.b_raw = lin->b0;
                fr.tip_registration.error_raw = lin->error;
                fr.tip_registration.inlier = lin->inlier;
                fr.tip_robust_scale = unary_factor->last_linearization_scale();
            }
        }

        if (opts_.gate.enabled && !opts_.gate.external_decision) {
            const Eigen::Isometry3f d = last_keyframe_pose_.inverse() * fr.current_pose;
            const bool time_hit = opts_.gate.min_time_seconds > 0.0f && last_keyframe_time_ >= 0.0 &&
                                  timestamp - last_keyframe_time_ >= opts_.gate.min_time_seconds;
            const bool inlier_ok = opts_.gate.min_inlier_ratio <= 0.0f ||
                                   fr.inlier_ratio > opts_.gate.min_inlier_ratio;
            const bool is_keyframe = inlier_ok &&
                                     (!has_keyframe_ ||
                                      d.translation().norm() >= opts_.gate.min_translation ||
                                      Eigen::AngleAxisf(d.rotation()).angle() >= opts_.gate.min_rotation ||
                                      time_hit);
            if (is_keyframe) {
                last_keyframe_pose_ = fr.current_pose;
                last_keyframe_time_ = timestamp;
                has_keyframe_ = true;
            }
            finalize_frame(fr, is_keyframe);
            return fr;
        }

        if (!opts_.gate.external_decision) {
            finalize_frame(fr, true);
        }

        return fr;
    }

    SlidingWindow& window() { return window_; }
    const SlidingWindow& window() const { return window_; }

private:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// @brief Frozen robust scale for marginalization: the frame-level ladder's
    ///        floor (final rung) when enabled, else 0% letting factors use fixed default scale
    static float frozen_marg_scale(const GraphSolverParams& solver_params) {
        return solver_params.robust.enable ? solver_params.robust.min_scale : 0.0f;
    }

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
