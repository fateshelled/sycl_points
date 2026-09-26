#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>

#include <Eigen/Geometry>

#include "sycl_points/algorithms/graph/gicp_factor.hpp"
#include "sycl_points/algorithms/graph/bootstrap_state_prior_factor.hpp"
#include "sycl_points/algorithms/graph/graph_solver.hpp"
#include "sycl_points/algorithms/graph/imu_preintegration_factor.hpp"
#include "sycl_points/algorithms/graph/nav_state_prior_factor.hpp"
#include "sycl_points/algorithms/graph/sliding_window.hpp"
#include "sycl_points/algorithms/graph/velocity_update.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
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

    /// @brief Tight-coupling IMU edge context. The preintegration accumulates
    ///        from the source keyframe node and is added as a full 15-DOF binary
    ///        factor against the current tip. Scale-free, host-only: it never
    ///        participates in the robust ladder or LiDAR observability.
    struct ImuEdgeContext {
        bool enable = false;
        NodeId source_id = INVALID_NODE_ID;
        std::shared_ptr<const imu::IMUPreintegration> preintegration = nullptr;
        Eigen::Isometry3f T_imu_to_lidar = Eigen::Isometry3f::Identity();
        Eigen::Vector3f gravity = Eigen::Vector3f(0.0f, 0.0f, -9.80665f);
        /// @brief Initial navigation state for the new tip node. The unary pose
        ///        factor alone cannot observe velocity/bias, so the tip is seeded
        ///        from the source keyframe (or the pipeline's current estimate).
        Eigen::Vector3f tip_velocity = Eigen::Vector3f::Zero();
        Eigen::Vector3f tip_accel_bias = Eigen::Vector3f::Zero();
        Eigen::Vector3f tip_gyro_bias = Eigen::Vector3f::Zero();
        /// @brief Add a weak navigation-state prior (velocity/bias only) to the
        ///        tip. Fixes the preintegration gauge so single-edge frames stay
        ///        solvable; the sigmas are loose by design.
        bool add_nav_prior = false;
        float nav_prior_sigma_velocity = 1.0f;     ///< [m/s]
        float nav_prior_sigma_accel_bias = 0.5f;   ///< [m/s^2]
        float nav_prior_sigma_gyro_bias = 0.1f;    ///< [rad/s]
        float root_prior_sigma_pose = 1e-4f;
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
        NodeState current_state;
        bool has_current_state = false;
        bool converged = false;
        size_t iterations = 0;
        float error = 0.0f;
        bool keyframe = true;
        /// @brief Solver outcome of the last optimize() pass for this frame.
        ///        MAX_ITERATIONS (default) means the loop simply ran out of
        ///        iterations; an invalid status means the estimate is unusable.
        GraphSolver::Status solver_status = GraphSolver::Status::MAX_ITERATIONS;
        /// @brief LiDAR-only tip observability from the final solver pass.
        GraphSolver::ObservabilityDiagnostics observability;
        size_t solver_inner_iterations = 0;
        size_t solver_accepted_steps = 0;
        size_t solver_rejected_steps = 0;
        float solver_final_lambda = 0.0f;
        float solver_tip_translation_step = 0.0f;
        float solver_tip_rotation_step = 0.0f;
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
        /// @brief Keyframe/submap lifecycle freeze payload (freeze_ready only on a
        ///        Successful eviction): the oldest node left the active window this
        ///        frame, and the pipeline must insert its scan into the fixed
        ///        submap at its final optimized pose. A scan is thereby never
        ///        both an active PoseNode and fixed map geometry
        ///        (ActiveKeyframeScans ∩ FixedSubmapScans = empty). Force-dropped
        ///        nodes are NOT frozen: emergency eviction drops the node without
        ///        a prior and the map simply misses that scan.
        bool freeze_ready = false;
        Eigen::Isometry3f frozen_pose = Eigen::Isometry3f::Identity();
        std::shared_ptr<PointCloudShared> frozen_cloud = nullptr;
        double frozen_timestamp = 0.0;
    };

    struct Checkpoint {
        SlidingWindow::Checkpoint window;
        Eigen::Isometry3f last_keyframe_pose = Eigen::Isometry3f::Identity();
        double last_keyframe_time = -1.0;
        bool has_keyframe = false;
    };

    Checkpoint checkpoint() const {
        return {window_.checkpoint(), last_keyframe_pose_, last_keyframe_time_, has_keyframe_};
    }

    void restore(const Checkpoint& checkpoint) {
        window_.restore(checkpoint.window);
        last_keyframe_pose_ = checkpoint.last_keyframe_pose;
        last_keyframe_time_ = checkpoint.last_keyframe_time;
        has_keyframe_ = checkpoint.has_keyframe;
    }

    /// @brief Apply the authoritative keyframe decision for the current tip.
    ///
    /// `keep` decides only graph retention: whether the tip stays an active
    /// PoseNode. Submap insertion is a SEPARATE event: on eviction the oldest
    /// node is marginalized and its final optimized pose + cloud are reported
    /// in the FrameResult freeze payload for the pipeline to insert into the
    /// fixed submap.
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
            if (m.status == SlidingWindow::MarginalizationStatus::Success) {
                frame_result.marginalization_action = SlidingWindow::MarginalizationAction::None;
                // Graph/map lifecycle transition: the state is gone from the
                // graph, its geometry becomes fixed map data. Inserted at the
                // optimized pose the node had exactly at eviction time.
                frame_result.freeze_ready = true;
                frame_result.frozen_pose = m.evicted_pose;
                frame_result.frozen_cloud = m.evicted_cloud;
                frame_result.frozen_timestamp = m.evicted_timestamp;
            } else if (m.status == SlidingWindow::MarginalizationStatus::NotRequired) {
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
        return this->process_frame(source_cloud, submap_cloud, submap_knn, source_knn, initial_pose, timestamp,
                                   reg_params, vu, ImuEdgeContext());
    }

    FrameResult process_frame(std::shared_ptr<PointCloudShared> source_cloud,
                              std::shared_ptr<const PointCloudShared> submap_cloud,
                              std::shared_ptr<const knn::KNNBase> submap_knn,
                              std::shared_ptr<knn::KNNBase> source_knn,
                              const Eigen::Isometry3f& initial_pose, double timestamp,
                              const registration::RegistrationParams& reg_params,
                               const VelocityUpdateContext& vu,
                               const ImuEdgeContext& imu) {
        const auto checkpoint = this->checkpoint();
        try {
            return process_frame_impl(std::move(source_cloud), std::move(submap_cloud),
                                      std::move(submap_knn), std::move(source_knn), initial_pose,
                                      timestamp, reg_params, vu, imu, checkpoint);
        } catch (...) {
            this->restore(checkpoint);
            throw;
        }
    }

    NodeId add_bootstrap_node(const Eigen::Isometry3f& pose, double timestamp,
                              std::shared_ptr<PointCloudShared> cloud,
                              std::shared_ptr<knn::KNNBase> knn,
                              const NodeState& state,
                              const ImuEdgeContext& prior) {
        const NodeId id = window_.add_node(pose, timestamp, std::move(cloud), std::move(knn),
                                           state.velocity, state.accel_bias, state.gyro_bias);
        auto node = window_.get_node(id);
        NodeState reference = state;
        reference.pose = pose;
        window_.add_factor(std::make_shared<BootstrapStatePriorFactor>(
            node, reference, prior.root_prior_sigma_pose, prior.nav_prior_sigma_velocity,
            prior.nav_prior_sigma_accel_bias, prior.nav_prior_sigma_gyro_bias));
        last_keyframe_pose_ = pose;
        last_keyframe_time_ = timestamp;
        has_keyframe_ = true;
        return id;
    }

private:
    FrameResult process_frame_impl(std::shared_ptr<PointCloudShared> source_cloud,
                                   std::shared_ptr<const PointCloudShared> submap_cloud,
                                   std::shared_ptr<const knn::KNNBase> submap_knn,
                                   std::shared_ptr<knn::KNNBase> source_knn,
                                   const Eigen::Isometry3f& initial_pose, double timestamp,
                                   const registration::RegistrationParams& reg_params,
                                   const VelocityUpdateContext& vu, const ImuEdgeContext& imu,
                                   const Checkpoint& checkpoint) {

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
        current_node->velocity = imu.tip_velocity;
        current_node->accel_bias = imu.tip_accel_bias;
        current_node->gyro_bias = imu.tip_gyro_bias;
        current_node->linearization_velocity = imu.tip_velocity;
        current_node->linearization_accel_bias = imu.tip_accel_bias;
        current_node->linearization_gyro_bias = imu.tip_gyro_bias;
        auto unary_factor = std::make_shared<UnaryGicpFactor>(
            queue_, current_id, current_node, submap_cloud, submap_knn, reg_params);
        window_.add_factor(unary_factor);

        // 2b. Binary GICP factors: current <-> each existing active window node.
        for (auto& node : window_.active_nodes()) {
            if (node->id == current_id) continue;
            if (!node->knn) continue;  // need a kNN on the target node's cloud
            window_.add_factor(std::make_shared<BinaryGicpFactor>(
                queue_, current_id, current_node, node->id, node, reg_params));
        }

        // 2c. IMU preintegration edge (tight coupling): full 15-DOF binary factor
        //     between the source keyframe and the current tip. Host-only and
        //     scale-free, so it is deliberately excluded from the robust ladder
        //     and from LiDAR observability.
        if (imu.enable && imu.preintegration && imu.source_id != INVALID_NODE_ID) {
            auto src_node = window_.get_node(imu.source_id);
            if (src_node && src_node->id != current_id && imu.preintegration->get_dt_total() > 0.0) {
                window_.add_factor(std::make_shared<ImuPreintegrationFactor>(
                    src_node, current_node, *imu.preintegration, imu.T_imu_to_lidar, imu.gravity));
            }
        }

        // 2d. Weak navigation-state prior on the tip (velocity/bias only). This
        //     fixes the preintegration gauge (a single edge leaves a
        //     constant-velocity/bias trade-off) without constraining the
        //     LiDAR-observed geometry. Host-only and scale-free.
        if (imu.add_nav_prior) {
            NodeState ref;
            ref.pose = current_node->pose;
            ref.velocity = imu.tip_velocity;
            ref.accel_bias = imu.tip_accel_bias;
            ref.gyro_bias = imu.tip_gyro_bias;
            window_.add_factor(std::make_shared<NavStatePriorFactor>(
                current_node, ref, imu.nav_prior_sigma_velocity, imu.nav_prior_sigma_accel_bias,
                imu.nav_prior_sigma_gyro_bias));
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
                    fr.observability = result.observability;
                    fr.solver_inner_iterations += result.inner_iterations;
                    fr.solver_accepted_steps += result.accepted_steps;
                    fr.solver_rejected_steps += result.rejected_steps;
                    fr.solver_final_lambda = result.final_lambda;
                    fr.solver_tip_translation_step = result.final_tip_translation_step;
                    fr.solver_tip_rotation_step = result.final_tip_rotation_step;
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
                this->restore(checkpoint);
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
        if (cur) {
            fr.current_state = cur->state();
            fr.has_current_state = true;
        }
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

public:
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

    Eigen::Isometry3f last_keyframe_pose_ = Eigen::Isometry3f::Identity();
    double last_keyframe_time_ = -1.0;
    bool has_keyframe_ = false;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
