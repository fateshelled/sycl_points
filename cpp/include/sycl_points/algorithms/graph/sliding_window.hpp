#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "sycl_points/algorithms/graph/gicp_factor.hpp"
#include "sycl_points/algorithms/graph/marginalization_prior.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/algorithms/graph/relative_pose_factor.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Sliding-window management for the local pose graph.
///
/// Owns the active pose nodes and the factors connecting them, and holds the
/// current marginalization prior. Phase 1 uses only node/factor management;
/// marginalization is added in Phase 2.
class SlidingWindow {
public:
    /// @brief Outcome (failure reason) of a single marginalize_oldest() attempt.
    ///        The pipeline action taken afterwards is reported separately via
    ///        MarginalizationAction, so the reason is never lost.
    enum class MarginalizationStatus {
        NotRequired,          ///< window within max size; nothing done
        Success,              ///< oldest node absorbed into the Markov-blanket prior
        NonFiniteSystem,      ///< factor linearization / Schur output contained NaN or Inf
        DecompositionFailed,  ///< H_mm not usable even with escalated lambda
    };

    /// @brief Pipeline action taken after a marginalization attempt.
    enum class MarginalizationAction {
        None,         ///< no action (NotRequired / Success)
        Deferred,     ///< failure: oldest node kept for a next-frame retry
        ForceDropped, ///< persistent failure: oldest node dropped without a prior
    };

    struct MarginalizationResult {
        MarginalizationStatus status = MarginalizationStatus::NotRequired;
        NodeId marginalized_node = INVALID_NODE_ID;
        /// @brief Last lambda actually attempted for the Schur regularization.
        float lambda_used = 0.0f;
    };

    /// @brief Frozen robust scale marginalization linearizes with. Should be the
    ///        final robust ladder rung's scale (GraphSolverParams::RobustSchedule
    ///        ::min_scale when the frame-level ladder is enabled, else 0 for the
    ///        factor's own fixed default scale), so weights do not change in
    ///        meaning the moment a node leaves the window.
    explicit SlidingWindow(size_t max_window_size = 5, float marginalization_lambda = 1e-6f,
                           float marginalization_scale = 0.0f)
        : max_window_size_(max_window_size), marginalization_lambda_(marginalization_lambda),
          marginalization_scale_(marginalization_scale) {
        if (max_window_size_ == 0 || !std::isfinite(marginalization_lambda_) ||
            marginalization_lambda_ <= 0.0f) {
            throw std::invalid_argument(
                "[SlidingWindow] window size and marginalization lambda must be positive");
        }
        if (!std::isfinite(marginalization_scale_) || marginalization_scale_ < 0.0f) {
            throw std::invalid_argument("[SlidingWindow] marginalization_scale must be finite and >= 0");
        }
    }

    NodeId add_node(const Eigen::Isometry3f& initial_pose, double timestamp,
                    std::shared_ptr<PointCloudShared> cloud = nullptr,
                    std::shared_ptr<knn::KNNBase> knn = nullptr) {
        auto node = std::make_shared<PoseNode>();
        node->id = next_id_++;
        node->timestamp = timestamp;
        node->pose = initial_pose;
        node->linearization_pose = initial_pose;
        node->cloud = std::move(cloud);
        node->knn = std::move(knn);
        node->type = (nodes_.empty()) ? PoseNode::Type::CURRENT : PoseNode::Type::ACTIVE_WINDOW;
        nodes_.push_back(node);
        return node->id;
    }

    void add_factor(std::shared_ptr<GicpFactorBase> factor) { factors_.push_back(std::move(factor)); }

    /// @brief End the robust ladder for all annealing factors (frame end, after
    ///        the ladder has reached its floor): locks each factor's scale at the
    ///        last value used. Scale-free factors are unaffected.
    void finalize_robust() {
        for (auto& f : factors_) {
            f->freeze();
        }
    }

    /// @brief Drop a transient (non-promoted) tip node and every factor that
    ///        touches it. Unlike marginalization no information is preserved:
    ///        the observation lives on only in the returned pose, and the next
    ///        frame re-observes the same region through its fresh tip star.
    void remove_node(NodeId id) {
        factors_.erase(std::remove_if(factors_.begin(), factors_.end(),
                                      [&](const auto& f) {
                                          auto [s, t] = f->node_ids();
                                          return s == id || t == id;
                                      }),
                       factors_.end());
        nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
                                    [&](const auto& n) { return n->id == id; }),
                     nodes_.end());
        nodes_by_id_.erase(id);
    }

    std::shared_ptr<PoseNode> get_node(NodeId id) {
        auto it = nodes_by_id_.find(id);
        if (it != nodes_by_id_.end()) return it->second;
        for (auto& n : nodes_)
            if (n->id == id) {
                nodes_by_id_[id] = n;
                return n;
            }
        return nullptr;
    }

    std::vector<std::shared_ptr<PoseNode>>& active_nodes() { return nodes_; }
    const std::vector<std::shared_ptr<GicpFactorBase>>& factors() const { return factors_; }
    const MarginalizationPrior& prior() const { return prior_; }
    size_t window_size() const { return nodes_.size(); }
    size_t max_window_size() const { return max_window_size_; }
    float marginalization_lambda() const { return marginalization_lambda_; }

    /// @brief Sparse-chain topology maintenance: drop point-cloud binary factors
    ///        that do not touch `keep_tip` (their scan-to-scan information is
    ///        re-expressed by the new tip's fresh star), and make sure the
    ///        adjacent pair (convert_a, convert_b) keeps a chain constraint by
    ///        converting the matching binary into a RelativePoseFactor frozen at
    ///        the current estimates (a relative factor already present is kept
    ///        as-is). Idempotent w.r.t. repeated calls with the same pair.
    void prune_point_cloud_binaries(NodeId keep_tip, NodeId convert_a, NodeId convert_b,
                                    const RelativePoseParams& rel_params = RelativePoseParams()) {
        bool chain_present = false;
        std::vector<std::shared_ptr<GicpFactorBase>> kept;
        kept.reserve(factors_.size());
        for (auto& f : factors_) {
            const auto [s, t] = f->node_ids();
            if (!f->is_point_cloud_binary()) {
                if (f->node_ids() == std::make_pair(convert_a, convert_b)) chain_present = true;
                kept.push_back(f);
                continue;
            }
            if (s == keep_tip) {
                kept.push_back(f);  // fresh star edge of the current tip
                continue;
            }
            if (!chain_present && (s == convert_a || s == convert_b) &&
                (t == convert_a || t == convert_b)) {
                auto na = get_node(convert_a);
                auto nb = get_node(convert_b);
                const Eigen::Isometry3f G = na->pose.inverse() * nb->pose;
                kept.push_back(std::make_shared<RelativePoseFactor>(convert_a, na, convert_b, nb,
                                                                    G, rel_params));
                chain_present = true;
                continue;
            }
            // stale point-cloud binary -> dropped
        }
        if (!chain_present && convert_a != INVALID_NODE_ID && convert_b != INVALID_NODE_ID) {
            auto na = get_node(convert_a);
            auto nb = get_node(convert_b);
            if (na && nb) {
                const Eigen::Isometry3f G = na->pose.inverse() * nb->pose;
                kept.push_back(std::make_shared<RelativePoseFactor>(convert_a, na, convert_b, nb,
                                                                    G, rel_params));
            }
        }
        factors_ = std::move(kept);
    }

    /// @brief Marginalize the oldest active node via Schur complement, producing a
    ///        dense prior over its Markov blanket. Only factors touching the removed
    ///        node are absorbed; surviving factors remain represented exactly once.
    ///
    /// Marginalization linearizes with the frozen robust scale of the final ladder
    /// rung, so the robust measurement model (whose weights the optimizer actually
    /// adopted) is what gets baked into the prior. Numerical stability is provided
    /// by lambda escalation on H_mm, never by dropping robust weights:
    /// - non-finite H/b -> NonFiniteSystem immediately (regularization cannot fix NaN)
    /// - LDLT info failure or poor eigenvalue conditioning
    ///   (lambda_min < kMinConditionRatio * lambda_max) -> retry with lambda *= 10
    MarginalizationResult marginalize_oldest(const sycl_utils::DeviceQueue& queue) {
        MarginalizationResult result;
        if (nodes_.size() <= max_window_size_) {
            result.status = MarginalizationStatus::NotRequired;
            return result;
        }

        auto oldest = nodes_.front();
        NodeId marginalize_id = oldest->id;
        std::vector<NodeId> local_ids = {marginalize_id};
        auto add_local_id = [&](NodeId id) {
            if (id != INVALID_NODE_ID &&
                std::find(local_ids.begin(), local_ids.end(), id) == local_ids.end()) {
                local_ids.push_back(id);
            }
        };
        for (const NodeId id : prior_.node_ids) add_local_id(id);
        for (const auto& f : factors_) {
            const auto [sid, tid] = f->node_ids();
            if (sid == marginalize_id || tid == marginalize_id) {
                add_local_id(sid);
                add_local_id(tid);
            }
        }

        const size_t K = local_ids.size();
        Eigen::MatrixXf H_all = Eigen::MatrixXf::Zero(6 * K, 6 * K);
        Eigen::VectorXf b_all = Eigen::VectorXf::Zero(6 * K);
        std::unordered_map<NodeId, int> id_to_idx;
        for (int i = 0; i < static_cast<int>(K); ++i) id_to_idx[local_ids[i]] = i;

        for (auto& f : factors_) {
            const auto [sid, tid] = f->node_ids();
            if (sid != marginalize_id && tid != marginalize_id) continue;
            f->clear_cache();
            // Keep the robust weights the optimizer actually adopted. Each factor is
            // re-linearized with its own frozen scale (scale_now falls back to the
            // window's marginalization_scale_ only for factors still annealing), so a
            // ladder that ended early (e.g. redeskew failure) bakes in the scale the
            // factor last actually used, not the nominal floor.
            auto lin = f->linearize(queue, f->scale_now(marginalization_scale_));
            int si = id_to_idx[sid];
            H_all.block<6, 6>(6 * si, 6 * si) += lin.H00;
            b_all.segment<6>(6 * si) += lin.b0;
            if (tid != INVALID_NODE_ID) {
                int ti = id_to_idx[tid];
                H_all.block<6, 6>(6 * ti, 6 * ti) += lin.H11;
                H_all.block<6, 6>(6 * si, 6 * ti) += lin.H01;
                H_all.block<6, 6>(6 * ti, 6 * si) += lin.H01.transpose();
                b_all.segment<6>(6 * ti) += lin.b1;
            }
        }

        // Step 2: carry the existing prior exactly once.
        if (prior_.is_valid()) {
            std::vector<Eigen::Isometry3f> poses;
            poses.reserve(prior_.node_ids.size());
            for (const NodeId id : prior_.node_ids) poses.push_back(get_node(id)->pose);
            const auto c = prior_.evaluate(poses);
            for (size_t i = 0; i < prior_.node_ids.size(); ++i) {
                const int pi = id_to_idx[prior_.node_ids[i]];
                b_all.segment<6>(6 * pi) += c.b.segment<6>(6 * i);
                for (size_t j = 0; j < prior_.node_ids.size(); ++j) {
                    const int pj = id_to_idx[prior_.node_ids[j]];
                    H_all.block<6, 6>(6 * pi, 6 * pj) += c.H.block<6, 6>(6 * i, 6 * j);
                }
            }
        }

        // Step 3: Schur complement to eliminate the oldest node (index 0).
        // NaN/Inf cannot be fixed by regularization -> single deferred-retry status.
        if (!H_all.allFinite() || !b_all.allFinite()) {
            if (verbose()) {
                std::cerr << "[SlidingWindow] marginalization linearization is non-finite"
                          << " (node " << marginalize_id << ")" << std::endl;
            }
            result.status = MarginalizationStatus::NonFiniteSystem;
            return result;
        }
        const Eigen::Matrix<float, 6, 6> H_mm = H_all.block<6, 6>(0, 0);

        // Eigen LDLT reports Success even on numerically singular input, so gate
        // on the eigenvalue conditioning as well and escalate lambda until the
        // Schur complement reads a stable H_mm (per-frame retry, never robust
        // weight removal).
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt_mm;
        float lambda_used = marginalization_lambda_;
        bool usable = false;
        for (int escalation = 0; escalation <= kMaxLambdaEscalations; ++escalation) {
            const float attempted = lambda_used;
            const Eigen::Matrix<float, 6, 6> H_mm_reg =
                H_mm + attempted * Eigen::Matrix<float, 6, 6>::Identity();
            if (!H_mm_reg.allFinite()) {
                result.status = MarginalizationStatus::NonFiniteSystem;
                return result;
            }
            bool conditioned = false;
            if (const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eig(H_mm_reg); eig.info() == Eigen::Success) {
                const auto ev = eig.eigenvalues();
                // H is PSD up to noise; a non-positive or badly conditioned span is
                // treated the same as a decomposition failure.
                const float ev_max = ev.maxCoeff();
                const float ev_min = ev.minCoeff();
                conditioned = ev_max > 0.0f && ev_min >= kMinConditionRatio * ev_max;
            }
            ldlt_mm.compute(H_mm_reg);
            if (ldlt_mm.info() == Eigen::Success && conditioned) {
                usable = true;
                break;
            }
            if (verbose()) {
                std::cerr << "[SlidingWindow] marginalization lambda escalation"
                          << " (escalation=" << (escalation + 1) << "/" << kMaxLambdaEscalations
                          << ", lambda=" << attempted
                          << (ldlt_mm.info() == Eigen::Success ? ", poor conditioning" : ", LDLT failed")
                          << ")" << std::endl;
            }
            // Do not escalate past the final attempt: lambda_used must stay equal to
            // the value actually tried so the failure log / MarginalizationResult::
            // lambda_used report the last attempted lambda, not the next (unused) one.
            if (escalation < kMaxLambdaEscalations) {
                lambda_used *= 10.0f;
            }
        }
        if (!usable) {
            if (verbose()) {
                std::cerr << "[SlidingWindow] marginalization decomposition failed"
                          << " (node " << marginalize_id << ", lambda up to " << lambda_used << ")"
                          << std::endl;
            }
            result.status = MarginalizationStatus::DecompositionFailed;
            return result;
        }

        const int r_size = static_cast<int>(6 * (K - 1));
        Eigen::MatrixXf H_mr = H_all.block(0, 6, 6, r_size);
        Eigen::Matrix<float, 6, 1> b_m = b_all.head<6>();
        Eigen::MatrixXf H_rr = H_all.block(6, 6, r_size, r_size);
        Eigen::VectorXf b_r = b_all.segment(6, r_size);

        Eigen::MatrixXf H_prior_new = H_rr - H_mr.transpose() * ldlt_mm.solve(H_mr);
        Eigen::VectorXf b_prior_new = b_r - H_mr.transpose() * ldlt_mm.solve(b_m);
        // Finite H_mm/H_mr do not guarantee a finite Schur complement on every
        // build; mirror GraphSolver's allFinite gate (NON_FINITE_SYSTEM).
        if (!H_prior_new.allFinite() || !b_prior_new.allFinite()) {
            if (verbose()) {
                std::cerr << "[SlidingWindow] marginalization produced a non-finite"
                          << " prior (node " << marginalize_id << ")" << std::endl;
            }
            result.status = MarginalizationStatus::NonFiniteSystem;
            return result;
        }

        // Step 4: retain the complete reduced system over the Markov blanket.
        MarginalizationPrior new_prior;
        new_prior.node_ids.assign(local_ids.begin() + 1, local_ids.end());
        for (const NodeId id : new_prior.node_ids) {
            new_prior.linearization_poses.push_back(get_node(id)->pose);
        }
        new_prior.H_prior = 0.5f * (H_prior_new + H_prior_new.transpose());
        new_prior.b_prior = b_prior_new;
        new_prior.error_constant = 0.0f;

        // Step 5: drop factors/nodes touching the marginalized node, replace prior.
        factors_.erase(std::remove_if(factors_.begin(), factors_.end(),
                                      [&](const auto& f) {
                                          auto [s, t] = f->node_ids();
                                          return s == marginalize_id || t == marginalize_id;
                                      }),
                       factors_.end());
        nodes_.erase(nodes_.begin());
        nodes_by_id_.erase(marginalize_id);
        prior_ = std::move(new_prior);

        result.status = MarginalizationStatus::Success;
        result.marginalized_node = marginalize_id;
        result.lambda_used = lambda_used;
        return result;
    }

    /// @brief Degraded fallback for persistent marginalization failure: drop the
    ///        oldest node and every factor touching it. No prior is built (that
    ///        information is lost), but the window stays bounded and the pipeline
    ///        keeps running. Also purges the node from the prior when it is
    ///        referenced there.
    /// @return the dropped NodeId, or INVALID_NODE_ID when the window is empty.
    NodeId force_drop_oldest() {
        if (nodes_.empty()) return INVALID_NODE_ID;
        const NodeId id = nodes_.front()->id;
        const bool in_prior =
            std::find(prior_.node_ids.begin(), prior_.node_ids.end(), id) != prior_.node_ids.end();
        remove_node(id);
        if (in_prior) prior_ = MarginalizationPrior{};
        return id;
    }

private:
    static constexpr int kMaxLambdaEscalations = 3;     ///< lambda *= 10 retries per frame
    static constexpr float kMinConditionRatio = 1e-6f;  ///< required lambda_min/lambda_max of H_mm_reg

    /// @brief Verbose diagnostics gate (escalation / failure logs). Enabled with
    ///        the SYCL_POINTS_VERBOSE environment variable.
    static bool verbose() { return std::getenv("SYCL_POINTS_VERBOSE") != nullptr; }

    size_t max_window_size_ = 5;
    float marginalization_lambda_ = 1e-6f;
    float marginalization_scale_ = 0.0f;
    NodeId next_id_ = 0;
    std::vector<std::shared_ptr<PoseNode>> nodes_;
    std::vector<std::shared_ptr<GicpFactorBase>> factors_;
    MarginalizationPrior prior_;
    std::unordered_map<NodeId, std::shared_ptr<PoseNode>> nodes_by_id_;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
