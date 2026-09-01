#pragma once

#include <algorithm>
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
    explicit SlidingWindow(size_t max_window_size = 5) : max_window_size_(max_window_size) {}

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

    /// @brief Where the Schur-complement prior produced by marginalize_oldest is
    ///        anchored. Newest preserves the original clique-era behavior;
    ///        OldestSurviving follows the dominant chain coupling of the
    ///        marginalized node (sparse-chain topology).
    enum class MarginalizationAnchor { Newest, OldestSurviving };

    /// @brief Marginalize the oldest active node via Schur complement, producing a
    ///        star-shaped unary prior on the newest node. Returns the marginalized
    ///        node id, or INVALID_NODE_ID if the window has not exceeded its max size.
    NodeId marginalize_oldest(const sycl_utils::DeviceQueue& queue,
                              MarginalizationAnchor anchor = MarginalizationAnchor::Newest) {
        if (nodes_.size() <= max_window_size_) return INVALID_NODE_ID;

        auto oldest = nodes_.front();
        NodeId marginalize_id = oldest->id;
        const size_t K = nodes_.size();
        const int m_idx = 0;
        const int r_size = static_cast<int>(6 * (K - 1));

        // Step 1: linearize all factors at the current poses into a dense 6K x 6K system.
        Eigen::MatrixXf H_all = Eigen::MatrixXf::Zero(6 * K, 6 * K);
        Eigen::VectorXf b_all = Eigen::VectorXf::Zero(6 * K);
        std::unordered_map<NodeId, int> id_to_idx;
        for (int i = 0; i < static_cast<int>(K); ++i) id_to_idx[nodes_[i]->id] = i;

        for (auto& f : factors_) {
            f->clear_cache();
            auto lin = f->linearize(queue, f->scale_now(0.0f));
            auto [sid, tid] = f->node_ids();
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

        // Step 2: add existing prior.
        if (prior_.is_valid()) {
            int pi = id_to_idx[prior_.node_ids[0]];
            auto c = prior_.evaluate(nodes_[pi]->pose);
            H_all.block<6, 6>(6 * pi, 6 * pi) += c.H;
            b_all.segment<6>(6 * pi) += c.b;
        }

        // Step 3: Schur complement to eliminate the oldest node (index 0).
        Eigen::Matrix<float, 6, 6> H_mm = H_all.block<6, 6>(0, 0);
        Eigen::Matrix<float, 6, 6> H_mm_reg = H_mm + marginalization_lambda_ * Eigen::Matrix<float, 6, 6>::Identity();
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt_mm(H_mm_reg);
        if (ldlt_mm.info() != Eigen::Success) return INVALID_NODE_ID;  // fallback

        Eigen::MatrixXf H_mr = H_all.block(0, 6, 6, r_size);
        Eigen::Matrix<float, 6, 1> b_m = b_all.head<6>();
        Eigen::MatrixXf H_rr = H_all.block(6, 6, r_size, r_size);
        Eigen::VectorXf b_r = b_all.segment(6, r_size);

        Eigen::MatrixXf H_prior_new = H_rr - H_mr.transpose() * ldlt_mm.solve(H_mr);
        Eigen::VectorXf b_prior_new = b_r - H_mr.transpose() * ldlt_mm.solve(b_m);

        // Step 4: star-shaped prior on the chosen anchor node (diagonal 6x6 block).
        const int anchor_pos =
            (anchor == MarginalizationAnchor::Newest) ? static_cast<int>(K) - 1 : 1;
        MarginalizationPrior new_prior;
        new_prior.node_ids = {nodes_[anchor_pos]->id};
        new_prior.linearization_poses = {nodes_[anchor_pos]->pose};
        const int anchor_reduced = anchor_pos - 1;  // index after dropping nodes_[0]
        new_prior.H_prior = eigen_utils::ensure_symmetric<6>(
            H_prior_new.block<6, 6>(6 * anchor_reduced, 6 * anchor_reduced));
        new_prior.b_prior = b_prior_new.segment<6>(6 * anchor_reduced);
        new_prior.error_constant = 0.0f;

        // Step 5: drop factors/nodes touching the marginalized node, replace prior.
        factors_.erase(std::remove_if(factors_.begin(), factors_.end(),
                                      [&](const auto& f) {
                                          auto [s, t] = f->node_ids();
                                          return s == marginalize_id || t == marginalize_id;
                                      }),
                       factors_.end());
        nodes_.erase(nodes_.begin());
        for (auto& kv : nodes_by_id_) {
            if (kv.first == marginalize_id) {
                nodes_by_id_.erase(kv.first);
                break;
            }
        }
        prior_ = std::move(new_prior);

        (void)m_idx;
        return marginalize_id;
    }

private:
    size_t max_window_size_ = 5;
    float marginalization_lambda_ = 1e-6f;
    NodeId next_id_ = 0;
    std::vector<std::shared_ptr<PoseNode>> nodes_;
    std::vector<std::shared_ptr<GicpFactorBase>> factors_;
    MarginalizationPrior prior_;
    std::unordered_map<NodeId, std::shared_ptr<PoseNode>> nodes_by_id_;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
