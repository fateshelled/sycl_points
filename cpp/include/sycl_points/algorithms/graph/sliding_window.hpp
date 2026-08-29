#pragma once

#include <unordered_map>
#include <vector>

#include "sycl_points/algorithms/graph/gicp_factor.hpp"
#include "sycl_points/algorithms/graph/marginalization_prior.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"

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

    /// @brief Marginalize the oldest active node via Schur complement, producing a
    ///        star-shaped unary prior on the newest node. Returns the marginalized
    ///        node id, or INVALID_NODE_ID if the window has not exceeded its max size.
    NodeId marginalize_oldest(const sycl_utils::DeviceQueue& queue) {
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
            auto lin = f->linearize(queue);
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

        // Step 4: star-shaped prior on the newest remaining node (diagonal 6x6 block).
        MarginalizationPrior new_prior;
        new_prior.node_ids = {nodes_.back()->id};
        new_prior.linearization_poses = {nodes_.back()->pose};
        const int newest_reduced = static_cast<int>(K) - 2;
        new_prior.H_prior = eigen_utils::ensure_symmetric<6>(
            H_prior_new.block<6, 6>(6 * newest_reduced, 6 * newest_reduced));
        new_prior.b_prior = b_prior_new.segment<6>(6 * newest_reduced);
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
