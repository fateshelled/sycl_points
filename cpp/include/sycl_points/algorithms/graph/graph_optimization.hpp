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
    GraphOptimization(const sycl_utils::DeviceQueue& queue,
                      const GraphSolverParams& solver_params = GraphSolverParams(),
                      size_t max_window_size = 5)
        : queue_(queue), solver_(queue_, solver_params), window_(max_window_size) {}

    struct FrameResult {
        Eigen::Isometry3f current_pose;
        bool converged = false;
        size_t iterations = 0;
        float error = 0.0f;
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

        // 4. Marginalize the oldest node once the window exceeds its max size.
        if (window_.window_size() > window_.max_window_size()) {
            window_.marginalize_oldest(queue_);
        }

        // 5. Results.
        auto node = window_.get_node(current_id);
        FrameResult fr;
        fr.current_pose = node ? node->pose : initial_pose;
        fr.converged = result.converged;
        fr.iterations = result.iterations;
        fr.error = result.final_error;
        return fr;
    }

    SlidingWindow& window() { return window_; }
    const SlidingWindow& window() const { return window_; }

private:
    sycl_utils::DeviceQueue queue_;
    GraphSolver solver_;
    SlidingWindow window_;

    std::shared_ptr<const PointCloudShared> submap_;
    std::shared_ptr<const knn::KNNBase> submap_knn_;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
