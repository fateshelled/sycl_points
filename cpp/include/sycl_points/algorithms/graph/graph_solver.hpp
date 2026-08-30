#pragma once

#include <algorithm>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/algorithms/graph/sliding_window.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Gauss-Newton solver parameters for the local pose graph.
struct GraphSolverParams {
    size_t max_iterations = 10;
    float convergence_rotation = 1e-4f;      // [rad]
    float convergence_translation = 1e-4f;   // [m]
    float relinearize_rotation_thresh = 0.02f;
    float relinearize_translation_thresh = 0.05f;
    float marginalization_lambda = 1e-6f;
};

/// @brief Gauss-Newton solver over the sliding-window pose graph.
class GraphSolver {
public:
    struct Result {
        bool converged = false;
        size_t iterations = 0;
        float final_error = 0.0f;
    };

    GraphSolver(const sycl_utils::DeviceQueue& queue,
                const GraphSolverParams& params = GraphSolverParams())
        : queue_(queue), params_(params) {}

    Result optimize(SlidingWindow& window) {
        Result result;
        for (size_t iter = 0; iter < params_.max_iterations; ++iter) {
            auto sys = assemble(window);

            Eigen::LDLT<Eigen::MatrixXf> ldlt(sys.H);
            if (ldlt.info() != Eigen::Success) {
                Eigen::MatrixXf H_reg =
                    sys.H + params_.marginalization_lambda *
                                Eigen::MatrixXf::Identity(sys.H.rows(), sys.H.cols());
                ldlt.compute(H_reg);
                if (ldlt.info() != Eigen::Success) break;
            }
            Eigen::VectorXf delta = ldlt.solve(-sys.b);

            bool converged = true;
            for (size_t i = 0; i < sys.node_ids.size(); ++i) {
                Eigen::Matrix<float, 6, 1> d = delta.segment<6>(6 * i);
                if (d.head<3>().norm() > params_.convergence_rotation ||
                    d.tail<3>().norm() > params_.convergence_translation)
                    converged = false;
            }

            for (size_t i = 0; i < sys.node_ids.size(); ++i) {
                auto node = window.get_node(sys.node_ids[i]);
                if (!node) continue;
                const Eigen::Matrix<float, 6, 1> d = delta.segment<6>(6 * i);
                node->pose = Eigen::Isometry3f(node->pose.matrix() * eigen_utils::lie::se3_exp(d));
            }

            result.final_error = sys.error;
            result.iterations = iter + 1;
            if (converged) {
                result.converged = true;
                break;
            }
        }
        return result;
    }

private:
    struct LinearizedSystem {
        Eigen::MatrixXf H;
        Eigen::VectorXf b;
        float error = 0.0f;
        std::vector<NodeId> node_ids;
    };

    LinearizedSystem assemble(SlidingWindow& window) {
        auto& nodes = window.active_nodes();
        const size_t K = nodes.size();
        LinearizedSystem sys;
        sys.H = Eigen::MatrixXf::Zero(6 * K, 6 * K);
        sys.b = Eigen::VectorXf::Zero(6 * K);
        std::unordered_map<NodeId, int> idx;
        for (int i = 0; i < static_cast<int>(K); ++i) {
            sys.node_ids.push_back(nodes[i]->id);
            idx[nodes[i]->id] = i;
        }

        for (auto& factor : window.factors()) {
            auto lin = factor->get_linearization(
                queue_, params_.relinearize_rotation_thresh, params_.relinearize_translation_thresh);
            auto [sid, tid] = factor->node_ids();
            int si = idx.at(sid);
            sys.H.block<6, 6>(6 * si, 6 * si) += lin.H00;
            sys.b.segment<6>(6 * si) += lin.b0;
            sys.error += lin.error;
            if (tid != INVALID_NODE_ID && idx.count(tid)) {
                int ti = idx.at(tid);
                sys.H.block<6, 6>(6 * ti, 6 * ti) += lin.H11;
                sys.H.block<6, 6>(6 * si, 6 * ti) += lin.H01;
                sys.H.block<6, 6>(6 * ti, 6 * si) += lin.H01.transpose();
                sys.b.segment<6>(6 * ti) += lin.b1;
            }
        }

        const auto& prior = window.prior();
        if (prior.is_valid()) {
            auto it = std::find(sys.node_ids.begin(), sys.node_ids.end(), prior.node_ids[0]);
            if (it != sys.node_ids.end()) {
                int pi = static_cast<int>(std::distance(sys.node_ids.begin(), it));
                auto c = prior.evaluate(nodes[pi]->pose);
                sys.H.block<6, 6>(6 * pi, 6 * pi) += c.H;
                sys.b.segment<6>(6 * pi) += c.b;
                sys.error += c.error;
            }
        }
        return sys;
    }

    sycl_utils::DeviceQueue queue_;
    GraphSolverParams params_;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
