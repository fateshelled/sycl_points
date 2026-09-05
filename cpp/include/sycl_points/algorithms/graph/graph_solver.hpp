#pragma once

#include <algorithm>
#include <cmath>
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

    /// @brief Graduated non-convexity (robust scale ladder) for the per-frame
    ///        tip factors. Disabled by default (existing behavior preserved).
    struct RobustSchedule {
        bool enable = false;
        float init_scale = 10.0f;    ///< starting (convexified) robust scale
        float min_scale = 1.25f;     ///< ladder floor; also the frozen scale of old factors
        size_t levels = 4;           ///< number of rungs
        size_t iters_per_level = 2;  ///< GN iterations spent on each rung
        /// @brief Re-linearize (with KNN) on every rung change during annealing so
        ///        the ladder is actually applied even when the pose drift stays under
        ///        the relinearization threshold. Costs ~levels full tip linearizations
        ///        per frame (== the align-path ladder); recommended when enable=true.
        bool relinearize_per_rung = false;
    };
    RobustSchedule robust;
};

/// @brief Robust ladder scale for GN iteration `iter` under schedule r.
///        Returns 0 when disabled (=> factors use their configured default),
///        otherwise geometrically descends init_scale -> min_scale across
///        `levels` rungs, holding each rung for `iters_per_level` iterations.
inline float robust_ladder_scale(const GraphSolverParams::RobustSchedule& r, size_t iter) {
    if (!r.enable || r.levels == 0 || r.min_scale <= 0.0f || r.init_scale <= 0.0f) {
        return 0.0f;
    }
    const size_t per = std::max<size_t>(1, r.iters_per_level);
    const size_t level = std::min(iter / per, r.levels - 1);
    const float alpha =
        r.levels == 1
            ? 0.0f
            : std::pow(r.min_scale / r.init_scale, 1.0f / static_cast<float>(r.levels - 1));
    return std::max(r.min_scale, r.init_scale * std::pow(alpha, static_cast<float>(level)));
}

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

    const GraphSolverParams& params() const { return params_; }

    Result optimize(SlidingWindow& window) {
        Result result;
        for (size_t iter = 0; iter < params_.max_iterations; ++iter) {
            auto sys = assemble(window, robust_ladder_scale(params_.robust, iter));

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
            // With an active ladder, small steps alone must not stop the loop:
            // convergence is only granted once the schedule reached its floor
            // (mirrors align-path RobustAligner running every level).
            const size_t ladder_iters =
                std::max<size_t>(1, params_.robust.levels) *
                std::max<size_t>(1, params_.robust.iters_per_level);
            const bool ladder_done =
                !params_.robust.enable || (iter + 1) >= ladder_iters;
            if (converged && ladder_done) {
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

    LinearizedSystem assemble(SlidingWindow& window, float ladder_scale) {
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
                queue_, params_.relinearize_rotation_thresh, params_.relinearize_translation_thresh,
                ladder_scale);
            auto [sid, tid] = factor->node_ids();
            int si = idx.at(sid);
            sys.H.block<6, 6>(6 * si, 6 * si) += lin.H00;
            sys.error += lin.error;
            bool has_target = tid != INVALID_NODE_ID && idx.count(tid);
            if (has_target) {
                int ti = idx.at(tid);
                sys.H.block<6, 6>(6 * ti, 6 * ti) += lin.H11;
                sys.H.block<6, 6>(6 * si, 6 * ti) += lin.H01;
                sys.H.block<6, 6>(6 * ti, 6 * si) += lin.H01.transpose();
            }

            // When a factor reuses a cached (delayed) linearization, b0/b1 are stale
            // at the current poses. Keep the Jacobians frozen but refresh the gradient
            // with the linear-model correction g = b + H * [ds; dt], where ds/dt is the
            // pose change since the linearization point. Without this, stale gradients
            // make Gauss-Newton stall instead of converging.
            Eigen::Matrix<float, 6, 1> g0 = lin.b0;
            Eigen::Matrix<float, 6, 1> g1 = lin.b1;
            const Eigen::Isometry3f& src_lin = lin.source_linearization_pose;
            const Eigen::Matrix<float, 6, 1> ds = eigen_utils::lie::se3_log(
                src_lin.inverse() * window.get_node(sid)->pose);
            if (ds.norm() > 0.0f) {
                g0 += lin.H00 * ds;
                if (has_target) {
                    g1 += lin.H01.transpose() * ds;
                }
            }
            if (has_target) {
                int ti = idx.at(tid);
                const Eigen::Matrix<float, 6, 1> dt = eigen_utils::lie::se3_log(
                    lin.target_linearization_pose.inverse() * window.get_node(tid)->pose);
                if (dt.norm() > 0.0f) {
                    g0 += lin.H01 * dt;
                    g1 += lin.H11 * dt;
                }
                sys.b.segment<6>(6 * ti) += g1;
            }
            sys.b.segment<6>(6 * si) += g0;
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
