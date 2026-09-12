#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/algorithms/graph/sliding_window.hpp"
#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Nonlinear solver parameters for the local pose graph.
struct GraphSolverParams {
    registration::OptimizationMethod optimization_method =
        registration::OptimizationMethod::GAUSS_NEWTON;
    registration::RegistrationOptimizationParams::LevenbergMarquardt lm;
    size_t max_iterations = 10;
    float convergence_rotation = 1e-4f;      // [rad]
    float convergence_translation = 1e-4f;   // [m]
    float relinearize_rotation_thresh = 0.02f;
    float relinearize_translation_thresh = 0.05f;
    float solver_damping_lambda = 1e-6f;
    float marginalization_lambda = 1e-6f;

    /// @brief Step stability bounds for one Gauss-Newton direction. LDLT reports
    ///        Success even on singular / ill-conditioned systems, so a finite but
    ///        huge step would silently move every pose; the solver rejects such a
    ///        step and retries with escalated damping. Bounds are deliberately
    ///        generous (normal odometry steps are well below them) and are not
    ///        exposed as ROS parameters.
    float max_step_translation = 10.0f;  ///< [m] per-iteration translation bound
    float max_step_rotation = 1.0f;      ///< [rad] per-iteration rotation bound

    /// @brief Print per-iteration solver logs (mirrors RegistrationFactorParams::verbose
    ///        on the LO align path; wired from graph/factor/verbose).
    bool verbose = false;

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
        bool relinearize_per_rung = true;
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

/// @brief Ladder scale for rung `level` (frame-level schedule: one optimize()
///        pass per rung). Same formula as robust_ladder_scale(); used by
///        GraphOptimization when it drives the ladder across passes, mirroring
///        the align path's RobustAligner (one full solve per rung).
inline float robust_ladder_scale_at_level(const GraphSolverParams::RobustSchedule& r, size_t level) {
    if (!r.enable || r.levels == 0 || r.min_scale <= 0.0f || r.init_scale <= 0.0f) {
        return 0.0f;
    }
    const size_t lvl = std::min(level, r.levels - 1);
    const float alpha =
        r.levels == 1
            ? 0.0f
            : std::pow(r.min_scale / r.init_scale, 1.0f / static_cast<float>(r.levels - 1));
    return std::max(r.min_scale, r.init_scale * std::pow(alpha, static_cast<float>(lvl)));
}

/// @brief Gauss-Newton solver over the sliding-window pose graph.
class GraphSolver {
public:
    enum class Status {
        MAX_ITERATIONS,
        CONVERGED,
        NON_FINITE_SYSTEM,
        DECOMPOSITION_FAILED,
        NON_FINITE_STEP,
        NON_FINITE_OBJECTIVE,
        UNSTABLE_STEP,
        NO_PROGRESS,
    };

    struct Result {
        bool converged = false;
        size_t iterations = 0;
        size_t inner_iterations = 0;
        size_t accepted_steps = 0;
        size_t rejected_steps = 0;
        float final_error = 0.0f;
        float final_lambda = 0.0f;
        Status status = Status::MAX_ITERATIONS;

        bool valid() const { return valid_status(status); }
    };

    /// @brief A non-finite system, decomposition failure, or non-finite/unstable
    ///        step leaves the current estimate unusable; MAX_ITERATIONS is still
    ///        a best-effort valid result. Callers (pipeline) must discard the
    ///        frame when this returns false.
    static bool valid_status(Status status) {
        return status != Status::NON_FINITE_SYSTEM && status != Status::DECOMPOSITION_FAILED &&
               status != Status::NON_FINITE_STEP && status != Status::NON_FINITE_OBJECTIVE &&
               status != Status::UNSTABLE_STEP;
    }

    GraphSolver(const sycl_utils::DeviceQueue& queue,
                const GraphSolverParams& params = GraphSolverParams())
        : queue_(queue), params_(params) {
        // Reject invalid configuration up front: a zero iteration count never
        // solves, and non-finite thresholds or non-positive lambdas would
        // silently corrupt the optimization. Negative convergence thresholds
        // are allowed on purpose: they disable the convergence check (never-
        // converge mode, used by the per-call iteration-limit test).
        if (params_.max_iterations == 0 || !std::isfinite(params_.convergence_rotation) ||
            !std::isfinite(params_.convergence_translation) ||
            !std::isfinite(params_.solver_damping_lambda) || params_.solver_damping_lambda <= 0.0f ||
            !std::isfinite(params_.marginalization_lambda) || params_.marginalization_lambda <= 0.0f) {
            throw std::invalid_argument("[GraphSolver] invalid solver parameters");
        }
        if (params_.optimization_method != registration::OptimizationMethod::GAUSS_NEWTON &&
            params_.optimization_method != registration::OptimizationMethod::LEVENBERG_MARQUARDT) {
            throw std::invalid_argument("[GraphSolver] unsupported optimization method");
        }
        if (params_.lm.max_inner_iterations == 0 || !std::isfinite(params_.lm.lambda_factor) ||
            params_.lm.lambda_factor <= 1.0f || !std::isfinite(params_.lm.init_lambda) ||
            !std::isfinite(params_.lm.min_lambda) || !std::isfinite(params_.lm.max_lambda) ||
            params_.lm.min_lambda <= 0.0f || params_.lm.init_lambda < params_.lm.min_lambda ||
            params_.lm.init_lambda > params_.lm.max_lambda) {
            throw std::invalid_argument("[GraphSolver] invalid LM parameters");
        }
        if (params_.robust.enable &&
            (params_.robust.levels == 0 || params_.robust.iters_per_level == 0 ||
             !std::isfinite(params_.robust.init_scale) || params_.robust.init_scale <= 0.0f ||
             !std::isfinite(params_.robust.min_scale) || params_.robust.min_scale <= 0.0f ||
             params_.robust.init_scale < params_.robust.min_scale)) {
            throw std::invalid_argument("[GraphSolver] invalid robust schedule");
        }
    }

    const GraphSolverParams& params() const { return params_; }

    /// @brief Run the configured nonlinear optimizer over the sliding window.
    /// @param robust_scale_override When set, every iteration linearizes with
    ///        this fixed scale and the internal robust ladder (and its
    ///        ladder-done convergence gating) is bypassed: the caller drives
    ///        the frame-level schedule, mirroring the align path's
    ///        RobustAligner (one full solve per rung). When unset, the behavior
    ///        is unchanged (internal ladder when params_.robust.enable).
    Result optimize(SlidingWindow& window, std::optional<float> robust_scale_override = std::nullopt,
                    std::optional<size_t> max_iterations_override = std::nullopt) {
        Result result;
        const size_t max_iterations = max_iterations_override.value_or(params_.max_iterations);
        float lm_lambda = params_.lm.init_lambda;
        result.final_lambda = params_.optimization_method == registration::OptimizationMethod::LEVENBERG_MARQUARDT
                                  ? lm_lambda
                                  : params_.solver_damping_lambda;
        for (size_t iter = 0; iter < max_iterations; ++iter) {
            auto sys = assemble(window, robust_scale_override.value_or(robust_ladder_scale(params_.robust, iter)));

            result.final_error = sys.error;
            if (!sys.H.allFinite() || !sys.b.allFinite() || !std::isfinite(sys.error)) {
                result.status = Status::NON_FINITE_SYSTEM;
                break;
            }

            if (params_.optimization_method == registration::OptimizationMethod::LEVENBERG_MARQUARDT) {
                const auto current_poses = collect_poses(window, sys.node_ids);
                const auto current_eval = evaluate_objective(
                    window, sys.node_ids, current_poses, &sys.stale_factors,
                    sys.current_error_base, sys.current_inliers_base, false);
                result.final_error = current_eval.error;
                if (!current_eval.finite) {
                    result.status = Status::NON_FINITE_OBJECTIVE;
                    break;
                }

                bool accepted = false;
                bool saw_finite_trial = false;
                bool saw_converged_rejected_trial = false;
                bool converged = false;
                float accepted_max_dt = 0.0f;
                float accepted_max_dr = 0.0f;
                Status failure = Status::DECOMPOSITION_FAILED;
                for (size_t inner = 0; inner < params_.lm.max_inner_iterations; ++inner) {
                    ++result.inner_iterations;
                    Eigen::VectorXf delta;
                    float max_dt = 0.0f;
                    float max_dr = 0.0f;
                    if (!solve_damped(sys, lm_lambda, delta, max_dt, max_dr, failure)) {
                        ++result.rejected_steps;
                    } else {
                        auto trial_poses = current_poses;
                        for (size_t i = 0; i < trial_poses.size(); ++i) {
                            trial_poses[i] = Eigen::Isometry3f(
                                trial_poses[i].matrix() *
                                eigen_utils::lie::se3_exp(delta.segment<6>(6 * i)));
                        }
                        const auto trial_eval = evaluate_objective(window, sys.node_ids, trial_poses);
                        if (trial_eval.finite) {
                            saw_finite_trial = true;
                            if (trial_eval.error <= current_eval.error) {
                                apply_poses(window, sys.node_ids, trial_poses);
                                result.final_error = trial_eval.error;
                                result.accepted_steps++;
                                accepted = true;
                                accepted_max_dt = max_dt;
                                accepted_max_dr = max_dr;
                                converged = step_converged(delta, sys.node_ids.size());
                                lm_lambda = std::clamp(lm_lambda / params_.lm.lambda_factor,
                                                       params_.lm.min_lambda, params_.lm.max_lambda);
                                break;
                            }
                            // At a local minimum, float noise can make every
                            // non-zero trial microscopically worse. A rejected
                            // step below the convergence thresholds proves no
                            // meaningful update remains; keep the current pose
                            // and report convergence without committing it.
                            saw_converged_rejected_trial =
                                saw_converged_rejected_trial ||
                                step_converged(delta, sys.node_ids.size());
                        } else {
                            failure = Status::NON_FINITE_OBJECTIVE;
                        }
                        ++result.rejected_steps;
                    }
                    lm_lambda = std::clamp(lm_lambda * params_.lm.lambda_factor,
                                           params_.lm.min_lambda, params_.lm.max_lambda);
                }
                result.final_lambda = lm_lambda;
                result.iterations = iter + 1;
                if (params_.verbose) {
                    std::cout << "iter [" << iter << "] "
                              << "error: " << result.final_error << ", "
                              << "lambda: " << lm_lambda << ", "
                              << "accepted: " << accepted << ", "
                              << "dt: " << accepted_max_dt << ", "
                              << "dr: " << accepted_max_dr << std::endl;
                }
                if (!accepted) {
                    if (saw_converged_rejected_trial) {
                        result.converged = true;
                        result.status = Status::CONVERGED;
                    } else {
                        result.status = saw_finite_trial ? Status::NO_PROGRESS : failure;
                    }
                    break;
                }

                const size_t ladder_iters =
                    std::max<size_t>(1, params_.robust.levels) *
                    std::max<size_t>(1, params_.robust.iters_per_level);
                const bool ladder_done = robust_scale_override.has_value() || !params_.robust.enable ||
                                         (iter + 1) >= ladder_iters;
                if (converged && ladder_done) {
                    result.converged = true;
                    result.status = Status::CONVERGED;
                    break;
                }
                continue;
            }

            // Eigen LDLT reports Success even on numerically singular input, so a
            // naive solve can emit a huge but finite step that silently moves
            // every pose. Gate the direction on step magnitude and eigenvalue
            // conditioning, escalating damping until it is usable (mirrors the
            // marginalization ladder); give up when the system stays unusable.
            float lambda = params_.solver_damping_lambda;
            Status failure = Status::DECOMPOSITION_FAILED;
            Eigen::VectorXf delta;
            bool accepted = false;
            for (int escalation = 0; escalation <= kMaxDampingEscalations && !accepted; ++escalation) {
                if (escalation > 0 && params_.verbose) {
                    std::cout << "solver damping escalation " << escalation << "/" << kMaxDampingEscalations
                              << ", lambda=" << lambda << std::endl;
                }
                const Eigen::MatrixXf H_reg =
                    sys.H + lambda * Eigen::MatrixXf::Identity(sys.H.rows(), sys.H.cols());
                Eigen::LDLT<Eigen::MatrixXf> ldlt(H_reg);
                if (ldlt.info() != Eigen::Success) {
                    failure = Status::DECOMPOSITION_FAILED;
                } else {
                    delta = ldlt.solve(-sys.b);
                    if (!delta.allFinite()) {
                        failure = Status::NON_FINITE_STEP;
                    } else {
                        float max_dt = 0.0f;
                        float max_dr = 0.0f;
                        for (size_t i = 0; i < sys.node_ids.size(); ++i) {
                            const Eigen::Matrix<float, 6, 1> d = delta.segment<6>(6 * i);
                            max_dr = std::max(max_dr, d.head<3>().norm());
                            max_dt = std::max(max_dt, d.tail<3>().norm());
                        }
                        if (max_dt > params_.max_step_translation || max_dr > params_.max_step_rotation) {
                            failure = Status::UNSTABLE_STEP;
                        } else if (!is_well_conditioned(H_reg)) {
                            failure = Status::DECOMPOSITION_FAILED;
                        } else {
                            accepted = true;
                        }
                    }
                }
                if (!accepted && escalation < kMaxDampingEscalations) lambda *= 10.0f;
            }
            if (!accepted) {
                result.status = failure;
                break;
            }

            bool converged = true;
            float max_dt = 0.0f;
            float max_dr = 0.0f;
            for (size_t i = 0; i < sys.node_ids.size(); ++i) {
                Eigen::Matrix<float, 6, 1> d = delta.segment<6>(6 * i);
                max_dr = std::max(max_dr, d.head<3>().norm());
                max_dt = std::max(max_dt, d.tail<3>().norm());
                if (d.head<3>().norm() > params_.convergence_rotation ||
                    d.tail<3>().norm() > params_.convergence_translation)
                    converged = false;
            }

            if (params_.verbose) {
                std::cout << "iter [" << iter << "] ";
                std::cout << "error: " << sys.error << ", ";
                std::cout << "inlier: " << sys.inliers << ", ";
                std::cout << "dt: " << max_dt << ", ";
                std::cout << "dr: " << max_dr << std::endl;
            }

            for (size_t i = 0; i < sys.node_ids.size(); ++i) {
                auto node = window.get_node(sys.node_ids[i]);
                if (!node) continue;
                const Eigen::Matrix<float, 6, 1> d = delta.segment<6>(6 * i);
                node->pose = Eigen::Isometry3f(node->pose.matrix() * eigen_utils::lie::se3_exp(d));
            }

            result.final_error = sys.error;
            result.final_lambda = lambda;
            result.iterations = iter + 1;
            // With an active ladder, small steps alone must not stop the loop:
            // convergence is only granted once the schedule reached its floor
            // (mirrors align-path RobustAligner running every level). When the
            // caller drives a fixed-scale schedule, the ladder gating is the
            // caller's concern: grant convergence normally.
            const size_t ladder_iters =
                std::max<size_t>(1, params_.robust.levels) *
                std::max<size_t>(1, params_.robust.iters_per_level);
            const bool ladder_done = robust_scale_override.has_value() || !params_.robust.enable ||
                                     (iter + 1) >= ladder_iters;
            if (converged && ladder_done) {
                result.converged = true;
                result.status = Status::CONVERGED;
                break;
            }
        }
        return result;
    }

private:
    static constexpr int kMaxDampingEscalations = 3;    ///< solver lambda *= 10 retries per iteration
    static constexpr float kMinConditionRatio = 1e-6f;  ///< required lambda_min/lambda_max of the damped H

    /// @brief A PSD (up to noise) Hessian whose eigenvalue span is too small is
    ///        treated the same as a decomposition failure even when LDLT itself
    ///        reports Success.
    static bool is_well_conditioned(const Eigen::MatrixXf& H) {
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(H);
        if (eig.info() != Eigen::Success) return false;
        const float ev_max = eig.eigenvalues().maxCoeff();
        const float ev_min = eig.eigenvalues().minCoeff();
        return ev_max > 0.0f && ev_min >= kMinConditionRatio * ev_max;
    }

    struct LinearizedSystem {
        Eigen::MatrixXf H;
        Eigen::VectorXf b;
        float error = 0.0f;
        size_t inliers = 0;
        std::vector<NodeId> node_ids;
        float current_error_base = 0.0f;
        size_t current_inliers_base = 0;
        std::vector<GraphFactorBase::Ptr> stale_factors;
    };

    struct ObjectiveEvaluation {
        float error = 0.0f;
        size_t inliers = 0;
        bool finite = true;
    };

    static std::vector<Eigen::Isometry3f> collect_poses(
        SlidingWindow& window, const std::vector<NodeId>& node_ids) {
        std::vector<Eigen::Isometry3f> poses;
        poses.reserve(node_ids.size());
        for (const NodeId id : node_ids) {
            const auto node = window.get_node(id);
            if (!node) throw std::logic_error("[GraphSolver] active node not found");
            poses.push_back(node->pose);
        }
        return poses;
    }

    static void apply_poses(SlidingWindow& window, const std::vector<NodeId>& node_ids,
                            const std::vector<Eigen::Isometry3f>& poses) {
        for (size_t i = 0; i < node_ids.size(); ++i) {
            const auto node = window.get_node(node_ids[i]);
            if (!node) throw std::logic_error("[GraphSolver] active node not found");
            node->pose = poses[i];
        }
    }

    ObjectiveEvaluation evaluate_objective(const SlidingWindow& window,
                                            const std::vector<NodeId>& node_ids,
                                            const std::vector<Eigen::Isometry3f>& poses,
                                            const std::vector<GraphFactorBase::Ptr>* factors = nullptr,
                                            float base_error = 0.0f,
                                            size_t base_inliers = 0,
                                            bool include_prior = true) const {
        auto pose_of = [&](NodeId id) -> const Eigen::Isometry3f& {
            const auto it = std::find(node_ids.begin(), node_ids.end(), id);
            if (it == node_ids.end()) throw std::logic_error("[GraphSolver] factor node not active");
            return poses[static_cast<size_t>(std::distance(node_ids.begin(), it))];
        };

        ObjectiveEvaluation eval;
        eval.error = base_error;
        eval.inliers = base_inliers;
        eval.finite = std::isfinite(base_error);
        if (!eval.finite) return eval;
        const Eigen::Isometry3f fixed_target = Eigen::Isometry3f::Identity();
        const auto& selected_factors = factors ? *factors : window.factors();
        std::vector<FactorErrorEvaluation> pending;
        pending.reserve(selected_factors.size());
        sycl_utils::events all_events;
        for (const auto& factor : selected_factors) {
            const auto [sid, tid] = factor->node_ids();
            pending.push_back(factor->compute_error_async(
                pose_of(sid), tid == INVALID_NODE_ID ? fixed_target : pose_of(tid)));
            all_events += pending.back().events;
        }
        // Every GPU factor is now in flight. Waiting here, once per objective,
        // allows independent queues/factors to overlap instead of serializing
        // submit -> wait -> submit -> wait in the factor loop.
        all_events.wait_and_throw();
        for (const auto& evaluation : pending) {
            const auto [error, inlier] = evaluation.collect();
            eval.error += error;
            eval.inliers += inlier;
            if (!std::isfinite(error) || !std::isfinite(eval.error)) {
                eval.finite = false;
                return eval;
            }
        }

        const auto& prior = window.prior();
        if (include_prior && prior.is_valid()) {
            std::vector<Eigen::Isometry3f> prior_poses;
            prior_poses.reserve(prior.node_ids.size());
            for (const NodeId id : prior.node_ids) prior_poses.push_back(pose_of(id));
            const auto contribution = prior.evaluate(prior_poses);
            eval.error += contribution.error;
            if (!std::isfinite(contribution.error) || !std::isfinite(eval.error)) eval.finite = false;
        }
        return eval;
    }

    bool solve_damped(const LinearizedSystem& sys, float lambda, Eigen::VectorXf& delta,
                      float& max_dt, float& max_dr, Status& failure) const {
        const Eigen::MatrixXf H_reg =
            sys.H + lambda * Eigen::MatrixXf::Identity(sys.H.rows(), sys.H.cols());
        Eigen::LDLT<Eigen::MatrixXf> ldlt(H_reg);
        if (ldlt.info() != Eigen::Success) {
            failure = Status::DECOMPOSITION_FAILED;
            return false;
        }
        delta = ldlt.solve(-sys.b);
        if (!delta.allFinite()) {
            failure = Status::NON_FINITE_STEP;
            return false;
        }
        max_dt = 0.0f;
        max_dr = 0.0f;
        for (size_t i = 0; i < sys.node_ids.size(); ++i) {
            const Eigen::Matrix<float, 6, 1> d = delta.segment<6>(6 * i);
            max_dr = std::max(max_dr, d.head<3>().norm());
            max_dt = std::max(max_dt, d.tail<3>().norm());
        }
        if (max_dt > params_.max_step_translation || max_dr > params_.max_step_rotation) {
            failure = Status::UNSTABLE_STEP;
            return false;
        }
        if (!is_well_conditioned(H_reg)) {
            failure = Status::DECOMPOSITION_FAILED;
            return false;
        }
        return true;
    }

    bool step_converged(const Eigen::VectorXf& delta, size_t node_count) const {
        for (size_t i = 0; i < node_count; ++i) {
            const Eigen::Matrix<float, 6, 1> d = delta.segment<6>(6 * i);
            if (d.head<3>().norm() > params_.convergence_rotation ||
                d.tail<3>().norm() > params_.convergence_translation) {
                return false;
            }
        }
        return true;
    }

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
            sys.error += lin.error;
            sys.inliers += lin.inlier;
            if (factor->last_get_relinearized()) {
                sys.current_error_base += lin.error;
                sys.current_inliers_base += lin.inlier;
            } else {
                sys.stale_factors.push_back(factor);
            }
            bool has_target = tid != INVALID_NODE_ID && idx.count(tid);

            // Stale cached linearizations live in offset-from-linearization
            // coordinates o = Log(T_lin^-1 T). With right perturbations the
            // offset moves as o(delta) = o + Jr(o) delta (BCH; Jr = Jl(-o)^-1),
            // so BOTH the quadratic term and the gradient must be transported
            // into the update's current tangent:
            //     H = U^T H_lin U (block-wise),  g = U^T (H_lin o + b).
            // Leaving H un-transported would keep the stale factor's curvature
            // expressed in the old coordinates and bend the GN step; without
            // the whole transport stale models make Gauss-Newton stall.
            const Eigen::Isometry3f& src_lin = lin.source_linearization_pose;
            const Eigen::Matrix<float, 6, 1> ds =
                eigen_utils::lie::se3_log(src_lin.inverse() * window.get_node(sid)->pose);
            const Eigen::Matrix<float, 6, 6> U_s = eigen_utils::lie::se3_right_jacobian(ds);
            Eigen::Matrix<float, 6, 1> q0 = lin.b0;
            Eigen::Matrix<float, 6, 1> q1 = lin.b1;
            if (ds.norm() > 0.0f) {
                q0 += lin.H00 * ds;
                if (has_target) {
                    q1 += lin.H01.transpose() * ds;
                }
            }
            sys.H.block<6, 6>(6 * si, 6 * si) += U_s.transpose() * lin.H00 * U_s;
            if (has_target) {
                int ti = idx.at(tid);
                const Eigen::Matrix<float, 6, 1> dt =
                    eigen_utils::lie::se3_log(lin.target_linearization_pose.inverse() *
                                              window.get_node(tid)->pose);
                const Eigen::Matrix<float, 6, 6> U_t = eigen_utils::lie::se3_right_jacobian(dt);
                if (dt.norm() > 0.0f) {
                    q0 += lin.H01 * dt;
                    q1 += lin.H11 * dt;
                }
                // Transported cross block must stay symmetric: complete H_st
                // first and add its transpose, NOT H01^T U_s U_t.
                const Eigen::Matrix<float, 6, 6> H_st = U_s.transpose() * lin.H01 * U_t;
                sys.H.block<6, 6>(6 * ti, 6 * ti) += U_t.transpose() * lin.H11 * U_t;
                sys.H.block<6, 6>(6 * si, 6 * ti) += H_st;
                sys.H.block<6, 6>(6 * ti, 6 * si) += H_st.transpose();
                sys.b.segment<6>(6 * ti) += U_t.transpose() * q1;
            }
            sys.b.segment<6>(6 * si) += U_s.transpose() * q0;
        }

        const auto& prior = window.prior();
        if (prior.is_valid()) {
            bool all_present = true;
            std::vector<int> prior_indices;
            std::vector<Eigen::Isometry3f> poses;
            prior_indices.reserve(prior.node_ids.size());
            poses.reserve(prior.node_ids.size());
            for (const NodeId id : prior.node_ids) {
                const auto it = idx.find(id);
                if (it == idx.end()) {
                    all_present = false;
                    break;
                }
                prior_indices.push_back(it->second);
                poses.push_back(window.get_node(id)->pose);
            }
            if (all_present) {
                const auto c = prior.evaluate(poses);
                for (size_t i = 0; i < prior_indices.size(); ++i) {
                    const int pi = prior_indices[i];
                    sys.b.segment<6>(6 * pi) += c.b.segment<6>(6 * i);
                    for (size_t j = 0; j < prior_indices.size(); ++j) {
                        const int pj = prior_indices[j];
                        sys.H.block<6, 6>(6 * pi, 6 * pj) += c.H.block<6, 6>(6 * i, 6 * j);
                    }
                }
                sys.error += c.error;
                sys.current_error_base += c.error;
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
