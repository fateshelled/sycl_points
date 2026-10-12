#pragma once

#include <cmath>
#include <functional>
#include <optional>

#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief True if the relative pose (current vs linearization point) moved beyond
///        the relinearization thresholds. Rotation/translation norms follow the
///        same SE(3) convention as the solver (se3_log packs rotation in head<3>()
///        and translation in tail<3>()).
inline bool relinearization_needed(const Eigen::Isometry3f& current, const Eigen::Isometry3f& lin,
                                   float rot_th, float trans_th) {
    const Eigen::Matrix<float, 6, 1> e = eigen_utils::lie::se3_log(lin.inverse() * current);
    return e.head<3>().norm() > rot_th || e.tail<3>().norm() > trans_th;
}

/// @brief Relative-pose measurement handed over when a point-cloud binary
///        factor is converted into a chain RelativePoseFactor: the frozen
///        relative pose G plus the 6x6 information matrix projected from the
///        binary's joint Hessian (anisotropic, with the robust weights the
///        optimizer actually adopted).
struct RelativePoseMeasurement {
    Eigen::Isometry3f G = Eigen::Isometry3f::Identity();
    Eigen::Matrix<float, 6, 6> information = Eigen::Matrix<float, 6, 6>::Zero();
    /// @brief Linear gradient term (per-twist) captured on the same snapshot:
    ///        the source binary factor's energy model is E(r) = 1/2 r^T Omega r
    ///        + gradient^T r about G, not a pure quadratic (its residual at the
    ///        linearization snapshot is generally non-zero). Zero for a pure
    ///        quadratic measurement.
    Eigen::Matrix<float, 6, 1> gradient = Eigen::Matrix<float, 6, 1>::Zero();
};

/// @brief A submitted factor objective evaluation. GPU factors return pending
///        events and a non-blocking collector; host factors use the default
///        immediate implementation. This lets GraphSolver submit every factor
///        before performing one aggregate wait.
struct FactorErrorEvaluation {
    sycl_utils::events events;
    std::function<std::pair<float, uint32_t>()> collect;
};

/// @brief Abstract base for all sliding-window graph factors (point-cloud GICP
///        factors, host-only chain relatives, and test mocks alike).
///
/// A factor connects either one node (unary, against a fixed target such as a
/// submap) or two nodes (binary, between two pose estimates). The linearization
/// point is captured at linearize() call time so the solver controls when
/// relinearization happens.
class GraphFactorBase {
public:
    using Ptr = std::shared_ptr<GraphFactorBase>;

    struct RuntimeState {
        std::optional<FactorLinearization> cached_linearization;
        bool annealing = false;
        bool force_relinearize_on_scale = false;
        bool last_get_relinearized = false;
        float last_scale = 0.0f;
        float frozen_scale = 0.0f;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    virtual ~GraphFactorBase() = default;

    /// @brief Linearize the factor at the current node estimates.
    ///        @param scale robust loss scale override; <=0 means "use the factor's
    ///        configured default". Scale-free factors (chain relatives) ignore it.
    virtual FactorLinearization linearize(const sycl_utils::DeviceQueue& queue, float scale = 0.0f) = 0;

    /// @brief Evaluate the robust error at the given source/target poses using
    ///        the correspondences and effective robust scale cached by the most
    ///        recent linearization. The evaluation must not mutate node poses,
    ///        factor caches, or correspondence state. Calling this before the
    ///        first linearization is a programming error.
    virtual std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f& src_pose,
                                                      const Eigen::Isometry3f& tgt_pose) const = 0;

    virtual FactorErrorEvaluation compute_error_async(const Eigen::Isometry3f& src_pose,
                                                       const Eigen::Isometry3f& tgt_pose) const {
        const auto result = compute_error(src_pose, tgt_pose);
        return {sycl_utils::events{}, [result]() { return result; }};
    }

    /// @brief State-aware objective evaluation. Pose-only factors inherit the
    ///        default, which forwards to the pose-based virtuals above; state
    ///        factors (IMU preintegration) override this to consume velocity and
    ///        biases. The solver always evaluates this form.
    virtual std::pair<float, uint32_t> compute_error_state(const NodeState& src_state,
                                                           const NodeState& tgt_state) const {
        return compute_error(src_state.pose, tgt_state.pose);
    }

    virtual FactorErrorEvaluation compute_error_state_async(const NodeState& src_state,
                                                            const NodeState& tgt_state) const {
        return compute_error_async(src_state.pose, tgt_state.pose);
    }

    /// @brief IDs of the two connected nodes. For a unary factor the target
    ///        id is INVALID_NODE_ID (fixed target).
    virtual std::pair<NodeId, NodeId> node_ids() const = 0;

    /// @brief Whether the factor should be relinearized given the latest poses.
    virtual bool needs_relinearization(const Eigen::Isometry3f& src, const Eigen::Isometry3f& tgt,
                                       float rot_th, float trans_th) const = 0;

    /// @brief True for point-cloud based binary factors (BinaryGicpFactor). The
    ///        sparse-chain topology prunes/converts exactly these; host-only
    ///        factors (chain relatives) are always kept.
    virtual bool is_point_cloud_binary() const { return false; }

    /// @brief True when the factor constrains the full 15-DOF node state
    ///        (pose + velocity + biases) instead of only the pose sub-block.
    ///        The solver selects its 15-DOF layout when any submitted factor
    ///        reports true, and otherwise keeps the legacy 6-DOF pose-only
    ///        layout (so pose-only graphs are numerically unchanged).
    virtual bool uses_full_state() const { return false; }

    /// @brief True only for LiDAR measurement factors whose transported
    ///        Hessian contributes to geometric observability. Motion-chain
    ///        factors, marginalization priors, and test/host constraints are
    ///        deliberately excluded from degeneracy detection.
    virtual bool contributes_lidar_observability() const { return false; }

    /// @brief Relative-pose measurement (G, information) captured from this
    ///        factor's latest cached linearization, used when the factor is
    ///        converted into a chain RelativePoseFactor. nullopt by default;
    ///        point-cloud binary factors override this. Callers fall back to
    ///        the sigma-based chain factor on nullopt — no extra linearization
    ///        is ever run just for the conversion.
    virtual std::optional<RelativePoseMeasurement> make_relative_pose_measurement() const {
        return std::nullopt;
    }

    /// @brief Return the linearization, reusing a cached result when the connected
    ///        node poses have not moved beyond the relinearization thresholds.
    ///        The cache lives in the base class so every factor type shares identical
    ///        reuse semantics; subclasses implement only linearize() and
    ///        needs_relinearization().
    ///        @param ladder_scale current robust-schedule ladder scale for annealing
    ///        factors (see begin_annealing/freeze). A cache hit keeps its weights
    ///        ("lag" is deliberate and bounded: scale only decreases during annealing).
    virtual FactorLinearization get_linearization(const sycl_utils::DeviceQueue& queue,
                                                 float relinearize_rotation_thresh,
                                                 float relinearize_translation_thresh,
                                                 float ladder_scale = 0.0f) {
        const float s_eff = scale_now(ladder_scale);
        const bool rung_changed = force_relin_on_scale_ && annealing_ &&
                                  std::fabs(s_eff - last_scale_) >
                                      1e-3f * std::fabs(s_eff > 0.0f ? s_eff : 1.0f);
        if (cached_lin_ && !rung_changed &&
            !needs_relinearization(Eigen::Isometry3f::Identity(),
                                   Eigen::Isometry3f::Identity(),
                                   relinearize_rotation_thresh,
                                   relinearize_translation_thresh)) {
            last_get_relinearized_ = false;
            return *cached_lin_;
        }
        cached_lin_ = this->linearize(queue, scale_now(ladder_scale));
        last_scale_ = scale_now(ladder_scale);
        last_get_relinearized_ = true;
        return *cached_lin_;
    }

    /// @brief Whether the most recent get_linearization() call refreshed this
    ///        factor rather than returning its cached model.
    bool last_get_relinearized() const { return last_get_relinearized_; }

    /// @brief Drop any cached linearization so the next get_linearization re-computes.
    virtual void clear_cache() { cached_lin_.reset(); }

    /// @brief Retarget a fixed-target (unary) factor onto a new point cloud and
    ///        its matching kNN as ONE generation, discarding the linearization
    ///        cache while preserving the robust-scale state. Returns true when
    ///        the factor owns such a target (UnaryGicpFactor); the default is a
    ///        no-op for every other factor type (binary, chain, priors, mocks).
    ///        The cloud and kNN must always be replaced together: pairing a
    ///        cloud with a kNN built from a different generation yields
    ///        correspondences against the wrong target.
    virtual bool set_fixed_target(std::shared_ptr<const PointCloudShared>, std::shared_ptr<const knn::KNNBase>) {
        return false;
    }

    /// @brief Current fixed-target generation of this factor (nullptr when the
    ///        factor has no fixed target). Mirrors set_fixed_target so callers
    ///        can verify that a retarget replaced the cloud and kNN together.
    virtual std::shared_ptr<const PointCloudShared> fixed_target_cloud() const { return nullptr; }
    virtual std::shared_ptr<const knn::KNNBase> fixed_target_knn() const { return nullptr; }

    /// @brief Pointer to the cached linearization, or nullptr when not cached.
    ///        Lets callers read the last-used statistics (error / inlier) without
    ///        triggering a re-linearization.
    const FactorLinearization* cached_linearization() const { return cached_lin_ ? &*cached_lin_ : nullptr; }

    /// @brief Scale actually used by this factor for the next relinearization:
    ///        the live ladder while annealing, the locked value once frozen.
    float scale_now(float ladder_scale) const {
        return annealing_ ? ladder_scale : frozen_scale_;
    }

    bool is_annealing() const { return annealing_; }

    /// @brief Robust scale used by the most recent actual linearization.
    ///        A non-positive value means the factor's configured default scale.
    float last_linearization_scale() const { return last_scale_; }

    /// @brief When true, an annealing factor relinearizes on every ladder rung
    ///        change (align-style full KNN per level) instead of letting the
    ///        cached weights sleep until the pose threshold trips. Only affects
    ///        the per-frame tip group; frozen factors are untouched.
    void set_robust_force_mode(bool on) { force_relin_on_scale_ = on; }

    RuntimeState runtime_state() const {
        return {cached_lin_, annealing_, force_relin_on_scale_, last_get_relinearized_, last_scale_,
                frozen_scale_};
    }

    void restore_runtime_state(const RuntimeState& state) {
        cached_lin_ = state.cached_linearization;
        annealing_ = state.annealing;
        force_relin_on_scale_ = state.force_relinearize_on_scale;
        last_get_relinearized_ = state.last_get_relinearized;
        last_scale_ = state.last_scale;
        frozen_scale_ = state.frozen_scale;
    }

protected:
    /// @brief Mark this factor as participating in the robust ladder (called by
    ///        constructors of scale-dependent factors: unary/binary point-cloud ones).
    void begin_annealing() { annealing_ = true; }

public:
    /// @brief End annealing: lock the factor's robust scale at whatever value it
    ///        last linearized with (normally the ladder floor at frame end).
    void freeze() {
        if (!annealing_) return;
        annealing_ = false;
        frozen_scale_ = last_scale_;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    mutable std::optional<FactorLinearization> cached_lin_;
    bool annealing_ = false;        ///< factors opt in via begin_annealing()
    bool force_relin_on_scale_ = false;  ///< relinearize per ladder rung (robust force mode)
    bool last_get_relinearized_ = false; ///< result of the most recent cache lookup
    float last_scale_ = 0.0f;       ///< scale used by the most recent linearize()
    float frozen_scale_ = 0.0f;     ///< locked scale after freeze(); 0 = default_scale
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
