#pragma once

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sycl_points/algorithms/graph/graph_factor.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Parameters for the lightweight relative-pose (odometry-chain) factor.
struct RelativePoseParams {
    float sigma_rotation = 5e-3f;      // [rad]   per-axis, between keyframes
    float sigma_translation = 2e-2f;   // [m]     per-axis, between keyframes
};

/// @brief Binary factor anchoring the relative pose between two nodes to a
///        frozen measurement G (captured from scan-to-scan registration at
///        keyframe conversion time, or later from IMU preintegration).
///
/// Residual convention (right update, same as the rest of the solver):
///     r = se3_log( G^-1 * T_src^-1 * T_tgt ),  twist packed [rot; trans].
/// Jacobians: J_tgt = Jl(r) and J_src = -Jl(r) * Ad(T_tgt^-1 T_src), where Ad is
/// the SE(3) adjoint in [rot; trans] packing ([[R,0],[[t]x R, R]]) and Jl is
/// the SE(3) left Jacobian (truncated exponential series). With the residual
/// kept small by the linearization thresholds this reduces to the familiar
/// -Ad(T_tgt^-1 T_src) approximation. The information matrix is diagonal
/// diag(1/sigma_rotation^2 * I3, 1/sigma_translation^2 * I3) when built from
/// RelativePoseParams, or an arbitrary symmetric PSD matrix when constructed
/// from a projected measurement (see
/// relative_pose_measurement_from_linearization).
///
/// No point-cloud work happens in linearize(), so needs_relinearization()
/// always returns true and the factor stays gradient-exact at negligible cost.
class RelativePoseFactor : public GraphFactorBase {
public:
    RelativePoseFactor(NodeId src_id, std::shared_ptr<PoseNode> src_node, NodeId tgt_id,
                       std::shared_ptr<PoseNode> tgt_node, const Eigen::Isometry3f& G,
                       const RelativePoseParams& params = RelativePoseParams())
        : src_id_(src_id), src_node_(std::move(src_node)), tgt_id_(tgt_id),
          tgt_node_(std::move(tgt_node)), G_(G), Omega_(make_information(params)) {}

    /// @brief Construct with an explicit information matrix (e.g. projected from
    ///        a BinaryGicpFactor's joint Hessian). The matrix is symmetrized;
    ///        PSD projection and scaling are the caller's responsibility.
    RelativePoseFactor(NodeId src_id, std::shared_ptr<PoseNode> src_node, NodeId tgt_id,
                       std::shared_ptr<PoseNode> tgt_node, const Eigen::Isometry3f& G,
                       const Eigen::Matrix<float, 6, 6>& information)
        : src_id_(src_id), src_node_(std::move(src_node)), tgt_id_(tgt_id),
          tgt_node_(std::move(tgt_node)), G_(G),
          Omega_(0.5f * (information + information.transpose()).eval()) {
        if (!Omega_.allFinite()) {
            throw std::invalid_argument("[RelativePoseFactor] information matrix must be finite");
        }
    }

    static Eigen::Matrix<float, 6, 6> make_information(const RelativePoseParams& params) {
        Eigen::Matrix<float, 6, 6> omega = Eigen::Matrix<float, 6, 6>::Zero();
        omega.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() / (params.sigma_rotation * params.sigma_rotation);
        omega.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() / (params.sigma_translation * params.sigma_translation);
        return omega;
    }

    /// @brief Adjoint of T on twists packed [rot; trans]: [[R, 0], [[t]x R, R]].
    static Eigen::Matrix<float, 6, 6> adjoint(const Eigen::Isometry3f& T) {
        Eigen::Matrix<float, 6, 6> ad = Eigen::Matrix<float, 6, 6>::Zero();
        ad.block<3, 3>(0, 0) = T.linear();
        ad.block<3, 3>(3, 0) = eigen_utils::lie::skew(Eigen::Vector3f(T.translation())) * T.linear();
        ad.block<3, 3>(3, 3) = T.linear();
        return ad;
    }

    /// @brief Adjoint matrix of a twist xi = [omega; rho] (se3 bracket).
    static Eigen::Matrix<float, 6, 6> adjoint_of_twist(const Eigen::Matrix<float, 6, 1>& xi) {
        Eigen::Matrix<float, 6, 6> ad = Eigen::Matrix<float, 6, 6>::Zero();
        ad.block<3, 3>(0, 0) = eigen_utils::lie::skew(Eigen::Vector3f(xi.head<3>()));
        ad.block<3, 3>(3, 0) = eigen_utils::lie::skew(Eigen::Vector3f(xi.tail<3>()));
        ad.block<3, 3>(3, 3) = eigen_utils::lie::skew(Eigen::Vector3f(xi.head<3>()));
        return ad;
    }

    /// @brief SE(3) left Jacobian Jl(xi) = sum_k ad(xi)^k / (k+1)!.
    ///        Truncated exponential series; 12 terms stay well below float
    ///        precision for |xi| in the radian/meter range used here.
    static Eigen::Matrix<float, 6, 6> left_jacobian(const Eigen::Matrix<float, 6, 1>& xi) {
        const Eigen::Matrix<float, 6, 6> a = adjoint_of_twist(xi);
        Eigen::Matrix<float, 6, 6> result = Eigen::Matrix<float, 6, 6>::Identity();
        Eigen::Matrix<float, 6, 6> power = Eigen::Matrix<float, 6, 6>::Identity();
        float factorial = 1.0f;
        for (int k = 1; k <= 12; ++k) {
            power = power * a;
            factorial *= static_cast<float>(k + 1);
            result += power / factorial;
        }
        return result;
    }

    FactorLinearization linearize(const sycl_utils::DeviceQueue&, float /*scale*/ = 0.0f) override {
        src_node_->linearization_pose = src_node_->pose;
        tgt_node_->linearization_pose = tgt_node_->pose;
        return linearize_at(src_node_->pose, tgt_node_->pose);
    }

    std::pair<NodeId, NodeId> node_ids() const override { return {src_id_, tgt_id_}; }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f& src,
                                             const Eigen::Isometry3f& tgt) const override {
        const Eigen::Matrix<float, 6, 1> r = residual(src, tgt);
        return {0.5f * (r.transpose() * Omega_ * r)(0, 0), 1};
    }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return true;  // linearization is host-only and cheap; keep it always fresh
    }

    /// @brief The information matrix this factor constrains with (isotropic
    ///        sigma model or a projected measurement).
    const Eigen::Matrix<float, 6, 6>& information() const { return Omega_; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Eigen::Matrix<float, 6, 1> residual(const Eigen::Isometry3f& src,
                                        const Eigen::Isometry3f& tgt) const {
        return eigen_utils::lie::se3_log(G_.inverse() * (src.inverse() * tgt));
    }

    FactorLinearization linearize_at(const Eigen::Isometry3f& src,
                                     const Eigen::Isometry3f& tgt) const {
        const Eigen::Matrix<float, 6, 1> r = residual(src, tgt);
        const Eigen::Matrix<float, 6, 6> jl = left_jacobian(r);
        const Eigen::Matrix<float, 6, 6> J0 = -jl * adjoint(tgt.inverse() * src);  // d r / d right(src)
        const Eigen::Matrix<float, 6, 6> J1 = jl;                                   // d r / d right(tgt)
        const Eigen::Matrix<float, 6, 6> J0t_omega = J0.transpose() * Omega_;
        const Eigen::Matrix<float, 6, 6> J1t_omega = J1.transpose() * Omega_;

        FactorLinearization lin;
        lin.H00 = J0t_omega * J0;
        lin.H11 = J1t_omega * J1;
        lin.H01 = J0t_omega * J1;
        lin.b0 = J0t_omega * r;
        lin.b1 = J1t_omega * r;
        lin.error = 0.5f * (r.transpose() * Omega_ * r)(0, 0);
        lin.inlier = 1;
        lin.source_linearization_pose = src;
        lin.target_linearization_pose = tgt;
        return lin;
    }

    NodeId src_id_ = INVALID_NODE_ID;
    NodeId tgt_id_ = INVALID_NODE_ID;
    std::shared_ptr<PoseNode> src_node_;
    std::shared_ptr<PoseNode> tgt_node_;
    Eigen::Isometry3f G_ = Eigen::Isometry3f::Identity();
    Eigen::Matrix<float, 6, 6> Omega_ = Eigen::Matrix<float, 6, 6>::Zero();
};

/// @brief Project a cached binary factor linearization onto a relative-pose
///        measurement (G, 6x6 information) for sparse-chain conversion.
///
/// A point-cloud binary factor constrains only the relative pose of its two
/// nodes: a common rigid motion of both frames (the gauge) leaves every point
/// residual unchanged, so its 12x12 joint Hessian carries at most 6 DoF of
/// information and factors exactly through the relative-pose Jacobian
///     J = [-Ad(T_tgt^-1 T_src) | I]
/// (right increments; r = 0 at the linearization snapshot because G is captured
/// from the same poses, so Jl = I; same convention as linearize_at) as
///     H_joint = J^T Omega J.
/// Omega is recovered with the right pseudo-inverse J+ = J^T (J J^T)^-1 —
/// J is always full row rank since J J^T = Ad Ad^T + I:
///     Omega = J+^T H_joint J+
/// The result keeps the anisotropy and the robust weights the optimizer
/// adopted. Degenerate directions stay weak: rounding-induced negative
/// eigenvalues are clipped to zero and no eigenvalue floor is added — the
/// solver's damping ladder provides the numerical safety.
///
/// @return nullopt when the linearization is unusable: missing/zero/non-finite
///         Hessian, decomposition trouble, or a reconstruction residual
///         indicating the Hessian does not live on the relative-pose subspace
///         (e.g. an absolute-pose prior mixed into the factor). Callers fall
///         back to the sigma-based chain factor. Host-only math; no GPU work.
inline std::optional<RelativePoseMeasurement> relative_pose_measurement_from_linearization(
    const FactorLinearization& lin) {
    constexpr float kMinInformationTrace = 1e-12f;   // reject an all-zero Hessian
    constexpr float kReconstructionTolerance = 1e-2f;  // relative Frobenius guard

    const Eigen::Isometry3f& T_src = lin.source_linearization_pose;
    const Eigen::Isometry3f& T_tgt = lin.target_linearization_pose;
    if (!T_src.matrix().allFinite() || !T_tgt.matrix().allFinite()) return std::nullopt;

    Eigen::Matrix<float, 12, 12> H_joint = Eigen::Matrix<float, 12, 12>::Zero();
    H_joint.block<6, 6>(0, 0) = lin.H00;
    H_joint.block<6, 6>(0, 6) = lin.H01;
    H_joint.block<6, 6>(6, 0) = lin.H01.transpose();
    H_joint.block<6, 6>(6, 6) = lin.H11;
    H_joint = (0.5f * (H_joint + H_joint.transpose())).eval();
    if (!H_joint.allFinite() || H_joint.trace() <= kMinInformationTrace) return std::nullopt;

    const Eigen::Isometry3f G = T_src.inverse() * T_tgt;
    Eigen::Matrix<float, 6, 12> J = Eigen::Matrix<float, 6, 12>::Zero();
    J.block<6, 6>(0, 0) = -RelativePoseFactor::adjoint(T_tgt.inverse() * T_src);
    J.block<6, 6>(0, 6).setIdentity();

    const Eigen::Matrix<float, 6, 6> JJt = J * J.transpose();
    const Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt(JJt);
    if (ldlt.info() != Eigen::Success) return std::nullopt;
    const Eigen::Matrix<float, 12, 6> J_pinv =
        J.transpose() * ldlt.solve(Eigen::Matrix<float, 6, 6>::Identity());

    Eigen::Matrix<float, 6, 6> omega = J_pinv.transpose() * H_joint * J_pinv;
    omega = (0.5f * (omega + omega.transpose())).eval();
    if (!omega.allFinite()) return std::nullopt;

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eig(omega);
    if (eig.info() != Eigen::Success) return std::nullopt;
    omega = eig.eigenvectors() * eig.eigenvalues().cwiseMax(0.0f).asDiagonal() *
            eig.eigenvectors().transpose();
    omega = (0.5f * (omega + omega.transpose())).eval();
    if (!omega.allFinite() || omega.trace() <= kMinInformationTrace) return std::nullopt;

    // Consistency guard: H_joint must be reproducible from the projected
    // information (its null space must be exactly the common-motion gauge).
    // A large residual means the Hessian carries off-subspace information;
    // fall back to the sigma model instead of corrupting the chain.
    const Eigen::Matrix<float, 12, 12> reconstructed = J.transpose() * omega * J;
    const float h_norm = H_joint.norm();
    const float recon_error = (H_joint - reconstructed).norm();
    if (!std::isfinite(h_norm) || !std::isfinite(recon_error) ||
        recon_error > kReconstructionTolerance * h_norm) {
        return std::nullopt;
    }

    RelativePoseMeasurement measurement;
    measurement.G = G;
    measurement.information = omega;
    return measurement;
}

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
