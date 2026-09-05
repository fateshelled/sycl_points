#pragma once

#include <memory>
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
/// diag(1/sigma_rotation^2 * I3, 1/sigma_translation^2 * I3).
///
/// No point-cloud work happens in linearize(), so needs_relinearization()
/// always returns true and the factor stays gradient-exact at negligible cost.
class RelativePoseFactor : public GicpFactorBase {
public:
    RelativePoseFactor(NodeId src_id, std::shared_ptr<PoseNode> src_node, NodeId tgt_id,
                       std::shared_ptr<PoseNode> tgt_node, const Eigen::Isometry3f& G,
                       const RelativePoseParams& params = RelativePoseParams())
        : src_id_(src_id), src_node_(std::move(src_node)), tgt_id_(tgt_id),
          tgt_node_(std::move(tgt_node)), G_(G), Omega_(make_information(params)) {}

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

    FactorLinearization linearize(const sycl_utils::DeviceQueue&, float /*scale*/ = 0.0f,
                                  bool /*raw*/ = false) override {
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

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
