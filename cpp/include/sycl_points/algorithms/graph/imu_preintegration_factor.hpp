#pragma once

#include <memory>
#include <stdexcept>

#include <Eigen/Dense>

#include "sycl_points/algorithms/graph/graph_factor.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Tightly-coupled IMU preintegration edge factor between two 15-DOF
///        graph nodes.
///
/// Residual (15-dim, ordered to match the preintegration covariance
/// [p, rot, v, accel-bias, gyro-bias]):
///   r_p  = R_iw^T (p_j - p_i - v_i·dt - ½ g·dt²) - Δp
///   r_R  = Log( ΔR^T R_iw^T R_jw )
///   r_v  = R_iw^T (v_j - v_i - g·dt) - Δv
///   r_ba = ba_j - ba_i
///   r_bg = bg_j - bg_i
/// where R_iw = R_i · R_lidar_imu (T_imu_to_lidar.rotation(); the IMU/LiDAR
/// lever arm is deliberately ignored, matching the IEKF LIO convention) and
/// (ΔR, Δv, Δp) are the bias-corrected preintegrated measurements.
///
/// Node tangent convention (matches the solver):
///   [ δθ(3, node right-perturbation) | δt(3, body translation) |
///     δv(3, world) | δba(3) | δbg(3) ]
/// so ∂r_p/∂δt_i = -R_lidar_imu^T etc. The bias random-walk is folded in
/// through the covariance's bias blocks, which the preintegrator fills from
/// the configured bias random-walk noise densities.
///
/// Jacobians are analytic and re-linearized on every solver assemble (host-only
/// math, no GPU work), so stale-linearization transport is never exercised.
class ImuPreintegrationFactor : public GraphFactorBase {
public:
    ImuPreintegrationFactor(std::shared_ptr<PoseNode> source_node, std::shared_ptr<PoseNode> target_node,
                            imu::IMUPreintegration preintegration, const Eigen::Isometry3f& T_imu_to_lidar,
                            const Eigen::Vector3f& gravity)
        : source_node_(std::move(source_node)),
          target_node_(std::move(target_node)),
          preint_(std::move(preintegration)),
          R_lidar_imu_(T_imu_to_lidar.rotation()),
          gravity_(gravity) {
        const auto& cov = preint_.get_raw().covariance;
        if (!cov.allFinite() || !imu::compute_imu_information(cov, information_)) {
            throw std::invalid_argument(
                "[ImuPreintegrationFactor] preintegration covariance is not invertible; enable IMU noise "
                "densities (gyro/accel noise and bias random walk)");
        }
        if (preint_.get_dt_total() <= 0.0) {
            throw std::invalid_argument("[ImuPreintegrationFactor] empty preintegration window");
        }
    }

    /// @brief Evaluate the residual and its analytic Jacobians at the given
    ///        endpoint states. Exposed for numerical-Jacobian tests.
    void evaluate(const NodeState& si, const NodeState& sj, Eigen::Matrix<float, 15, 1>& r,
                  Eigen::Matrix<float, 15, 15>& Ji, Eigen::Matrix<float, 15, 15>& Jj) const {
        imu::IMUBias bias;
        bias.gyro_bias = si.gyro_bias;
        bias.accel_bias = si.accel_bias;
        const imu::PreintegrationResult c = preint_.get_corrected(bias);
        const float dt = static_cast<float>(c.dt_total);

        const Eigen::Matrix3f R_i = si.pose.rotation();
        const Eigen::Matrix3f R_j = sj.pose.rotation();
        const Eigen::Vector3f p_i = si.pose.translation();
        const Eigen::Vector3f p_j = sj.pose.translation();
        const Eigen::Matrix3f R_imu_i = R_i * R_lidar_imu_;
        const Eigen::Matrix3f R_imu_j = R_j * R_lidar_imu_;
        const Eigen::Matrix3f R_imu_i_t = R_imu_i.transpose();

        const Eigen::Vector3f u = p_j - p_i - si.velocity * dt - 0.5f * gravity_ * dt * dt;
        const Eigen::Vector3f w_p = R_imu_i_t * u;
        const Eigen::Vector3f w_v = R_imu_i_t * (sj.velocity - si.velocity - gravity_ * dt);

        Eigen::Matrix3f R_rel = c.Delta_R.transpose() * R_imu_i_t * R_imu_j;
        const Eigen::Vector3f r_R =
            eigen_utils::lie::so3_log(eigen_utils::geometry::rotation_matrix_to_quaternion(R_rel));

        r.segment<3>(0) = w_p - c.Delta_p;
        r.segment<3>(3) = r_R;
        r.segment<3>(6) = w_v - c.Delta_v;
        r.segment<3>(9) = sj.accel_bias - si.accel_bias;
        r.segment<3>(12) = sj.gyro_bias - si.gyro_bias;

        const Eigen::Matrix3f R_i2l_t = R_lidar_imu_.transpose();
        const Eigen::Matrix3f Jr_inv = imu::so3_right_jacobian_inverse(r_R);
        const Eigen::Matrix3f Jl_inv = imu::so3_right_jacobian_inverse(-r_R);
        const Eigen::Matrix3f skew_p = eigen_utils::lie::skew(w_p);
        const Eigen::Matrix3f skew_v = eigen_utils::lie::skew(w_v);

        Ji.setZero();
        Jj.setZero();
        // position residual rows
        Ji.block<3, 3>(0, 0) = skew_p * R_i2l_t;
        Ji.block<3, 3>(0, 3) = -R_i2l_t;
        Ji.block<3, 3>(0, 6) = -R_imu_i_t * dt;
        Ji.block<3, 3>(0, 9) = -c.J.J_p_ba;
        Ji.block<3, 3>(0, 12) = -c.J.J_p_bg;
        Jj.block<3, 3>(0, 3) = R_imu_i_t * R_j;
        // rotation residual rows
        Ji.block<3, 3>(3, 0) = -Jl_inv * c.Delta_R.transpose() * R_i2l_t;
        Ji.block<3, 3>(3, 12) = -Jl_inv * c.J.J_R_bg;
        Jj.block<3, 3>(3, 0) = Jr_inv * R_i2l_t;
        // velocity residual rows
        Ji.block<3, 3>(6, 0) = skew_v * R_i2l_t;
        Ji.block<3, 3>(6, 6) = -R_imu_i_t;
        Ji.block<3, 3>(6, 9) = -c.J.J_v_ba;
        Ji.block<3, 3>(6, 12) = -c.J.J_v_bg;
        Jj.block<3, 3>(6, 6) = R_imu_i_t;
        // bias random-walk rows
        Ji.block<3, 3>(9, 9) = -Eigen::Matrix3f::Identity();
        Jj.block<3, 3>(9, 9) = Eigen::Matrix3f::Identity();
        Ji.block<3, 3>(12, 12) = -Eigen::Matrix3f::Identity();
        Jj.block<3, 3>(12, 12) = Eigen::Matrix3f::Identity();
    }

    /// @brief Residual only, using this factor's node estimates.
    Eigen::Matrix<float, 15, 1> residual(const NodeState& si, const NodeState& sj) const {
        Eigen::Matrix<float, 15, 1> r;
        Eigen::Matrix<float, 15, 15> Ji, Jj;
        evaluate(si, sj, r, Ji, Jj);
        return r;
    }

    FactorLinearization linearize(const sycl_utils::DeviceQueue&, float /*scale*/ = 0.0f) override {
        const NodeState si = source_node_->state();
        const NodeState sj = target_node_->state();
        Eigen::Matrix<float, 15, 1> r;
        Eigen::Matrix<float, 15, 15> Ji, Jj;
        evaluate(si, sj, r, Ji, Jj);

        const Eigen::Matrix<float, 15, 15> Hii = Ji.transpose() * information_ * Ji;
        const Eigen::Matrix<float, 15, 15> Hij = Ji.transpose() * information_ * Jj;
        const Eigen::Matrix<float, 15, 15> Hjj = Jj.transpose() * information_ * Jj;
        const Eigen::Matrix<float, 15, 1> bi = Ji.transpose() * information_ * r;
        const Eigen::Matrix<float, 15, 1> bj = Jj.transpose() * information_ * r;

        FactorLinearization lin;
        lin.full_state = true;
        lin.H00_full = 0.5f * (Hii + Hii.transpose());
        lin.H01_full = Hij;
        lin.H11_full = 0.5f * (Hjj + Hjj.transpose());
        lin.b0_full = bi;
        lin.b1_full = bj;
        lin.error = 0.5f * r.dot(information_ * r);
        lin.inlier = 1;
        lin.source_linearization_pose = si.pose;
        lin.target_linearization_pose = sj.pose;
        return lin;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&, const Eigen::Isometry3f&) const override {
        // Pose-only entry point: consume the current node states (the solver
        // uses compute_error_state).
        return compute_error_state(source_node_->state(), target_node_->state());
    }

    std::pair<float, uint32_t> compute_error_state(const NodeState& si, const NodeState& sj) const override {
        const Eigen::Matrix<float, 15, 1> r = residual(si, sj);
        return {0.5f * r.dot(information_ * r), 1};
    }

    FactorErrorEvaluation compute_error_state_async(const NodeState& si, const NodeState& sj) const override {
        const auto result = compute_error_state(si, sj);
        return {sycl_utils::events{}, [result]() { return result; }};
    }

    std::pair<NodeId, NodeId> node_ids() const override { return {source_node_->id, target_node_->id}; }

    bool uses_full_state() const override { return true; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return true;  // host-only, cheap, and keeps the model gradient-exact
    }

    /// @brief Preintegration window duration [s].
    double dt_total() const { return preint_.get_dt_total(); }

    /// @brief Linearization bias used by the preintegrator (source-node bias at
    ///        reset time), exposed so the pipeline can decide on re-integration.
    const imu::IMUBias& linearization_bias() const { return preint_.linearization_bias(); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    std::shared_ptr<PoseNode> source_node_;
    std::shared_ptr<PoseNode> target_node_;
    imu::IMUPreintegration preint_;
    Eigen::Matrix<float, 15, 15> information_ = Eigen::Matrix<float, 15, 15>::Zero();
    Eigen::Matrix3f R_lidar_imu_ = Eigen::Matrix3f::Identity();
    Eigen::Vector3f gravity_ = Eigen::Vector3f(0.0f, 0.0f, -9.80665f);
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
