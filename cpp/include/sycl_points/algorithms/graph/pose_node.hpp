#pragma once

#include <cstdint>
#include <limits>
#include <memory>

#include <Eigen/Dense>

#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Node identifier type used throughout the graph optimizer.
using NodeId = uint64_t;

/// @brief Sentinel value for an invalid / absent node (e.g. a fixed target).
static constexpr NodeId INVALID_NODE_ID = std::numeric_limits<NodeId>::max();

/// @brief DOF of a state-augmented node: SE(3) pose (6) + velocity (3) +
///        accel bias (3) + gyro bias (3).
static constexpr int kNodeDof = 15;
/// @brief DOF of the pose sub-block of a node (Eigen [rot; trans] se3 twist).
static constexpr int kPoseDof = 6;
/// @brief Start offset of the velocity sub-block inside the 15-vector.
static constexpr int kVelOffset = 6;
/// @brief Start offset of the accelerometer-bias sub-block inside the 15-vector.
static constexpr int kAccBiasOffset = 9;
/// @brief Start offset of the gyroscope-bias sub-block inside the 15-vector.
static constexpr int kGyrBiasOffset = 12;

/// @brief Full navigation state attached to a graph node.
///
/// Pose follows the graph's existing SE(3) convention (Eigen Isometry3f with a
/// rotation-first [rot; trans] right-perturbation twist), so all point-cloud
/// factors continue to see the same pose model. Velocity is expressed in the
/// world frame, biases in the IMU body frame; all three are perturbed
/// additively. The tangent ordering is therefore:
///   [0:6) pose (se3), [6:9) velocity, [9:12) accel bias, [12:15) gyro bias.
struct NodeState {
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
    Eigen::Vector3f accel_bias = Eigen::Vector3f::Zero();
    Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief A node in the sliding-window pose graph.
struct PoseNode {
    enum class Type { ACTIVE_WINDOW, CURRENT, MARGINALIZED };

    NodeId id = INVALID_NODE_ID;
    double timestamp = 0.0;
    Eigen::Isometry3f initial_pose = Eigen::Isometry3f::Identity();        // frame prediction anchor
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();               // current estimate
    Eigen::Isometry3f linearization_pose = Eigen::Isometry3f::Identity();  // linearization point
    std::shared_ptr<PointCloudShared> cloud = nullptr;                     // kept for relinearization
    std::shared_ptr<knn::KNNBase> knn = nullptr;                           // kNN built on `cloud` (binary factors)

    // Navigation state (world-frame velocity, body-frame biases). Pose-only
    // factors simply ignore these; state factors (IMU) constrain them.
    Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
    Eigen::Vector3f accel_bias = Eigen::Vector3f::Zero();
    Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();
    Eigen::Vector3f linearization_velocity = Eigen::Vector3f::Zero();
    Eigen::Vector3f linearization_accel_bias = Eigen::Vector3f::Zero();
    Eigen::Vector3f linearization_gyro_bias = Eigen::Vector3f::Zero();

    bool has_covariance = false;
    Type type = Type::ACTIVE_WINDOW;
    bool needs_relinearization = false;

    /// @brief Full navigation state snapshot of the current estimate.
    NodeState state() const {
        NodeState s;
        s.pose = pose;
        s.velocity = velocity;
        s.accel_bias = accel_bias;
        s.gyro_bias = gyro_bias;
        return s;
    }

    /// @brief Full navigation state snapshot at the current linearization point.
    NodeState linearization_state() const {
        NodeState s;
        s.pose = linearization_pose;
        s.velocity = linearization_velocity;
        s.accel_bias = linearization_accel_bias;
        s.gyro_bias = linearization_gyro_bias;
        return s;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Advance a node estimate by a 15-DOF tangent increment.
///        Pose uses a right SE(3) update (rotation-first twist); velocity and
///        biases are additive.
inline void apply_node_delta(PoseNode& node, const Eigen::Matrix<float, kNodeDof, 1>& delta) {
    node.pose = Eigen::Isometry3f(node.pose.matrix() * eigen_utils::lie::se3_exp(delta.head<kPoseDof>()));
    node.velocity += delta.segment<3>(kVelOffset);
    node.accel_bias += delta.segment<3>(kAccBiasOffset);
    node.gyro_bias += delta.segment<3>(kGyrBiasOffset);
}

/// @brief Advance a navigation state by a 15-DOF tangent increment (same
///        convention as the PoseNode overload).
inline void apply_node_delta(NodeState& state, const Eigen::Matrix<float, kNodeDof, 1>& delta) {
    state.pose = Eigen::Isometry3f(state.pose.matrix() * eigen_utils::lie::se3_exp(delta.head<kPoseDof>()));
    state.velocity += delta.segment<3>(kVelOffset);
    state.accel_bias += delta.segment<3>(kAccBiasOffset);
    state.gyro_bias += delta.segment<3>(kGyrBiasOffset);
}

/// @brief Linearization result of a single factor.
///
/// Pose-only factors (point-cloud GICP, chain relatives, mocks) keep filling the
/// 6x6 `H00/H01/H11` / 6x1 `b0/b1` blocks, which the solver places on the pose
/// sub-block of each connected node. State factors (IMU preintegration, bias
/// random walk) fill the `*_full` 15-DOF blocks and set `full_state = true`;
/// the solver then uses those instead.
struct FactorLinearization {
    // Pose-block linearization (used when full_state == false).
    Eigen::Matrix<float, 6, 6> H00 = Eigen::Matrix<float, 6, 6>::Zero();  // source-source
    Eigen::Matrix<float, 6, 6> H01 = Eigen::Matrix<float, 6, 6>::Zero();  // source-target
    Eigen::Matrix<float, 6, 6> H11 = Eigen::Matrix<float, 6, 6>::Zero();  // target-target
    Eigen::Matrix<float, 6, 1> b0 = Eigen::Matrix<float, 6, 1>::Zero();
    Eigen::Matrix<float, 6, 1> b1 = Eigen::Matrix<float, 6, 1>::Zero();

    // Full 15-DOF linearization (used when full_state == true). The tangent
    // ordering is [pose(6), velocity(3), accel bias(3), gyro bias(3)].
    bool full_state = false;
    Eigen::Matrix<float, kNodeDof, kNodeDof> H00_full = Eigen::Matrix<float, kNodeDof, kNodeDof>::Zero();
    Eigen::Matrix<float, kNodeDof, kNodeDof> H01_full = Eigen::Matrix<float, kNodeDof, kNodeDof>::Zero();
    Eigen::Matrix<float, kNodeDof, kNodeDof> H11_full = Eigen::Matrix<float, kNodeDof, kNodeDof>::Zero();
    Eigen::Matrix<float, kNodeDof, 1> b0_full = Eigen::Matrix<float, kNodeDof, 1>::Zero();
    Eigen::Matrix<float, kNodeDof, 1> b1_full = Eigen::Matrix<float, kNodeDof, 1>::Zero();

    float error = 0.0f;
    uint32_t inlier = 0;
    Eigen::Isometry3f source_linearization_pose = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f target_linearization_pose = Eigen::Isometry3f::Identity();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Expand a factor's cached linearization into `dof`-sized dense blocks.
///        Pose-only linearizations are placed on the leading pose sub-block
///        (0:6); full-state linearizations are used as-is when dof == 15.
inline void expand_linearization(const FactorLinearization& lin, int dof, Eigen::MatrixXf& H00,
                                 Eigen::MatrixXf& H01, Eigen::MatrixXf& H11, Eigen::VectorXf& b0,
                                 Eigen::VectorXf& b1) {
    H00 = Eigen::MatrixXf::Zero(dof, dof);
    H01 = Eigen::MatrixXf::Zero(dof, dof);
    H11 = Eigen::MatrixXf::Zero(dof, dof);
    b0 = Eigen::VectorXf::Zero(dof);
    b1 = Eigen::VectorXf::Zero(dof);
    if (dof == kNodeDof && lin.full_state) {
        H00 = lin.H00_full;
        H01 = lin.H01_full;
        H11 = lin.H11_full;
        b0 = lin.b0_full;
        b1 = lin.b1_full;
    } else {
        H00.block<6, 6>(0, 0) = lin.H00;
        H01.block<6, 6>(0, 0) = lin.H01;
        H11.block<6, 6>(0, 0) = lin.H11;
        b0.head<6>() = lin.b0;
        b1.head<6>() = lin.b1;
    }
}

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
