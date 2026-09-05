#pragma once

#include <cstdint>
#include <limits>
#include <memory>

#include <Eigen/Dense>

#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/points/point_cloud.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Node identifier type used throughout the graph optimizer.
using NodeId = uint64_t;

/// @brief Sentinel value for an invalid / absent node (e.g. a fixed target).
static constexpr NodeId INVALID_NODE_ID = std::numeric_limits<NodeId>::max();

/// @brief A pose node in the sliding-window pose graph.
struct PoseNode {
    enum class Type { ACTIVE_WINDOW, CURRENT, MARGINALIZED };

    NodeId id = INVALID_NODE_ID;
    double timestamp = 0.0;
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();               // current estimate
    Eigen::Isometry3f linearization_pose = Eigen::Isometry3f::Identity();  // linearization point
    std::shared_ptr<PointCloudShared> cloud = nullptr;                     // kept for relinearization
    std::shared_ptr<knn::KNNBase> knn = nullptr;                           // kNN built on `cloud` (binary factors)
    bool has_covariance = false;
    Type type = Type::ACTIVE_WINDOW;
    bool needs_relinearization = false;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Linearization result of a single factor, expressed as 6x6 blocks
///        in the (source, target) node frame.
struct FactorLinearization {
    Eigen::Matrix<float, 6, 6> H00 = Eigen::Matrix<float, 6, 6>::Zero();  // source-source
    Eigen::Matrix<float, 6, 6> H01 = Eigen::Matrix<float, 6, 6>::Zero();  // source-target
    Eigen::Matrix<float, 6, 6> H11 = Eigen::Matrix<float, 6, 6>::Zero();  // target-target
    Eigen::Matrix<float, 6, 1> b0 = Eigen::Matrix<float, 6, 1>::Zero();
    Eigen::Matrix<float, 6, 1> b1 = Eigen::Matrix<float, 6, 1>::Zero();
    float error = 0.0f;
    uint32_t inlier = 0;
    Eigen::Isometry3f source_linearization_pose = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f target_linearization_pose = Eigen::Isometry3f::Identity();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
