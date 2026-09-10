#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>

// Disable Eigen SIMD: this TU compiles SYCL device kernels (graph_factor_kernel)
// for which Eigen's SSE packet specializations are invalid. See graph_factor_kernel.hpp.
#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE
#endif

#include <Eigen/Dense>
#include <sycl/sycl.hpp>

#include "sycl_points/algorithms/deskew/relative_pose_deskew.hpp"
#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/graph/gicp_factor.hpp"
#include "sycl_points/algorithms/graph/graph_optimization.hpp"
#include "sycl_points/algorithms/graph/graph_solver.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/algorithms/graph/relative_pose_factor.hpp"
#include "sycl_points/algorithms/graph/sliding_window.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace {

using namespace sycl_points;
using namespace sycl_points::algorithms;

// ---------------------------------------------------------------------------
// Test utilities
// ---------------------------------------------------------------------------

sycl_utils::DeviceQueue make_queue() {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    return sycl_utils::DeviceQueue(device);
}

PointCloudShared::Ptr make_cube_cloud(const sycl_utils::DeviceQueue& queue, size_t n, float half,
                                      std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-half, half);
    PointCloudCPU cpu;
    cpu.points->resize(n);
    for (size_t i = 0; i < n; ++i) {
        (*cpu.points)[i] = PointType(dist(gen), dist(gen), dist(gen), 1.0f);
    }
    return std::make_shared<PointCloudShared>(queue, cpu);
}

PointCloudShared::Ptr transform_cloud(const sycl_utils::DeviceQueue& queue, const PointCloudShared& src,
                                      const Eigen::Isometry3f& T) {
    PointCloudCPU cpu;
    cpu.points->resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        Eigen::Vector4f p = src.points->at(i);
        Eigen::Vector4f tp = T.matrix() * p;
        (*cpu.points)[i] = PointType(tp.x(), tp.y(), tp.z(), 1.0f);
    }
    return std::make_shared<PointCloudShared>(queue, cpu);
}

void estimate_covariances(const knn::KNNBase& knn, PointCloudShared& cloud) {
    covariance::estimate_async(knn, cloud, 10).wait_and_throw();
}

registration::RegistrationParams gicp_params(float max_corr = 2.0f) {
    registration::RegistrationFactorParams fp;
    fp.reg_type = registration::RegType::GICP;
    fp.max_correspondence_distance = max_corr;
    registration::RegistrationOptimizationParams op;
    return registration::RegistrationParams(fp, op);
}

void expect_pose_near(const Eigen::Isometry3f& a, const Eigen::Isometry3f& b, float tol_t, float tol_r) {
    const Eigen::Isometry3f diff = a.inverse() * b;
    EXPECT_LT(diff.translation().norm(), tol_t);
    Eigen::AngleAxisf aa(diff.rotation());
    float angle = aa.angle();
    if (angle > 3.14159265f) angle = 2.0f * 3.14159265f - angle;
    EXPECT_LT(angle, tol_r);
}

// Synthetic factor: anchors a node to a known target pose.
// Residual follows the same convention as GICP: r = log( T_target^{-1} * T ).
class AnchorFactor : public graph::GicpFactorBase {
public:
    AnchorFactor(std::shared_ptr<graph::PoseNode> node, const Eigen::Isometry3f& target, float weight)
        : node_(std::move(node)), target_(target), w_(weight) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        const Eigen::Matrix<float, 6, 1> r = eigen_utils::lie::se3_log(target_.inverse() * node_->pose);
        const Eigen::Matrix<float, 6, 6> J = Eigen::Matrix<float, 6, 6>::Identity();
        const Eigen::Matrix<float, 6, 6> Omega = w_ * Eigen::Matrix<float, 6, 6>::Identity();
        graph::FactorLinearization lin;
        lin.source_linearization_pose = node_->pose;
        lin.H00 = J.transpose() * Omega * J;
        lin.b0 = J.transpose() * Omega * r;
        lin.error = 0.5f * r.transpose() * Omega * r;
        lin.inlier = 1;
        return lin;
    }

    std::pair<graph::NodeId, graph::NodeId> node_ids() const override { return {node_->id, graph::INVALID_NODE_ID}; }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f& src_pose,
                                             const Eigen::Isometry3f&) const override {
        const Eigen::Matrix<float, 6, 1> r = eigen_utils::lie::se3_log(target_.inverse() * src_pose);
        return {0.5f * r.transpose() * (w_ * Eigen::Matrix<float, 6, 6>::Identity()) * r, 1};
    }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return false;
    }

private:
    std::shared_ptr<graph::PoseNode> node_;
    Eigen::Isometry3f target_;
    float w_;
};

// Synthetic binary factor: anchors the relative pose between two nodes to
// identity (i.e. target and source should coincide). Jacobian convention:
// J0 = I (source), J1 = -I (target), matching the right-update solver.
class BinaryAnchorFactor : public graph::GicpFactorBase {
public:
    BinaryAnchorFactor(std::shared_ptr<graph::PoseNode> src, std::shared_ptr<graph::PoseNode> tgt, float weight)
        : src_(std::move(src)), tgt_(std::move(tgt)), w_(weight) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        const Eigen::Matrix<float, 6, 1> r = eigen_utils::lie::se3_log(tgt_->pose.inverse() * src_->pose);
        const Eigen::Matrix<float, 6, 6> Omega = w_ * Eigen::Matrix<float, 6, 6>::Identity();
        graph::FactorLinearization lin;
        lin.source_linearization_pose = src_->pose;
        lin.target_linearization_pose = tgt_->pose;
        lin.H00 = Omega;     // J0^T Omega J0
        lin.H11 = Omega;     // J1^T Omega J1
        lin.H01 = -Omega;    // J0^T Omega J1
        lin.b0 = Omega * r;  // J0^T Omega r
        lin.b1 = -Omega * r; // J1^T Omega r
        lin.error = 0.5f * r.transpose() * Omega * r;
        lin.inlier = 1;
        return lin;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f& s, const Eigen::Isometry3f& t) const override {
        const Eigen::Matrix<float, 6, 1> r = eigen_utils::lie::se3_log(t.inverse() * s);
        return {0.5f * r.transpose() * (w_ * Eigen::Matrix<float, 6, 6>::Identity()) * r, 1};
    }

    std::pair<graph::NodeId, graph::NodeId> node_ids() const override { return {src_->id, tgt_->id}; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float, float) const override {
        return false;
    }

private:
    std::shared_ptr<graph::PoseNode> src_;
    std::shared_ptr<graph::PoseNode> tgt_;
    float w_;
};

class NonFiniteFactor : public graph::GicpFactorBase {
public:
    explicit NonFiniteFactor(std::shared_ptr<graph::PoseNode> node) : node_(std::move(node)) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        graph::FactorLinearization lin;
        lin.source_linearization_pose = node_->pose;
        lin.H00.setIdentity();
        lin.b0.setZero();
        lin.error = std::numeric_limits<float>::quiet_NaN();
        lin.inlier = 1;
        return lin;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&,
                                             const Eigen::Isometry3f&) const override {
        return {std::numeric_limits<float>::quiet_NaN(), 1};
    }

    std::pair<graph::NodeId, graph::NodeId> node_ids() const override {
        return {node_->id, graph::INVALID_NODE_ID};
    }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return false;
    }

private:
    std::shared_ptr<graph::PoseNode> node_;
};

// ---------------------------------------------------------------------------
// SlidingWindow management (host-only)
// ---------------------------------------------------------------------------

TEST(ResidualNormTest, ClampsFiniteNegativeButPropagatesNonFiniteValues) {
    EXPECT_FLOAT_EQ(registration::kernel::residual_norm_from_squared_error(-1e-5f), 0.0f);
    EXPECT_FLOAT_EQ(registration::kernel::residual_norm_from_squared_error(4.0f), 2.0f);
    EXPECT_TRUE(std::isnan(registration::kernel::residual_norm_from_squared_error(
        std::numeric_limits<float>::quiet_NaN())));
    EXPECT_TRUE(std::isinf(registration::kernel::residual_norm_from_squared_error(
        std::numeric_limits<float>::infinity())));
}

TEST(ResidualNormTest, RotationDivergenceUsesUnaryHalfSquaredConvention) {
    constexpr float divergence = 2.0f;
    const float residual = registration::kernel::residual_norm_from_squared_error(
        0.5f * divergence * divergence);
    EXPECT_NEAR(residual, divergence / std::sqrt(2.0f), 1e-6f);
}

class GraphSlidingWindowTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue = make_queue();
};

TEST_F(GraphSlidingWindowTest, AddNodesAndFactors) {
    graph::SlidingWindow window(5);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);

    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id0), Eigen::Isometry3f::Identity(), 1.0f));
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id1), Eigen::Isometry3f::Identity(), 1.0f));
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id2), Eigen::Isometry3f::Identity(), 1.0f));

    EXPECT_EQ(window.window_size(), 3U);
    EXPECT_EQ(window.factors().size(), 3U);
    EXPECT_FALSE(window.prior().is_valid());
}

TEST_F(GraphSlidingWindowTest, UsesConfiguredMarginalizationLambda) {
    graph::GraphSolverParams params;
    params.marginalization_lambda = 1e-3f;
    graph::GraphOptimization optimizer(queue, params, 5);
    EXPECT_FLOAT_EQ(optimizer.window().marginalization_lambda(), 1e-3f);
}

TEST_F(GraphSlidingWindowTest, MarginalizeOldestShrinksWindow) {
    graph::SlidingWindow window(2);  // max window size = 2
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);

    // Only the factor touching id0 is absorbed; the id1-id2 factor stays live.
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id0), window.get_node(id1), 5.0f));
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id1), window.get_node(id2), 5.0f));

    const graph::NodeId marginalized = window.marginalize_oldest(queue);
    EXPECT_NE(marginalized, graph::INVALID_NODE_ID);
    EXPECT_EQ(window.window_size(), 2U);
    EXPECT_TRUE(window.prior().is_valid());
    ASSERT_EQ(window.prior().node_ids.size(), 1U);
    EXPECT_EQ(window.prior().node_ids[0], id1);
    EXPECT_EQ(window.get_node(marginalized), nullptr);
}

TEST_F(GraphSlidingWindowTest, MarginalizationDoesNotAbsorbSurvivingFactors) {
    graph::SlidingWindow window(2);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id0), Eigen::Isometry3f::Identity(), 10.0f));
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id0), window.get_node(id1), 5.0f));
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id2), Eigen::Isometry3f::Identity(), 7.0f));

    ASSERT_NE(window.marginalize_oldest(queue), graph::INVALID_NODE_ID);
    ASSERT_EQ(window.prior().node_ids.size(), 1U);
    EXPECT_EQ(window.prior().node_ids[0], id1);
    ASSERT_EQ(window.factors().size(), 1U);
    EXPECT_EQ(window.factors()[0]->node_ids().first, id2);
}

TEST_F(GraphSlidingWindowTest, MarginalizationPreservesMarkovBlanketCoupling) {
    graph::SlidingWindow window(2);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id0), Eigen::Isometry3f::Identity(), 10.0f));
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id0), window.get_node(id1), 5.0f));
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id0), window.get_node(id2), 3.0f));

    ASSERT_NE(window.marginalize_oldest(queue), graph::INVALID_NODE_ID);
    ASSERT_EQ(window.prior().node_ids.size(), 2U);
    EXPECT_EQ(window.prior().node_ids[0], id1);
    EXPECT_EQ(window.prior().node_ids[1], id2);
    EXPECT_GT((window.prior().H_prior.block<6, 6>(0, 6).norm()), 1e-3f);
}

TEST_F(GraphSlidingWindowTest, ExternalKeyframeDecisionKeepsOrDropsTip) {
    graph::GraphOptimization::Options opts;
    opts.gate.enabled = true;
    opts.gate.external_decision = true;
    graph::GraphOptimization optimizer(queue, graph::GraphSolverParams(), 5, opts);

    graph::GraphOptimization::FrameResult dropped;
    dropped.current_node_id = optimizer.window().add_node(Eigen::Isometry3f::Identity(), 0.0);
    optimizer.finalize_frame(dropped, false);
    EXPECT_TRUE(dropped.finalized);
    EXPECT_FALSE(dropped.keyframe);
    EXPECT_EQ(optimizer.window().window_size(), 0U);

    graph::GraphOptimization::FrameResult kept;
    kept.current_node_id = optimizer.window().add_node(Eigen::Isometry3f::Identity(), 1.0);
    optimizer.finalize_frame(kept, true);
    EXPECT_TRUE(kept.finalized);
    EXPECT_TRUE(kept.keyframe);
    EXPECT_EQ(optimizer.window().window_size(), 1U);
}

// ---------------------------------------------------------------------------
// Solver convergence with synthetic anchors (host-only)
// ---------------------------------------------------------------------------

class GraphSolverTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue = make_queue();
};

TEST_F(GraphSolverTest, ConvergesToAnchorTargets) {
    graph::SlidingWindow window(5);
    Eigen::Isometry3f t0 = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f t1 = Eigen::Isometry3f::Identity();
    t1.translate(Eigen::Vector3f(0.3f, 0.0f, 0.0f));
    Eigen::Isometry3f t2 = Eigen::Isometry3f::Identity();
    t2.rotate(Eigen::AngleAxisf(0.2f, Eigen::Vector3f::UnitZ()));

    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);

    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id0), t0, 10.0f));
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id1), t1, 10.0f));
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id2), t2, 10.0f));

    graph::GraphSolver solver(queue);
    auto result = solver.optimize(window);

    EXPECT_TRUE(result.converged);
    expect_pose_near(window.get_node(id0)->pose, t0, 1e-3f, 1e-3f);
    expect_pose_near(window.get_node(id1)->pose, t1, 1e-3f, 1e-3f);
    expect_pose_near(window.get_node(id2)->pose, t2, 1e-3f, 1e-3f);
}

TEST_F(GraphSolverTest, AnchorConstrainsNode) {
    graph::SlidingWindow window(5);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    window.add_node(Eigen::Isometry3f::Identity(), 1.0);

    Eigen::Isometry3f prior_pose = Eigen::Isometry3f::Identity();
    prior_pose.translate(Eigen::Vector3f(0.5f, 0.0f, 0.0f));

    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id0), prior_pose, 50.0f));
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(1), Eigen::Isometry3f::Identity(), 1.0f));

    graph::GraphSolver solver(queue);
    solver.optimize(window);

    expect_pose_near(window.get_node(id0)->pose, prior_pose, 1e-3f, 1e-3f);
}

TEST_F(GraphSolverTest, RejectsNonFiniteSystemWithoutUpdatingPose) {
    graph::SlidingWindow window(5);
    Eigen::Isometry3f initial = Eigen::Isometry3f::Identity();
    initial.translate(Eigen::Vector3f(0.3f, -0.2f, 0.1f));
    const graph::NodeId id = window.add_node(initial, 0.0);
    window.add_factor(std::make_shared<NonFiniteFactor>(window.get_node(id)));

    graph::GraphSolver solver(queue);
    const auto result = solver.optimize(window);

    EXPECT_FALSE(result.converged);
    EXPECT_FALSE(result.valid());
    EXPECT_EQ(result.status, graph::GraphSolver::Status::NON_FINITE_SYSTEM);
    EXPECT_TRUE(window.get_node(id)->pose.matrix().isApprox(initial.matrix()));
}

TEST_F(GraphSolverTest, SolvesDenseMarginalizationPrior) {
    graph::SlidingWindow window(2);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    Eigen::Isometry3f pose1 = Eigen::Isometry3f::Identity();
    pose1.translate(Eigen::Vector3f(0.2f, 0.0f, 0.0f));
    Eigen::Isometry3f pose2 = Eigen::Isometry3f::Identity();
    pose2.translate(Eigen::Vector3f(-0.1f, 0.0f, 0.0f));
    const graph::NodeId id1 = window.add_node(pose1, 1.0);
    const graph::NodeId id2 = window.add_node(pose2, 2.0);
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id0), Eigen::Isometry3f::Identity(), 10.0f));
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id0), window.get_node(id1), 5.0f));
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id0), window.get_node(id2), 3.0f));

    ASSERT_NE(window.marginalize_oldest(queue), graph::INVALID_NODE_ID);
    const auto result = graph::GraphSolver(queue).optimize(window);

    EXPECT_TRUE(result.valid());
    EXPECT_TRUE(result.converged);
    expect_pose_near(window.get_node(id1)->pose, Eigen::Isometry3f::Identity(), 1e-3f, 1e-3f);
    expect_pose_near(window.get_node(id2)->pose, Eigen::Isometry3f::Identity(), 1e-3f, 1e-3f);
}

TEST_F(GraphSolverTest, HonorsPerCallIterationLimit) {
    graph::SlidingWindow window(5);
    const graph::NodeId id = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    Eigen::Isometry3f target = Eigen::Isometry3f::Identity();
    target.translate(Eigen::Vector3f(1.0f, 0.0f, 0.0f));
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id), target, 1.0f));

    graph::GraphSolverParams params;
    params.max_iterations = 8;
    params.convergence_rotation = -1.0f;
    params.convergence_translation = -1.0f;
    const auto result = graph::GraphSolver(queue, params).optimize(window, 2.0f, 2U);

    EXPECT_EQ(result.iterations, 2U);
    EXPECT_FALSE(result.converged);
}

// ---------------------------------------------------------------------------
// End-to-end GICP (real factors, SYCL)
// ---------------------------------------------------------------------------

class GraphGicpTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue = make_queue();
    const size_t n_points = 3000;
    const float half = 1.0f;

    struct CloudBundle {
        PointCloudShared::Ptr cloud;
        std::shared_ptr<knn::KNNBase> knn;
    };

    CloudBundle build_bundle(const PointCloudShared::Ptr& c) {
        CloudBundle b;
        b.cloud = c;
        b.knn = knn::KDTree::build(queue, *c);
        estimate_covariances(*b.knn, *b.cloud);
        return b;
    }
};

TEST_F(GraphGicpTest, UnaryRecoversKnownTransform) {
    std::mt19937 gen(42);
    auto submap = make_cube_cloud(queue, n_points, half, gen);
    auto submap_bundle = build_bundle(submap);

    // Scan = submap expressed in a frame displaced by T_gt (scan = T_gt^{-1} * submap),
    // so the factor should recover T_gt.
    Eigen::Isometry3f T_gt = Eigen::Isometry3f::Identity();
    T_gt.translate(Eigen::Vector3f(0.15f, 0.0f, 0.0f));
    T_gt.rotate(Eigen::AngleAxisf(0.05f, Eigen::Vector3f::UnitZ()));
    auto scan = build_bundle(transform_cloud(queue, *submap, T_gt.inverse()));

    graph::SlidingWindow window(5);
    const graph::NodeId id = window.add_node(Eigen::Isometry3f::Identity(), 0.0, scan.cloud, scan.knn);
    auto node = window.get_node(id);

    auto params = gicp_params();
    window.add_factor(std::make_shared<graph::UnaryGicpFactor>(queue, id, node, submap_bundle.cloud,
                                                               submap_bundle.knn, params));

    graph::GraphSolver solver(queue);
    auto result = solver.optimize(window);
    EXPECT_TRUE(result.converged);

    expect_pose_near(window.get_node(id)->pose, T_gt, 0.05f, 0.05f);
}

// Marginalization must linearize factors with the RAW (unweighted) Hessian so the
// Schur-complement prior stays well conditioned. A strongly reweighted loss (GM at
// a tiny scale) collapses the robust Hessian; the raw path must recover the full
// information and match an explicitly NONE-typed factor.
TEST_F(GraphGicpTest, MarginalizationKeepsFrozenRobustWeights) {
    std::mt19937 gen(11);
    auto submap = make_cube_cloud(queue, n_points, half, gen);
    auto submap_bundle = build_bundle(submap);

    Eigen::Isometry3f T_gt = Eigen::Isometry3f::Identity();
    T_gt.translate(Eigen::Vector3f(0.4f, 0.0f, 0.0f));
    auto scan = build_bundle(transform_cloud(queue, *submap, T_gt.inverse()));

    // Align the node to a *wrong* pose (identity): the true offset is T_gt, so the
    // GICP residuals are far larger than a tiny robust scale and GM down-weights
    // the factor. Marginalization at the frozen scale must keep that behavior.
    graph::SlidingWindow window(5);
    const graph::NodeId id = window.add_node(Eigen::Isometry3f::Identity(), 0.0, scan.cloud, scan.knn);
    auto node = window.get_node(id);

    auto gm_params = gicp_params();
    gm_params.robust.type = robust::RobustLossType::GEMAN_MCCLURE;

    graph::UnaryGicpFactor factor(queue, id, node, submap_bundle.cloud, submap_bundle.knn, gm_params);
    const float tiny_scale = 0.05f;
    auto robust_lin = factor.linearize(queue, tiny_scale);

    // Small scale drives GM weights toward 0, so the robust Hessian stays below
    // the unweighted one: the outlier is not promoted back into the prior.
    auto none_params = gicp_params();
    none_params.robust.type = robust::RobustLossType::NONE;
    graph::UnaryGicpFactor none_factor(queue, id, node, submap_bundle.cloud, submap_bundle.knn,
                                       none_params);
    auto none_lin = none_factor.linearize(queue, tiny_scale);

    EXPECT_GT(none_lin.H00.trace(), 2.0f * robust_lin.H00.trace());
    // Same correspondences, only the per-point weight changes.
    EXPECT_EQ(none_lin.inlier, robust_lin.inlier);
    EXPECT_GT(robust_lin.inlier, 1500u);
}

TEST_F(GraphGicpTest, MarginalizationConsistency) {
    std::mt19937 gen(7);
    auto submap = make_cube_cloud(queue, n_points, half, gen);
    auto submap_bundle = build_bundle(submap);

    Eigen::Isometry3f T_rel = Eigen::Isometry3f::Identity();
    T_rel.translate(Eigen::Vector3f(0.2f, 0.0f, 0.0f));
    // node1's scan is submap in node1's frame (scan = T_rel^{-1} * submap) so the binary
    // factor (source=node1, target=node0=submap) recovers T_rel for node1.
    auto scan1 = build_bundle(transform_cloud(queue, *submap, T_rel.inverse()));

    auto build_problem = [&]() {
        graph::SlidingWindow w(1);  // max 1 -> 2 nodes triggers marginalization of id0
        const graph::NodeId id0 = w.add_node(Eigen::Isometry3f::Identity(), 0.0, submap_bundle.cloud,
                                              submap_bundle.knn);
        const graph::NodeId id1 = w.add_node(T_rel, 1.0, scan1.cloud, scan1.knn);
        auto n0 = w.get_node(id0);
        auto n1 = w.get_node(id1);

        auto params = gicp_params();
        // Anchor node0 to identity via a unary factor against the submap.
        w.add_factor(std::make_shared<graph::UnaryGicpFactor>(queue, id0, n0, submap_bundle.cloud,
                                                              submap_bundle.knn, params));
        // Couple node1 to node0.
        w.add_factor(std::make_shared<graph::BinaryGicpFactor>(queue, id1, n1, id0, n0, params));
        return w;
    };

    // Full solve keeps both nodes.
    auto win_full = build_problem();
    graph::GraphSolver(queue).optimize(win_full);
    Eigen::Isometry3f full_n1 = win_full.get_node(1)->pose;

    // Marginalized solve: optimize, drop the oldest node (node0), re-optimize the
    // single remaining node (node1) which now carries the Schur prior.
    auto win_marg = build_problem();
    graph::GraphSolver solver(queue);
    solver.optimize(win_marg);
    const graph::NodeId marginalized = win_marg.marginalize_oldest(queue);
    ASSERT_NE(marginalized, graph::INVALID_NODE_ID);
    EXPECT_EQ(win_marg.get_node(marginalized), nullptr);
    solver.optimize(win_marg);
    Eigen::Isometry3f marg_n1 = win_marg.get_node(1)->pose;

    // With a single remaining node the star-shaped Schur prior is exact, so the
    // recovered pose must match the full solve.
    expect_pose_near(full_n1, marg_n1, 1e-3f, 1e-3f);
    expect_pose_near(marg_n1, T_rel, 0.05f, 0.05f);
}

// ---------------------------------------------------------------------------
// Delayed relinearization: cache reuse (host-only; no SYCL kernels needed)
// ---------------------------------------------------------------------------

// Synthetic factor that runs a trivial CPU "linearization" and counts how many
// times linearize() is invoked. Reuse is decided by the real threshold check
// (relinearization_needed) so we can assert the base-class cache semantics.
class CountingGicpFactor : public graph::GicpFactorBase {
public:
    CountingGicpFactor(std::shared_ptr<graph::PoseNode> node, float rot_th, float trans_th)
        : node_(std::move(node)), rot_th_(rot_th), trans_th_(trans_th) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        ++linearize_calls;
        node_->linearization_pose = node_->pose;  // mirror real factors
        graph::FactorLinearization lin;
        lin.source_linearization_pose = node_->pose;
        lin.H00.setIdentity();
        lin.b0.setZero();
        lin.error = 0.0f;
        lin.inlier = 1;
        return lin;
    }

    std::pair<graph::NodeId, graph::NodeId> node_ids() const override { return {node_->id, graph::INVALID_NODE_ID}; }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&, const Eigen::Isometry3f&) const override {
        return {0.0f, 1};
    }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float, float) const override {
        return graph::relinearization_needed(node_->pose, node_->linearization_pose, rot_th_, trans_th_);
    }

    int linearize_calls = 0;

private:
    std::shared_ptr<graph::PoseNode> node_;
    float rot_th_, trans_th_;
};

class GraphCacheTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue = make_queue();
};

TEST_F(GraphCacheTest, RelinearizationThresholdLogic) {
    const Eigen::Isometry3f a = Eigen::Isometry3f::Identity();
    EXPECT_FALSE(graph::relinearization_needed(a, a, 0.02f, 0.05f));

    Eigen::Isometry3f b = Eigen::Isometry3f::Identity();
    b.translate(Eigen::Vector3f(0.01f, 0.0f, 0.0f));  // 0.01 m < 0.05 m
    EXPECT_FALSE(graph::relinearization_needed(a, b, 0.02f, 0.05f));

    Eigen::Isometry3f c = Eigen::Isometry3f::Identity();
    c.translate(Eigen::Vector3f(0.1f, 0.0f, 0.0f));  // 0.1 m > 0.05 m
    EXPECT_TRUE(graph::relinearization_needed(a, c, 0.02f, 0.05f));

    Eigen::Isometry3f d = Eigen::Isometry3f::Identity();
    d.rotate(Eigen::AngleAxisf(0.01f, Eigen::Vector3f::UnitZ()));  // 0.01 rad < 0.02 rad
    EXPECT_FALSE(graph::relinearization_needed(a, d, 0.02f, 0.05f));

    Eigen::Isometry3f e = Eigen::Isometry3f::Identity();
    e.rotate(Eigen::AngleAxisf(0.1f, Eigen::Vector3f::UnitZ()));  // 0.1 rad > 0.02 rad
    EXPECT_TRUE(graph::relinearization_needed(a, e, 0.02f, 0.05f));
}

TEST_F(GraphCacheTest, ReusesLinearizationUntilPoseMovesBeyondThreshold) {
    graph::SlidingWindow window(5);
    const auto id = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    auto node = window.get_node(id);
    CountingGicpFactor f(node, 0.02f, 0.05f);

    // First call must linearize.
    auto lin1 = f.get_linearization(queue, 0.02f, 0.05f);
    EXPECT_EQ(f.linearize_calls, 1);

    // Same pose -> reuse cached result (no new linearize).
    auto lin2 = f.get_linearization(queue, 0.02f, 0.05f);
    EXPECT_EQ(f.linearize_calls, 1);
    EXPECT_EQ(lin1.source_linearization_pose.matrix(), lin2.source_linearization_pose.matrix());

    // Move pose beyond threshold; linearization_pose stays stale -> must re-linearize.
    Eigen::Isometry3f moved = Eigen::Isometry3f::Identity();
    moved.translate(Eigen::Vector3f(0.2f, 0.0f, 0.0f));
    node->pose = moved;
    auto lin3 = f.get_linearization(queue, 0.02f, 0.05f);
    EXPECT_EQ(f.linearize_calls, 2);
    // After re-linearize, linearization_pose must track the new pose.
    expect_pose_near(node->linearization_pose, node->pose, 1e-6f, 1e-6f);

    // Pose unchanged since last relinearize -> reuse again.
    auto lin4 = f.get_linearization(queue, 0.02f, 0.05f);
    EXPECT_EQ(f.linearize_calls, 2);

    // clear_cache forces a re-linearize.
    f.clear_cache();
    auto lin5 = f.get_linearization(queue, 0.02f, 0.05f);
    EXPECT_EQ(f.linearize_calls, 3);
}

// ---------------------------------------------------------------------------
// RelativePoseFactor: independent Jacobian / information check + chain solve
// ---------------------------------------------------------------------------

// A 6D right-perturbation of an Isometry: T * exp(v).
static Eigen::Isometry3f rright(const Eigen::Isometry3f& T, const Eigen::Matrix<float, 6, 1>& v) {
    return Eigen::Isometry3f(T.matrix() * eigen_utils::lie::se3_exp(v));
}

class RelativePoseTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue = make_queue();
};

// Numerically differentiate the residual of a RelativePoseFactor and confirm
// the assembled b = J^T Omega r and H = J^T Omega J match what linearize()
// returns. This independently validates the -Adjoint(Tt^-1 Ts) source Jacobian
// (a plain -I would fail the H00 test whenever G has an off-axis translation).
TEST_F(RelativePoseTest, JacobianAndInformationAreConsistent) {
    auto win_src = std::make_shared<graph::PoseNode>();
    auto win_tgt = std::make_shared<graph::PoseNode>();
    win_src->id = 0;
    win_tgt->id = 1;

    Eigen::Isometry3f Ts = Eigen::Isometry3f::Identity();
    Ts.translate(Eigen::Vector3f(0.2f, -0.1f, 0.05f));
    Ts.rotate(Eigen::AngleAxisf(0.13f, Eigen::Vector3f::UnitZ()));
    Eigen::Isometry3f Tt = Eigen::Isometry3f::Identity();
    Tt.translate(Eigen::Vector3f(0.5f, 0.3f, -0.2f));
    Tt.rotate(Eigen::AngleAxisf(-0.21f, Eigen::Vector3f::UnitY()));
    Eigen::Isometry3f G = Eigen::Isometry3f::Identity();
    G.translate(Eigen::Vector3f(0.35f, 0.4f, -0.25f));
    G.rotate(Eigen::AngleAxisf(0.07f, Eigen::Vector3f(0.3f, 0.6f, 0.7f).normalized()));

    win_src->pose = Ts;
    win_tgt->pose = Tt;

    graph::RelativePoseParams rp;
    graph::RelativePoseFactor f(0, win_src, 1, win_tgt, G, rp);
    const Eigen::Matrix<float, 6, 6> Omega = graph::RelativePoseFactor::make_information(rp);

    auto residual = [&](const Eigen::Isometry3f& s, const Eigen::Isometry3f& t) {
        return eigen_utils::lie::se3_log(G.inverse() * (s.inverse() * t));
    };
    const Eigen::Matrix<float, 6, 1> r0 = residual(Ts, Tt);

    // Numerical Jacobians d r / d(right perturbation) at the linearization pose.
    const float eps = 1e-3f;
    Eigen::Matrix<float, 6, 6> Js = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 6> Jt = Eigen::Matrix<float, 6, 6>::Zero();
    for (int k = 0; k < 6; ++k) {
        Eigen::Matrix<float, 6, 1> e = Eigen::Matrix<float, 6, 1>::Zero();
        e[k] = eps;
        Js.col(k) = (residual(rright(Ts, e), Tt) - residual(rright(Ts, -e), Tt)) / (2 * eps);
        Jt.col(k) = (residual(Ts, rright(Tt, e)) - residual(Ts, rright(Tt, -e))) / (2 * eps);
    }

    const Eigen::Matrix<float, 6, 1> b0_num = Js.transpose() * Omega * r0;
    const Eigen::Matrix<float, 6, 1> b1_num = Jt.transpose() * Omega * r0;
    const Eigen::Matrix<float, 6, 6> H00_num = Js.transpose() * Omega * Js;
    const Eigen::Matrix<float, 6, 6> H01_num = Js.transpose() * Omega * Jt;
    const Eigen::Matrix<float, 6, 6> H11_num = Jt.transpose() * Omega * Jt;

    auto lin = f.linearize(queue);
    auto rel_err = [](const Eigen::Matrix<float, 6, 1>& a, const Eigen::Matrix<float, 6, 1>& b) {
        return (a - b).norm() / std::max(1.0f, a.norm());
    };
    auto rel_err_m = [](const Eigen::Matrix<float, 6, 6>& a, const Eigen::Matrix<float, 6, 6>& b) {
        return (a - b).norm() / std::max(1.0f, a.norm());
    };
    EXPECT_LT(rel_err(lin.b0, b0_num), 2e-2f);   // float32 FD noise ~1%
    EXPECT_LT(rel_err(lin.b1, b1_num), 2e-2f);
    EXPECT_LT(rel_err_m(lin.H00, H00_num), 2e-2f);
    EXPECT_LT(rel_err_m(lin.H01, H01_num), 2e-2f);
    EXPECT_LT(rel_err_m(lin.H11, H11_num), 2e-2f);

    // Sanity: a plain -I source Jacobian (ignoring the adjoint) must NOT match,
    // otherwise the adjoint term is doing nothing and the test is vacuous.
    const Eigen::Matrix<float, 6, 6> H00_naive = Eigen::Matrix<float, 6, 6>::Identity() * Omega;
    EXPECT_GT(rel_err_m(lin.H00, H00_naive), 1e-2f);
}

// A 3-node chain (anchor T0, relative T0->T1, relative T1->T2) with exact
// measurements must be recovered, and needs_relinearization() is always true
// so the base cache never goes stale for this factor.
TEST_F(RelativePoseTest, ChainRecoversWithAnchor) {
    graph::SlidingWindow window(5);
    Eigen::Isometry3f T0 = Eigen::Isometry3f::Identity();
    T0.translate(Eigen::Vector3f(1.0f, 0.5f, 0.0f));
    Eigen::Isometry3f G01 = Eigen::Isometry3f::Identity();
    G01.translate(Eigen::Vector3f(0.5f, 0.0f, 0.0f));
    G01.rotate(Eigen::AngleAxisf(0.1f, Eigen::Vector3f::UnitZ()));
    Eigen::Isometry3f G12 = Eigen::Isometry3f::Identity();
    G12.translate(Eigen::Vector3f(0.0f, 0.4f, 0.0f));
    G12.rotate(Eigen::AngleAxisf(-0.15f, Eigen::Vector3f::UnitZ()));
    const Eigen::Isometry3f T1 = T0 * G01;
    const Eigen::Isometry3f T2 = T1 * G12;

    const graph::NodeId id0 = window.add_node(T0, 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);

    // Anchor node 0 so the chain has an absolute reference.
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id0), T0, 100.0f));
    window.add_factor(std::make_shared<graph::RelativePoseFactor>(
        id0, window.get_node(id0), id1, window.get_node(id1), G01));
    window.add_factor(std::make_shared<graph::RelativePoseFactor>(
        id1, window.get_node(id1), id2, window.get_node(id2), G12));

    graph::GraphSolver solver(queue);
    auto result = solver.optimize(window);
    EXPECT_TRUE(result.converged);
    expect_pose_near(window.get_node(id1)->pose, T1, 1e-2f, 1e-2f);
    expect_pose_near(window.get_node(id2)->pose, T2, 1e-2f, 1e-2f);
}

// ---------------------------------------------------------------------------
// Topology invariants: sparse_chain vs clique (real GICP factors, SYCL)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Topology invariants: sparse_chain vs clique (real GICP factors, SYCL)
// ---------------------------------------------------------------------------

class GraphTopologyTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue = make_queue();

    graph::GraphOptimization::FrameResult feed_frame(graph::GraphOptimization& opt,
                                                     const PointCloudShared::Ptr& submap,
                                                     const std::shared_ptr<knn::KNNBase>& submap_knn,
                                                     const Eigen::Isometry3f& T_gt, double t) {
        auto scan = transform_cloud(queue, *submap, T_gt.inverse());
        auto knn = knn::KDTree::build(queue, *scan);
        estimate_covariances(*knn, *scan);
        return opt.process_frame(scan, submap, submap_knn, knn, T_gt, t, gicp_params());
    }

    struct TopoCounts {
        size_t pc_binary = 0;
        size_t chain = 0;
        size_t unary = 0;
    };

    static TopoCounts count_factors(const graph::SlidingWindow& w) {
        TopoCounts c;
        for (auto& f : w.factors()) {
            if (f->is_point_cloud_binary()) {
                ++c.pc_binary;
            } else if (std::dynamic_pointer_cast<const graph::RelativePoseFactor>(f)) {
                ++c.chain;
            } else {
                ++c.unary;
            }
        }
        return c;
    }

    PointCloudShared::Ptr make_submap(std::mt19937& gen,
                                      std::shared_ptr<knn::KNNBase>& knn_out) {
        auto submap = make_cube_cloud(queue, 3000, 1.0f, gen);
        auto knn = knn::KDTree::build(queue, *submap);
        estimate_covariances(*knn, *submap);
        knn_out = knn;
        return submap;
    }
};

// In sparse_chain mode the point-cloud star must only ever touch the current
// tip, older adjacent pairs live on chain RelativePoseFactors, and the
// marginalization prior must anchor to the oldest surviving node.
TEST_F(GraphTopologyTest, SparseChainKeepsStarAtTipOnly) {
    std::mt19937 gen(7);
    std::shared_ptr<knn::KNNBase> submap_knn;
    auto submap = make_submap(gen, submap_knn);

    graph::GraphOptimization::Options opts;
    opts.binary_topology = graph::GraphOptimization::BinaryTopology::sparse_chain;
    graph::GraphOptimization opt(queue, graph::GraphSolverParams(), 4, opts);

    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f step = Eigen::Isometry3f::Identity();
    step.translate(Eigen::Vector3f(0.02f, 0.0f, 0.0f));
    step.rotate(Eigen::AngleAxisf(0.004f, Eigen::Vector3f::UnitZ()));

    for (size_t f = 0; f < 7; ++f) {
        T = T * step;
        auto fr = feed_frame(opt, submap, submap_knn, T, 0.1 * static_cast<double>(f));
        EXPECT_TRUE(fr.converged) << "frame " << f;
        expect_pose_near(fr.current_pose, T, 0.05f, 0.05f);

        auto& w = opt.window();
        if (f + 1 >= 4) {  // window full from here on
            const auto c = count_factors(w);
            EXPECT_EQ(c.pc_binary, w.window_size() - 1) << "frame " << f;
            EXPECT_EQ(c.chain, w.window_size() - 2) << "frame " << f;
            EXPECT_EQ(c.unary, w.window_size()) << "frame " << f;
            // every point-cloud binary touches the tip
            const graph::NodeId tip = w.active_nodes().back()->id;
            for (auto& fac : w.factors()) {
                if (fac->is_point_cloud_binary()) {
                    EXPECT_EQ(fac->node_ids().first, tip) << "frame " << f;
                }
            }
            if (w.prior().is_valid()) {
                EXPECT_EQ(w.prior().node_ids[0], w.active_nodes().front()->id) << "frame " << f;
            }
        }
    }
}

// Clique mode must keep the legacy structure (all pairs of point-cloud
// binaries, no chain factors, prior anchored to the newest node).
TEST_F(GraphTopologyTest, CliqueModeKeepsLegacyStructure) {
    std::mt19937 gen(7);
    std::shared_ptr<knn::KNNBase> submap_knn;
    auto submap = make_submap(gen, submap_knn);

    graph::GraphOptimization::Options opts;
    opts.binary_topology = graph::GraphOptimization::BinaryTopology::clique;
    graph::GraphOptimization opt(queue, graph::GraphSolverParams(), 4, opts);

    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f step = Eigen::Isometry3f::Identity();
    step.translate(Eigen::Vector3f(0.02f, 0.0f, 0.0f));
    step.rotate(Eigen::AngleAxisf(0.004f, Eigen::Vector3f::UnitZ()));

    for (size_t f = 0; f < 7; ++f) {
        T = T * step;
        auto fr = feed_frame(opt, submap, submap_knn, T, 0.1 * static_cast<double>(f));
        EXPECT_TRUE(fr.converged) << "frame " << f;

        auto& w = opt.window();
        if (f + 1 >= 4) {
            const size_t K = w.window_size();
            const auto c = count_factors(w);
            EXPECT_EQ(c.pc_binary, K * (K - 1) / 2) << "frame " << f;
            EXPECT_EQ(c.chain, 0u) << "frame " << f;
            if (w.prior().is_valid()) {
                for (const auto& node : w.active_nodes()) {
                    EXPECT_NE(std::find(w.prior().node_ids.begin(), w.prior().node_ids.end(), node->id),
                              w.prior().node_ids.end()) << "frame " << f;
                }
            }
        }
    }
}

// Keyframe gate: with small steps and a gate that fires only on accumulated
// motion, far fewer nodes persist than frames, yet the current pose must still
// track the (submap-anchored) trajectory.
TEST_F(GraphTopologyTest, KeyframeGateThinsPersistentNodes) {
    std::mt19937 gen(7);
    std::shared_ptr<knn::KNNBase> submap_knn;
    auto submap = make_submap(gen, submap_knn);

    graph::GraphOptimization::Options opts;
    opts.binary_topology = graph::GraphOptimization::BinaryTopology::sparse_chain;
    opts.gate.enabled = true;
    opts.gate.min_translation = 0.05f;  // fires every ~3rd frame at 0.02 steps
    graph::GraphOptimization opt(queue, graph::GraphSolverParams(), 8, opts);

    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f step = Eigen::Isometry3f::Identity();
    step.translate(Eigen::Vector3f(0.02f, 0.0f, 0.0f));

    size_t keyframes = 0;
    for (size_t f = 0; f < 9; ++f) {
        T = T * step;
        auto fr = feed_frame(opt, submap, submap_knn, T, 0.1 * static_cast<double>(f));
        EXPECT_TRUE(fr.converged) << "frame " << f;
        expect_pose_near(fr.current_pose, T, 0.05f, 0.05f);
        if (fr.keyframe) ++keyframes;
    }
    // 9 frames but only ~4 should have persisted as keyframes.
    EXPECT_GT(keyframes, 1u);
    EXPECT_LT(keyframes, 9u);
    EXPECT_LE(opt.window().window_size(), keyframes);

    // Persistent graph is pure keyframes + chain (no stale point-cloud star
    // left after a transient tip is dropped): binaries <= nodes-1.
    const auto c = count_factors(opt.window());
    EXPECT_LE(c.pc_binary + c.chain, opt.window().window_size() * (opt.window().window_size() - 1) / 2);
}

// ---------------------------------------------------------------------------
// Binary factor RegType coverage (linearize_binary<reg>)
// ---------------------------------------------------------------------------

class GraphBinaryRegTypesTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue = make_queue();
};

TEST_F(GraphBinaryRegTypesTest, SupportedTypesProduceSaneFactors) {
    std::mt19937 gen(5);
    auto cloud = make_cube_cloud(queue, 2000, 0.8f, gen);
    auto knn = knn::KDTree::build(queue, *cloud);
    estimate_covariances(*knn, *cloud);

    const registration::RegType types[] = {
        registration::RegType::POINT_TO_POINT, registration::RegType::POINT_TO_PLANE,
        registration::RegType::POINT_TO_DISTRIBUTION, registration::RegType::GICP};

    for (const auto type : types) {
        registration::RegistrationParams params = gicp_params();
        params.reg_type = type;
        graph::BinaryGicpLinearizer linearizer(queue, params);
        const int ti = static_cast<int>(type);

        // True relative pose: ~zero residual, full inliers, positive information.
        auto at_true = linearizer.linearize(*cloud, *knn, Eigen::Matrix4f::Identity(), *cloud,
                                            Eigen::Matrix4f::Identity());
        EXPECT_GT(at_true.inlier, 1500u) << "type=" << ti;
        EXPECT_LT(at_true.error, 1e-2f) << "type=" << ti;
        EXPECT_GT(at_true.H00.trace(), 0.0f) << "type=" << ti;

        // Wrong relative pose: error and gradient must grow.
        Eigen::Isometry3f pert = Eigen::Isometry3f::Identity();
        pert.translate(Eigen::Vector3f(0.15f, 0.0f, 0.0f));
        auto at_err = linearizer.linearize(*cloud, *knn, Eigen::Matrix4f::Identity(), *cloud, pert.matrix());
        EXPECT_GT(at_err.error, at_true.error) << "type=" << ti;
        EXPECT_GT(at_err.b0.norm(), 1e-2f) << "type=" << ti;
        // H00 must stay symmetric PSD-ish: diagonal blocks positive.
        EXPECT_GT(at_err.H00(0, 0), 0.0f) << "type=" << ti;
        EXPECT_GT(at_err.H11(3, 3), 0.0f) << "type=" << ti;
    }
}

TEST_F(GraphBinaryRegTypesTest, GenzIsRejectedForBinaryFactors) {
    std::mt19937 gen(5);
    auto cloud = make_cube_cloud(queue, 1000, 0.8f, gen);
    auto knn = knn::KDTree::build(queue, *cloud);
    estimate_covariances(*knn, *cloud);

    registration::RegistrationParams params = gicp_params();
    params.reg_type = registration::RegType::GENZ;
    graph::BinaryGicpLinearizer linearizer(queue, params);
    EXPECT_THROW(linearizer.linearize(*cloud, *knn, Eigen::Matrix4f::Identity(), *cloud,
                                      Eigen::Matrix4f::Identity()),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// Robust scale ladder (GNC)
// ---------------------------------------------------------------------------

TEST(RobustScheduleTest, LadderDescendsToFloor) {
    graph::GraphSolverParams::RobustSchedule r;
    // disabled => 0 (factors fall back to their configured default)
    EXPECT_FLOAT_EQ(graph::robust_ladder_scale(r, 0), 0.0f);

    r.enable = true;  // init 10 -> min 1.25, 4 levels x 2 iters: 10,5,2.5,1.25
    EXPECT_FLOAT_EQ(graph::robust_ladder_scale(r, 0), 10.0f);
    EXPECT_FLOAT_EQ(graph::robust_ladder_scale(r, 2), 5.0f);
    EXPECT_FLOAT_EQ(graph::robust_ladder_scale(r, 4), 2.5f);
    EXPECT_FLOAT_EQ(graph::robust_ladder_scale(r, 6), 1.25f);
    EXPECT_FLOAT_EQ(graph::robust_ladder_scale(r, 100), 1.25f);  // clamped at floor
    for (size_t k = 1; k <= 10; ++k) {
        EXPECT_LE(graph::robust_ladder_scale(r, k), graph::robust_ladder_scale(r, k - 1));
    }
}

// Synthetic factor that records the scale it is asked to linearize with.
class ScaleProbeFactor : public graph::GicpFactorBase {
public:
    ScaleProbeFactor() { begin_annealing(); }

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float scale) override {
        seen_scales.push_back(scale);
        graph::FactorLinearization lin;
        lin.H00.setIdentity();
        return lin;
    }
    std::pair<graph::NodeId, graph::NodeId> node_ids() const override {
        return {0, graph::INVALID_NODE_ID};
    }
    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&,
                                             const Eigen::Isometry3f&) const override {
        return {0.0f, 1};
    }
    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return true;  // always relinearize so every requested scale is observable
    }
    std::vector<float> seen_scales;
};

TEST(RobustScheduleTest, FreezeLocksLastUsedScale) {
    sycl_utils::DeviceQueue queue = make_queue();
    ScaleProbeFactor f;
    f.get_linearization(queue, 0.02f, 0.05f, 5.0f);
    f.freeze();
    // After freeze the ladder value is ignored; the locked scale is reused.
    f.get_linearization(queue, 0.02f, 0.05f, 1.0f);
    ASSERT_EQ(f.seen_scales.size(), 2u);
    EXPECT_FLOAT_EQ(f.seen_scales[0], 5.0f);
    EXPECT_FLOAT_EQ(f.seen_scales[1], 5.0f);
}

// Deterministic wiring proof: the solver's ladder reaches linearize() only at
// rung changes when relinearize_per_rung is set, and never when caches hold.
class CachedProbeFactor : public graph::GicpFactorBase {
public:
    CachedProbeFactor() { begin_annealing(); }

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float scale) override {
        seen_scales.push_back(scale);
        graph::FactorLinearization lin;
        lin.H00.setIdentity();
        // b0 is evaluated at pose=identity while the solver keeps moving the node
        // away from it -> the gradient correction g = b + H*delta never vanishes,
        // so the GN loop uses all 8 iterations (2 per rung x 4 rungs).
        lin.b0 = Eigen::Matrix<float, 6, 1>::Constant(0.1f);
        lin.source_linearization_pose = Eigen::Isometry3f::Identity();
        return lin;
    }
    std::pair<graph::NodeId, graph::NodeId> node_ids() const override {
        return {0, graph::INVALID_NODE_ID};
    }
    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&,
                                             const Eigen::Isometry3f&) const override {
        return {0.0f, 1};
    }
    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return false;  // pose-based relinearization disabled: only the rung can force it
    }
    std::vector<float> seen_scales;
};

TEST(RobustScheduleTest, RungForceControlsRelinearizations) {
    sycl_utils::DeviceQueue queue = make_queue();
    graph::GraphSolverParams sp;
    sp.max_iterations = 8;  // exactly levels x iters_per_level
    sp.robust.enable = true;
    sp.robust.iters_per_level = 2;
    sp.robust.levels = 4;

    {  // force off: single linearization, cached weights sleep for the frame
        graph::SlidingWindow window(5);
        const graph::NodeId id = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
        (void)id;
        auto f = std::make_shared<CachedProbeFactor>();
        window.add_factor(f);
        graph::GraphSolver solver(queue, sp);
        f->set_robust_force_mode(false);
        solver.optimize(window);
        ASSERT_EQ(f->seen_scales.size(), 1u);
        EXPECT_FLOAT_EQ(f->seen_scales[0], 10.0f);  // rung 0 at frame start
    }
    {  // force on: relinearize exactly once per rung (4 rungs in 8 iters)
        graph::SlidingWindow window(5);
        window.add_node(Eigen::Isometry3f::Identity(), 0.0);
        auto f = std::make_shared<CachedProbeFactor>();
        window.add_factor(f);
        graph::GraphSolver solver(queue, sp);
        f->set_robust_force_mode(true);
        solver.optimize(window);
        ASSERT_EQ(f->seen_scales.size(), 4u);
        EXPECT_FLOAT_EQ(f->seen_scales[0], 10.0f);
        EXPECT_FLOAT_EQ(f->seen_scales[1], 5.0f);
        EXPECT_FLOAT_EQ(f->seen_scales[2], 2.5f);
        EXPECT_FLOAT_EQ(f->seen_scales[3], 1.25f);
    }
}

// ---------------------------------------------------------------------------
// VelocityUpdate: tip-only constant-velocity deskew + re-solve
// ---------------------------------------------------------------------------

// Build a scan captured during constant-velocity motion: point i is the world
// point observed by a body at pose T_start * exp(tau * delta_twist) (tau cycles
// over [0, 1) via i % steps), with per-point timestamp offset tau*1000 ms.
PointCloudShared::Ptr make_distorted_scan(const sycl_utils::DeviceQueue& queue, const PointCloudShared& world,
                                          const Eigen::Isometry3f& T_start,
                                          const Eigen::Matrix<float, 6, 1>& delta_twist, size_t steps) {
    auto cloud = std::make_shared<PointCloudShared>(queue);
    const size_t n = world.size();
    cloud->points->resize(n);
    cloud->timestamp_offsets->resize(n);
    for (size_t i = 0; i < n; ++i) {
        const float tau = static_cast<float>(i % steps) / static_cast<float>(steps);
        const Eigen::Isometry3f pose(T_start.matrix() * eigen_utils::lie::se3_exp(delta_twist * tau));
        const Eigen::Vector4f p = pose.inverse().matrix() * world.points->at(i);
        (*cloud->points)[i] = PointType(p.x(), p.y(), p.z(), 1.0f);
        (*cloud->timestamp_offsets)[i] = static_cast<TimestampOffset>(tau * 1000.0f);
    }
    cloud->start_time_ms = 0.0;
    cloud->end_time_ms = 1000.0;
    return cloud;
}

class GraphVelocityUpdateTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue = make_queue();

    void estimate_covs(PointCloudShared& cloud) {
        auto knn = knn::KDTree::build(queue, cloud);
        covariance::estimate_async(*knn, cloud, 10).wait_and_throw();
    }

    // Pipeline analog: estimate covariances on the raw scan, then apply the
    // iter-0 deskew (kernel rotates normals/covs). Returns the deskewed copy.
    PointCloudShared::Ptr deskew_iter0(const PointCloudShared::Ptr& raw, const Eigen::Isometry3f& prev_pose,
                                       const Eigen::Isometry3f& init_pose, float dt) {
        estimate_covs(*raw);
        auto deskewed = std::make_shared<PointCloudShared>(queue);
        EXPECT_TRUE(
            deskew::deskew_point_cloud_constant_velocity(*raw, *deskewed, prev_pose, init_pose, dt));
        return deskewed;
    }
};

// Deskewing the tip scan with the (prev, refined) basis must recover the true
// tip pose; registering the raw distorted scan instead leaves a mid-scan bias.
TEST_F(GraphVelocityUpdateTest, DeskewRecoversUndistortedTipPose) {
    std::mt19937 gen(31);
    auto submap = make_cube_cloud(queue, 3000, 1.0f, gen);
    auto submap_knn = knn::KDTree::build(queue, *submap);
    estimate_covariances(*submap_knn, *submap);

    // Constant-velocity twist: 0.2 m + 0.05 rad per inter-frame interval.
    Eigen::Matrix<float, 6, 1> delta = Eigen::Matrix<float, 6, 1>::Zero();
    delta.head<3>() = Eigen::Vector3f(0.0f, 0.0f, 0.05f);
    delta.tail<3>() = Eigen::Vector3f(0.2f, 0.0f, 0.0f);
    const Eigen::Isometry3f T_prev = Eigen::Isometry3f::Identity();
    const Eigen::Isometry3f T_gt(T_prev.matrix() * eigen_utils::lie::se3_exp(delta));

    auto raw = make_distorted_scan(queue, *submap, T_gt, delta, 10);
    auto raw_knn = knn::KDTree::build(queue, *raw);
    estimate_covariances(*raw_knn, *raw);

    float err_on = -1.0f, err_off = -1.0f;
    {  // velocity update ON (2 deskew+re-solve rounds)
        graph::GraphOptimization opt(queue, graph::GraphSolverParams(), 4);
        auto deskewed = this->deskew_iter0(raw, T_prev, T_gt, 1.0f);
        graph::GraphOptimization::VelocityUpdateContext vu;
        vu.enable = true;
        vu.iterations = 2;
        vu.prev_pose = T_prev;
        vu.dt = 1.0f;
        vu.raw_source = raw;
        const auto fr = opt.process_frame(deskewed, submap, submap_knn, nullptr, T_gt, 1.0, gicp_params(), vu);
        err_on = (fr.current_pose.inverse() * T_gt).translation().norm();
        expect_pose_near(fr.current_pose, T_gt, 0.03f, 0.03f);
        ASSERT_NE(fr.tip_cloud, nullptr);
        // The tip cloud must be a deskewed snapshot, not the raw scan.
        // (Point 0 has offset tau=0: deskew is the identity there, so check a
        // mid-scan point.)
        EXPECT_GT(((*fr.tip_cloud->points)[5].head<3>() - (*raw->points)[5].head<3>()).norm(), 1e-3f);
        // Retained tip (gate disabled): deferred kNN built exactly once here.
        const auto tip = opt.window().active_nodes().back();
        ASSERT_NE(tip->knn, nullptr);
        EXPECT_EQ(tip->cloud.get(), fr.tip_cloud.get());
    }
    {  // velocity update OFF: distorted scan -> mid-scan bias
        graph::GraphOptimization opt(queue, graph::GraphSolverParams(), 4);
        const auto fr = opt.process_frame(raw, submap, submap_knn, raw_knn, T_gt, 1.0, gicp_params());
        err_off = (fr.current_pose.inverse() * T_gt).translation().norm();
        EXPECT_GT(err_off, 0.03f);
    }
    EXPECT_LT(err_on, 0.02f);
    EXPECT_GT(err_off, 2.0f * err_on);
}

// The velocity update must swap ONLY the tip's cloud; older nodes keep their
// cloud/knn pointers, and the tip's kNN is built once at retention (nullptr
// from the caller, built inside process_frame).
TEST_F(GraphVelocityUpdateTest, TipOnlyCloudSwapWithDeferredKnn) {
    std::mt19937 gen(7);
    auto submap = make_cube_cloud(queue, 3000, 1.0f, gen);
    auto submap_knn = knn::KDTree::build(queue, *submap);
    estimate_covariances(*submap_knn, *submap);

    graph::GraphOptimization opt(queue, graph::GraphSolverParams(), 4);
    Eigen::Matrix<float, 6, 1> delta = Eigen::Matrix<float, 6, 1>::Zero();
    delta.tail<3>() = Eigen::Vector3f(0.02f, 0.0f, 0.0f);
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();

    std::map<graph::NodeId, std::pair<const void*, const void*>> before;
    for (size_t f = 0; f < 3; ++f) {
        const Eigen::Isometry3f T_prev = T;
        T = Eigen::Isometry3f(T.matrix() * eigen_utils::lie::se3_exp(delta));
        auto raw = make_distorted_scan(queue, *submap, T, delta, 10);
        const size_t iters = (f == 0) ? 1 : 2;  // f=0: single deskew, no inner swap
        auto source = this->deskew_iter0(raw, T_prev, T, 1.0f);

        // Snapshot every existing node's cloud/knn pointers.
        for (const auto& node : opt.window().active_nodes()) {
            before[node->id] = {node->cloud.get(), node->knn.get()};
        }

        graph::GraphOptimization::VelocityUpdateContext vu;
        vu.enable = true;
        vu.iterations = iters;
        vu.prev_pose = T_prev;
        vu.dt = 1.0f;
        vu.raw_source = raw;
        const auto fr = opt.process_frame(source, submap, submap_knn, nullptr, T, 0.1 * static_cast<double>(f),
                                          gicp_params(), vu);

        expect_pose_near(fr.current_pose, T, 0.03f, 0.03f);
        ASSERT_NE(fr.tip_cloud, nullptr);
        const auto tip = opt.window().active_nodes().back();
        EXPECT_EQ(tip->cloud.get(), fr.tip_cloud.get());
        // Deferred kNN: caller passed nullptr, retention builds it once.
        ASSERT_NE(tip->knn, nullptr);
        if (iters == 1) {
            EXPECT_EQ(tip->cloud.get(), source.get());  // no inner round -> no swap
        } else {
            EXPECT_NE(tip->cloud.get(), source.get());  // inner round swapped the cloud
        }
        // Older nodes must be untouched (same pointers).
        for (const auto& [id, ptrs] : before) {
            const auto node = opt.window().get_node(id);
            ASSERT_NE(node, nullptr);
            EXPECT_EQ(node->cloud.get(), ptrs.first) << "frame " << f << " node " << id;
            EXPECT_EQ(node->knn.get(), ptrs.second) << "frame " << f << " node " << id;
        }
        before.clear();
    }
}

TEST_F(GraphVelocityUpdateTest, KeepsSampledPointSetAcrossRedeskewRounds) {
    std::mt19937 gen(71);
    auto submap = make_cube_cloud(queue, 1000, 1.0f, gen);
    auto submap_knn = knn::KDTree::build(queue, *submap);
    estimate_covariances(*submap_knn, *submap);

    Eigen::Matrix<float, 6, 1> delta = Eigen::Matrix<float, 6, 1>::Zero();
    delta.tail<3>() = Eigen::Vector3f(0.05f, 0.0f, 0.0f);
    const Eigen::Isometry3f previous_pose = Eigen::Isometry3f::Identity();
    const Eigen::Isometry3f current_pose(eigen_utils::lie::se3_exp(delta));
    auto raw = make_distorted_scan(queue, *submap, current_pose, delta, 10);
    this->estimate_covs(*raw);

    constexpr size_t sampled_size = 192;
    PointCloudCPU sampled_cpu;
    sampled_cpu.points->resize(sampled_size);
    sampled_cpu.covs->resize(sampled_size);
    sampled_cpu.timestamp_offsets->resize(sampled_size);
    for (size_t i = 0; i < sampled_size; ++i) {
        (*sampled_cpu.points)[i] = (*raw->points)[i];
        (*sampled_cpu.covs)[i] = (*raw->covs)[i];
        (*sampled_cpu.timestamp_offsets)[i] = (*raw->timestamp_offsets)[i];
    }
    sampled_cpu.start_time_ms = raw->start_time_ms;
    sampled_cpu.end_time_ms = raw->end_time_ms;
    auto sampled_raw = std::make_shared<PointCloudShared>(queue, sampled_cpu);
    auto sampled_deskewed = std::make_shared<PointCloudShared>(queue);
    ASSERT_TRUE(deskew::deskew_point_cloud_constant_velocity(
        *sampled_raw, *sampled_deskewed, previous_pose, current_pose, 1.0f));

    graph::GraphOptimization::VelocityUpdateContext vu;
    vu.enable = true;
    vu.iterations = 2;
    vu.prev_pose = previous_pose;
    vu.dt = 1.0f;
    vu.raw_source = sampled_raw;

    graph::GraphSolverParams solver_params;
    solver_params.max_iterations = 1;
    graph::GraphOptimization opt(queue, solver_params, 4);
    const auto fr = opt.process_frame(sampled_deskewed, submap, submap_knn, nullptr,
                                      current_pose, 1.0, gicp_params(), vu);

    ASSERT_NE(fr.tip_cloud, nullptr);
    EXPECT_EQ(fr.tip_cloud->size(), sampled_size);
    EXPECT_GE(fr.inlier_ratio, 0.0f);
    EXPECT_LE(fr.inlier_ratio, 1.0f);
    EXPECT_GT(fr.tip_registration.inlier, 0U);
    EXPECT_TRUE(fr.tip_registration.H_raw.allFinite());
    EXPECT_GT(fr.tip_registration.H_raw.trace(), 0.0f);
}

// A scan without per-point timestamps disables the update inside
// process_frame: the caller-provided cloud and kNN are used as-is.
TEST_F(GraphVelocityUpdateTest, NoTimestampsFallsBackToPlainPath) {
    std::mt19937 gen(41);
    auto submap = make_cube_cloud(queue, 3000, 1.0f, gen);
    auto submap_knn = knn::KDTree::build(queue, *submap);
    estimate_covariances(*submap_knn, *submap);
    auto scan = make_cube_cloud(queue, 3000, 1.0f, gen);  // no timestamps
    auto scan_knn = knn::KDTree::build(queue, *scan);
    estimate_covariances(*scan_knn, *scan);

    graph::GraphOptimization opt(queue, graph::GraphSolverParams(), 4);
    graph::GraphOptimization::VelocityUpdateContext vu;
    vu.enable = true;
    vu.iterations = 2;
    vu.dt = 1.0f;
    vu.raw_source = scan;  // has_timestamps() == false -> inactive
    const auto fr = opt.process_frame(scan, submap, submap_knn, scan_knn, Eigen::Isometry3f::Identity(), 1.0,
                                      gicp_params(), vu);
    EXPECT_EQ(fr.tip_cloud, nullptr);
    const auto tip = opt.window().active_nodes().back();
    EXPECT_EQ(tip->cloud.get(), scan.get());
    EXPECT_EQ(tip->knn.get(), scan_knn.get());
}

// The fixed-scale override must reach linearize() instead of the internal
// ladder (frame-level robust schedule driven by process_frame).
TEST(RobustScheduleTest, FixedScaleOverrideReachesLinearize) {
    sycl_utils::DeviceQueue queue = make_queue();
    graph::GraphSolverParams sp;
    sp.max_iterations = 8;
    sp.robust.enable = true;
    sp.robust.iters_per_level = 2;
    sp.robust.levels = 4;
    graph::GraphSolver solver(queue, sp);

    {  // no override: internal ladder, needs_relinearization=true keeps GN
        // running until the ladder reaches its floor (8 linearizations)
        graph::SlidingWindow window(5);
        window.add_node(Eigen::Isometry3f::Identity(), 0.0);
        auto f = std::make_shared<ScaleProbeFactor>();
        window.add_factor(f);
        solver.optimize(window);
        ASSERT_EQ(f->seen_scales.size(), 8u);
        EXPECT_FLOAT_EQ(f->seen_scales[0], 10.0f);
        EXPECT_FLOAT_EQ(f->seen_scales[2], 5.0f);
        EXPECT_FLOAT_EQ(f->seen_scales[4], 2.5f);
        EXPECT_FLOAT_EQ(f->seen_scales[6], 1.25f);
    }
    {  // override 3.0: the ladder (and its gating) is bypassed; converged on
        // iteration 0 at scale 3.0
        graph::SlidingWindow window(5);
        window.add_node(Eigen::Isometry3f::Identity(), 0.0);
        auto f = std::make_shared<ScaleProbeFactor>();
        window.add_factor(f);
        solver.optimize(window, 3.0f);
        ASSERT_EQ(f->seen_scales.size(), 1u);
        EXPECT_FLOAT_EQ(f->seen_scales[0], 3.0f);
    }
}

// TEMP repro: multi-frame run with HUBER + ladder + velocity update + keyframe gate
// Planar (noisy floor) environment: raw covariances are strongly anisotropic, so
// binary factors (no plane normalization) operate on near-singular RCR blocks.
PointCloudShared::Ptr make_noisy_plane_cloud(const sycl_utils::DeviceQueue& queue, size_t n, float extent,
                                             float noise, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-extent, extent);
    std::normal_distribution<float> nz(0.0f, noise);
    PointCloudCPU cpu;
    cpu.points->resize(n);
    for (size_t i = 0; i < n; ++i) {
        (*cpu.points)[i] = PointType(dist(gen), dist(gen), nz(gen), 1.0f);
    }
    return std::make_shared<PointCloudShared>(queue, cpu);
}

TEST(GraphReproTest, MultiFrameErrorStaysFinite) {
    sycl_utils::DeviceQueue queue = make_queue();
    std::mt19937 gen(17);
    auto submap = make_noisy_plane_cloud(queue, 20000, 5.0f, 0.005f, gen);
    auto submap_knn = knn::KDTree::build(queue, *submap);
    estimate_covariances(*submap_knn, *submap);

    graph::GraphSolverParams sp;
    sp.max_iterations = 8;
    sp.robust.enable = true;
    sp.robust.init_scale = 10.0f;
    sp.robust.min_scale = 1.25f;
    sp.robust.levels = 3;
    sp.robust.iters_per_level = 2;

    graph::GraphOptimization::Options opts;
    opts.gate.enabled = true;
    opts.gate.min_translation = 0.5f;

    auto params = gicp_params();
    params.robust.type = robust::RobustLossType::HUBER;

    graph::GraphOptimization opt(queue, sp, 5, opts);

    Eigen::Matrix<float, 6, 1> delta = Eigen::Matrix<float, 6, 1>::Zero();
    delta.head<3>() = Eigen::Vector3f(0.0f, 0.0f, 0.01f);
    delta.tail<3>() = Eigen::Vector3f(0.3f, 0.0f, 0.0f);
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();

    for (size_t f = 0; f < 20; ++f) {
        const Eigen::Isometry3f T_prev = T;
        T = Eigen::Isometry3f(T.matrix() * eigen_utils::lie::se3_exp(delta));
        auto raw = make_distorted_scan(queue, *submap, T, delta, 10);
        auto deskewed = std::make_shared<PointCloudShared>(queue);
        ASSERT_TRUE(deskew::deskew_point_cloud_constant_velocity(*raw, *deskewed, T_prev, T, 1.0f));
        estimate_covariances(*knn::KDTree::build(queue, *deskewed), *deskewed);

        graph::GraphOptimization::VelocityUpdateContext vu;
        vu.enable = true;
        vu.iterations = 1;
        vu.prev_pose = T_prev;
        vu.dt = 1.0f;
        vu.raw_source = raw;
        // Imperfect motion prediction: perturb the initial guess.
        Eigen::Matrix<float, 6, 1> pred_err = Eigen::Matrix<float, 6, 1>::Zero();
        pred_err.head<3>() = Eigen::Vector3f(0.0f, 0.0f, 0.01f);
        pred_err.tail<3>() = Eigen::Vector3f(0.02f, 0.01f, 0.0f);
        const Eigen::Isometry3f init_T(T.matrix() * eigen_utils::lie::se3_exp(pred_err));
        const auto fr = opt.process_frame(deskewed, submap, submap_knn, nullptr, init_T,
                                          0.1 * static_cast<double>(f), params, vu);
        std::cout << "frame " << f << " error " << fr.error << " inlier? iter " << fr.iterations
                  << " converged " << fr.converged << std::endl;
        EXPECT_TRUE(std::isfinite(fr.error)) << "frame " << f;
        EXPECT_TRUE(fr.current_pose.matrix().allFinite()) << "frame " << f;
    }
}

}  // namespace
