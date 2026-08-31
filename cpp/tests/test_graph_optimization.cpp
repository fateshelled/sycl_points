#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

// Disable Eigen SIMD: this TU compiles SYCL device kernels (graph_factor_kernel)
// for which Eigen's SSE packet specializations are invalid. See graph_factor_kernel.hpp.
#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE
#endif

#include <Eigen/Dense>
#include <sycl/sycl.hpp>

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

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&) override {
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

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&) override {
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

// ---------------------------------------------------------------------------
// SlidingWindow management (host-only)
// ---------------------------------------------------------------------------

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

TEST_F(GraphSlidingWindowTest, MarginalizeOldestShrinksWindow) {
    graph::SlidingWindow window(2);  // max window size = 2
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);

    // Couple the chain so the marginalized prior carries information about the
    // newest node (n2) through n1.
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id0), window.get_node(id1), 5.0f));
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id1), window.get_node(id2), 5.0f));

    const graph::NodeId marginalized = window.marginalize_oldest(queue);
    EXPECT_NE(marginalized, graph::INVALID_NODE_ID);
    EXPECT_EQ(window.window_size(), 2U);
    EXPECT_TRUE(window.prior().is_valid());
    ASSERT_EQ(window.prior().node_ids.size(), 1U);
    EXPECT_EQ(window.prior().node_ids[0], id2);  // star-shaped prior on newest node
    EXPECT_EQ(window.get_node(marginalized), nullptr);
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

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&) override {
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
                EXPECT_EQ(w.prior().node_ids[0], w.active_nodes().back()->id) << "frame " << f;
            }
        }
    }
}

}  // namespace
