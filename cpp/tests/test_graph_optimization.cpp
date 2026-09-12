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
class AnchorFactor : public graph::GraphFactorBase {
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
class BinaryAnchorFactor : public graph::GraphFactorBase {
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

class NonFiniteFactor : public graph::GraphFactorBase {
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

// Rank-deficient Hessian mock: H_mm ends up with one zero eigenvalue, which
// Eigen LDLT happily "succeeds" on. Marginalization must catch the poor
// conditioning and escalate lambda instead.
class WeakRankFactor : public graph::GraphFactorBase {
public:
    WeakRankFactor(std::shared_ptr<graph::PoseNode> src, std::shared_ptr<graph::PoseNode> tgt)
        : src_(std::move(src)), tgt_(std::move(tgt)) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        graph::FactorLinearization lin;
        lin.source_linearization_pose = src_->pose;
        lin.target_linearization_pose = tgt_->pose;
        lin.H00 = Eigen::Matrix<float, 6, 6>::Identity();
        lin.H00(5, 5) = 0.0f;  // singular in rotation-z
        lin.b0.setZero();
        lin.H11 = Eigen::Matrix<float, 6, 6>::Identity();
        // H01 remains zero: the conditioning problem is confined to H_mm.
        lin.error = 0.0f;
        lin.inlier = 1;
        return lin;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&,
                                             const Eigen::Isometry3f&) const override {
        return {0.0f, 1};
    }

    std::pair<graph::NodeId, graph::NodeId> node_ids() const override {
        return {src_->id, tgt_->id};
    }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return false;
    }

private:
    std::shared_ptr<graph::PoseNode> src_;
    std::shared_ptr<graph::PoseNode> tgt_;
};

// Non-finite Hessian mock: a finite b but a NaN information block. Marginalization
// must report NonFiniteSystem instead of silently building a corrupted prior.
class NonFiniteHessianFactor : public graph::GraphFactorBase {
public:
    explicit NonFiniteHessianFactor(std::shared_ptr<graph::PoseNode> node)
        : node_(std::move(node)) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        graph::FactorLinearization lin;
        lin.source_linearization_pose = node_->pose;
        lin.H00.setIdentity();
        lin.H00(0, 0) = std::numeric_limits<float>::quiet_NaN();
        lin.b0.setZero();
        lin.error = 0.0f;
        lin.inlier = 1;
        return lin;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&,
                                             const Eigen::Isometry3f&) const override {
        return {0.0f, 1};
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

// Rank-one Hessian mock with configurable scale/gradient along translation-z:
// Eigen LDLT reports Success on the singular system, so a naive solve would
// apply a huge finite step. The solver must gate the step and escalate damping.
class RankOneSingularFactor : public graph::GraphFactorBase {
public:
    RankOneSingularFactor(std::shared_ptr<graph::PoseNode> node, float h_scale, float b_scale)
        : node_(std::move(node)), h_scale_(h_scale), b_scale_(b_scale) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        graph::FactorLinearization lin;
        lin.source_linearization_pose = node_->pose;
        const Eigen::Matrix<float, 6, 1> axis = Eigen::Matrix<float, 6, 1>::Unit(5);
        lin.H00 = h_scale_ * axis * axis.transpose();
        lin.b0 = b_scale_ * axis;
        lin.error = 0.0f;
        lin.inlier = 1;
        return lin;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&,
                                             const Eigen::Isometry3f&) const override {
        return {0.0f, 1};
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
    float h_scale_;
    float b_scale_;
};

// Synthetic factor with a fixed joint linearization: exercises the
// relative-pose measurement projection without GPU point clouds.
class SyntheticJointFactor : public graph::GraphFactorBase {
public:
    SyntheticJointFactor(std::shared_ptr<graph::PoseNode> src, std::shared_ptr<graph::PoseNode> tgt,
                         graph::FactorLinearization lin)
        : src_(std::move(src)), tgt_(std::move(tgt)), lin_(std::move(lin)) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override { return lin_; }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&,
                                             const Eigen::Isometry3f&) const override {
        return {0.0f, 1};
    }

    std::pair<graph::NodeId, graph::NodeId> node_ids() const override { return {src_->id, tgt_->id}; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return true;
    }

    std::optional<graph::RelativePoseMeasurement> make_relative_pose_measurement() const override {
        const graph::FactorLinearization* lin = cached_linearization();
        if (lin == nullptr) return std::nullopt;
        return graph::relative_pose_measurement_from_linearization(*lin);
    }

private:
    std::shared_ptr<graph::PoseNode> src_;
    std::shared_ptr<graph::PoseNode> tgt_;
    graph::FactorLinearization lin_;
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

TEST_F(GraphSlidingWindowTest, RejectsInvalidWindowSize) {
    EXPECT_THROW(graph::SlidingWindow(0), std::invalid_argument);
}

TEST_F(GraphSlidingWindowTest, MarginalizationDiagnosticsTrackOutcomes) {
    graph::SlidingWindow window(1);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);

    // NotRequired is not an attempt.
    EXPECT_EQ(window.marginalize_oldest(queue).status,
              graph::SlidingWindow::MarginalizationStatus::NotRequired);
    EXPECT_EQ(window.marginalization_diagnostics().attempts, 0u);

    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    window.add_factor(std::make_shared<NonFiniteHessianFactor>(window.get_node(id0)));
    EXPECT_EQ(window.marginalize_oldest(queue).status,
              graph::SlidingWindow::MarginalizationStatus::NonFiniteSystem);
    EXPECT_EQ(window.marginalization_diagnostics().attempts, 1u);
    EXPECT_EQ(window.marginalization_diagnostics().non_finite_system, 1u);
    EXPECT_EQ(window.marginalization_diagnostics().success, 0u);

    // The degraded fallback drop is counted, and the window recovers: with a
    // healthy factor the next marginalization succeeds and tracks the lambda.
    EXPECT_EQ(window.force_drop_oldest(), id0);
    EXPECT_EQ(window.marginalization_diagnostics().force_dropped, 1u);
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id1), Eigen::Isometry3f::Identity(), 10.0f));
    window.add_node(Eigen::Isometry3f::Identity(), 2.0);
    EXPECT_EQ(window.marginalize_oldest(queue).status, graph::SlidingWindow::MarginalizationStatus::Success);
    EXPECT_EQ(window.marginalization_diagnostics().attempts, 2u);
    EXPECT_EQ(window.marginalization_diagnostics().success, 1u);
    EXPECT_FLOAT_EQ(window.marginalization_diagnostics().max_lambda_used, 1e-6f);
}

TEST_F(GraphSlidingWindowTest, MarginalizeOldestShrinksWindow) {
    graph::SlidingWindow window(2);  // max window size = 2
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);

    // Only the factor touching id0 is absorbed; the id1-id2 factor stays live.
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id0), window.get_node(id1), 5.0f));
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id1), window.get_node(id2), 5.0f));

    const auto marginalized = window.marginalize_oldest(queue);
    ASSERT_EQ(marginalized.status, graph::SlidingWindow::MarginalizationStatus::Success);
    EXPECT_EQ(window.window_size(), 2U);
    EXPECT_TRUE(window.prior().is_valid());
    ASSERT_EQ(window.prior().node_ids.size(), 1U);
    EXPECT_EQ(window.prior().node_ids[0], id1);
    EXPECT_EQ(window.get_node(marginalized.marginalized_node), nullptr);
}

TEST_F(GraphSlidingWindowTest, MarginalizationDoesNotAbsorbSurvivingFactors) {
    graph::SlidingWindow window(2);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id0), Eigen::Isometry3f::Identity(), 10.0f));
    window.add_factor(std::make_shared<BinaryAnchorFactor>(window.get_node(id0), window.get_node(id1), 5.0f));
    window.add_factor(std::make_shared<AnchorFactor>(window.get_node(id2), Eigen::Isometry3f::Identity(), 7.0f));

    ASSERT_EQ(window.marginalize_oldest(queue).status, graph::SlidingWindow::MarginalizationStatus::Success);
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

    ASSERT_EQ(window.marginalize_oldest(queue).status, graph::SlidingWindow::MarginalizationStatus::Success);
    ASSERT_EQ(window.prior().node_ids.size(), 2U);
    EXPECT_EQ(window.prior().node_ids[0], id1);
    EXPECT_EQ(window.prior().node_ids[1], id2);
    EXPECT_GT((window.prior().H_prior.block<6, 6>(0, 6).norm()), 1e-3f);
}

TEST_F(GraphSlidingWindowTest, MarginalizationEscalatesLambdaOnPoorConditioning) {
    // Eigen LDLT reports Success even for the rank-deficient H_ss this factor
    // produces, so the eigenvalue conditioning check must fire and the per-frame
    // lambda escalation (base 1e-9 -> x10 each retry) must recover a usable Schur.
    graph::SlidingWindow window(2, 1e-7f);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);
    window.add_factor(std::make_shared<WeakRankFactor>(window.get_node(id0), window.get_node(id1)));

    const auto m = window.marginalize_oldest(queue);
    ASSERT_EQ(m.status, graph::SlidingWindow::MarginalizationStatus::Success);
    EXPECT_NEAR(m.lambda_used, 1e-5f, 1e-6f);
    EXPECT_EQ(window.window_size(), 2U);
    EXPECT_TRUE(window.prior().is_valid());
    EXPECT_EQ(window.prior().node_ids[0], id1);
}

TEST_F(GraphSlidingWindowTest, MarginalizationRejectsNonFiniteSystem) {
    graph::SlidingWindow window(2);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    const graph::NodeId id2 = window.add_node(Eigen::Isometry3f::Identity(), 2.0);
    window.add_factor(std::make_shared<NonFiniteHessianFactor>(window.get_node(id0)));

    const auto m = window.marginalize_oldest(queue);
    EXPECT_EQ(m.status, graph::SlidingWindow::MarginalizationStatus::NonFiniteSystem);
    // Nothing was dropped: the node stays for a next-frame retry.
    EXPECT_EQ(window.window_size(), 3U);
    EXPECT_FALSE(window.prior().is_valid());
}

TEST_F(GraphSlidingWindowTest, FinalizeFrameDefersThenForceDropsOnPersistentFailure) {
    graph::GraphSolverParams params;
    graph::GraphOptimization optimizer(queue, params, 1);
    auto& window = optimizer.window();

    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    window.add_factor(std::make_shared<NonFiniteHessianFactor>(window.get_node(id0)));

    auto finalize_with_new_frame = [&](graph::NodeId tip) {
        graph::GraphOptimization::FrameResult fr;
        fr.current_node_id = tip;
        optimizer.finalize_frame(fr, /*keep=*/true);
        return fr;
    };

    // Frames 1-2: failure is deferred (window stays at max+2). The failure reason
    // (NonFiniteSystem) is preserved independently of the Deferred action.
    graph::NodeId tip = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    auto fr1 = finalize_with_new_frame(tip);
    EXPECT_EQ(fr1.marginalization_action, graph::SlidingWindow::MarginalizationAction::Deferred);
    EXPECT_EQ(fr1.marginalization_status, graph::SlidingWindow::MarginalizationStatus::NonFiniteSystem);
    EXPECT_NE(fr1.marginalization_action, graph::SlidingWindow::MarginalizationAction::ForceDropped);

    tip = window.add_node(Eigen::Isometry3f::Identity(), 2.0);
    auto fr2 = finalize_with_new_frame(tip);
    EXPECT_EQ(fr2.marginalization_action, graph::SlidingWindow::MarginalizationAction::Deferred);
    EXPECT_EQ(fr2.marginalization_status, graph::SlidingWindow::MarginalizationStatus::NonFiniteSystem);

    // Frame 3: the cap (max + 2) is exceeded -> the oldest node is force-dropped
    // (no prior), which bounds the window and removes the offending factor.
    tip = window.add_node(Eigen::Isometry3f::Identity(), 3.0);
    auto fr3 = finalize_with_new_frame(tip);
    EXPECT_EQ(fr3.marginalization_action, graph::SlidingWindow::MarginalizationAction::ForceDropped);
    EXPECT_EQ(fr3.marginalization_status,
              graph::SlidingWindow::MarginalizationStatus::NonFiniteSystem);
    EXPECT_EQ(window.window_size(), 3U);
    EXPECT_EQ(window.get_node(id0), nullptr);
    // The stale-prior guard: no prior may reference the dropped node.
    EXPECT_FALSE(window.prior().is_valid());

    // Recovery: with the NaN factor gone the next finalize marginalizes normally
    // (one node per frame; the window stays bounded at max + 2).
    tip = window.add_node(Eigen::Isometry3f::Identity(), 4.0);
    auto fr4 = finalize_with_new_frame(tip);
    EXPECT_EQ(fr4.marginalization_status, graph::SlidingWindow::MarginalizationStatus::Success);
    EXPECT_EQ(fr4.marginalization_action, graph::SlidingWindow::MarginalizationAction::None);
    EXPECT_LE(window.window_size(), window.max_window_size() + 2);
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

// A singular system with a huge gradient: LDLT "succeeds" and would emit a
// ~1000 m finite step. The step gate must reject it and the damping ladder
// must fail the solve instead of moving the pose.
TEST_F(GraphSolverTest, RejectsUnstableStepOnSingularSystemWithoutUpdatingPose) {
    graph::SlidingWindow window(5);
    Eigen::Isometry3f initial = Eigen::Isometry3f::Identity();
    initial.translate(Eigen::Vector3f(0.3f, -0.2f, 0.1f));
    const graph::NodeId id = window.add_node(initial, 0.0);
    window.add_factor(std::make_shared<RankOneSingularFactor>(window.get_node(id), 1e6f, 1e9f));

    graph::GraphSolver solver(queue);
    const auto result = solver.optimize(window);

    EXPECT_FALSE(result.converged);
    EXPECT_FALSE(result.valid());
    EXPECT_EQ(result.status, graph::GraphSolver::Status::UNSTABLE_STEP);
    EXPECT_TRUE(window.get_node(id)->pose.matrix().isApprox(initial.matrix()));
}

// An ill-conditioned (rank-one, near-null-space) system with a zero gradient
// produces a stable but meaningless direction; the conditioning gate must fail
// the solve instead of trusting the LDLT success.
TEST_F(GraphSolverTest, RejectsIllConditionedSystemWithoutUpdatingPose) {
    graph::SlidingWindow window(5);
    Eigen::Isometry3f initial = Eigen::Isometry3f::Identity();
    initial.translate(Eigen::Vector3f(0.1f, 0.2f, -0.3f));
    const graph::NodeId id = window.add_node(initial, 0.0);
    window.add_factor(std::make_shared<RankOneSingularFactor>(window.get_node(id), 1e12f, 0.0f));

    graph::GraphSolver solver(queue);
    const auto result = solver.optimize(window);

    EXPECT_FALSE(result.converged);
    EXPECT_FALSE(result.valid());
    EXPECT_EQ(result.status, graph::GraphSolver::Status::DECOMPOSITION_FAILED);
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

    ASSERT_EQ(window.marginalize_oldest(queue).status, graph::SlidingWindow::MarginalizationStatus::Success);
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

TEST_F(GraphSolverTest, RejectsInvalidIterationAndRobustSettings) {
    graph::GraphSolverParams params;
    params.max_iterations = 0;
    EXPECT_THROW(graph::GraphSolver(queue, params), std::invalid_argument);

    params.max_iterations = 1;
    params.robust.enable = true;
    params.robust.levels = 0;
    EXPECT_THROW(graph::GraphSolver(queue, params), std::invalid_argument);
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

// Marginalization must keep the robust weights the optimizer actually adopted:
// a measurement rejected by the (frozen) robust loss must not be restored to
// full weight when its node leaves the window. Conditioning is handled by
// lambda escalation on H_mm, never by un-weighting the objective.
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
    const auto marginalized = win_marg.marginalize_oldest(queue);
    ASSERT_EQ(marginalized.status, graph::SlidingWindow::MarginalizationStatus::Success);
    EXPECT_EQ(win_marg.get_node(marginalized.marginalized_node), nullptr);
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
class CountingGicpFactor : public graph::GraphFactorBase {
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
        // Mirror the real GICP factors: judge against this factor's own cached
        // linearization, not the shared PoseNode::linearization_pose.
        const graph::FactorLinearization* lin = cached_linearization();
        if (lin == nullptr) return true;
        return graph::relinearization_needed(node_->pose, lin->source_linearization_pose, rot_th_,
                                             trans_th_);
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

// Two counting factors hang on the SAME node. The first factor relinearizes and
// refreshes the shared PoseNode::linearization_pose; the second must still
// judge against its own cached linearization and relinearize too. Judging by
// the node pose would wrongly report "no movement" for the second factor.
TEST_F(GraphCacheTest, SiblingFactorsRelinearizeIndependently) {
    graph::SlidingWindow window(5);
    const auto id = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    auto node = window.get_node(id);
    CountingGicpFactor f1(node, 0.02f, 0.05f);
    CountingGicpFactor f2(node, 0.02f, 0.05f);

    // Initial linearizations populate both caches.
    f1.get_linearization(queue, 0.02f, 0.05f);
    f2.get_linearization(queue, 0.02f, 0.05f);
    ASSERT_EQ(f1.linearize_calls, 1);
    ASSERT_EQ(f2.linearize_calls, 1);

    // Move the node beyond the threshold, then relinearize only f1.
    Eigen::Isometry3f moved = Eigen::Isometry3f::Identity();
    moved.translate(Eigen::Vector3f(0.2f, 0.0f, 0.0f));
    node->pose = moved;
    f1.get_linearization(queue, 0.02f, 0.05f);
    EXPECT_EQ(f1.linearize_calls, 2);
    // f1's linearize() refreshed node->linearization_pose (now node->pose);
    // f2 must judge against ITS OWN stale cache and relinearize as well.
    f2.get_linearization(queue, 0.02f, 0.05f);
    EXPECT_EQ(f2.linearize_calls, 2);

    // Small move within the threshold: both factors keep their fresh caches.
    node->pose = Eigen::Isometry3f(moved.matrix());
    node->pose.translate(Eigen::Vector3f(0.01f, 0.0f, 0.0f));
    f1.get_linearization(queue, 0.02f, 0.05f);
    f2.get_linearization(queue, 0.02f, 0.05f);
    EXPECT_EQ(f1.linearize_calls, 2);
    EXPECT_EQ(f2.linearize_calls, 2);
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

static Eigen::Isometry3f random_pose(std::mt19937& gen, float trans_scale, float rot_scale) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.translate(Eigen::Vector3f(dist(gen), dist(gen), dist(gen)) * trans_scale);
    Eigen::Vector3f axis(dist(gen), dist(gen), dist(gen));
    axis.normalize();
    T.rotate(Eigen::AngleAxisf(dist(gen) * rot_scale, axis));
    return T;
}

// The t/Jacobian for large residuals: the plain-Jl(r) approximation drifts
// quadratically with ||r||, so the numeric-vs-analytic agreement must hold far
// beyond the (tiny) linearization thresholds.
TEST_F(RelativePoseTest, JacobianMatchesNumericalAtLargeResiduals) {
    auto win_src = std::make_shared<graph::PoseNode>();
    auto win_tgt = std::make_shared<graph::PoseNode>();
    win_src->id = 0;
    win_tgt->id = 1;

    for (int trial = 0; trial < 6; ++trial) {
        std::mt19937 gen(static_cast<unsigned int>(trial + 1) * 17);
        const Eigen::Isometry3f Ts = random_pose(gen, 2.0f, 0.6f);
        const Eigen::Isometry3f Tt = random_pose(gen, 2.0f, 0.6f);
        const Eigen::Isometry3f G = random_pose(gen, 2.0f, 0.6f);

        win_src->pose = Ts;
        win_tgt->pose = Tt;

        graph::RelativePoseParams rp;
        graph::RelativePoseFactor f(0, win_src, 1, win_tgt, G, rp);
        const Eigen::Matrix<float, 6, 6> Omega = graph::RelativePoseFactor::make_information(rp);

        auto residual = [&](const Eigen::Isometry3f& s, const Eigen::Isometry3f& t) {
            return eigen_utils::lie::se3_log(G.inverse() * (s.inverse() * t));
        };
        const Eigen::Matrix<float, 6, 1> r0 = residual(Ts, Tt);
        ASSERT_GT(r0.norm(), 0.5f);  // ensure the trial really stresses ||r||

        const float eps = 1e-3f;
        Eigen::Matrix<float, 6, 6> Js = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 6> Jt = Eigen::Matrix<float, 6, 6>::Zero();
        for (int k = 0; k < 6; ++k) {
            Eigen::Matrix<float, 6, 1> e = Eigen::Matrix<float, 6, 1>::Zero();
            e[k] = eps;
            Js.col(k) = (residual(rright(Ts, e), Tt) - residual(rright(Ts, -e), Tt)) / (2 * eps);
            Jt.col(k) = (residual(Ts, rright(Tt, e)) - residual(Ts, rright(Tt, -e))) / (2 * eps);
        }

        auto lin = f.linearize(queue);
        auto rel_err_m = [](const Eigen::Matrix<float, 6, 6>& a, const Eigen::Matrix<float, 6, 6>& b) {
            return (a - b).norm() / std::max(1.0f, a.norm());
        };
        // b^T Omega b / H: b_num = J^T Omega r
        const Eigen::Matrix<float, 6, 1> b0_num = Js.transpose() * Omega * r0;
        const Eigen::Matrix<float, 6, 1> b1_num = Jt.transpose() * Omega * r0;
        const Eigen::Matrix<float, 6, 6> H00_num = Js.transpose() * Omega * Js;
        const Eigen::Matrix<float, 6, 6> H01_num = Js.transpose() * Omega * Jt;
        const Eigen::Matrix<float, 6, 6> H11_num = Jt.transpose() * Omega * Jt;
        // float32 FD noise grows with ||r||; 3% relative keeps the check
        // meaningful (the buggy plain-Jl implementation fails by ~30% here).
        EXPECT_LT(rel_err_m(lin.H00, H00_num), 3e-2f) << "trial " << trial;
        EXPECT_LT(rel_err_m(lin.H01, H01_num), 3e-2f) << "trial " << trial;
        EXPECT_LT(rel_err_m(lin.H11, H11_num), 3e-2f) << "trial " << trial;
        EXPECT_LT((lin.b0 - b0_num).norm() / std::max(1.0f, b0_num.norm()), 6e-2f) << "trial " << trial;
        EXPECT_LT((lin.b1 - b1_num).norm() / std::max(1.0f, b1_num.norm()), 6e-2f) << "trial " << trial;
    }
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
// Relative-pose measurement projection: 12-DoF joint Hessian -> 6-DoF
// relative-pose information for sparse-chain conversion (host-only math)
// ---------------------------------------------------------------------------

namespace {

Eigen::Matrix<float, 6, 12> relative_jacobian(const Eigen::Isometry3f& T_src,
                                              const Eigen::Isometry3f& T_tgt) {
    Eigen::Matrix<float, 6, 12> J = Eigen::Matrix<float, 6, 12>::Zero();
    J.block<6, 6>(0, 0) = -graph::RelativePoseFactor::adjoint(T_tgt.inverse() * T_src);
    J.block<6, 6>(0, 6).setIdentity();
    return J;
}

graph::FactorLinearization make_joint_linearization(const Eigen::Isometry3f& T_src,
                                                    const Eigen::Isometry3f& T_tgt,
                                                    const Eigen::Matrix<float, 6, 6>& omega,
                                                    const Eigen::Matrix<float, 6, 1>& gradient =
                                                        Eigen::Matrix<float, 6, 1>::Zero()) {
    const Eigen::Matrix<float, 6, 12> J = relative_jacobian(T_src, T_tgt);
    const Eigen::Matrix<float, 12, 12> H = J.transpose() * omega * J;
    const Eigen::Matrix<float, 12, 1> b = J.transpose() * gradient;
    graph::FactorLinearization lin;
    lin.H00 = H.block<6, 6>(0, 0);
    lin.H01 = H.block<6, 6>(0, 6);
    lin.H11 = H.block<6, 6>(6, 6);
    lin.b0 = b.segment<6>(0);
    lin.b1 = b.segment<6>(6);
    lin.source_linearization_pose = T_src;
    lin.target_linearization_pose = T_tgt;
    lin.inlier = 1;
    return lin;
}

Eigen::Isometry3f test_pose(float tx, float ty, float tz, float yaw) {
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.translate(Eigen::Vector3f(tx, ty, tz));
    T.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    return T;
}

}  // namespace

// The projection must recover the information exactly when the joint Hessian
// is genuinely of the form J^T Omega J (the gauge null space aligns).
TEST(GraphRelativePoseInfoTest, RecoversIsotropicInformation) {
    const Eigen::Isometry3f T_src = test_pose(0.3f, -0.2f, 0.1f, 0.2f);
    const Eigen::Isometry3f T_tgt = test_pose(0.8f, -0.1f, -0.2f, -0.15f);

    Eigen::Matrix<float, 6, 6> omega_iso = Eigen::Matrix<float, 6, 6>::Zero();
    omega_iso.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() / (5e-3f * 5e-3f);
    omega_iso.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() / (2e-2f * 2e-2f);

    const auto lin = make_joint_linearization(T_src, T_tgt, omega_iso);
    const auto m = graph::relative_pose_measurement_from_linearization(lin);
    ASSERT_TRUE(m.has_value());

    EXPECT_TRUE(m->G.matrix().isApprox((T_src.inverse() * T_tgt).matrix(), 1e-6f));
    EXPECT_LT((m->information - omega_iso).norm(), 1e-4f * omega_iso.norm());
}

// The projected information keeps the anisotropy of the source Hessian: weak
// directions (e.g. yaw in a corridor) stay weak in the chain factor.
TEST(GraphRelativePoseInfoTest, PreservesAnisotropicWeakDirections) {
    const Eigen::Isometry3f T_src = test_pose(0.1f, 0.2f, -0.1f, 0.3f);
    const Eigen::Isometry3f T_tgt = test_pose(0.6f, -0.3f, 0.2f, 0.1f);

    Eigen::Matrix<float, 6, 6> omega = Eigen::Matrix<float, 6, 6>::Zero();
    // strong roll/pitch and y/z translation, weak yaw and x translation
    omega.diagonal() << 1e6f, 1e6f, 1e2f, 1e2f, 1e6f, 1e6f;

    const auto lin = make_joint_linearization(T_src, T_tgt, omega);
    const auto m = graph::relative_pose_measurement_from_linearization(lin);
    ASSERT_TRUE(m.has_value());

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eig(m->information);
    ASSERT_EQ(eig.info(), Eigen::Success);
    std::vector<float> got(eig.eigenvalues().data(), eig.eigenvalues().data() + 6);
    std::sort(got.begin(), got.end());
    const std::vector<float> expected{1e2f, 1e2f, 1e6f, 1e6f, 1e6f, 1e6f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], 1e-2f * expected[i]) << "eigenvalue " << i;
    }
}

// Degenerate directions keep their near-zero information (no artificial
// stiffening); the PSD clip removes rounding-induced negative eigenvalues.
TEST(GraphRelativePoseInfoTest, KeepsDegenerateDirectionsNearZero) {
    const Eigen::Isometry3f T_src = test_pose(-0.4f, 0.1f, 0.2f, -0.2f);
    const Eigen::Isometry3f T_tgt = test_pose(0.2f, 0.4f, 0.1f, 0.25f);

    Eigen::Matrix<float, 6, 6> omega = Eigen::Matrix<float, 6, 6>::Zero();
    omega.diagonal() << 1e4f, 1e4f, 1e4f, 1e4f, 1e4f, 0.0f;  // yaw unobservable

    const auto lin = make_joint_linearization(T_src, T_tgt, omega);
    const auto m = graph::relative_pose_measurement_from_linearization(lin);
    ASSERT_TRUE(m.has_value());

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eig(m->information);
    ASSERT_EQ(eig.info(), Eigen::Success);
    EXPECT_GE(eig.eigenvalues().minCoeff(), -1e-6f * eig.eigenvalues().maxCoeff());
    EXPECT_LT(eig.eigenvalues().minCoeff(), 1e-2f * eig.eigenvalues().maxCoeff());
}

// Unusable inputs must fall back (nullopt): NaN Hessian, all-zero Hessian
// (never linearized), and a Hessian carrying off-subspace information (e.g. an
// absolute-pose prior mixed in) which the reconstruction guard rejects.
TEST(GraphRelativePoseInfoTest, RejectsUnusableLinearizations) {
    const Eigen::Isometry3f T_src = test_pose(0.2f, 0.0f, 0.1f, 0.1f);
    const Eigen::Isometry3f T_tgt = test_pose(0.5f, 0.2f, 0.0f, -0.1f);

    auto lin_nan = make_joint_linearization(T_src, T_tgt, Eigen::Matrix<float, 6, 6>::Identity());
    lin_nan.H00(0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(graph::relative_pose_measurement_from_linearization(lin_nan).has_value());

    graph::FactorLinearization lin_zero;  // all-zero Hessian, identity poses
    EXPECT_FALSE(graph::relative_pose_measurement_from_linearization(lin_zero).has_value());

    graph::FactorLinearization lin_full;
    lin_full.H00 = Eigen::Matrix<float, 6, 6>::Identity();
    lin_full.H11 = Eigen::Matrix<float, 6, 6>::Identity();
    lin_full.source_linearization_pose = T_src;
    lin_full.target_linearization_pose = T_tgt;
    // Joint Hessian = I_12: full rank over the 12-DoF space, i.e. information
    // outside the relative-pose subspace (rank 6). Reconstruction must fail.
    EXPECT_FALSE(graph::relative_pose_measurement_from_linearization(lin_full).has_value());
}

// The joint gradient must survive the projection: a binary factor linearized
// with non-zero b0/b1 (frozen correspondences at a shifted pose) converts into
// a chain factor with the same energy model 1/2 r^T Omega r + g^T r, so the
// Hessian AND the gradient both match before/after the conversion.
TEST(GraphRelativePoseInfoTest, ProjectsGradientAndConvertedFactorMatches) {
    const Eigen::Isometry3f T_src = test_pose(0.3f, -0.2f, 0.1f, 0.2f);
    const Eigen::Isometry3f T_tgt = test_pose(0.8f, -0.1f, -0.2f, -0.15f);

    Eigen::Matrix<float, 6, 6> omega = Eigen::Matrix<float, 6, 6>::Zero();
    omega.diagonal() << 1e4f, 2e3f, 5e3f, 7e2f, 4e4f, 9e2f;

    Eigen::Matrix<float, 6, 1> g;
    g << 0.4f, -1.5f, 2.0f, 0.1f, -0.3f, 0.8f;

    const auto lin = make_joint_linearization(T_src, T_tgt, omega, g);
    const auto m = graph::relative_pose_measurement_from_linearization(lin);
    ASSERT_TRUE(m.has_value());

    EXPECT_LT((m->information - omega).norm(), 1e-4f * omega.norm());
    EXPECT_LT((m->gradient - g).norm() / std::max(1.0f, g.norm()), 1e-4f);
    EXPECT_TRUE(m->G.matrix().isApprox((T_src.inverse() * T_tgt).matrix(), 1e-6f));

    // The converted chain factor must reproduce the binary's joint model:
    // with nodes at the snapshot, r = 0, so b = J^T g and H = J^T Omega J.
    auto queue = make_queue();
    graph::SlidingWindow window(4);
    const graph::NodeId id0 = window.add_node(T_src, 0.0);
    const graph::NodeId id1 = window.add_node(T_tgt, 1.0);
    window.add_factor(std::make_shared<graph::RelativePoseFactor>(
        id0, window.get_node(id0), id1, window.get_node(id1), m->G, m->information, m->gradient));

    auto& factor = *std::dynamic_pointer_cast<graph::RelativePoseFactor>(window.factors().front());
    const auto lin_chain = factor.linearize(queue);
    const Eigen::Matrix<float, 6, 12> J = relative_jacobian(T_src, T_tgt);
    // The chain factor uses the corrected Jr(r) Jacobian, but at the snapshot
    // r = 0 so Jr = I and both Jacobians coincide.
    EXPECT_LT((lin_chain.H00 - J.block<6, 6>(0, 0).transpose() * omega * J.block<6, 6>(0, 0)).norm(),
              1e-4f * lin_chain.H00.norm());
    EXPECT_LT((lin_chain.H11 - J.block<6, 6>(0, 6).transpose() * omega * J.block<6, 6>(0, 6)).norm(),
              1e-4f * lin_chain.H11.norm());
    EXPECT_LT((lin_chain.b0 - J.block<6, 6>(0, 0).transpose() * g).norm() /
                  std::max(1.0f, lin_chain.b0.norm()),
              1e-4f);
    EXPECT_LT((lin_chain.b1 - J.block<6, 6>(0, 6).transpose() * g).norm() /
                  std::max(1.0f, lin_chain.b1.norm()),
              1e-4f);

    // compute_error must use the same energy model incl. the linear term: at a
    // displaced pose it must equal linearize_at's error there.
    const Eigen::Isometry3f S = test_pose(0.9f, 0.2f, -0.1f, 0.35f);
    const Eigen::Isometry3f T = test_pose(0.4f, -0.3f, 0.3f, -0.05f);
    window.get_node(id0)->pose = S;
    window.get_node(id1)->pose = T;
    const auto lin_disp = factor.linearize(queue);
    const auto err_pair = factor.compute_error(S, T);
    EXPECT_LT(err_pair.first - lin_disp.error, 1e-3f * std::max(1.0f, lin_disp.error));
    EXPECT_EQ(err_pair.second, lin_disp.inlier);
}

// A gradient with content outside the relative-pose subspace (e.g. an
// absolute-pose prior mixed into the binary factor) must be rejected, exactly
// like the Hessian case, instead of silently corrupting the chain.
TEST(GraphRelativePoseInfoTest, RejectsOffSubspaceGradient) {
    const Eigen::Isometry3f T_src = test_pose(0.2f, 0.0f, 0.1f, 0.1f);
    const Eigen::Isometry3f T_tgt = test_pose(0.5f, 0.2f, 0.0f, -0.1f);

    auto lin = make_joint_linearization(T_src, T_tgt, Eigen::Matrix<float, 6, 6>::Identity(),
                                        Eigen::Matrix<float, 6, 1>::Ones());
    lin.b0(0, 0) += 5.0f;  // break the b_joint = J^T g relation
    EXPECT_FALSE(graph::relative_pose_measurement_from_linearization(lin).has_value());
}

// The marginalization prior must transport its cached Schur model into the
// current right-tangent: with e = Log(T_lin^-1 T) and right perturbations,
// dE/ddelta = Jr(e)^T (H_prior e + b_prior). The gradient must match numerical
// differentiation at displaced poses — the Euclidean rule (Jr = I) fails here
// — and the transported Hessian must stay symmetric and reduce to H_prior /
// b_prior exactly at the linearization poses.
TEST(MarginalizationPriorTangentTest, GradientMatchesNumericalAtDisplacedPoses) {
    std::mt19937 gen(42);
    std::normal_distribution<float> ndist(0.0f, 1.0f);

    for (int trial = 0; trial < 4; ++trial) {
        graph::MarginalizationPrior prior;
        const size_t n = 2;
        prior.node_ids = {10, 20};

        // Random SPD prior Hessian.
        Eigen::MatrixXf M = Eigen::MatrixXf::Zero(6 * n, 6 * n);
        for (Eigen::Index i = 0; i < M.rows(); ++i) {
            for (Eigen::Index j = 0; j <= i; ++j) {
                M(i, j) = ndist(gen);
            }
        }
        prior.H_prior = M.transpose() * M + 1e-2f * Eigen::MatrixXf::Identity(6 * n, 6 * n);
        prior.b_prior = Eigen::VectorXf::Zero(6 * n);
        for (Eigen::Index i = 0; i < prior.b_prior.size(); ++i) prior.b_prior(i) = ndist(gen);

        for (size_t i = 0; i < n; ++i) {
            prior.linearization_poses.push_back(random_pose(gen, 1.0f, 0.3f));
        }
        std::vector<Eigen::Isometry3f> poses;
        for (size_t i = 0; i < n; ++i) {
            poses.push_back(prior.linearization_poses[i] * random_pose(gen, 0.8f, 0.4f));
        }
        ASSERT_TRUE(prior.is_valid());

        const graph::MarginalizationPrior::PriorContribution exact = prior.evaluate(poses);

        // Numerical gradient wrt right perturbations of each current pose.
        const float eps = 1e-3f;
        for (size_t i = 0; i < n; ++i) {
            Eigen::Matrix<float, 6, 1> grad_num = Eigen::Matrix<float, 6, 1>::Zero();
            for (int k = 0; k < 6; ++k) {
                Eigen::Matrix<float, 6, 1> d = Eigen::Matrix<float, 6, 1>::Zero();
                d(k) = eps;
                auto perturbed = poses;
                perturbed[i] = rright(poses[i], d);
                const float e_plus = prior.evaluate(perturbed).error;
                d(k) = -eps;
                perturbed[i] = rright(poses[i], d);
                const float e_minus = prior.evaluate(perturbed).error;
                grad_num(k) = (e_plus - e_minus) / (2 * eps);
            }
            const Eigen::Matrix<float, 6, 1> grad_exact = exact.b.segment<6>(6 * i);
            EXPECT_LT((grad_num - grad_exact).norm() / std::max(1.0f, grad_num.norm()), 1e-2f)
                << "trial " << trial << " node " << i;
        }

        // Symmetry check on the transported Hessian (random SPD in, so the
        // transport must keep a symmetric PSD-like shape, symmetric in
        // particular).
        const Eigen::MatrixXf asym = exact.H - exact.H.transpose();
        EXPECT_LT(asym.norm(), 1e-5f * exact.H.norm()) << "trial " << trial;

        // Degenerate case: at the linearization poses the transport is the
        // identity (A = Jr(0) = I), so the model must equal H_prior / b_prior
        // exactly.
        std::vector<Eigen::Isometry3f> at_lin = prior.linearization_poses;
        const graph::MarginalizationPrior::PriorContribution start = prior.evaluate(at_lin);
        EXPECT_LT((start.H - prior.H_prior).norm(), 1e-5f * prior.H_prior.norm());
        EXPECT_LT((start.b - prior.b_prior).norm(), 1e-5f * std::max(1.0f, prior.b_prior.norm()));
    }
}

// ---------------------------------------------------------------------------
// Stale-cache tangent transport in the solver + sparse-chain ordering (host)
// ---------------------------------------------------------------------------

// Binary-factor mock whose linearization NEVER changes: get_linearization()
// caches this fixed model on the first call and every later call reuses it
// regardless of the node poses, so the solver always assembles from a stale
// model transported into the current tangent.
class FixedLinearizationBinaryFactor : public graph::GraphFactorBase {
public:
    FixedLinearizationBinaryFactor(graph::NodeId s_id, graph::NodeId t_id,
                                   graph::FactorLinearization lin)
        : s_id_(s_id), t_id_(t_id), lin_(std::move(lin)) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        return lin_;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&,
                                             const Eigen::Isometry3f&) const override {
        return {0.0f, 1};
    }

    std::pair<graph::NodeId, graph::NodeId> node_ids() const override { return {s_id_, t_id_}; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return false;
    }

private:
    graph::NodeId s_id_, t_id_;
    graph::FactorLinearization lin_;
};

// The solver must assemble a stale cached binary factor EXACTLY as the
// transported model U^T H_lin U / U^T (H_lin o + b) predicts: the one Gauss-Newton
// step it takes must equal the analytic step of that transported quadratic.
// The buggy variant (cross block not finished before transpose) both breaks
// symmetry and changes the lower triangle the LDLT reads, shifting the step.
TEST(GraphSolverTransportTest, StaleCachedBinaryMatchesTangentTransportModel) {
    auto queue = make_queue();
    graph::SlidingWindow window(4);

    const Eigen::Isometry3f T_lin_s = test_pose(0.2f, 0.1f, 0.0f, 0.05f);
    const Eigen::Isometry3f T_lin_t = test_pose(0.9f, 0.3f, -0.2f, -0.3f);

    graph::FactorLinearization lin;
    lin.H00 = Eigen::Matrix<float, 6, 6>::Identity();
    // Keep the model well conditioned (D = 0.1 * ones -> block eigenvalues 1.6/0.4).
    lin.H01 = 0.1f * Eigen::Matrix<float, 6, 6>::Ones();
    lin.H11 = Eigen::Matrix<float, 6, 6>::Identity();
    lin.b0 << 0.03f, -0.05f, 0.07f, 0.01f, 0.02f, -0.04f;
    lin.b1 << -0.02f, 0.04f, 0.03f, -0.03f, 0.01f, 0.03f;
    lin.error = 0.0f;
    lin.inlier = 1;
    lin.source_linearization_pose = T_lin_s;
    lin.target_linearization_pose = T_lin_t;

    const graph::NodeId s_id = window.add_node(T_lin_s, 0.0);
    const graph::NodeId t_id = window.add_node(T_lin_t, 1.0);
    window.add_factor(std::make_shared<FixedLinearizationBinaryFactor>(s_id, t_id, lin));

    graph::GraphSolver solver(queue);
    graph::GraphSolverParams params;
    params.max_iterations = 1;

    // First call: poses on the snapshot, cache the linearization.
    solver.optimize(window, std::nullopt, std::optional<size_t>(1));
    const Eigen::Isometry3f s_start = T_lin_s;
    const Eigen::Isometry3f t_start = T_lin_t;

    // Second call: nodes far from the snapshot (cache reuse is guaranteed by
    // needs_relinearization()==false), so the transport U = Jr(o) is active.
    const Eigen::Matrix<float, 6, 1> ds_vec(0.3f, -0.2f, 0.1f, 0.2f, 0.1f, -0.05f);
    const Eigen::Matrix<float, 6, 1> dt_vec(-0.1f, 0.25f, 0.15f, -0.05f, 0.3f, 0.1f);
    const Eigen::Isometry3f s_disp = Eigen::Isometry3f(s_start.matrix() *
                                                       eigen_utils::lie::se3_exp(ds_vec));
    const Eigen::Isometry3f t_disp = Eigen::Isometry3f(t_start.matrix() *
                                                       eigen_utils::lie::se3_exp(dt_vec));
    window.get_node(s_id)->pose = s_disp;
    window.get_node(t_id)->pose = t_disp;

    ASSERT_EQ(solver.optimize(window, std::nullopt, std::optional<size_t>(1)).iterations, 1U);

    // Expected: exact transported model, one Gauss-Newton step.
    const Eigen::Matrix<float, 6, 1> ds =
        eigen_utils::lie::se3_log(T_lin_s.inverse() * s_disp);
    const Eigen::Matrix<float, 6, 1> dt =
        eigen_utils::lie::se3_log(T_lin_t.inverse() * t_disp);
    const Eigen::Matrix<float, 6, 6> U_s = eigen_utils::lie::se3_right_jacobian(ds);
    const Eigen::Matrix<float, 6, 6> U_t = eigen_utils::lie::se3_right_jacobian(dt);
    Eigen::Matrix<float, 12, 12> H_exp = Eigen::Matrix<float, 12, 12>::Zero();
    H_exp.block<6, 6>(0, 0) = U_s.transpose() * lin.H00 * U_s;
    H_exp.block<6, 6>(6, 6) = U_t.transpose() * lin.H11 * U_t;
    const Eigen::Matrix<float, 6, 6> H_st = U_s.transpose() * lin.H01 * U_t;
    H_exp.block<6, 6>(0, 6) = H_st;
    H_exp.block<6, 6>(6, 0) = H_st.transpose();
    Eigen::MatrixXf H_reg = H_exp + params.solver_damping_lambda *
                                          Eigen::MatrixXf::Identity(12, 12);
    Eigen::Matrix<float, 12, 1> q;
    const Eigen::Matrix<float, 6, 1> q0 = lin.b0 + lin.H00 * ds + lin.H01 * dt;
    const Eigen::Matrix<float, 6, 1> q1 = lin.b1 + lin.H01.transpose() * ds + lin.H11 * dt;
    q << U_s.transpose() * q0, U_t.transpose() * q1;
    const Eigen::LDLT<Eigen::MatrixXf> ldlt(H_reg);
    ASSERT_EQ(ldlt.info(), Eigen::Success);
    const Eigen::VectorXf delta_expect = ldlt.solve(-q);
    ASSERT_TRUE(delta_expect.allFinite());

    const auto near = [&](const Eigen::Isometry3f& start, const Eigen::Matrix<float, 6, 1>& d_exp,
                          const graph::NodeId id) {
        const Eigen::Isometry3f got = window.get_node(id)->pose;
        const Eigen::Matrix<float, 6, 1> d_got = eigen_utils::lie::se3_log(start.inverse() * got);
        EXPECT_LT((d_got - d_exp).norm(), 1e-3f * std::max(1.0f, d_exp.norm()));
    };
    near(s_disp, delta_expect.segment<6>(0), s_id);
    near(t_disp, delta_expect.segment<6>(6), t_id);
}

// The chain conversion must keep the binary factor's FULL endpoint ordering:
// its measurement (G, Omega, gradient) lives in the source->target tangent of
// the binary (source = the older tip at capture time). Reordering the endpoints
// (the previous behaviour) fed G_AB into a factor wired B<-A and left a huge
// non-zero residual at the snapshot.
class PruneBinaryMock : public graph::GraphFactorBase {
public:
    PruneBinaryMock(graph::NodeId s_id, graph::NodeId t_id, graph::FactorLinearization lin,
                    bool has_measurement)
        : s_id_(s_id), t_id_(t_id), lin_(std::move(lin)), has_measurement_(has_measurement) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        return lin_;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&,
                                             const Eigen::Isometry3f&) const override {
        return {0.0f, 1};
    }

    std::pair<graph::NodeId, graph::NodeId> node_ids() const override { return {s_id_, t_id_}; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return false;
    }

    bool is_point_cloud_binary() const override { return true; }

    std::optional<graph::RelativePoseMeasurement> make_relative_pose_measurement() const override {
        if (!has_measurement_) return std::nullopt;
        return graph::relative_pose_measurement_from_linearization(lin_);
    }

private:
    graph::NodeId s_id_, t_id_;
    graph::FactorLinearization lin_;
    bool has_measurement_;
};

TEST(SparseChainPruneOrderTest, ConversionKeepsBinaryEndpointOrdering) {
    auto queue = make_queue();

    const Eigen::Isometry3f T_A = test_pose(0.3f, -0.2f, 0.1f, 0.4f);
    const Eigen::Isometry3f T_B = test_pose(1.0f, 0.4f, 0.0f, -0.2f);

    Eigen::Matrix<float, 6, 6> omega = Eigen::Matrix<float, 6, 6>::Zero();
    omega.diagonal() << 1e4f, 2e3f, 5e3f, 7e2f, 4e4f, 9e2f;
    Eigen::Matrix<float, 6, 1> g;
    g << 0.4f, -1.5f, 2.0f, 0.1f, -0.3f, 0.8f;
    // Snapshot in the binary's own ordering: source = B (was tip), target = A.
    const auto lin = make_joint_linearization(T_B, T_A, omega, g);

    graph::SlidingWindow window(6);
    const graph::NodeId idA = window.add_node(T_A, 0.0);
    const graph::NodeId idB = window.add_node(T_B, 1.0);
    const graph::NodeId idC = window.add_node(Eigen::Isometry3f::Identity(), 2.0);
    window.add_factor(std::make_shared<PruneBinaryMock>(idB, idA, lin, /*has_measurement*/ true));

    window.prune_point_cloud_binaries(idC, idA, idB);

    const auto chain = std::dynamic_pointer_cast<graph::RelativePoseFactor>(
        window.factors().front());
    ASSERT_TRUE(chain);
    // Endpoint order preserved (B -> A, reversed w.r.t. convert_a/convert_b).
    EXPECT_EQ(chain->node_ids().first, idB);
    EXPECT_EQ(chain->node_ids().second, idA);
    // The projected model matches the projection inputs.
    EXPECT_LT((chain->information() - omega).norm(), 1e-4f * omega.norm());

    // Linearizing the chain factor at the snapshot must reproduce the binary
    // model exactly (residual 0, same H/b): proves G/Omega/gradient ordering.
    const auto lin_chain = chain->linearize(queue);
    const Eigen::Matrix<float, 6, 12> J = relative_jacobian(T_B, T_A);
    EXPECT_LT((lin_chain.H00 - J.block<6, 6>(0, 0).transpose() * omega * J.block<6, 6>(0, 0)).norm(),
              1e-4f * omega.norm() * omega.norm());
    EXPECT_LT((lin_chain.b0 - J.block<6, 6>(0, 0).transpose() * g).norm(),
              1e-5f * std::max(1.0f, g.norm()));
    EXPECT_LT(std::fabs(lin_chain.error), 1e-4f);
}

// Without a cached linearization the fallback keeps the binary ordering and
// builds G from the current poses in that same ordering (zero residual).
TEST(SparseChainPruneOrderTest, FallbackKeepsGeometryConsistent) {
    auto queue = make_queue();
    graph::SlidingWindow window(6);

    const Eigen::Isometry3f T_A = test_pose(0.4f, 0.1f, -0.1f, 0.3f);
    const Eigen::Isometry3f T_B = test_pose(1.2f, -0.2f, 0.05f, -0.1f);
    graph::FactorLinearization lin;  // zero Hessian: no usable measurement
    lin.source_linearization_pose = T_B;
    lin.target_linearization_pose = T_A;
    lin.inlier = 1;

    const graph::NodeId idA = window.add_node(T_A, 0.0);
    const graph::NodeId idB = window.add_node(T_B, 1.0);
    const graph::NodeId idC = window.add_node(Eigen::Isometry3f::Identity(), 2.0);
    window.add_factor(std::make_shared<PruneBinaryMock>(idB, idA, lin, /*has_measurement*/ false));

    window.prune_point_cloud_binaries(idC, idA, idB);

    const auto chain = std::dynamic_pointer_cast<graph::RelativePoseFactor>(
        window.factors().front());
    ASSERT_TRUE(chain);
    EXPECT_EQ(chain->node_ids().first, idB);
    EXPECT_EQ(chain->node_ids().second, idA);
    const auto lin_chain = chain->linearize(queue);
    // snapshot poses -> zero residual => zero energy.
    EXPECT_LT(std::fabs(lin_chain.error), 1e-6f);
}

// ---------------------------------------------------------------------------


// G and Omega must come from the same linearization snapshot (the cached
// poses), not from the nodes' current poses.
TEST(GraphRelativePoseInfoTest, MeasurementUsesLinearizationSnapshot) {
    const Eigen::Isometry3f T_lin_src = test_pose(0.3f, 0.0f, 0.0f, 0.1f);
    const Eigen::Isometry3f T_lin_tgt = test_pose(0.7f, 0.1f, 0.0f, 0.2f);
    // Nodes have since moved beyond the lin snapshot.
    const Eigen::Isometry3f T_node_src = test_pose(0.31f, 0.0f, 0.0f, 0.1f);
    const Eigen::Isometry3f T_node_tgt = test_pose(0.72f, 0.1f, 0.0f, 0.2f);

    auto queue = make_queue();
    graph::SlidingWindow window(4);
    const graph::NodeId id0 = window.add_node(T_node_src, 0.0);
    const graph::NodeId id1 = window.add_node(T_node_tgt, 1.0);
    window.add_factor(std::make_shared<SyntheticJointFactor>(
        window.get_node(id0), window.get_node(id1),
        make_joint_linearization(T_lin_src, T_lin_tgt, Eigen::Matrix<float, 6, 6>::Identity())));

    auto& factor = *std::dynamic_pointer_cast<SyntheticJointFactor>(window.factors().front());
    factor.get_linearization(queue, 0.0f, 0.0f);  // populate the cache
    const auto m = factor.make_relative_pose_measurement();
    ASSERT_TRUE(m.has_value());
    EXPECT_TRUE(m->G.matrix().isApprox((T_lin_src.inverse() * T_lin_tgt).matrix(), 1e-6f));
    EXPECT_FALSE(m->G.matrix().isApprox((T_node_src.inverse() * T_node_tgt).matrix(), 1e-3f));
}

// A real BinaryGicpFactor yields a finite PSD measurement once linearized, and
// reports nullopt before its first linearization (fallback contract).
TEST(GraphRelativePoseInfoTest, BinaryFactorMeasurementAndFallback) {
    auto queue = make_queue();
    std::mt19937 gen(11);
    const Eigen::Isometry3f T_a = test_pose(0.2f, 0.0f, 0.0f, 0.0f);
    const Eigen::Isometry3f T_b = test_pose(0.6f, 0.1f, 0.0f, 0.05f);

    auto cloud_a = make_cube_cloud(queue, 800, 1.0f, gen);
    auto cloud_b = transform_cloud(queue, *cloud_a, (T_b.inverse() * T_a));
    auto knn_a = knn::KDTree::build(queue, *cloud_a);
    auto knn_b = knn::KDTree::build(queue, *cloud_b);
    estimate_covariances(*knn_a, *cloud_a);
    estimate_covariances(*knn_b, *cloud_b);

    graph::SlidingWindow window(4);
    const graph::NodeId id_a = window.add_node(T_a, 0.0, cloud_a, knn_a);
    const graph::NodeId id_b = window.add_node(T_b, 1.0, cloud_b, knn_b);
    auto factor = std::make_shared<graph::BinaryGicpFactor>(queue, id_a, window.get_node(id_a),
                                                            id_b, window.get_node(id_b), gicp_params());

    EXPECT_FALSE(factor->make_relative_pose_measurement().has_value());

    factor->get_linearization(queue, 1.0f, 1.0f);  // populates the cache
    const auto m = factor->make_relative_pose_measurement();
    ASSERT_TRUE(m.has_value());
    EXPECT_TRUE(m->G.matrix().isApprox((T_a.inverse() * T_b).matrix(), 1e-4f));
    EXPECT_TRUE(m->information.allFinite());
    EXPECT_GT(m->information.trace(), 0.0f);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eig(m->information);
    ASSERT_EQ(eig.info(), Eigen::Success);
    EXPECT_GE(eig.eigenvalues().minCoeff(), -1e-3f * eig.eigenvalues().maxCoeff());
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
                for (const auto& node : w.active_nodes()) {
                    EXPECT_NE(std::find(w.prior().node_ids.begin(), w.prior().node_ids.end(), node->id),
                              w.prior().node_ids.end()) << "frame " << f;
                }
            }
        }
    }
}

// A solver failure (non-finite system) must abort the frame: remaining
// ladder/velocity rounds stop, the failed tip is discarded and the surviving
// poses are restored (frame-local rollback), and the FrameResult reports the
// invalid status so the pipeline can discard the frame instead of updating the
// map / odometry. Sparse-chain bookkeeping committed before the solve is
// intentionally retained.
TEST_F(GraphTopologyTest, ProcessFrameDiscardsFailedTipAndRestoresPoses) {
    std::mt19937 gen(7);
    std::shared_ptr<knn::KNNBase> submap_knn;
    auto submap = make_submap(gen, submap_knn);

    graph::GraphOptimization opt(queue, graph::GraphSolverParams(), 4);

    // Frames 0-1: healthy solves prime the window (2 nodes, no chain yet).
    auto fr0 = feed_frame(opt, submap, submap_knn, Eigen::Isometry3f::Identity(), 0.0);
    EXPECT_TRUE(fr0.converged);
    const graph::NodeId node0 = fr0.current_node_id;
    Eigen::Isometry3f step = Eigen::Isometry3f::Identity();
    step.translate(Eigen::Vector3f(0.02f, 0.0f, 0.0f));
    auto fr1 = feed_frame(opt, submap, submap_knn, step, 0.1);
    EXPECT_TRUE(fr1.converged);
    const size_t nodes_before = opt.window().window_size();
    const Eigen::Isometry3f pose0_before = opt.window().get_node(node0)->pose;

    // Inject a non-finite factor on the surviving node, then feed the next scan:
    // the assembled system contains NaN and the solver must abort the frame.
    // This frame's prune (n=3) first converts the (node0, node1) adjacent pair
    // into a chain RelativePoseFactor; that bookkeeping predates the solve.
    opt.window().add_factor(std::make_shared<NonFiniteHessianFactor>(opt.window().get_node(node0)));
    auto fr = feed_frame(opt, submap, submap_knn, step, 0.2);

    EXPECT_FALSE(fr.solver_valid());
    EXPECT_EQ(fr.solver_status, graph::GraphSolver::Status::NON_FINITE_SYSTEM);
    EXPECT_EQ(fr.current_node_id, graph::INVALID_NODE_ID);
    EXPECT_EQ(fr.tip_cloud, nullptr);

    // Frame-local rollback: the failed tip is gone, the surviving nodes keep
    // their pre-frame poses, and the injected factor remains. The sparse-chain
    // conversion committed before the solve (the (node0, node1) chain
    // RelativePoseFactor) is retained on purpose.
    auto& w = opt.window();
    EXPECT_EQ(w.window_size(), nodes_before);
    EXPECT_EQ(w.get_node(node0)->id, node0);
    EXPECT_TRUE(w.get_node(node0)->pose.matrix().isApprox(pose0_before.matrix(), 1e-6f));
    size_t chain_factors = 0;
    for (const auto& f : w.factors()) {
        if (auto chain = std::dynamic_pointer_cast<const graph::RelativePoseFactor>(f)) {
            ++chain_factors;
            // The conversion inherits the GICP-derived anisotropic information
            // from the binary's cached linearization (cache existed here), not
            // the isotropic sigma model.
            EXPECT_TRUE(chain->information().allFinite());
            EXPECT_GT(chain->information().trace(), 0.0f);
            EXPECT_FALSE(chain->information().isApprox(
                graph::RelativePoseFactor::make_information(graph::RelativePoseParams()), 1e-3f));
        }
    }
    EXPECT_EQ(chain_factors, 1u);

    // Recovery: dropping the offending node (and the injected factor with it)
    // makes the next frame solve normally again.
    opt.window().force_drop_oldest();
    auto fr2 = feed_frame(opt, submap, submap_knn, step, 0.3);
    EXPECT_TRUE(fr2.converged);
    EXPECT_EQ(fr2.solver_status, graph::GraphSolver::Status::CONVERGED);
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
class ScaleProbeFactor : public graph::GraphFactorBase {
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

// P1 regression: marginalization must re-linearize each factor with its own frozen
// robust scale (the scale it last actually used), not the window's nominal
// marginalization_scale_ fallback. This catches a ladder that ended early
// (e.g. redeskew failure) leaving last_scale_ != min_scale.
TEST_F(GraphSlidingWindowTest, MarginalizationUsesFrozenRobustScale) {
    // Window fallback scale (3.0) differs from the factor's frozen scale (7.0).
    graph::SlidingWindow window(/*max_window_size=*/1, /*marginalization_lambda=*/1e-6f,
                               /*marginalization_scale=*/3.0f);
    auto probe = std::make_shared<ScaleProbeFactor>();
    // Simulate a robust ladder that stopped at scale 7.0 (not the floor): drive one
    // linearization at 7.0, then freeze so the adopted scale is locked.
    probe->get_linearization(queue, 0.02f, 0.05f, 7.0f);
    probe->freeze();
    ASSERT_FALSE(probe->is_annealing());

    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    window.add_factor(probe);
    window.add_node(Eigen::Isometry3f::Identity(), 1.0);  // push the window over max

    const auto m = window.marginalize_oldest(queue);
    ASSERT_EQ(m.status, graph::SlidingWindow::MarginalizationStatus::Success);
    // Every linearization invoked by marginalization used the frozen 7.0 scale;
    // the 3.0 fallback must never have been passed to the factor.
    ASSERT_GE(probe->seen_scales.size(), 1u);
    bool saw_fallback = false;
    for (float s : probe->seen_scales) {
        EXPECT_FLOAT_EQ(s, 7.0f);
        if (s == 3.0f) saw_fallback = true;
    }
    EXPECT_FALSE(saw_fallback);
}

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
class CachedProbeFactor : public graph::GraphFactorBase {
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
