#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <sycl/sycl.hpp>

#include "sycl_points/algorithms/graph/graph_solver.hpp"
#include "sycl_points/algorithms/graph/graph_optimization.hpp"
#include "sycl_points/algorithms/graph/imu_preintegration_factor.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/algorithms/graph/relative_pose_factor.hpp"
#include "sycl_points/algorithms/graph/sliding_window.hpp"
#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/pipeline/graph_odometry.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace {

using namespace sycl_points;
using namespace sycl_points::algorithms;

sycl_utils::DeviceQueue make_queue() {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    return sycl_utils::DeviceQueue(device);
}

std::vector<imu::IMUMeasurement> make_constant_imu(double t0, double T, int n_steps,
                                                   const Eigen::Vector3f& gyro, const Eigen::Vector3f& accel) {
    std::vector<imu::IMUMeasurement> meas;
    for (int i = 0; i <= n_steps; ++i) {
        imu::IMUMeasurement m;
        m.timestamp = t0 + T * static_cast<double>(i) / static_cast<double>(n_steps);
        m.gyro = gyro;
        m.accel = accel;
        meas.push_back(m);
    }
    return meas;
}

imu::IMUPreintegrationParams make_params() {
    imu::IMUPreintegrationParams p;
    p.gravity = Eigen::Vector3f(0.0f, 0.0f, -9.80665f);
    p.gyro_noise_density = 1e-2f;
    p.accel_noise_density = 1e-1f;
    p.gyro_bias_rw_density = 1e-2f;
    p.accel_bias_rw_density = 1e-2f;
    return p;
}

Eigen::Matrix3f exp_so3(const Eigen::Vector3f& phi) {
    return Eigen::Matrix3f(
        eigen_utils::geometry::quaternion_to_rotation_matrix(eigen_utils::lie::so3_exp(phi)));
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

graph::NodeState random_state(std::mt19937& gen) {
    std::uniform_real_distribution<float> pdist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> rdist(-0.2f, 0.2f);
    std::uniform_real_distribution<float> vdist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> bdist(-0.01f, 0.01f);
    graph::NodeState s;
    s.pose = Eigen::Isometry3f::Identity();
    s.pose.linear() = exp_so3(Eigen::Vector3f(rdist(gen), rdist(gen), rdist(gen)));
    s.pose.translation() = Eigen::Vector3f(pdist(gen), pdist(gen), pdist(gen));
    s.velocity = Eigen::Vector3f(vdist(gen), vdist(gen), vdist(gen));
    s.accel_bias = Eigen::Vector3f(bdist(gen), bdist(gen), bdist(gen));
    s.gyro_bias = Eigen::Vector3f(bdist(gen), bdist(gen), bdist(gen));
    return s;
}

/// Test-only full-state anchor: residual
///   [Log(T_tgt^-1 T); v - v_t; ba - ba_t; bg - bg_t]
/// with an isotropic weight. Exercises the solver's 15-DOF layout independent of
/// the IMU factor.
class FullStateAnchorFactor : public graph::GraphFactorBase {
public:
    FullStateAnchorFactor(std::shared_ptr<graph::PoseNode> node, const graph::NodeState& target, float weight)
        : node_(std::move(node)), target_(target), w_(weight) {}

    graph::FactorLinearization linearize(const sycl_utils::DeviceQueue&, float) override {
        const graph::NodeState s = node_->state();
        Eigen::Matrix<float, 15, 1> r;
        r.segment<6>(0) = Eigen::Matrix<float, 6, 1>(eigen_utils::lie::se3_log(target_.pose.inverse() * s.pose));
        r.segment<3>(6) = s.velocity - target_.velocity;
        r.segment<3>(9) = s.accel_bias - target_.accel_bias;
        r.segment<3>(12) = s.gyro_bias - target_.gyro_bias;

        const Eigen::Matrix<float, 6, 6> J_pose =
            eigen_utils::lie::se3_right_jacobian(r.segment<6>(0));
        Eigen::Matrix<float, 15, 15> J = Eigen::Matrix<float, 15, 15>::Zero();
        J.block<6, 6>(0, 0) = J_pose;
        J.block<3, 3>(6, 6).setIdentity();
        J.block<3, 3>(9, 9).setIdentity();
        J.block<3, 3>(12, 12).setIdentity();
        const Eigen::Matrix<float, 15, 15> Omega = w_ * Eigen::Matrix<float, 15, 15>::Identity();

        graph::FactorLinearization lin;
        lin.full_state = true;
        lin.H00_full = J.transpose() * Omega * J;
        lin.b0_full = J.transpose() * Omega * r;
        lin.error = 0.5f * w_ * r.squaredNorm();
        lin.inlier = 1;
        lin.source_linearization_pose = s.pose;
        return lin;
    }

    std::pair<float, uint32_t> compute_error(const Eigen::Isometry3f&, const Eigen::Isometry3f&) const override {
        const graph::NodeState s = node_->state();
        const Eigen::Matrix<float, 6, 1> ep = eigen_utils::lie::se3_log(target_.pose.inverse() * s.pose);
        const float e2 = ep.squaredNorm() + (s.velocity - target_.velocity).squaredNorm() +
                         (s.accel_bias - target_.accel_bias).squaredNorm() +
                         (s.gyro_bias - target_.gyro_bias).squaredNorm();
        return {0.5f * w_ * e2, 1};
    }

    std::pair<float, uint32_t> compute_error_state(const graph::NodeState& s,
                                                   const graph::NodeState&) const override {
        const Eigen::Matrix<float, 6, 1> ep = eigen_utils::lie::se3_log(target_.pose.inverse() * s.pose);
        const float e2 = ep.squaredNorm() + (s.velocity - target_.velocity).squaredNorm() +
                         (s.accel_bias - target_.accel_bias).squaredNorm() +
                         (s.gyro_bias - target_.gyro_bias).squaredNorm();
        return {0.5f * w_ * e2, 1};
    }

    std::pair<graph::NodeId, graph::NodeId> node_ids() const override {
        return {node_->id, graph::INVALID_NODE_ID};
    }

    bool uses_full_state() const override { return true; }

    bool needs_relinearization(const Eigen::Isometry3f&, const Eigen::Isometry3f&, float,
                               float) const override {
        return true;
    }

private:
    std::shared_ptr<graph::PoseNode> node_;
    graph::NodeState target_;
    float w_;
};

class GraphLioTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue = make_queue();
};

pipeline::graph_odometry::GraphOdometryParams graph_pipeline_params() {
    pipeline::graph_odometry::GraphOdometryParams params;
    params.device.vendor = "default";
    params.device.type = "";
    params.imu.enable = true;
    params.imu.deskew.enable = true;
    params.imu.initial_alignment.enable = true;
    params.motion_prediction.mode = pipeline::lidar_odometry::MotionPredictionMode::IMU_SE3;
    params.imu.preintegration.gyro_noise_density = 1e-2f;
    params.imu.preintegration.accel_noise_density = 1e-3f;
    params.imu.preintegration.gyro_bias_rw_density = 1e-5f;
    params.imu.preintegration.accel_bias_rw_density = 1e-4f;
    return params;
}

TEST(GraphPipelineImu, RejectsUnsupportedLeverArm) {
    auto params = graph_pipeline_params();
    params.imu.T_imu_to_lidar.translation().x() = 0.1f;
    EXPECT_THROW(pipeline::graph_odometry::GraphOdometryPipeline pipeline(params), std::invalid_argument);
}

TEST(GraphPipelineImu, BootstrapWaitsForImuBoundary) {
    pipeline::graph_odometry::GraphOdometryPipeline pipeline(graph_pipeline_params());
    using Coverage = pipeline::graph_odometry::GraphOdometryPipeline::IMUCoverage;
    EXPECT_EQ(pipeline.get_frame_imu_coverage(10.0), Coverage::waiting_for_future);

    imu::IMUMeasurement measurement;
    measurement.timestamp = 9.9;
    pipeline.add_imu_measurement(measurement);
    EXPECT_EQ(pipeline.get_frame_imu_coverage(10.0), Coverage::waiting_for_future);

    measurement.timestamp = 10.1;
    pipeline.add_imu_measurement(measurement);
    EXPECT_EQ(pipeline.get_frame_imu_coverage(10.0), Coverage::ready);
    EXPECT_EQ(pipeline.get_frame_imu_coverage(9.8), Coverage::start_expired);
}

TEST(GraphPipelineImu, BootstrapAfterRejectedFrameCommitsOneRoot) {
    auto params = graph_pipeline_params();
    params.graph.lio.keyframe_imu_history_max_samples = 2;
    params.graph.registration.min_num_points = 3;
    params.graph.registration.factor.reg_type = registration::RegType::POINT_TO_POINT;
    params.submap.point_random_sampling_num = 32;
    params.scan.preprocess.box_filter.enable = false;
    params.scan.preprocess.angle_incidence_filter.enable = false;
    params.scan.downsampling.polar.enable = false;
    params.scan.downsampling.random.enable = false;
    params.scan.intensity_correction.enable = false;
    params.covariance_estimation.m_estimation.enable = false;
    pipeline::graph_odometry::GraphOdometryPipeline pipeline(params);

    imu::IMUMeasurement measurement;
    measurement.accel = Eigen::Vector3f(0.0f, 0.0f, 9.80665f);
    for (int i = 0; i <= 12; ++i) {
        measurement.timestamp = i * 0.1;
        pipeline.add_imu_measurement(measurement);
    }

    std::mt19937 gen(42);
    auto small_cloud = make_cube_cloud(*pipeline.get_device_queue(), 2, 1.0f, gen);
    auto set_scan_times = [](const PointCloudShared::Ptr& scan) {
        scan->resize_timestamps(scan->size());
        std::fill(scan->timestamp_offsets->begin(), scan->timestamp_offsets->end(), 100);
        (*scan->timestamp_offsets)[0] = 0;
        scan->start_time_ms = 900.0;
        scan->end_time_ms = 1000.0;
    };
    set_scan_times(small_cloud);
    EXPECT_EQ(pipeline.process(small_cloud, 1.0),
              pipeline::graph_odometry::GraphOdometryPipeline::ResultType::small_number_of_points);
    EXPECT_EQ(pipeline.get_graph_window().window_size(), 0u);

    auto cloud = make_cube_cloud(*pipeline.get_device_queue(), 100, 1.0f, gen);
    set_scan_times(cloud);
    EXPECT_EQ(pipeline.process(cloud, 1.0),
              pipeline::graph_odometry::GraphOdometryPipeline::ResultType::first_frame);
    EXPECT_EQ(pipeline.get_graph_window().window_size(), 1u);
    EXPECT_EQ(pipeline.get_keyframe_poses().size(), 1u);
    EXPECT_EQ(pipeline.get_frame_imu_coverage(1.1),
              pipeline::graph_odometry::GraphOdometryPipeline::IMUCoverage::recovery_required);
}

TEST_F(GraphLioTest, BootstrapNodeHasOneFullStateAnchor) {
    std::mt19937 gen(5);
    auto cloud = make_cube_cloud(queue, 200, 0.5f, gen);
    auto cloud_knn = knn::KDTree::build(queue, *cloud);
    graph::GraphOptimization opt(queue, graph::GraphSolverParams(), 4);

    graph::NodeState state;
    state.pose.translation() = Eigen::Vector3f(1.0f, -2.0f, 0.5f);
    state.velocity = Eigen::Vector3f(0.2f, 0.1f, -0.1f);
    state.accel_bias = Eigen::Vector3f(0.01f, 0.0f, -0.02f);
    state.gyro_bias = Eigen::Vector3f(0.001f, -0.002f, 0.0f);

    graph::GraphOptimization::ImuEdgeContext prior;
    const graph::NodeId root = opt.add_bootstrap_node(
        state.pose, 1.0, cloud, cloud_knn, state, prior);

    ASSERT_NE(root, graph::INVALID_NODE_ID);
    ASSERT_EQ(opt.window().window_size(), 1u);
    ASSERT_EQ(opt.window().factors().size(), 1u);
    EXPECT_TRUE(opt.window().factors().front()->uses_full_state());
    const auto linearization = opt.window().factors().front()->linearize(queue);
    EXPECT_GT((linearization.H00_full.topLeftCorner<6, 6>().trace()), 0.0f);
    EXPECT_GT((linearization.H00_full.bottomRightCorner<9, 9>().trace()), 0.0f);
}

// The solver selects the 15-DOF layout when a full-state factor is present and
// converges to a consistent set of states.
TEST_F(GraphLioTest, SolverConvergesWithFullStateFactors) {
    std::mt19937 gen(7);
    const auto s0 = random_state(gen);
    const auto s1 = random_state(gen);

    graph::SlidingWindow window(4);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    auto n0 = window.get_node(id0);
    auto n1 = window.get_node(id1);
    // Start far from the targets.
    n0->pose = s0.pose;
    n0->velocity = Eigen::Vector3f(1.0f, -1.0f, 0.5f);
    n1->pose = s1.pose;
    n1->pose.translation() += Eigen::Vector3f(0.4f, -0.3f, 0.2f);
    n1->velocity = Eigen::Vector3f(-0.7f, 0.6f, 0.5f);

    window.add_factor(std::make_shared<FullStateAnchorFactor>(n0, s0, 50.0f));
    window.add_factor(std::make_shared<FullStateAnchorFactor>(n1, s1, 50.0f));

    graph::GraphSolver solver(queue);
    const auto result = solver.optimize(window);

    ASSERT_TRUE(result.valid());
    EXPECT_TRUE(result.converged);
    const graph::NodeState r0 = n0->state();
    const graph::NodeState r1 = n1->state();
    EXPECT_LT((r0.velocity - s0.velocity).norm(), 1e-3f);
    EXPECT_LT((r1.velocity - s1.velocity).norm(), 1e-3f);
    EXPECT_LT((r1.pose.translation() - s1.pose.translation()).norm(), 1e-3f);
}

// Marginalization must absorb a 15-DOF node into a valid 15-DOF prior that
// reproduces the reduced system at its linearization point.
TEST_F(GraphLioTest, MarginalizesFullStateNodeIntoPrior) {
    std::mt19937 gen(11);
    const auto s0 = random_state(gen);
    const auto s1 = random_state(gen);

    graph::SlidingWindow window(2);
    const graph::NodeId id0 = window.add_node(s0.pose, 0.0, nullptr, nullptr, s0.velocity, s0.accel_bias,
                                              s0.gyro_bias);
    const graph::NodeId id1 = window.add_node(s1.pose, 1.0, nullptr, nullptr, s1.velocity, s1.accel_bias,
                                              s1.gyro_bias);
    window.add_factor(std::make_shared<FullStateAnchorFactor>(window.get_node(id0), s0, 20.0f));
    window.add_factor(std::make_shared<FullStateAnchorFactor>(window.get_node(id1), s1, 20.0f));
    // Chain edge so the marginalized node's Markov blanket contains id1.
    window.add_factor(std::make_shared<graph::RelativePoseFactor>(
        id0, window.get_node(id0), id1, window.get_node(id1), s0.pose.inverse() * s1.pose,
        graph::RelativePoseParams{}));
    window.add_node(s1.pose, 2.0);

    const auto m = window.marginalize_oldest(queue);
    ASSERT_EQ(m.status, graph::SlidingWindow::MarginalizationStatus::Success);
    EXPECT_EQ(window.prior().dof, graph::kNodeDof);
    ASSERT_TRUE(window.prior().is_valid());
    ASSERT_EQ(window.prior().node_ids.size(), 1u);

    // The prior must evaluate to a finite reduced system at its linearization.
    const auto node = window.get_node(window.prior().node_ids[0]);
    std::vector<graph::NodeState, Eigen::aligned_allocator<graph::NodeState>> states;
    states.push_back(node->state());
    const auto c = window.prior().evaluate_states(states);
    EXPECT_TRUE(c.H.allFinite());
    EXPECT_TRUE(c.b.allFinite());
    EXPECT_EQ(c.H.rows(), graph::kNodeDof);
    EXPECT_TRUE(std::isfinite(c.error));
}

// With an IMU edge plus full-state anchors the whole system (poses, velocity,
// biases) stays well conditioned and the solver reports a valid result.
TEST_F(GraphLioTest, SolvesWithImuEdgeAndAnchors) {
    const Eigen::Vector3f gyro(0.1f, -0.05f, 0.2f);
    const Eigen::Vector3f accel(0.2f, -0.1f, 9.9f);
    auto preint = imu::IMUPreintegration(make_params());
    imu::IMUBias bias0;
    preint.reset(bias0, Eigen::Matrix<float, 15, 15>::Zero());
    preint.integrate_batch(make_constant_imu(0.0, 0.3, 30, gyro, accel));

    graph::SlidingWindow window(4);
    const graph::NodeId id0 = window.add_node(Eigen::Isometry3f::Identity(), 0.0);
    const graph::NodeId id1 = window.add_node(Eigen::Isometry3f::Identity(), 1.0);
    auto n0 = window.get_node(id0);
    auto n1 = window.get_node(id1);

    graph::NodeState anchor0 = n0->state();
    graph::NodeState anchor1 = n1->state();
    anchor1.pose.translation() = Eigen::Vector3f(0.05f, 0.02f, 0.3f);

    window.add_factor(std::make_shared<FullStateAnchorFactor>(n0, anchor0, 100.0f));
    window.add_factor(std::make_shared<FullStateAnchorFactor>(n1, anchor1, 1.0f));
    window.add_factor(std::make_shared<graph::ImuPreintegrationFactor>(
        n0, n1, preint, Eigen::Isometry3f::Identity(), make_params().gravity));

    graph::GraphSolver solver(queue);
    const auto result = solver.optimize(window);
    EXPECT_TRUE(result.valid());
    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(n0->pose.matrix().allFinite());
    EXPECT_TRUE(n1->pose.matrix().allFinite());
    EXPECT_TRUE(n1->velocity.allFinite());
    EXPECT_TRUE(n1->accel_bias.allFinite());
    EXPECT_TRUE(n1->gyro_bias.allFinite());
}

// End-to-end Phase 2.4 wiring: GraphOptimization must attach the preintegration
// edge to the source keyframe and seed the new tip with the navigation state.
TEST_F(GraphLioTest, GraphOptimizationAttachesImuEdgeAndSeedsNavState) {
    std::mt19937 gen(3);
    const size_t n_points = 400;
    const float half = 0.6f;
    auto cloud = make_cube_cloud(queue, n_points, half, gen);
    auto knn = knn::KDTree::build(queue, *cloud);
    estimate_covariances(*knn, *cloud);
    auto submap = make_cube_cloud(queue, n_points, half, gen);
    auto submap_knn = knn::KDTree::build(queue, *submap);
    estimate_covariances(*submap_knn, *submap);

    graph::GraphOptimization opt(queue, graph::GraphSolverParams(), 5);
    auto fr1 = opt.process_frame(cloud, submap, submap_knn, knn, Eigen::Isometry3f::Identity(), 0.0, gicp_params());
    ASSERT_TRUE(fr1.solver_valid());
    const graph::NodeId source_id = fr1.current_node_id;
    ASSERT_NE(source_id, graph::INVALID_NODE_ID);

    // Stationary preintegration: consistent with identity motion, zero velocity
    // and zero bias, so the tightly-coupled solve stays well conditioned.
    auto preint = std::make_shared<imu::IMUPreintegration>(make_params());
    preint->reset(imu::IMUBias{}, Eigen::Matrix<float, 15, 15>::Zero(), Eigen::Matrix3f::Identity());
    preint->integrate_batch(
        make_constant_imu(0.0, 0.3, 30, Eigen::Vector3f::Zero(), Eigen::Vector3f(0.0f, 0.0f, 9.80665f)));

    auto knn2 = knn::KDTree::build(queue, *cloud);
    graph::GraphOptimization::ImuEdgeContext imu_edge;
    imu_edge.enable = true;
    imu_edge.source_id = source_id;
    imu_edge.preintegration = preint;
    imu_edge.gravity = Eigen::Vector3f(0.0f, 0.0f, -9.80665f);
    auto src = opt.window().get_node(source_id);
    ASSERT_TRUE(src != nullptr);
    imu_edge.tip_velocity = src->velocity;
    imu_edge.tip_accel_bias = src->accel_bias;
    imu_edge.tip_gyro_bias = src->gyro_bias;
    imu_edge.add_nav_prior = true;

    auto fr2 = opt.process_frame(cloud, submap, submap_knn, knn2, Eigen::Isometry3f::Identity(), 0.3,
                                 gicp_params(), graph::GraphOptimization::VelocityUpdateContext(), imu_edge);
    EXPECT_TRUE(fr2.solver_valid());
    ASSERT_TRUE(fr2.has_current_state);

    bool has_full_state_factor = false;
    for (const auto& f : opt.window().factors()) {
        if (f->uses_full_state()) has_full_state_factor = true;
    }
    EXPECT_TRUE(has_full_state_factor);
    auto tip = opt.window().get_node(fr2.current_node_id);
    ASSERT_TRUE(tip != nullptr);
    EXPECT_TRUE(tip->velocity.allFinite());
    EXPECT_TRUE(tip->accel_bias.allFinite());
    EXPECT_TRUE(tip->gyro_bias.allFinite());
    EXPECT_TRUE(fr2.current_state.pose.matrix().isApprox(tip->pose.matrix(), 1e-6f));
    EXPECT_TRUE(fr2.current_state.velocity.isApprox(tip->velocity, 1e-6f));
}

}  // namespace
