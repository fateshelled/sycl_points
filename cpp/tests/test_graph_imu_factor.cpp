#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <sycl/sycl.hpp>

#include "sycl_points/algorithms/graph/imu_preintegration_factor.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
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
    meas.reserve(static_cast<size_t>(n_steps) + 1);
    for (int i = 0; i <= n_steps; ++i) {
        const double t = t0 + T * static_cast<double>(i) / static_cast<double>(n_steps);
        imu::IMUMeasurement m;
        m.timestamp = t;
        m.gyro = gyro;
        m.accel = accel;
        meas.push_back(m);
    }
    return meas;
}

imu::IMUPreintegrationParams make_params() {
    imu::IMUPreintegrationParams p;
    p.gravity = Eigen::Vector3f(0.0f, 0.0f, -9.80665f);
    // Moderate noise keeps the covariance well conditioned without making the
    // information scale so large that finite differences lose precision.
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

graph::NodeState random_state(std::mt19937& gen) {
    std::uniform_real_distribution<float> pdist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> rdist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> vdist(-0.4f, 0.4f);
    std::uniform_real_distribution<float> bdist(-0.02f, 0.02f);
    graph::NodeState s;
    s.pose = Eigen::Isometry3f::Identity();
    s.pose.linear() = exp_so3(Eigen::Vector3f(rdist(gen), rdist(gen), rdist(gen)));
    s.pose.translation() = Eigen::Vector3f(pdist(gen), pdist(gen), pdist(gen));
    s.velocity = Eigen::Vector3f(vdist(gen), vdist(gen), vdist(gen));
    s.accel_bias = Eigen::Vector3f(bdist(gen), bdist(gen), bdist(gen));
    s.gyro_bias = Eigen::Vector3f(bdist(gen), bdist(gen), bdist(gen));
    return s;
}

class GraphImuFactorTest : public ::testing::Test {
protected:
    sycl_utils::DeviceQueue queue = make_queue();
};

// Analytic Jacobians of the 15-D residual w.r.t. BOTH endpoint states must match
// central finite differences on the solver's manifold update.
TEST_F(GraphImuFactorTest, AnalyticJacobiansMatchFiniteDifferences) {
    const Eigen::Vector3f gyro(0.15f, -0.08f, 0.22f);
    const Eigen::Vector3f accel(0.3f, -0.2f, 9.9f);
    auto preint = imu::IMUPreintegration(make_params());
    imu::IMUBias bias0;
    bias0.gyro_bias = Eigen::Vector3f(0.01f, -0.02f, 0.005f);
    bias0.accel_bias = Eigen::Vector3f(0.03f, -0.01f, 0.02f);
    preint.reset(bias0, Eigen::Matrix<float, 15, 15>::Zero());
    preint.integrate_batch(make_constant_imu(0.0, 0.25, 25, gyro, accel));

    auto src = std::make_shared<graph::PoseNode>();
    auto tgt = std::make_shared<graph::PoseNode>();
    src->id = 0;
    tgt->id = 1;
    Eigen::Isometry3f T_il = Eigen::Isometry3f::Identity();
    T_il.linear() = exp_so3(Eigen::Vector3f(0.2f, -0.1f, 0.3f));
    T_il.translation() = Eigen::Vector3f(0.35f, -0.22f, 0.14f);
    graph::ImuPreintegrationFactor factor(src, tgt, preint, T_il, make_params().gravity);

    std::mt19937 gen(42);
    const graph::NodeState si = random_state(gen);
    const graph::NodeState sj = random_state(gen);

    Eigen::Matrix<float, 15, 1> r;
    Eigen::Matrix<float, 15, 15> Ji, Jj;
    factor.evaluate(si, sj, r, Ji, Jj);

    constexpr float kEps = 1e-4f;
    auto numeric_column = [&](bool perturb_source, int col) {
        Eigen::Matrix<float, 15, 1> d = Eigen::Matrix<float, 15, 1>::Zero();
        d[col] = kEps;
        graph::NodeState sp = si;
        graph::NodeState tp = sj;
        graph::NodeState sm = si;
        graph::NodeState tm = sj;
        if (perturb_source) {
            graph::apply_node_delta(sp, d);
            graph::apply_node_delta(sm, Eigen::Matrix<float, 15, 1>(-d));
        } else {
            graph::apply_node_delta(tp, d);
            graph::apply_node_delta(tm, Eigen::Matrix<float, 15, 1>(-d));
        }
        const Eigen::Matrix<float, 15, 1> rp = factor.residual(sp, tp);
        const Eigen::Matrix<float, 15, 1> rm = factor.residual(sm, tm);
        return (rp - rm) / (2.0f * kEps);
    };

    for (int col = 0; col < 15; ++col) {
        const Eigen::Matrix<float, 15, 1> num_i = numeric_column(true, col);
        const Eigen::Matrix<float, 15, 1> num_j = numeric_column(false, col);
        for (int row = 0; row < 15; ++row) {
            const float denom_i = std::max(1.0f, std::abs(Ji(row, col)));
            const float denom_j = std::max(1.0f, std::abs(Jj(row, col)));
            EXPECT_LT(std::abs(Ji(row, col) - num_i[row]), 2e-3f * denom_i)
                << "source col " << col << " row " << row;
            EXPECT_LT(std::abs(Jj(row, col) - num_j[row]), 2e-3f * denom_j)
                << "target col " << col << " row " << row;
        }
    }
}

// States built to satisfy the preintegration exactly must produce a (near)
// zero residual and gradient.
TEST_F(GraphImuFactorTest, ConsistentStatesHaveZeroResidual) {
    const Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);
    const Eigen::Vector3f accel(0.0f, 0.0f, 9.80665f);
    auto preint = imu::IMUPreintegration(make_params());
    imu::IMUBias bias0;
    preint.reset(bias0, Eigen::Matrix<float, 15, 15>::Zero());
    preint.integrate_batch(make_constant_imu(0.0, 0.5, 50, gyro, accel));

    const auto& raw = preint.get_raw();
    const float dt = static_cast<float>(raw.dt_total);
    const Eigen::Vector3f g(0.0f, 0.0f, -9.80665f);

    graph::NodeState si;
    si.pose = Eigen::Isometry3f::Identity();
    si.velocity = Eigen::Vector3f(1.0f, 0.5f, -0.2f);
    si.accel_bias = bias0.accel_bias;
    si.gyro_bias = bias0.gyro_bias;

    graph::NodeState sj;
    sj.pose = Eigen::Isometry3f::Identity();
    sj.pose.linear() = si.pose.linear() * raw.Delta_R;
    sj.pose.translation() = si.pose.translation() + si.velocity * dt + 0.5f * g * dt * dt +
                            si.pose.linear() * raw.Delta_p;
    sj.velocity = si.velocity + g * dt + si.pose.linear() * raw.Delta_v;
    sj.accel_bias = si.accel_bias;
    sj.gyro_bias = si.gyro_bias;

    auto src = std::make_shared<graph::PoseNode>();
    auto tgt = std::make_shared<graph::PoseNode>();
    src->id = 0;
    tgt->id = 1;
    graph::ImuPreintegrationFactor factor(src, tgt, preint, Eigen::Isometry3f::Identity(), g);
    const Eigen::Matrix<float, 15, 1> r = factor.residual(si, sj);
    EXPECT_LT(r.norm(), 1e-3f) << "r = " << r.transpose();
}

TEST_F(GraphImuFactorTest, RotatingLeverArmHasZeroPositionResidual) {
    auto preint = imu::IMUPreintegration(make_params());
    preint.reset(imu::IMUBias{}, Eigen::Matrix<float, 15, 15>::Zero());
    preint.integrate_batch(make_constant_imu(0.0, 0.25, 25, Eigen::Vector3f(0.0f, 0.0f, 0.8f),
                                              Eigen::Vector3f(0.0f, 0.0f, 9.80665f)));

    Eigen::Isometry3f T_il = Eigen::Isometry3f::Identity();
    T_il.linear() = exp_so3(Eigen::Vector3f(0.2f, -0.1f, 0.3f));
    T_il.translation() = Eigen::Vector3f(0.3f, -0.2f, 0.1f);
    const auto& c = preint.get_raw();
    const float dt = static_cast<float>(c.dt_total);
    const Eigen::Vector3f g = make_params().gravity;
    graph::NodeState si;
    si.pose.linear() = exp_so3(Eigen::Vector3f(-0.1f, 0.15f, 0.2f));
    si.velocity = Eigen::Vector3f(0.4f, -0.1f, 0.2f);
    graph::NodeState sj = si;
    const Eigen::Matrix3f R_imu_i = si.pose.rotation() * T_il.rotation();
    sj.pose.linear() = R_imu_i * c.Delta_R * T_il.rotation().transpose();
    sj.pose.translation() = si.pose.translation() + si.pose.rotation() * T_il.translation() +
                            si.velocity * dt + 0.5f * g * dt * dt + R_imu_i * c.Delta_p -
                            sj.pose.rotation() * T_il.translation();
    sj.velocity = si.velocity + g * dt + R_imu_i * c.Delta_v;

    auto src = std::make_shared<graph::PoseNode>();
    auto tgt = std::make_shared<graph::PoseNode>();
    graph::ImuPreintegrationFactor factor(src, tgt, preint, T_il, g);
    EXPECT_LT(factor.residual(si, sj).norm(), 1e-3f);
    Eigen::Isometry3f rotation_only = Eigen::Isometry3f::Identity();
    rotation_only.linear() = T_il.rotation();
    graph::ImuPreintegrationFactor without_arm(src, tgt, preint, rotation_only, g);
    EXPECT_GT(without_arm.residual(si, sj).head<3>().norm(), 1e-2f);
}

// The factor advertises full-state usage so the solver picks the 15-DOF layout,
// and linearize() emits dense 15x15 blocks.
TEST_F(GraphImuFactorTest, LinearizeProducesFullStateBlocks) {
    auto preint = imu::IMUPreintegration(make_params());
    preint.reset(imu::IMUBias{}, Eigen::Matrix<float, 15, 15>::Zero());
    preint.integrate_batch(
        make_constant_imu(0.0, 0.2, 20, Eigen::Vector3f(0.1f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 9.8f)));
    auto src = std::make_shared<graph::PoseNode>();
    auto tgt = std::make_shared<graph::PoseNode>();
    src->id = 0;
    tgt->id = 1;
    tgt->pose.translation() = Eigen::Vector3f(0.1f, 0.0f, 0.0f);

    graph::ImuPreintegrationFactor factor(src, tgt, preint, Eigen::Isometry3f::Identity(),
                                          Eigen::Vector3f(0.0f, 0.0f, -9.80665f));
    EXPECT_TRUE(factor.uses_full_state());
    const graph::FactorLinearization lin = factor.linearize(queue);
    EXPECT_TRUE(lin.full_state);
    EXPECT_TRUE(lin.H00_full.allFinite());
    EXPECT_TRUE(lin.H11_full.allFinite());
    EXPECT_TRUE(lin.b0_full.allFinite());
    // H must be symmetric PSD (information weighted).
    EXPECT_LT((lin.H00_full - lin.H00_full.transpose()).norm(), 1e-4f * lin.H00_full.norm());
    EXPECT_GE(lin.error, 0.0f);
}

// The bias random-walk term folded into the joint residual must penalize a
// bias difference between the endpoints, with the covariance's bias block
// (filled from the random-walk noise densities) providing the weight.
TEST_F(GraphImuFactorTest, BiasRandomWalkPenalizesBiasDifference) {
    const Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);
    const Eigen::Vector3f accel(0.0f, 0.0f, 9.80665f);
    auto preint = imu::IMUPreintegration(make_params());
    preint.reset(imu::IMUBias{}, Eigen::Matrix<float, 15, 15>::Zero());
    preint.integrate_batch(make_constant_imu(0.0, 0.5, 50, gyro, accel));

    const auto& raw = preint.get_raw();
    const float dt = static_cast<float>(raw.dt_total);
    const Eigen::Vector3f g(0.0f, 0.0f, -9.80665f);

    graph::NodeState si;
    si.pose = Eigen::Isometry3f::Identity();
    si.velocity = Eigen::Vector3f(1.0f, 0.5f, -0.2f);

    graph::NodeState sj;
    sj.pose = Eigen::Isometry3f::Identity();
    sj.pose.linear() = si.pose.linear() * raw.Delta_R;
    sj.pose.translation() =
        si.pose.translation() + si.velocity * dt + 0.5f * g * dt * dt + si.pose.linear() * raw.Delta_p;
    sj.velocity = si.velocity + g * dt + si.pose.linear() * raw.Delta_v;

    auto src = std::make_shared<graph::PoseNode>();
    auto tgt = std::make_shared<graph::PoseNode>();
    src->id = 0;
    tgt->id = 1;
    graph::ImuPreintegrationFactor factor(src, tgt, preint, Eigen::Isometry3f::Identity(), g);

    const float error0 = factor.compute_error_state(si, sj).first;
    EXPECT_LT(error0, 1e-3f);

    graph::NodeState sj_bias = sj;
    sj_bias.accel_bias = Eigen::Vector3f(0.02f, -0.01f, 0.015f);
    sj_bias.gyro_bias = Eigen::Vector3f(0.005f, 0.01f, -0.008f);
    const float error1 = factor.compute_error_state(si, sj_bias).first;
    EXPECT_GT(error1, error0 + 1e-3f);

    // The bias blocks of the linearized Hessian must be nonzero and symmetric.
    const graph::FactorLinearization lin = factor.linearize(queue);
    const Eigen::Matrix<float, 6, 6> H_bias = lin.H00_full.bottomRightCorner<6, 6>();
    EXPECT_GT(H_bias.trace(), 1e-3f);
    EXPECT_LT((H_bias - H_bias.transpose()).norm(), 1e-5f * H_bias.norm());

    // A bias difference enters the gradient on both endpoints (opposite signs).
    graph::NodeState sj_shift = sj;
    sj_shift.accel_bias = Eigen::Vector3f(0.01f, 0.0f, 0.0f);
    Eigen::Matrix<float, 15, 1> r;
    Eigen::Matrix<float, 15, 15> Ji, Jj;
    factor.evaluate(si, sj_shift, r, Ji, Jj);
    EXPECT_NEAR(r.segment<3>(9).x(), 0.01f, 1e-5f);
    EXPECT_LT((Ji.block<3, 3>(9, 9).norm()), 5.0f);
    EXPECT_GT((Jj.block<3, 3>(9, 9).diagonal().minCoeff()), 0.5f);
}

}  // namespace
