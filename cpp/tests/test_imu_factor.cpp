#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Dense>

#include "sycl_points/imu/imu_factor.hpp"

namespace imu = sycl_points::imu;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Build a diagonal 15×15 covariance matrix from a scalar (σ² for all).
static Eigen::Matrix<double, 15, 15> diag_cov(double sigma_sq) {
    return sigma_sq * Eigen::Matrix<double, 15, 15>::Identity();
}

/// Rotation matrix around Z axis by angle_rad (double precision).
static Eigen::Matrix3d rot_z(double angle_rad) {
    const double c = std::cos(angle_rad);
    const double s = std::sin(angle_rad);
    Eigen::Matrix3d R;
    R << c, -s, 0,
         s,  c, 0,
         0,  0, 1;
    return R;
}

// ─── tests ───────────────────────────────────────────────────────────────────

// 1. H_imu is the inverse of P_pred for a diagonal covariance.
TEST(ImuFactor, HessianIsInverseOfCovariance) {
    const double sigma_sq = 0.04;
    const auto P_pred = diag_cov(sigma_sq);

    imu::State x_pred, x_op;  // identical states → residual = 0

    Eigen::Matrix<double, 15, 15> H;
    Eigen::Matrix<double, 15, 1>  b;
    imu::compute_imu_hessian_gradient(x_pred, x_op, P_pred, H, b);

    // H · P_pred should equal the identity matrix
    const Eigen::Matrix<double, 15, 15> product = H * P_pred;
    EXPECT_TRUE(product.isApprox(Eigen::Matrix<double, 15, 15>::Identity(), 1e-10))
        << "H * P_pred =\n" << product;
}

// 2. When x_op == x_pred (zero error) the gradient b_imu must be zero.
TEST(ImuFactor, ZeroResidualGivesZeroGradient) {
    imu::State x_pred, x_op;

    Eigen::Matrix<double, 15, 15> H;
    Eigen::Matrix<double, 15, 1>  b;
    imu::compute_imu_hessian_gradient(x_pred, x_op, diag_cov(0.01), H, b);

    EXPECT_TRUE(b.isZero(1e-12)) << "b =\n" << b.transpose();
}

// 3. Position residual only: b reflects the position error scaled by H.
TEST(ImuFactor, PositionResidualOnlyAffectsPositionBlock) {
    imu::State x_pred, x_op;
    x_op.position = Eigen::Vector3d(1.0, 2.0, 3.0);  // pred stays at zero

    const double sigma_sq = 0.25;
    const auto P = diag_cov(sigma_sq);

    Eigen::Matrix<double, 15, 15> H;
    Eigen::Matrix<double, 15, 1>  b;
    imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b);

    // For a diagonal covariance H = (1/σ²) I, so b[0:3] = (1/σ²) * Δp
    const double h_diag = 1.0 / sigma_sq;
    EXPECT_NEAR(b(0), h_diag * 1.0, 1e-10);
    EXPECT_NEAR(b(1), h_diag * 2.0, 1e-10);
    EXPECT_NEAR(b(2), h_diag * 3.0, 1e-10);

    // Non-position blocks should remain zero
    EXPECT_TRUE(b.segment<12>(3).isZero(1e-12))
        << "Non-position part of b is non-zero: " << b.segment<12>(3).transpose();
}

// 4. Velocity residual only: b reflects the velocity error.
TEST(ImuFactor, VelocityResidualOnly) {
    imu::State x_pred, x_op;
    x_op.velocity = Eigen::Vector3d(0.5, -1.0, 0.0);

    const double sigma_sq = 1.0;
    const auto P = diag_cov(sigma_sq);

    Eigen::Matrix<double, 15, 15> H;
    Eigen::Matrix<double, 15, 1>  b;
    imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b);

    // H = I for σ²=1, so b[6:9] = Δv
    EXPECT_NEAR(b(6),  0.5, 1e-10);
    EXPECT_NEAR(b(7), -1.0, 1e-10);
    EXPECT_NEAR(b(8),  0.0, 1e-10);

    // All other blocks must be zero
    EXPECT_TRUE(b.segment<6>(0).isZero(1e-12));
    EXPECT_TRUE(b.segment<6>(9).isZero(1e-12));
}

// 5. Gyro and accel bias residuals.
TEST(ImuFactor, BiasResiduals) {
    imu::State x_pred, x_op;
    x_op.accel_bias = Eigen::Vector3d(0.1, 0.2, 0.3);
    x_op.gyro_bias  = Eigen::Vector3d(0.4, 0.5, 0.6);

    const double sigma_sq = 2.0;
    const auto P = diag_cov(sigma_sq);

    Eigen::Matrix<double, 15, 15> H;
    Eigen::Matrix<double, 15, 1>  b;
    imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b);

    const double inv_s2 = 1.0 / sigma_sq;
    // Accel bias block (indices 9-11)
    EXPECT_NEAR(b(9),  inv_s2 * 0.1, 1e-10);
    EXPECT_NEAR(b(10), inv_s2 * 0.2, 1e-10);
    EXPECT_NEAR(b(11), inv_s2 * 0.3, 1e-10);
    // Gyro bias block (indices 12-14)
    EXPECT_NEAR(b(12), inv_s2 * 0.4, 1e-10);
    EXPECT_NEAR(b(13), inv_s2 * 0.5, 1e-10);
    EXPECT_NEAR(b(14), inv_s2 * 0.6, 1e-10);

    EXPECT_TRUE(b.segment<9>(0).isZero(1e-12));
}

// 6. Rotation residual: Log(R_pred^T * R_op) around Z axis.
TEST(ImuFactor, RotationResidualSO3Log) {
    imu::State x_pred, x_op;
    const double angle = M_PI / 6.0;  // 30 degrees

    x_pred.rotation = Eigen::Matrix3d::Identity();   // R_pred = I
    x_op.rotation   = rot_z(angle);                  // R_op   = Rz(30°)

    // Expected: Log(I^T * Rz(30°)) = Log(Rz(30°)) = (0, 0, 30°)
    const auto P = diag_cov(1.0);

    Eigen::Matrix<double, 15, 15> H;
    Eigen::Matrix<double, 15, 1>  b;
    imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b);

    // H = I (σ²=1), so b[3:6] = rotation vector
    EXPECT_NEAR(b(3), 0.0,   1e-10);
    EXPECT_NEAR(b(4), 0.0,   1e-10);
    EXPECT_NEAR(b(5), angle, 1e-10);

    // Non-rotation blocks must be zero
    EXPECT_TRUE(b.segment<3>(0).isZero(1e-12));
    EXPECT_TRUE(b.segment<9>(6).isZero(1e-12));
}

// 7. Anti-symmetry: swapping x_pred and x_op negates the rotation residual.
TEST(ImuFactor, RotationResidualAntisymmetry) {
    const double angle = M_PI / 4.0;

    imu::State x_a, x_b;
    x_a.rotation = Eigen::Matrix3d::Identity();
    x_b.rotation = rot_z(angle);

    const auto P = diag_cov(1.0);
    Eigen::Matrix<double, 15, 15> H;
    Eigen::Matrix<double, 15, 1>  b_ab, b_ba;

    // r = Log(R_a^T R_b)
    imu::compute_imu_hessian_gradient(x_a, x_b, P, H, b_ab);
    // r = Log(R_b^T R_a)
    imu::compute_imu_hessian_gradient(x_b, x_a, P, H, b_ba);

    // The two rotation residuals should be negatives of each other
    EXPECT_TRUE(b_ab.segment<3>(3).isApprox(-b_ba.segment<3>(3), 1e-10))
        << "b_ab_rot = " << b_ab.segment<3>(3).transpose()
        << "\nb_ba_rot = " << b_ba.segment<3>(3).transpose();
}

// 8. H_imu is symmetric positive-definite (Cholesky succeeds, eigenvalues > 0).
TEST(ImuFactor, HessianIsSymmetricPositiveDefinite) {
    imu::State x_pred, x_op;
    x_op.position  = Eigen::Vector3d(1.0, -0.5, 0.2);
    x_op.velocity  = Eigen::Vector3d(0.3, 0.1, -0.4);
    x_op.gyro_bias = Eigen::Vector3d(0.01, -0.02, 0.005);

    // Non-diagonal P (a small perturbation to make it interesting)
    Eigen::Matrix<double, 15, 15> P = diag_cov(0.1);
    P(0, 1) = P(1, 0) = 0.005;
    P(6, 7) = P(7, 6) = 0.003;

    Eigen::Matrix<double, 15, 15> H;
    Eigen::Matrix<double, 15, 1>  b;
    imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b);

    // Symmetry
    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-12)) << "H is not symmetric";

    // Positive definiteness via Cholesky
    const Eigen::LLT<Eigen::Matrix<double, 15, 15>> llt(H);
    EXPECT_EQ(llt.info(), Eigen::Success) << "H is not positive-definite";
}

// 9. b_imu = H_imu * r  (definition check via manual residual).
TEST(ImuFactor, GradientEqualsHTimesResidual) {
    imu::State x_pred, x_op;
    x_op.position  = Eigen::Vector3d(0.3, -0.1, 0.5);
    x_op.rotation  = rot_z(0.2);
    x_op.velocity  = Eigen::Vector3d(-0.2, 0.4, 0.0);
    x_op.accel_bias = Eigen::Vector3d(0.01, -0.02, 0.03);
    x_op.gyro_bias  = Eigen::Vector3d(0.005, 0.003, -0.001);

    const auto P = diag_cov(0.05);

    Eigen::Matrix<double, 15, 15> H;
    Eigen::Matrix<double, 15, 1>  b;
    imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b);

    // Reconstruct the residual manually and verify b = H * r
    Eigen::Matrix<double, 15, 1> r;
    r.segment<3>(0)  = x_op.position  - x_pred.position;
    const Eigen::Matrix3d R_rel = x_pred.rotation.transpose() * x_op.rotation;
    const Eigen::AngleAxisd aa(R_rel);
    r.segment<3>(3)  = (aa.angle() < 1e-10) ? Eigen::Vector3d::Zero().eval()
                                             : (aa.angle() * aa.axis()).eval();
    r.segment<3>(6)  = x_op.velocity   - x_pred.velocity;
    r.segment<3>(9)  = x_op.accel_bias - x_pred.accel_bias;
    r.segment<3>(12) = x_op.gyro_bias  - x_pred.gyro_bias;

    const Eigen::Matrix<double, 15, 1> b_expected = H * r;
    EXPECT_TRUE(b.isApprox(b_expected, 1e-12))
        << "b        = " << b.transpose()
        << "\nb_expected = " << b_expected.transpose();
}

// 10. Near-zero rotation produces a near-zero rotation residual (no NaN/Inf).
TEST(ImuFactor, NearIdentityRotationIsNumericallyStable) {
    imu::State x_pred, x_op;
    // A very small rotation (< 1e-10 rad) to exercise the small-angle guard
    const double tiny = 1e-11;
    x_op.rotation = rot_z(tiny);

    Eigen::Matrix<double, 15, 15> H;
    Eigen::Matrix<double, 15, 1>  b;
    imu::compute_imu_hessian_gradient(x_pred, x_op, diag_cov(1.0), H, b);

    EXPECT_TRUE(b.allFinite()) << "b contains NaN or Inf";
    EXPECT_NEAR(b.segment<3>(3).norm(), 0.0, 1e-9);
}
