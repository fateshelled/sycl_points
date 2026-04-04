#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Dense>

#include "sycl_points/imu/imu_factor.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace imu = sycl_points::imu;

// ─── helpers ─────────────────────────────────────────────────────────────────

static constexpr float kEps      = 1e-5f;  // general tolerance
static constexpr float kEpsTight = 1e-6f;  // tight tolerance for algebraic checks

/// Build a diagonal 15×15 covariance matrix from a scalar (σ² for all).
static Eigen::Matrix<float, 15, 15> diag_cov(float sigma_sq) {
    return sigma_sq * Eigen::Matrix<float, 15, 15>::Identity();
}

/// Rotation matrix around Z axis by angle_rad (float precision).
static Eigen::Matrix3f rot_z(float angle_rad) {
    const float c = std::cos(angle_rad);
    const float s = std::sin(angle_rad);
    Eigen::Matrix3f R;
    R << c, -s, 0,
         s,  c, 0,
         0,  0, 1;
    return R;
}

// ─── tests ───────────────────────────────────────────────────────────────────

// 1. H_imu is the inverse of P_pred for a diagonal covariance.
TEST(ImuFactor, HessianIsInverseOfCovariance) {
    const float sigma_sq = 0.04f;
    const auto P_pred = diag_cov(sigma_sq);

    imu::State x_pred, x_op;  // identical states → residual = 0

    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b;
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_pred, x_op, P_pred, H, b));

    // H · P_pred should equal the identity matrix
    const Eigen::Matrix<float, 15, 15> product = H * P_pred;
    EXPECT_TRUE(product.isApprox(Eigen::Matrix<float, 15, 15>::Identity(), kEpsTight))
        << "H * P_pred =\n" << product;
}

// 2. When x_op == x_pred (zero error) the gradient b_imu must be zero.
TEST(ImuFactor, ZeroResidualGivesZeroGradient) {
    imu::State x_pred, x_op;

    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b;
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_pred, x_op, diag_cov(0.01f), H, b));

    EXPECT_TRUE(b.isZero(kEpsTight)) << "b =\n" << b.transpose();
}

// 3. Position residual only: b reflects the position error scaled by H.
TEST(ImuFactor, PositionResidualOnlyAffectsPositionBlock) {
    imu::State x_pred, x_op;
    x_op.position = Eigen::Vector3f(1.0f, 2.0f, 3.0f);  // pred stays at zero

    const float sigma_sq = 0.25f;
    const auto P = diag_cov(sigma_sq);

    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b;
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b));

    // For a diagonal covariance H = (1/σ²) I, so b[0:3] = (1/σ²) * Δp
    const float h_diag = 1.0f / sigma_sq;
    EXPECT_NEAR(b(0), h_diag * 1.0f, kEps);
    EXPECT_NEAR(b(1), h_diag * 2.0f, kEps);
    EXPECT_NEAR(b(2), h_diag * 3.0f, kEps);

    // Non-position blocks should remain zero
    EXPECT_TRUE(b.segment<12>(3).isZero(kEpsTight))
        << "Non-position part of b is non-zero: " << b.segment<12>(3).transpose();
}

// 4. Velocity residual only: b reflects the velocity error.
TEST(ImuFactor, VelocityResidualOnly) {
    imu::State x_pred, x_op;
    x_op.velocity = Eigen::Vector3f(0.5f, -1.0f, 0.0f);

    const float sigma_sq = 1.0f;
    const auto P = diag_cov(sigma_sq);

    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b;
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b));

    // H = I for σ²=1, so b[6:9] = Δv
    EXPECT_NEAR(b(6),  0.5f, kEps);
    EXPECT_NEAR(b(7), -1.0f, kEps);
    EXPECT_NEAR(b(8),  0.0f, kEps);

    // All other blocks must be zero
    EXPECT_TRUE(b.segment<6>(0).isZero(kEpsTight));
    EXPECT_TRUE(b.segment<6>(9).isZero(kEpsTight));
}

// 5. Gyro and accel bias residuals.
TEST(ImuFactor, BiasResiduals) {
    imu::State x_pred, x_op;
    x_op.accel_bias = Eigen::Vector3f(0.1f, 0.2f, 0.3f);
    x_op.gyro_bias  = Eigen::Vector3f(0.4f, 0.5f, 0.6f);

    const float sigma_sq = 2.0f;
    const auto P = diag_cov(sigma_sq);

    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b;
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b));

    const float inv_s2 = 1.0f / sigma_sq;
    // Accel bias block (indices 9-11)
    EXPECT_NEAR(b(9),  inv_s2 * 0.1f, kEps);
    EXPECT_NEAR(b(10), inv_s2 * 0.2f, kEps);
    EXPECT_NEAR(b(11), inv_s2 * 0.3f, kEps);
    // Gyro bias block (indices 12-14)
    EXPECT_NEAR(b(12), inv_s2 * 0.4f, kEps);
    EXPECT_NEAR(b(13), inv_s2 * 0.5f, kEps);
    EXPECT_NEAR(b(14), inv_s2 * 0.6f, kEps);

    EXPECT_TRUE(b.segment<9>(0).isZero(kEpsTight));
}

// 6. Rotation residual: Log(R_pred^T * R_op) around Z axis.
TEST(ImuFactor, RotationResidualSO3Log) {
    imu::State x_pred, x_op;
    const float angle = static_cast<float>(M_PI) / 6.0f;  // 30 degrees

    x_pred.rotation = Eigen::Matrix3f::Identity();  // R_pred = I
    x_op.rotation   = rot_z(angle);                 // R_op   = Rz(30°)

    // Expected: Log(I^T * Rz(30°)) = Log(Rz(30°)) = (0, 0, 30°)
    const auto P = diag_cov(1.0f);

    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b;
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b));

    // H = I (σ²=1), so b[3:6] = rotation vector
    EXPECT_NEAR(b(3), 0.0f,  kEps);
    EXPECT_NEAR(b(4), 0.0f,  kEps);
    EXPECT_NEAR(b(5), angle, kEps);

    // Non-rotation blocks must be zero
    EXPECT_TRUE(b.segment<3>(0).isZero(kEpsTight));
    EXPECT_TRUE(b.segment<9>(6).isZero(kEpsTight));
}

// 7. Anti-symmetry: swapping x_pred and x_op negates the rotation residual.
TEST(ImuFactor, RotationResidualAntisymmetry) {
    const float angle = static_cast<float>(M_PI) / 4.0f;

    imu::State x_a, x_b;
    x_a.rotation = Eigen::Matrix3f::Identity();
    x_b.rotation = rot_z(angle);

    const auto P = diag_cov(1.0f);
    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b_ab, b_ba;

    // r = Log(R_a^T R_b)
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_a, x_b, P, H, b_ab));
    // r = Log(R_b^T R_a)
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_b, x_a, P, H, b_ba));

    // The two rotation residuals should be negatives of each other
    EXPECT_TRUE(b_ab.segment<3>(3).isApprox(-b_ba.segment<3>(3), kEps))
        << "b_ab_rot = " << b_ab.segment<3>(3).transpose()
        << "\nb_ba_rot = " << b_ba.segment<3>(3).transpose();
}

// 8. H_imu is symmetric positive-definite (Cholesky succeeds, eigenvalues > 0).
TEST(ImuFactor, HessianIsSymmetricPositiveDefinite) {
    imu::State x_pred, x_op;
    x_op.position  = Eigen::Vector3f(1.0f, -0.5f, 0.2f);
    x_op.velocity  = Eigen::Vector3f(0.3f, 0.1f, -0.4f);
    x_op.gyro_bias = Eigen::Vector3f(0.01f, -0.02f, 0.005f);

    // Non-diagonal P (a small perturbation to make it interesting)
    Eigen::Matrix<float, 15, 15> P = diag_cov(0.1f);
    P(0, 1) = P(1, 0) = 0.005f;
    P(6, 7) = P(7, 6) = 0.003f;

    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b;
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b));

    // Symmetry
    EXPECT_TRUE(H.isApprox(H.transpose(), kEpsTight)) << "H is not symmetric";

    // Positive definiteness via Cholesky
    const Eigen::LLT<Eigen::Matrix<float, 15, 15>> llt(H);
    EXPECT_EQ(llt.info(), Eigen::Success) << "H is not positive-definite";
}

// 9. b_imu = H_imu * r  (definition check via manual residual).
TEST(ImuFactor, GradientEqualsHTimesResidual) {
    imu::State x_pred, x_op;
    x_op.position   = Eigen::Vector3f(0.3f, -0.1f, 0.5f);
    x_op.rotation   = rot_z(0.2f);
    x_op.velocity   = Eigen::Vector3f(-0.2f, 0.4f, 0.0f);
    x_op.accel_bias = Eigen::Vector3f(0.01f, -0.02f, 0.03f);
    x_op.gyro_bias  = Eigen::Vector3f(0.005f, 0.003f, -0.001f);

    const auto P = diag_cov(0.05f);

    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b;
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_pred, x_op, P, H, b));

    // Reconstruct the residual via the same eigen_utils path and verify b = H * r
    Eigen::Matrix<float, 15, 1> r;
    r.segment<3>(imu::State::kIdxPos)     = x_op.position  - x_pred.position;
    const Eigen::Matrix3f R_rel = x_pred.rotation.transpose() * x_op.rotation;
    const Eigen::Vector4f q_rel = sycl_points::eigen_utils::geometry::rotation_matrix_to_quaternion(R_rel);
    r.segment<3>(imu::State::kIdxRot)     = sycl_points::eigen_utils::lie::so3_log(q_rel);
    r.segment<3>(imu::State::kIdxVel)     = x_op.velocity   - x_pred.velocity;
    r.segment<3>(imu::State::kIdxAccBias) = x_op.accel_bias - x_pred.accel_bias;
    r.segment<3>(imu::State::kIdxGyrBias) = x_op.gyro_bias  - x_pred.gyro_bias;

    const Eigen::Matrix<float, 15, 1> b_expected = H * r;
    EXPECT_TRUE(b.isApprox(b_expected, kEpsTight))
        << "b        = " << b.transpose()
        << "\nb_expected = " << b_expected.transpose();
}

// 10. Near-zero rotation produces a near-zero rotation residual (no NaN/Inf).
TEST(ImuFactor, NearIdentityRotationIsNumericallyStable) {
    imu::State x_pred, x_op;
    // A very small rotation to exercise the small-angle guard in so3_log
    const float tiny = 1e-8f;
    x_op.rotation = rot_z(tiny);

    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b;
    EXPECT_TRUE(imu::compute_imu_hessian_gradient(x_pred, x_op, diag_cov(1.0f), H, b));

    EXPECT_TRUE(b.allFinite()) << "b contains NaN or Inf";
    EXPECT_NEAR(b.segment<3>(imu::State::kIdxRot).norm(), 0.0f, kEps);
}

// 11. kIdx* constants match the expected state-vector layout.
TEST(ImuFactor, StateIndexConstantsAreCorrect) {
    EXPECT_EQ(imu::State::kIdxPos,     0);
    EXPECT_EQ(imu::State::kIdxRot,     3);
    EXPECT_EQ(imu::State::kIdxVel,     6);
    EXPECT_EQ(imu::State::kIdxAccBias, 9);
    EXPECT_EQ(imu::State::kIdxGyrBias, 12);
}

// 12. Ill-conditioned (zero) covariance returns zero H and b (LDLT guard).
TEST(ImuFactor, IllConditionedCovarianceReturnsZero) {
    imu::State x_pred, x_op;
    x_op.position = Eigen::Vector3f(1.0f, 2.0f, 3.0f);

    // A zero matrix is not positive-definite → LDLT must detect this.
    const Eigen::Matrix<float, 15, 15> P_zero = Eigen::Matrix<float, 15, 15>::Zero();

    Eigen::Matrix<float, 15, 15> H;
    Eigen::Matrix<float, 15, 1>  b;
    EXPECT_FALSE(imu::compute_imu_hessian_gradient(x_pred, x_op, P_zero, H, b));

    EXPECT_TRUE(H.isZero()) << "H should be zero for ill-conditioned P";
    EXPECT_TRUE(b.isZero()) << "b should be zero for ill-conditioned P";
}
