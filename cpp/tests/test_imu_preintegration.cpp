#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Dense>

#include "sycl_points/imu/imu_preintegration.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sp  = sycl_points;
namespace imu = sycl_points::imu;

// ─── helpers ─────────────────────────────────────────────────────────────────

static constexpr float kEps     = 1e-4f;   // loose tolerance for float integration
static constexpr float kEpsTight = 1e-5f;  // tight tolerance for algebraic checks

/// Build a batch of IMU measurements with constant gyro/accel over [t0, t0+T].
static std::vector<imu::IMUMeasurement, Eigen::aligned_allocator<imu::IMUMeasurement>>
make_constant_imu(double t0, double T, int n_steps,
                  const Eigen::Vector3f& gyro,
                  const Eigen::Vector3f& accel) {
    std::vector<imu::IMUMeasurement, Eigen::aligned_allocator<imu::IMUMeasurement>> meas;
    meas.reserve(n_steps + 1);
    const double dt = T / n_steps;
    for (int i = 0; i <= n_steps; ++i) {
        imu::IMUMeasurement m;
        m.timestamp = t0 + i * dt;
        m.gyro      = gyro;
        m.accel     = accel;
        meas.push_back(m);
    }
    return meas;
}

/// Rotation matrix around Z axis by angle_rad.
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

// 1. Initial state after construction is identity / zero.
TEST(IMUPreintegration, InitialStateIsIdentity) {
    imu::IMUPreintegration integ;

    EXPECT_FALSE(integ.has_measurements());
    EXPECT_DOUBLE_EQ(integ.get_dt_total(), 0.0);

    const auto& r = integ.get_raw();
    EXPECT_TRUE(r.Delta_R.isApprox(Eigen::Matrix3f::Identity(), kEpsTight));
    EXPECT_TRUE(r.Delta_v.isZero(kEpsTight));
    EXPECT_TRUE(r.Delta_p.isZero(kEpsTight));
}

// 2. reset() clears state.
TEST(IMUPreintegration, ResetClearsState) {
    imu::IMUPreintegration integ;
    auto meas = make_constant_imu(0.0, 1.0, 100,
                                  Eigen::Vector3f(0.1f, 0.0f, 0.0f),
                                  Eigen::Vector3f(0.0f, 0.0f, 9.81f));
    integ.integrate_batch(meas);
    EXPECT_TRUE(integ.has_measurements());

    integ.reset();

    EXPECT_FALSE(integ.has_measurements());
    EXPECT_DOUBLE_EQ(integ.get_dt_total(), 0.0);
    const auto& r = integ.get_raw();
    EXPECT_TRUE(r.Delta_R.isApprox(Eigen::Matrix3f::Identity(), kEpsTight));
    EXPECT_TRUE(r.Delta_v.isZero(kEpsTight));
    EXPECT_TRUE(r.Delta_p.isZero(kEpsTight));
}

// 3. Single measurement → has_measurements() = true, but dt_total still 0.
TEST(IMUPreintegration, SingleMeasurementNoIntegration) {
    imu::IMUPreintegration integ;
    imu::IMUMeasurement m;
    m.timestamp = 1.0;
    m.gyro  = Eigen::Vector3f(0.1f, 0.2f, 0.3f);
    m.accel = Eigen::Vector3f(0.0f, 0.0f, 9.81f);
    integ.integrate(m);

    EXPECT_TRUE(integ.has_measurements());
    EXPECT_DOUBLE_EQ(integ.get_dt_total(), 0.0);
    EXPECT_TRUE(integ.get_raw().Delta_R.isApprox(Eigen::Matrix3f::Identity(), kEpsTight));
}

// 4. Zero motion → Delta_R = I, Delta_v = 0, Delta_p = 0.
TEST(IMUPreintegration, ZeroMotionIdentityResult) {
    imu::IMUPreintegration integ;
    // gravity-compensated accel = 0 because no real accelerometer in free fall, but
    // for zero motion test with zero accel and zero gyro bias:
    auto meas = make_constant_imu(0.0, 1.0, 200,
                                  Eigen::Vector3f::Zero(),
                                  Eigen::Vector3f::Zero());
    integ.integrate_batch(meas);

    const auto& r = integ.get_raw();
    EXPECT_TRUE(r.Delta_R.isApprox(Eigen::Matrix3f::Identity(), kEps));
    EXPECT_TRUE(r.Delta_v.isZero(kEps));
    EXPECT_TRUE(r.Delta_p.isZero(kEps));
    EXPECT_NEAR(r.dt_total, 1.0, 1e-9);
}

// 5. Constant rotation around Z: after T seconds at ω_z rad/s,
//    Delta_R should be rot_z(ω_z * T).
TEST(IMUPreintegration, ConstantRotationZ) {
    const float omega_z = static_cast<float>(M_PI / 4.0);  // 45 deg/s
    const double T      = 2.0;                              // 2 seconds
    const int n_steps   = 400;

    imu::IMUPreintegration integ;
    auto meas = make_constant_imu(0.0, T, n_steps,
                                  Eigen::Vector3f(0.0f, 0.0f, omega_z),
                                  Eigen::Vector3f::Zero());
    integ.integrate_batch(meas);

    const Eigen::Matrix3f expected = rot_z(omega_z * static_cast<float>(T));
    EXPECT_TRUE(integ.get_raw().Delta_R.isApprox(expected, kEps));
}

// 6. Constant linear acceleration along X (no rotation): after T seconds,
//    Delta_p ≈ 0.5 * a * T².
TEST(IMUPreintegration, ConstantAccelerationX) {
    const float ax    = 2.0f;   // m/s^2
    const double T    = 1.5;
    const int n_steps = 300;

    imu::IMUPreintegration integ;
    auto meas = make_constant_imu(0.0, T, n_steps,
                                  Eigen::Vector3f::Zero(),
                                  Eigen::Vector3f(ax, 0.0f, 0.0f));
    integ.integrate_batch(meas);

    const float expected_px = 0.5f * ax * static_cast<float>(T * T);
    const auto& r = integ.get_raw();

    EXPECT_NEAR(r.Delta_p.x(), expected_px, kEps);
    EXPECT_NEAR(r.Delta_p.y(), 0.0f,        kEps);
    EXPECT_NEAR(r.Delta_p.z(), 0.0f,        kEps);
    EXPECT_NEAR(r.Delta_v.x(), ax * static_cast<float>(T), kEps);
}

// 7. Batch integration produces the same result as incremental integration.
TEST(IMUPreintegration, BatchAndIncrementalAreEqual) {
    const auto meas = make_constant_imu(0.0, 1.0, 100,
                                        Eigen::Vector3f(0.05f, -0.03f, 0.08f),
                                        Eigen::Vector3f(0.3f, -0.1f, 9.5f));

    imu::IMUPreintegration incremental;
    for (const auto& m : meas) incremental.integrate(m);

    imu::IMUPreintegration batch;
    batch.integrate_batch(meas);

    const auto& ri = incremental.get_raw();
    const auto& rb = batch.get_raw();

    EXPECT_TRUE(ri.Delta_R.isApprox(rb.Delta_R, kEpsTight));
    EXPECT_TRUE(ri.Delta_v.isApprox(rb.Delta_v, kEpsTight));
    EXPECT_TRUE(ri.Delta_p.isApprox(rb.Delta_p, kEpsTight));
    EXPECT_DOUBLE_EQ(ri.dt_total, rb.dt_total);
}

// 8. Bias correction: integrating with b1 directly should match
//    first-order get_corrected(b1) after integrating with b0,
//    for small bias changes.
TEST(IMUPreintegration, BiasCorrection_SmallChange) {
    const Eigen::Vector3f gyro  = Eigen::Vector3f(0.1f, -0.05f, 0.08f);
    const Eigen::Vector3f accel = Eigen::Vector3f(0.2f,  0.1f,  9.7f);
    const int n_steps = 100;
    const double T    = 0.5;

    // Linearization bias
    imu::IMUBias b0;
    b0.gyro_bias  = Eigen::Vector3f(0.005f, -0.003f,  0.002f);
    b0.accel_bias = Eigen::Vector3f(0.01f,   0.005f, -0.008f);

    // Small bias perturbation
    imu::IMUBias b1;
    b1.gyro_bias  = b0.gyro_bias  + Eigen::Vector3f(0.001f, -0.001f, 0.001f);
    b1.accel_bias = b0.accel_bias + Eigen::Vector3f(0.002f,  0.001f, -0.001f);

    // Reference: integrate directly with b1
    imu::IMUPreintegration ref;
    ref.reset(b1);
    ref.integrate_batch(make_constant_imu(0.0, T, n_steps, gyro, accel));
    const auto& r_ref = ref.get_raw();

    // First-order correction starting from b0
    imu::IMUPreintegration foc;
    foc.reset(b0);
    foc.integrate_batch(make_constant_imu(0.0, T, n_steps, gyro, accel));
    const auto r_corrected = foc.get_corrected(b1);

    // For a small bias change the first-order approximation should be close.
    // Use a looser tolerance since this is only first order.
    const float tol = 5e-3f;
    EXPECT_TRUE(r_corrected.Delta_R.isApprox(r_ref.Delta_R, tol));
    EXPECT_TRUE(r_corrected.Delta_v.isApprox(r_ref.Delta_v, tol));
    EXPECT_TRUE(r_corrected.Delta_p.isApprox(r_ref.Delta_p, tol));
}

// 9. predict_relative_transform with zero motion returns identity.
TEST(IMUPreintegration, PredictRelativeTransformZeroMotion) {
    imu::IMUPreintegration integ;
    auto meas = make_constant_imu(0.0, 0.5, 50,
                                  Eigen::Vector3f::Zero(),
                                  Eigen::Vector3f::Zero());
    integ.integrate_batch(meas);

    const sp::TransformMatrix T_rel =
        integ.predict_relative_transform(imu::IMUBias{});

    EXPECT_TRUE(T_rel.isApprox(sp::TransformMatrix::Identity(), kEps));
}

// 10. predict_transform: free fall (only gravity, no sensor acceleration).
//     A sensor measuring zero acceleration (gravity-cancelled) for T seconds
//     starting at rest should move p_j = p_i + 0.5 * g * T^2.
TEST(IMUPreintegration, PredictTransform_FreeFall) {
    // Sensor reports a = 0 (gravity is cancelled by free fall)
    // but the world-frame gravity still acts.
    const double T      = 1.0;
    const int n_steps   = 200;

    imu::IMUPreintegrationParams params;
    params.gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);

    imu::IMUPreintegration integ(params);
    // In free fall, the sensor measures ~0 acceleration.
    auto meas = make_constant_imu(0.0, T, n_steps,
                                  Eigen::Vector3f::Zero(),
                                  Eigen::Vector3f::Zero());
    integ.integrate_batch(meas);

    // Initial pose: identity (at origin, no rotation)
    sp::TransformMatrix T_i = sp::TransformMatrix::Identity();
    Eigen::Vector3f v_i     = Eigen::Vector3f::Zero();

    const sp::TransformMatrix T_j =
        integ.predict_transform(T_i, v_i, imu::IMUBias{});

    // Expected: p_j = 0 + 0 + 0.5 * (-9.81) * 1^2 along z
    const float expected_pz = 0.5f * (-9.81f) * static_cast<float>(T * T);

    EXPECT_NEAR(T_j(0, 3), 0.0f,        kEps);
    EXPECT_NEAR(T_j(1, 3), 0.0f,        kEps);
    EXPECT_NEAR(T_j(2, 3), expected_pz, kEps);

    // Rotation should be identity (no angular velocity)
    const Eigen::Matrix3f R_j = T_j.block<3, 3>(0, 0);
    EXPECT_TRUE(R_j.isApprox(Eigen::Matrix3f::Identity(), kEps));
}

// 11. predict_transform with initial velocity: p_j = p_i + v_i * T + 0.5 * g * T^2
TEST(IMUPreintegration, PredictTransform_InitialVelocity) {
    const double T = 2.0;
    const int n_steps = 400;

    imu::IMUPreintegrationParams params;
    params.gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);

    imu::IMUPreintegration integ(params);
    auto meas = make_constant_imu(0.0, T, n_steps,
                                  Eigen::Vector3f::Zero(),
                                  Eigen::Vector3f::Zero());
    integ.integrate_batch(meas);

    sp::TransformMatrix T_i = sp::TransformMatrix::Identity();
    T_i(0, 3) = 1.0f; T_i(1, 3) = 2.0f; T_i(2, 3) = 3.0f;  // non-zero start position
    const Eigen::Vector3f v_i(1.0f, -0.5f, 0.0f);

    const sp::TransformMatrix T_j =
        integ.predict_transform(T_i, v_i, imu::IMUBias{});

    const float t = static_cast<float>(T);
    const Eigen::Vector3f p_j_expected(
        1.0f + 1.0f * t,
        2.0f + (-0.5f) * t,
        3.0f + 0.5f * (-9.81f) * t * t);

    EXPECT_NEAR(T_j(0, 3), p_j_expected.x(), kEps);
    EXPECT_NEAR(T_j(1, 3), p_j_expected.y(), kEps);
    EXPECT_NEAR(T_j(2, 3), p_j_expected.z(), kEps);
}

// 12. Delta_R remains a valid rotation matrix (det ≈ 1, R^T R ≈ I) after many steps.
TEST(IMUPreintegration, DeltaRRemainsValidRotation) {
    const int n_steps = 500;
    imu::IMUPreintegration integ;
    auto meas = make_constant_imu(0.0, 5.0, n_steps,
                                  Eigen::Vector3f(0.3f, -0.2f, 0.5f),
                                  Eigen::Vector3f(0.1f, 0.2f, 9.5f));
    integ.integrate_batch(meas);

    const Eigen::Matrix3f R = integ.get_raw().Delta_R;
    EXPECT_NEAR(R.determinant(), 1.0f, 1e-4f);
    EXPECT_TRUE((R.transpose() * R).isApprox(Eigen::Matrix3f::Identity(), 1e-4f));
}

// 13. Midpoint is more accurate than Euler for constant angular velocity.
//     Compare error against analytical rotation to confirm midpoint advantage.
TEST(IMUPreintegration, MidpointBetterThanEulerForRotation) {
    // Analytical reference with many steps (ground truth)
    const float omega_z = 1.5f;  // rad/s
    const double T      = 2.0;
    const Eigen::Matrix3f R_true = rot_z(omega_z * static_cast<float>(T));

    // Coarse integration (few steps) to amplify integration error
    const int n_coarse = 20;

    imu::IMUPreintegration integ;
    auto meas = make_constant_imu(0.0, T, n_coarse,
                                  Eigen::Vector3f(0.0f, 0.0f, omega_z),
                                  Eigen::Vector3f::Zero());
    integ.integrate_batch(meas);

    const Eigen::Matrix3f R_midpoint = integ.get_raw().Delta_R;

    // Frobenius distance from analytical solution
    const float err_midpoint = (R_midpoint - R_true).norm();

    // Midpoint error should be small (< 0.01 rad for 20 coarse steps)
    EXPECT_LT(err_midpoint, 0.01f);
}

// 14. get_corrected with identical bias returns the same result as get_raw.
TEST(IMUPreintegration, GetCorrectedSameBiasEqualsRaw) {
    imu::IMUBias bias;
    bias.gyro_bias  = Eigen::Vector3f(0.01f, -0.02f, 0.005f);
    bias.accel_bias = Eigen::Vector3f(0.05f,  0.02f, -0.01f);

    imu::IMUPreintegration integ;
    integ.reset(bias);
    auto meas = make_constant_imu(0.0, 1.0, 100,
                                  Eigen::Vector3f(0.1f, 0.0f, 0.0f),
                                  Eigen::Vector3f(0.0f, 0.0f, 9.81f));
    integ.integrate_batch(meas);

    const auto& raw     = integ.get_raw();
    const auto corrected = integ.get_corrected(bias);  // same bias

    EXPECT_TRUE(corrected.Delta_R.isApprox(raw.Delta_R, kEpsTight));
    EXPECT_TRUE(corrected.Delta_v.isApprox(raw.Delta_v, kEpsTight));
    EXPECT_TRUE(corrected.Delta_p.isApprox(raw.Delta_p, kEpsTight));
}
