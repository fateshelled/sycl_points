#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <numbers>

#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/pipeline/lidar_odometry.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sp = sycl_points;
namespace imu = sycl_points::imu;
namespace lo = sycl_points::pipeline::lidar_odometry;

// ─── helpers ─────────────────────────────────────────────────────────────────

static constexpr float kEps = 1e-4f;

/// Build a batch of IMU measurements at rest (gravity-only accel, zero gyro)
/// over [t0, t0+T] with n_steps intervals.
static std::vector<imu::IMUMeasurement> make_static_imu(
    double t0, double T, int n_steps, const Eigen::Vector3f& gravity_body = Eigen::Vector3f(0.0f, 0.0f, -9.81f)) {
    std::vector<imu::IMUMeasurement> meas;
    meas.reserve(static_cast<size_t>(n_steps + 1));
    const double dt = T / n_steps;
    for (int i = 0; i <= n_steps; ++i) {
        imu::IMUMeasurement m;
        m.timestamp = t0 + i * dt;
        m.gyro = Eigen::Vector3f::Zero();
        // At rest, accelerometer reads the negation of gravity in body frame.
        // With gravity (0,0,-9.81) in world and body = world, accel = -g = (0,0,9.81).
        // But IMUPreintegration gravity compensation assumes accel = specific_force = a - g_body,
        // so at rest a=0 → accel = -g_body = (0,0,9.81) in world-aligned body frame.
        m.accel = -gravity_body;  // specific force at rest
        meas.push_back(m);
    }
    return meas;
}

/// Create minimal pipeline params suitable for unit tests (uses Intel CPU device).
static lo::LidarOdometryParams make_test_params() {
    lo::LidarOdometryParams p;
    p.device.vendor = "default";
    p.device.type = "";

    // Disable IMU initial alignment
    p.imu.initial_alignment.enable = false;

    // Disable most preprocessing to keep it simple
    p.scan.downsampling.polar.enable = false;
    p.scan.downsampling.voxel.enable = false;
    p.scan.downsampling.random.enable = false;
    p.scan.intensity_correction.enable = false;
    p.scan.preprocess.box_filter.enable = false;
    p.scan.preprocess.angle_incidence_filter.enable = false;

    // Use voxel hash map to avoid occupancy grid complexity
    p.submap.map_type = sp::pipeline::odometry::SubmapMapType::VOXEL_HASH_MAP;
    p.submap.voxel_size = 0.5f;

    // Simple POINT_TO_POINT ICP
    p.registration.factor.reg_type = sp::algorithms::registration::RegType::POINT_TO_POINT;
    p.lo.registration.max_iterations = 5;
    p.registration.factor.max_correspondence_distance = 100.0f;
    p.registration.min_num_points = 3;

    // No rotation constraint term
    p.registration.factor.rotation_constraint.enable = false;

    // Disable covariance M-estimation
    p.covariance_estimation.m_estimation.enable = false;

    return p;
}

/// Build a PointCloudShared with n copies of the same XY plane of points
/// (varying x from 0 to n-1, y=0, z=0).
static sp::PointCloudShared::Ptr make_flat_cloud(const sp::sycl_utils::DeviceQueue& queue, size_t n) {
    auto cloud = std::make_shared<sp::PointCloudShared>(queue);
    cloud->points->resize(n);
    for (size_t i = 0; i < n; ++i) {
        cloud->points->at(i) = sp::PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
    }
    return cloud;
}

// ─── tests ───────────────────────────────────────────────────────────────────

// 1. IMU parameter defaults are sane (pure C++ - no SYCL device needed).
TEST(LidarOdometryIMU, IMUParamDefaults) {
    lo::Parameters p;
    EXPECT_FALSE(p.imu.enable);
    EXPECT_EQ(p.motion_prediction.mode, lo::MotionPredictionMode::GYRO_LIDAR_CV);
    EXPECT_TRUE(p.imu.T_imu_to_lidar.isApprox(Eigen::Isometry3f::Identity()));
    EXPECT_NEAR(p.imu.preintegration.gravity.norm(), 9.80665f, 1e-3f);
    EXPECT_TRUE(p.imu.bias.gyro_bias.isZero());
    EXPECT_TRUE(p.imu.bias.accel_bias.isZero());
}

TEST(LidarOdometryIMU, MotionPredictionModeConversion) {
    EXPECT_EQ(lo::MotionPredictionMode_from_string("lidar_cv"), lo::MotionPredictionMode::LIDAR_CV);
    EXPECT_EQ(lo::MotionPredictionMode_from_string("GYRO_LIDAR_CV"), lo::MotionPredictionMode::GYRO_LIDAR_CV);
    EXPECT_EQ(lo::MotionPredictionMode_from_string("imu_se3"), lo::MotionPredictionMode::IMU_SE3);
    EXPECT_EQ(lo::MotionPredictionMode_to_string(lo::MotionPredictionMode::GYRO_LIDAR_CV), "GYRO_LIDAR_CV");
    EXPECT_THROW(lo::MotionPredictionMode_from_string("invalid"), std::runtime_error);
}

TEST(LidarOdometryIMU, GyroLidarCVFusionKeepsCVTranslation) {
    lo::MotionPredictor::Params params;
    params.mode = lo::MotionPredictionMode::GYRO_LIDAR_CV;
    lo::MotionPredictor predictor(params);

    Eigen::Isometry3f odom = Eigen::Isometry3f::Identity();
    odom.translation() = Eigen::Vector3f(2.0f, -1.0f, 0.5f);

    const Eigen::Matrix3f delta_R_imu = Eigen::AngleAxisf(0.4f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    lo::MotionPredictionCandidates candidates;
    candidates.gyro_delta_rotation_lidar = delta_R_imu;

    const auto reg_result = std::make_shared<sp::algorithms::registration::RegistrationResult>();
    const Eigen::Isometry3f fused =
        predictor.predict(Eigen::Vector3f(1.0f, 2.0f, 3.0f), Eigen::AngleAxisf(0.2f, Eigen::Vector3f::UnitX()), odom,
                          1.0f, reg_result, false, candidates);

    EXPECT_TRUE(fused.translation().isApprox(odom.translation() + Eigen::Vector3f(1.0f, 2.0f, 3.0f), kEps));
    EXPECT_TRUE(fused.rotation().isApprox(delta_R_imu, kEps));
}

// 2. add_imu_measurement() is a no-op when IMU is disabled (guard on null ptr).
TEST(LidarOdometryIMU, AddMeasurementNoOpWhenDisabled) {
    auto params = make_test_params();
    params.imu.enable = false;

    lo::LiDAROdometryPipeline pipeline(params);

    imu::IMUMeasurement m;
    m.timestamp = 0.0;
    m.gyro = Eigen::Vector3f::Zero();
    m.accel = Eigen::Vector3f(0.0f, 0.0f, 9.81f);

    // Must not crash
    EXPECT_NO_THROW(pipeline.add_imu_measurement(m));
    EXPECT_NO_THROW(pipeline.add_imu_measurement(m));
}

// 3. add_imu_measurement() feeds the integrator when IMU is enabled.
TEST(LidarOdometryIMU, AddMeasurementDoesNotCrashWhenEnabled) {
    auto params = make_test_params();
    params.imu.enable = true;

    lo::LiDAROdometryPipeline pipeline(params);

    const auto batch = make_static_imu(0.0, 0.1, 10);
    for (const auto& m : batch) {
        EXPECT_NO_THROW(pipeline.add_imu_measurement(m));
    }
}

// 4. IMU motion prediction math with identity extrinsic and zero motion.
//    A stationary IMU (specific force = -gravity_body, gyro = 0) should
//    produce a near-identity relative transform (gravity compensated).
TEST(LidarOdometryIMU, ZeroMotionIMUGivesNearIdentityRelativeTransform) {
    imu::IMUPreintegrationParams imu_params;
    imu_params.gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);

    imu::IMUPreintegration integrator(imu_params);

    // Reset at identity world orientation
    integrator.reset(imu::IMUBias());

    // Feed static IMU data (100 Hz for 0.1 s)
    const auto batch = make_static_imu(0.0, 0.1, 10);
    integrator.integrate_batch(batch);

    EXPECT_TRUE(integrator.has_measurements());

    const sp::TransformMatrix T_rel =
        integrator.predict_relative_transform(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero(), imu::IMUBias());

    // Gravity-compensated relative rotation should be identity
    const Eigen::Matrix3f R_rel = T_rel.block<3, 3>(0, 0);
    EXPECT_TRUE(R_rel.isApprox(Eigen::Matrix3f::Identity(), kEps)) << "R_rel =\n" << R_rel;

    // Translation should be near zero (device at rest)
    const Eigen::Vector3f t_rel = T_rel.block<3, 1>(0, 3);
    EXPECT_LT(t_rel.norm(), kEps) << "t_rel = " << t_rel.transpose();
}

// 5. IMU extrinsic conversion: 90° rotation around Z from IMU to LiDAR.
//    A pure X-translation in the IMU frame should become a Y-translation in
//    the LiDAR frame after the extrinsic conversion.
TEST(LidarOdometryIMU, ExtrinsicConversionRotatesRelativeTranslation) {
    // Extrinsic: IMU is rotated 90° around Z relative to LiDAR
    // p_lidar = R_90z * p_imu
    Eigen::Isometry3f T_imu_to_lidar = Eigen::Isometry3f::Identity();
    const float c = std::cos(std::numbers::pi_v<float> / 2.0f);
    const float s = std::sin(std::numbers::pi_v<float> / 2.0f);
    Eigen::Matrix3f R90z;
    R90z << c, -s, 0, s, c, 0, 0, 0, 1;
    T_imu_to_lidar.linear() = R90z;

    // Construct a known T_imu_rel with pure X-translation = 1 m
    Eigen::Isometry3f T_imu_rel_iso = Eigen::Isometry3f::Identity();
    T_imu_rel_iso.translation() = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

    // Apply: T_lidar_rel = T_imu_to_lidar * T_imu_rel * T_imu_to_lidar^{-1}
    const Eigen::Isometry3f T_lidar_rel = T_imu_to_lidar * T_imu_rel_iso * T_imu_to_lidar.inverse();

    // Expected: translation becomes Y = 1 (because R90z * [1,0,0] = [0,1,0])
    const Eigen::Vector3f t_lidar = T_lidar_rel.translation();
    EXPECT_NEAR(t_lidar.x(), 0.0f, kEps);
    EXPECT_NEAR(t_lidar.y(), 1.0f, kEps);
    EXPECT_NEAR(t_lidar.z(), 0.0f, kEps);
}

// 6. Process first frame succeeds and IMU reset is called (no crash).
TEST(LidarOdometryIMU, FirstFrameWithIMUEnabled) {
    auto params = make_test_params();
    params.imu.enable = true;

    lo::LiDAROdometryPipeline pipeline(params);

    // Feed IMU data before the first frame
    const auto batch = make_static_imu(0.0, 0.05, 5);
    for (const auto& m : batch) {
        pipeline.add_imu_measurement(m);
    }

    // Build a minimal scan (flat cloud, x = 0..9)
    const auto& queue = *pipeline.get_device_queue();
    const auto scan = make_flat_cloud(queue, 10);

    const auto result = pipeline.process(scan, 0.05);
    EXPECT_EQ(result, lo::LiDAROdometryPipeline::ResultType::first_frame);
}

// 7. Second frame with IMU enabled uses IMU prediction path (no crash, success).
TEST(LidarOdometryIMU, SecondFrameWithIMUEnabled) {
    auto params = make_test_params();
    params.imu.enable = true;

    lo::LiDAROdometryPipeline pipeline(params);

    const auto& queue = *pipeline.get_device_queue();
    const auto scan = make_flat_cloud(queue, 20);

    // First frame
    {
        const auto batch = make_static_imu(0.0, 0.1, 10);
        for (const auto& m : batch) pipeline.add_imu_measurement(m);
        const auto r = pipeline.process(scan, 0.1);
        EXPECT_EQ(r, lo::LiDAROdometryPipeline::ResultType::first_frame);
    }

    // Second frame: feed IMU between frames, then process
    {
        const auto batch = make_static_imu(0.1, 0.1, 10);
        for (const auto& m : batch) pipeline.add_imu_measurement(m);
        const auto r = pipeline.process(scan, 0.2);
        EXPECT_EQ(r, lo::LiDAROdometryPipeline::ResultType::success);
    }
}

// 8. Second frame with IMU disabled falls back to adaptive prediction (no crash).
TEST(LidarOdometryIMU, SecondFrameWithIMUDisabledFallsBack) {
    auto params = make_test_params();
    params.imu.enable = false;
    params.motion_prediction.mode = lo::MotionPredictionMode::GYRO_LIDAR_CV;

    lo::LiDAROdometryPipeline pipeline(params);

    const auto& queue = *pipeline.get_device_queue();
    const auto scan = make_flat_cloud(queue, 20);

    {
        const auto r = pipeline.process(scan, 0.1);
        EXPECT_EQ(r, lo::LiDAROdometryPipeline::ResultType::first_frame);
    }
    {
        const auto r = pipeline.process(scan, 0.2);
        EXPECT_EQ(r, lo::LiDAROdometryPipeline::ResultType::success);
    }
}

TEST(LidarOdometryIMU, IMUAvailableWithLidarAndGyroCVModes) {
    for (const auto mode : {lo::MotionPredictionMode::LIDAR_CV, lo::MotionPredictionMode::GYRO_LIDAR_CV}) {
        auto params = make_test_params();
        params.imu.enable = true;
        params.motion_prediction.mode = mode;

        lo::LiDAROdometryPipeline pipeline(params);
        const auto scan = make_flat_cloud(*pipeline.get_device_queue(), 20);

        for (const auto& m : make_static_imu(0.0, 0.1, 10)) pipeline.add_imu_measurement(m);
        EXPECT_EQ(pipeline.process(scan, 0.1), lo::LiDAROdometryPipeline::ResultType::first_frame);

        for (const auto& m : make_static_imu(0.1, 0.1, 10)) pipeline.add_imu_measurement(m);
        EXPECT_EQ(pipeline.process(scan, 0.2), lo::LiDAROdometryPipeline::ResultType::success)
            << "mode=" << lo::MotionPredictionMode_to_string(mode);
    }
}

// 9. Repeated add_imu_measurement + process cycles work correctly.
TEST(LidarOdometryIMU, MultipleFramesWithIMU) {
    auto params = make_test_params();
    params.imu.enable = true;

    lo::LiDAROdometryPipeline pipeline(params);

    const auto& queue = *pipeline.get_device_queue();
    const auto scan = make_flat_cloud(queue, 20);

    double t = 0.0;

    // First frame
    {
        auto batch = make_static_imu(t, 0.1, 10);
        for (const auto& m : batch) pipeline.add_imu_measurement(m);
        pipeline.process(scan, t + 0.1);
        t += 0.1;
    }

    // Three more frames
    for (int i = 0; i < 3; ++i) {
        auto batch = make_static_imu(t, 0.1, 10);
        for (const auto& m : batch) pipeline.add_imu_measurement(m);
        const auto r = pipeline.process(scan, t + 0.1);
        EXPECT_EQ(r, lo::LiDAROdometryPipeline::ResultType::success) << "Failed at frame " << (i + 2);
        t += 0.1;
    }
}

TEST(LidarOdometryIMU, RejectsSparseFirstFrame) {
    auto params = make_test_params();
    lo::LiDAROdometryPipeline pipeline(params);
    const auto sparse_scan = make_flat_cloud(*pipeline.get_device_queue(), params.registration.min_num_points);

    const Eigen::Isometry3f initial_odom = pipeline.get_odom();
    EXPECT_EQ(pipeline.process(sparse_scan, 0.1), lo::LiDAROdometryPipeline::ResultType::small_number_of_points);
    EXPECT_TRUE(pipeline.get_odom().isApprox(initial_odom));
}

TEST(LidarOdometryIMU, SparseInitializedFrameUsesPredictionOnlyFallback) {
    auto params = make_test_params();
    params.motion_prediction.mode = lo::MotionPredictionMode::LIDAR_CV;
    lo::LiDAROdometryPipeline pipeline(params);

    const auto normal_scan = make_flat_cloud(*pipeline.get_device_queue(), 20);
    const auto sparse_scan = make_flat_cloud(*pipeline.get_device_queue(), params.registration.min_num_points);

    ASSERT_EQ(pipeline.process(normal_scan, 0.1), lo::LiDAROdometryPipeline::ResultType::first_frame);
    EXPECT_EQ(pipeline.process(normal_scan, 0.2), lo::LiDAROdometryPipeline::ResultType::success);
    const size_t keyframe_count = pipeline.get_keyframe_poses().size();

    EXPECT_EQ(pipeline.process(sparse_scan, 0.3), lo::LiDAROdometryPipeline::ResultType::prediction_only);
    EXPECT_EQ(pipeline.get_registration_result().inlier, 0u);
    EXPECT_FALSE(pipeline.get_registration_result().converged);
    EXPECT_EQ(pipeline.get_keyframe_poses().size(), keyframe_count);

    // The fallback frame is accepted as the current time boundary.
    EXPECT_EQ(pipeline.process(normal_scan, 0.25), lo::LiDAROdometryPipeline::ResultType::old_timestamp);
    EXPECT_EQ(pipeline.process(normal_scan, 0.4), lo::LiDAROdometryPipeline::ResultType::success);
}

TEST(LidarOdometryIMU, ConsecutiveSparseFramesWithIMUKeepPropagating) {
    auto params = make_test_params();
    params.imu.enable = true;
    params.motion_prediction.mode = lo::MotionPredictionMode::IMU_SE3;
    lo::LiDAROdometryPipeline pipeline(params);

    const auto normal_scan = make_flat_cloud(*pipeline.get_device_queue(), 20);
    const auto sparse_scan = make_flat_cloud(*pipeline.get_device_queue(), params.registration.min_num_points);

    for (const auto& m : make_static_imu(0.0, 0.1, 10)) pipeline.add_imu_measurement(m);
    ASSERT_EQ(pipeline.process(normal_scan, 0.1), lo::LiDAROdometryPipeline::ResultType::first_frame);

    for (const auto& m : make_static_imu(0.1, 0.1, 10)) pipeline.add_imu_measurement(m);
    EXPECT_EQ(pipeline.process(sparse_scan, 0.2), lo::LiDAROdometryPipeline::ResultType::prediction_only);

    for (const auto& m : make_static_imu(0.2, 0.1, 10)) pipeline.add_imu_measurement(m);
    EXPECT_EQ(pipeline.process(sparse_scan, 0.3), lo::LiDAROdometryPipeline::ResultType::prediction_only);
    EXPECT_TRUE(pipeline.get_odom().matrix().allFinite());

    for (const auto& m : make_static_imu(0.3, 0.1, 10)) pipeline.add_imu_measurement(m);
    EXPECT_EQ(pipeline.process(normal_scan, 0.4), lo::LiDAROdometryPipeline::ResultType::success);
}

TEST(LidarOdometryIMU, VelocityCorrectorPreservesPredictionOnlyVelocity) {
    imu::IMUPreintegrationParams imu_params;
    imu_params.gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);
    imu::IMUPreintegration integrator(imu_params);
    integrator.reset(imu::IMUBias());
    integrator.integrate_batch(make_static_imu(0.0, 0.1, 10));

    imu::IMUVelocityCorrector corrector;
    const Eigen::Vector3f initial_velocity(1.0f, -0.5f, 0.25f);
    EXPECT_TRUE(
        corrector.get_reset_velocity(integrator, imu::IMUBias(), initial_velocity).isApprox(initial_velocity, kEps));

    const Eigen::Vector3f propagated = corrector.update_without_icp(Eigen::Matrix3f::Identity(), imu_params.gravity);
    EXPECT_TRUE(propagated.isApprox(initial_velocity, 1e-3f));

    integrator.reset(imu::IMUBias());
    integrator.integrate_batch(make_static_imu(0.1, 0.1, 10));
    const Eigen::Vector3f next = corrector.get_reset_velocity(integrator, imu::IMUBias(), Eigen::Vector3f::Zero());
    EXPECT_TRUE(next.isApprox(propagated, kEps));
}
