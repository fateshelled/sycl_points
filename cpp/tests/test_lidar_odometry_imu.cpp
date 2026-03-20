#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Dense>

#include "sycl_points/imu/imu_preintegration.hpp"
#include "sycl_points/pipeline/lidar_odometry.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sp  = sycl_points;
namespace imu = sycl_points::imu;
namespace lo  = sycl_points::pipeline::lidar_odometry;

// ─── helpers ─────────────────────────────────────────────────────────────────

static constexpr float kEps = 1e-4f;

/// Build a batch of IMU measurements at rest (gravity-only accel, zero gyro)
/// over [t0, t0+T] with n_steps intervals.
static std::vector<imu::IMUMeasurement, Eigen::aligned_allocator<imu::IMUMeasurement>>
make_static_imu(double t0, double T, int n_steps,
                const Eigen::Vector3f& gravity_body = Eigen::Vector3f(0.0f, 0.0f, -9.81f)) {
    std::vector<imu::IMUMeasurement, Eigen::aligned_allocator<imu::IMUMeasurement>> meas;
    meas.reserve(static_cast<size_t>(n_steps + 1));
    const double dt = T / n_steps;
    for (int i = 0; i <= n_steps; ++i) {
        imu::IMUMeasurement m;
        m.timestamp = t0 + i * dt;
        m.gyro      = Eigen::Vector3f::Zero();
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
    p.device.vendor = "intel";
    p.device.type   = "cpu";

    // Disable most preprocessing to keep it simple
    p.scan.downsampling.polar.enable  = false;
    p.scan.downsampling.voxel.enable  = false;
    p.scan.downsampling.random.enable = false;
    p.scan.intensity_correction.enable = false;
    p.scan.preprocess.box_filter.enable             = false;
    p.scan.preprocess.angle_incidence_filter.enable = false;

    // Use voxel hash map to avoid occupancy grid complexity
    p.submap.map_type   = lo::SubmapMapType::VOXEL_HASH_MAP;
    p.submap.voxel_size = 0.5f;

    // Simple POINT_TO_POINT ICP
    p.registration.pipeline.registration.reg_type =
        sp::algorithms::registration::RegType::POINT_TO_POINT;
    p.registration.pipeline.registration.max_iterations = 5;
    p.registration.pipeline.registration.max_correspondence_distance = 100.0f;
    p.registration.min_num_points = 3;

    // No rotation constraint or photometric term
    p.registration.pipeline.registration.rotation_constraint.enable = false;
    p.registration.pipeline.registration.photometric.enable         = false;

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
    EXPECT_TRUE(p.imu.T_imu_to_lidar.isApprox(Eigen::Isometry3f::Identity()));
    EXPECT_NEAR(p.imu.preintegration.gravity.norm(), 9.81f, 1e-3f);
    EXPECT_TRUE(p.imu.bias.gyro_bias.isZero());
    EXPECT_TRUE(p.imu.bias.accel_bias.isZero());
}

// 2. add_imu_measurement() is a no-op when IMU is disabled (guard on null ptr).
TEST(LidarOdometryIMU, AddMeasurementNoOpWhenDisabled) {
    auto params        = make_test_params();
    params.imu.enable  = false;

    lo::LiDAROdometryPipeline pipeline(params);

    imu::IMUMeasurement m;
    m.timestamp = 0.0;
    m.gyro      = Eigen::Vector3f::Zero();
    m.accel     = Eigen::Vector3f(0.0f, 0.0f, 9.81f);

    // Must not crash
    EXPECT_NO_THROW(pipeline.add_imu_measurement(m));
    EXPECT_NO_THROW(pipeline.add_imu_measurement(m));
}

// 3. add_imu_measurement() feeds the integrator when IMU is enabled.
TEST(LidarOdometryIMU, AddMeasurementDoesNotCrashWhenEnabled) {
    auto params       = make_test_params();
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
    integrator.reset(imu::IMUBias(), Eigen::Matrix3f::Identity());

    // Feed static IMU data (100 Hz for 0.1 s)
    const auto batch = make_static_imu(0.0, 0.1, 10);
    integrator.integrate_batch(batch);

    EXPECT_TRUE(integrator.has_measurements());

    const sp::TransformMatrix T_rel = integrator.predict_relative_transform(imu::IMUBias());

    // Gravity-compensated relative rotation should be identity
    const Eigen::Matrix3f R_rel = T_rel.block<3, 3>(0, 0);
    EXPECT_TRUE(R_rel.isApprox(Eigen::Matrix3f::Identity(), kEps))
        << "R_rel =\n" << R_rel;

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
    const float c = std::cos(M_PIf / 2.0f);
    const float s = std::sin(M_PIf / 2.0f);
    Eigen::Matrix3f R90z;
    R90z << c, -s, 0,
            s,  c, 0,
            0,  0, 1;
    T_imu_to_lidar.linear() = R90z;

    // Construct a known T_imu_rel with pure X-translation = 1 m
    Eigen::Isometry3f T_imu_rel_iso = Eigen::Isometry3f::Identity();
    T_imu_rel_iso.translation()     = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

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
    auto params       = make_test_params();
    params.imu.enable = true;

    lo::LiDAROdometryPipeline pipeline(params);

    // Feed IMU data before the first frame
    const auto batch = make_static_imu(0.0, 0.05, 5);
    for (const auto& m : batch) {
        pipeline.add_imu_measurement(m);
    }

    // Build a minimal scan (flat cloud, x = 0..9)
    const auto& queue = *pipeline.get_device_queue();
    const auto  scan  = make_flat_cloud(queue, 10);

    const auto result = pipeline.process(scan, 0.05);
    EXPECT_EQ(result, lo::LiDAROdometryPipeline::ResultType::first_frame);
}

// 7. Second frame with IMU enabled uses IMU prediction path (no crash, success).
TEST(LidarOdometryIMU, SecondFrameWithIMUEnabled) {
    auto params       = make_test_params();
    params.imu.enable = true;

    lo::LiDAROdometryPipeline pipeline(params);

    const auto& queue = *pipeline.get_device_queue();
    const auto  scan  = make_flat_cloud(queue, 20);

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
    auto params       = make_test_params();
    params.imu.enable = false;

    lo::LiDAROdometryPipeline pipeline(params);

    const auto& queue = *pipeline.get_device_queue();
    const auto  scan  = make_flat_cloud(queue, 20);

    {
        const auto r = pipeline.process(scan, 0.1);
        EXPECT_EQ(r, lo::LiDAROdometryPipeline::ResultType::first_frame);
    }
    {
        const auto r = pipeline.process(scan, 0.2);
        EXPECT_EQ(r, lo::LiDAROdometryPipeline::ResultType::success);
    }
}

// 9. Repeated add_imu_measurement + process cycles work correctly.
TEST(LidarOdometryIMU, MultipleFramesWithIMU) {
    auto params       = make_test_params();
    params.imu.enable = true;

    lo::LiDAROdometryPipeline pipeline(params);

    const auto& queue = *pipeline.get_device_queue();
    const auto  scan  = make_flat_cloud(queue, 20);

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
        EXPECT_EQ(r, lo::LiDAROdometryPipeline::ResultType::success)
            << "Failed at frame " << (i + 2);
        t += 0.1;
    }
}
