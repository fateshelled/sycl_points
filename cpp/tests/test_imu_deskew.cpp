#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <deque>

#include "sycl_points/algorithms/deskew/imu_deskew.hpp"
#include "sycl_points/algorithms/deskew/relative_pose_deskew.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points::algorithms::deskew {
namespace {

// ─── helpers ─────────────────────────────────────────────────────────────────

static constexpr float kEps = 5e-3f;  // 5 mm tolerance

/// Build a SYCL device queue using the default CPU selector.
static sycl_utils::DeviceQueue make_queue() {
    sycl::device dev(sycl_utils::device_selector::default_selector_v);
    return sycl_utils::DeviceQueue(dev);
}

/// Generate IMU measurements with constant gyro and accel over [t0, t0+T].
static std::deque<imu::IMUMeasurement> make_imu_buffer(
    double t0, double T, int n_steps, const Eigen::Vector3f& gyro,
    const Eigen::Vector3f& specific_force /* = accel_meas = a - g_body */) {
    std::deque<imu::IMUMeasurement> buf;
    const double dt = T / n_steps;
    for (int i = 0; i <= n_steps; ++i) {
        imu::IMUMeasurement m;
        m.timestamp = t0 + i * dt;
        m.gyro = gyro;
        m.accel = specific_force;
        buf.push_back(m);
    }
    return buf;
}

// ─── tests ───────────────────────────────────────────────────────────────────

// 1. Deskew a scan with pure constant-rate rotation around Z.
//    Points observed at various times during the scan should be corrected to
//    their world-frame coordinates (i.e. what they look like at scan start).
TEST(IMUDeskewTest, PureRotationDeskew) {
    auto queue = make_queue();
    PointCloudShared cloud(queue);

    constexpr double scan_start_sec = 1.0;
    constexpr double scan_duration_sec = 0.1;  // 100 ms
    cloud.start_time_ms = scan_start_sec * 1e3;
    cloud.end_time_ms = (scan_start_sec + scan_duration_sec) * 1e3;

    // Sensor rotates at 2π/4 = π/2 rad/s around Z over 100 ms → 9° rotation.
    const float omega_z = static_cast<float>(M_PI / 2.0);

    // Five world-frame points (known ground truth).
    const std::vector<Eigen::Vector3f> world_points = {
        {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.5f}, {-1.0f, 0.5f, 0.0f}, {0.5f, -0.5f, 1.0f},
    };

    // Five equally-spaced timestamps within the scan.
    const std::vector<float> offsets_ms = {0.0f, 25.0f, 50.0f, 75.0f, 100.0f};

    for (size_t i = 0; i < world_points.size(); ++i) {
        const float t_rel_sec = offsets_ms[i] * 1e-3f;

        // Pose of sensor at time t: rotated around Z by omega_z * t_rel
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
        pose.linear() = Eigen::AngleAxisf(omega_z * t_rel_sec, Eigen::Vector3f::UnitZ()).toRotationMatrix();

        // Point in sensor frame at time t (what the LiDAR measures).
        const Eigen::Vector3f p_sensor = pose.inverse() * world_points[i];

        PointType pt;
        pt << p_sensor.x(), p_sensor.y(), p_sensor.z(), 1.0f;
        cloud.points->push_back(pt);
        cloud.timestamp_offsets->push_back(offsets_ms[i]);
    }

    // IMU at 200 Hz over the scan window (with a generous bracket).
    // specific_force = 0 because gravity is zeroed out in preint_params — any
    // non-zero accel value would be integrated without cancellation and pollute
    // the translation channel, breaking the pure-rotation invariant.
    const auto imu_buf = make_imu_buffer(scan_start_sec - 0.02, scan_duration_sec + 0.04, 24,
                                         Eigen::Vector3f(0.0f, 0.0f, omega_z),  // gyro = omega_z around Z
                                         Eigen::Vector3f::Zero());              // specific force = 0 (gravity disabled)

    // No-gravity model: set gravity to zero for a clean rotation-only test.
    imu::IMUPreintegrationParams preint_params;
    preint_params.gravity = Eigen::Vector3f::Zero();

    PointCloudShared deskewed(queue);
    IMUDeskewStatus status;
    const bool ok = deskew_point_cloud_imu(cloud, deskewed, imu_buf, scan_start_sec,
                                           Eigen::Isometry3f::Identity(),  // T_imu_to_lidar = I
                                           imu::IMUBias(), preint_params,
                                           Eigen::Matrix3f::Identity(),  // R_world_imu = I
                                           &status);

    ASSERT_TRUE(ok) << "deskew failed with status " << static_cast<int>(status);
    ASSERT_EQ(deskewed.size(), world_points.size());

    for (size_t i = 0; i < world_points.size(); ++i) {
        const Eigen::Vector3f corrected = (*deskewed.points)[i].head<3>();
        EXPECT_NEAR((corrected - world_points[i]).norm(), 0.0f, kEps)
            << "Point " << i << ": corrected=" << corrected.transpose() << "  expected=" << world_points[i].transpose();
    }
}

// 2. Pure translation (zero gyro, zero gravity): constant acceleration.
//    After deskew all points should be at their scan-start sensor positions.
TEST(IMUDeskewTest, PureTranslationDeskew) {
    auto queue = make_queue();
    PointCloudShared cloud(queue);

    constexpr double scan_start_sec = 2.0;
    constexpr double scan_duration_sec = 0.1;
    cloud.start_time_ms = scan_start_sec * 1e3;
    cloud.end_time_ms = (scan_start_sec + scan_duration_sec) * 1e3;

    // Sensor accelerates at 1 m/s² along X (starting from rest).
    // At time t: position offset = 0.5 * 1.0 * t^2.
    const float accel_x = 1.0f;

    // Three world points.
    const std::vector<Eigen::Vector3f> world_pts = {
        {2.0f, 0.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
    };
    const std::vector<float> offsets_ms = {0.0f, 50.0f, 100.0f};

    for (size_t i = 0; i < world_pts.size(); ++i) {
        const float t = offsets_ms[i] * 1e-3f;
        // Sensor displacement from scan start: 0.5 * accel_x * t^2
        const Eigen::Vector3f sensor_pos(0.5f * accel_x * t * t, 0.0f, 0.0f);
        // Point in sensor frame = world_point - sensor_position (no rotation)
        const Eigen::Vector3f p_sensor = world_pts[i] - sensor_pos;

        PointType pt;
        pt << p_sensor.x(), p_sensor.y(), p_sensor.z(), 1.0f;
        cloud.points->push_back(pt);
        cloud.timestamp_offsets->push_back(offsets_ms[i]);
    }

    // IMU measurements: accel_x only, no gyro, gravity zeroed out.
    const auto imu_buf = make_imu_buffer(scan_start_sec - 0.02, scan_duration_sec + 0.04, 24,
                                         Eigen::Vector3f::Zero(),              // no rotation
                                         Eigen::Vector3f(accel_x, 0.0f, 0.0f)  // specific force
    );

    imu::IMUPreintegrationParams preint_params;
    preint_params.gravity = Eigen::Vector3f::Zero();  // no gravity for clean test

    PointCloudShared deskewed(queue);
    IMUDeskewStatus status;
    const bool ok = deskew_point_cloud_imu(cloud, deskewed, imu_buf, scan_start_sec, Eigen::Isometry3f::Identity(),
                                           imu::IMUBias(), preint_params, Eigen::Matrix3f::Identity(), &status);

    ASSERT_TRUE(ok) << "deskew failed: " << static_cast<int>(status);
    for (size_t i = 0; i < world_pts.size(); ++i) {
        const Eigen::Vector3f corrected = (*deskewed.points)[i].head<3>();
        EXPECT_NEAR((corrected - world_pts[i]).norm(), 0.0f, kEps) << "Point " << i << " mismatch";
    }
}

// 3. In-place alias: input and output are the same cloud — no corruption.
TEST(IMUDeskewTest, InPlaceAliasNoCorruption) {
    auto queue = make_queue();
    PointCloudShared cloud(queue);

    constexpr double scan_start_sec = 0.5;
    cloud.start_time_ms = scan_start_sec * 1e3;
    cloud.end_time_ms = cloud.start_time_ms + 100.0;

    for (int i = 0; i < 10; ++i) {
        PointType pt;
        pt << static_cast<float>(i), 0.0f, 0.0f, 1.0f;
        cloud.points->push_back(pt);
        cloud.timestamp_offsets->push_back(static_cast<float>(i) * 10.0f);
    }

    const auto imu_buf = make_imu_buffer(scan_start_sec - 0.02, 0.14, 20, Eigen::Vector3f(0.0f, 0.0f, 0.1f),
                                         Eigen::Vector3f(0.0f, 0.0f, 9.81f));

    imu::IMUPreintegrationParams preint_params;
    preint_params.gravity = Eigen::Vector3f::Zero();

    // Pass cloud as both input and output (in-place).
    IMUDeskewStatus status;
    const bool ok = deskew_point_cloud_imu(cloud, cloud, imu_buf, scan_start_sec, Eigen::Isometry3f::Identity(),
                                           imu::IMUBias(), preint_params, Eigen::Matrix3f::Identity(), &status);

    ASSERT_TRUE(ok) << "in-place deskew failed: " << static_cast<int>(status);
    ASSERT_EQ(cloud.size(), 10u);
    for (size_t i = 0; i < cloud.size(); ++i) {
        for (int d = 0; d < 4; ++d) {
            EXPECT_TRUE(std::isfinite((*cloud.points)[i][d]));
        }
    }
}

// 4. Empty IMU buffer → insufficient_imu_coverage.
TEST(IMUDeskewTest, InsufficientIMUDataReturnsFalse) {
    auto queue = make_queue();
    PointCloudShared cloud(queue);
    cloud.start_time_ms = 1000.0;
    cloud.end_time_ms = 1100.0;
    PointType pt;
    pt << 1.0f, 0.0f, 0.0f, 1.0f;
    cloud.points->push_back(pt);
    cloud.timestamp_offsets->push_back(0.0f);

    std::deque<imu::IMUMeasurement> empty_buf;

    IMUDeskewStatus status;
    PointCloudShared output(queue);
    const bool ok = deskew_point_cloud_imu(cloud, output, empty_buf, 1.0, Eigen::Isometry3f::Identity(), imu::IMUBias(),
                                           imu::IMUPreintegrationParams(), Eigen::Matrix3f::Identity(), &status);

    EXPECT_FALSE(ok);
    EXPECT_EQ(status, IMUDeskewStatus::insufficient_imu_coverage);
}

// 5. Cloud without timestamp offsets → no_timestamps.
TEST(IMUDeskewTest, NoTimestampsReturnsFalse) {
    auto queue = make_queue();
    PointCloudShared cloud(queue);
    cloud.start_time_ms = 1000.0;
    cloud.end_time_ms = 1100.0;
    PointType pt;
    pt << 1.0f, 0.0f, 0.0f, 1.0f;
    cloud.points->push_back(pt);
    // timestamp_offsets is intentionally left empty (no timestamps).

    const auto imu_buf = make_imu_buffer(0.98, 0.14, 20, Eigen::Vector3f::Zero(), Eigen::Vector3f(0.0f, 0.0f, 9.81f));

    IMUDeskewStatus status;
    PointCloudShared output(queue);
    const bool ok = deskew_point_cloud_imu(cloud, output, imu_buf, 1.0, Eigen::Isometry3f::Identity(), imu::IMUBias(),
                                           imu::IMUPreintegrationParams(), Eigen::Matrix3f::Identity(), &status);

    EXPECT_FALSE(ok);
    EXPECT_EQ(status, IMUDeskewStatus::no_timestamps);
}

// 6. Zero scan duration → invalid_scan_duration.
TEST(IMUDeskewTest, ZeroScanDurationReturnsFalse) {
    auto queue = make_queue();
    PointCloudShared cloud(queue);
    cloud.start_time_ms = 1000.0;
    cloud.end_time_ms = 1000.0;  // same → duration = 0
    PointType pt;
    pt << 1.0f, 0.0f, 0.0f, 1.0f;
    cloud.points->push_back(pt);
    cloud.timestamp_offsets->push_back(0.0f);

    const auto imu_buf = make_imu_buffer(0.98, 0.14, 20, Eigen::Vector3f::Zero(), Eigen::Vector3f(0.0f, 0.0f, 9.81f));

    IMUDeskewStatus status;
    PointCloudShared output(queue);
    const bool ok = deskew_point_cloud_imu(cloud, output, imu_buf, 1.0, Eigen::Isometry3f::Identity(), imu::IMUBias(),
                                           imu::IMUPreintegrationParams(), Eigen::Matrix3f::Identity(), &status);

    EXPECT_FALSE(ok);
    EXPECT_EQ(status, IMUDeskewStatus::invalid_scan_duration);
}

// 7. IMU data covers only the first half → insufficient_imu_coverage.
TEST(IMUDeskewTest, PartialIMUCoverageReturnsFalse) {
    auto queue = make_queue();
    PointCloudShared cloud(queue);
    cloud.start_time_ms = 1000.0;
    cloud.end_time_ms = 1100.0;
    PointType pt;
    pt << 1.0f, 0.0f, 0.0f, 1.0f;
    cloud.points->push_back(pt);
    cloud.timestamp_offsets->push_back(0.0f);

    // Only covers [0.98, 1.04] — does not reach scan_end = 1.1 s.
    // With kMarginSec = 0.05 the coverage threshold is scan_end - kMarginSec = 1.05.
    // Last measurement at 1.04 < 1.05 → coverage check fails as expected.
    const auto imu_buf = make_imu_buffer(0.98, 0.06, 10, Eigen::Vector3f::Zero(), Eigen::Vector3f(0.0f, 0.0f, 9.81f));

    IMUDeskewStatus status;
    PointCloudShared output(queue);
    const bool ok = deskew_point_cloud_imu(cloud, output, imu_buf, 1.0, Eigen::Isometry3f::Identity(), imu::IMUBias(),
                                           imu::IMUPreintegrationParams(), Eigen::Matrix3f::Identity(), &status);

    EXPECT_FALSE(ok);
    EXPECT_EQ(status, IMUDeskewStatus::insufficient_imu_coverage);
}

// 8. Compare IMU deskew with constant-velocity deskew for identical constant motion.
//    Both methods should produce points close to the world-frame positions.
TEST(IMUDeskewTest, MatchesConstantVelocityApproximately) {
    auto queue = make_queue();

    constexpr double scan_start_sec = 0.0;
    constexpr double scan_duration_sec = 0.1;

    const float omega_z = static_cast<float>(M_PI / 4.0);  // 45°/s

    const Eigen::Transform<float, 3, 1> start_pose = Eigen::Transform<float, 3, 1>::Identity();
    Eigen::Transform<float, 3, 1> end_pose = Eigen::Transform<float, 3, 1>::Identity();
    end_pose.linear() =
        Eigen::AngleAxisf(omega_z * static_cast<float>(scan_duration_sec), Eigen::Vector3f::UnitZ()).toRotationMatrix();

    const Eigen::Vector3f world_point(1.0f, 0.5f, 0.0f);
    const std::vector<float> offsets_ms = {0.0f, 50.0f, 100.0f};

    // Build identical clouds for both methods.
    PointCloudShared cloud_imu(queue), cloud_cv(queue);
    cloud_imu.start_time_ms = cloud_cv.start_time_ms = scan_start_sec * 1e3;
    cloud_imu.end_time_ms = cloud_cv.end_time_ms = (scan_start_sec + scan_duration_sec) * 1e3;

    for (float offset_ms : offsets_ms) {
        const float t = offset_ms * 1e-3f;
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
        pose.linear() = Eigen::AngleAxisf(omega_z * t, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        const Eigen::Vector3f p_sensor = pose.inverse() * world_point;

        PointType pt;
        pt << p_sensor.x(), p_sensor.y(), p_sensor.z(), 1.0f;
        cloud_imu.points->push_back(pt);
        cloud_imu.timestamp_offsets->push_back(offset_ms);
        cloud_cv.points->push_back(pt);
        cloud_cv.timestamp_offsets->push_back(offset_ms);
    }

    // IMU deskew (zero gravity for a clean comparison).
    // specific_force = 0 so the disabled gravity term does not pollute translation.
    const auto imu_buf = make_imu_buffer(scan_start_sec - 0.02, scan_duration_sec + 0.04, 24,
                                         Eigen::Vector3f(0.0f, 0.0f, omega_z), Eigen::Vector3f::Zero());

    imu::IMUPreintegrationParams preint_params;
    preint_params.gravity = Eigen::Vector3f::Zero();

    PointCloudShared deskewed_imu(queue), deskewed_cv(queue);
    ASSERT_TRUE(deskew_point_cloud_imu(cloud_imu, deskewed_imu, imu_buf, scan_start_sec, Eigen::Isometry3f::Identity(),
                                       imu::IMUBias(), preint_params, Eigen::Matrix3f::Identity()));
    ASSERT_TRUE(deskew_point_cloud_constant_velocity(cloud_cv, deskewed_cv, start_pose, end_pose));

    // Both methods should correct points close to the world-frame position.
    for (size_t i = 0; i < offsets_ms.size(); ++i) {
        const Eigen::Vector3f p_imu = (*deskewed_imu.points)[i].head<3>();
        const Eigen::Vector3f p_cv = (*deskewed_cv.points)[i].head<3>();
        EXPECT_NEAR((p_imu - world_point).norm(), 0.0f, kEps) << "IMU deskew: point " << i << " error";
        EXPECT_NEAR((p_cv - world_point).norm(), 0.0f, kEps) << "CV deskew: point " << i << " error";
    }
}

// 9. Normals and covariances are rotated consistently with points.
TEST(IMUDeskewTest, NormalsAndCovariancesRotated) {
    auto queue = make_queue();
    PointCloudShared cloud(queue);

    constexpr double scan_start_sec = 0.0;
    constexpr double scan_duration_sec = 0.1;
    cloud.start_time_ms = scan_start_sec * 1e3;
    cloud.end_time_ms = (scan_start_sec + scan_duration_sec) * 1e3;

    const float omega_z = static_cast<float>(M_PI / 2.0);

    const Eigen::Vector3f world_point(1.0f, 1.0f, 0.0f);
    const Eigen::Vector3f world_normal = Eigen::Vector3f::UnitZ();
    Eigen::Matrix3f world_cov = Eigen::Matrix3f::Zero();
    world_cov.diagonal() = Eigen::Vector3f(0.01f, 0.02f, 0.03f);

    const std::vector<float> offsets_ms = {0.0f, 50.0f, 100.0f};

    for (float offset_ms : offsets_ms) {
        const float t = offset_ms * 1e-3f;
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
        pose.linear() = Eigen::AngleAxisf(omega_z * t, Eigen::Vector3f::UnitZ()).toRotationMatrix();

        const Eigen::Vector3f p_sensor = pose.inverse() * world_point;
        const Eigen::Vector3f n_sensor = pose.linear().transpose() * world_normal;
        const Eigen::Matrix3f cov_sensor = pose.linear().transpose() * world_cov * pose.linear();

        PointType pt;
        pt << p_sensor.x(), p_sensor.y(), p_sensor.z(), 1.0f;
        cloud.points->push_back(pt);
        cloud.timestamp_offsets->push_back(offset_ms);

        Normal normal;
        normal.head<3>() = n_sensor;
        normal.w() = 0.0f;
        cloud.normals->push_back(normal);

        Covariance cov = Covariance::Zero();
        cov.topLeftCorner<3, 3>() = cov_sensor;
        cloud.covs->push_back(cov);
    }

    const auto imu_buf = make_imu_buffer(scan_start_sec - 0.02, scan_duration_sec + 0.04, 24,
                                         Eigen::Vector3f(0.0f, 0.0f, omega_z), Eigen::Vector3f::Zero());

    imu::IMUPreintegrationParams preint_params;
    preint_params.gravity = Eigen::Vector3f::Zero();

    PointCloudShared deskewed(queue);
    ASSERT_TRUE(deskew_point_cloud_imu(cloud, deskewed, imu_buf, scan_start_sec, Eigen::Isometry3f::Identity(),
                                       imu::IMUBias(), preint_params, Eigen::Matrix3f::Identity()));

    for (size_t i = 0; i < offsets_ms.size(); ++i) {
        const Eigen::Vector3f p_corrected = (*deskewed.points)[i].head<3>();
        EXPECT_NEAR((p_corrected - world_point).norm(), 0.0f, kEps) << "point " << i;

        const Eigen::Vector3f n_corrected = (*deskewed.normals)[i].head<3>();
        EXPECT_NEAR((n_corrected - world_normal).norm(), 0.0f, kEps) << "normal " << i;

        const Eigen::Matrix3f cov_corrected = (*deskewed.covs)[i].topLeftCorner<3, 3>();
        EXPECT_NEAR((cov_corrected - world_cov).norm(), 0.0f, kEps) << "covariance " << i;
    }
}

}  // namespace
}  // namespace sycl_points::algorithms::deskew
