#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <deque>
#include <vector>

#include "sycl_points/imu/imu_preintegration.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace deskew {

/// @brief A single entry in the IMU-derived pose trajectory used for deskewing.
///
/// Stores the relative pose of the LiDAR frame at a given time with respect to
/// the LiDAR frame at the scan start. The quaternion convention is (x, y, z, w).
/// Using a flat struct of primitives makes the data safe to pass as a raw pointer
/// into SYCL device kernels.
struct IMUTrajectoryPose {
    float q[4];       ///< Unit quaternion [x, y, z, w] representing relative rotation.
    float t[3];       ///< Relative translation in the scan-start LiDAR frame [m].
    float timestamp;  ///< Time from scan start [s].
};

static_assert(sizeof(IMUTrajectoryPose) == 32, "IMUTrajectoryPose size mismatch");

/// @brief Return code for deskew_point_cloud_imu().
enum class IMUDeskewStatus {
    success,
    insufficient_imu_coverage,  ///< IMU data does not bracket the full scan window.
    no_timestamps,              ///< Input cloud has no per-point timestamp offsets.
    invalid_scan_duration,      ///< Scan duration is zero or negative.
    empty_cloud,                ///< Input cloud is empty.
};

namespace detail {

/// @brief Quaternion Hamilton product (x, y, z, w convention).
///        Uses element-wise scalar access to avoid SSE intrinsics in device code.
SYCL_EXTERNAL inline Eigen::Vector4f quat_mult(const Eigen::Vector4f& a, const Eigen::Vector4f& b) {
    Eigen::Vector4f result;
    result(0) = a(3) * b(0) + a(0) * b(3) + a(1) * b(2) - a(2) * b(1);
    result(1) = a(3) * b(1) - a(0) * b(2) + a(1) * b(3) + a(2) * b(0);
    result(2) = a(3) * b(2) + a(0) * b(1) - a(1) * b(0) + a(2) * b(3);
    result(3) = a(3) * b(3) - a(0) * b(0) - a(1) * b(1) - a(2) * b(2);
    return result;
}

/// @brief Spherical linear interpolation between two unit quaternions.
///        Uses so3_log / so3_exp (SYCL_EXTERNAL) for stable interpolation.
///        Element-wise scalar operations are used throughout to avoid SSE intrinsics.
SYCL_EXTERNAL inline Eigen::Vector4f quat_slerp(const Eigen::Vector4f& q0, const Eigen::Vector4f& q1, float alpha) {
    // Ensure the shorter arc is taken: flip q1 if dot product is negative.
    Eigen::Vector4f q1_ = q1;
    if (eigen_utils::dot<4>(q0, q1_) < 0.0f) {
        // Element-wise negation — avoids SSE pnegate on Eigen::Vector4f.
        q1_(0) *= -1.0f;
        q1_(1) *= -1.0f;
        q1_(2) *= -1.0f;
        q1_(3) *= -1.0f;
    }
    // Conjugate of q0: (-x, -y, -z, w)
    Eigen::Vector4f q0_conj;
    q0_conj(0) = -q0(0);
    q0_conj(1) = -q0(1);
    q0_conj(2) = -q0(2);
    q0_conj(3) = q0(3);
    // Relative rotation: q0^{-1} * q1
    const Eigen::Vector4f delta_q = quat_mult(q0_conj, q1_);
    // SO(3) log → scale → exp
    const Eigen::Vector3f omega = eigen_utils::lie::so3_log(delta_q);
    const Eigen::Vector4f delta_q_scaled = eigen_utils::lie::so3_exp(omega * alpha);
    return quat_mult(q0, delta_q_scaled);
}

/// @brief Interpolate a pose trajectory entry at the given normalized parameter alpha.
///        Returns the interpolated rotation matrix and translation vector.
///        All Eigen::Vector4f operations are contained within this SYCL_EXTERNAL function
///        to prevent SSE intrinsics from being inlined into device kernel code.
SYCL_EXTERNAL inline void interpolate_trajectory_pose(const IMUTrajectoryPose* traj, size_t lo, size_t hi, float alpha,
                                                      Eigen::Matrix3f& R_out, Eigen::Vector3f& t_out) {
    // Load quaternions element-by-element to avoid Vector4f 4-arg constructor SSE.
    Eigen::Vector4f q0, q1;
    q0(0) = traj[lo].q[0];
    q0(1) = traj[lo].q[1];
    q0(2) = traj[lo].q[2];
    q0(3) = traj[lo].q[3];
    q1(0) = traj[hi].q[0];
    q1(1) = traj[hi].q[1];
    q1(2) = traj[hi].q[2];
    q1(3) = traj[hi].q[3];

    const Eigen::Vector4f q_interp = quat_slerp(q0, q1, alpha);
    R_out = eigen_utils::geometry::quaternion_to_rotation_matrix(q_interp);

    // Element-wise LERP for translation.
    const float inv_alpha = 1.0f - alpha;
    t_out(0) = traj[lo].t[0] * inv_alpha + traj[hi].t[0] * alpha;
    t_out(1) = traj[lo].t[1] * inv_alpha + traj[hi].t[1] * alpha;
    t_out(2) = traj[lo].t[2] * inv_alpha + traj[hi].t[2] * alpha;
}

}  // namespace detail

/// @brief Deskew a point cloud using buffered IMU measurements (SE3, pre-processing step).
///
/// All points are brought into the sensor frame at scan-start time by integrating
/// the buffered IMU measurements over the scan window and applying per-point SE3
/// corrections via a SYCL parallel kernel. Gravity is compensated using
/// @p R_world_body_i consistent with IMUPreintegration::predict_relative_transform().
///
/// This function should be called *before* downsampling and ICP so that every
/// raw point is corrected. If the IMU buffer does not bracket the full scan window
/// the function returns false and leaves the output cloud unmodified.
///
/// @param input_cloud           Raw scan with per-point timestamp offsets [ms].
/// @param output_cloud          Deskewed scan. May alias @p input_cloud for in-place use.
/// @param imu_buffer            Time-ordered buffer of IMU measurements [absolute s].
/// @param scan_start_time_sec   Absolute time of scan start matching
///                              @c input_cloud.start_time_ms * 1e-3 [s].
/// @param T_imu_to_lidar        Extrinsic: pose of IMU body frame expressed in LiDAR frame.
/// @param bias                  Current gyroscope and accelerometer bias estimate.
/// @param preintegration_params Gravity vector and other integration settings.
/// @param R_world_body_i        Rotation from IMU body to world frame at scan start.
///                              Used for gravity compensation in Delta_p.
///                              Typically: odom.rotation() * T_imu_to_lidar.rotation().
/// @param[out] status           Optional detailed result code.
/// @return true on success, false if prerequisites are not met.
inline bool deskew_point_cloud_imu(
    const PointCloudShared& input_cloud, PointCloudShared& output_cloud,
    const std::deque<imu::IMUMeasurement, Eigen::aligned_allocator<imu::IMUMeasurement>>& imu_buffer,
    double scan_start_time_sec, const Eigen::Isometry3f& T_imu_to_lidar, const imu::IMUBias& bias,
    const imu::IMUPreintegrationParams& preintegration_params, const Eigen::Matrix3f& R_world_body_i,
    IMUDeskewStatus* status = nullptr) {
    auto set_status = [&](IMUDeskewStatus s) {
        if (status) *status = s;
    };

    if (!input_cloud.queue.ptr || !output_cloud.queue.ptr) {
        throw std::runtime_error("[deskew_point_cloud_imu] SYCL queue is not initialized");
    }
    if (input_cloud.queue.ptr->get_context() != output_cloud.queue.ptr->get_context()) {
        throw std::runtime_error(
            "[deskew_point_cloud_imu] input_cloud and output_cloud must share the same SYCL context");
    }

    const size_t cloud_size = input_cloud.size();
    if (cloud_size == 0) {
        set_status(IMUDeskewStatus::empty_cloud);
        return false;
    }
    if (!input_cloud.has_timestamps()) {
        set_status(IMUDeskewStatus::no_timestamps);
        return false;
    }

    const double scan_duration_sec = (input_cloud.end_time_ms - input_cloud.start_time_ms) * 1e-3;
    if (scan_duration_sec <= 0.0) {
        set_status(IMUDeskewStatus::invalid_scan_duration);
        return false;
    }
    const double scan_end_sec = scan_start_time_sec + scan_duration_sec;

    // -----------------------------------------------------------------------
    // Step 1: Filter IMU buffer to the scan window with a generous margin.
    // -----------------------------------------------------------------------
    constexpr double kMarginSec = 0.05;  // 50 ms — covers up to 50 Hz IMU
    std::vector<imu::IMUMeasurement, Eigen::aligned_allocator<imu::IMUMeasurement>> filtered;
    filtered.reserve(256);
    for (const auto& m : imu_buffer) {
        if (m.timestamp >= scan_start_time_sec - kMarginSec && m.timestamp <= scan_end_sec + kMarginSec) {
            filtered.push_back(m);
        }
    }

    // We need at least 2 measurements and coverage of the full scan window.
    if (filtered.size() < 2) {
        set_status(IMUDeskewStatus::insufficient_imu_coverage);
        return false;
    }
    if (filtered.front().timestamp > scan_start_time_sec + kMarginSec ||
        filtered.back().timestamp < scan_end_sec - kMarginSec) {
        set_status(IMUDeskewStatus::insufficient_imu_coverage);
        return false;
    }

    // -----------------------------------------------------------------------
    // Step 2: Build a virtual IMU measurement at exactly scan_start_time_sec
    //         by linearly interpolating adjacent measurements.
    // -----------------------------------------------------------------------
    imu::IMUMeasurement m_start;
    m_start.timestamp = scan_start_time_sec;

    // Find the first measurement at or after scan_start.
    auto it_next = std::lower_bound(filtered.begin(), filtered.end(), scan_start_time_sec,
                                    [](const imu::IMUMeasurement& m, double t) { return m.timestamp < t; });

    if (it_next == filtered.begin()) {
        // All measurements are at or after scan_start — use the first one as-is.
        m_start.gyro = it_next->gyro;
        m_start.accel = it_next->accel;
    } else if (it_next == filtered.end()) {
        // All measurements are before scan_start — use the last one as-is.
        m_start.gyro = filtered.back().gyro;
        m_start.accel = filtered.back().accel;
        it_next = filtered.end();
    } else {
        // Interpolate between the bracketing measurements.
        const auto& prev_m = *std::prev(it_next);
        const float alpha =
            static_cast<float>((scan_start_time_sec - prev_m.timestamp) / (it_next->timestamp - prev_m.timestamp));
        m_start.gyro = prev_m.gyro * (1.0f - alpha) + it_next->gyro * alpha;
        m_start.accel = prev_m.accel * (1.0f - alpha) + it_next->accel * alpha;
    }

    // -----------------------------------------------------------------------
    // Step 3: Integrate IMU from scan_start to build a relative-pose trajectory.
    //         Each entry is the LiDAR-frame relative pose at that IMU timestamp,
    //         with gravity compensated as in IMUPreintegration::predict_relative_transform().
    // -----------------------------------------------------------------------
    std::vector<IMUTrajectoryPose> traj_cpu;
    traj_cpu.reserve(filtered.size() + 1);

    // Identity pose at scan start (t = 0).
    {
        IMUTrajectoryPose identity{};
        identity.q[0] = 0.0f;
        identity.q[1] = 0.0f;
        identity.q[2] = 0.0f;
        identity.q[3] = 1.0f;
        identity.t[0] = identity.t[1] = identity.t[2] = 0.0f;
        identity.timestamp = 0.0f;
        traj_cpu.push_back(identity);
    }

    imu::IMUPreintegration local_integrator(preintegration_params);
    local_integrator.reset(bias, R_world_body_i);
    local_integrator.integrate(m_start);  // stores as prev; no integration step yet

    for (auto it = it_next; it != filtered.end(); ++it) {
        if (it->timestamp > scan_end_sec + kMarginSec) break;

        local_integrator.integrate(*it);

        const float t_rel_sec = static_cast<float>(it->timestamp - scan_start_time_sec);
        if (t_rel_sec < 0.0f) continue;

        // predict_relative_transform() applies gravity compensation using dt_total,
        // which equals t_rel_sec because the integrator was reset at scan_start.
        const sycl_points::TransformMatrix T_imu_rel = local_integrator.predict_relative_transform(bias);

        // Convert to LiDAR-frame relative transform:
        //   T_lidar_rel = T_imu_to_lidar * T_imu_rel * T_imu_to_lidar^{-1}
        Eigen::Isometry3f T_imu_rel_iso = Eigen::Isometry3f::Identity();
        T_imu_rel_iso.linear() = T_imu_rel.block<3, 3>(0, 0);
        T_imu_rel_iso.translation() = T_imu_rel.block<3, 1>(0, 3);
        const Eigen::Isometry3f T_lidar_rel = T_imu_to_lidar * T_imu_rel_iso * T_imu_to_lidar.inverse();

        // Store as flat quaternion + translation for SYCL kernel access.
        const Eigen::Quaternionf q_lidar(T_lidar_rel.rotation());
        IMUTrajectoryPose entry{};
        entry.q[0] = q_lidar.x();
        entry.q[1] = q_lidar.y();
        entry.q[2] = q_lidar.z();
        entry.q[3] = q_lidar.w();
        entry.t[0] = T_lidar_rel.translation().x();
        entry.t[1] = T_lidar_rel.translation().y();
        entry.t[2] = T_lidar_rel.translation().z();
        entry.timestamp = t_rel_sec;
        traj_cpu.push_back(entry);
    }

    // Need at least 2 entries to interpolate (identity + at least one more).
    if (traj_cpu.size() < 2) {
        set_status(IMUDeskewStatus::insufficient_imu_coverage);
        return false;
    }

    // Verify the trajectory covers the full scan duration.
    const float scan_duration_f = static_cast<float>(scan_duration_sec);
    if (traj_cpu.back().timestamp < scan_duration_f - static_cast<float>(kMarginSec)) {
        set_status(IMUDeskewStatus::insufficient_imu_coverage);
        return false;
    }

    // -----------------------------------------------------------------------
    // Step 4: Copy trajectory into SYCL shared memory.
    // -----------------------------------------------------------------------
    const size_t traj_size = traj_cpu.size();
    shared_vector<IMUTrajectoryPose> traj_sycl(*input_cloud.queue.ptr);
    traj_sycl.assign(traj_cpu.begin(), traj_cpu.end());

    // -----------------------------------------------------------------------
    // Step 5: Mirror input metadata into output (if not in-place).
    // -----------------------------------------------------------------------
    if (&input_cloud != &output_cloud) {
        output_cloud.start_time_ms = input_cloud.start_time_ms;
        output_cloud.end_time_ms = input_cloud.end_time_ms;

        output_cloud.timestamp_offsets->assign(input_cloud.timestamp_offsets->begin(),
                                               input_cloud.timestamp_offsets->end());
        output_cloud.points->resize(cloud_size);

        if (input_cloud.has_normal()) {
            output_cloud.normals->resize(cloud_size);
        } else {
            output_cloud.normals->clear();
        }
        if (input_cloud.has_cov()) {
            output_cloud.covs->resize(cloud_size);
        } else {
            output_cloud.covs->clear();
        }
        if (input_cloud.has_rgb()) {
            output_cloud.rgb->assign(input_cloud.rgb->begin(), input_cloud.rgb->end());
        } else {
            output_cloud.rgb->clear();
        }
        if (input_cloud.has_intensity()) {
            output_cloud.intensities->assign(input_cloud.intensities->begin(), input_cloud.intensities->end());
        } else {
            output_cloud.intensities->clear();
        }
        if (input_cloud.has_color_gradient()) {
            output_cloud.color_gradients->resize(cloud_size);
        } else {
            output_cloud.color_gradients->clear();
        }
        if (input_cloud.has_intensity_gradient()) {
            output_cloud.intensity_gradients->resize(cloud_size);
        } else {
            output_cloud.intensity_gradients->clear();
        }
    }

    // -----------------------------------------------------------------------
    // Step 6: SYCL kernel — per-point SE3 correction via trajectory lookup.
    // -----------------------------------------------------------------------
    auto deskew_event = input_cloud.queue.ptr->submit([&](sycl::handler& h) {
        const auto work_group_size = input_cloud.queue.get_work_group_size();
        const auto global_size = input_cloud.queue.get_global_size(input_cloud.size());

        const auto* points_in = input_cloud.points->data();
        const auto* normals_in = input_cloud.has_normal() ? input_cloud.normals->data() : nullptr;
        const auto* covs_in = input_cloud.has_cov() ? input_cloud.covs->data() : nullptr;
        const auto* color_gradients_in =
            input_cloud.has_color_gradient() ? input_cloud.color_gradients->data() : nullptr;
        const auto* intensity_gradients_in =
            input_cloud.has_intensity_gradient() ? input_cloud.intensity_gradients->data() : nullptr;

        auto* points_out = output_cloud.points->data();
        auto* normals_out = output_cloud.has_normal() ? output_cloud.normals->data() : nullptr;
        auto* covs_out = output_cloud.has_cov() ? output_cloud.covs->data() : nullptr;
        auto* color_gradients_out = output_cloud.has_color_gradient() ? output_cloud.color_gradients->data() : nullptr;
        auto* intensity_gradients_out =
            output_cloud.has_intensity_gradient() ? output_cloud.intensity_gradients->data() : nullptr;
        const auto* timestamps = input_cloud.timestamp_offsets->data();
        const auto* traj_ptr = traj_sycl.data();

        const bool process_normals = normals_in != nullptr && normals_out != nullptr;
        const bool process_covs = covs_in != nullptr && covs_out != nullptr;
        const bool process_color_gradients = color_gradients_in != nullptr && color_gradients_out != nullptr;
        const bool process_intensity_gradients =
            intensity_gradients_in != nullptr && intensity_gradients_out != nullptr;

        h.parallel_for(                                       //
            sycl::nd_range<1>(global_size, work_group_size),  //
            [=](sycl::nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= cloud_size) return;

                // Convert per-point offset [ms] to time from scan start [s].
                const float t_sec = timestamps[idx] * 1e-3f;

                if (!sycl::isfinite(t_sec)) {
                    // Copy the point unchanged if its timestamp is invalid.
                    eigen_utils::copy<4, 1>(points_in[idx], points_out[idx]);
                    if (process_normals) eigen_utils::copy<4, 1>(normals_in[idx], normals_out[idx]);
                    if (process_covs) eigen_utils::copy<4, 4>(covs_in[idx], covs_out[idx]);
                    if (process_color_gradients)
                        eigen_utils::copy<3, 3>(color_gradients_in[idx], color_gradients_out[idx]);
                    if (process_intensity_gradients)
                        eigen_utils::copy<3, 1>(intensity_gradients_in[idx], intensity_gradients_out[idx]);
                    return;
                }

                // ---- Binary search for the enclosing trajectory interval ----
                size_t lo = 0;
                size_t hi = traj_size - 1;
                while (hi - lo > 1) {
                    const size_t mid = (lo + hi) / 2;
                    if (traj_ptr[mid].timestamp <= t_sec)
                        lo = mid;
                    else
                        hi = mid;
                }

                // ---- Interpolation factor ----
                const float t_lo = traj_ptr[lo].timestamp;
                const float t_hi = traj_ptr[hi].timestamp;
                float alpha = 0.0f;
                if (t_hi > t_lo) {
                    alpha = sycl::clamp((t_sec - t_lo) / (t_hi - t_lo), 0.0f, 1.0f);
                }

                // ---- SLERP + LERP inside SYCL_EXTERNAL to avoid SSE intrinsics ----
                // All Eigen::Vector4f operations (load, copy, slerp) are encapsulated
                // inside interpolate_trajectory_pose so the kernel lambda itself never
                // touches a Vector4f (which triggers pload/pstore on Packet4f).
                Eigen::Matrix3f R_interp;
                Eigen::Vector3f t_interp;
                detail::interpolate_trajectory_pose(traj_ptr, lo, hi, alpha, R_interp, t_interp);

                // ---- Apply SE3: p_out = R * p_in + t ----
                const Eigen::Vector3f p3 = points_in[idx].template head<3>();
                const Eigen::Vector3f Rp = eigen_utils::multiply<3, 3>(R_interp, p3);
                // Element-wise scalar writes — avoids Eigen::Vector4f copy assignment (SSE pstore).
                points_out[idx](0) = Rp(0) + t_interp(0);
                points_out[idx](1) = Rp(1) + t_interp(1);
                points_out[idx](2) = Rp(2) + t_interp(2);
                points_out[idx](3) = points_in[idx](3);

                // ---- Rotate frame-dependent attributes ----
                if (process_normals) {
                    normals_out[idx].setZero();
                    const Eigen::Vector3f normal_in = normals_in[idx].template head<3>();
                    normals_out[idx].template head<3>() = eigen_utils::multiply<3, 3>(R_interp, normal_in);
                }
                if (process_covs) {
                    covs_out[idx].setZero();
                    const Eigen::Matrix3f cov_in = covs_in[idx].topLeftCorner<3, 3>();
                    const Eigen::Matrix3f R_interp_t = eigen_utils::transpose<3, 3>(R_interp);
                    const Eigen::Matrix3f rotated_cov =
                        eigen_utils::multiply<3, 3, 3>(R_interp, eigen_utils::multiply<3, 3, 3>(cov_in, R_interp_t));
                    covs_out[idx].topLeftCorner<3, 3>() = rotated_cov;
                }
                if (process_color_gradients) {
                    color_gradients_out[idx].setZero();
                    color_gradients_out[idx] =
                        eigen_utils::multiply<3, 3, 3>(color_gradients_in[idx], eigen_utils::transpose<3, 3>(R_interp));
                }
                if (process_intensity_gradients) {
                    intensity_gradients_out[idx].setZero();
                    intensity_gradients_out[idx] = eigen_utils::multiply<3, 3>(R_interp, intensity_gradients_in[idx]);
                }
            });
    });

    deskew_event.wait_and_throw();

    set_status(IMUDeskewStatus::success);
    return true;
}

}  // namespace deskew
}  // namespace algorithms
}  // namespace sycl_points
