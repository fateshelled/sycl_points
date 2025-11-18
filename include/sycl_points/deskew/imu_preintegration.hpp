#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/points/types.hpp>
#include <vector>

namespace sycl_points {

/// @brief Preintegrate IMU rotations and output cumulative orientations.
/// @param imu_samples Sequence of IMU measurements ordered by timestamp.
/// @return Quaternion sequence aligned with @p imu_samples.
struct IMUMotionState {
    /// @brief Integrated orientation up to the sample timestamp.
    Eigen::Quaternionf orientation{Eigen::Quaternionf::Identity()};
    /// @brief Integrated position in the world frame.
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    /// @brief Integrated velocity in the world frame.
    Eigen::Vector3f velocity{Eigen::Vector3f::Zero()};
};

/// @brief Preintegrate IMU motion and output cumulative states.
/// @param imu_samples Sequence of IMU measurements ordered by timestamp.
/// @param gyro_bias Constant gyroscope bias to subtract from measurements.
/// @param accel_bias Constant accelerometer bias to subtract from measurements.
/// @param gravity Gravity acceleration expressed in the world frame (subtracted after
/// rotation).
/// @return Motion state sequence aligned with @p imu_samples.
inline std::vector<IMUMotionState> preintegrate_imu_motion(const IMUDataContainerCPU& imu_samples,
                                                          const Eigen::Vector3f& gyro_bias =
                                                              Eigen::Vector3f::Zero(),
                                                          const Eigen::Vector3f& accel_bias =
                                                              Eigen::Vector3f::Zero(),
                                                          const Eigen::Vector3f& gravity =
                                                              Eigen::Vector3f::Zero()) {
    std::vector<IMUMotionState> motion_states;
    motion_states.reserve(imu_samples.size());
    if (imu_samples.empty()) {
        return motion_states;
    }

    IMUMotionState accumulated_state;
    motion_states.push_back(accumulated_state);

    for (size_t idx = 1; idx < imu_samples.size(); ++idx) {
        const IMUData& prev_sample = imu_samples[idx - 1];
        const IMUData& curr_sample = imu_samples[idx];
        const double delta_time = curr_sample.timestamp_seconds() - prev_sample.timestamp_seconds();
        if (delta_time <= 0.0) {
            // Non-positive dt: propagate previous state.
            motion_states.push_back(accumulated_state);
            continue;
        }

        const Eigen::Vector3f average_raw_gyro = 0.5f * (prev_sample.angular_velocity + curr_sample.angular_velocity);
        const Eigen::Vector3f average_gyro = average_raw_gyro - gyro_bias;
        const Eigen::Vector3f delta_theta = average_gyro * static_cast<float>(delta_time);
        const float angle = delta_theta.norm();

        Eigen::Quaternionf delta_orientation;
        if (angle < 1e-6f) {
            // Use first-order quaternion update to avoid precision loss.
            delta_orientation =
                Eigen::Quaternionf(1.0f, 0.5f * delta_theta.x(), 0.5f * delta_theta.y(), 0.5f * delta_theta.z());
            delta_orientation.normalize();
        } else {
            const Eigen::Vector3f axis = delta_theta / angle;
            delta_orientation = Eigen::AngleAxisf(angle, axis);
        }

        const auto prev_orientation = accumulated_state.orientation;
        accumulated_state.orientation = (prev_orientation * delta_orientation).normalized();

        // Use the average acceleration and beginning-of-interval orientation to
        // integrate translational motion in the world frame. Biases are removed
        // before rotation and gravity is subtracted in the world frame so that the
        // integrator can model free-fall or stationary scenarios accurately.
        const Eigen::Vector3f average_raw_acceleration =
            0.5f * (prev_sample.linear_acceleration + curr_sample.linear_acceleration);
        const Eigen::Vector3f average_acceleration = average_raw_acceleration - accel_bias;
        const Eigen::Vector3f acceleration_world = prev_orientation * average_acceleration - gravity;
        accumulated_state.position += accumulated_state.velocity * static_cast<float>(delta_time) +
                                      0.5f * acceleration_world * static_cast<float>(delta_time * delta_time);
        accumulated_state.velocity += acceleration_world * static_cast<float>(delta_time);

        motion_states.push_back(accumulated_state);
    }

    return motion_states;
}

inline IMUMotionState interpolate_motion_state(const IMUDataContainerCPU& imu_samples,
                                               const std::vector<IMUMotionState>& motion_states, double timestamp) {
    const auto comparator = [](const IMUData& sample, double t) { return sample.timestamp_seconds() < t; };

    auto upper = std::lower_bound(imu_samples.begin(), imu_samples.end(), timestamp, comparator);
    if (upper == imu_samples.begin()) {
        return motion_states.front();
    }
    if (upper == imu_samples.end()) {
        return motion_states.back();
    }

    const size_t next_idx = static_cast<size_t>(std::distance(imu_samples.begin(), upper));
    const size_t prev_idx = next_idx - 1;

    const double t0 = imu_samples[prev_idx].timestamp_seconds();
    const double t1 = imu_samples[next_idx].timestamp_seconds();
    const double ratio = (t1 - t0) <= 0.0 ? 0.0 : std::clamp((timestamp - t0) / (t1 - t0), 0.0, 1.0);

    IMUMotionState interpolated_state;
    interpolated_state.orientation =
        motion_states[prev_idx].orientation.slerp(static_cast<float>(ratio), motion_states[next_idx].orientation);
    interpolated_state.position =
        motion_states[prev_idx].position +
        static_cast<float>(ratio) * (motion_states[next_idx].position - motion_states[prev_idx].position);
    interpolated_state.velocity =
        motion_states[prev_idx].velocity +
        static_cast<float>(ratio) * (motion_states[next_idx].velocity - motion_states[prev_idx].velocity);
    return interpolated_state;
}

/// @brief Deskew a point cloud by removing rotational and translational motion
/// derived from IMU data.
///
/// The corrected points are expressed in the world frame (base frame) rather
/// than the sensor frame at the start timestamp.
/// @param cloud Point cloud that will be updated in-place.
/// @param imu_samples Time ordered IMU samples used for interpolation.
/// @param gyro_bias Constant gyroscope bias to subtract from measurements.
/// @param accel_bias Constant accelerometer bias to subtract from measurements.
/// @param gravity Gravity acceleration expressed in the world frame (subtracted after
/// rotation).
/// @return true when deskewing is applied, false if prerequisites are not met.
inline bool deskew_point_cloud(PointCloudCPU& cloud, const IMUDataContainerCPU& imu_samples,
                               const Eigen::Vector3f& gyro_bias = Eigen::Vector3f::Zero(),
                               const Eigen::Vector3f& accel_bias = Eigen::Vector3f::Zero(),
                               const Eigen::Vector3f& gravity = Eigen::Vector3f::Zero()) {
    if (cloud.size() == 0 || !cloud.has_timestamps() || imu_samples.size() < 2) {
        return false;
    }

    const std::vector<IMUMotionState> motion_states = preintegrate_imu_motion(imu_samples, gyro_bias,
                                                                             accel_bias, gravity);
    if (motion_states.size() != imu_samples.size()) {
        return false;
    }

    const double timestamp_base = static_cast<double>(cloud.timestamp_base_ns) * 1e-9;

    for (size_t idx = 0; idx < cloud.size(); ++idx) {
        const double timestamp_seconds = timestamp_base + static_cast<double>((*cloud.timestamp_offsets)[idx]) * 1e-9;
        if (!std::isfinite(timestamp_seconds)) {
            continue;
        }

        const double clamped_time = std::clamp(timestamp_seconds, imu_samples.front().timestamp_seconds(),
                                               imu_samples.back().timestamp_seconds());
        const IMUMotionState motion_state = interpolate_motion_state(imu_samples, motion_states, clamped_time);
        const Eigen::Vector3f original_point = (*cloud.points)[idx].head<3>();
        // Transform point from sensor frame to world frame to compensate for motion.
        const Eigen::Vector3f corrected_point = motion_state.orientation * original_point + motion_state.position;
        (*cloud.points)[idx].head<3>() = corrected_point;
    }

    return true;
}

}  // namespace sycl_points
