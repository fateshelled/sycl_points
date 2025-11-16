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
inline std::vector<Eigen::Quaternionf> PreintegrateIMURotations(
    const IMUDataContainerCPU &imu_samples) {
  std::vector<Eigen::Quaternionf> orientations;
  orientations.reserve(imu_samples.size());
  if (imu_samples.empty()) {
    return orientations;
  }

  Eigen::Quaternionf accumulated_orientation = Eigen::Quaternionf::Identity();
  orientations.push_back(accumulated_orientation);

  for (size_t idx = 1; idx < imu_samples.size(); ++idx) {
    const IMUData &prev_sample = imu_samples[idx - 1];
    const IMUData &curr_sample = imu_samples[idx];
    const double delta_time = curr_sample.timestamp - prev_sample.timestamp;
    if (delta_time <= 0.0) {
      // Non-positive dt: propagate previous orientation.
      orientations.push_back(accumulated_orientation);
      continue;
    }

    const Eigen::Vector3f average_gyro =
        0.5f * (prev_sample.angular_velocity + curr_sample.angular_velocity);
    const Eigen::Vector3f delta_theta =
        average_gyro * static_cast<float>(delta_time);
    const float angle = delta_theta.norm();

    Eigen::Quaternionf delta_orientation;
    if (angle < 1e-6f) {
      // Use first-order quaternion update to avoid precision loss.
      delta_orientation = Eigen::Quaternionf(1.0f, 0.5f * delta_theta.x(),
                                             0.5f * delta_theta.y(),
                                             0.5f * delta_theta.z());
      delta_orientation.normalize();
    } else {
      const Eigen::Vector3f axis = delta_theta / angle;
      delta_orientation = Eigen::AngleAxisf(angle, axis);
    }

    accumulated_orientation =
        (accumulated_orientation * delta_orientation).normalized();
    orientations.push_back(accumulated_orientation);
  }

  return orientations;
}

/// @brief Deskew a point cloud by removing rotational motion derived from IMU data.
/// @param cloud Point cloud that will be updated in-place.
/// @param imu_samples Time ordered IMU samples used for interpolation.
/// @return true when deskewing is applied, false if prerequisites are not met.
inline bool DeskewPointCloudRotations(PointCloudCPU &cloud,
                                      const IMUDataContainerCPU &imu_samples) {
  if (cloud.size() == 0 || !cloud.has_timestamps() || imu_samples.size() < 2) {
    return false;
  }

  const std::vector<Eigen::Quaternionf> orientations =
      PreintegrateIMURotations(imu_samples);
  if (orientations.size() != imu_samples.size()) {
    return false;
  }

  const auto clamp_ratio = [](double value, double min_value,
                              double max_value) -> double {
    return std::max(min_value, std::min(value, max_value));
  };

  const auto interpolate_orientation = [&](double timestamp) {
    const auto comparator = [](const IMUData &sample, double t) {
      return sample.timestamp < t;
    };

    auto upper = std::lower_bound(imu_samples.begin(), imu_samples.end(),
                                  timestamp, comparator);
    if (upper == imu_samples.begin()) {
      return orientations.front();
    }
    if (upper == imu_samples.end()) {
      return orientations.back();
    }

    const size_t next_idx =
        static_cast<size_t>(std::distance(imu_samples.begin(), upper));
    const size_t prev_idx = next_idx - 1;

    const double t0 = imu_samples[prev_idx].timestamp;
    const double t1 = imu_samples[next_idx].timestamp;
    const double ratio = (t1 - t0) <= 0.0
                             ? 0.0
                             : clamp_ratio((timestamp - t0) / (t1 - t0), 0.0,
                                           1.0);
    return orientations[prev_idx].slerp(static_cast<float>(ratio),
                                        orientations[next_idx]);
  };

  const double timestamp_base =
      static_cast<double>(cloud.timestamp_base_ns) * 1e-9;

  for (size_t idx = 0; idx < cloud.size(); ++idx) {
    const double timestamp_seconds =
        timestamp_base + static_cast<double>((*cloud.timestamp_offsets)[idx]) *
                               1e-9;
    if (!std::isfinite(timestamp_seconds)) {
      continue;
    }

    const double clamped_time =
        std::max(imu_samples.front().timestamp,
                 std::min(timestamp_seconds, imu_samples.back().timestamp));
    const Eigen::Quaternionf orientation = interpolate_orientation(clamped_time);
    const Eigen::Vector3f original_point = (*cloud.points)[idx].head<3>();
    const Eigen::Vector3f corrected_point =
        orientation.conjugate() * original_point;
    (*cloud.points)[idx].head<3>() = corrected_point;
  }

  return true;
}

}  // namespace sycl_points

