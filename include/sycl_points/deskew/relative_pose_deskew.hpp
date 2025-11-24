#pragma once

#include <Eigen/Geometry>
#include <algorithm>

#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

/// @brief Constant body velocity expressed as linear and angular components.
struct ConstantBodyVelocity {
    /// @brief Linear velocity in the body frame (m/s).
    Eigen::Vector3f linear_velocity{Eigen::Vector3f::Zero()};
    /// @brief Angular velocity in the body frame (rad/s).
    Eigen::Vector3f angular_velocity{Eigen::Vector3f::Zero()};
};

/// @brief Estimate constant body velocity from two consecutive relative poses.
/// @param previous_relative_pose Relative pose at the beginning of the interval.
/// @param current_relative_pose Relative pose at the end of the interval.
/// @param delta_time_seconds Time difference between the two poses in seconds (float precision).
/// @param velocity Output linear and angular velocity expressed in the body frame.
/// @return true if the estimation succeeds, false when @p delta_time_seconds is non-positive.
inline bool estimate_constant_body_velocity(const Eigen::Transform<float, 3, 1>& previous_relative_pose,
                                            const Eigen::Transform<float, 3, 1>& current_relative_pose,
                                            float delta_time_seconds, ConstantBodyVelocity& velocity) {
    if (delta_time_seconds <= 0.0f) {
        velocity = ConstantBodyVelocity{};
        return false;
    }

    // Compute motion between the two poses and map it to the twist vector.
    const Eigen::Transform<float, 3, 1> delta_pose = previous_relative_pose.inverse() * current_relative_pose;
    const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(delta_pose);

    const float inv_dt = 1.0f / delta_time_seconds;
    velocity.angular_velocity = delta_twist.head<3>() * inv_dt;
    velocity.linear_velocity = delta_twist.tail<3>() * inv_dt;
    return true;
}

/// @brief Deskew a point cloud assuming constant body velocity between two poses.
///
/// Points are transformed from the sensor frame at their sampling time into the
/// frame of @p previous_relative_pose, compensating both rotation and
/// translation. Normals and covariances are rotated using the angular velocity
/// model to keep them roughly aligned with the deskewed points, while color and
/// intensity gradients are cleared.
/// @param cloud Point cloud to be updated in-place. Timestamps must be present.
/// @param previous_relative_pose Relative pose at the start of the scan interval.
/// @param current_relative_pose Relative pose at the end of the scan interval.
/// @param delta_time_seconds Time difference between the two poses in seconds (float precision).
/// @return true when deskewing succeeds, false if prerequisites are not met.
inline bool deskew_point_cloud_constant_velocity(PointCloudShared& cloud,
                                                 const Eigen::Transform<float, 3, 1>& previous_relative_pose,
                                                 const Eigen::Transform<float, 3, 1>& current_relative_pose,
                                                 float delta_time_seconds) {
    if (cloud.size() == 0 || !cloud.has_timestamps()) {
        return false;
    }

    if (delta_time_seconds <= 0.0f) {
        return false;
    }

    // Compute motion between the two poses and map it to the twist vector.
    const Eigen::Transform<float, 3, 1> delta_pose = previous_relative_pose.inverse() * current_relative_pose;
    const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(delta_pose);

    ConstantBodyVelocity velocity;
    const float inv_dt = 1.0f / delta_time_seconds;
    velocity.angular_velocity = delta_twist.head<3>() * inv_dt;
    velocity.linear_velocity = delta_twist.tail<3>() * inv_dt;

    // Hint to the runtime that host-side access will follow for shared memory.
    cloud.queue.set_accessed_by_host(cloud.timestamp_offsets->data(), cloud.timestamp_offsets->size());
    cloud.queue.set_accessed_by_host(cloud.points->data(), cloud.points->size());
    if (cloud.has_normal()) {
        cloud.queue.set_accessed_by_host(cloud.normals->data(), cloud.normals->size());
    }
    if (cloud.has_cov()) {
        cloud.queue.set_accessed_by_host(cloud.covs->data(), cloud.covs->size());
    }

    for (size_t idx = 0; idx < cloud.size(); ++idx) {
        const float timestamp_seconds = static_cast<float>((*cloud.timestamp_offsets)[idx]) * 1e-3f;
        if (!std::isfinite(timestamp_seconds)) {
            continue;
        }

        // Normalize with respect to the scan duration using timestamp offsets directly.
        const float normalized_time =
            delta_time_seconds > 0.0f ? std::clamp(timestamp_seconds / delta_time_seconds, 0.0f, 1.0f) : 0.0f;

        const Eigen::Vector<float, 6> scaled_twist = (delta_twist * normalized_time).eval();
        const Eigen::Transform<float, 3, 1> point_motion = eigen_utils::lie::se3_exp(scaled_twist);
        const Eigen::Transform<float, 3, 1> point_pose = previous_relative_pose * point_motion;

        const Eigen::Vector3f corrected_point = point_pose * (*cloud.points)[idx].head<3>();
        (*cloud.points)[idx].head<3>() = corrected_point;

        // Rotate normals and covariances using the integrated angular velocity.
        const Eigen::Vector3f integrated_omega = scaled_twist.head<3>();
        const Eigen::Matrix3f rotation = eigen_utils::lie::so3_exp(integrated_omega).toRotationMatrix();
        if (cloud.has_normal()) {
            (*cloud.normals)[idx].head<3>() = rotation * (*cloud.normals)[idx].head<3>();
        }
        if (cloud.has_cov()) {
            Eigen::Matrix3f rotated_cov = rotation * (*cloud.covs)[idx].topLeftCorner<3, 3>() * rotation.transpose();
            (*cloud.covs)[idx].topLeftCorner<3, 3>() = rotated_cov;
        }
    }
    if (cloud.has_color_gradient()) {
        cloud.color_gradients->clear();
    }
    if (cloud.has_intensity_gradient()) {
        cloud.intensity_gradients->clear();
    }

    return true;
}

}  // namespace sycl_points

