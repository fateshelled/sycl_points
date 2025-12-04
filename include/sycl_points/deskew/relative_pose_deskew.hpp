#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

/// @brief Deskew a point cloud assuming constant body velocity between two poses.
///
/// Points are transformed from the sensor frame at their sampling time into the
/// frame of @p previous_relative_pose, compensating both rotation and
/// translation. Normals and covariances are rotated using the angular velocity
/// model to keep them roughly aligned with the deskewed points, while color and
/// intensity gradients are cleared.
/// @param input_cloud Point cloud with timestamps. The data in this cloud is never modified.
/// @param output_cloud Point cloud receiving the deskewed data. Containers will be resized as needed.
/// @param previous_relative_pose Relative pose at the start of the scan interval.
/// @param current_relative_pose Relative pose at the end of the scan interval.
/// @param delta_time_seconds Time difference between the two poses in seconds (float precision).
/// @return true when deskewing succeeds, false if prerequisites are not met.
inline bool deskew_point_cloud_constant_velocity(const PointCloudShared& input_cloud, PointCloudShared& output_cloud,
                                                 const Eigen::Transform<float, 3, 1>& previous_relative_pose,
                                                 const Eigen::Transform<float, 3, 1>& current_relative_pose,
                                                 float delta_time_seconds) {
    if (input_cloud.size() == 0 || !input_cloud.has_timestamps()) {
        return false;
    }

    if (delta_time_seconds <= 0.0f) {
        return false;
    }

    if (&input_cloud != &output_cloud) {
        // Copy metadata so the output stays synchronized with the input timing information.
        output_cloud.start_time_ms = input_cloud.start_time_ms;
        output_cloud.end_time_ms = input_cloud.end_time_ms;

        // Mirror input fields into the output so non-deskewed attributes remain intact.
        output_cloud.timestamp_offsets->assign(input_cloud.timestamp_offsets->begin(),
                                               input_cloud.timestamp_offsets->end());
        output_cloud.points->assign(input_cloud.points->begin(), input_cloud.points->end());

        if (input_cloud.has_normal()) {
            output_cloud.normals->assign(input_cloud.normals->begin(), input_cloud.normals->end());
        } else {
            output_cloud.normals->clear();
        }
        if (input_cloud.has_cov()) {
            output_cloud.covs->assign(input_cloud.covs->begin(), input_cloud.covs->end());
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
    }
    if (output_cloud.has_color_gradient()) {
        output_cloud.color_gradients->clear();
    }
    if (output_cloud.has_intensity_gradient()) {
        output_cloud.intensity_gradients->clear();
    }

    // Compute motion between the two poses and map it to the twist vector.
    const Eigen::Transform<float, 3, 1> delta_pose = previous_relative_pose.inverse() * current_relative_pose;
    const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(delta_pose);
    for (size_t idx = 0; idx < input_cloud.size(); ++idx) {
        const float timestamp_seconds = (*input_cloud.timestamp_offsets)[idx] * 1e-3f;
        if (!std::isfinite(timestamp_seconds)) {
            continue;
        }

        // Normalize with respect to the scan duration using timestamp offsets directly.
        const float normalized_time =
            delta_time_seconds > 0.0f ? std::clamp(timestamp_seconds / delta_time_seconds, 0.0f, 1.0f) : 0.0f;

        const Eigen::Vector<float, 6> scaled_twist = (delta_twist * normalized_time).eval();

        const Eigen::Transform<float, 3, 1> point_motion = eigen_utils::lie::se3_exp(scaled_twist);
        const Eigen::Vector3f corrected_point = point_motion * (*input_cloud.points)[idx].head<3>();
        (*output_cloud.points)[idx].head<3>() = corrected_point;

        // Rotate normals and covariances using the integrated angular velocity.
        const Eigen::Vector3f integrated_omega = scaled_twist.head<3>();
        const Eigen::Matrix3f rotation = eigen_utils::lie::so3_exp(integrated_omega).toRotationMatrix();
        if (input_cloud.has_normal() && output_cloud.has_normal()) {
            (*output_cloud.normals)[idx].head<3>() = rotation * (*input_cloud.normals)[idx].head<3>();
        }
        if (input_cloud.has_cov() && output_cloud.has_cov()) {
            Eigen::Matrix3f rotated_cov =
                rotation * (*input_cloud.covs)[idx].topLeftCorner<3, 3>() * rotation.transpose();
            (*output_cloud.covs)[idx].topLeftCorner<3, 3>() = rotated_cov;
        }
    }

    return true;
}

}  // namespace sycl_points
