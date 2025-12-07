#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
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
/// @param input_cloud Point cloud with timestamps. The data in this cloud is modified during in-place operation (i.e.
/// `&input_cloud == &output_cloud`).
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
    std::array<float, 6> delta_twist_array{};
    std::copy(delta_twist.data(), delta_twist.data() + delta_twist.size(), delta_twist_array.begin());

    const auto work_group_size = input_cloud.queue.get_work_group_size();
    const auto global_size = input_cloud.queue.get_global_size(input_cloud.size());
    const size_t cloud_size = output_cloud.size();

    // Cache raw pointers for the device kernel.
    auto* points_in = input_cloud.points->data();
    auto* points_out = output_cloud.points->data();
    auto* normals_in = input_cloud.has_normal() ? input_cloud.normals->data() : nullptr;
    auto* normals_out = output_cloud.has_normal() ? output_cloud.normals->data() : nullptr;
    auto* covs_in = input_cloud.has_cov() ? input_cloud.covs->data() : nullptr;
    auto* covs_out = output_cloud.has_cov() ? output_cloud.covs->data() : nullptr;
    auto* timestamps = output_cloud.timestamp_offsets->data();

    const bool process_normals = normals_in != nullptr && normals_out != nullptr;
    const bool process_covs = covs_in != nullptr && covs_out != nullptr;

    // Launch device kernel to deskew each point independently.
    sycl::event deskew_event = input_cloud.queue.ptr->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t idx = item.get_global_linear_id();
            if (idx >= cloud_size) {
                return;
            }

            const float timestamp_seconds = timestamps[idx] * 1e-3f;
            if (!sycl::isfinite(timestamp_seconds)) {
                return;
            }

            // Normalize with respect to the scan duration using timestamp offsets directly.
            const float normalized_time = sycl::clamp(timestamp_seconds / delta_time_seconds, 0.0f, 1.0f);

            Eigen::Vector<float, 6> scaled_twist;
            for (size_t i = 0; i < delta_twist_array.size(); ++i) {
                scaled_twist[static_cast<Eigen::Index>(i)] = delta_twist_array[i] * normalized_time;
            }

            const Eigen::Matrix4f point_motion = eigen_utils::lie::se3_exp(scaled_twist);

            // Apply rotation and translation to deskew the point in the device kernel.
            const auto& [rotation, translation] = eigen_utils::geometry::matrix4_to_isometry3(point_motion);

            const Eigen::Vector3f point_in = points_in[idx].template head<3>();
            const Eigen::Vector3f rotated_point = eigen_utils::multiply<3, 3>(rotation, point_in);
            points_out[idx].template head<3>() = eigen_utils::add<3, 1>(rotated_point, translation);

            // Rotate normals and covariances using the integrated angular velocity.
            const Eigen::Vector3f integrated_omega = scaled_twist.template head<3>();
            const Eigen::Matrix3f rotation_omega =
                eigen_utils::geometry::quaternion_to_rotation_matrix(eigen_utils::lie::so3_exp(integrated_omega));
            if (process_normals) {
                const Eigen::Vector3f normal_in = normals_in[idx].template head<3>();
                normals_out[idx].template head<3>() = eigen_utils::multiply<3, 3>(rotation_omega, normal_in);
            }
            if (process_covs) {
                const Eigen::Matrix3f cov_in = covs_in[idx].topLeftCorner<3, 3>();
                const Eigen::Matrix3f rotation_omega_t = eigen_utils::transpose<3, 3>(rotation_omega);
                const Eigen::Matrix3f rotated_cov = eigen_utils::multiply<3, 3, 3>(
                    rotation_omega, eigen_utils::multiply<3, 3, 3>(cov_in, rotation_omega_t));
                covs_out[idx].topLeftCorner<3, 3>() = rotated_cov;
            }
        });
    });

    deskew_event.wait();

    return true;
}

}  // namespace sycl_points
