#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <stdexcept>

#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace deskew {

/// @brief Deskew a point cloud assuming constant body velocity between two poses.
///
/// Points are transformed from the sensor frame at their sampling time into the
/// frame of @p previous_relative_pose, compensating both rotation and
/// translation. Normals, covariances, and photometric gradients are rotated
/// using the angular velocity model to keep them aligned with the deskewed
/// points.
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
    if (!input_cloud.queue.ptr || !output_cloud.queue.ptr) {
        throw std::runtime_error("[deskew_point_cloud_constant_velocity] SYCL queue is not initialized");
    }

    if (input_cloud.queue.ptr->get_context() != output_cloud.queue.ptr->get_context()) {
        throw std::runtime_error(
            "[deskew_point_cloud_constant_velocity] input_cloud and output_cloud must share the same SYCL context");
    }

    const size_t cloud_size = input_cloud.size();
    if (cloud_size == 0 || !input_cloud.has_timestamps()) {
        return false;
    }

    if (delta_time_seconds <= 0.0f) {
        return false;
    }

    const float scan_duration_seconds =
        static_cast<float>((input_cloud.end_time_ms - input_cloud.start_time_ms) * 1e-3);
    if (scan_duration_seconds <= 0.0f) {
        return false;
    }

    if (&input_cloud != &output_cloud) {
        // Copy metadata so the output stays synchronized with the input timing information.
        output_cloud.start_time_ms = input_cloud.start_time_ms;
        output_cloud.end_time_ms = input_cloud.end_time_ms;

        // Mirror input fields into the output so non-deskewed attributes remain intact.
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
            // copy
            output_cloud.rgb->assign(input_cloud.rgb->begin(), input_cloud.rgb->end());
        } else {
            output_cloud.rgb->clear();
        }
        if (input_cloud.has_intensity()) {
            // copy
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

    // Launch device kernel to deskew each point independently.
    auto deskew_event = input_cloud.queue.ptr->submit([&](sycl::handler& h) {
        const auto work_group_size = input_cloud.queue.get_work_group_size();
        const auto global_size = input_cloud.queue.get_global_size(input_cloud.size());

        // Compute motion between the two poses and map it to the twist vector.
        const Eigen::Transform<float, 3, 1> delta_pose = previous_relative_pose.inverse() * current_relative_pose;
        const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(delta_pose);
        std::array<float, 6> delta_twist_array{};
        std::copy(delta_twist.data(), delta_twist.data() + delta_twist.size(), delta_twist_array.begin());

        // Cache raw pointers for the device kernel.
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
        auto* timestamps = output_cloud.timestamp_offsets->data();

        const bool process_normals = normals_in != nullptr && normals_out != nullptr;
        const bool process_covs = covs_in != nullptr && covs_out != nullptr;
        const bool process_color_gradients = color_gradients_in != nullptr && color_gradients_out != nullptr;
        const bool process_intensity_gradients =
            intensity_gradients_in != nullptr && intensity_gradients_out != nullptr;

        h.parallel_for(                                       //
            sycl::nd_range<1>(global_size, work_group_size),  //
            [=](sycl::nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= cloud_size) {
                    return;
                }

                // millisec to sec
                const float timestamp_seconds = timestamps[idx] * 1e-3f;
                if (!sycl::isfinite(timestamp_seconds)) {
                    // copy to output
                    eigen_utils::copy<4, 1>(points_in[idx], points_out[idx]);
                    if (process_normals) {
                        eigen_utils::copy<4, 1>(normals_in[idx], normals_out[idx]);
                    }
                    if (process_covs) {
                        eigen_utils::copy<4, 4>(covs_in[idx], covs_out[idx]);
                    }
                    if (process_color_gradients) {
                        eigen_utils::copy<3, 3>(color_gradients_in[idx], color_gradients_out[idx]);
                    }
                    if (process_intensity_gradients) {
                        eigen_utils::copy<3, 1>(intensity_gradients_in[idx], intensity_gradients_out[idx]);
                    }
                    return;
                }

                // Normalize with respect to the scan duration derived from the point cloud timestamps.
                const float normalized_time = sycl::clamp(timestamp_seconds / scan_duration_seconds, 0.0f, 1.0f);

                Eigen::Vector<float, 6> scaled_twist;
                for (size_t i = 0; i < delta_twist_array.size(); ++i) {
                    scaled_twist[i] = delta_twist_array[i] * normalized_time;
                }

                const Eigen::Matrix4f point_motion = eigen_utils::lie::se3_exp(scaled_twist);

                // Apply motion to deskew the point in the device kernel.
                points_out[idx] = eigen_utils::multiply<4, 4>(point_motion, points_in[idx]);

                // Rotate frame-dependent attributes using the integrated angular velocity.
                const Eigen::Vector3f integrated_omega = scaled_twist.template head<3>();
                const Eigen::Matrix3f rotation_omega =
                    eigen_utils::geometry::quaternion_to_rotation_matrix(eigen_utils::lie::so3_exp(integrated_omega));
                if (process_normals) {
                    normals_out[idx].setZero();
                    const Eigen::Vector3f normal_in = normals_in[idx].template head<3>();
                    normals_out[idx].template head<3>() = eigen_utils::multiply<3, 3>(rotation_omega, normal_in);
                }
                if (process_covs) {
                    covs_out[idx].setZero();
                    const Eigen::Matrix3f cov_in = covs_in[idx].topLeftCorner<3, 3>();
                    const Eigen::Matrix3f rotation_omega_t = eigen_utils::transpose<3, 3>(rotation_omega);
                    const Eigen::Matrix3f rotated_cov = eigen_utils::multiply<3, 3, 3>(
                        rotation_omega, eigen_utils::multiply<3, 3, 3>(cov_in, rotation_omega_t));
                    covs_out[idx].topLeftCorner<3, 3>() = rotated_cov;
                }
                if (process_color_gradients) {
                    color_gradients_out[idx].setZero();
                    color_gradients_out[idx] = eigen_utils::multiply<3, 3, 3>(
                        color_gradients_in[idx], eigen_utils::transpose<3, 3>(rotation_omega));
                }
                if (process_intensity_gradients) {
                    intensity_gradients_out[idx].setZero();
                    intensity_gradients_out[idx] =
                        eigen_utils::multiply<3, 3>(rotation_omega, intensity_gradients_in[idx]);
                }
            });
    });

    deskew_event.wait_and_throw();

    return true;
}

}  // namespace deskew
}  // namespace algorithms
}  // namespace sycl_points
