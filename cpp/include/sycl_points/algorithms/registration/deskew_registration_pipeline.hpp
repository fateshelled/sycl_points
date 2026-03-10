#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>

#include "sycl_points/algorithms/deskew/relative_pose_deskew.hpp"
#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Pipeline for deskew loop scheduling and deskew execution.
class DeskewRegistrationPipeline {
public:
    explicit DeskewRegistrationPipeline(size_t velocity_update_iter)
        : deskew_levels_(std::max<size_t>(1, velocity_update_iter)) {}

    template <typename PoseGetter, typename Func>
    void run(const PointCloudShared& source, PointCloudShared& deskewed, const TransformMatrix& prev_pose, float dt,
             bool verbose, PoseGetter&& pose_getter, Func&& func) const {
        for (size_t deskew_iter = 0; deskew_iter < deskew_levels_; ++deskew_iter) {
            const TransformMatrix current_pose = pose_getter();

            if (verbose) {
                std::cout << "deskewed: " << deskew_iter << std::endl;
            }

            const Eigen::Isometry3f delta_pose = Eigen::Isometry3f(prev_pose).inverse() * Eigen::Isometry3f(current_pose);
            const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(delta_pose);
            const float delta_angle = delta_twist.head<3>().norm();
            const float delta_dist = delta_twist.tail<3>().norm();

            if (verbose) {
                std::cout << "deskewed[" << deskew_iter << "]: angle=" << delta_angle << ", dist=" << delta_dist
                          << std::endl;
            }

            // Recompute source points using the latest pose estimate before each optimization pass.
            deskew::deskew_point_cloud_constant_velocity(source, deskewed, Eigen::Isometry3f(prev_pose),
                                                         Eigen::Isometry3f(current_pose), dt);

            func(deskew_iter, deskewed);
        }
    }

private:
    size_t deskew_levels_ = 1;
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
