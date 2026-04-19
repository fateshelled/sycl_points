#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <iostream>
#include <memory>

#include "sycl_points/algorithms/registration/result.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_odometry {

/// @brief Predicts the next-frame initial pose using velocity from the previous frame,
///        with optional degeneracy-based scaling of rotation and translation components.
class AdaptiveMotionPredictor {
public:
    using Ptr = std::shared_ptr<AdaptiveMotionPredictor>;

    struct Params {
        struct AdaptiveAxis {
            bool enable = true;
            float factor_min = 0.2f;
            float factor_max = 1.0f;
            float min_eigenvalue_low = 1.0f;
            float min_eigenvalue_high = 10.0f;
        };

        struct Adaptive {
            AdaptiveAxis rotation = {.enable = true,
                                     .factor_min = 0.2f,
                                     .factor_max = 1.0f,
                                     .min_eigenvalue_low = 5.0f,
                                     .min_eigenvalue_high = 10.0f};
            AdaptiveAxis translation;
        };

        float static_factor = 0.5f;
        bool verbose = false;
        Adaptive adaptive;
    };

    explicit AdaptiveMotionPredictor(const Params& params) : params_(params) {}

    /// @brief Predict the initial ICP pose for the current frame.
    /// @param linear_velocity   [m/s] in previous LiDAR body frame
    /// @param angular_velocity  [rad/s]
    /// @param odom              Current T_odom_to_lidar
    /// @param dt                Frame interval [s]
    /// @param reg_result        Previous registration result (used for Hessian-based degeneracy factors)
    /// @param registrated       Whether a valid registration result exists
    /// @return Predicted absolute pose T_odom_to_lidar_curr
    Eigen::Isometry3f predict(const Eigen::Vector3f& linear_velocity, const Eigen::AngleAxisf& angular_velocity,
                              const Eigen::Isometry3f& odom, float dt,
                              const algorithms::registration::RegistrationResult::Ptr& reg_result,
                              bool registrated) const {
        float rot_factor = params_.static_factor;
        float trans_factor = params_.static_factor;

        if (registrated && reg_result->inlier > 0) {
            if (params_.adaptive.rotation.enable) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_rot(reg_result->H_raw.block<3, 3>(0, 0));
                if (solver_rot.info() == Eigen::Success) {
                    const float low = params_.adaptive.rotation.min_eigenvalue_low;
                    const float high = params_.adaptive.rotation.min_eigenvalue_high;
                    const float max_factor = params_.adaptive.rotation.factor_max;
                    const float min_factor = params_.adaptive.rotation.factor_min;

                    const float min_eig_ratio = solver_rot.eigenvalues().minCoeff() / reg_result->inlier;
                    const float score = std::clamp((min_eig_ratio - low) / std::max(high - low, 1e-6f), 0.0f, 1.0f);
                    // When degenerate (score→0), apply full motion (max_factor) to prevent map distortion
                    // by maintaining constant-velocity assumption. When well-constrained (score→1),
                    // ICP can correct the pose, so motion prediction is dampened (min_factor).
                    rot_factor = max_factor * (1.0f - score) + min_factor * score;

                    if (params_.verbose) {
                        std::cout << "[motion predictor] rot: factor=" << rot_factor << ", eigen value=["
                                  << solver_rot.eigenvalues().transpose() / reg_result->inlier << "]" << std::endl;
                    }
                }
            }

            if (params_.adaptive.translation.enable) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_trans(reg_result->H_raw.block<3, 3>(3, 3));
                if (solver_trans.info() == Eigen::Success) {
                    const float low = params_.adaptive.translation.min_eigenvalue_low;
                    const float high = params_.adaptive.translation.min_eigenvalue_high;
                    const float max_factor = params_.adaptive.translation.factor_max;
                    const float min_factor = params_.adaptive.translation.factor_min;

                    const float min_eig_ratio = solver_trans.eigenvalues().minCoeff() / reg_result->inlier;
                    const float score = std::clamp((min_eig_ratio - low) / std::max(high - low, 1e-6f), 0.0f, 1.0f);
                    // Same intent as rot_factor: trust motion model more when degenerate.
                    trans_factor = max_factor * (1.0f - score) + min_factor * score;
                }
                if (params_.verbose) {
                    std::cout << "[motion predictor] trans: factor=" << trans_factor << ", eigen value=["
                              << solver_trans.eigenvalues().transpose() / reg_result->inlier << "]" << std::endl;
                }
            }
        }

        const Eigen::Vector3f delta_trans = linear_velocity * dt;
        const Eigen::AngleAxisf delta_angle_axis(angular_velocity.angle() * dt, angular_velocity.axis());

        const Eigen::Vector3f predicted_trans = odom.translation() + odom.rotation() * (delta_trans * trans_factor);
        const Eigen::Quaternionf predicted_rot =
            Eigen::Quaternionf(odom.rotation()) *
            Eigen::AngleAxisf(delta_angle_axis.angle() * rot_factor, delta_angle_axis.axis());

        Eigen::Isometry3f init_T = Eigen::Isometry3f::Identity();
        init_T.translation() = predicted_trans;
        init_T.rotate(predicted_rot.normalized());
        return init_T;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Params params_;
};

}  // namespace lidar_odometry
}  // namespace pipeline
}  // namespace sycl_points
