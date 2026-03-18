#pragma once

#include <algorithm>
#include <iostream>
#include <memory>

#include "sycl_points/algorithms/registration/pipeline/aligner.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {
namespace pipeline {

/// @brief Repeats deskew and registration assuming constant sensor velocity
class VelocityUpdateAligner {
public:
    using Ptr = std::shared_ptr<VelocityUpdateAligner>;

    /// @brief Constructor
    /// @param aligner Wrapped aligner to execute after each deskew update
    /// @param velocity_update_iter Number of deskew and re-alignment iterations
    /// @param verbose If true, print deskew progress
    VelocityUpdateAligner(RegistrationAligner aligner, size_t velocity_update_iter, bool verbose = false)
        : aligner_(std::move(aligner)), velocity_update_iter_(velocity_update_iter), verbose_(verbose) {}

    /// @brief Constructor
    /// @param registration Registration backend to wrap
    /// @param velocity_update_iter Number of deskew and re-alignment iterations
    /// @param verbose If true, print deskew progress
    VelocityUpdateAligner(const Registration::Ptr& registration, size_t velocity_update_iter, bool verbose = false)
        : VelocityUpdateAligner(make_registration_aligner(registration), velocity_update_iter, verbose) {}

    /// @brief Aligns point clouds while iteratively updating the deskewed source cloud
    /// @param source Source point cloud
    /// @param target Target point cloud
    /// @param target_knn KNN structure built on the target point cloud
    /// @param initial_guess Initial transformation matrix
    /// @param options Per-call execution options including previous pose and time delta
    /// @note If the source has no timestamps, deskew is skipped and the original source cloud is forwarded instead.
    /// @return Registration result from the final deskew iteration
    RegistrationResult align(const PointCloudShared& source, const PointCloudShared& target,
                             const knn::KNNBase& target_knn,
                             const TransformMatrix& initial_guess = TransformMatrix::Identity(),
                             const Registration::ExecutionOptions& options = Registration::ExecutionOptions()) const {
        RegistrationResult result;
        result.T.matrix() = initial_guess;

        if (source.size() == 0) {
            return result;
        }

        // Use a dedicated working buffer so deskew writes into a non-aliased output cloud.
        this->deskewed_pc_ = std::make_shared<PointCloudShared>(source.queue);

        if (!source.has_timestamps()) {
            if (this->verbose_) {
                std::cout << "deskew skipped: source has no timestamps" << std::endl;
            }
            *this->deskewed_pc_ = source;
            return this->aligner_(*this->deskewed_pc_, target, target_knn, result.T.matrix(), options);
        }

        const size_t deskew_levels = std::max<size_t>(1, this->velocity_update_iter_);

        for (size_t deskew_iter = 0; deskew_iter < deskew_levels; ++deskew_iter) {
            if (this->verbose_) {
                std::cout << "deskewed: " << deskew_iter << std::endl;
            }

            const Eigen::Isometry3f delta_pose = Eigen::Isometry3f(options.prev_pose).inverse() * result.T;
            const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(delta_pose);
            const float delta_angle = delta_twist.head<3>().norm();
            const float delta_dist = delta_twist.tail<3>().norm();
            if (this->verbose_) {
                std::cout << "deskewed[" << deskew_iter << "]: angle=" << delta_angle << ", dist=" << delta_dist
                          << std::endl;
            }

            deskew::deskew_point_cloud_constant_velocity(source, *this->deskewed_pc_,
                                                         Eigen::Isometry3f(options.prev_pose), result.T);
            result = this->aligner_(*this->deskewed_pc_, target, target_knn, result.T.matrix(), options);
        }

        return result;
    }

    /// @brief Returns the deskewed source point cloud used by the most recent align() call
    const PointCloudShared* get_deskewed_point_cloud() const { return this->deskewed_pc_.get(); }

    /// @brief Exposes this aligner as a RegistrationAligner
    RegistrationAligner make_aligner() const {
        return [this](const PointCloudShared& source, const PointCloudShared& target, const knn::KNNBase& target_knn,
                      const TransformMatrix& initial_guess, const Registration::ExecutionOptions& options) {
            return this->align(source, target, target_knn, initial_guess, options);
        };
    }

private:
    RegistrationAligner aligner_;
    size_t velocity_update_iter_ = 1;
    bool verbose_ = false;
    mutable PointCloudShared::Ptr deskewed_pc_ = nullptr;
};

}  // namespace pipeline
}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
