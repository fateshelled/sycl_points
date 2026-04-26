#pragma once

#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/pipeline/lidar_odometry_params.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_inertial_odometry {

/// @brief Parameters for the LiDAR-Inertial Odometry pipeline.
///
/// Extends lidar_odometry::Parameters with an LIO-specific optimization block.
/// IMU is always required; params_.imu.enable is forced true in the pipeline.
struct Parameters : public lidar_odometry::Parameters {
    struct LIO {
        /// Maximum number of Gauss-Newton iterations per frame.
        size_t max_iterations = 10;
        // Convergence threshold
        algorithms::registration::RegistrationParams::Criteria criteria;
        /// Regularization factor for velocity and bias
        float invalid_regularization_factor = 1e4f;
    };

    LIO lio;
};

}  // namespace lidar_inertial_odometry
}  // namespace pipeline
}  // namespace sycl_points
