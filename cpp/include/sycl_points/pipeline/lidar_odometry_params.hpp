#pragma once

#include "sycl_points/pipeline/adaptive_motion_predictor.hpp"
#include "sycl_points/pipeline/odometry_common_params.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_odometry {

/// @brief Parameters specific to the LiDAR-only odometry pipeline.
struct Parameters : public odometry::CommonParameters {
    using MotionPrediction = AdaptiveMotionPredictor::Params;

    MotionPrediction motion_prediction;
};

}  // namespace lidar_odometry
}  // namespace pipeline
}  // namespace sycl_points
