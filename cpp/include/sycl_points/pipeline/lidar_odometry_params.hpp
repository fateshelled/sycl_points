#pragma once

#include "sycl_points/algorithms/registration/registration_pipeline_params.hpp"
#include "sycl_points/pipeline/adaptive_motion_predictor.hpp"
#include "sycl_points/pipeline/odometry_common_params.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_odometry {

/// @brief Parameters specific to the LiDAR-only odometry pipeline.
struct Parameters : public odometry::CommonParameters {
    using MotionPrediction = AdaptiveMotionPredictor::Params;

    struct LO {
        struct Registration {
            using Criteria = algorithms::registration::RegistrationConvergenceCriteria;

            size_t max_iterations = 20;
            Criteria criteria;
            algorithms::registration::RegistrationOptimizationParams optimization;
            algorithms::registration::DegenerateRegularizationParams degenerate_regularization;
            algorithms::registration::MapPriorParams map_prior;
        };

        struct Pipeline {
            algorithms::registration::RegistrationRobustScheduleParams robust;
            algorithms::registration::RegistrationVelocityUpdateParams velocity_update;
        };

        Registration registration;
        Pipeline pipeline;
    };

    MotionPrediction motion_prediction;
    LO lo;

    algorithms::registration::RegistrationPipelineParams make_registration_pipeline_params() const {
        algorithms::registration::RegistrationPipelineParams result;
        result.registration =
            algorithms::registration::RegistrationParams(registration.factor, lo.registration.optimization);
        result.registration.max_iterations = lo.registration.max_iterations;
        result.registration.criteria = lo.registration.criteria;
        result.registration.degenerate_reg = lo.registration.degenerate_regularization;
        result.registration.map_prior = lo.registration.map_prior;
        result.random_sampling = registration_sampling;
        result.robust = lo.pipeline.robust;
        result.velocity_update = lo.pipeline.velocity_update;
        return result;
    }
};

}  // namespace lidar_odometry
}  // namespace pipeline
}  // namespace sycl_points
