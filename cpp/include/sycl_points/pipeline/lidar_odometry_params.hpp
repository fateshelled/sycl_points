#pragma once

#include "sycl_points/algorithms/registration/registration_pipeline_params.hpp"
#include "sycl_points/pipeline/motion_predictor.hpp"
#include "sycl_points/pipeline/odometry_common_params.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_odometry {

/// @brief Parameters specific to the LiDAR-only odometry pipeline.
struct Parameters : public odometry::CommonParameters {
    using MotionPrediction = MotionPredictor::Params;

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

    /// @brief Sliding-window graph optimizer (local BA) parameters.
    struct Graph {
        size_t window_size = 5;                  ///< persistent keyframe nodes
        size_t solver_iterations = 10;           ///< GN iterations per frame
        float convergence_translation = 1e-4f;   ///< [m]
        float convergence_rotation = 1e-4f;      ///< [rad]
        float relinearize_translation_thresh = 0.05f;  ///< [m] delayed relinearization
        float relinearize_rotation_thresh = 0.02f;     ///< [rad]
        float marginalization_lambda = 1e-6f;    ///< fallback regularization
        float chain_sigma_rotation = 5e-3f;      ///< [rad] RelativePoseFactor info
        float chain_sigma_translation = 2e-2f;   ///< [m]
        // Robust scale ladder (GNC) for per-frame tip factors. Disabled by
        // default: factors then use registration/robust/default_scale as before.
        bool robust_enable = false;
        float robust_init_scale = 10.0f;
        float robust_min_scale = 1.25f;
        size_t robust_levels = 4;
        size_t robust_iters_per_level = 2;
        bool robust_relinearize_per_rung = false;
    };

    MotionPrediction motion_prediction;
    LO lo;
    Graph graph;

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
