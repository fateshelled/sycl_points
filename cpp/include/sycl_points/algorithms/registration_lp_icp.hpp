#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>

#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/algorithms/knn/knn.hpp>
#include <sycl_points/algorithms/localizability_constrained_optimizer.hpp>
#include <sycl_points/algorithms/localizability_detection.hpp>
#include <sycl_points/algorithms/registration_factor.hpp>
#include <sycl_points/algorithms/registration_result.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {

namespace algorithms {

namespace registration {

/// @brief Registration result with localizability information
struct LPRegistrationResult : public RegistrationResult {
    /// Localizability detection result (valid for first iteration)
    localizability::LocalizabilityResult localizability;

    /// Flag indicating if localizability detection was performed
    bool has_localizability = false;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Parameters for LP-ICP registration
struct LPRegistrationParams {
    /// Base registration parameters
    struct Criteria {
        float translation = 1e-3f;  // translation tolerance
        float rotation = 1e-3f;     // rotation tolerance [rad]
    };

    /// Registration type
    RegType reg_type = RegType::POINT_TO_PLANE;

    /// Maximum iterations
    size_t max_iterations = 20;

    /// Damping factor
    float lambda = 1e-6f;

    /// Maximum correspondence distance
    float max_correspondence_distance = 2.0f;

    /// Convergence criteria
    Criteria criteria;

    /// Localizability detection parameters
    localizability::LocalizabilityParams localizability_params;

    /// Constrained optimizer parameters
    localizability::ConstrainedOptimizerParams optimizer_params;

    /// Enable localizability-aware optimization
    bool enable_localizability = true;

    /// Enable debug output
    bool verbose = false;
};

/// @brief LP-ICP: Localizability-aware Point-to-Plane ICP Registration
/// @details Implements the LP-ICP framework that detects and handles
/// degenerate cases (tunnels, corridors, etc.) through localizability analysis
class LPRegistration {
public:
    using Ptr = std::shared_ptr<LPRegistration>;

    /// @brief Constructor
    /// @param queue SYCL device queue
    /// @param params Registration parameters
    LPRegistration(const sycl_utils::DeviceQueue& queue, const LPRegistrationParams& params = LPRegistrationParams())
        : params_(params), queue_(queue) {
        neighbors_ = std::make_shared<shared_vector<knn::KNNResult>>(1, knn::KNNResult(), *queue_.ptr);
        neighbors_->at(0).allocate(queue_, 1, 1);

        localizability_detector_ =
            std::make_shared<localizability::LocalizabilityDetection>(queue_, params_.localizability_params);
        constrained_optimizer_ =
            std::make_shared<localizability::ConstrainedOptimizer>(params_.optimizer_params);
    }

    /// @brief Perform LP-ICP registration
    /// @param source Source point cloud
    /// @param target Target point cloud
    /// @param target_knn Target KNN search structure
    /// @param initial_guess Initial transformation matrix
    /// @return Registration result with localizability information
    LPRegistrationResult align(const PointCloudShared& source, const PointCloudShared& target,
                                const knn::KNNBase& target_knn,
                                const TransformMatrix& initial_guess = TransformMatrix::Identity());

    /// @brief Get parameters
    const LPRegistrationParams& getParams() const { return params_; }

    /// @brief Set parameters
    void setParams(const LPRegistrationParams& params) {
        params_ = params;
        localizability_detector_->setParams(params_.localizability_params);
        constrained_optimizer_->setParams(params_.optimizer_params);
    }

    /// @brief Get localizability detector
    localizability::LocalizabilityDetection::Ptr getLocalizabilityDetector() const { return localizability_detector_; }

    /// @brief Get constrained optimizer
    localizability::ConstrainedOptimizer::Ptr getConstrainedOptimizer() const { return constrained_optimizer_; }

private:
    LPRegistrationParams params_;
    sycl_utils::DeviceQueue queue_;

    shared_vector_ptr<knn::KNNResult> neighbors_;
    std::shared_ptr<localizability::LocalizabilityDetection> localizability_detector_;
    std::shared_ptr<localizability::ConstrainedOptimizer> constrained_optimizer_;

    /// @brief Check convergence
    bool isConverged(const Eigen::Vector<float, 6>& delta) const {
        return delta.head<3>().norm() < params_.criteria.rotation &&
               delta.tail<3>().norm() < params_.criteria.translation;
    }

    /// @brief Linearize registration factor
    LinearizedResult linearize(const PointCloudShared& source, const PointCloudShared& target,
                                const TransformMatrix& T, float max_correspondence_distance);
};

// Implementation

inline LPRegistrationResult LPRegistration::align(const PointCloudShared& source, const PointCloudShared& target,
                                                    const knn::KNNBase& target_knn, const TransformMatrix& initial_guess) {
    const size_t N = source.size();
    LPRegistrationResult result;
    result.T.matrix() = initial_guess;

    if (N == 0) {
        return result;
    }

    // Validate requirements for point-to-plane ICP
    if (params_.reg_type == RegType::POINT_TO_PLANE || params_.reg_type == RegType::GENZ) {
        if (!target.has_normal()) {
            if (target.has_cov()) {
                covariance::compute_normals_from_covariances(target);
            } else {
                throw std::runtime_error(
                    "[LPRegistration] Normal vectors required for point-to-plane ICP. "
                    "Please compute normals or covariances before registration.");
            }
        }
    }

    const float max_corr_dist = params_.max_correspondence_distance;
    localizability::LocalizabilityResult localizability_result;
    bool first_iteration = true;

    for (size_t iter = 0; iter < params_.max_iterations; ++iter) {
        // Step 1: Nearest neighbor search
        auto knn_event = target_knn.nearest_neighbor_search_async(source, (*neighbors_)[0], {}, result.T.matrix());
        knn_event.evs.back().wait();

        // Step 2: Linearize registration factor
        const LinearizedResult linearized = linearize(source, target, result.T.matrix(), max_corr_dist);

        // Step 3: Localizability detection (first iteration only)
        if (first_iteration && params_.enable_localizability) {
            localizability_result = localizability_detector_->detect(linearized.H, source, target, (*neighbors_)[0],
                                                                      result.T.matrix(), max_corr_dist);
            result.localizability = localizability_result;
            result.has_localizability = true;
            first_iteration = false;

            if (params_.verbose) {
                std::cout << "[LPRegistration] Localizability categories: ";
                for (size_t j = 0; j < 6; ++j) {
                    switch (localizability_result.aggregate.categories[j]) {
                        case localizability::LocalizabilityCategory::FULL:
                            std::cout << "F";
                            break;
                        case localizability::LocalizabilityCategory::PARTIAL:
                            std::cout << "P";
                            break;
                        case localizability::LocalizabilityCategory::NONE:
                            std::cout << "N";
                            break;
                    }
                }
                std::cout << std::endl;
            }
        }

        // Step 4: Solve optimization
        Eigen::Vector<float, 6> delta;
        if (params_.enable_localizability && result.has_localizability) {
            // Use constrained optimization
            delta = constrained_optimizer_->solve(linearized.H, linearized.b, localizability_result);
        } else {
            // Standard Gauss-Newton update
            const Eigen::Matrix<float, 6, 6> H_damped =
                linearized.H + params_.lambda * Eigen::Matrix<float, 6, 6>::Identity();
            delta = H_damped.ldlt().solve(-linearized.b);
        }

        // Step 5: Update pose
        result.T = result.T * Eigen::Isometry3f(eigen_utils::lie::se3_exp(delta));
        result.iterations = iter + 1;
        result.H = linearized.H;
        result.b = linearized.b;
        result.error = linearized.error;
        result.inlier = linearized.inlier;

        if (params_.verbose) {
            std::cout << "[LPRegistration] iter " << iter << ": error=" << result.error
                      << ", inlier=" << result.inlier << ", dt=" << delta.tail<3>().norm()
                      << ", dr=" << delta.head<3>().norm() << std::endl;
        }

        // Step 6: Check convergence
        if (isConverged(delta)) {
            result.converged = true;
            break;
        }
    }

    return result;
}

inline LinearizedResult LPRegistration::linearize(const PointCloudShared& source, const PointCloudShared& target,
                                                    const TransformMatrix& T, float max_correspondence_distance) {
    const size_t N = source.size();
    const float max_dist_sq = max_correspondence_distance * max_correspondence_distance;
    const auto T_sycl = eigen_utils::to_sycl_vec(T);

    LinearizedResult result;

    const auto* source_pts = source.points_ptr();
    const auto* target_pts = target.points_ptr();
    const auto* target_normals = target.has_normal() ? target.normals_ptr() : nullptr;
    const auto* source_covs = source.has_cov() ? source.covs_ptr() : nullptr;
    const auto* target_covs = target.has_cov() ? target.covs_ptr() : nullptr;
    const auto* neighbor_indices = (*neighbors_)[0].indices->data();
    const auto* neighbor_distances = (*neighbors_)[0].distances->data();

    for (size_t i = 0; i < N; ++i) {
        if (neighbor_distances[i] > max_dist_sq) {
            continue;
        }

        const int32_t target_idx = neighbor_indices[i];
        if (target_idx < 0) {
            continue;
        }

        const PointType& source_pt = source_pts[i];
        const PointType& target_pt = target_pts[target_idx];
        const Normal target_normal =
            target_normals ? target_normals[target_idx] : Normal(0.0f, 0.0f, 1.0f, 0.0f);
        const Covariance source_cov = source_covs ? source_covs[i] : Covariance::Identity();
        const Covariance target_cov = target_covs ? target_covs[target_idx] : Covariance::Identity();

        float residual_norm = 0.0f;
        LinearizedResult factor;

        switch (params_.reg_type) {
            case RegType::POINT_TO_POINT:
                factor = kernel::linearize_point_to_point(T_sycl, source_pt, target_pt, residual_norm);
                break;
            case RegType::POINT_TO_PLANE:
                factor = kernel::linearize_point_to_plane(T_sycl, source_pt, target_pt, target_normal, residual_norm);
                break;
            case RegType::GICP:
                factor = kernel::linearize_gicp(T_sycl, source_pt, source_cov, target_pt, target_cov, residual_norm);
                break;
            case RegType::GENZ: {
                float pt2pt_norm = 0.0f, pt2pl_norm = 0.0f;
                const auto pt2pt =
                    kernel::linearize_point_to_point(T_sycl, source_pt, target_pt, pt2pt_norm);
                const auto pt2pl =
                    kernel::linearize_point_to_plane(T_sycl, source_pt, target_pt, target_normal, pt2pl_norm);
                const float alpha = 0.5f;  // GenZ alpha
                factor.H = eigen_utils::add<6, 6>(eigen_utils::multiply<6, 6>(pt2pt.H, 1.0f - alpha),
                                                   eigen_utils::multiply<6, 6>(pt2pl.H, alpha));
                factor.b = eigen_utils::add<6, 1>(eigen_utils::multiply<6>(pt2pt.b, 1.0f - alpha),
                                                   eigen_utils::multiply<6>(pt2pl.b, alpha));
                factor.error = pt2pt.error * (1.0f - alpha) + pt2pl.error * alpha;
                factor.inlier = 1;
                residual_norm = pt2pt_norm * (1.0f - alpha) + pt2pl_norm * alpha;
            } break;
        }

        // Accumulate
        eigen_utils::add_inplace<6, 6>(result.H, factor.H);
        eigen_utils::add_inplace<6, 1>(result.b, factor.b);
        result.error += factor.error;
        result.inlier += factor.inlier;
    }

    return result;
}

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
