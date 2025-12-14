#pragma once

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <vector>

#include <sycl_points/algorithms/knn/knn.hpp>
#include <sycl_points/algorithms/registration_factor.hpp>
#include <sycl_points/algorithms/registration_result.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace localizability {

/// @brief Localizability category for each 6-DoF direction
enum class LocalizabilityCategory {
    FULL = 0,     ///< Fully localizable - no additional constraints needed
    PARTIAL = 1,  ///< Partially localizable - soft constraints applied
    NONE = 2      ///< Non-localizable - hard constraints applied
};

/// @brief Parameters for localizability detection
struct LocalizabilityParams {
    /// Noise filtering threshold (h_f in paper)
    float noise_threshold = 0.03f;

    /// High contribution threshold (h_u in paper)
    float high_contribution_threshold = 0.4998f;

    /// Thresholds for tri-value classification
    float T1 = 50.0f;  ///< Threshold for L_f >= T1 -> Full
    float T2 = 30.0f;  ///< Threshold for L_u >= T2 -> Full
    float T3 = 15.0f;  ///< Threshold for L_f >= T3 && L_u >= T4 -> Partial
    float T4 = 9.0f;   ///< Threshold for L_u >= T4 -> Partial

    /// Enable debug output
    bool verbose = false;
};

/// @brief Localizability contribution for a single correspondence
struct LocalizabilityContribution {
    /// Rotation direction contributions (3D vector projected to eigenvectors)
    Eigen::Vector3f F_r = Eigen::Vector3f::Zero();

    /// Translation direction contributions (3D vector projected to eigenvectors)
    Eigen::Vector3f F_t = Eigen::Vector3f::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Aggregated localizability for each 6-DoF direction
struct LocalizabilityAggregate {
    /// Medium contribution sums L_f (after noise filtering)
    Eigen::Vector<float, 6> L_f = Eigen::Vector<float, 6>::Zero();

    /// High contribution sums L_u
    Eigen::Vector<float, 6> L_u = Eigen::Vector<float, 6>::Zero();

    /// Category for each direction [rx, ry, rz, tx, ty, tz]
    std::array<LocalizabilityCategory, 6> categories = {LocalizabilityCategory::NONE, LocalizabilityCategory::NONE,
                                                         LocalizabilityCategory::NONE, LocalizabilityCategory::NONE,
                                                         LocalizabilityCategory::NONE, LocalizabilityCategory::NONE};

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Soft constraint for partially localizable directions
struct SoftConstraint {
    /// Direction index (0-5 for rx, ry, rz, tx, ty, tz)
    size_t direction_index = 0;

    /// Constraint value delta_x'_ji from small-scale ICP
    float constraint_value = 0.0f;

    /// Eigenvector for this direction
    Eigen::Vector<float, 6> eigenvector = Eigen::Vector<float, 6>::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Hard constraint matrix for non-localizable directions
struct HardConstraint {
    /// Constraint matrix D where D * delta_x = 0
    Eigen::Matrix<float, Eigen::Dynamic, 6> D;

    /// Number of hard constraints
    size_t num_constraints = 0;

    HardConstraint() : D(6, 6), num_constraints(0) { D.setZero(); }

    void addConstraint(const Eigen::Vector<float, 6>& v) {
        if (num_constraints < 6) {
            D.row(num_constraints) = v.transpose();
            ++num_constraints;
        }
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Result of localizability detection
struct LocalizabilityResult {
    /// Aggregated localizability information
    LocalizabilityAggregate aggregate;

    /// Rotation eigenvectors (columns are eigenvectors, sorted by eigenvalue ascending)
    Eigen::Matrix3f V_r = Eigen::Matrix3f::Identity();

    /// Translation eigenvectors (columns are eigenvectors, sorted by eigenvalue ascending)
    Eigen::Matrix3f V_t = Eigen::Matrix3f::Identity();

    /// Rotation eigenvalues (sorted ascending)
    Eigen::Vector3f eigenvalues_r = Eigen::Vector3f::Zero();

    /// Translation eigenvalues (sorted ascending)
    Eigen::Vector3f eigenvalues_t = Eigen::Vector3f::Zero();

    /// Soft constraints for partially localizable directions
    std::vector<SoftConstraint> soft_constraints;

    /// Hard constraints for non-localizable directions
    HardConstraint hard_constraint;

    /// Full 6x6 eigenvectors (combined rotation and translation)
    Eigen::Matrix<float, 6, 6> V = Eigen::Matrix<float, 6, 6>::Identity();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

namespace kernel {

/// @brief Compute SE(3) Jacobian for a point correspondence
/// @param T Transform matrix as sycl float4 array
/// @param source_pt Source point
/// @return 3x6 Jacobian matrix [rotation_part | translation_part]
SYCL_EXTERNAL inline Eigen::Matrix<float, 3, 6> compute_jacobian_3x6(const std::array<sycl::float4, 4>& T,
                                                                      const PointType& source_pt) {
    Eigen::Matrix<float, 3, 6> J = Eigen::Matrix<float, 3, 6>::Zero();

    // Rotation part: R * skew(source_pt)
    const Eigen::Matrix3f skewed = eigen_utils::lie::skew(source_pt);
    const Eigen::Matrix3f R = eigen_utils::from_sycl_vec(T).block<3, 3>(0, 0);
    const Eigen::Matrix3f T_skewed = eigen_utils::multiply<3, 3, 3>(R, skewed);

    // First 3 columns: rotation part
    J.block<3, 3>(0, 0) = T_skewed;

    // Last 3 columns: translation part = -R
    J.block<3, 3>(0, 3) = eigen_utils::multiply<3, 3>(R, -1.0f);

    return J;
}

/// @brief Compute Jacobian for point-to-plane correspondence
/// @param T Transform matrix
/// @param source_pt Source point
/// @param target_normal Target surface normal
/// @return 1x6 Jacobian row vector
SYCL_EXTERNAL inline Eigen::Matrix<float, 1, 6> compute_point_to_plane_jacobian(const std::array<sycl::float4, 4>& T,
                                                                                  const PointType& source_pt,
                                                                                  const Normal& target_normal) {
    const Eigen::Matrix<float, 3, 6> J_3x6 = compute_jacobian_3x6(T, source_pt);
    const Eigen::Vector3f n = target_normal.head<3>();

    // Project onto normal: n^T * J
    return eigen_utils::multiply<1, 3, 6>(n.transpose(), J_3x6);
}

/// @brief Compute localizability contribution for a single correspondence
/// @param J_i Jacobian matrix for correspondence i (1x6 or 3x6)
/// @param V_r Rotation eigenvectors (3x3, columns are eigenvectors)
/// @param V_t Translation eigenvectors (3x3, columns are eigenvectors)
/// @param normalize_rotation Whether to normalize rotation Jacobian (for scale unification)
/// @return Localizability contribution for this correspondence
SYCL_EXTERNAL inline LocalizabilityContribution compute_contribution(const Eigen::Matrix<float, 1, 6>& J_i,
                                                                       const Eigen::Matrix3f& V_r,
                                                                       const Eigen::Matrix3f& V_t,
                                                                       bool normalize_rotation) {
    LocalizabilityContribution contrib;

    // Extract rotation and translation parts of Jacobian
    Eigen::Vector3f J_rot = J_i.block<1, 3>(0, 0).transpose();
    const Eigen::Vector3f J_trans = J_i.block<1, 3>(0, 3).transpose();

    // Normalize rotation Jacobian for scale unification (Eq. 23 in paper)
    if (normalize_rotation) {
        const float norm = eigen_utils::frobenius_norm<3>(J_rot);
        if (norm > 1.0f) {
            J_rot = eigen_utils::multiply<3>(J_rot, 1.0f / norm);
        }
    }

    // Project onto eigenvectors (Eq. 23-26 in paper)
    // F_ri = |J_rot^T * v_ri|^2 for each eigenvector v_ri
    for (size_t j = 0; j < 3; ++j) {
        const Eigen::Vector3f v_r = V_r.col(j);
        const Eigen::Vector3f v_t = V_t.col(j);

        const float proj_r = eigen_utils::dot<3>(J_rot, v_r);
        const float proj_t = eigen_utils::dot<3>(J_trans, v_t);

        contrib.F_r(j) = proj_r * proj_r;
        contrib.F_t(j) = proj_t * proj_t;
    }

    return contrib;
}

}  // namespace kernel

/// @brief Localizability Detection Module for LP-ICP
class LocalizabilityDetection {
public:
    using Ptr = std::shared_ptr<LocalizabilityDetection>;

    /// @brief Constructor
    /// @param queue SYCL device queue
    /// @param params Detection parameters
    LocalizabilityDetection(const sycl_utils::DeviceQueue& queue,
                             const LocalizabilityParams& params = LocalizabilityParams())
        : params_(params), queue_(queue) {}

    /// @brief Detect localizability from Hessian matrix and correspondences
    /// @param H Hessian matrix (6x6)
    /// @param source Source point cloud
    /// @param target Target point cloud
    /// @param neighbors KNN search results (correspondences)
    /// @param T Current transformation matrix
    /// @param max_correspondence_distance Maximum correspondence distance
    /// @return Localizability detection result
    LocalizabilityResult detect(const Eigen::Matrix<float, 6, 6>& H, const PointCloudShared& source,
                                 const PointCloudShared& target, const knn::KNNResult& neighbors,
                                 const TransformMatrix& T, float max_correspondence_distance);

    /// @brief Get parameters
    const LocalizabilityParams& getParams() const { return params_; }

    /// @brief Set parameters
    void setParams(const LocalizabilityParams& params) { params_ = params; }

private:
    LocalizabilityParams params_;
    sycl_utils::DeviceQueue queue_;

    /// @brief Decompose Hessian into rotation and translation submatrices
    void decomposeHessian(const Eigen::Matrix<float, 6, 6>& H, Eigen::Matrix3f& H_r, Eigen::Matrix3f& H_t,
                          Eigen::Vector3f& eigenvalues_r, Eigen::Vector3f& eigenvalues_t, Eigen::Matrix3f& V_r,
                          Eigen::Matrix3f& V_t);

    /// @brief Compute localizability contributions for all correspondences
    std::vector<LocalizabilityContribution> computeContributions(const PointCloudShared& source,
                                                                   const PointCloudShared& target,
                                                                   const knn::KNNResult& neighbors,
                                                                   const TransformMatrix& T,
                                                                   float max_correspondence_distance,
                                                                   const Eigen::Matrix3f& V_r,
                                                                   const Eigen::Matrix3f& V_t);

    /// @brief Apply noise filtering and high contribution selection
    void filterContributions(const std::vector<LocalizabilityContribution>& contributions,
                              std::vector<Eigen::Vector<float, 6>>& F_f, std::vector<Eigen::Vector<float, 6>>& F_u);

    /// @brief Aggregate contributions and classify each direction
    LocalizabilityAggregate aggregateAndClassify(const std::vector<Eigen::Vector<float, 6>>& F_f,
                                                   const std::vector<Eigen::Vector<float, 6>>& F_u);

    /// @brief Build constraints for partial and non-localizable directions
    void buildConstraints(LocalizabilityResult& result, const PointCloudShared& source, const PointCloudShared& target,
                           const knn::KNNResult& neighbors, const TransformMatrix& T, float max_correspondence_distance,
                           const std::vector<Eigen::Vector<float, 6>>& F_f);
};

// Implementation

inline void LocalizabilityDetection::decomposeHessian(const Eigen::Matrix<float, 6, 6>& H, Eigen::Matrix3f& H_r,
                                                       Eigen::Matrix3f& H_t, Eigen::Vector3f& eigenvalues_r,
                                                       Eigen::Vector3f& eigenvalues_t, Eigen::Matrix3f& V_r,
                                                       Eigen::Matrix3f& V_t) {
    // Extract rotation and translation submatrices from Hessian
    // H = [H_r   H_rt]
    //     [H_rt^T H_t]
    H_r = H.block<3, 3>(0, 0);
    H_t = H.block<3, 3>(3, 3);

    // Ensure symmetry
    H_r = eigen_utils::ensure_symmetric<3>(H_r);
    H_t = eigen_utils::ensure_symmetric<3>(H_t);

    // Eigenvalue decomposition
    eigen_utils::symmetric_eigen_decomposition_3x3(H_r, eigenvalues_r, V_r);
    eigen_utils::symmetric_eigen_decomposition_3x3(H_t, eigenvalues_t, V_t);
}

inline std::vector<LocalizabilityContribution> LocalizabilityDetection::computeContributions(
    const PointCloudShared& source, const PointCloudShared& target, const knn::KNNResult& neighbors,
    const TransformMatrix& T, float max_correspondence_distance, const Eigen::Matrix3f& V_r,
    const Eigen::Matrix3f& V_t) {
    const size_t N = source.size();
    const float max_dist_sq = max_correspondence_distance * max_correspondence_distance;
    const auto T_sycl = eigen_utils::to_sycl_vec(T);

    std::vector<LocalizabilityContribution> contributions;
    contributions.reserve(N);

    const auto* source_pts = source.points_ptr();
    const auto* target_pts = target.points_ptr();
    const auto* target_normals = target.has_normal() ? target.normals_ptr() : nullptr;
    const auto* neighbor_indices = neighbors.indices->data();
    const auto* neighbor_distances = neighbors.distances->data();

    for (size_t i = 0; i < N; ++i) {
        // Skip invalid correspondences
        if (neighbor_distances[i] > max_dist_sq) {
            continue;
        }

        const int32_t target_idx = neighbor_indices[i];
        if (target_idx < 0) {
            continue;
        }

        const PointType& source_pt = source_pts[i];
        const Normal target_normal =
            target_normals ? target_normals[target_idx] : Normal(0.0f, 0.0f, 1.0f, 0.0f);

        // Compute Jacobian for this correspondence (point-to-plane)
        const Eigen::Matrix<float, 1, 6> J_i = kernel::compute_point_to_plane_jacobian(T_sycl, source_pt, target_normal);

        // Compute contribution with rotation scale unification
        contributions.push_back(kernel::compute_contribution(J_i, V_r, V_t, true));
    }

    return contributions;
}

inline void LocalizabilityDetection::filterContributions(const std::vector<LocalizabilityContribution>& contributions,
                                                          std::vector<Eigen::Vector<float, 6>>& F_f,
                                                          std::vector<Eigen::Vector<float, 6>>& F_u) {
    const float h_f = params_.noise_threshold;
    const float h_u = params_.high_contribution_threshold;

    F_f.clear();
    F_u.clear();
    F_f.reserve(contributions.size());
    F_u.reserve(contributions.size());

    for (const auto& contrib : contributions) {
        Eigen::Vector<float, 6> f_f = Eigen::Vector<float, 6>::Zero();
        Eigen::Vector<float, 6> f_u = Eigen::Vector<float, 6>::Zero();

        // Combine rotation and translation contributions
        Eigen::Vector<float, 6> F;
        F.head<3>() = contrib.F_r;
        F.tail<3>() = contrib.F_t;

        // Apply thresholds
        for (size_t j = 0; j < 6; ++j) {
            // Noise filtering: set to 0 if below threshold
            if (F(j) >= h_f) {
                f_f(j) = F(j);
            }

            // High contribution selection
            if (F(j) >= h_u) {
                f_u(j) = F(j);
            }
        }

        F_f.push_back(f_f);
        F_u.push_back(f_u);
    }
}

inline LocalizabilityAggregate LocalizabilityDetection::aggregateAndClassify(
    const std::vector<Eigen::Vector<float, 6>>& F_f, const std::vector<Eigen::Vector<float, 6>>& F_u) {
    LocalizabilityAggregate agg;

    // Sum contributions for each direction
    for (size_t i = 0; i < F_f.size(); ++i) {
        for (size_t j = 0; j < 6; ++j) {
            agg.L_f(j) += F_f[i](j);
            agg.L_u(j) += F_u[i](j);
        }
    }

    // Classify each direction using tri-value classification
    for (size_t j = 0; j < 6; ++j) {
        if (agg.L_f(j) >= params_.T1 || agg.L_u(j) >= params_.T2) {
            // Full localizability
            agg.categories[j] = LocalizabilityCategory::FULL;
        } else if (agg.L_f(j) >= params_.T3 && agg.L_u(j) >= params_.T4) {
            // Partial localizability
            agg.categories[j] = LocalizabilityCategory::PARTIAL;
        } else {
            // Non-localizable
            agg.categories[j] = LocalizabilityCategory::NONE;
        }
    }

    return agg;
}

inline void LocalizabilityDetection::buildConstraints(LocalizabilityResult& result, const PointCloudShared& source,
                                                        const PointCloudShared& target, const knn::KNNResult& neighbors,
                                                        const TransformMatrix& T, float max_correspondence_distance,
                                                        const std::vector<Eigen::Vector<float, 6>>& F_f) {
    const float max_dist_sq = max_correspondence_distance * max_correspondence_distance;
    const auto T_sycl = eigen_utils::to_sycl_vec(T);

    // Build 6x6 eigenvector matrix V
    result.V.setZero();
    result.V.block<3, 3>(0, 0) = result.V_r;
    result.V.block<3, 3>(3, 3) = result.V_t;

    for (size_t j = 0; j < 6; ++j) {
        const Eigen::Vector<float, 6> v_j = result.V.col(j);

        if (result.aggregate.categories[j] == LocalizabilityCategory::PARTIAL) {
            // Partial: compute soft constraint using correspondences with medium contributions
            SoftConstraint soft;
            soft.direction_index = j;
            soft.eigenvector = v_j;

            // Select correspondences that contribute to L_f for this direction
            // and perform small-scale optimization
            float sum_contribution = 0.0f;
            float weighted_delta = 0.0f;

            const auto* source_pts = source.points_ptr();
            const auto* target_pts = target.points_ptr();
            const auto* target_normals = target.has_normal() ? target.normals_ptr() : nullptr;
            const auto* neighbor_indices = neighbors.indices->data();
            const auto* neighbor_distances = neighbors.distances->data();

            size_t valid_idx = 0;
            for (size_t i = 0; i < source.size(); ++i) {
                if (neighbor_distances[i] > max_dist_sq) {
                    continue;
                }

                const int32_t target_idx = neighbor_indices[i];
                if (target_idx < 0) {
                    continue;
                }

                // Check if this correspondence has medium contribution for direction j
                if (valid_idx < F_f.size() && F_f[valid_idx](j) > 0.0f) {
                    const PointType& source_pt = source_pts[i];
                    const PointType& target_pt = target_pts[target_idx];

                    // Transform source point
                    PointType transformed_source;
                    transform::kernel::transform_point(source_pt, transformed_source, T_sycl);

                    // Compute residual projected onto direction
                    const Eigen::Vector3f residual = (target_pt - transformed_source).head<3>();

                    // Project residual onto eigenvector direction
                    // For rotation: use angular component, for translation: use linear component
                    float delta;
                    if (j < 3) {
                        // Rotation direction
                        const Eigen::Vector3f v_rot = v_j.head<3>();
                        delta = eigen_utils::dot<3>(residual, eigen_utils::cross(v_rot, source_pt.head<3>()));
                    } else {
                        // Translation direction
                        const Eigen::Vector3f v_trans = v_j.tail<3>();
                        delta = eigen_utils::dot<3>(residual, v_trans);
                    }

                    weighted_delta += F_f[valid_idx](j) * delta;
                    sum_contribution += F_f[valid_idx](j);
                }
                ++valid_idx;
            }

            if (sum_contribution > 0.0f) {
                soft.constraint_value = weighted_delta / sum_contribution;
            }

            result.soft_constraints.push_back(soft);
        } else if (result.aggregate.categories[j] == LocalizabilityCategory::NONE) {
            // None: add hard constraint
            result.hard_constraint.addConstraint(v_j);
        }
        // Full: no additional constraints needed
    }
}

inline LocalizabilityResult LocalizabilityDetection::detect(const Eigen::Matrix<float, 6, 6>& H,
                                                              const PointCloudShared& source,
                                                              const PointCloudShared& target,
                                                              const knn::KNNResult& neighbors, const TransformMatrix& T,
                                                              float max_correspondence_distance) {
    LocalizabilityResult result;

    // Step 1: Decompose Hessian and compute eigenvectors
    Eigen::Matrix3f H_r, H_t;
    decomposeHessian(H, H_r, H_t, result.eigenvalues_r, result.eigenvalues_t, result.V_r, result.V_t);

    // Step 2: Compute localizability contributions for all correspondences
    std::vector<LocalizabilityContribution> contributions =
        computeContributions(source, target, neighbors, T, max_correspondence_distance, result.V_r, result.V_t);

    // Step 3: Filter contributions (noise filtering and high contribution selection)
    std::vector<Eigen::Vector<float, 6>> F_f, F_u;
    filterContributions(contributions, F_f, F_u);

    // Step 4: Aggregate and classify each direction
    result.aggregate = aggregateAndClassify(F_f, F_u);

    // Step 5: Build constraints for partial and non-localizable directions
    buildConstraints(result, source, target, neighbors, T, max_correspondence_distance, F_f);

    if (params_.verbose) {
        std::cout << "[LocalizabilityDetection] L_f: " << result.aggregate.L_f.transpose() << std::endl;
        std::cout << "[LocalizabilityDetection] L_u: " << result.aggregate.L_u.transpose() << std::endl;
        std::cout << "[LocalizabilityDetection] Categories: ";
        for (size_t j = 0; j < 6; ++j) {
            switch (result.aggregate.categories[j]) {
                case LocalizabilityCategory::FULL:
                    std::cout << "Full ";
                    break;
                case LocalizabilityCategory::PARTIAL:
                    std::cout << "Partial ";
                    break;
                case LocalizabilityCategory::NONE:
                    std::cout << "None ";
                    break;
            }
        }
        std::cout << std::endl;
    }

    return result;
}

}  // namespace localizability

}  // namespace algorithms

}  // namespace sycl_points
