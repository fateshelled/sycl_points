#pragma once

#include <Eigen/Dense>
#include <memory>

#include <sycl_points/algorithms/localizability_detection.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace localizability {

/// @brief Parameters for constrained optimization with localizability
struct ConstrainedOptimizerParams {
    /// Weight for soft constraint penalty (lambda_s in paper)
    float soft_constraint_weight = 1.0f;

    /// Damping factor for optimization
    float damping = 1e-6f;

    /// Enable debug output
    bool verbose = false;
};

/// @brief Constrained optimizer using localizability detection results
/// @details Implements the optimization module of LP-ICP framework
/// that applies soft and hard constraints based on localizability analysis
class ConstrainedOptimizer {
public:
    using Ptr = std::shared_ptr<ConstrainedOptimizer>;

    /// @brief Constructor
    /// @param params Optimizer parameters
    ConstrainedOptimizer(const ConstrainedOptimizerParams& params = ConstrainedOptimizerParams()) : params_(params) {}

    /// @brief Solve constrained optimization problem
    /// @param H Original Hessian matrix (6x6)
    /// @param b Original gradient vector (6x1)
    /// @param localizability Localizability detection result
    /// @return Optimal delta for pose update
    Eigen::Vector<float, 6> solve(const Eigen::Matrix<float, 6, 6>& H, const Eigen::Vector<float, 6>& b,
                                   const LocalizabilityResult& localizability);

    /// @brief Get parameters
    const ConstrainedOptimizerParams& getParams() const { return params_; }

    /// @brief Set parameters
    void setParams(const ConstrainedOptimizerParams& params) { params_ = params; }

private:
    ConstrainedOptimizerParams params_;

    /// @brief Apply soft constraints to Hessian and gradient
    /// @param H Hessian matrix (modified in-place)
    /// @param b Gradient vector (modified in-place)
    /// @param soft_constraints Soft constraints from localizability detection
    void applySoftConstraints(Eigen::Matrix<float, 6, 6>& H, Eigen::Vector<float, 6>& b,
                               const std::vector<SoftConstraint>& soft_constraints);

    /// @brief Solve with hard constraints using null space projection
    /// @param H Hessian matrix
    /// @param b Gradient vector
    /// @param hard_constraint Hard constraint matrix D
    /// @return Optimal delta satisfying D * delta = 0
    Eigen::Vector<float, 6> solveWithHardConstraints(const Eigen::Matrix<float, 6, 6>& H,
                                                       const Eigen::Vector<float, 6>& b,
                                                       const HardConstraint& hard_constraint);
};

// Implementation

inline Eigen::Vector<float, 6> ConstrainedOptimizer::solve(const Eigen::Matrix<float, 6, 6>& H,
                                                             const Eigen::Vector<float, 6>& b,
                                                             const LocalizabilityResult& localizability) {
    // Copy matrices for modification
    Eigen::Matrix<float, 6, 6> H_mod = H;
    Eigen::Vector<float, 6> b_mod = b;

    // Apply soft constraints for partially localizable directions
    if (!localizability.soft_constraints.empty()) {
        applySoftConstraints(H_mod, b_mod, localizability.soft_constraints);
    }

    // Apply hard constraints for non-localizable directions
    if (localizability.hard_constraint.num_constraints > 0) {
        return solveWithHardConstraints(H_mod, b_mod, localizability.hard_constraint);
    }

    // No hard constraints: standard solve with damping
    const Eigen::Matrix<float, 6, 6> H_damped =
        eigen_utils::add<6, 6>(H_mod, eigen_utils::multiply<6, 6>(Eigen::Matrix<float, 6, 6>::Identity(), params_.damping));

    // Solve using Cholesky decomposition
    Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt(H_damped);
    if (ldlt.info() != Eigen::Success) {
        // Fallback to more robust SVD if Cholesky fails
        Eigen::JacobiSVD<Eigen::Matrix<float, 6, 6>> svd(H_damped, Eigen::ComputeFullU | Eigen::ComputeFullV);
        return svd.solve(-b_mod);
    }

    return ldlt.solve(-b_mod);
}

inline void ConstrainedOptimizer::applySoftConstraints(Eigen::Matrix<float, 6, 6>& H, Eigen::Vector<float, 6>& b,
                                                         const std::vector<SoftConstraint>& soft_constraints) {
    // Soft constraint term: lambda_s * sum_j (v_j^T * delta_x - delta_x'_j)^2
    // This adds to Hessian: lambda_s * v_j * v_j^T
    // This adds to gradient: -lambda_s * delta_x'_j * v_j

    const float lambda_s = params_.soft_constraint_weight;

    for (const auto& constraint : soft_constraints) {
        const Eigen::Vector<float, 6>& v = constraint.eigenvector;
        const float delta_x_prime = constraint.constraint_value;

        // Add to Hessian: lambda_s * v * v^T
        const Eigen::Matrix<float, 6, 6> vvT = eigen_utils::outer<6>(v, v);
        eigen_utils::add_inplace<6, 6>(H, eigen_utils::multiply<6, 6>(vvT, lambda_s));

        // Add to gradient: -lambda_s * delta_x'_j * v
        eigen_utils::add_inplace<6, 1>(b, eigen_utils::multiply<6>(v, -lambda_s * delta_x_prime));
    }

    if (params_.verbose) {
        std::cout << "[ConstrainedOptimizer] Applied " << soft_constraints.size() << " soft constraints" << std::endl;
    }
}

inline Eigen::Vector<float, 6> ConstrainedOptimizer::solveWithHardConstraints(const Eigen::Matrix<float, 6, 6>& H,
                                                                                 const Eigen::Vector<float, 6>& b,
                                                                                 const HardConstraint& hard_constraint) {
    // Hard constraint: D * delta_x = 0
    // Solve in the null space of D using null space projection

    const size_t m = hard_constraint.num_constraints;
    if (m == 0) {
        // No hard constraints
        const Eigen::Matrix<float, 6, 6> H_damped =
            eigen_utils::add<6, 6>(H, eigen_utils::multiply<6, 6>(Eigen::Matrix<float, 6, 6>::Identity(), params_.damping));
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt(H_damped);
        return ldlt.solve(-b);
    }

    if (m >= 6) {
        // All directions are constrained - return zero
        if (params_.verbose) {
            std::cout << "[ConstrainedOptimizer] All 6 DoF constrained, returning zero" << std::endl;
        }
        return Eigen::Vector<float, 6>::Zero();
    }

    // Get the constraint matrix D (m x 6)
    Eigen::MatrixXf D = hard_constraint.D.topRows(m);

    // Compute null space of D using SVD
    // D = U * S * V^T
    // Null space of D is the columns of V corresponding to zero singular values
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(D, Eigen::ComputeFullV);
    const Eigen::MatrixXf& V = svd.matrixV();

    // Null space basis: columns of V from m to 5 (6-m columns)
    const size_t null_dim = 6 - m;
    Eigen::Matrix<float, 6, Eigen::Dynamic> N = V.rightCols(null_dim);

    // Project optimization into null space
    // H_reduced = N^T * H * N
    // b_reduced = N^T * b
    Eigen::MatrixXf H_reduced = N.transpose() * H * N;
    Eigen::VectorXf b_reduced = N.transpose() * b;

    // Add damping to reduced system
    H_reduced += params_.damping * Eigen::MatrixXf::Identity(null_dim, null_dim);

    // Solve reduced system
    Eigen::LDLT<Eigen::MatrixXf> ldlt(H_reduced);
    Eigen::VectorXf delta_reduced;
    if (ldlt.info() != Eigen::Success) {
        // Fallback to SVD
        Eigen::JacobiSVD<Eigen::MatrixXf> svd_solve(H_reduced, Eigen::ComputeFullU | Eigen::ComputeFullV);
        delta_reduced = svd_solve.solve(-b_reduced);
    } else {
        delta_reduced = ldlt.solve(-b_reduced);
    }

    // Map back to full space: delta = N * delta_reduced
    Eigen::Vector<float, 6> delta = N * delta_reduced;

    if (params_.verbose) {
        std::cout << "[ConstrainedOptimizer] Solved with " << m << " hard constraints" << std::endl;
        std::cout << "[ConstrainedOptimizer] Null space dimension: " << null_dim << std::endl;
    }

    return delta;
}

}  // namespace localizability

}  // namespace algorithms

}  // namespace sycl_points
