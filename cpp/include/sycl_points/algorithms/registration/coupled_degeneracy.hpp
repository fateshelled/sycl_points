#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Block ordering of a 6x6 pose Hessian.
///
/// Two orderings coexist in this codebase, so the caller must state which one
/// it passes. The raw ICP Hessian produced by registration uses
/// `[delta_omega; delta_t]` (rotation first); the local pose Hessian assembled
/// in `apply_directional_icp_weighting` uses `[delta_t; delta_omega]`.
enum class PoseHessianOrder {
    /// @brief [rotation (0-2); translation (3-5)] — Registration 6x6 convention.
    rotation_first,
    /// @brief [translation (0-2); rotation (3-5)] — LIO directional weighting H_pose convention.
    translation_first
};

/// @brief Unit-balance matrix for the coupled pose analysis.
///
/// A 6x6 pose Hessian mixes metre and radian units, so a raw 6x6 eigen test
/// would compare incomparable scales and a block-diagonal test would discard the
/// translation-rotation coupling. The analysis instead balances the two blocks
/// with a representative length `L` (a rotation of `theta` corresponds to a
/// lever-arm displacement `L*theta`) and then diagonalises the full 6x6 matrix:
///
///   H_balanced = S^-1 (H / inlier) S^-1,   S = diag(L, L, L, 1, 1, 1)  (rotation first)
///
/// The eigenvalues of the balanced Hessian are unit-consistent, and its
/// eigenvectors are the coupled 6D motions (translation combined with rotation)
/// ordered from weakest to strongest. This is the approach already used by the
/// graph solver's LiDAR observability regularisation.
inline Eigen::Matrix<double, 6, 6> coupled_balance_matrix(double representative_length, PoseHessianOrder order) {
    Eigen::Matrix<double, 6, 6> balance = Eigen::Matrix<double, 6, 6>::Identity();
    const double length = std::max(representative_length, 1e-6);
    if (order == PoseHessianOrder::rotation_first) {
        balance.diagonal().head<3>().setConstant(length);
    } else {
        balance.diagonal().tail<3>().setConstant(length);
    }
    return balance;
}

/// @brief Unit-balanced eigendecomposition of a 6x6 pose Hessian.
struct CoupledEigenAnalysis {
    bool valid = false;
    Eigen::Matrix<double, 6, 6> balance = Eigen::Matrix<double, 6, 6>::Identity();          ///< S
    Eigen::Matrix<double, 6, 6> balance_inverse = Eigen::Matrix<double, 6, 6>::Identity();  ///< S^-1
    /// @brief Ascending eigenvalues of the balanced, inlier-normalised Hessian.
    Eigen::Vector<double, 6> normalized_eigenvalues = Eigen::Vector<double, 6>::Zero();
    /// @brief Orthonormal eigenvectors (columns) of the balanced Hessian.
    Eigen::Matrix<double, 6, 6> normalized_eigenvectors = Eigen::Matrix<double, 6, 6>::Identity();
};

/// @brief Diagonalise the unit-balanced, inlier-normalised pose Hessian.
///
/// The input is promoted to double and symmetrised before the addition of the
/// balance matrix. Negative eigenvalues from floating-point error are clamped to
/// zero for ranking.
///
/// @param H    6x6 pose Hessian (PSD; Gauss-Newton information matrix).
/// @param order Block ordering of @p H.
/// @param representative_length Representative length [m] balancing rotation and translation.
/// @param inlier Inlier count used to normalise the Hessian.
inline CoupledEigenAnalysis compute_coupled_eigen_analysis(const Eigen::Matrix<float, 6, 6>& H, PoseHessianOrder order,
                                                           double representative_length, double inlier) {
    CoupledEigenAnalysis result;
    if (!(inlier > 0.0)) {
        return result;
    }
    const Eigen::Matrix<double, 6, 6> Hd = H.cast<double>();
    if (!Hd.allFinite()) {
        return result;
    }

    const Eigen::Matrix<double, 6, 6> balance = coupled_balance_matrix(representative_length, order);
    const Eigen::Matrix<double, 6, 6> balance_inverse = balance.inverse();
    Eigen::Matrix<double, 6, 6> balanced =
        balance_inverse * (0.5 * (Hd + Hd.transpose()) / inlier) * balance_inverse;
    balanced = 0.5 * (balanced + balanced.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(balanced);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
        !solver.eigenvectors().allFinite()) {
        return result;
    }

    result.balance = balance;
    result.balance_inverse = balance_inverse;
    result.normalized_eigenvalues = solver.eigenvalues().cwiseMax(0.0);
    result.normalized_eigenvectors = solver.eigenvectors();
    result.valid = true;
    return result;
}

/// @brief Modified Gram-Schmidt over 6D pose directions.
///
/// Returns an orthonormal basis of the span of @p directions; dependent
/// directions (norm below @p tolerance after projection) are dropped so a
/// projector built from the result has the true weak-subspace rank.
inline std::vector<Eigen::Matrix<double, 6, 1>> orthonormalize_directions(
    std::vector<Eigen::Matrix<double, 6, 1>> directions, double tolerance = 1e-6) {
    std::vector<Eigen::Matrix<double, 6, 1>> basis;
    basis.reserve(directions.size());
    for (auto& direction : directions) {
        for (const auto& b : basis) {
            direction -= b.dot(direction) * b;
        }
        const double norm = direction.norm();
        if (norm > tolerance) {
            basis.push_back(direction / norm);
        }
    }
    return basis;
}

/// @brief A normalized pose direction with an associated information scale.
struct ScaledDirection {
    Eigen::Matrix<double, 6, 1> direction = Eigen::Matrix<double, 6, 1>::Zero();
    double scale = 1.0;
};

/// @brief Gram-Schmidt that carries a per-direction scale.
///
/// Used by the directional ICP weighting, where each weak coupled direction has
/// its own attenuation factor. A candidate that is dependent on an earlier basis
/// vector is dropped and its scale is merged (minimum) into the basis vector it
/// is closest to, so a strongly attenuated physical mode is not lost when the
/// second representation of the same mode is deduplicated.
inline std::vector<ScaledDirection> orthonormalize_scaled_directions(
    std::vector<ScaledDirection> directions, double tolerance = 1e-6) {
    std::vector<ScaledDirection> basis;
    basis.reserve(directions.size());
    for (auto candidate : directions) {
        double scale = candidate.scale;
        int dominant = -1;
        double dominant_abs = 0.0;
        for (int b = 0; b < static_cast<int>(basis.size()); ++b) {
            const double projection = basis[b].direction.dot(candidate.direction);
            candidate.direction -= projection * basis[b].direction;
            if (std::abs(projection) > dominant_abs) {
                dominant_abs = std::abs(projection);
                dominant = b;
            }
        }
        const double norm = candidate.direction.norm();
        if (norm > tolerance) {
            candidate.direction /= norm;
            candidate.scale = scale;
            basis.push_back(candidate);
        } else if (dominant >= 0) {
            // Dependent on an existing basis vector: merge the scale rather than
            // discarding it, keeping the more conservative (smaller) value.
            basis[dominant].scale = std::min(basis[dominant].scale, scale);
        }
    }
    return basis;
}

/// @brief Map selected balanced eigenvectors to orthonormal original-space directions.
///
/// The balanced eigenvector `v_k` corresponds to the original-space motion
/// `S v_k`. Those motions are generally not orthogonal, so they are
/// orthonormalised; the resulting basis spans the same weak subspace, which is
/// what a projector or a directional filter needs.
inline std::vector<Eigen::Matrix<double, 6, 1>> coupled_weak_directions(
    const CoupledEigenAnalysis& analysis, const std::vector<int>& weak_indices) {
    std::vector<Eigen::Matrix<double, 6, 1>> directions;
    directions.reserve(weak_indices.size());
    for (const int index : weak_indices) {
        directions.push_back(analysis.balance * analysis.normalized_eigenvectors.col(index));
    }
    return orthonormalize_directions(std::move(directions));
}

/// @brief Map selected balanced eigenvectors to scaled original-space directions.
inline std::vector<ScaledDirection> coupled_weak_scaled_directions(
    const CoupledEigenAnalysis& analysis, const std::vector<int>& weak_indices,
    const std::vector<double>& scales) {
    std::vector<ScaledDirection> directions;
    directions.reserve(weak_indices.size());
    for (size_t i = 0; i < weak_indices.size(); ++i) {
        ScaledDirection direction;
        direction.direction = analysis.balance * analysis.normalized_eigenvectors.col(weak_indices[i]);
        direction.scale = i < scales.size() ? scales[i] : 1.0;
        directions.push_back(direction);
    }
    return orthonormalize_scaled_directions(std::move(directions));
}

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
