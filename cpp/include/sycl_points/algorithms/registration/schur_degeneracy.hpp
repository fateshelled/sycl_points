#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

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

/// @brief Schur-complement (marginal) information of a 6x6 pose Hessian.
///
/// The independent analysis of the translation and rotation 3x3 diagonal blocks
/// discards the cross terms, so a motion that couples the two (for example a
/// cylinder-axis translation combined with a rotation about the same axis) can
/// appear well constrained in both blocks while the full 6x6 pose space is still
/// weakly constrained. Marginalizing one block given the other exposes that:
///
///   S_t = H_tt - H_tr * pinv(H_rr) * H_rt   (translation info after rotation compensates)
///   S_r = H_rr - H_rt * pinv(H_tt) * H_tr   (rotation info after translation compensates)
///
/// Each Schur matrix keeps its own physical unit (translation information vs
/// rotation information), so the existing per-block thresholds remain
/// meaningful — unlike a combined 6x6 eigenvalue test which would mix meters and
/// radians.
struct SchurDirections {
    bool valid = false;
    Eigen::Matrix3d S_translation = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d S_rotation = Eigen::Matrix3d::Zero();
    /// @brief Ascending eigenvalues of S_translation (translation information units).
    Eigen::Vector3d trans_eigenvalues = Eigen::Vector3d::Zero();
    Eigen::Matrix3d trans_eigenvectors = Eigen::Matrix3d::Identity();
    /// @brief Ascending eigenvalues of S_rotation (rotation information units).
    Eigen::Vector3d rot_eigenvalues = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rot_eigenvectors = Eigen::Matrix3d::Identity();
    /// @brief Numerical rank of the diagonal block that was marginalized out.
    int translation_block_rank = 0;
    int rotation_block_rank = 0;
};

/// @brief Rank-aware pseudo-inverse of a symmetric 3x3 block.
///
/// Eigenvalues below `max(absolute_cutoff, relative_cutoff * lambda_max)` are
/// treated as the null space and contribute nothing, so a singular block (the
/// degenerate case of interest) yields a minimum-norm inverse instead of a NaN.
/// @param rank_out Optional participation count of the non-truncated eigenvalues.
inline Eigen::Matrix3d schur_pseudo_inverse(const Eigen::Matrix3d& block, double relative_cutoff,
                                            double absolute_cutoff, int* rank_out = nullptr) {
    const Eigen::Matrix3d symmetric = 0.5 * (block + block.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(symmetric);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
        !solver.eigenvectors().allFinite()) {
        if (rank_out) *rank_out = 0;
        return Eigen::Matrix3d::Zero();
    }

    const double largest = std::max(0.0, solver.eigenvalues().maxCoeff());
    const double cutoff = std::max(std::max(0.0, absolute_cutoff), std::max(0.0, relative_cutoff) * largest);
    Eigen::Vector3d inverse = Eigen::Vector3d::Zero();
    int rank = 0;
    for (int i = 0; i < 3; ++i) {
        if (solver.eigenvalues()(i) > cutoff) {
            inverse(i) = 1.0 / solver.eigenvalues()(i);
            ++rank;
        }
    }
    if (rank_out) *rank_out = rank;
    return solver.eigenvectors() * inverse.asDiagonal() * solver.eigenvectors().transpose();
}

/// @brief Compute the translation/rotation Schur complements and their eigendecompositions.
///
/// The input is promoted to double before any subtraction: the Schur complement
/// is a difference of large, nearly equal terms and loses precision quickly in
/// float. The computation is host-only and O(1), so the extra precision is free.
///
/// @param H    6x6 pose Hessian (PSD; Gauss-Newton information matrix).
/// @param order Block ordering of @p H.
/// @param relative_cutoff Relative eigenvalue cutoff for the block pseudo-inverse.
/// @param absolute_cutoff Absolute eigenvalue cutoff for the block pseudo-inverse.
/// @return Schur directions; `valid` is false when the decomposition failed or produced non-finite values.
inline SchurDirections compute_schur_directions(const Eigen::Matrix<float, 6, 6>& H, PoseHessianOrder order,
                                                double relative_cutoff = 1e-6, double absolute_cutoff = 1e-9) {
    SchurDirections result;

    const Eigen::Matrix<double, 6, 6> Hd = H.cast<double>();
    if (!Hd.allFinite()) {
        return result;
    }
    const Eigen::Matrix<double, 6, 6> symmetric = 0.5 * (Hd + Hd.transpose());

    Eigen::Matrix3d H_tt;
    Eigen::Matrix3d H_rr;
    Eigen::Matrix3d H_tr;  // d2 / (d_t d_theta)
    Eigen::Matrix3d H_rt;  // d2 / (d_theta d_t)
    if (order == PoseHessianOrder::translation_first) {
        H_tt = symmetric.block<3, 3>(0, 0);
        H_rr = symmetric.block<3, 3>(3, 3);
        H_tr = symmetric.block<3, 3>(0, 3);
        H_rt = symmetric.block<3, 3>(3, 0);
    } else {
        H_rr = symmetric.block<3, 3>(0, 0);
        H_tt = symmetric.block<3, 3>(3, 3);
        H_rt = symmetric.block<3, 3>(0, 3);
        H_tr = symmetric.block<3, 3>(3, 0);
    }

    const Eigen::Matrix3d pinv_rr =
        schur_pseudo_inverse(H_rr, relative_cutoff, absolute_cutoff, &result.rotation_block_rank);
    const Eigen::Matrix3d pinv_tt =
        schur_pseudo_inverse(H_tt, relative_cutoff, absolute_cutoff, &result.translation_block_rank);

    Eigen::Matrix3d S_t = H_tt - H_tr * pinv_rr * H_rt;
    Eigen::Matrix3d S_r = H_rr - H_rt * pinv_tt * H_tr;
    S_t = 0.5 * (S_t + S_t.transpose());
    S_r = 0.5 * (S_r + S_r.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> trans_eig(S_t);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> rot_eig(S_r);
    if (trans_eig.info() != Eigen::Success || rot_eig.info() != Eigen::Success ||
        !trans_eig.eigenvalues().allFinite() || !rot_eig.eigenvalues().allFinite() ||
        !trans_eig.eigenvectors().allFinite() || !rot_eig.eigenvectors().allFinite()) {
        return result;
    }

    result.S_translation = S_t;
    result.S_rotation = S_r;
    // Floating-point cancellation can leave tiny negative eigenvalues; clamp for ranking.
    result.trans_eigenvalues = trans_eig.eigenvalues().cwiseMax(0.0);
    result.trans_eigenvectors = trans_eig.eigenvectors();
    result.rot_eigenvalues = rot_eig.eigenvalues().cwiseMax(0.0);
    result.rot_eigenvectors = rot_eig.eigenvectors();
    result.valid = true;
    return result;
}

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
