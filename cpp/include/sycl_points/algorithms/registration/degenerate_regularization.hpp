#pragma once

#include <Eigen/Dense>

#include <iostream>

#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/algorithms/registration/schur_degeneracy.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace registration {

enum class DegenerateRegularizationType {
    none = 0,
    /// @brief Informed, Constrained, Aligned: A Field Analysis on Degeneracy-aware Point Cloud Registration in the Wild
    /// @authors Turcan Tuna, Julian Nubert, Patrick Pfreundschuh, Cesar Cadena, Shehryar Khattak, Marco Hutter
    /// @cite https://arxiv.org/abs/2408.11809
    /// @date 2024
    /// @note Non linear optimization with Tikhonov regularization
    nl_reg,
    /// @note Truncated SVD. Degenerate components of the pose update are set to zero.
    tsvd,
    /// @note Linear least-squares with subspace Tikhonov regularization.
    l_reg,
    /// @note Project the unconstrained solution onto the observable subspace.
    solution_remap,
    /// @note Zero-update equality constraints along degenerate directions.
    eq_constraint
};

inline DegenerateRegularizationType DegenerateRegularizationType_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(str.begin(), str.end(), upper.begin(), [](u_char c) { return std::toupper(c); });

    if (upper.compare("NONE") == 0) {
        return DegenerateRegularizationType::none;
    } else if (upper.compare("NL-REG") == 0 || upper.compare("NL_REG") == 0) {
        return DegenerateRegularizationType::nl_reg;
    } else if (upper.compare("TSVD") == 0) {
        return DegenerateRegularizationType::tsvd;
    } else if (upper.compare("L-REG") == 0 || upper.compare("L_REG") == 0) {
        return DegenerateRegularizationType::l_reg;
    } else if (upper.compare("SOLUTION-REMAP") == 0 || upper.compare("SOLUTION_REMAP") == 0) {
        return DegenerateRegularizationType::solution_remap;
    } else if (upper.compare("EQ-CONSTRAINT") == 0 || upper.compare("EQ_CONSTRAINT") == 0) {
        return DegenerateRegularizationType::eq_constraint;
    }
    std::string error_str = "[DegenerateRegularizationType_from_string] Invalid DegenerateRegularizationType str [";
    error_str += str;
    error_str += "]";
    throw std::runtime_error(error_str);
}

struct DegenerateRegularizationParams {
    DegenerateRegularizationType type = DegenerateRegularizationType::none;
    bool verbose = false;
    float rot_eigenvalue_threshold = 10.0f;
    float trans_eigenvalue_threshold = 1.0f;
    float base_factor = 1.0f;
    float linear_factor = 440.0f;
    /// @brief Analyse the coupled translation/rotation Schur complements instead
    ///        of the independent 3x3 diagonal blocks. This catches coupled
    ///        degenerate motions (e.g. translation along, and rotation about, a
    ///        cylinder axis) that the block-diagonal analysis misses. The
    ///        default keeps the existing block-diagonal behaviour. Schur
    ///        eigenvalues are marginal and therefore no larger than the
    ///        corresponding block eigenvalues, so the thresholds may need
    ///        retuning when this is enabled.
    bool use_schur_complement = false;
    /// @brief Relative eigenvalue cutoff for the Schur block pseudo-inverse.
    double schur_relative_cutoff = 1e-6;
    /// @brief Absolute eigenvalue cutoff for the Schur block pseudo-inverse.
    double schur_absolute_cutoff = 1e-9;
};

class DegenerateRegularization {
public:
    void set_params(const DegenerateRegularizationParams& params) { this->params_ = params; }

    LinearizedResult regularize(const LinearizedResult& linearized_result, const Eigen::Isometry3f& current_pose,
                                const Eigen::Isometry3f& initial_guess) const {
        return this->regularize_impl(linearized_result, current_pose, initial_guess);
    }

private:
    DegenerateRegularizationParams params_;

    LinearizedResult regularize_impl(const LinearizedResult& linearized_result, const Eigen::Isometry3f& current_pose,
                                     const Eigen::Isometry3f& initial_guess) const {
        LinearizedResult ret = linearized_result;
        const auto inlier = linearized_result.inlier;
        if (inlier == 0) {
            return ret;
        }

        if (this->params_.type == DegenerateRegularizationType::none) {
            return ret;
        }

        Eigen::Vector3f rot_eigenvalues = Eigen::Vector3f::Zero();
        Eigen::Matrix3f rot_eigenvectors = Eigen::Matrix3f::Identity();
        Eigen::Vector3f trans_eigenvalues = Eigen::Vector3f::Zero();
        Eigen::Matrix3f trans_eigenvectors = Eigen::Matrix3f::Identity();

        if (this->params_.use_schur_complement) {
            // Coupled translation/rotation marginal information. Catches combined
            // degenerate motions (translation coupled with rotation) that the
            // independent 3x3 diagonal blocks cannot see.
            const SchurDirections schur =
                compute_schur_directions(linearized_result.H, PoseHessianOrder::rotation_first,
                                         this->params_.schur_relative_cutoff, this->params_.schur_absolute_cutoff);
            if (!schur.valid) {
                return ret;
            }
            rot_eigenvalues = schur.rot_eigenvalues.cast<float>();
            rot_eigenvectors = schur.rot_eigenvectors.cast<float>();
            trans_eigenvalues = schur.trans_eigenvalues.cast<float>();
            trans_eigenvectors = schur.trans_eigenvectors.cast<float>();

            if (this->params_.verbose) {
                std::cout << "[DegenerateRegularization] schur ranks: rotation=" << schur.rotation_block_rank
                          << ", translation=" << schur.translation_block_rank << std::endl;
            }
        } else {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_rot(linearized_result.H.block<3, 3>(0, 0));
            if (solver_rot.info() != Eigen::Success) {
                return ret;
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_trans(linearized_result.H.block<3, 3>(3, 3));
            if (solver_trans.info() != Eigen::Success) {
                return ret;
            }
            rot_eigenvalues = solver_rot.eigenvalues();
            rot_eigenvectors = solver_rot.eigenvectors();
            trans_eigenvalues = solver_trans.eigenvalues();
            trans_eigenvectors = solver_trans.eigenvectors();
        }

        if (this->params_.verbose) {
            const float inlier_f = static_cast<float>(inlier);
            std::cout << "[DegenerateRegularization] rotation eigenvalues/inlier: "
                      << (rot_eigenvalues / inlier_f).transpose() << std::endl;
            std::cout << "[DegenerateRegularization] translation eigenvalues/inlier: "
                      << (trans_eigenvalues / inlier_f).transpose() << std::endl;
        }

        if (this->params_.type == DegenerateRegularizationType::nl_reg) {
            const float rot_threshold = this->params_.rot_eigenvalue_threshold;
            const float trans_threshold = this->params_.trans_eigenvalue_threshold;
            const float lambda = this->params_.base_factor * inlier;

            Eigen::Matrix<float, 6, 6> H_penalty = Eigen::Matrix<float, 6, 6>::Zero();
            if (rot_threshold > 0.0f) {
                for (size_t i = 0; i < 3; ++i) {
                    const float val = rot_eigenvalues(i) / inlier;
                    if (val < rot_threshold) {
                        Eigen::Vector<float, 6> degenerate_vector = Eigen::Vector<float, 6>::Zero();
                        degenerate_vector.head<3>() = rot_eigenvectors.col(i);
                        H_penalty += lambda * (degenerate_vector * degenerate_vector.transpose());
                    }
                }
            }
            if (trans_threshold > 0.0f) {
                for (size_t i = 0; i < 3; ++i) {
                    const float val = trans_eigenvalues(i) / inlier;
                    if (val < trans_threshold) {
                        Eigen::Vector<float, 6> degenerate_vector = Eigen::Vector<float, 6>::Zero();
                        degenerate_vector.tail<3>() = trans_eigenvectors.col(i);
                        H_penalty += lambda * (degenerate_vector * degenerate_vector.transpose());
                    }
                }
            }
            const Eigen::Isometry3f delta_pose = initial_guess.inverse() * current_pose;
            const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(delta_pose);

            ret.H += H_penalty;
            ret.b += H_penalty * delta_twist;
            return ret;
        } else if (this->params_.type == DegenerateRegularizationType::tsvd ||
                   this->params_.type == DegenerateRegularizationType::l_reg ||
                   this->params_.type == DegenerateRegularizationType::solution_remap ||
                   this->params_.type == DegenerateRegularizationType::eq_constraint) {
            Eigen::Matrix<float, 6, 6> observable_projector = Eigen::Matrix<float, 6, 6>::Identity();
            const auto truncate_directions = [&](const Eigen::Vector3f& eigenvalues,
                                                 const Eigen::Matrix3f& eigenvectors, const float threshold,
                                                 const int offset) {
                if (threshold <= 0.0f) {
                    return;
                }
                for (Eigen::Index i = 0; i < 3; ++i) {
                    if (eigenvalues(i) / static_cast<float>(inlier) < threshold) {
                        Eigen::Vector<float, 6> direction = Eigen::Vector<float, 6>::Zero();
                        direction.segment<3>(offset) = eigenvectors.col(i);
                        observable_projector -= direction * direction.transpose();
                    }
                }
            };
            truncate_directions(rot_eigenvalues, rot_eigenvectors, this->params_.rot_eigenvalue_threshold, 0);
            truncate_directions(trans_eigenvalues, trans_eigenvectors, this->params_.trans_eigenvalue_threshold, 3);

            const Eigen::Matrix<float, 6, 6> degenerate_projector =
                Eigen::Matrix<float, 6, 6>::Identity() - observable_projector;
            if (this->params_.type == DegenerateRegularizationType::l_reg) {
                ret.H += this->params_.linear_factor * degenerate_projector;
            } else if (this->params_.type == DegenerateRegularizationType::solution_remap) {
                ret.solution_projector = observable_projector;
            } else {
                // The solvers consume normal equations rather than a precomputed pseudo-inverse. Replacing the
                // removed subspace with identity and zero gradient makes it non-singular without creating an update.
                // TSVD and zero-valued equality constraints both solve only in the observable subspace. The identity
                // formulation is algebraically equivalent to their truncated/reduced solve for these constraints.
                ret.H = observable_projector * linearized_result.H * observable_projector + degenerate_projector;
                ret.b = observable_projector * linearized_result.b;
            }
            return ret;
        }
        return ret;
    }
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
