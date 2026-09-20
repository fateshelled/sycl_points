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

        const float inlier_f = static_cast<float>(inlier);
        const float rot_threshold = this->params_.rot_eigenvalue_threshold;
        const float trans_threshold = this->params_.trans_eigenvalue_threshold;

        // Collect the weak pose directions as 6D vectors. The block-diagonal path
        // zero-pads the weak block eigenvectors. The Schur path lifts each weak
        // Schur eigenvector to the coupled 6D twist that minimizes the local
        // quadratic model in the other block; the lifted set is orthonormalized
        // so the projector removes exactly the coupled weak subspace and keeps
        // observable coupled motions that zero-padding would have discarded.
        std::vector<Eigen::Matrix<double, 6, 1>> weak_directions;

        if (this->params_.use_schur_complement) {
            const SchurDirections schur =
                compute_schur_directions(linearized_result.H, PoseHessianOrder::rotation_first,
                                         this->params_.schur_relative_cutoff, this->params_.schur_absolute_cutoff);
            if (!schur.valid) {
                return ret;
            }
            if (this->params_.verbose) {
                const double inlier_d = static_cast<double>(inlier);
                std::cout << "[DegenerateRegularization] schur ranks: rotation=" << schur.rotation_block_rank
                          << ", translation=" << schur.translation_block_rank << std::endl;
                std::cout << "[DegenerateRegularization] schur rotation eigenvalues/inlier: "
                          << (schur.rot_eigenvalues / inlier_d).transpose() << std::endl;
                std::cout << "[DegenerateRegularization] schur translation eigenvalues/inlier: "
                          << (schur.trans_eigenvalues / inlier_d).transpose() << std::endl;
            }
            if (trans_threshold > 0.0f) {
                for (int i = 0; i < 3; ++i) {
                    if (schur.trans_eigenvalues(i) / static_cast<double>(inlier) < trans_threshold) {
                        weak_directions.push_back(lift_translation_direction(
                            schur, schur.trans_eigenvectors.col(i), PoseHessianOrder::rotation_first));
                    }
                }
            }
            if (rot_threshold > 0.0f) {
                for (int i = 0; i < 3; ++i) {
                    if (schur.rot_eigenvalues(i) / static_cast<double>(inlier) < rot_threshold) {
                        weak_directions.push_back(lift_rotation_direction(
                            schur, schur.rot_eigenvectors.col(i), PoseHessianOrder::rotation_first));
                    }
                }
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
            if (this->params_.verbose) {
                std::cout << "[DegenerateRegularization] rotation eigenvalues/inlier: "
                          << (solver_rot.eigenvalues() / inlier_f).transpose() << std::endl;
                std::cout << "[DegenerateRegularization] translation eigenvalues/inlier: "
                          << (solver_trans.eigenvalues() / inlier_f).transpose() << std::endl;
            }
            if (rot_threshold > 0.0f) {
                for (int i = 0; i < 3; ++i) {
                    if (solver_rot.eigenvalues()(i) / inlier_f < rot_threshold) {
                        Eigen::Matrix<double, 6, 1> direction = Eigen::Matrix<double, 6, 1>::Zero();
                        direction.head<3>() = solver_rot.eigenvectors().col(i).cast<double>();
                        weak_directions.push_back(direction);
                    }
                }
            }
            if (trans_threshold > 0.0f) {
                for (int i = 0; i < 3; ++i) {
                    if (solver_trans.eigenvalues()(i) / inlier_f < trans_threshold) {
                        Eigen::Matrix<double, 6, 1> direction = Eigen::Matrix<double, 6, 1>::Zero();
                        direction.tail<3>() = solver_trans.eigenvectors().col(i).cast<double>();
                        weak_directions.push_back(direction);
                    }
                }
            }
        }

        weak_directions = orthonormalize_directions(weak_directions);

        Eigen::Matrix<float, 6, 6> degenerate_projector = Eigen::Matrix<float, 6, 6>::Zero();
        for (const auto& direction : weak_directions) {
            const Eigen::Matrix<float, 6, 1> d = direction.cast<float>();
            degenerate_projector.noalias() += d * d.transpose();
        }
        const Eigen::Matrix<float, 6, 6> observable_projector =
            Eigen::Matrix<float, 6, 6>::Identity() - degenerate_projector;

        if (this->params_.type == DegenerateRegularizationType::nl_reg) {
            const Eigen::Matrix<float, 6, 6> H_penalty = (this->params_.base_factor * inlier_f) * degenerate_projector;
            const Eigen::Isometry3f delta_pose = initial_guess.inverse() * current_pose;
            const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(delta_pose);
            ret.H += H_penalty;
            ret.b += H_penalty * delta_twist;
            return ret;
        } else if (this->params_.type == DegenerateRegularizationType::tsvd ||
                   this->params_.type == DegenerateRegularizationType::l_reg ||
                   this->params_.type == DegenerateRegularizationType::solution_remap ||
                   this->params_.type == DegenerateRegularizationType::eq_constraint) {
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
