#pragma once

#include <Eigen/Dense>

#include <iostream>

#include "sycl_points/algorithms/registration/linearized_result.hpp"
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

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_rot(linearized_result.H.block<3, 3>(0, 0));
        if (solver_rot.info() != Eigen::Success) {
            return ret;
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_trans(linearized_result.H.block<3, 3>(3, 3));
        if (solver_trans.info() != Eigen::Success) {
            return ret;
        }

        if (this->params_.verbose) {
            const float inlier_f = static_cast<float>(inlier);
            std::cout << "[DegenerateRegularization] rotation eigenvalues/inlier: "
                      << (solver_rot.eigenvalues() / inlier_f).transpose() << std::endl;
            std::cout << "[DegenerateRegularization] translation eigenvalues/inlier: "
                      << (solver_trans.eigenvalues() / inlier_f).transpose() << std::endl;
        }

        if (this->params_.type == DegenerateRegularizationType::nl_reg) {
            const float rot_threshold = this->params_.rot_eigenvalue_threshold;
            const float trans_threshold = this->params_.trans_eigenvalue_threshold;
            const float lambda = this->params_.base_factor * inlier;

            Eigen::Matrix<float, 6, 6> H_penalty = Eigen::Matrix<float, 6, 6>::Zero();
            if (rot_threshold > 0.0f) {
                for (size_t i = 0; i < 3; ++i) {
                    const float val = solver_rot.eigenvalues()(i) / inlier;
                    if (val < rot_threshold) {
                        Eigen::Vector<float, 6> degenerate_vector = Eigen::Vector<float, 6>::Zero();
                        degenerate_vector.head<3>() = solver_rot.eigenvectors().col(i);
                        H_penalty += lambda * (degenerate_vector * degenerate_vector.transpose());
                    }
                }
            }
            if (trans_threshold > 0.0f) {
                for (size_t i = 0; i < 3; ++i) {
                    const float val = solver_trans.eigenvalues()(i) / inlier;
                    if (val < trans_threshold) {
                        Eigen::Vector<float, 6> degenerate_vector = Eigen::Vector<float, 6>::Zero();
                        degenerate_vector.tail<3>() = solver_trans.eigenvectors().col(i);
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
            const auto truncate_directions = [&](const auto& solver, const float threshold, const int offset) {
                if (threshold <= 0.0f) {
                    return;
                }
                for (Eigen::Index i = 0; i < 3; ++i) {
                    if (solver.eigenvalues()(i) / static_cast<float>(inlier) < threshold) {
                        Eigen::Vector<float, 6> direction = Eigen::Vector<float, 6>::Zero();
                        direction.segment<3>(offset) = solver.eigenvectors().col(i);
                        observable_projector -= direction * direction.transpose();
                    }
                }
            };
            truncate_directions(solver_rot, this->params_.rot_eigenvalue_threshold, 0);
            truncate_directions(solver_trans, this->params_.trans_eigenvalue_threshold, 3);

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
