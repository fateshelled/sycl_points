#pragma once

#include <Eigen/Dense>

#include <iostream>
#include <utility>
#include <vector>

#include "sycl_points/algorithms/registration/coupled_degeneracy.hpp"
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
    /// @brief Analyse the coupled translation/rotation modes with a unit-balanced
    ///        full 6x6 eigendecomposition instead of the independent 3x3 diagonal
    ///        blocks. The independent analysis discards the cross terms, so a
    ///        motion that couples translation and rotation (e.g. translation along,
    ///        and rotation about, a cylinder axis) can appear well constrained in
    ///        both blocks while the full pose space is weakly constrained. The
    ///        coupled analysis catches those modes; the balanced eigenvector is
    ///        used directly, so no lift step is needed. The default keeps the
    ///        existing block-diagonal behaviour.
    bool use_coupled_degeneracy = false;
    /// @brief Weak-mode threshold on the balanced, inlier-normalised eigenvalues
    ///        used when `use_coupled_degeneracy` is enabled (replaces the
    ///        per-block thresholds for that path).
    float coupled_eigenvalue_threshold = 1.0f;
    /// @brief Representative length [m] balancing the rotation and translation
    ///        blocks of the coupled analysis (a rotation of theta corresponds to a
    ///        displacement of `representative_length * theta`). `<= 0` estimates it
    ///        per frame from the Hessian trace ratio (~= weighted RMS point range),
    ///        which adapts to the scene distance; a positive value is used fixed.
    float coupled_representative_length = 0.0f;
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

        // Collect the weak pose directions as 6D vectors. The block-diagonal path
        // zero-pads the weak block eigenvectors. The coupled path takes the weak
        // unit-balanced 6x6 eigenvectors directly, which already combine
        // translation and rotation, so exactly the coupled weak subspace is
        // removed and observable coupled motions are preserved.
        std::vector<Eigen::Matrix<double, 6, 1>> weak_directions;

        if (this->params_.use_coupled_degeneracy) {
            // A non-positive or non-finite configured length means "estimate it
            // from the Hessian" (~= weighted RMS point range); a positive finite
            // value is used directly.
            double length = static_cast<double>(this->params_.coupled_representative_length);
            if (!std::isfinite(length) || length <= 0.0) {
                length = estimate_representative_length(linearized_result.H, PoseHessianOrder::rotation_first, 1.0);
            }
            const CoupledEigenAnalysis analysis = compute_coupled_eigen_analysis(
                linearized_result.H, PoseHessianOrder::rotation_first, length, static_cast<double>(inlier));
            if (!analysis.valid) {
                return ret;
            }
            if (this->params_.verbose) {
                std::cout << "[DegenerateRegularization] coupled eigenvalues/inlier: "
                          << analysis.normalized_eigenvalues.transpose() << std::endl;
                std::cout << "[DegenerateRegularization] coupled representative_length: " << length << std::endl;
            }
            std::vector<int> weak;
            for (int k = 0; k < 6; ++k) {
                if (analysis.normalized_eigenvalues(k) < this->params_.coupled_eigenvalue_threshold) {
                    weak.push_back(k);
                }
            }
            weak_directions = coupled_weak_directions(analysis, weak);
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
            if (this->params_.rot_eigenvalue_threshold > 0.0f) {
                for (int i = 0; i < 3; ++i) {
                    if (solver_rot.eigenvalues()(i) / inlier_f < this->params_.rot_eigenvalue_threshold) {
                        Eigen::Matrix<double, 6, 1> direction = Eigen::Matrix<double, 6, 1>::Zero();
                        direction.head<3>() = solver_rot.eigenvectors().col(i).cast<double>();
                        weak_directions.push_back(direction);
                    }
                }
            }
            if (this->params_.trans_eigenvalue_threshold > 0.0f) {
                for (int i = 0; i < 3; ++i) {
                    if (solver_trans.eigenvalues()(i) / inlier_f < this->params_.trans_eigenvalue_threshold) {
                        Eigen::Matrix<double, 6, 1> direction = Eigen::Matrix<double, 6, 1>::Zero();
                        direction.tail<3>() = solver_trans.eigenvectors().col(i).cast<double>();
                        weak_directions.push_back(direction);
                    }
                }
            }
            weak_directions = orthonormalize_directions(std::move(weak_directions));
        }

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
