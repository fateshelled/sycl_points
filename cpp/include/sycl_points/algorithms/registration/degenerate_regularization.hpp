#pragma once

#include <Eigen/Dense>

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
    nl_reg
};

DegenerateRegularizationType DegenerateRegularizationType_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(str.begin(), str.end(), upper.begin(), [](u_char c) { return std::toupper(c); });

    if (upper.compare("NONE") == 0) {
        return DegenerateRegularizationType::none;
    } else if (upper.compare("NL-REG") == 0 || upper.compare("NL_REG") == 0) {
        return DegenerateRegularizationType::nl_reg;
    }
    std::string error_str = "[DegenerateRegularizationType_from_string] Invalid DegenerateRegularizationType str [";
    error_str += str;
    error_str += "]";
    throw std::runtime_error(error_str);
}

struct DegenerateRegularizationParams {
    DegenerateRegularizationType type = DegenerateRegularizationType::none;
    float rot_eig_threshold = 10.0f;
    float trans_eig_threshold = 1.0f;
    float base_factor = 1.0f;
};

class DegenerateRegularization {
public:
    void set_params(const DegenerateRegularizationParams& params) { this->params_ = params; }

    bool regularize(LinearizedResult& linearized_result, const Eigen::Isometry3f& current_pose,
                    const Eigen::Isometry3f& initial_guess) {
        return this->regularize_impl(linearized_result, current_pose, initial_guess);
    }

private:
    DegenerateRegularizationParams params_;

    bool regularize_impl(LinearizedResult& linearized_result, const Eigen::Isometry3f& current_pose,
                         const Eigen::Isometry3f& initial_guess) const {
        const auto inlier = linearized_result.inlier;
        if (inlier == 0) {
            return false;
        }

        if (this->params_.type == DegenerateRegularizationType::none) {
            return true;
        } else if (this->params_.type == DegenerateRegularizationType::nl_reg) {
            const float rot_threshold = this->params_.rot_eig_threshold;
            const float trans_threshold = this->params_.trans_eig_threshold;
            const float lambda = this->params_.base_factor * inlier;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_rot(linearized_result.H.block<3, 3>(0, 0));
            if (solver_rot.info() != Eigen::Success) {
                return false;
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_trans(linearized_result.H.block<3, 3>(3, 3));
            if (solver_trans.info() != Eigen::Success) {
                return false;
            }

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

            linearized_result.H += H_penalty;
            linearized_result.b += H_penalty * delta_twist;
            return true;
        }
        return false;
    }
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
