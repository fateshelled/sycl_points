#pragma once

#include <Eigen/Dense>

#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

struct MapPriorParams {
    bool enabled = false;
    /// @brief Process noise added to each rotation diagonal of H_raw^{-1} before inverting.
    ///        Larger value = weaker (more permissive) prior on rotation.
    float rot_process_noise = 0.01f;
    /// @brief Process noise added to each translation diagonal of H_raw^{-1} before inverting.
    ///        Larger value = weaker (more permissive) prior on translation.
    float trans_process_noise = 0.01f;
};

/// @brief MAP estimation prior using the previous frame's Hessian as the information matrix.
///
/// Adds a Gaussian prior N(T_pred, Omega_prior^{-1}) to the GICP normal equations each
/// iteration.  The information matrix is computed once per frame as:
///
///   Omega_prior = (H_raw_prev^{-1} + Q)^{-1}
///
/// where Q is a diagonal process-noise matrix that prevents the prior from being too
/// tight in directions that were well-constrained in the previous frame.
///
/// The resulting updates to the normal equations (H * delta = -b, solve(-b) convention):
///
///   H_total = H_gicp + Omega_prior
///   b_total = b_gicp + Omega_prior * e_prior
///
/// where e_prior = Log(T_pred^{-1} * T_est) is the deviation of the current estimate
/// from the predicted pose on the SE(3) manifold.
///
/// In degenerate directions H_raw_prev is small, so Omega_prior is also small and
/// nl_reg's Tikhonov penalty dominates — the two mechanisms are complementary.
class MapPrior {
public:
    void set_params(const MapPriorParams& params) { params_ = params; }

    /// @brief Precompute Omega_prior for the upcoming align() call.
    ///        Call this once per frame, after motion prediction and before align().
    /// @param H_raw_prev  Unregularized Hessian (RegistrationResult::H_raw) from the previous frame.
    /// @param T_pred      Predicted pose used as the initial guess for the current frame.
    void update(const Eigen::Matrix<float, 6, 6>& H_raw_prev, const Eigen::Isometry3f& T_pred) {
        has_prior_ = false;
        if (!params_.enabled) return;

        // Q = diag(rot_noise * I_3, trans_noise * I_3)
        Eigen::Matrix<float, 6, 6> Q = Eigen::Matrix<float, 6, 6>::Zero();
        Q.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * params_.rot_process_noise;
        Q.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * params_.trans_process_noise;

        // Sigma_prior = H_raw_prev^{-1} + Q
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt_h(H_raw_prev);
        if (ldlt_h.info() != Eigen::Success || ldlt_h.vectorD().minCoeff() <= 0.0f) return;
        const Eigen::Matrix<float, 6, 6> Sigma_prior =
            ldlt_h.solve(Eigen::Matrix<float, 6, 6>::Identity()) + Q;

        // Omega_prior = Sigma_prior^{-1}
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt_s(Sigma_prior);
        if (ldlt_s.info() != Eigen::Success || ldlt_s.vectorD().minCoeff() <= 0.0f) return;
        Omega_prior_ = ldlt_s.solve(Eigen::Matrix<float, 6, 6>::Identity());

        T_pred_ = T_pred;
        has_prior_ = true;
    }

    /// @brief Apply the MAP prior to the normal equations for one optimizer iteration.
    /// @param in    Linearized result after GICP (and optional degenerate regularization).
    /// @param T_est Current pose estimate at this iteration.
    /// @return Modified LinearizedResult with prior terms added.
    LinearizedResult apply(const LinearizedResult& in, const Eigen::Isometry3f& T_est) const {
        if (!is_active()) return in;

        // e_prior = Log(T_pred^{-1} * T_est): deviation of the current estimate from the prediction
        const Eigen::Vector<float, 6> e_prior = eigen_utils::lie::se3_log(T_pred_.inverse() * T_est);

        LinearizedResult ret = in;
        ret.H += Omega_prior_;
        ret.b += Omega_prior_ * e_prior;
        return ret;
    }

    bool is_active() const { return params_.enabled && has_prior_; }

private:
    MapPriorParams params_;
    bool has_prior_ = false;
    Eigen::Matrix<float, 6, 6> Omega_prior_ = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Isometry3f T_pred_ = Eigen::Isometry3f::Identity();
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
