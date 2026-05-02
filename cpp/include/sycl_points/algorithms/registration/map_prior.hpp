#pragma once

#include <Eigen/Dense>

#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

struct MapPriorParams {
    bool enabled = false;
    /// @brief Process noise for each rotation axis (diagonal of Q).
    ///        Larger value = weaker (more permissive) prior on rotation.
    float rot_process_noise = 0.01f;
    /// @brief Process noise for each translation axis (diagonal of Q).
    ///        Larger value = weaker (more permissive) prior on translation.
    float trans_process_noise = 0.01f;
};

/// @brief MAP estimation prior using the previous frame's Hessian as the information matrix.
///
/// Adds a Gaussian prior N(T_pred, Omega_prior^{-1}) to the GICP normal equations each
/// iteration.  The information matrix is computed once per frame via the Matrix Inversion
/// Lemma to avoid directly inverting H_raw_prev (which may be singular in degenerate cases):
///
///   R = Q^{-1}  (trivial: diagonal)
///   Omega_prior = (H^{-1} + Q)^{-1} = R - R(H + R)^{-1}R
///
/// Since R is positive-definite, (H + R) is always invertible even when H is singular,
/// making this formulation robust to degenerate environments.
///
/// H_raw_prev is expressed in the previous sensor frame.  Before computing Omega_prior,
/// it is rotated into the current sensor frame using the rotation-only Adjoint:
///
///   Ad = block_diag(R_rel, R_rel),  R_rel = R_prev^T * R_pred
///   H_curr = Ad^T * H_raw_prev * Ad
///
/// The resulting updates to the normal equations (H * delta = -b, solve(-b) convention):
///
///   H_total = H_gicp + Omega_prior
///   b_total = b_gicp + Omega_prior * e_prior
///   error_total = error_gicp + 0.5 * e_prior^T * Omega_prior * e_prior
///
/// where e_prior = Log(T_pred^{-1} * T_est) is the deviation of the current estimate
/// from the predicted pose on the SE(3) manifold.
///
/// In degenerate directions H_raw_prev is small, so Omega_prior is also small and
/// nl_reg's Tikhonov penalty dominates — the two mechanisms are complementary.
class MapPrior {
public:
    void set_params(const MapPriorParams& params) { params_ = params; }

    /// @brief Precompute Omega_prior and T_pred_inv for the upcoming align() call.
    ///        Call this once per frame, after motion prediction and before align().
    /// @param H_raw_prev  Unregularized Hessian (RegistrationResult::H_raw) from the previous frame.
    /// @param T_pred      Predicted pose used as the initial guess for the current frame.
    /// @param T_prev      Previous odometry pose (used to compute relative rotation for Adjoint).
    void update(const Eigen::Matrix<float, 6, 6>& H_raw_prev, const Eigen::Isometry3f& T_pred,
                const Eigen::Isometry3f& T_prev) {
        has_prior_ = false;
        if (!params_.enabled) return;

        // Rotate H_raw_prev into the current sensor frame using the rotation-only Adjoint.
        // Ad = block_diag(R_rel, R_rel), R_rel = R_prev^T * R_pred
        const Eigen::Matrix3f R_rel = T_prev.rotation().transpose() * T_pred.rotation();
        Eigen::Matrix<float, 6, 6> Ad = Eigen::Matrix<float, 6, 6>::Zero();
        Ad.block<3, 3>(0, 0) = R_rel;
        Ad.block<3, 3>(3, 3) = R_rel;
        const Eigen::Matrix<float, 6, 6> H_curr = Ad.transpose() * H_raw_prev * Ad;

        // R = Q^{-1}: diagonal, trivially computed from process-noise parameters
        const float inv_rot = 1.0f / std::max(params_.rot_process_noise, 1e-6f);
        const float inv_trans = 1.0f / std::max(params_.trans_process_noise, 1e-6f);
        Eigen::Matrix<float, 6, 6> R = Eigen::Matrix<float, 6, 6>::Zero();
        R.diagonal() << inv_rot, inv_rot, inv_rot, inv_trans, inv_trans, inv_trans;

        // Omega_prior = (H^{-1} + Q)^{-1} = R - R(H + R)^{-1}R
        // (H + R) is always PD since R is PD, so this is robust to singular H.
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt(H_curr + R);
        if (ldlt.info() != Eigen::Success) return;
        Omega_prior_ = R - R * ldlt.solve(R);

        // Precompute inverse once; reused every iteration inside apply() and prior_error()
        T_pred_inv_ = T_pred.inverse();
        has_prior_ = true;
    }

    /// @brief Apply the MAP prior to the normal equations for one optimizer iteration.
    ///        Adds Omega_prior to H and b, and adds the prior's scalar cost to error.
    /// @param in    Linearized result after GICP (and optional degenerate regularization).
    /// @param T_est Current pose estimate at this iteration.
    /// @return Modified LinearizedResult with prior terms added.
    LinearizedResult apply(const LinearizedResult& in, const Eigen::Isometry3f& T_est) const {
        if (!is_active()) return in;

        // e_prior = Log(T_pred^{-1} * T_est): deviation of the current estimate from the prediction
        const Eigen::Vector<float, 6> e_prior = eigen_utils::lie::se3_log(T_pred_inv_ * T_est);
        const Eigen::Vector<float, 6> Omega_e = Omega_prior_ * e_prior;

        LinearizedResult ret = in;
        ret.H += Omega_prior_;
        ret.b += Omega_e;
        ret.error += 0.5f * e_prior.dot(Omega_e);
        return ret;
    }

    /// @brief Compute the scalar prior cost at a given pose.
    ///        Used to augment compute_error() results for LM/Dogleg step acceptance.
    float prior_error(const Eigen::Isometry3f& T_est) const {
        if (!is_active()) return 0.0f;
        const Eigen::Vector<float, 6> e = eigen_utils::lie::se3_log(T_pred_inv_ * T_est);
        return 0.5f * e.dot(Omega_prior_ * e);
    }

    bool is_active() const { return params_.enabled && has_prior_; }

private:
    MapPriorParams params_;
    bool has_prior_ = false;
    Eigen::Matrix<float, 6, 6> Omega_prior_ = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Isometry3f T_pred_inv_ = Eigen::Isometry3f::Identity();
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
