#pragma once

#include <Eigen/Dense>
#include <cmath>

#include "sycl_points/algorithms/registration/linearized_result.hpp"
#include "sycl_points/algorithms/registration/result.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

struct MapPriorParams {
    bool enabled = false;
    /// @brief Sigma contribution at unit (1 rad) inter-frame rotation [rad].
    ///        Q_rot = rot_base_sigma^2 + rot_vel_sigma^2 * |delta_rot|   [rad^2]
    ///        Interpretation: at |delta_rot| = 1 rad the std-dev contribution is rot_vel_sigma.
    ///        At smaller motion, the contribution scales as rot_vel_sigma * sqrt(|delta_rot|).
    float rot_vel_sigma = 1.0f;
    /// @brief Sigma contribution at unit (1 m) inter-frame translation [m].
    ///        Q_trans = trans_base_sigma^2 + trans_vel_sigma^2 * |delta_trans|   [m^2]
    ///        Interpretation: at |delta_trans| = 1 m the std-dev contribution is trans_vel_sigma.
    ///        At smaller motion, the contribution scales as trans_vel_sigma * sqrt(|delta_trans|).
    float trans_vel_sigma = 1.0f;
    /// @brief Isotropic baseline rotation std-dev [rad].
    ///        Squared and added to Q_rot to model acceleration-induced prediction uncertainty,
    ///        which keeps the prior responsive to sudden motion regardless of current velocity.
    float rot_base_sigma = 3.16e-2f;  // sqrt(1e-3) rad ~= 1.81 deg
    /// @brief Isotropic baseline translation std-dev [m].
    ///        Squared and added to Q_trans to model acceleration-induced prediction uncertainty,
    ///        which keeps the prior responsive to sudden motion regardless of current velocity.
    float trans_base_sigma = 1e-2f;  // sqrt(1e-4) m = 1 cm
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
/// H_raw is treated as J^T * Sigma^{-1} * J under the assumption that the per-point
/// Mahalanobis residual r^T Sigma_i^{-1} r has unit variance.  Real data deviates from
/// this, so H_raw is calibrated by the reduced chi-squared statistic of the previous
/// registration before being used as an information matrix:
///
///   DOF   = 3 * N_inlier - 6      (3 = GICP residual dim per inlier, 6 = SE(3) params)
///   s^2   = error / DOF           (residual variance estimate; ~1.0 under perfect modelling)
///   H_cal = H_raw_prev / s^2      (Sigma_pose ~= s^2 * H_raw^{-1}  =>  Lambda = H_raw / s^2)
///
/// Large s^2 (under-fit residuals or sparse inliers) loosens the prior; small s^2 tightens
/// it.  No clamping is applied: callers should monitor degenerate inputs.
///
/// H_cal is expressed in the previous sensor frame.  Before computing Omega_prior, it is
/// rotated into the current sensor frame using the rotation-only Adjoint:
///
///   Ad       = block_diag(R_rel, R_rel),  R_rel = R_opt_prev^T * R_pred
///   H_curr   = Ad^T * H_cal * Ad
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
/// The process noise Q is computed adaptively from the predicted inter-frame motion:
///
///   Q_rot   = rot_base_sigma^2   + rot_vel_sigma^2   * |delta_rot|
///   Q_trans = trans_base_sigma^2 + trans_vel_sigma^2 * |delta_trans|
///
/// The velocity-proportional term loosens the prior during fast motion (CV-model uncertainty),
/// while the additive baseline (base_sigma^2) is an isotropic acceleration-noise term that
/// keeps the prior responsive to sudden motion even when current velocity is small or zero.
/// Linear (|delta|) rather than quadratic (delta^2) scaling is used to keep Q within a
/// practical dynamic range across the typical 0.01–1.0 m/frame motion regime — quadratic
/// scaling would make the prior nearly vanish at high speed.  Both base_sigma and vel_sigma
/// are parameterised as std-dev (units of [rad] / [m]); vel_sigma is the σ contribution at
/// unit motion (1 rad / 1 m).  Both appear squared in the variance formula.
/// In degenerate directions H_raw_prev is small, so Omega_prior is also small and
/// nl_reg's Tikhonov penalty dominates — the two mechanisms are complementary.
class MapPrior {
public:
    void set_params(const MapPriorParams& params) {
        this->params_ = params;
        this->has_prior_ = false;
    }

    /// @brief Precompute Omega_prior and T_pred_inv for the upcoming align() call.
    ///        Call this once per frame, after motion prediction and before align().
    /// @param prev_result  Registration result of the previous frame.
    ///                     Uses H_raw (unregularized Hessian), T (optimized pose), error
    ///                     (sum of Mahalanobis-weighted squared residuals) and inlier count
    ///                     (used to scale H_raw via the reduced chi-squared statistic).
    /// @param T_pred       Predicted pose used as the initial guess for the current frame.
    void update(const RegistrationResult& prev_result, const Eigen::Isometry3f& T_pred) {
        this->has_prior_ = false;
        if (!this->params_.enabled) return;

        // Reduced chi-squared scaling: convert the raw Hessian (J^T Sigma^{-1} J under
        // unit-variance residual assumption) into a calibrated information matrix.
        //   DOF = 3 * N_inlier - 6   (3 = GICP residual dim, 6 = SE(3) params, both fixed)
        //   s^2 = error / DOF        (no clamp; degenerate inputs disable the prior)
        // Skip when DOF is non-positive (inlier <= 2) or when error is non-finite.
        const float dof = 3.0f * static_cast<float>(prev_result.inlier) - 6.0f;
        if (dof <= 0.0f) return;
        if (!std::isfinite(prev_result.error) || prev_result.error <= 0.0f) return;
        const float s_sq = prev_result.error / dof;
        const Eigen::Matrix<float, 6, 6> H_calibrated = prev_result.H_raw / s_sq;

        // R_rel = R_opt_prev^T * R_pred: relative rotation from the optimized previous frame
        // to the predicted current frame.  H_raw was built at prev_result.T, so this is
        // the correct rotation for the Adjoint transformation and for the per-axis delta.
        const Eigen::Matrix3f R_rel = prev_result.T.rotation().transpose() * T_pred.rotation();

        // Per-axis inter-frame delta expressed in T_pred body frame.
        const Eigen::AngleAxisf aa(R_rel);
        const Eigen::Vector3f delta_rot_body = aa.axis() * aa.angle();
        const Eigen::Vector3f delta_trans_body =
            T_pred.rotation().transpose() * (T_pred.translation() - prev_result.T.translation());

        // Per-axis process noise Q: velocity-proportional term plus an isotropic baseline.
        // Linear (|delta|) rather than quadratic scaling — quadratic would make the prior
        // nearly vanish at high speed (delta ~ 1m/frame).  Both vel_sigma and base_sigma are
        // parameterised as std-dev (units of [rad] / [m]) and appear squared in the variance
        // formula.  The baseline (base_sigma^2) is always added so the prior stays responsive
        // to sudden acceleration regardless of current speed.
        //   Q_rot[i]   = rot_base_sigma^2   + rot_vel_sigma^2   * |delta_rot_body[i]|
        //   Q_trans[i] = trans_base_sigma^2 + trans_vel_sigma^2 * |delta_trans_body[i]|
        const float rot_var_per_unit = this->params_.rot_vel_sigma * this->params_.rot_vel_sigma;
        const float trans_var_per_unit = this->params_.trans_vel_sigma * this->params_.trans_vel_sigma;
        const float rot_base_var = this->params_.rot_base_sigma * this->params_.rot_base_sigma;
        const float trans_base_var = this->params_.trans_base_sigma * this->params_.trans_base_sigma;
        const Eigen::Vector3f q_rot =
            delta_rot_body.cwiseAbs() * rot_var_per_unit + Eigen::Vector3f::Constant(rot_base_var);
        const Eigen::Vector3f q_trans =
            delta_trans_body.cwiseAbs() * trans_var_per_unit + Eigen::Vector3f::Constant(trans_base_var);

        // Rotate H_calibrated from T_opt_prev body frame into T_pred body frame via
        // rotation-only Adjoint: Ad = block_diag(R_rel, R_rel), H_curr = Ad^T * H_cal * Ad
        Eigen::Matrix<float, 6, 6> Ad = Eigen::Matrix<float, 6, 6>::Zero();
        Ad.block<3, 3>(0, 0) = R_rel;
        Ad.block<3, 3>(3, 3) = R_rel;
        const Eigen::Matrix<float, 6, 6> H_curr = Ad.transpose() * H_calibrated * Ad;

        // R = Q^{-1}: per-axis diagonal (safe because q[i] >= base_sigma^2 > 0 by construction)
        Eigen::Vector<float, 6> R_diag;
        R_diag.head<3>() = q_rot.cwiseInverse();
        R_diag.tail<3>() = q_trans.cwiseInverse();
        const Eigen::Matrix<float, 6, 6> R = R_diag.asDiagonal();

        // Omega_prior = (H^{-1} + Q)^{-1} = R - R(H + R)^{-1}R
        // (H + R) is always PD since R is PD, so this is robust to singular H.
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt(H_curr + R);
        if (ldlt.info() != Eigen::Success) return;
        this->Omega_prior_ = R - R * ldlt.solve(R);

        // Precompute inverse once; reused every iteration inside apply() and prior_error()
        this->T_pred_inv_ = T_pred.inverse();
        this->has_prior_ = true;
    }

    /// @brief Apply the MAP prior to the normal equations for one optimizer iteration.
    ///        Adds Omega_prior to H and b, and adds the prior's scalar cost to error.
    /// @param in    Linearized result after GICP (and optional degenerate regularization).
    /// @param T_est Current pose estimate at this iteration.
    /// @return Modified LinearizedResult with prior terms added.
    LinearizedResult apply(const LinearizedResult& in, const Eigen::Isometry3f& T_est) const {
        if (!is_active()) return in;

        // e_prior = Log(T_pred^{-1} * T_est): deviation of the current estimate from the prediction
        const Eigen::Vector<float, 6> e_prior = eigen_utils::lie::se3_log(this->T_pred_inv_ * T_est);
        const Eigen::Vector<float, 6> Omega_e = this->Omega_prior_ * e_prior;

        LinearizedResult ret = in;
        ret.H += this->Omega_prior_;
        ret.b += Omega_e;
        ret.error += 0.5f * e_prior.dot(Omega_e);
        return ret;
    }

    /// @brief Compute the scalar prior cost at a given pose.
    ///        Used to augment compute_error() results for LM/Dogleg step acceptance.
    float prior_error(const Eigen::Isometry3f& T_est) const {
        if (!is_active()) return 0.0f;
        const Eigen::Vector<float, 6> e = eigen_utils::lie::se3_log(this->T_pred_inv_ * T_est);
        return 0.5f * e.dot(this->Omega_prior_ * e);
    }

    bool is_active() const { return this->params_.enabled && this->has_prior_; }

private:
    MapPriorParams params_;
    bool has_prior_ = false;
    Eigen::Matrix<float, 6, 6> Omega_prior_ = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Isometry3f T_pred_inv_ = Eigen::Isometry3f::Identity();
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
