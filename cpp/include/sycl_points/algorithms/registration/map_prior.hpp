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
    /// @brief Velocity-proportional scale for rotation noise.
    ///        Q_rot = max(rot_min_noise, rot_vel_scale * delta_rot^2)   [rad^2 / rad^2]
    float rot_vel_scale = 1.0f;
    /// @brief Velocity-proportional scale for translation noise.
    ///        Q_trans = max(trans_min_noise, trans_vel_scale * delta_trans^2)   [m^2 / m^2]
    float trans_vel_scale = 1.0f;
    /// @brief Floor for rotation process noise [rad^2].
    ///        Prevents over-constraining during sudden acceleration.
    float rot_min_noise = 1e-3f;
    /// @brief Floor for translation process noise [m^2].
    ///        Prevents over-constraining during sudden acceleration.
    float trans_min_noise = 1e-4f;
    /// @brief EMA smoothing factor for the Hessian across frames.
    ///        H_ema = exp(alpha * log(H_curr) + (1-alpha) * log(Ad_ema^T * H_ema_prev * Ad_ema))
    ///        1.0 = current frame only (no smoothing), smaller = more temporal smoothing.
    float hessian_ema_alpha = 1.0f;
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
/// When hessian_ema_alpha < 1.0, H_curr is blended with the history in Log-Euclidean space.
/// H_ema_ is stored in the body frame of T_pred from the previous update() call (T_pred_prev),
/// so its rotation into the current frame uses a separate Adjoint Ad_ema:
///
///   Ad_ema        = block_diag(R_rel_ema, R_rel_ema),  R_rel_ema = R_pred_prev^T * R_pred
///   H_ema_rotated = Ad_ema^T * H_ema_prev * Ad_ema
///   H_smoothed    = exp(alpha * log(H_curr) + (1-alpha) * log(H_ema_rotated))
///
/// Note that Ad uses R_opt_prev (the optimised pose of the previous frame) while Ad_ema
/// uses R_pred_prev (the predicted pose saved from the previous call).  These differ by the
/// registration correction of the previous frame and must not be conflated.
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
///   Q_rot   = max(rot_min_noise,   rot_vel_scale   * delta_rot^2)
///   Q_trans = max(trans_min_noise, trans_vel_scale * delta_trans^2)
///
/// This tightens the prior during slow/smooth motion and loosens it during fast motion,
/// while the floor (min_noise) prevents over-constraining during sudden acceleration.
/// In degenerate directions H_raw_prev is small, so Omega_prior is also small and
/// nl_reg's Tikhonov penalty dominates — the two mechanisms are complementary.
class MapPrior {
public:
    void set_params(const MapPriorParams& params) {
        params_ = params;
        has_prior_ = false;
        has_hessian_ema_ = false;
        H_ema_ = Eigen::Matrix<float, 6, 6>::Zero();
    }

    /// @brief Precompute Omega_prior and T_pred_inv for the upcoming align() call.
    ///        Call this once per frame, after motion prediction and before align().
    /// @param prev_result  Registration result of the previous frame.
    ///                     Uses H_raw (unregularized Hessian), T (optimized pose), error
    ///                     (sum of Mahalanobis-weighted squared residuals) and inlier count
    ///                     (used to scale H_raw via the reduced chi-squared statistic).
    /// @param T_pred       Predicted pose used as the initial guess for the current frame.
    void update(const RegistrationResult& prev_result, const Eigen::Isometry3f& T_pred) {
        has_prior_ = false;
        if (!params_.enabled) return;

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

        // Per-axis process noise Q: velocity-proportional with a floor for sudden acceleration.
        //   Q_rot[i]   = max(rot_min_noise,   rot_vel_scale   * delta_rot_body[i]^2)
        //   Q_trans[i] = max(trans_min_noise, trans_vel_scale * delta_trans_body[i]^2)
        const Eigen::Vector3f q_rot =
            (delta_rot_body.cwiseProduct(delta_rot_body) * params_.rot_vel_scale).cwiseMax(params_.rot_min_noise);
        const Eigen::Vector3f q_trans = (delta_trans_body.cwiseProduct(delta_trans_body) * params_.trans_vel_scale)
                                            .cwiseMax(params_.trans_min_noise);

        // Rotate H_calibrated from T_opt_prev body frame into T_pred body frame via
        // rotation-only Adjoint: Ad = block_diag(R_rel, R_rel), H_curr = Ad^T * H_cal * Ad
        Eigen::Matrix<float, 6, 6> Ad = Eigen::Matrix<float, 6, 6>::Zero();
        Ad.block<3, 3>(0, 0) = R_rel;
        Ad.block<3, 3>(3, 3) = R_rel;
        const Eigen::Matrix<float, 6, 6> H_curr = Ad.transpose() * H_calibrated * Ad;

        // Log-Euclidean EMA: blend H_curr with the rotated history.
        // H_ema_ is stored in the body frame of T_pred_ema_prev_ (the T_pred saved from
        // the previous call), so its rotation uses Ad_ema built from R_pred_prev^T * R_pred,
        // which is distinct from Ad (which uses R_opt_prev^T * R_pred).
        Eigen::Matrix<float, 6, 6> H_smoothed;
        if (has_hessian_ema_ && params_.hessian_ema_alpha < 1.0f) {
            const Eigen::Matrix3f R_rel_ema = T_pred_ema_prev_.rotation().transpose() * T_pred.rotation();
            Eigen::Matrix<float, 6, 6> Ad_ema = Eigen::Matrix<float, 6, 6>::Zero();
            Ad_ema.block<3, 3>(0, 0) = R_rel_ema;
            Ad_ema.block<3, 3>(3, 3) = R_rel_ema;
            const Eigen::Matrix<float, 6, 6> H_ema_rotated = Ad_ema.transpose() * H_ema_ * Ad_ema;
            H_smoothed = spd_exp(params_.hessian_ema_alpha * spd_log(H_curr) +
                                 (1.0f - params_.hessian_ema_alpha) * spd_log(H_ema_rotated));
        } else {
            H_smoothed = H_curr;
        }
        H_ema_ = H_smoothed;        // stored in T_pred body frame
        T_pred_ema_prev_ = T_pred;  // saved for computing Ad_ema at the next call
        has_hessian_ema_ = true;

        // R = Q^{-1}: per-axis diagonal (safe because q >= min_noise > 0)
        Eigen::Vector<float, 6> R_diag;
        R_diag.head<3>() = q_rot.cwiseInverse();
        R_diag.tail<3>() = q_trans.cwiseInverse();
        const Eigen::Matrix<float, 6, 6> R = R_diag.asDiagonal();

        // Omega_prior = (H^{-1} + Q)^{-1} = R - R(H + R)^{-1}R
        // (H + R) is always PD since R is PD, so this is robust to singular H.
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt(H_smoothed + R);
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
    // Eigenvalues are clamped to 1e-9 before log to guard against near-zero or slightly
    // negative values caused by floating-point rounding in near-singular H matrices.
    static Eigen::Matrix<float, 6, 6> spd_log(const Eigen::Matrix<float, 6, 6>& A) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eig(A);
        const auto log_eigs = eig.eigenvalues().cwiseMax(1e-9f).array().log().matrix();
        return eig.eigenvectors() * log_eigs.asDiagonal() * eig.eigenvectors().transpose();
    }

    static Eigen::Matrix<float, 6, 6> spd_exp(const Eigen::Matrix<float, 6, 6>& A) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eig(A);
        return eig.eigenvectors() * eig.eigenvalues().array().exp().matrix().asDiagonal() *
               eig.eigenvectors().transpose();
    }

    MapPriorParams params_;
    bool has_prior_ = false;
    bool has_hessian_ema_ = false;
    Eigen::Matrix<float, 6, 6> Omega_prior_ = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 6> H_ema_ = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Isometry3f T_pred_inv_ = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f T_pred_ema_prev_ = Eigen::Isometry3f::Identity();
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
