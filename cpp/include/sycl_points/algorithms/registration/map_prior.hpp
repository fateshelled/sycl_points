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

using MapPriorMatrix = Eigen::Matrix<float, 6, 6>;

/// @brief Process covariance for one prediction interval, expressed in the
///        destination pose's body frame.
inline MapPriorMatrix make_map_prior_process_covariance(const MapPriorParams& params, const Eigen::Isometry3f& from,
                                                        const Eigen::Isometry3f& to) {
    const Eigen::Matrix3f R_rel = from.rotation().transpose() * to.rotation();
    const Eigen::AngleAxisf aa(R_rel);
    const Eigen::Vector3f delta_rot_body = aa.axis() * aa.angle();
    const Eigen::Vector3f delta_trans_body = to.rotation().transpose() * (to.translation() - from.translation());

    const float rot_var_per_unit = params.rot_vel_sigma * params.rot_vel_sigma;
    const float trans_var_per_unit = params.trans_vel_sigma * params.trans_vel_sigma;
    const float rot_base_var = params.rot_base_sigma * params.rot_base_sigma;
    const float trans_base_var = params.trans_base_sigma * params.trans_base_sigma;

    MapPriorMatrix Q = MapPriorMatrix::Zero();
    Q.diagonal().head<3>() = delta_rot_body.cwiseAbs() * rot_var_per_unit + Eigen::Vector3f::Constant(rot_base_var);
    Q.diagonal().tail<3>() =
        delta_trans_body.cwiseAbs() * trans_var_per_unit + Eigen::Vector3f::Constant(trans_base_var);
    return Q;
}

/// @brief Rotate accumulated process covariance into @p to's body frame and
///        append the covariance of the new prediction interval.
inline MapPriorMatrix accumulate_map_prior_process_covariance(const MapPriorParams& params,
                                                              const MapPriorMatrix& accumulated,
                                                              const Eigen::Isometry3f& from,
                                                              const Eigen::Isometry3f& to) {
    const Eigen::Matrix3f R_rel = from.rotation().transpose() * to.rotation();
    MapPriorMatrix Ad = MapPriorMatrix::Zero();
    Ad.block<3, 3>(0, 0) = R_rel;
    Ad.block<3, 3>(3, 3) = R_rel;

    const MapPriorMatrix next = Ad.transpose() * accumulated * Ad + make_map_prior_process_covariance(params, from, to);
    return 0.5f * (next + next.transpose());
}

/// @brief MAP estimation prior using the previous frame's Hessian as the information matrix.
///
/// Adds a Gaussian prior N(T_pred, Omega_prior^{-1}) to the GICP normal equations each
/// iteration.  The information matrix is computed once per frame via the Matrix Inversion
/// Lemma to avoid directly inverting H_raw_prev (which may be singular in degenerate cases):
///
///   R = Q^{-1}
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
///   DOF   = 3 * N_inlier - 6                (3 = GICP residual dim per inlier, 6 = SE(3) params)
///   s^2   = max(1.0, 2 * error_raw / DOF)   (factor of 2 cancels the 0.5 in compute_robust_error;
///                                            clamp >= 1.0 prevents over-confident prior)
///   H_cal = H_raw_prev / s^2                (Sigma_pose ~= s^2 * H_raw^{-1} => Lambda = H_raw / s^2)
///
/// Large s^2 (under-fit residuals or sparse inliers) loosens the prior; the lower clamp
/// at 1.0 prevents over-tightening when residuals come out smaller than the model's
/// expected unit variance (over-fit, noise-free simulation, etc.).
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
        this->last_valid_result_.reset();
        this->last_frame_registered_ = false;
        this->accumulated_process_covariance_.setZero();
        this->accumulation_pose_ = Eigen::Isometry3f::Identity();
        this->accumulation_initialized_ = false;
    }

    /// @brief Accumulate process covariance over a single prediction interval
    ///        [from -> to].  No-op until a valid prior source has been registered
    ///        via notify_registration_success(); the first call seeds
    ///        accumulation_pose_ from @p from, subsequent calls ignore @p from and
    ///        chain from the internal accumulation pose.
    ///        This is the entry point for both the regular inter-frame path and
    ///        the prediction-only fallback path (called once per process()).
    void accumulate_process_covariance(const Eigen::Isometry3f& from, const Eigen::Isometry3f& to) {
        if (!this->params_.enabled || !this->last_valid_result_) return;

        if (!this->accumulation_initialized_) {
            this->accumulation_pose_ = from;
            this->accumulated_process_covariance_.setZero();
            this->accumulation_initialized_ = true;
        }

        this->accumulated_process_covariance_ = accumulate_map_prior_process_covariance(
            this->params_, this->accumulated_process_covariance_, this->accumulation_pose_, to);
        this->accumulation_pose_ = to;
    }

    /// @brief Commit the registration result of the current frame and refresh
    ///        prior-source bookkeeping.  Call once after align() returns.
    ///        - On a valid result: replace last_valid_result_, reset the
    ///          accumulated process covariance, and seed accumulation_pose_ at
    ///          @p T_post so the next prediction interval starts fresh.
    ///        - On an invalid/unusable result: keep last_valid_result_ unchanged
    ///          and continue accumulating process covariance up to @p T_post, so
    ///          the prior weakens across successive unusable frames exactly as
    ///          the prediction-only fallback does.
    void notify_registration_success(const RegistrationResult& result, const Eigen::Isometry3f& T_post) {
        if (this->is_valid_prior_source(result)) {
            this->last_valid_result_ = std::make_shared<RegistrationResult>(result);
            this->accumulated_process_covariance_.setZero();
            this->accumulation_pose_ = T_post;
            this->accumulation_initialized_ = true;
            this->last_frame_registered_ = true;
        } else {
            this->accumulate_process_covariance(this->accumulation_pose_, T_post);
            this->last_frame_registered_ = false;
        }
    }

    /// @brief Mark the most recent frame as prediction-only (no registration
    ///        was run).  Does not touch the accumulated process covariance,
    ///        which was already advanced by the per-frame
    ///        accumulate_process_covariance() call in process().
    void notify_prediction_only() { this->last_frame_registered_ = false; }

    /// @brief Precompute Omega_prior and T_pred_inv for the upcoming align()
    ///        call using the internally tracked prior source and accumulated
    ///        process covariance.  Call once per frame, after motion prediction
    ///        and before align().  No-op when no valid prior source exists or
    ///        when accumulation has not been initialised yet.
    void prepare_for_align(const Eigen::Isometry3f& T_pred) {
        this->has_prior_ = false;
        if (!this->params_.enabled) return;
        if (!this->last_valid_result_ || !this->accumulation_initialized_) return;
        const RegistrationResult& prev_result = *this->last_valid_result_;
        const MapPriorMatrix& process_covariance = this->accumulated_process_covariance_;

        // Reduced chi-squared scaling: convert the raw Hessian (J^T Sigma^{-1} J under
        // unit-variance residual assumption) into a calibrated information matrix.
        //   DOF = 3 * N_inlier - 6                (3 = GICP residual dim, 6 = SE(3) params)
        //   s^2 = max(1.0, 2 * error_raw / DOF)
        // The factor of 2 undoes the 0.5 prefactor in compute_robust_error so that s^2
        // matches the standard reduced chi-squared definition (sum of squared residuals / DOF).
        // The lower clamp at 1.0 prevents over-tightening when fit error is below the model's
        // expected unit-variance level (over-fit, noise-free sim).  Skip when DOF is
        // non-positive (inlier <= 2) or when error is non-finite/negative.
        const float dof = 3.0f * static_cast<float>(prev_result.inlier) - 6.0f;
        if (dof <= 0.0f) return;
        if (!std::isfinite(prev_result.error_raw) || prev_result.error_raw < 0.0f) return;
        const float s_sq = std::max(1.0f, 2.0f * prev_result.error_raw / dof);
        const MapPriorMatrix H_calibrated = prev_result.H_raw / s_sq;

        // R_rel = R_opt_prev^T * R_pred: relative rotation from the optimized previous frame
        // to the predicted current frame.  H_raw was built at prev_result.T, so this is
        // the correct rotation for the Adjoint transformation and for the per-axis delta.
        const Eigen::Matrix3f R_rel = prev_result.T.rotation().transpose() * T_pred.rotation();

        // Per-axis inter-frame delta expressed in T_pred body frame.
        // Recompute from process_covariance diagonal for the diagonal Q inversion.
        const Eigen::Vector<float, 6> q_diag = process_covariance.diagonal();

        // Rotate H_calibrated from T_opt_prev body frame into T_pred body frame via
        // rotation-only Adjoint: Ad = block_diag(R_rel, R_rel), H_curr = Ad^T * H_cal * Ad
        MapPriorMatrix Ad = MapPriorMatrix::Zero();
        Ad.block<3, 3>(0, 0) = R_rel;
        Ad.block<3, 3>(3, 3) = R_rel;
        const MapPriorMatrix H_curr = Ad.transpose() * H_calibrated * Ad;

        // R = Q^{-1}: per-axis diagonal (safe because q[i] >= base_sigma^2 > 0 by construction)
        Eigen::Vector<float, 6> R_diag;
        R_diag.head<3>() = q_diag.head<3>().cwiseInverse();
        R_diag.tail<3>() = q_diag.tail<3>().cwiseInverse();
        const MapPriorMatrix R = R_diag.asDiagonal();

        // Omega_prior = (H^{-1} + Q)^{-1} = R - R(H + R)^{-1}R
        // (H + R) is always PD since R is PD, so this is robust to singular H.
        Eigen::LDLT<MapPriorMatrix> ldlt(H_curr + R);
        if (ldlt.info() != Eigen::Success) return;
        this->Omega_prior_ = R - R * ldlt.solve(R);
        this->Omega_prior_ = 0.5f * (this->Omega_prior_ + this->Omega_prior_.transpose());
        if (!this->Omega_prior_.allFinite()) return;

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
    const MapPriorMatrix& information_matrix() const { return this->Omega_prior_; }

    /// @brief Whether a registration result valid enough to anchor the MAP
    ///        prior is currently held.  Exposed so upstream pipelines can feed
    ///        motion predictors that gate Hessian-based adaptive weighting on
    ///        the availability of a prior source.
    bool has_valid_prior_source() const { return this->last_valid_result_ != nullptr; }
    /// @brief Last registration result accepted as a prior source (or null).
    const RegistrationResult::Ptr& last_valid_result() const { return this->last_valid_result_; }
    /// @brief Whether the most recent frame produced a usable registration
    ///        (true) or fell back to prediction-only / unusable result (false).
    bool last_frame_registered() const { return this->last_frame_registered_; }

private:
    /// @brief Heuristic validity test for a registration result being used as
    ///        the prior source.  Mirrors the conditions originally inlined in
    ///        the LiDAR odometry pipeline (lidar_odometry.hpp).
    static bool is_valid_prior_source(const RegistrationResult& result) {
        return result.T.matrix().allFinite() && result.H_raw.allFinite() && std::isfinite(result.error_raw) &&
               result.error_raw >= 0.0f && result.inlier > 2;
    }

    MapPriorParams params_;
    bool has_prior_ = false;
    MapPriorMatrix Omega_prior_ = MapPriorMatrix::Zero();
    Eigen::Isometry3f T_pred_inv_ = Eigen::Isometry3f::Identity();

    /// @brief Last registration result accepted as the MAP prior source.
    ///        Reset only by set_params() or by notify_registration_success()
    ///        receiving a valid result; retained across prediction-only and
    ///        unusable-result frames to keep Omega_prior anchored.
    RegistrationResult::Ptr last_valid_result_ = nullptr;
    /// @brief True iff the most recent frame produced a usable registration.
    ///        Read by motion-prediction callers gating Hessian-based adaptive
    ///        weighting.
    bool last_frame_registered_ = false;
    /// @brief Process covariance accumulated across prediction intervals since
    ///        the last valid registration.  Expressed in accumulation_pose_'s
    ///        body frame; rotated forward each time accumulate_process_covariance()
    ///        is called and reset to zero by notify_registration_success().
    MapPriorMatrix accumulated_process_covariance_ = MapPriorMatrix::Zero();
    /// @brief Body-frame anchor for the next accumulate_process_covariance()
    ///        step.  Updated to @p to after every accumulation.
    Eigen::Isometry3f accumulation_pose_ = Eigen::Isometry3f::Identity();
    /// @brief False until the first accumulate_process_covariance() call
    ///        following a valid registration seeds accumulation_pose_.
    ///        prepare_for_align() refuses to build a prior until this is true.
    bool accumulation_initialized_ = false;
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
