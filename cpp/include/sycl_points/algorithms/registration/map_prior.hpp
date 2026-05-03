#pragma once

#include <Eigen/Dense>

#include "sycl_points/algorithms/registration/linearized_result.hpp"
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
    }

    /// @brief Precompute Omega_prior and T_pred_inv for the upcoming align() call.
    ///        Call this once per frame, after motion prediction and before align().
    /// @param H_raw_prev   Unregularized Hessian (RegistrationResult::H_raw) from the previous frame.
    ///                     This was computed at T_opt_prev, so the Adjoint must use T_opt_prev.
    /// @param T_opt_prev   Optimized pose of the previous frame (= odom_ in LiDAROdometryPipeline).
    /// @param T_pred       Predicted pose used as the initial guess for the current frame.
    void update(const Eigen::Matrix<float, 6, 6>& H_raw_prev, const Eigen::Isometry3f& T_opt_prev,
                const Eigen::Isometry3f& T_pred) {
        has_prior_ = false;
        if (!params_.enabled) return;

        // R_rel = R_opt_prev^T * R_pred: relative rotation from the optimized previous frame
        // to the predicted current frame.  H_raw_prev was built at T_opt_prev, so this is
        // the correct rotation for the Adjoint transformation and for the per-axis delta.
        const Eigen::Matrix3f R_rel = T_opt_prev.rotation().transpose() * T_pred.rotation();

        // Per-axis inter-frame delta expressed in T_pred body frame.
        const Eigen::AngleAxisf aa(R_rel);
        const Eigen::Vector3f delta_rot_body = aa.axis() * aa.angle();
        const Eigen::Vector3f delta_trans_body =
            T_pred.rotation().transpose() * (T_pred.translation() - T_opt_prev.translation());

        // Per-axis process noise Q: velocity-proportional with a floor for sudden acceleration.
        //   Q_rot[i]   = max(rot_min_noise,   rot_vel_scale   * delta_rot_body[i]^2)
        //   Q_trans[i] = max(trans_min_noise, trans_vel_scale * delta_trans_body[i]^2)
        const Eigen::Vector3f q_rot =
            (delta_rot_body.cwiseProduct(delta_rot_body) * params_.rot_vel_scale).cwiseMax(params_.rot_min_noise);
        const Eigen::Vector3f q_trans = (delta_trans_body.cwiseProduct(delta_trans_body) * params_.trans_vel_scale)
                                            .cwiseMax(params_.trans_min_noise);

        // Rotate H_raw_prev from T_opt_prev body frame into T_pred body frame via rotation-only
        // Adjoint: Ad = block_diag(R_rel, R_rel), H_curr = Ad^T * H_raw_prev * Ad
        Eigen::Matrix<float, 6, 6> Ad = Eigen::Matrix<float, 6, 6>::Zero();
        Ad.block<3, 3>(0, 0) = R_rel;
        Ad.block<3, 3>(3, 3) = R_rel;
        const Eigen::Matrix<float, 6, 6> H_curr = Ad.transpose() * H_raw_prev * Ad;

        // R = Q^{-1}: per-axis diagonal (safe because q >= min_noise > 0)
        Eigen::Vector<float, 6> R_diag;
        R_diag.head<3>() = q_rot.cwiseInverse();
        R_diag.tail<3>() = q_trans.cwiseInverse();
        const Eigen::Matrix<float, 6, 6> R = R_diag.asDiagonal();

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
