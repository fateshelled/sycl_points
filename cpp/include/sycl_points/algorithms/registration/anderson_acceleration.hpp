#pragma once

#include <deque>

#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {

/// @brief Anderson acceleration for fixed-point iterations on SE(3).
///
/// Applies Anderson(m) acceleration (Walker & Ni 2011, Type 2) to the
/// outer Registration optimization loop.  Each call to apply() takes the
/// T proposed by the underlying optimizer and returns an extrapolated T
/// that should converge faster.
///
/// The iterate is represented in the Lie algebra se(3) relative to the
/// initial guess:  x_k = se3_log(T_k * T_initial^{-1}).
/// The residual is f_k = x_{k+1} - x_k (effective update step).
///
/// Anderson mixing step (window size m_k = min(m, k)):
///   F_k = [f_{k-m_k+1}-f_{k-m_k}, ..., f_k-f_{k-1}]  (6 x m_k)
///   X_k = [x_{k-m_k+1}-x_{k-m_k}, ..., x_{k+1}-x_k]  (6 x m_k)
///   gamma* = argmin_gamma ||f_k - F_k * gamma||_2^2
///   x_{k+1}^AA = x_{k+1} - (1 - beta) * f_k - (beta * F_k + X_k) * gamma*
///   T_{k+1}^AA = se3_exp(x_{k+1}^AA) * T_initial
class AndersonAcceleration {
public:
    AndersonAcceleration() = default;

    /// @brief Reset the acceleration state.
    /// @param window_size History window size m (Anderson(m))
    /// @param beta Mixing parameter: 1.0 = pure Anderson acceleration
    void reset(size_t window_size, float beta) {
        window_size_ = window_size;
        beta_ = beta;
        x_hist_.clear();
        f_hist_.clear();
        initial_T_inv_cached_ = false;
    }

    /// @brief Apply Anderson acceleration to the current iterate.
    /// @param input_T  Transformation proposed by the underlying optimizer this iteration
    /// @param initial_T  The initial guess passed to Registration::align()
    /// @return Accelerated transformation
    Eigen::Isometry3f apply(const Eigen::Isometry3f& input_T, const Eigen::Isometry3f& initial_T) {
        // Cache initial_T^{-1} on the first call (constant throughout the align() loop)
        if (!initial_T_inv_cached_) {
            initial_T_inv_ = initial_T.inverse();
            initial_T_inv_cached_ = true;
        }

        // Compute relative transform in Lie algebra: x_{k+1} = log(input_T * initial_T^{-1})
        const Eigen::Isometry3f rel = input_T * initial_T_inv_;
        const Eigen::Vector<float, 6> x_new = eigen_utils::lie::se3_log(rel);

        if (x_hist_.empty()) {
            // First iteration: just store and return as-is
            x_hist_.push_back(x_new);
            return input_T;
        }

        // Compute residual f_k = x_{k+1} - x_k
        const Eigen::Vector<float, 6> f_new = x_new - x_hist_.back();
        f_hist_.push_back(f_new);

        // Keep history bounded.  To construct m_k = window_size_ columns in F_k / X_k
        // we need window_size_+1 residuals (to form window_size_ differences), so allow
        // f_hist_ to hold up to window_size_+1 entries before trimming.
        while (x_hist_.size() > window_size_ + 1) {
            x_hist_.pop_front();
        }
        while (f_hist_.size() > window_size_ + 1) {
            f_hist_.pop_front();
        }

        // Store current iterate AFTER trimming
        x_hist_.push_back(x_new);

        const int m_k = static_cast<int>(f_hist_.size()) - 1;
        if (m_k <= 0) {
            // Not enough history yet for mixing
            return input_T;
        }

        // Build F_k (6 x m_k): column differences of residuals
        // Build X_k (6 x m_k): column differences of iterates
        // Since f_hist_[j] = x_hist_[j+1] - x_hist_[j], X_k can be built directly
        // from f_hist_ without recomputing differences from x_hist_.
        Eigen::MatrixXf F_k(6, m_k);
        Eigen::MatrixXf X_k(6, m_k);
        for (int j = 0; j < m_k; ++j) {
            F_k.col(j) = f_hist_[j + 1] - f_hist_[j];
            X_k.col(j) = f_hist_[j];
        }

        // Solve unconstrained LS: gamma* = argmin ||f_k - F_k * gamma||_2
        // Use column-pivoting QR for robustness against rank-deficient F_k
        const Eigen::Vector<float, 6>& f_k = f_hist_.back();
        const Eigen::VectorXf gamma = F_k.colPivHouseholderQr().solve(f_k);

        // Guard against numerical failures (NaN/Inf) that can arise in degenerate cases
        if (!gamma.allFinite()) {
            return input_T;
        }

        // Accelerated iterate (Walker & Ni 2011, Type 2 with damping beta):
        //   x_acc = x_{k+1} - (1 - beta) * f_k - (beta * F_k + X_k) * gamma*
        const Eigen::Vector<float, 6> x_acc = x_new - (1.0f - beta_) * f_k - (beta_ * F_k + X_k) * gamma;

        // Retract back to SE(3): T_acc = se3_exp(x_acc) * initial_T
        const Eigen::Isometry3f T_acc =
            Eigen::Isometry3f(eigen_utils::lie::se3_exp(x_acc)) * initial_T;
        return T_acc;
    }

private:
    size_t window_size_ = 5;
    float beta_ = 1.0f;

    // Cached inverse of initial_T (constant throughout an align() call)
    Eigen::Isometry3f initial_T_inv_;
    bool initial_T_inv_cached_ = false;

    // x_hist_[i] = se3_log(T_i * T_initial^{-1})
    std::deque<Eigen::Vector<float, 6>> x_hist_;
    // f_hist_[i] = x_hist_[i+1] - x_hist_[i]
    std::deque<Eigen::Vector<float, 6>> f_hist_;
};

}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
