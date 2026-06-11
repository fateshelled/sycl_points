#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/imu/imu_initial_alignment.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/algorithms/lio/lio_registration.hpp"
#include "sycl_points/algorithms/registration/dogleg_step.hpp"
#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/pipeline/lidar_inertial_odometry_params.hpp"
#include "sycl_points/pipeline/pointcloud_processing.hpp"
#include "sycl_points/pipeline/submapping.hpp"
#include "sycl_points/utils/time_utils.hpp"

// ---------------------------------------------------------------------------
// LiDAR-Inertial Odometry Pipeline
//
// Each frame runs an iterative optimisation loop (Gauss-Newton, Levenberg-
// Marquardt, or Powell dogleg, selected by optimization_method) that combines
// the ICP Hessian/gradient (6×6, from SYCL parallel reduction) with the IMU
// prior Hessian/gradient (15×15) into a unified 15-DOF normal equation solved
// by LDLT.  GN commits the damped step unconditionally; LM and dogleg accept or
// reject each trial step on the combined cost (reduced-chi²-weighted robust ICP
// error with frozen correspondences + IMU prior Mahalanobis distance).
//
// Frame convention: single-stage, REP-105 style (FAST-LIO2-like).
//   odom ──(estimated, dynamic)──▶ lidar     (odom is the gravity-aligned world frame)
//
//   The 15-DOF IEKF state x_ is the LiDAR pose expressed directly in the gravity-
//   aligned odom (world) frame:
//     x_.position  = odom-frame position of the LiDAR origin [m]
//     x_.rotation  = R_odom_lidar  (SO(3))
//     x_.velocity  = odom-frame velocity of the LiDAR body [m/s]
//     x_.accel_bias / gyro_bias = IMU biases (body frame)
//
//   The initial orientation is Rz(user_yaw) * R_gravity (roll/pitch from the IMU
//   initial alignment), so odom is gravity-aligned and the preintegration gravity
//   stays (0,0,-g) canonical.  The ROS node converts to base_link via the static
//   T_base_link_to_lidar extrinsic when broadcasting odom → base_link.
//
// The ICP Hessian is embedded into the 15-DOF LiDAR-frame error-state.  The
// IMU preintegration covariance lives in the IMU error-state and is rotated
// into the LiDAR error-state via transform_covariance_imu_to_lidar() before
// being passed to compute_imu_hessian_gradient(); the inverse Jacobian is
// applied in reset_imu_preintegration() when handing P_post back to the
// preintegrator.
// ---------------------------------------------------------------------------

namespace sycl_points {
namespace pipeline {
namespace lidar_inertial_odometry {

class LidarInertialOdometryPipeline {
public:
    using Ptr = std::shared_ptr<LidarInertialOdometryPipeline>;
    using ConstPtr = std::shared_ptr<const LidarInertialOdometryPipeline>;

    enum class ResultType : std::int8_t {
        success = 0,
        first_frame,
        waiting_initial_alignment,
        error = 100,
        old_timestamp,
        small_number_of_points,
    };

    explicit LidarInertialOdometryPipeline(const Parameters& params) {
        params_ = params;
        params_.imu.enable = true;  // IMU is mandatory for LIO
        // Ensure the IMU buffer is large enough to satisfy the alignment window.
        // The pop_front rule keeps span ≤ buffer_duration_sec, so equal values can fail.
        if (params_.imu.initial_alignment.enable) {
            const double need = static_cast<double>(params_.imu.initial_alignment.required_duration_sec) + 0.2;
            if (params_.imu.buffer_duration_sec < need) {
                params_.imu.buffer_duration_sec = need;
            }
        }
        initialize();
    }

    // -------------------------------------------------------------------------
    // Getters
    // -------------------------------------------------------------------------
    auto get_device_queue() const { return this->queue_ptr_; }
    const auto& get_error_message() const { return this->error_message_; }
    const auto& get_current_processing_time() const { return this->current_processing_time_; }
    const auto& get_total_processing_times() const { return this->total_processing_times_; }

    /// @brief Current / previous LiDAR pose in the odom (world) frame.
    ///        odom is gravity-aligned; T_odom_to_lidar is tracked directly (single-stage).
    const auto& get_odom() const { return this->odom_; }
    const auto& get_prev_odom() const { return this->prev_odom_; }
    const auto& get_last_keyframe_pose() const { return this->submap_->get_last_keyframe_pose(); }

    /// @brief Keyframe poses in the odom (world) frame.
    const auto& get_keyframe_poses() const { return this->submap_->get_keyframe_poses(); }

    const PointCloudShared& get_preprocessed_point_cloud() const { return *this->preprocessed_pc_; }
    const PointCloudShared& get_submap_point_cloud() const { return this->submap_->get_submap_point_cloud(); }
    const PointCloudShared& get_last_keyframe_point_cloud() const {
        return this->submap_->get_last_keyframe_point_cloud();
    }
    const auto& get_registration_result() const { return *this->reg_result_; }
    const imu::State& get_lio_state() const { return this->x_; }

    // -------------------------------------------------------------------------
    // IMU measurement feed (thread-safe)
    // -------------------------------------------------------------------------
    void add_imu_measurement(const imu::IMUMeasurement& meas) {
        std::lock_guard<std::mutex> lock(this->imu_mutex_);
        if (!meas.accel.allFinite() || !meas.gyro.allFinite()) return;
        if (!imu_buffer_.empty() && meas.timestamp <= imu_buffer_.back().timestamp) return;
        const double latest = meas.timestamp;
        this->imu_buffer_.push_back(meas);
        while (latest - this->imu_buffer_.front().timestamp > this->params_.imu.buffer_duration_sec) {
            this->imu_buffer_.pop_front();
        }
    }

    std::deque<imu::IMUMeasurement> get_imu_buffer() const {
        std::lock_guard<std::mutex> lock(this->imu_mutex_);
        return this->imu_buffer_;
    }

    // -------------------------------------------------------------------------
    // Main process call (one LiDAR frame)
    // -------------------------------------------------------------------------
    ResultType process(const PointCloudShared::Ptr scan, double timestamp) {
        this->error_message_.clear();

        // Initial roll/pitch alignment from stationary IMU samples.
        // Runs once before the first scan is accepted as the reference frame.
        if (this->is_first_frame_ && this->alignment_estimator_->enabled() && !this->alignment_estimator_->is_done()) {
            const imu::IMUBias current_bias{this->x_.gyro_bias, this->x_.accel_bias};
            const auto out = this->alignment_estimator_->try_align(timestamp, this->get_imu_buffer(), current_bias);
            if (out.status != imu::InitialAlignmentEstimator::Status::success) {
                this->error_message_ = std::string("initial_alignment: ") + out.error_message;
                return ResultType::waiting_initial_alignment;
            }
            this->apply_initial_alignment(out);
        }

        if (this->last_frame_time_ > 0.0) {
            const float dt = static_cast<float>(timestamp - this->last_frame_time_);
            if (dt > 0.0f) {
                dt_ = dt;
            } else {
                this->error_message_ = "old timestamp";
                return ResultType::old_timestamp;
            }
        }
        this->clear_current_processing_time();

        // Preprocessing
        double dt_preprocessing = 0.0;
        {
            try {
                time_utils::measure_execution([&]() { this->preprocess(scan); }, dt_preprocessing);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("preprocess: ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
        }

        // Covariance estimation
        {
            double dt_cov = 0.0;
            try {
                time_utils::measure_execution([&]() { this->compute_covariances(); }, dt_cov);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("compute_covariances: ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::compute_covariances, dt_cov);
        }

        // Refine filter (angle incidence filter + intensity zscore)
        {
            double dt_refine = 0.0;
            try {
                time_utils::measure_execution([&]() { this->refine_filter(this->preprocessed_pc_); }, dt_refine);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("refine_filter: ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            dt_preprocessing += dt_refine;
            this->add_delta_time(ProcessName::preprocessing, dt_preprocessing);
        }

        if (this->preprocessed_pc_->size() <= this->params_.registration.min_num_points) {
            this->error_message_ = "point cloud size is too small";
            return ResultType::small_number_of_points;
        }

        // Integrate IMU measurements for this window
        {
            this->imu_batch_.clear();
            std::lock_guard<std::mutex> lock(this->imu_mutex_);
            this->imu_batch_.reserve(this->imu_buffer_.size());
            for (const auto& m : this->imu_buffer_) {
                if (m.timestamp <= this->last_imu_reset_timestamp_) continue;
                if (m.timestamp > timestamp) break;
                this->imu_batch_.push_back(m);
            }
        }
        this->imu_preintegration_->integrate_batch(this->imu_batch_);

        // First frame: initialize state and submap, no registration
        if (this->is_first_frame_) {
            try {
                // Anchor the first keyframe at the post-alignment LiDAR pose (odom_ in
                // the odom/world frame) so that subsequent add_frame() calls share the
                // same reference frame.  Without this, the first submap content lives
                // in the sensor frame while later frames live in odom — visible as a
                // small (gravity-correction) tilt mismatch in rviz until the voxel hash
                // map accumulates enough data to swap in.
                this->submap_->add_first_frame(*this->preprocessed_pc_, timestamp, this->odom_);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("build_submap (first frame): ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->is_first_frame_ = false;
            this->last_frame_time_ = timestamp;
            this->last_imu_reset_timestamp_ = timestamp;

            // odom_/prev_odom_ were initialized in initialize() from params and may have
            // been updated by apply_initial_alignment().  Confirm x_.position/rotation from
            // odom_ here so the aligned rotation (when applicable) propagates without
            // re-reading mutated params.
            this->x_.position = this->odom_.translation();
            this->x_.rotation = this->odom_.rotation();
            this->x_.velocity = Eigen::Vector3f::Zero();
            // x_.accel_bias / gyro_bias already set in initialize();
            // x_.gyro_bias may also have been updated by apply_initial_alignment().

            this->reset_imu_preintegration();
            return ResultType::first_frame;
        }

        // LIO registration
        {
            double dt_reg = 0.0;
            try {
                *this->reg_result_ = time_utils::measure_execution([&]() { return this->lio_registration(); }, dt_reg);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("lio_registration: ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::registration, dt_reg);
        }
        this->last_imu_reset_timestamp_ = timestamp;

        // Submapping
        {
            double dt_sub = 0.0;
            try {
                time_utils::measure_execution([&]() { this->submapping(*this->reg_result_, timestamp); }, dt_sub);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("submapping: ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            add_delta_time(ProcessName::build_submap, dt_sub);
        }

        this->prev_odom_ = this->odom_;
        this->odom_ = this->reg_result_->T;
        this->last_frame_time_ = timestamp;

        return ResultType::success;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // -------------------------------------------------------------------------
    // Member variables
    // -------------------------------------------------------------------------
    sycl_utils::DeviceQueue::Ptr queue_ptr_ = nullptr;

    PointCloudShared::Ptr preprocessed_pc_ = nullptr;
    PointCloudShared::Ptr registration_input_pc_ = nullptr;  // random-sampled source for ICP linearization
    bool is_first_frame_ = true;
    imu::InitialAlignmentEstimator::Ptr alignment_estimator_ = nullptr;
    pointcloud_processing::ProcessingContext processing_ctx_;

    shared_vector_ptr<float> icp_weights_ = nullptr;

    pointcloud_processing::PCProcessor::Ptr pc_processor_ = nullptr;
    algorithms::registration::Registration::Ptr registration_ = nullptr;

    algorithms::registration::RegistrationResult::Ptr reg_result_ = nullptr;

    /// Previous / current LiDAR pose (T_odom_to_lidar) in the odom (world) frame.
    /// Kept in sync with the IEKF state x_ (odom_ == state_to_pose(x_) after each frame).
    Eigen::Isometry3f prev_odom_;
    Eigen::Isometry3f odom_;

    submapping::Submap::Ptr submap_ = nullptr;

    double last_frame_time_ = -1.0;
    double last_imu_reset_timestamp_ = -1.0;
    float dt_ = -1.0f;

    Parameters params_;

    // LIO state (LiDAR frame convention, 15-DOF: position, rotation, velocity, accel_bias, gyro_bias)
    imu::State x_;
    // Posterior covariance passed as initial covariance to the next IMU reset window. (LiDAR frame, 15×15)
    Eigen::Matrix<float, 15, 15> P_post_ = Eigen::Matrix<float, 15, 15>::Zero();

    imu::IMUPreintegration::Ptr imu_preintegration_ = nullptr;
    /// Window-start IMU orientation/velocity snapshotted at the latest preintegration
    /// reset.  These act as the linearization point for predict_relative_transform()
    /// and must remain frozen across IEKF iterations even when x_ is being updated.
    Eigen::Matrix3f imu_R_world_at_reset_ = Eigen::Matrix3f::Identity();
    Eigen::Vector3f imu_v_world_at_reset_ = Eigen::Vector3f::Zero();

    std::deque<imu::IMUMeasurement> imu_buffer_;
    mutable std::mutex imu_mutex_;
    std::vector<imu::IMUMeasurement> imu_batch_;

    std::string error_message_;

    enum class ProcessName { preprocessing, compute_covariances, registration, build_submap };
    const std::map<ProcessName, std::string> pn_map_ = {
        {ProcessName::preprocessing, "1. preprocessing"},
        {ProcessName::compute_covariances, "2. compute covariances"},
        {ProcessName::registration, "3. lio registration"},
        {ProcessName::build_submap, "4. build submap"},
    };
    std::map<std::string, double> current_processing_time_;
    std::map<std::string, std::vector<double>> total_processing_times_;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    static Eigen::Isometry3f state_to_pose(const imu::State& s) {
        Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
        T.linear() = s.rotation;
        T.translation() = s.position;
        return T;
    }

    bool is_lio_converged(const Eigen::Matrix<float, 15, 1>& delta) const {
        return delta.segment<3>(imu::State::kIdxRot).norm() < params_.lio.criteria.rotation &&
               delta.segment<3>(imu::State::kIdxPos).norm() < params_.lio.criteria.translation;
    }

    /// @brief Decide whether the IMU bias states are observable in this window.
    ///
    /// Returns true (always update biases) unless freeze_on_low_excitation is set,
    /// in which case the window must show gyro or specific-force variation above the
    /// configured thresholds.  Near-stationary windows return false so the caller can
    /// hold the biases fixed instead of letting them absorb measurement noise.
    ///
    /// Both deviations are measured on the full 3-D vector, not its magnitude.  Using
    /// the accel magnitude alone would miss a constant-rate turn: the gravity vector
    /// rotates in the body frame so the accel components vary while |a| stays ≈ g, and
    /// the gyro is constant so its deviation is ~0 — the window would be wrongly judged
    /// unobservable and freeze the gyro bias exactly when rotation makes it observable.
    bool imu_bias_observable() const {
        const auto& be = this->params_.lio.bias_estimation;
        if (!be.freeze_on_low_excitation) return true;
        if (this->imu_batch_.size() < 2) return false;

        Eigen::Vector3f gyro_mean = Eigen::Vector3f::Zero();
        Eigen::Vector3f accel_mean = Eigen::Vector3f::Zero();
        for (const auto& m : this->imu_batch_) {
            gyro_mean += m.gyro;
            accel_mean += m.accel;
        }
        const float n = static_cast<float>(this->imu_batch_.size());
        gyro_mean /= n;
        accel_mean /= n;

        float gyro_dev = 0.0f;
        float accel_dev = 0.0f;
        for (const auto& m : this->imu_batch_) {
            gyro_dev = std::max(gyro_dev, (m.gyro - gyro_mean).norm());
            accel_dev = std::max(accel_dev, (m.accel - accel_mean).norm());
        }
        return gyro_dev > be.gyro_excitation_threshold || accel_dev > be.accel_excitation_threshold;
    }

    /// @brief Clamp a bias vector to a maximum L2 norm (no-op when max_norm <= 0).
    static void clamp_bias_norm(Eigen::Vector3f& bias, float max_norm) {
        if (max_norm <= 0.0f) return;
        const float norm = bias.norm();
        if (norm > max_norm) bias *= (max_norm / norm);
    }

    void reset_imu_preintegration() {
        const Eigen::Isometry3f& T_i2l = this->params_.imu.T_imu_to_lidar;
        const Eigen::Matrix3f R_world_imu = x_.rotation * T_i2l.rotation();

        Eigen::Matrix<float, 15, 15> P_initial = this->P_post_;

        // Add fixed-sigma floors to P_initial before resetting the integrator.
        // Velocity floor: ensures P_pred[p,p] ≳ (fd_velocity_sigma × dt)², keeping
        //   H_imu[p,p] on the same scale as H_icp regardless of accel_noise_density.
        // Rotation floor: same mechanism for H_imu[φ,φ] vs gyro_noise_density.
        const float sv2 = this->params_.lio.fd_velocity_sigma * this->params_.lio.fd_velocity_sigma;
        P_initial.block<3, 3>(imu::State::kIdxVel, imu::State::kIdxVel) += sv2 * Eigen::Matrix3f::Identity();
        const float sr2 = this->params_.lio.icp_rotation_sigma * this->params_.lio.icp_rotation_sigma;
        P_initial.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot) += sr2 * Eigen::Matrix3f::Identity();

        // P_post_ is in LiDAR error-state frame; IMUPreintegration expects IMU body frame.
        // Convert before passing so that get_raw().covariance → transform_covariance_imu_to_lidar
        // yields the correct P_pred_lidar without double-transformation.
        const Eigen::Matrix<float, 15, 15> P_initial_imu =
            algorithms::lio::transform_covariance_lidar_to_imu(P_initial, T_i2l, this->x_.rotation);

        this->imu_preintegration_->reset({this->x_.gyro_bias, this->x_.accel_bias}, P_initial_imu);
        this->imu_R_world_at_reset_ = R_world_imu;
        this->imu_v_world_at_reset_ = this->x_.velocity;
    }

    /// @brief Predict the 15-DOF state from IMU preintegration.
    imu::State predict_state() const {
        const Eigen::Isometry3f& T_i2l = this->params_.imu.T_imu_to_lidar;
        const imu::IMUBias current_bias{this->x_.gyro_bias, this->x_.accel_bias};

        // Relative pose prediction (gravity + initial velocity already compensated)
        const TransformMatrix T_imu_rel_mat = this->imu_preintegration_->predict_relative_transform(
            this->imu_R_world_at_reset_, this->imu_v_world_at_reset_, current_bias);
        Eigen::Isometry3f T_imu_rel = Eigen::Isometry3f::Identity();
        T_imu_rel.linear() = T_imu_rel_mat.block<3, 3>(0, 0);
        T_imu_rel.translation() = T_imu_rel_mat.block<3, 1>(0, 3);

        // Convert IMU-relative transform to LiDAR-relative: T_lidar_rel = T_i2l * T_imu_rel * T_i2l^{-1}
        const Eigen::Isometry3f T_lidar_rel = T_i2l * T_imu_rel * T_i2l.inverse();
        const Eigen::Isometry3f T_pred = state_to_pose(x_) * T_lidar_rel;

        // Velocity prediction in world frame: v_j = v_i + g*dt + R_world_imu * Delta_v
        const auto c = this->imu_preintegration_->get_corrected(current_bias);
        const Eigen::Matrix3f R_world_imu = this->x_.rotation * T_i2l.rotation();
        const float dt_f = static_cast<float>(c.dt_total);

        imu::State pred;
        pred.position = T_pred.translation();
        pred.rotation = T_pred.rotation();
        pred.velocity = this->x_.velocity + this->params_.imu.preintegration.gravity * dt_f + R_world_imu * c.Delta_v;
        pred.accel_bias = this->x_.accel_bias;
        pred.gyro_bias = this->x_.gyro_bias;
        return pred;
    }

    /// @brief Combined LIO Gauss-Newton optimization for one LiDAR frame.
    algorithms::registration::RegistrationResult lio_registration() {
        // ---- Build prior from IMU preintegration ----
        imu::State x_pred = predict_state();

        // Transform IMU-frame preintegration covariance into the LiDAR error-state frame.
        const Eigen::Matrix<float, 15, 15> P_pred = algorithms::lio::transform_covariance_imu_to_lidar(
            this->imu_preintegration_->get_raw().covariance, this->params_.imu.T_imu_to_lidar, x_pred.rotation);

        Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Zero();
        Eigen::Matrix<float, 15, 1> b_imu = Eigen::Matrix<float, 15, 1>::Zero();
        const bool imu_valid = imu::compute_imu_hessian_gradient(x_pred, x_pred, P_pred, H_imu, b_imu);

        // ---- Prepare ICP source (random sampling) ----
        const auto& rs = this->params_.registration.pipeline.random_sampling;
        const PointCloudShared* source =
            (rs.enable && this->preprocessed_pc_->size() > rs.num)
                ? (this->pc_processor_->random_sampling(*this->preprocessed_pc_, *this->registration_input_pc_, rs.num),
                   this->registration_input_pc_.get())
                : this->preprocessed_pc_.get();

        // ---- Gauss-Newton loop ----
        imu::State x_op = x_pred;

        algorithms::registration::Registration::ExecutionOptions options;
        options.dt = this->dt_;
        options.prev_pose = this->odom_.matrix();

        const TransformMatrix T_initial = state_to_pose(x_pred).matrix();
        const auto& robust = this->params_.registration.pipeline.robust;

        // When excitation is too low the bias states are unobservable; hold them
        // fixed for this frame so the optimizer does not absorb noise into them.
        const bool update_bias = imu_bias_observable();

        // Per-point residual dimension for the chi-squared DOF estimate:
        // point-to-plane constrains 1 dimension per correspondence; point-to-point /
        // GICP / point-to-distribution use 3-D residuals.  GenZ blends both — using
        // the smaller dimension is the conservative (loosening-only) choice.
        const auto reg_type = this->params_.registration.pipeline.registration.reg_type;
        const float icp_residual_dim = (reg_type == algorithms::registration::RegType::POINT_TO_PLANE ||
                                        reg_type == algorithms::registration::RegType::GENZ)
                                           ? 1.0f
                                           : 3.0f;

        algorithms::registration::LinearizedResult last_icp;
        last_icp.inlier = 0;
        size_t actual_iterations = 0;

        // Undamped normal-equation Hessian of the last solved iteration; used to
        // recover the posterior covariance without the damping term.
        Eigen::Matrix<float, 15, 15> H_undamped = Eigen::Matrix<float, 15, 15>::Zero();
        bool has_H_undamped = false;

        // Optimizer selection shared with the LO registration backend.
        // GN: cheapest — damped solve only (λ escalated when H is ill-conditioned).
        // LM/DOGLEG: step acceptance on the combined cost (weighted robust ICP error
        //            with frozen correspondences + IMU prior Mahalanobis cost), at one
        //            extra device error-evaluation pass per trial step.
        const auto opt_method = this->params_.registration.pipeline.registration.optimization_method;
        const auto& gn_params = this->params_.registration.pipeline.registration.gn;
        const auto& lm_params = this->params_.registration.pipeline.registration.lm;
        const auto& dl_params = this->params_.registration.pipeline.registration.dogleg;
        float lm_lambda = lm_params.init_lambda;
        float trust_region_radius = dl_params.initial_trust_region_radius;
        const auto clamp_radius = [&](float radius) {
            return std::clamp(radius, dl_params.min_trust_region_radius, dl_params.max_trust_region_radius);
        };

        // ---- Step-acceptance cost helpers (LM / dogleg) ----
        // The combined LIO cost evaluated at an arbitrary operating state x:
        //   J(x) = icp_weight · E_icp(x) + ½ · r(x)ᵀ H_imu r(x)
        // where E_icp is the robust ICP error evaluated with the correspondences
        // frozen at the current linearisation point (compute_error_frozen), and the
        // IMU term is the prior Mahalanobis distance (H_imu = P_pred⁻¹ is constant
        // across iterations; only the residual r = x ⊖ x_pred changes).  GN does not
        // need this — it commits the damped step unconditionally — but LM and dogleg
        // accept/reject trial steps by comparing J(x_trial) against J(x_op).
        const auto imu_cost = [&](const imu::State& x) -> float {
            if (!imu_valid) return 0.0f;
            const Eigen::Matrix<float, 15, 1> r = imu::compute_manifold_residual(x_pred, x);
            return 0.5f * r.dot(H_imu * r);
        };

        // First-order bias freeze: solve the full coupled system and then drop the bias
        // increment when the window lacks excitation (update_bias == false).  Because H
        // couples pose/velocity with the bias states, the retained pose/velocity step
        // technically assumes the bias also moves, so this is a slight inconsistency.
        // The iterative re-linearization absorbs most of it and the approximation is
        // empirically stable (PR #177 eval).  Applied to every trial step so that the
        // state evaluated by the LM/dogleg acceptance test matches the state retracted
        // on acceptance.  A fully consistent freeze would, when !update_bias, zero the
        // bias cross-terms / set the bias block of lio.H to identity and lio.b's bias
        // segment to zero BEFORE the solve, and restore P_post_'s bias block from P_pred
        // (zero cross-covariance) after it.
        const auto apply_bias_freeze = [&](Eigen::Matrix<float, 15, 1>& d) {
            if (!update_bias) {
                d.segment<3>(imu::State::kIdxAccBias).setZero();
                d.segment<3>(imu::State::kIdxGyrBias).setZero();
            }
        };

        for (size_t iter = 0; iter < this->params_.lio.total_iterations; ++iter) {
            ++actual_iterations;

            if (robust.auto_scale && this->params_.lio.total_iterations > 1) {
                const float t = static_cast<float>(iter) / static_cast<float>(this->params_.lio.total_iterations - 1);
                options.robust_scale =
                    std::max(robust.init_scale * std::pow(robust.min_scale / robust.init_scale, t), robust.min_scale);
            }

            const TransformMatrix T_op = state_to_pose(x_op).matrix();
            last_icp = this->registration_->compute_linearized_result(*source, this->submap_->get_submap_point_cloud(),
                                                                      this->submap_->get_submap_kdtree(), T_op,
                                                                      T_initial, options);

            // H_imu = P_pred^{-1} is constant for this frame, so we only recompute
            // the residual r and b_imu = H_imu · r (no LDLT inversion needed).
            if (iter > 0 && imu_valid) {
                imu::compute_imu_gradient(x_pred, x_op, H_imu, b_imu);
            }

            // Combine ICP + IMU into 15×15 normal equations.
            // Reduced chi-squared calibration of the ICP information (same scheme as
            // MapPrior::update): the ICP Hessian assumes unit-variance Mahalanobis
            // residuals, which real data violates.  Calibrate by the actual residual
            // statistics so a poor fit (map mismatch, degeneracy, sensor noise above
            // the model) loosens the ICP factor relative to the IMU prior:
            //   DOF = d·N_inlier − 6        (d = per-point residual dim, 6 = SE(3) params)
            //   s²  = max(1, 2·error/DOF)   (factor 2 cancels the ½ in the robust error;
            //                                clamp ≥ 1 so the factor is never tightened)
            //   w   = 1 / s²
            // Note: error is evaluated at the current annealed robust scale.  Early
            // (coarse-scale) iterations see larger error → stronger downweighting →
            // prediction-driven steps; the calibration relaxes toward w = 1 as the
            // scale shrinks and the fit converges, which matches the GNC intent.
            float icp_weight = 1.0f;
            const float icp_dof = icp_residual_dim * static_cast<float>(last_icp.inlier) - 6.0f;
            if (icp_dof > 0.0f && std::isfinite(last_icp.error) && last_icp.error >= 0.0f) {
                icp_weight = 1.0f / std::max(1.0f, 2.0f * last_icp.error / icp_dof);
            }
            algorithms::lio::LIOLinearizedResult lio;
            algorithms::lio::add_icp_factor(lio, last_icp, x_op.rotation, icp_weight);
            if (imu_valid) {
                algorithms::lio::add_imu_factor(lio, H_imu, b_imu);
            } else {
                const float kReg = this->params_.lio.invalid_regularization_factor;
                lio.H.block<3, 3>(imu::State::kIdxVel, imu::State::kIdxVel) += kReg * Eigen::Matrix3f::Identity();
                lio.H.block<3, 3>(imu::State::kIdxAccBias, imu::State::kIdxAccBias) +=
                    kReg * Eigen::Matrix3f::Identity();
                lio.H.block<3, 3>(imu::State::kIdxGyrBias, imu::State::kIdxGyrBias) +=
                    kReg * Eigen::Matrix3f::Identity();
            }

            // ICP error at a trial state, reusing the correspondences frozen by the
            // compute_linearized_result() call above (same source cloud).  Scaled by the
            // reduced chi-squared icp_weight so the trial cost matches the information
            // baked into lio.H/lio.b.
            const auto icp_cost = [&](const imu::State& x) -> float {
                const auto [e, in] = this->registration_->compute_error_frozen(
                    *source, this->submap_->get_submap_point_cloud(), state_to_pose(x).matrix(), options);
                (void)in;
                return icp_weight * e;
            };

            const Eigen::Matrix<float, 15, 15> I15 = Eigen::Matrix<float, 15, 15>::Identity();
            Eigen::Matrix<float, 15, 1> delta = Eigen::Matrix<float, 15, 1>::Zero();
            // step_accepted: a usable increment was produced and should be retracted.
            //   GN  — true unless the damped solve fails (then we stop: cannot progress).
            //   LM  — true when an inner trial lowered the combined cost.
            //   DL  — true when the gain ratio cleared eta1.
            // A rejected LM/dogleg step leaves x_op untouched and re-linearises next
            // iteration with the adjusted lambda / trust-region radius, so it must NOT
            // run the convergence test (delta == 0 would otherwise falsely converge).
            bool step_accepted = false;
            bool stop = false;

            switch (opt_method) {
                case algorithms::registration::OptimizationMethod::GAUSS_NEWTON: {
                    if (algorithms::lio::solve_ldlt(lio.H + gn_params.lambda * I15, lio.b, delta)) {
                        apply_bias_freeze(delta);
                        step_accepted = true;
                    } else {
                        stop = true;  // ill-conditioned: keep x_op and end the loop
                    }
                    break;
                }
                case algorithms::registration::OptimizationMethod::LEVENBERG_MARQUARDT: {
                    // Damped trials around the fixed linearisation point: accept the first
                    // λ whose step lowers the combined cost (then relax λ), otherwise grow
                    // λ and retry up to max_inner_iterations.  One frozen-correspondence
                    // error pass per trial.
                    const float current_cost = icp_cost(x_op) + imu_cost(x_op);
                    for (size_t inner = 0; inner < lm_params.max_inner_iterations; ++inner) {
                        Eigen::Matrix<float, 15, 1> trial_delta = Eigen::Matrix<float, 15, 1>::Zero();
                        if (algorithms::lio::solve_ldlt(lio.H + lm_lambda * I15, lio.b, trial_delta)) {
                            apply_bias_freeze(trial_delta);
                            const imu::State x_trial = algorithms::lio::retract(x_op, trial_delta);
                            const float trial_cost = icp_cost(x_trial) + imu_cost(x_trial);
                            if (trial_cost <= current_cost) {
                                delta = trial_delta;
                                step_accepted = true;
                                lm_lambda = std::clamp(lm_lambda / lm_params.lambda_factor, lm_params.min_lambda,
                                                       lm_params.max_lambda);
                                break;
                            }
                        }
                        lm_lambda =
                            std::clamp(lm_lambda * lm_params.lambda_factor, lm_params.min_lambda, lm_params.max_lambda);
                    }
                    break;
                }
                case algorithms::registration::OptimizationMethod::POWELL_DOGLEG: {
                    // Trust-region step (GN / Cauchy / dogleg blend) accepted on the gain
                    // ratio rho = actual / predicted cost reduction.  predicted_reduction is
                    // recomputed on the (possibly bias-frozen) step so it stays consistent
                    // with the increment actually evaluated and retracted.
                    const float current_cost = icp_cost(x_op) + imu_cost(x_op);
                    trust_region_radius = clamp_radius(trust_region_radius);
                    const algorithms::registration::DoglegStep<15> dl =
                        algorithms::registration::compute_dogleg_step<15>(lio.H, lio.b, trust_region_radius);
                    Eigen::Matrix<float, 15, 1> trial_delta = dl.p;
                    apply_bias_freeze(trial_delta);
                    const float predicted_reduction =
                        -(lio.b.dot(trial_delta) + 0.5f * trial_delta.dot(lio.H * trial_delta));
                    if (predicted_reduction <= 0.0f) {
                        trust_region_radius = clamp_radius(trust_region_radius * dl_params.gamma_decrease);
                        break;
                    }
                    const imu::State x_trial = algorithms::lio::retract(x_op, trial_delta);
                    const float trial_cost = icp_cost(x_trial) + imu_cost(x_trial);
                    const float rho = (current_cost - trial_cost) / predicted_reduction;
                    if (rho < dl_params.eta1) {
                        trust_region_radius = clamp_radius(trust_region_radius * dl_params.gamma_decrease);
                        break;
                    }
                    delta = trial_delta;
                    step_accepted = true;
                    if (rho > dl_params.eta2 && dl.step_norm >= trust_region_radius * 0.99f) {
                        trust_region_radius = clamp_radius(trust_region_radius * dl_params.gamma_increase);
                    }
                    break;
                }
            }
            H_undamped = lio.H;
            has_H_undamped = true;

            if (step_accepted) {
                x_op = algorithms::lio::retract(x_op, delta);
                if (is_lio_converged(delta)) break;
            } else if (stop) {
                break;
            }
            // LM / dogleg rejection: x_op unchanged; retry next iteration with the
            // updated lambda / trust-region radius.
        }

        // Posterior covariance from the undamped Hessian.  delta is solved with the
        // damping term λI for step control, but including λI in the inverse would
        // understate P_post_ and over-tighten the next window's IMU prior.
        if (has_H_undamped) {
            Eigen::LDLT<Eigen::Matrix<float, 15, 15>> ldlt(H_undamped);
            if (ldlt.info() == Eigen::Success && ldlt.vectorD().minCoeff() > 0.0f) {
                this->P_post_.setIdentity();
                ldlt.solveInPlace(this->P_post_);  // P_post = H⁻¹
            }
            // When H is not positive definite, keep the previous posterior instead of
            // collapsing it to zero (zero would mean perfect confidence at the reset).
        }

        // ---- Update state and reset IMU preintegration ----
        const Eigen::Vector3f prev_position = this->x_.position;
        const Eigen::Matrix3f prev_rotation = this->x_.rotation;
        this->x_.position = x_op.position;
        this->x_.rotation = x_op.rotation;
        this->x_.accel_bias = x_op.accel_bias;
        this->x_.gyro_bias = x_op.gyro_bias;
        clamp_bias_norm(this->x_.accel_bias, this->params_.lio.bias_estimation.max_accel_bias);
        clamp_bias_norm(this->x_.gyro_bias, this->params_.lio.bias_estimation.max_gyro_bias);
        if (this->dt_ > 0.0f) {
            const Eigen::Vector3f v_fd = (this->x_.position - prev_position) / this->dt_;
            const auto c = this->imu_preintegration_->get_corrected(imu::IMUBias{x_op.gyro_bias, x_op.accel_bias});
            Eigen::Vector3f v_finite_difference;
            if (c.dt_total > 1e-6) {
                const Eigen::Matrix3f R_world_imu_prev = prev_rotation * this->params_.imu.T_imu_to_lidar.rotation();
                const Eigen::Vector3f a_world = this->params_.imu.preintegration.gravity +
                                                R_world_imu_prev * c.Delta_v / static_cast<float>(c.dt_total);
                v_finite_difference = v_fd + 0.5f * a_world * this->dt_;
            } else {
                v_finite_difference = v_fd;
            }
            // Blend the IEKF filter velocity with the finite-difference velocity.
            // blend == 1.0 reproduces the legacy pure-FD estimate.
            const float blend = std::clamp(this->params_.lio.velocity_fd_blend, 0.0f, 1.0f);
            this->x_.velocity = (1.0f - blend) * x_op.velocity + blend * v_finite_difference;
        } else {
            this->x_.velocity = x_op.velocity;
        }
        reset_imu_preintegration();

        algorithms::registration::RegistrationResult result;
        result.T = state_to_pose(this->x_);
        result.converged = true;
        result.iterations = actual_iterations;
        result.inlier = last_icp.inlier;
        result.error = last_icp.error;
        return result;
    }

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------
    void initialize() {
        // SYCL queue
        const auto dev =
            sycl_utils::device_selector::select_device(this->params_.device.vendor, this->params_.device.type);
        this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);

        // Point cloud buffers
        this->preprocessed_pc_ = std::make_shared<PointCloudShared>(*this->queue_ptr_);
        this->registration_input_pc_ = std::make_shared<PointCloudShared>(*this->queue_ptr_);

        this->icp_weights_ = std::make_shared<shared_vector<float>>(*this->queue_ptr_->ptr);

        // Initial pose (T_odom_to_lidar in the gravity-aligned odom/world frame).
        // params_.pose.initial = initial_base_link * T_base_link_to_lidar (composed by
        // the ROS node).  IMU initial alignment overwrites the rotation with the
        // gravity-corrected value in apply_initial_alignment().
        this->odom_ = this->params_.pose.initial;
        this->prev_odom_ = this->params_.pose.initial;

        // Point cloud processor
        this->pc_processor_ = std::make_shared<pointcloud_processing::PCProcessor>(
            *this->queue_ptr_, this->params_.scan, this->params_.covariance_estimation, this->params_.imu);

        // Submapping
        {
            this->submap_ = std::make_shared<submapping::Submap>(*this->queue_ptr_, this->params_);
        }

        // Registration (provides KNN + linearization backend)
        {
            auto& reg_params = this->params_.registration.pipeline;
            reg_params.velocity_update.enable = false;  // LIO controls its own update loop
            // In LIO the IMU prior already constrains degenerate pose directions toward
            // the prediction; the ICP degenerate regularization would double-count along
            // the same directions, so disable it unless explicitly requested.
            if (!this->params_.lio.use_icp_degenerate_regularization) {
                reg_params.registration.degenerate_reg.type =
                    algorithms::registration::DegenerateRegularizationType::none;
            }
            this->registration_ =
                std::make_shared<algorithms::registration::Registration>(*this->queue_ptr_, reg_params.registration);
            this->reg_result_ = std::make_shared<algorithms::registration::RegistrationResult>();
        }

        // IMU preintegration.
        // R_world_imu is the IMU's rotation in the odom (world) frame at startup.
        // The first-frame reset_imu_preintegration() will overwrite this with the
        // post-alignment value (gravity-corrected) before any IMU integration runs.
        {
            this->imu_preintegration_ = std::make_shared<imu::IMUPreintegration>(this->params_.imu.preintegration);
            const Eigen::Matrix3f R_world_imu =
                this->params_.pose.initial.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
            this->imu_preintegration_->reset(this->params_.imu.bias);
            this->imu_R_world_at_reset_ = R_world_imu;
            this->imu_v_world_at_reset_ = Eigen::Vector3f::Zero();
        }

        // Initial gravity-aligned alignment estimator
        this->alignment_estimator_ = std::make_shared<imu::InitialAlignmentEstimator>(
            this->params_.imu.initial_alignment, this->params_.imu.preintegration.gravity,
            this->params_.imu.T_imu_to_lidar);

        // Initialize state biases from params so preprocess() uses the correct bias
        // before the first frame's state initialization runs.
        this->x_.accel_bias = this->params_.imu.bias.accel_bias;
        this->x_.gyro_bias = this->params_.imu.bias.gyro_bias;

        this->clear_total_processing_times();
    }

    // -------------------------------------------------------------------------
    // Initial gravity-aligned alignment
    // -------------------------------------------------------------------------

    /// @brief Apply a successful alignment result to the LIO state.
    /// Replaces the navigation-state rotation with the gravity-corrected initial
    /// orientation (user yaw preserved, roll/pitch from gravity), updates the held gyro
    /// bias, and odometry.  The IMU preintegration reset is intentionally left to the
    /// subsequent first-frame block (which calls reset_imu_preintegration() with proper
    /// covariance handling) to avoid a redundant reset that would discard the P_post_
    /// floor adjustments.
    /// @note out.R_gravity_lidar is the gravity-aligned LiDAR rotation with yaw ≈ 0.
    ///       The user-specified yaw (from params_.pose.initial) is layered on the left so
    ///       x_/odom_ stay in the gravity-aligned odom/world frame.
    /// @note Currently only gyro_bias is updated.  If the estimator gains accel_bias
    ///       estimation, extend this method accordingly.
    void apply_initial_alignment(const imu::InitialAlignmentEstimator::Output& out) {
        const float yaw_user = imu::detail::yaw_from_rotation(this->params_.pose.initial.rotation());
        const Eigen::Matrix3f R_odom_lidar =
            Eigen::AngleAxisf(yaw_user, Eigen::Vector3f::UnitZ()).toRotationMatrix() * out.R_gravity_lidar;
        this->x_.rotation = R_odom_lidar;
        this->x_.gyro_bias = out.gyro_bias;
        this->odom_.linear() = R_odom_lidar;
        this->prev_odom_.linear() = R_odom_lidar;
    }

    // -------------------------------------------------------------------------
    // Preprocessing
    // -------------------------------------------------------------------------
    void preprocess(const PointCloudShared::Ptr scan) {
        // LIO passes the current state bias (x_) to deskew, not the static params bias.
        if (this->params_.imu.deskew.enable) {
            auto imu_buf = this->get_imu_buffer();
            const imu::IMUBias current_bias{this->x_.gyro_bias, this->x_.accel_bias};
            this->pc_processor_->deskew_with_imu(*scan, *scan, imu_buf, this->odom_, current_bias, this->x_.velocity);
        }
        this->pc_processor_->prefilter(*scan, *this->preprocessed_pc_);
    }

    void refine_filter(const PointCloudShared::Ptr scan) {
        this->pc_processor_->refine_filter(*scan, this->processing_ctx_);
    }

    void compute_covariances() {
        const auto reg_type = params_.registration.pipeline.registration.reg_type;
        const bool needs_covs = (reg_type == algorithms::registration::RegType::GICP ||
                                 this->params_.registration.pipeline.registration.rotation_constraint.enable ||
                                 this->params_.scan.preprocess.angle_incidence_filter.enable);
        const bool needs_gaussian =
            this->params_.scan.intensity_gaussian.enable && this->preprocessed_pc_->has_intensity();
        const bool needs_local_mean_norm =
            this->params_.scan.intensity_local_mean_norm.enable && this->preprocessed_pc_->has_intensity();

        if (!needs_covs && !needs_gaussian && !needs_local_mean_norm) return;

        this->processing_ctx_ = this->pc_processor_->prepare_context(*this->preprocessed_pc_);
        this->pc_processor_->compute_covariances(*this->preprocessed_pc_, this->processing_ctx_);
    }

    // -------------------------------------------------------------------------
    // Submapping (same as LiDAROdometryPipeline)
    // -------------------------------------------------------------------------
    bool submapping(const algorithms::registration::RegistrationResult& reg_result, double timestamp) {
        const auto reg_pc_ptr = this->registration_input_pc_;
        bool computed_icp_weights = false;
        const size_t total_samples = this->params_.submap.point_random_sampling_num;
        if (reg_pc_ptr->size() > total_samples) {
            // Robust ICP weighted mixed random sampling
            const auto robust_auto_scale = this->params_.registration.pipeline.robust.auto_scale;
            const float robust_scale = robust_auto_scale
                                           ? this->params_.registration.pipeline.robust.min_scale
                                           : this->params_.registration.pipeline.registration.robust.default_scale;
            this->registration_->compute_icp_robust_weights(*reg_pc_ptr, this->submap_->get_submap_point_cloud(),
                                                            this->submap_->get_submap_kdtree(), reg_result.T.matrix(),
                                                            robust_scale, *this->icp_weights_);
            computed_icp_weights = true;
        }

        const float inlier_ratio = reg_pc_ptr->size() > 0
                                       ? static_cast<float>(reg_result.inlier) / static_cast<float>(reg_pc_ptr->size())
                                       : 0.0f;

        if (computed_icp_weights) {
            return this->submap_->add_frame(*this->registration_input_pc_, reg_result, inlier_ratio, timestamp,
                                            this->icp_weights_);
        }
        return this->submap_->add_frame(*this->registration_input_pc_, reg_result, inlier_ratio, timestamp);
    }

    // -------------------------------------------------------------------------
    // Processing-time bookkeeping
    // -------------------------------------------------------------------------
    void clear_current_processing_time() {
        this->current_processing_time_.clear();
        for (const auto& [k, v] : pn_map_) this->current_processing_time_[v] = 0.0;
    }
    void clear_total_processing_times() {
        this->total_processing_times_.clear();
        for (const auto& [k, v] : pn_map_) this->total_processing_times_[v] = {};
    }
    void add_delta_time(ProcessName name, double dt) {
        this->total_processing_times_[pn_map_.at(name)].push_back(dt);
        this->current_processing_time_[pn_map_.at(name)] = dt;
    }
};

}  // namespace lidar_inertial_odometry
}  // namespace pipeline
}  // namespace sycl_points
