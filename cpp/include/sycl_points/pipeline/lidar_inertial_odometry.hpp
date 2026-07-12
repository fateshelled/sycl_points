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
#include "sycl_points/pipeline/lidar_inertial_odometry_params.hpp"
#include "sycl_points/pipeline/pointcloud_processing.hpp"
#include "sycl_points/pipeline/submapping.hpp"
#include "sycl_points/utils/time_utils.hpp"

// ---------------------------------------------------------------------------
// LiDAR-Inertial Odometry Pipeline
//
// Each frame prepares the predicted state, covariance, and ICP input, then
// delegates the tightly-coupled optimization loop to algorithms::lio::LIORegistration.
// This pipeline owns sensor buffering, preprocessing, state application, and
// submap updates; the ICP+IMU optimizer lives entirely under algorithms/lio.
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
        imu_only,
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

        const bool insufficient_points = this->preprocessed_pc_->size() <= this->params_.registration.min_num_points;

        // The first LiDAR frame initializes the submap and frame convention, so it
        // cannot fall back to IMU-only propagation.  Keep this check before IMU
        // integration to avoid advancing the preintegration state on a rejected
        // first frame.
        if (this->is_first_frame_ && insufficient_points) {
            this->error_message_ = "point cloud size is too small";
            return ResultType::small_number_of_points;
        }

        this->integrate_imu_window(timestamp);

        if (insufficient_points) {
            return this->process_imu_only(timestamp);
        }

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
                *this->reg_result_ = time_utils::measure_execution([&]() { return this->register_frame(); }, dt_reg);
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
    /// Point cloud actually used as the ICP source in the last register_frame() call.
    /// Points either to registration_input_pc_ (when random sampling ran) or to
    /// preprocessed_pc_ (when sampling was skipped).  submapping() must add this same
    /// cloud to the map; using registration_input_pc_ unconditionally would feed a
    /// stale/empty cloud whenever sampling was skipped.
    PointCloudShared::ConstPtr registration_source_pc_ = nullptr;
    bool is_first_frame_ = true;
    imu::InitialAlignmentEstimator::Ptr alignment_estimator_ = nullptr;
    pointcloud_processing::ProcessingContext processing_ctx_;

    shared_vector_ptr<float> icp_weights_ = nullptr;

    pointcloud_processing::PCProcessor::Ptr pc_processor_ = nullptr;
    algorithms::lio::LIORegistration::Ptr lio_registration_ = nullptr;

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
        const float sv2 = this->params_.lio.preintegration_reset.fd_velocity_sigma *
                          this->params_.lio.preintegration_reset.fd_velocity_sigma;
        P_initial.block<3, 3>(imu::State::kIdxVel, imu::State::kIdxVel) += sv2 * Eigen::Matrix3f::Identity();
        const float sr2 = this->params_.lio.preintegration_reset.icp_rotation_sigma *
                          this->params_.lio.preintegration_reset.icp_rotation_sigma;
        P_initial.block<3, 3>(imu::State::kIdxRot, imu::State::kIdxRot) += sr2 * Eigen::Matrix3f::Identity();

        // P_post_ uses LiDAR right-rotation error; IMUPreintegration uses IMU
        // right-rotation error. Position and velocity remain in the world frame.
        // Convert before passing so that get_raw().covariance → transform_covariance_imu_to_lidar
        // yields the correct P_pred_lidar without double-transformation.
        const Eigen::Matrix<float, 15, 15> P_initial_imu =
            algorithms::lio::transform_covariance_lidar_to_imu(P_initial, T_i2l, this->x_.rotation);

        this->imu_preintegration_->reset({this->x_.gyro_bias, this->x_.accel_bias}, P_initial_imu, R_world_imu);
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

    void integrate_imu_window(double timestamp) {
        this->imu_batch_.clear();
        {
            std::lock_guard<std::mutex> lock(this->imu_mutex_);
            this->imu_batch_.reserve(this->imu_buffer_.size());
            imu::build_measurement_window(this->imu_buffer_, this->last_imu_reset_timestamp_, timestamp,
                                          this->imu_batch_);
        }
        this->imu_preintegration_->integrate_batch(this->imu_batch_);
    }

    ResultType process_imu_only(double timestamp) {
        const imu::State predicted_state = this->predict_state();
        const Eigen::Matrix<float, 15, 15> predicted_covariance = algorithms::lio::transform_covariance_imu_to_lidar(
            this->imu_preintegration_->get_raw().covariance, this->params_.imu.T_imu_to_lidar,
            predicted_state.rotation);

        if (!predicted_state.position.allFinite() || !predicted_state.rotation.allFinite() ||
            !predicted_state.velocity.allFinite() || !predicted_covariance.allFinite()) {
            this->error_message_ = "imu-only propagation produced non-finite state or covariance";
            return ResultType::error;
        }

        // Accept the IMU prediction as the posterior because no LiDAR measurement
        // update is available for this frame.
        this->prev_odom_ = this->odom_;
        this->x_ = predicted_state;
        this->P_post_ = predicted_covariance;
        this->odom_ = state_to_pose(this->x_);

        // Do not leave the previous ICP result visible as the latest result.
        this->reg_result_->T = this->odom_;
        this->reg_result_->converged = true;
        this->reg_result_->iterations = 0;
        this->reg_result_->H.setZero();
        this->reg_result_->b.setZero();
        this->reg_result_->error = 0.0f;
        this->reg_result_->H_raw.setZero();
        this->reg_result_->b_raw.setZero();
        this->reg_result_->error_raw = 0.0f;
        this->reg_result_->inlier = 0;

        this->last_frame_time_ = timestamp;
        this->last_imu_reset_timestamp_ = timestamp;
        this->reset_imu_preintegration();

        this->error_message_ = "point cloud size is too small; propagated with IMU only";
        return ResultType::imu_only;
    }

    /// @brief Prepare one frame and delegate optimization to algorithms::lio.
    algorithms::registration::RegistrationResult register_frame() {
        const imu::State predicted_state = predict_state();
        const Eigen::Matrix<float, 15, 15> predicted_covariance = algorithms::lio::transform_covariance_imu_to_lidar(
            this->imu_preintegration_->get_raw().covariance, this->params_.imu.T_imu_to_lidar,
            predicted_state.rotation);

        const auto& sampling = this->params_.registration_sampling;
        PointCloudShared::ConstPtr source =
            (sampling.enable && this->preprocessed_pc_->size() > sampling.num)
                ? (this->pc_processor_->random_sampling(*this->preprocessed_pc_, *this->registration_input_pc_,
                                                        sampling.num),
                   this->registration_input_pc_)
                : this->preprocessed_pc_;
        this->registration_source_pc_ = source;

        auto result = this->lio_registration_->align(
            *source, this->submap_->get_submap_point_cloud(), this->submap_->get_submap_kdtree(), predicted_state,
            predicted_covariance, this->P_post_, this->imu_bias_observable(), this->dt_, this->odom_.matrix());

        this->P_post_ = result.posterior_covariance;
        this->x_ = result.state;
        clamp_bias_norm(this->x_.accel_bias, this->params_.lio.bias_estimation.max_accel_bias);
        clamp_bias_norm(this->x_.gyro_bias, this->params_.lio.bias_estimation.max_gyro_bias);
        reset_imu_preintegration();
        return result.registration_result;
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

        // Tightly-coupled LIO registration algorithm.
        {
            this->lio_registration_ = std::make_shared<algorithms::lio::LIORegistration>(
                *this->queue_ptr_, this->params_.registration.factor, this->params_.lio.registration);
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
            this->imu_preintegration_->reset(this->params_.imu.bias, Eigen::Matrix<float, 15, 15>::Zero(), R_world_imu);
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
        const auto reg_type = params_.registration.factor.reg_type;
        const bool needs_covs = (reg_type == algorithms::registration::RegType::GICP ||
                                 this->params_.registration.factor.rotation_constraint.enable ||
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
        // Add the same cloud that was used as the ICP source.  This is
        // registration_input_pc_ when random sampling ran, or preprocessed_pc_ when it
        // was skipped; registration_source_pc_ was set by register_frame().
        PointCloudShared::ConstPtr reg_pc_ptr =
            this->registration_source_pc_ != nullptr ? this->registration_source_pc_ : this->registration_input_pc_;
        bool computed_icp_weights = false;
        const size_t total_samples = this->params_.submap.point_random_sampling_num;
        if (reg_pc_ptr->size() > total_samples) {
            // Robust ICP weighted mixed random sampling
            const auto& robust = this->params_.lio.registration.robust;
            const auto robust_auto_scale = robust.auto_scale;
            const float robust_scale =
                robust_auto_scale ? robust.min_scale : this->params_.registration.factor.robust.default_scale;
            this->lio_registration_->registration_backend()->compute_icp_robust_weights(
                *reg_pc_ptr, this->submap_->get_submap_point_cloud(), this->submap_->get_submap_kdtree(),
                reg_result.T.matrix(), robust_scale, *this->icp_weights_);
            computed_icp_weights = true;
        }

        const float inlier_ratio = reg_pc_ptr->size() > 0
                                       ? static_cast<float>(reg_result.inlier) / static_cast<float>(reg_pc_ptr->size())
                                       : 0.0f;

        if (computed_icp_weights) {
            return this->submap_->add_frame(*reg_pc_ptr, reg_result, inlier_ratio, timestamp, this->icp_weights_);
        }
        return this->submap_->add_frame(*reg_pc_ptr, reg_result, inlier_ratio, timestamp);
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
