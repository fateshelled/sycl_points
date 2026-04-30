#pragma once

#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/algorithms/lio/lio_registration.hpp"
#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/pipeline/lidar_inertial_odometry_params.hpp"
#include "sycl_points/pipeline/pointcloud_processing.hpp"
#include "sycl_points/pipeline/submapping.hpp"
#include "sycl_points/utils/time_utils.hpp"

// ---------------------------------------------------------------------------
// LiDAR-Inertial Odometry Pipeline
//
// Each frame runs a Gauss-Newton loop that combines the ICP Hessian/gradient
// (6×6, from SYCL parallel reduction) with the IMU prior Hessian/gradient
// (15×15) into a unified 15-DOF normal equation solved by LDLT.
//
// State convention: LiDAR body frame in world
//   x_.position  = world-frame position of the LiDAR origin [m]
//   x_.rotation  = R_world_lidar  (SO(3), body-to-world)
//   x_.velocity  = world-frame velocity of the LiDAR body [m/s]
//   x_.accel_bias / gyro_bias = IMU biases (body frame)
//
// NOTE: The ICP Hessian is embedded into the 15-DOF LiDAR-frame error-state.
//       When T_imu_to_lidar ≠ Identity the preintegration covariance P_pred is
//       expressed in the IMU error-state, which introduces a small mismatch.
//       A full adjoint correction (P_lidar = Ad * P_imu * Ad^T) is left for a
//       future update.
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
        error = 100,
        old_timestamp,
        small_number_of_points,
    };

    explicit LidarInertialOdometryPipeline(const Parameters& params) {
        params_ = params;
        params_.imu.enable = true;  // IMU is mandatory for LIO
        initialize();
    }

    // -------------------------------------------------------------------------
    // Getters
    // -------------------------------------------------------------------------
    auto get_device_queue() const { return this->queue_ptr_; }
    const auto& get_error_message() const { return this->error_message_; }
    const auto& get_current_processing_time() const { return this->current_processing_time_; }
    const auto& get_total_processing_times() const { return this->total_processing_times_; }
    const auto& get_odom() const { return this->odom_; }
    const auto& get_prev_odom() const { return this->prev_odom_; }
    const auto& get_last_keyframe_pose() const { return this->submap_->get_last_keyframe_pose(); }
    const auto& get_keyframe_poses() const { return this->submap_->get_keyframe_poses(); }
    const PointCloudShared& get_preprocessed_point_cloud() const { return *this->preprocessed_pc_; }
    const PointCloudShared& get_submap_point_cloud() const { return this->submap_->get_submap_point_cloud(); }
    const PointCloudShared& get_keyframe_point_cloud() const { return this->submap_->get_keyframe_point_cloud(); }
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
                this->submap_->add_first_frame(*this->preprocessed_pc_, timestamp);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("build_submap (first frame): ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->is_first_frame_ = false;
            this->last_frame_time_ = timestamp;
            this->last_imu_reset_timestamp_ = timestamp;

            this->x_.position = this->params_.pose.initial.translation();
            this->x_.rotation = this->params_.pose.initial.rotation();
            this->x_.velocity = Eigen::Vector3f::Zero();
            this->x_.accel_bias = this->params_.imu.bias.accel_bias;
            this->x_.gyro_bias = this->params_.imu.bias.gyro_bias;
            // offset_R_L_I / offset_T_L_I already set in initialize(); keep as-is.

            this->odom_ = this->params_.pose.initial;
            this->prev_odom_ = this->params_.pose.initial;

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

    algorithms::knn::KNNResult knn_result_;
    shared_vector_ptr<float> icp_weights_ = nullptr;

    pointcloud_processing::PCProcessor::Ptr pc_processor_ = nullptr;
    algorithms::registration::Registration::Ptr registration_ = nullptr;

    algorithms::registration::RegistrationResult::Ptr reg_result_ = nullptr;

    Eigen::Isometry3f prev_odom_;
    Eigen::Isometry3f odom_;

    submapping::Submap::Ptr submap_ = nullptr;

    double last_frame_time_ = -1.0;
    double last_imu_reset_timestamp_ = -1.0;
    float dt_ = -1.0f;

    Parameters params_;

    // LIO state (LiDAR frame convention, 21-DOF including extrinsic)
    imu::State x_;
    // Posterior covariance passed as initial covariance to the next IMU reset window. (LiDAR frame, 21×21)
    Eigen::Matrix<float, 21, 21> P_post_ = Eigen::Matrix<float, 21, 21>::Zero();

    imu::IMUPreintegration::Ptr imu_preintegration_ = nullptr;

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

    /// @brief Return the effective LiDAR-IMU extrinsic to use for prediction/reset.
    /// When estimate_extrinsic is enabled, uses the current state x_; otherwise uses the fixed param.
    Eigen::Isometry3f effective_extrinsic() const { return this->effective_extrinsic_from(this->x_); }

    /// @brief Return effective extrinsic from an operating-point state (used inside IEKF loop).
    Eigen::Isometry3f effective_extrinsic_from(const imu::State& x_op) const {
        if (this->params_.lio.estimate_extrinsic) {
            Eigen::Isometry3f T;
            T.setIdentity();
            T.linear() = x_op.offset_R_L_I;
            T.translation() = x_op.offset_T_L_I;
            return T;
        }
        return this->params_.imu.T_imu_to_lidar;
    }

    bool is_lio_converged(const Eigen::Matrix<float, 21, 1>& delta) const {
        return delta.segment<3>(imu::State::kIdxRot).norm() < params_.lio.criteria.rotation &&
               delta.segment<3>(imu::State::kIdxPos).norm() < params_.lio.criteria.translation;
    }

    void reset_imu_preintegration() {
        // Use current state extrinsic when online calibration is enabled
        const Eigen::Isometry3f T_i2l_eff = effective_extrinsic();
        const Eigen::Matrix3f R_world_imu = x_.rotation * T_i2l_eff.rotation();

        // Extract the 15×15 navigation-state covariance block from P_post_ (21×21).
        // The extrinsic block is not passed to IMU preintegration (it doesn't model extrinsic).
        Eigen::Matrix<float, 15, 15> P_initial = this->P_post_.block<15, 15>(0, 0);

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
            algorithms::lio::transform_covariance_lidar_to_imu(P_initial, T_i2l_eff, this->x_.rotation);

        this->imu_preintegration_->reset(               //
            {this->x_.gyro_bias, this->x_.accel_bias},  //
            R_world_imu, this->x_.velocity, P_initial_imu);
    }

    /// @brief Predict the full 21-DOF state from IMU preintegration using the given extrinsic.
    /// @param T_i2l  Extrinsic to use for the prediction (may be from x_ or x_op).
    imu::State predict_state_with(const Eigen::Isometry3f& T_i2l) const {
        const imu::IMUBias current_bias{this->x_.gyro_bias, this->x_.accel_bias};

        // Relative pose prediction (gravity + initial velocity already compensated)
        const TransformMatrix T_imu_rel_mat = this->imu_preintegration_->predict_relative_transform(current_bias);
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
        pred.offset_R_L_I = T_i2l.rotation();
        pred.offset_T_L_I = T_i2l.translation();
        return pred;
    }

    /// @brief Predict from the current state extrinsic (convenience wrapper).
    imu::State predict_state() const { return predict_state_with(effective_extrinsic()); }

    /// @brief Compute the 15×6 Jacobian of the IMU prediction residual w.r.t. the extrinsic [δφ_ex | δt_ex].
    ///
    /// The navigation prediction (position, rotation, velocity) depends on the LiDAR-IMU extrinsic
    /// T_i2l = [R_li | t_li].  This Jacobian captures that coupling so the IEKF cross-terms are correct.
    ///
    /// Rows 0–2  (position):   -x_.rotation * ∂t_pred_rel/∂δex
    /// Rows 3–5  (rotation):    R_li * (I − R_imu_rel^T)   (columns 0–2 only; cols 3–5 = 0)
    /// Rows 6–8  (velocity):    x_.rotation * R_li * [Delta_v]×   (columns 0–2 only)
    /// Rows 9–14 (biases):      0
    Eigen::Matrix<float, 15, 6> compute_extrinsic_jacobian(const imu::State& x_op) const {
        Eigen::Matrix<float, 15, 6> J = Eigen::Matrix<float, 15, 6>::Zero();

        const Eigen::Isometry3f T_i2l = effective_extrinsic_from(x_op);
        const Eigen::Matrix3f R_li = T_i2l.rotation();
        const Eigen::Vector3f t_li = T_i2l.translation();

        const imu::IMUBias current_bias{x_op.gyro_bias, x_op.accel_bias};
        const TransformMatrix T_imu_rel_mat = this->imu_preintegration_->predict_relative_transform(current_bias);
        const Eigen::Matrix3f R_imu_rel = T_imu_rel_mat.block<3, 3>(0, 0);
        const Eigen::Vector3f t_imu_rel = T_imu_rel_mat.block<3, 1>(0, 3);
        const Eigen::Matrix3f R_pred_rel = R_li * R_imu_rel * R_li.transpose();

        // a = R_li^T * t_li  (lever-arm in IMU frame)
        const Eigen::Vector3f a = R_li.transpose() * t_li;

        // ∂t_pred_rel / ∂δφ_ex:
        //   t_pred_rel = (I − R_pred_rel) * t_li + R_li * t_imu_rel
        //   δt_pred_rel|_δφ = R_li * ([R_imu_rel*a]× − R_imu_rel*[a]× − [t_imu_rel]×) * δφ
        const Eigen::Matrix3f dpos_dphi = this->x_.rotation * R_li *
                                          (eigen_utils::lie::skew(Eigen::Vector3f(R_imu_rel * a)) -
                                           R_imu_rel * eigen_utils::lie::skew(Eigen::Vector3f(a)) -
                                           eigen_utils::lie::skew(Eigen::Vector3f(t_imu_rel)));

        // ∂t_pred_rel / ∂δt_ex:
        //   = I − R_pred_rel
        const Eigen::Matrix3f dpos_dt = this->x_.rotation * (Eigen::Matrix3f::Identity() - R_pred_rel);

        J.block<3, 3>(imu::State::kIdxPos, 0) = -dpos_dphi;
        J.block<3, 3>(imu::State::kIdxPos, 3) = -dpos_dt;

        // ∂r_rot / ∂δφ_ex = R_li * (I − R_imu_rel^T)   (δt_ex has no effect)
        J.block<3, 3>(imu::State::kIdxRot, 0) = R_li * (Eigen::Matrix3f::Identity() - R_imu_rel.transpose());

        // ∂r_vel / ∂δφ_ex: velocity prediction uses R_world_imu = x_.rotation * R_li
        //   ∂(R_world_imu * Delta_v) / ∂δφ_ex = x_.rotation * R_li * [Delta_v]×  (negate for residual)
        // Wait: r_vel = x_op.vel − x_pred.vel, and x_pred.vel depends on extrinsic via R_world_imu.
        //   ∂r_vel / ∂δφ_ex = −∂x_pred.vel / ∂δφ_ex = x_.rotation * R_li * [Delta_v]×
        // (sign: ∂(R_li*Exp(δφ)*Delta_v)/∂δφ = −R_li*[Delta_v]× → world: x_.rot*R_li*[Δv]×, negated gives +)
        const auto c = this->imu_preintegration_->get_corrected(current_bias);
        J.block<3, 3>(imu::State::kIdxVel, 0) = this->x_.rotation * R_li * eigen_utils::lie::skew(c.Delta_v);

        return J;
    }

    /// @brief Combined LIO Gauss-Newton optimization for one LiDAR frame.
    algorithms::registration::RegistrationResult lio_registration() {
        // ---- Build prior from IMU preintegration ----
        imu::State x_pred = predict_state();

        // Build 21×21 P_pred:
        //   - Upper-left 15×15: IMU preintegration covariance transformed to LiDAR frame.
        //   - Lower-right 6×6:  Extrinsic prior from the previous frame's posterior.
        const Eigen::Isometry3f T_i2l_init = effective_extrinsic();
        const Eigen::Matrix<float, 15, 15> P_nav_lidar = algorithms::lio::transform_covariance_imu_to_lidar(
            this->imu_preintegration_->get_raw().covariance, T_i2l_init, x_pred.rotation);

        Eigen::Matrix<float, 21, 21> P_pred_21 = Eigen::Matrix<float, 21, 21>::Zero();
        P_pred_21.block<15, 15>(0, 0) = P_nav_lidar;
        P_pred_21.block<6, 6>(imu::State::kIdxExRot, imu::State::kIdxExRot) =
            this->P_post_.block<6, 6>(imu::State::kIdxExRot, imu::State::kIdxExRot);

        // When extrinsic estimation is disabled, pin the extrinsic with a large stiffness
        // so the 21×21 system behaves identically to the original 15-DOF system.
        if (!this->params_.lio.estimate_extrinsic) {
            constexpr float kPin = 1e6f;
            P_pred_21.block<6, 6>(imu::State::kIdxExRot, imu::State::kIdxExRot) =
                (1.0f / kPin) * Eigen::Matrix<float, 6, 6>::Identity();
        }

        Eigen::Matrix<float, 21, 21> H_imu = Eigen::Matrix<float, 21, 21>::Zero();
        Eigen::Matrix<float, 21, 1> b_imu = Eigen::Matrix<float, 21, 1>::Zero();
        bool imu_valid = imu::compute_imu_hessian_gradient(x_pred, x_pred, P_pred_21, H_imu, b_imu);

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

        algorithms::registration::LinearizedResult last_icp;
        last_icp.inlier = 0;
        size_t actual_iterations = 0;

        for (size_t iter = 0; iter < this->params_.lio.max_iterations; ++iter) {
            ++actual_iterations;

            if (robust.auto_scale && this->params_.lio.max_iterations > 1) {
                const float t = static_cast<float>(iter) / static_cast<float>(this->params_.lio.max_iterations - 1);
                options.robust_scale =
                    std::max(robust.init_scale * std::pow(robust.min_scale / robust.init_scale, t), robust.min_scale);
            }

            const TransformMatrix T_op = state_to_pose(x_op).matrix();
            last_icp = this->registration_->compute_linearized_result(*source, this->submap_->get_submap_point_cloud(),
                                                                      this->submap_->get_submap_kdtree(), T_op,
                                                                      T_initial, options);

            // Update gradient b_imu at the current operating point.
            // H_imu = P_pred_21^{-1} is constant for this frame, so we only recompute
            // the residual r and b_imu = H_imu · r (no LDLT inversion needed).
            // When extrinsic estimation is enabled, also re-linearise the navigation
            // prediction using the current x_op extrinsic — but keep x_pred's extrinsic
            // fixed to the prior mean so that the extrinsic prior term is correctly enforced.
            if (iter > 0 && imu_valid) {
                if (this->params_.lio.estimate_extrinsic) {
                    const imu::State x_nav_pred = predict_state_with(effective_extrinsic_from(x_op));
                    x_pred.position = x_nav_pred.position;
                    x_pred.rotation = x_nav_pred.rotation;
                    x_pred.velocity = x_nav_pred.velocity;
                    // x_pred.offset_R_L_I / offset_T_L_I stay as the prior mean
                }
                imu::compute_imu_gradient(x_pred, x_op, H_imu, b_imu);
            }

            // Combine ICP + IMU into 21×21 normal equations.
            // When extrinsic estimation is enabled, add cross-terms that make the
            // extrinsic observable through the navigation-state residual. The copies
            // are only made in that case to avoid the 21×21 allocation on the common path.
            algorithms::lio::LIOLinearizedResult lio;
            algorithms::lio::add_icp_factor(lio, last_icp, x_op.rotation);
            if (imu_valid) {
                if (this->params_.lio.estimate_extrinsic) {
                    Eigen::Matrix<float, 21, 21> H_cross = H_imu;
                    Eigen::Matrix<float, 21, 1> b_cross = b_imu;
                    const Eigen::Matrix<float, 15, 6> J_nav_ex = compute_extrinsic_jacobian(x_op);
                    const Eigen::Matrix<float, 15, 15> Omega_nav = H_imu.block<15, 15>(0, 0);
                    const Eigen::Matrix<float, 15, 6> OJ = Omega_nav * J_nav_ex;
                    H_cross.block<15, 6>(0, imu::State::kIdxExRot) += OJ;
                    H_cross.block<6, 15>(imu::State::kIdxExRot, 0) += OJ.transpose();
                    H_cross.block<6, 6>(imu::State::kIdxExRot, imu::State::kIdxExRot) += J_nav_ex.transpose() * OJ;
                    b_cross.segment<6>(imu::State::kIdxExRot) += J_nav_ex.transpose() * b_imu.segment<15>(0);
                    algorithms::lio::add_imu_factor(lio, H_cross, b_cross);
                } else {
                    algorithms::lio::add_imu_factor(lio, H_imu, b_imu);
                }
            } else {
                const float kReg = this->params_.lio.invalid_regularization_factor;
                lio.H.block<3, 3>(imu::State::kIdxVel, imu::State::kIdxVel) += kReg * Eigen::Matrix3f::Identity();
                lio.H.block<3, 3>(imu::State::kIdxAccBias, imu::State::kIdxAccBias) +=
                    kReg * Eigen::Matrix3f::Identity();
                lio.H.block<3, 3>(imu::State::kIdxGyrBias, imu::State::kIdxGyrBias) +=
                    kReg * Eigen::Matrix3f::Identity();
                // Always pin extrinsic when IMU is invalid — ICP provides no extrinsic information.
                constexpr float kPin = 1e6f;
                lio.H.block<6, 6>(imu::State::kIdxExRot, imu::State::kIdxExRot) +=
                    kPin * Eigen::Matrix<float, 6, 6>::Identity();
            }

            Eigen::Matrix<float, 21, 1> delta;
            const float lambda = this->params_.registration.pipeline.registration.gn.lambda;
            if (!algorithms::lio::solve_ldlt(lio.H + lambda * Eigen::Matrix<float, 21, 21>::Identity(), lio.b, delta,
                                             &this->P_post_))
                break;

            x_op = algorithms::lio::retract(x_op, delta);

            if (is_lio_converged(delta)) break;
        }

        // ---- Update state and reset IMU preintegration ----
        const Eigen::Vector3f prev_position = this->x_.position;
        const Eigen::Matrix3f prev_rotation = this->x_.rotation;
        this->x_.position = x_op.position;
        this->x_.rotation = x_op.rotation;
        this->x_.accel_bias = x_op.accel_bias;
        this->x_.gyro_bias = x_op.gyro_bias;
        if (this->params_.lio.estimate_extrinsic) {
            this->x_.offset_R_L_I = x_op.offset_R_L_I;
            this->x_.offset_T_L_I = x_op.offset_T_L_I;
        }
        if (this->dt_ > 0.0f) {
            const Eigen::Vector3f v_fd = (this->x_.position - prev_position) / this->dt_;
            const auto c = this->imu_preintegration_->get_corrected(imu::IMUBias{x_op.gyro_bias, x_op.accel_bias});
            if (c.dt_total > 1e-6) {
                const Eigen::Matrix3f R_world_imu_prev = prev_rotation * effective_extrinsic_from(x_op).rotation();
                const Eigen::Vector3f a_world = this->params_.imu.preintegration.gravity +
                                                R_world_imu_prev * c.Delta_v / static_cast<float>(c.dt_total);
                this->x_.velocity = v_fd + 0.5f * a_world * this->dt_;
            } else {
                this->x_.velocity = v_fd;
            }
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

        // Initial pose
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
            this->registration_ =
                std::make_shared<algorithms::registration::Registration>(*this->queue_ptr_, reg_params.registration);
            this->reg_result_ = std::make_shared<algorithms::registration::RegistrationResult>();
        }

        // IMU preintegration
        {
            this->imu_preintegration_ = std::make_shared<imu::IMUPreintegration>(this->params_.imu.preintegration);
            const Eigen::Matrix3f R_world_imu =
                this->params_.pose.initial.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
            this->imu_preintegration_->reset(this->params_.imu.bias, R_world_imu);
        }

        // Initialize state biases from params so preprocess() uses the correct bias
        // before the first frame's state initialization runs.
        this->x_.accel_bias = this->params_.imu.bias.accel_bias;
        this->x_.gyro_bias = this->params_.imu.bias.gyro_bias;
        this->x_.offset_R_L_I = this->params_.imu.T_imu_to_lidar.rotation();
        this->x_.offset_T_L_I = this->params_.imu.T_imu_to_lidar.translation();

        // Initialize P_post_ extrinsic blocks with the initial uncertainty from params.
        // The navigation blocks start at zero; the extrinsic blocks hold the prior sigma.
        if (this->params_.lio.estimate_extrinsic) {
            const float sr2 = this->params_.lio.extrinsic_rotation_sigma * this->params_.lio.extrinsic_rotation_sigma;
            const float st2 =
                this->params_.lio.extrinsic_translation_sigma * this->params_.lio.extrinsic_translation_sigma;
            this->P_post_.block<3, 3>(imu::State::kIdxExRot, imu::State::kIdxExRot) = sr2 * Eigen::Matrix3f::Identity();
            this->P_post_.block<3, 3>(imu::State::kIdxExTrans, imu::State::kIdxExTrans) =
                st2 * Eigen::Matrix3f::Identity();
        }

        this->clear_total_processing_times();
    }

    // -------------------------------------------------------------------------
    // Preprocessing
    // -------------------------------------------------------------------------
    void preprocess(const PointCloudShared::Ptr scan) {
        // LIO passes the current state bias (x_) to deskew, not the static params bias.
        if (this->params_.imu.deskew.enable) {
            auto imu_buf = this->get_imu_buffer();
            const imu::IMUBias current_bias{this->x_.gyro_bias, this->x_.accel_bias};
            this->pc_processor_->deskew_with_imu(*scan, *scan, imu_buf, this->odom_, current_bias);
        }
        this->pc_processor_->prefilter(*scan, *this->preprocessed_pc_);
    }

    void refine_filter(const PointCloudShared::Ptr scan) {
        this->pc_processor_->refine_filter(*scan, this->knn_result_);
    }

    void compute_covariances() {
        const auto reg_type = params_.registration.pipeline.registration.reg_type;
        const bool needs_covs = (reg_type == algorithms::registration::RegType::GICP ||
                                 this->params_.registration.pipeline.registration.rotation_constraint.enable ||
                                 this->params_.scan.preprocess.angle_incidence_filter.enable);
        const bool needs_zscore = this->params_.scan.intensity_zscore.enable && this->preprocessed_pc_->has_intensity();

        if (!needs_covs && !needs_zscore) return;

        this->knn_result_ = this->pc_processor_->compute_covariances(*this->preprocessed_pc_);
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
