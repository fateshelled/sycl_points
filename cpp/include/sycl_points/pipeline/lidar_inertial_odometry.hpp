#pragma once

#include <cmath>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include "sycl_points/algorithms/deskew/imu_deskew.hpp"
#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/filter/intensity_correction.hpp"
#include "sycl_points/algorithms/filter/polar_downsampling.hpp"
#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/algorithms/filter/voxel_downsampling.hpp"
#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/lio/lio_registration.hpp"
#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/pipeline/lidar_inertial_odometry_params.hpp"
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
        {
            double dt_pre = 0.0, dt_aif = 0.0;
            try {
                time_utils::measure_execution([&]() { this->preprocess(scan); }, dt_pre);
                time_utils::measure_execution([&]() { this->angle_incidence_filter(this->preprocessed_pc_); }, dt_aif);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("preprocess: ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::preprocessing, dt_pre + dt_aif);
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

    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_ = nullptr;
    algorithms::filter::VoxelGrid::Ptr voxel_filter_ = nullptr;
    algorithms::filter::PolarGrid::Ptr polar_filter_ = nullptr;
    algorithms::registration::Registration::Ptr registration_ = nullptr;

    algorithms::registration::RegistrationResult::Ptr reg_result_ = nullptr;

    Eigen::Isometry3f prev_odom_;
    Eigen::Isometry3f odom_;

    submapping::Submap::Ptr submap_ = nullptr;

    double last_frame_time_ = -1.0;
    double last_imu_reset_timestamp_ = -1.0;
    float dt_ = -1.0f;

    Parameters params_;
    Eigen::Isometry3f T_lidar_to_imu_;  // cached inverse of params_.imu.T_imu_to_lidar

    // LIO state (LiDAR frame convention)
    imu::State x_;
    // Posterior covariance passed as initial covariance to the next IMU reset window.
    Eigen::Matrix<float, 15, 15> P_post_ = Eigen::Matrix<float, 15, 15>::Zero();

    imu::IMUPreintegration::Ptr imu_preintegration_ = nullptr;
    ;
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
        return delta.segment<3>(imu::State::kIdxRot).norm() < params_.lio.rotation_convergence &&
               delta.segment<3>(imu::State::kIdxPos).norm() < params_.lio.position_convergence;
    }

    void reset_imu_preintegration() {
        const Eigen::Matrix3f R_world_imu = x_.rotation * this->params_.imu.T_imu_to_lidar.rotation();
        this->imu_preintegration_->reset(               //
            {this->x_.accel_bias, this->x_.gyro_bias},  //
            R_world_imu, this->x_.velocity, this->P_post_);
    }

    /// @brief Predict the full 15-DOF state from IMU preintegration.
    imu::State predict_state() const {
        const imu::IMUBias current_bias{this->x_.accel_bias, this->x_.gyro_bias};
        const Eigen::Isometry3f& T_i2l = this->params_.imu.T_imu_to_lidar;

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
        return pred;
    }

    /// @brief Combined LIO Gauss-Newton optimization for one LiDAR frame.
    algorithms::registration::RegistrationResult lio_registration() {
        // ---- Build prior from IMU preintegration ----
        const imu::State x_pred = predict_state();

        // Transform P_pred from IMU error-state to LiDAR error-state.
        // P_pred from preintegration is expressed in IMU body frame [δp_imu, δφ_imu, …],
        // but our LIO state uses LiDAR body frame [δp_lidar, δφ_lidar, …].
        // The Jacobian J is evaluated at x_pred (the prior mean) and is constant per frame.
        const Eigen::Matrix<float, 15, 15> P_pred_lidar = algorithms::lio::transform_covariance_imu_to_lidar(
            this->imu_preintegration_->get_raw().covariance, this->params_.imu.T_imu_to_lidar, x_pred.rotation);

        Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Zero();
        Eigen::Matrix<float, 15, 1> b_imu = Eigen::Matrix<float, 15, 1>::Zero();
        bool imu_valid = imu::compute_imu_hessian_gradient(x_pred, x_pred, P_pred_lidar, H_imu, b_imu);

        // ---- Prepare ICP source (random sampling) ----
        // Respects registration/random_sampling params, same as RegistrationPipeline::align().
        const auto& rs = this->params_.registration.pipeline.random_sampling;
        const PointCloudShared* source = (rs.enable && this->preprocessed_pc_->size() > rs.num)
                                             ? (this->preprocess_filter_->random_sampling(
                                                    *this->preprocessed_pc_, *this->registration_input_pc_, rs.num),
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

            // Robust scale: geometric decay init_scale → min_scale across LIO iterations,
            // mirroring RobustAligner. When auto_scale=false, default_scale is used throughout.
            if (robust.auto_scale && this->params_.lio.max_iterations > 1) {
                const float t = static_cast<float>(iter) / static_cast<float>(this->params_.lio.max_iterations - 1);
                options.robust_scale =
                    std::max(robust.init_scale * std::pow(robust.min_scale / robust.init_scale, t), robust.min_scale);
            }

            // ICP Hessian/gradient at current operating point (SYCL device).
            // T_initial (IMU-predicted pose) is passed for degenerate regularization — consistent
            // with RegistrationPipeline::align() and prevents unbounded degenerate-direction updates.
            const TransformMatrix T_op = state_to_pose(x_op).matrix();
            last_icp = this->registration_->compute_linearized_result(*source, this->submap_->get_submap_point_cloud(),
                                                                      this->submap_->get_submap_kdtree(), T_op,
                                                                      T_initial, options);

            // Update IMU factor at current operating point (skip on first: already computed above)
            if (iter > 0) {
                imu_valid = imu::compute_imu_hessian_gradient(x_pred, x_op, P_pred_lidar, H_imu, b_imu);
            }

            // Combine ICP + IMU into 15×15 normal equations.
            // Pass x_op.rotation so add_icp_factor can rotate the body-frame ICP
            // translation gradient/Hessian into the world-frame LIO position space.
            algorithms::lio::LIOLinearizedResult lio;
            algorithms::lio::add_icp_factor(lio, last_icp, x_op.rotation);
            if (imu_valid) {
                algorithms::lio::add_imu_factor(lio, H_imu, b_imu);
            } else {
                // P_pred is singular: pin velocity and biases to their predicted values so the
                // 15×15 system stays full-rank and ICP can still correct pose.
                const float kReg = this->params_.lio.invalid_regularization_factor;
                lio.H.block<3, 3>(imu::State::kIdxVel, imu::State::kIdxVel) += kReg * Eigen::Matrix3f::Identity();
                lio.H.block<3, 3>(imu::State::kIdxAccBias, imu::State::kIdxAccBias) +=
                    kReg * Eigen::Matrix3f::Identity();
                lio.H.block<3, 3>(imu::State::kIdxGyrBias, imu::State::kIdxGyrBias) +=
                    kReg * Eigen::Matrix3f::Identity();
            }

            // P_post = H⁻¹: posterior covariance for IEKF-style bias accumulation.
            // Reuses the LDLT already factored for delta — negligible extra cost.
            Eigen::Matrix<float, 15, 1> delta;
            if (!algorithms::lio::solve_ldlt(lio, delta, &this->P_post_)) break;

            x_op = algorithms::lio::retract(x_op, delta);

            if (is_lio_converged(delta)) break;
        }

        // ---- Update state and reset IMU preintegration ----
        // Pose and biases come from the LIO optimisation.
        // Velocity is derived from the LiDAR displacement (finite difference) rather than
        // the LDLT solution: the velocity block of the 15×15 system is only weakly
        // constrained by the IMU (large accel noise → small H_imu[v,v]), and the
        // P_pred cross-terms between rotation and velocity can drive large, wrong δv
        // during fast turning.  Displacement-based velocity is simple and always consistent
        // with the pose update.
        const Eigen::Vector3f prev_position = this->x_.position;
        this->x_.position = x_op.position;
        this->x_.rotation = x_op.rotation;
        this->x_.accel_bias = x_op.accel_bias;
        this->x_.gyro_bias = x_op.gyro_bias;
        this->x_.velocity = (this->dt_ > 0.0f) ? (this->x_.position - prev_position) / this->dt_ : x_op.velocity;
        reset_imu_preintegration();

        // Fill RegistrationResult for compatibility with submapping
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
        this->T_lidar_to_imu_ = this->params_.imu.T_imu_to_lidar.inverse();

        // SYCL queue
        const auto dev =
            sycl_utils::device_selector::select_device(this->params_.device.vendor, this->params_.device.type);
        this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);

        // Point cloud buffers
        this->preprocessed_pc_.reset(new PointCloudShared(*this->queue_ptr_));
        this->registration_input_pc_.reset(new PointCloudShared(*this->queue_ptr_));

        this->icp_weights_ = std::make_shared<shared_vector<float>>(*this->queue_ptr_->ptr);

        // Initial pose
        this->odom_ = this->params_.pose.initial;
        this->prev_odom_ = this->params_.pose.initial;

        // Filters
        this->preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(*this->queue_ptr_);
        if (this->params_.scan.downsampling.voxel.enable) {
            this->voxel_filter_ = std::make_shared<algorithms::filter::VoxelGrid>(
                *this->queue_ptr_, this->params_.scan.downsampling.voxel.size);
        }
        if (this->params_.scan.downsampling.polar.enable) {
            const auto coord =
                algorithms::coordinate_system_from_string(this->params_.scan.downsampling.polar.coord_system);
            this->polar_filter_ = std::make_shared<algorithms::filter::PolarGrid>(
                *this->queue_ptr_, this->params_.scan.downsampling.polar.distance_size,
                this->params_.scan.downsampling.polar.elevation_size,
                this->params_.scan.downsampling.polar.azimuth_size, coord);
        }

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

        this->clear_total_processing_times();
    }

    // -------------------------------------------------------------------------
    // Preprocessing (same as LiDAROdometryPipeline)
    // -------------------------------------------------------------------------
    void preprocess(const PointCloudShared::Ptr scan) {
        if (this->params_.imu.deskew.enable) {
            const auto imu_buf = this->get_imu_buffer();
            const double scan_start_sec = scan->start_time_ms * 1e-3;
            const Eigen::Matrix3f R_world_imu = this->odom_.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
            algorithms::deskew::deskew_point_cloud_imu(*scan, *scan, imu_buf, scan_start_sec,
                                                       this->params_.imu.T_imu_to_lidar, this->params_.imu.bias,
                                                       this->params_.imu.preintegration, R_world_imu);
        }

        if (this->params_.scan.preprocess.box_filter.enable) {
            this->preprocess_filter_->box_filter(*scan, this->params_.scan.preprocess.box_filter.min,
                                                 this->params_.scan.preprocess.box_filter.max);
        }

        auto input_ptr = scan;
        auto output_ptr = this->preprocessed_pc_;
        bool grid_ds = false;
        if (this->params_.scan.downsampling.polar.enable) {
            grid_ds = true;
            this->polar_filter_->downsampling(*input_ptr, *output_ptr);
            input_ptr = output_ptr = this->preprocessed_pc_;
        }
        if (this->params_.scan.downsampling.voxel.enable) {
            grid_ds = true;
            this->voxel_filter_->downsampling(*input_ptr, *output_ptr);
            input_ptr = output_ptr = this->preprocessed_pc_;
        }
        if (!grid_ds) *this->preprocessed_pc_ = *scan;

        if (this->params_.scan.downsampling.random.enable) {
            this->preprocess_filter_->random_sampling(*this->preprocessed_pc_, *this->preprocessed_pc_,
                                                      this->params_.scan.downsampling.random.num);
        }
        if (this->params_.scan.intensity_correction.enable && this->preprocessed_pc_->has_intensity()) {
            algorithms::intensity_correction::correct_intensity(
                *this->preprocessed_pc_, this->params_.scan.intensity_correction.exp,
                this->params_.scan.intensity_correction.scale, this->params_.scan.intensity_correction.min_intensity,
                this->params_.scan.intensity_correction.max_intensity);
        }
    }

    void angle_incidence_filter(const PointCloudShared::Ptr scan) {
        if (this->params_.scan.preprocess.angle_incidence_filter.enable) {
            this->preprocess_filter_->angle_incidence_filter(
                *scan, this->params_.scan.preprocess.angle_incidence_filter.min_angle,
                this->params_.scan.preprocess.angle_incidence_filter.max_angle);
        }
    }

    void compute_covariances() {
        const auto reg_type = params_.registration.pipeline.registration.reg_type;
        if (reg_type == algorithms::registration::RegType::GICP ||
            this->params_.registration.pipeline.registration.rotation_constraint.enable ||
            this->params_.scan.preprocess.angle_incidence_filter.enable) {
            const auto src_tree = algorithms::knn::KDTree::build(*this->queue_ptr_, *this->preprocessed_pc_);
            auto events = src_tree->knn_search_async(*this->preprocessed_pc_,
                                                     this->params_.covariance_estimation.neighbor_num, knn_result_);
            if (params_.covariance_estimation.m_estimation.enable) {
                events += algorithms::covariance::compute_covariances_with_m_estimation_async(
                    this->knn_result_, *this->preprocessed_pc_, this->params_.covariance_estimation.m_estimation.type,
                    this->params_.covariance_estimation.m_estimation.mad_scale,
                    this->params_.covariance_estimation.m_estimation.min_robust_scale,
                    this->params_.covariance_estimation.m_estimation.max_iterations, events.evs);
            } else {
                events += algorithms::covariance::compute_covariances_async(this->knn_result_, *this->preprocessed_pc_,
                                                                            events.evs);
            }
            events.wait_and_throw();
        }
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

        const float inlier_ratio = static_cast<float>(reg_result.inlier) / static_cast<float>(reg_pc_ptr->size());

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
