#pragma once

#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include "sycl_points/algorithms/deskew/imu_deskew.hpp"
#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/feature/photometric_gradient.hpp"
#include "sycl_points/algorithms/filter/intensity_correction.hpp"
#include "sycl_points/algorithms/filter/polar_downsampling.hpp"
#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/algorithms/filter/voxel_downsampling.hpp"
#include "sycl_points/algorithms/imu/imu_factor.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/lio/lio_registration.hpp"
#include "sycl_points/algorithms/mapping/occupancy_grid_map.hpp"
#include "sycl_points/algorithms/mapping/voxel_hash_map.hpp"
#include "sycl_points/algorithms/registration/registration_pipeline.hpp"
#include "sycl_points/pipeline/lidar_inertial_odometry_params.hpp"
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
    auto get_device_queue() const { return queue_ptr_; }
    const auto& get_error_message() const { return error_message_; }
    const auto& get_current_processing_time() const { return current_processing_time_; }
    const auto& get_total_processing_times() const { return total_processing_times_; }
    const auto& get_odom() const { return odom_; }
    const auto& get_prev_odom() const { return prev_odom_; }
    const auto& get_last_keyframe_pose() const { return last_keyframe_pose_; }
    const auto& get_keyframe_poses() const { return keyframe_poses_; }
    const PointCloudShared& get_preprocessed_point_cloud() const { return *preprocessed_pc_; }
    const PointCloudShared& get_submap_point_cloud() const { return *submap_pc_ptr_; }
    const PointCloudShared& get_keyframe_point_cloud() const { return *keyframe_pc_; }
    const auto& get_registration_result() const { return *reg_result_; }
    const imu::State& get_lio_state() const { return x_; }

    // -------------------------------------------------------------------------
    // IMU measurement feed (thread-safe)
    // -------------------------------------------------------------------------
    void add_imu_measurement(const imu::IMUMeasurement& meas) {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (!meas.accel.allFinite() || !meas.gyro.allFinite()) return;
        if (!imu_buffer_.empty() && meas.timestamp <= imu_buffer_.back().timestamp) return;
        const double latest = meas.timestamp;
        imu_buffer_.push_back(meas);
        while (latest - imu_buffer_.front().timestamp > params_.imu.buffer_duration_sec) {
            imu_buffer_.pop_front();
        }
    }

    std::deque<imu::IMUMeasurement> get_imu_buffer() const {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        return imu_buffer_;
    }

    // -------------------------------------------------------------------------
    // Main process call (one LiDAR frame)
    // -------------------------------------------------------------------------
    ResultType process(const PointCloudShared::Ptr scan, double timestamp) {
        error_message_.clear();

        if (last_frame_time_ > 0.0) {
            const float dt = static_cast<float>(timestamp - last_frame_time_);
            if (dt > 0.0f) {
                dt_ = dt;
            } else {
                error_message_ = "old timestamp";
                return ResultType::old_timestamp;
            }
        }
        clear_current_processing_time();

        // Preprocessing
        {
            double dt_pre = 0.0, dt_aif = 0.0;
            try {
                time_utils::measure_execution([&]() { preprocess(scan); }, dt_pre);
                time_utils::measure_execution([&]() { angle_incidence_filter(preprocessed_pc_); }, dt_aif);
            } catch (const std::exception& e) {
                error_message_ = std::string("preprocess: ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << error_message_ << std::endl;
                return ResultType::error;
            }
            add_delta_time(ProcessName::preprocessing, dt_pre + dt_aif);
        }

        // Covariance estimation
        {
            double dt_cov = 0.0;
            try {
                time_utils::measure_execution([&]() { compute_covariances(); }, dt_cov);
            } catch (const std::exception& e) {
                error_message_ = std::string("compute_covariances: ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << error_message_ << std::endl;
                return ResultType::error;
            }
            add_delta_time(ProcessName::compute_covariances, dt_cov);
        }

        if (preprocessed_pc_->size() <= params_.registration.min_num_points) {
            error_message_ = "point cloud size is too small";
            return ResultType::small_number_of_points;
        }

        // Integrate IMU measurements for this window
        {
            imu_batch_.clear();
            std::lock_guard<std::mutex> lock(imu_mutex_);
            imu_batch_.reserve(imu_buffer_.size());
            for (const auto& m : imu_buffer_) {
                if (m.timestamp <= last_imu_reset_timestamp_) continue;
                if (m.timestamp > timestamp) break;
                imu_batch_.push_back(m);
            }
        }
        imu_preintegration_->integrate_batch(imu_batch_);

        // First frame: initialize state and submap, no registration
        if (is_first_frame_) {
            try {
                build_submap(preprocessed_pc_, params_.pose.initial);
            } catch (const std::exception& e) {
                error_message_ = std::string("build_submap (first frame): ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << error_message_ << std::endl;
                return ResultType::error;
            }
            is_first_frame_ = false;
            last_keyframe_time_ = timestamp;
            last_frame_time_ = timestamp;
            last_imu_reset_timestamp_ = timestamp;

            x_.position = params_.pose.initial.translation();
            x_.rotation = params_.pose.initial.rotation();
            x_.velocity = Eigen::Vector3f::Zero();
            x_.accel_bias = params_.imu.bias.accel_bias;
            x_.gyro_bias = params_.imu.bias.gyro_bias;

            odom_ = params_.pose.initial;
            prev_odom_ = params_.pose.initial;
            last_keyframe_pose_ = params_.pose.initial;
            keyframe_poses_.push_back(last_keyframe_pose_);

            reset_imu_preintegration();
            return ResultType::first_frame;
        }

        // LIO registration
        {
            double dt_reg = 0.0;
            try {
                *reg_result_ = time_utils::measure_execution([&]() { return lio_registration(); }, dt_reg);
            } catch (const std::exception& e) {
                error_message_ = std::string("lio_registration: ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << error_message_ << std::endl;
                return ResultType::error;
            }
            add_delta_time(ProcessName::registration, dt_reg);
        }
        last_imu_reset_timestamp_ = timestamp;

        // Submapping
        {
            double dt_sub = 0.0;
            try {
                time_utils::measure_execution([&]() { submapping(*reg_result_, timestamp); }, dt_sub);
            } catch (const std::exception& e) {
                error_message_ = std::string("submapping: ") + e.what();
                std::cerr << "[LidarInertialOdometry] " << error_message_ << std::endl;
                return ResultType::error;
            }
            add_delta_time(ProcessName::build_submap, dt_sub);
        }

        prev_odom_ = odom_;
        odom_ = reg_result_->T;
        last_frame_time_ = timestamp;
        registrated_ = true;

        return ResultType::success;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // -------------------------------------------------------------------------
    // Member variables
    // -------------------------------------------------------------------------
    sycl_utils::DeviceQueue::Ptr queue_ptr_;

    PointCloudShared::Ptr preprocessed_pc_;
    PointCloudShared::Ptr keyframe_pc_;
    PointCloudShared::Ptr submap_pc_ptr_;
    PointCloudShared::Ptr submap_pc_tmp_;
    bool is_first_frame_ = true;
    bool registrated_ = false;

    algorithms::mapping::VoxelHashMap::Ptr submap_voxel_;
    algorithms::mapping::OccupancyGridMap::Ptr occupancy_grid_;
    algorithms::knn::KDTree::Ptr submap_tree_;
    algorithms::knn::KNNResult knn_result_;
    shared_vector_ptr<float> icp_weights_;

    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_;
    algorithms::filter::VoxelGrid::Ptr voxel_filter_;
    algorithms::filter::PolarGrid::Ptr polar_filter_;
    algorithms::registration::RegistrationPipeline::Ptr registration_pipeline_;

    algorithms::registration::RegistrationResult::Ptr reg_result_;

    Eigen::Isometry3f prev_odom_;
    Eigen::Isometry3f odom_;
    Eigen::Isometry3f last_keyframe_pose_;
    std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> keyframe_poses_;

    double last_keyframe_time_ = -1.0;
    double last_frame_time_ = -1.0;
    double last_imu_reset_timestamp_ = -1.0;
    float dt_ = -1.0f;

    Parameters params_;

    // LIO state (LiDAR frame convention)
    imu::State x_;
    // Posterior covariance passed as initial covariance to the next IMU reset window.
    Eigen::Matrix<float, 15, 15> P_post_ = Eigen::Matrix<float, 15, 15>::Zero();

    imu::IMUPreintegration::Ptr imu_preintegration_;
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

    Eigen::Isometry3f state_to_pose(const imu::State& s) const {
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
        const Eigen::Isometry3f& T_i2l = params_.imu.T_imu_to_lidar;
        // R_world_imu = R_world_lidar * R_lidar_imu  (T_imu_to_lidar.rotation() = R_lidar_imu)
        const Eigen::Matrix3f R_world_imu = x_.rotation * T_i2l.rotation();
        imu_preintegration_->reset({x_.accel_bias, x_.gyro_bias}, R_world_imu, x_.velocity, P_post_);
    }

    /// @brief Predict the full 15-DOF state from IMU preintegration.
    imu::State predict_state() const {
        const imu::IMUBias current_bias{x_.accel_bias, x_.gyro_bias};
        const Eigen::Isometry3f& T_i2l = params_.imu.T_imu_to_lidar;

        // Relative pose prediction (gravity + initial velocity already compensated)
        const TransformMatrix T_imu_rel_mat = imu_preintegration_->predict_relative_transform(current_bias);
        Eigen::Isometry3f T_imu_rel = Eigen::Isometry3f::Identity();
        T_imu_rel.linear() = T_imu_rel_mat.block<3, 3>(0, 0);
        T_imu_rel.translation() = T_imu_rel_mat.block<3, 1>(0, 3);

        // Convert IMU-relative transform to LiDAR-relative: T_lidar_rel = T_i2l * T_imu_rel * T_i2l^{-1}
        const Eigen::Isometry3f T_lidar_rel = T_i2l * T_imu_rel * T_i2l.inverse();
        const Eigen::Isometry3f T_pred = state_to_pose(x_) * T_lidar_rel;

        // Velocity prediction in world frame: v_j = v_i + g*dt + R_world_imu * Delta_v
        const auto c = imu_preintegration_->get_corrected(current_bias);
        const Eigen::Matrix3f R_world_imu = x_.rotation * T_i2l.rotation();
        const float dt_f = static_cast<float>(c.dt_total);

        imu::State pred;
        pred.position = T_pred.translation();
        pred.rotation = T_pred.rotation();
        pred.velocity = x_.velocity + params_.imu.preintegration.gravity * dt_f + R_world_imu * c.Delta_v;
        pred.accel_bias = x_.accel_bias;
        pred.gyro_bias = x_.gyro_bias;
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
        const Eigen::Matrix<float, 15, 15> P_pred_lidar =
            algorithms::lio::transform_covariance_imu_to_lidar(imu_preintegration_->get_raw().covariance,
                                                                params_.imu.T_imu_to_lidar, x_pred.rotation);

        Eigen::Matrix<float, 15, 15> H_imu = Eigen::Matrix<float, 15, 15>::Zero();
        Eigen::Matrix<float, 15, 1> b_imu = Eigen::Matrix<float, 15, 1>::Zero();
        bool imu_valid = imu::compute_imu_hessian_gradient(x_pred, x_pred, P_pred_lidar, H_imu, b_imu);

        // ---- Gauss-Newton loop ----
        imu::State x_op = x_pred;

        algorithms::registration::Registration::ExecutionOptions options;
        options.dt = dt_;
        options.prev_pose = odom_.matrix();

        auto* reg = registration_pipeline_->registration().get();

        algorithms::registration::LinearizedResult last_icp;
        last_icp.inlier = 0;
        size_t actual_iterations = 0;

        for (size_t iter = 0; iter < params_.lio.max_iterations; ++iter) {
            ++actual_iterations;

            // ICP Hessian/gradient at current operating point (SYCL device).
            // Pass x_pred pose as initial_pose so degenerate regularization uses the same
            // reference as align() does — this prevents unbounded yaw updates when the
            // rotation Hessian from ICP is near-zero (e.g. during turning).
            const TransformMatrix T_op      = state_to_pose(x_op).matrix();
            const TransformMatrix T_initial = state_to_pose(x_pred).matrix();
            last_icp = reg->compute_linearized_result(*preprocessed_pc_, *submap_pc_ptr_, *submap_tree_,
                                                      T_op, T_initial, options);

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
                // IMU factor unavailable (P_pred singular or zero noise densities).
                // Add weak identity regularization to velocity and bias blocks so the
                // 15×15 system remains full-rank and ICP can still correct the pose.
                // The large information value (small σ) keeps v / b near their predicted values.
                constexpr float kReg = 1e4f;
                lio.H.block<3, 3>(imu::State::kIdxVel,     imu::State::kIdxVel)     += kReg * Eigen::Matrix3f::Identity();
                lio.H.block<3, 3>(imu::State::kIdxAccBias, imu::State::kIdxAccBias) += kReg * Eigen::Matrix3f::Identity();
                lio.H.block<3, 3>(imu::State::kIdxGyrBias, imu::State::kIdxGyrBias) += kReg * Eigen::Matrix3f::Identity();
            }

            // Solve H·δx = −b
            Eigen::Matrix<float, 15, 1> delta;
            if (!algorithms::lio::solve_ldlt(lio, delta)) break;

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
        const Eigen::Vector3f prev_position = x_.position;
        x_.position   = x_op.position;
        x_.rotation   = x_op.rotation;
        x_.accel_bias = x_op.accel_bias;
        x_.gyro_bias  = x_op.gyro_bias;
        x_.velocity   = (dt_ > 0.0f) ? (x_.position - prev_position) / dt_ : x_op.velocity;
        reset_imu_preintegration();

        // Fill RegistrationResult for compatibility with submapping
        algorithms::registration::RegistrationResult result;
        result.T = state_to_pose(x_);
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
        const auto dev = sycl_utils::device_selector::select_device(params_.device.vendor, params_.device.type);
        queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);
        icp_weights_ = std::make_shared<shared_vector<float>>(*queue_ptr_->ptr);

        // Point cloud buffers
        preprocessed_pc_.reset(new PointCloudShared(*queue_ptr_));
        keyframe_pc_.reset(new PointCloudShared(*queue_ptr_));
        submap_pc_ptr_.reset(new PointCloudShared(*queue_ptr_));
        submap_pc_tmp_.reset(new PointCloudShared(*queue_ptr_));

        // Initial pose
        odom_ = params_.pose.initial;
        prev_odom_ = params_.pose.initial;

        // Filters
        preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(*queue_ptr_);
        if (params_.scan.downsampling.voxel.enable) {
            voxel_filter_ =
                std::make_shared<algorithms::filter::VoxelGrid>(*queue_ptr_, params_.scan.downsampling.voxel.size);
        }
        if (params_.scan.downsampling.polar.enable) {
            const auto coord = algorithms::coordinate_system_from_string(params_.scan.downsampling.polar.coord_system);
            polar_filter_ = std::make_shared<algorithms::filter::PolarGrid>(
                *queue_ptr_, params_.scan.downsampling.polar.distance_size,
                params_.scan.downsampling.polar.elevation_size, params_.scan.downsampling.polar.azimuth_size, coord);
        }

        // Submap
        const auto submap_type = params_.submap.map_type;
        if (submap_type == lidar_odometry::SubmapMapType::OCCUPANCY_GRID_MAP) {
            occupancy_grid_ =
                std::make_shared<algorithms::mapping::OccupancyGridMap>(*queue_ptr_, params_.submap.voxel_size);
            occupancy_grid_->set_log_odds_hit(params_.submap.occupancy_grid_map.log_odds_hit);
            occupancy_grid_->set_log_odds_miss(params_.submap.occupancy_grid_map.log_odds_miss);
            occupancy_grid_->set_log_odds_limits(params_.submap.occupancy_grid_map.log_odds_limits_min,
                                                 params_.submap.occupancy_grid_map.log_odds_limits_max);
            occupancy_grid_->set_occupancy_threshold(params_.submap.occupancy_grid_map.occupied_threshold);
            occupancy_grid_->set_free_space_updates_enabled(
                params_.submap.occupancy_grid_map.enable_free_space_updates);
            occupancy_grid_->set_voxel_pruning_enabled(params_.submap.occupancy_grid_map.enable_pruning);
            occupancy_grid_->set_stale_frame_threshold(params_.submap.occupancy_grid_map.stale_frame_threshold);
            occupancy_grid_->set_covariance_aggregation_mode(params_.submap.covariance_aggregation_mode);
        } else {
            submap_voxel_ = std::make_shared<algorithms::mapping::VoxelHashMap>(*queue_ptr_, params_.submap.voxel_size);
            submap_voxel_->set_covariance_aggregation_mode(params_.submap.covariance_aggregation_mode);
        }

        // Registration pipeline (provides KNN + linearization backend)
        {
            auto reg_params = params_.registration.pipeline;
            reg_params.velocity_update.enable = false;  // LIO controls its own update loop
            registration_pipeline_ =
                std::make_shared<algorithms::registration::RegistrationPipeline>(*queue_ptr_, reg_params);
            reg_result_ = std::make_shared<algorithms::registration::RegistrationResult>();
        }

        // IMU preintegration
        imu_preintegration_ = std::make_shared<imu::IMUPreintegration>(params_.imu.preintegration);
        const Eigen::Matrix3f R_world_imu = params_.pose.initial.rotation() * params_.imu.T_imu_to_lidar.rotation();
        imu_preintegration_->reset(params_.imu.bias, R_world_imu);

        clear_total_processing_times();
    }

    // -------------------------------------------------------------------------
    // Preprocessing (same as LiDAROdometryPipeline)
    // -------------------------------------------------------------------------
    void preprocess(const PointCloudShared::Ptr scan) {
        if (params_.imu.deskew.enable) {
            auto imu_buf = get_imu_buffer();
            const double scan_start_sec = scan->start_time_ms * 1e-3;
            const Eigen::Matrix3f R_world_imu = odom_.rotation() * params_.imu.T_imu_to_lidar.rotation();
            algorithms::deskew::deskew_point_cloud_imu(*scan, *scan, imu_buf, scan_start_sec,
                                                       params_.imu.T_imu_to_lidar, params_.imu.bias,
                                                       params_.imu.preintegration, R_world_imu);
        }

        if (params_.scan.preprocess.box_filter.enable) {
            preprocess_filter_->box_filter(*scan, params_.scan.preprocess.box_filter.min,
                                           params_.scan.preprocess.box_filter.max);
        }

        auto input_ptr = scan;
        auto output_ptr = preprocessed_pc_;
        bool grid_ds = false;
        if (params_.scan.downsampling.polar.enable) {
            grid_ds = true;
            polar_filter_->downsampling(*input_ptr, *output_ptr);
            input_ptr = output_ptr = preprocessed_pc_;
        }
        if (params_.scan.downsampling.voxel.enable) {
            grid_ds = true;
            voxel_filter_->downsampling(*input_ptr, *output_ptr);
            input_ptr = output_ptr = preprocessed_pc_;
        }
        if (!grid_ds) *preprocessed_pc_ = *scan;

        if (params_.scan.downsampling.random.enable) {
            preprocess_filter_->random_sampling(*preprocessed_pc_, *preprocessed_pc_,
                                                params_.scan.downsampling.random.num);
        }
        if (params_.scan.intensity_correction.enable && preprocessed_pc_->has_intensity()) {
            algorithms::intensity_correction::correct_intensity(
                *preprocessed_pc_, params_.scan.intensity_correction.exp, params_.scan.intensity_correction.scale,
                params_.scan.intensity_correction.min_intensity, params_.scan.intensity_correction.max_intensity);
        }
    }

    void angle_incidence_filter(const PointCloudShared::Ptr scan) {
        if (params_.scan.preprocess.angle_incidence_filter.enable) {
            preprocess_filter_->angle_incidence_filter(*scan, params_.scan.preprocess.angle_incidence_filter.min_angle,
                                                       params_.scan.preprocess.angle_incidence_filter.max_angle);
        }
    }

    void compute_covariances() {
        const auto reg_type = params_.registration.pipeline.registration.reg_type;
        if (reg_type == algorithms::registration::RegType::GICP ||
            params_.registration.pipeline.registration.rotation_constraint.enable ||
            params_.scan.preprocess.angle_incidence_filter.enable) {
            const auto src_tree = algorithms::knn::KDTree::build(*queue_ptr_, *preprocessed_pc_);
            auto events =
                src_tree->knn_search_async(*preprocessed_pc_, params_.covariance_estimation.neighbor_num, knn_result_);
            if (params_.covariance_estimation.m_estimation.enable) {
                events += algorithms::covariance::compute_covariances_with_m_estimation_async(
                    knn_result_, *preprocessed_pc_, params_.covariance_estimation.m_estimation.type,
                    params_.covariance_estimation.m_estimation.mad_scale,
                    params_.covariance_estimation.m_estimation.min_robust_scale,
                    params_.covariance_estimation.m_estimation.max_iterations, events.evs);
            } else {
                events += algorithms::covariance::compute_covariances_async(knn_result_, *preprocessed_pc_, events.evs);
            }
            events.wait_and_throw();
        }
    }

    // -------------------------------------------------------------------------
    // Submapping (same as LiDAROdometryPipeline)
    // -------------------------------------------------------------------------
    void build_submap(const PointCloudShared::Ptr& cloud, const Eigen::Isometry3f& pose) {
        // Keyframe point cloud: uniform random sampling (no robust weights on first call)
        preprocess_filter_->random_sampling(*cloud, *keyframe_pc_, params_.submap.point_random_sampling_num);

        const auto submap_type = params_.submap.map_type;
        if (submap_type == lidar_odometry::SubmapMapType::OCCUPANCY_GRID_MAP) {
            occupancy_grid_->add_point_cloud(*keyframe_pc_, pose);
            occupancy_grid_->extract_occupied_points(*submap_pc_tmp_, pose, params_.submap.max_distance_range);
        } else {
            submap_voxel_->add_point_cloud(*keyframe_pc_, pose);
            submap_voxel_->downsampling(*submap_pc_tmp_, pose.translation(), params_.submap.max_distance_range);
        }

        if (is_first_frame_) {
            *submap_pc_ptr_ = sycl_points::algorithms::transform::transform_copy(*cloud, pose.matrix());
        } else if (submap_pc_tmp_->size() >= params_.registration.min_num_points) {
            submap_pc_ptr_ = submap_pc_tmp_;
        }

        submap_tree_ = algorithms::knn::KDTree::build(*queue_ptr_, *submap_pc_ptr_);

        const auto reg_type = params_.registration.pipeline.registration.reg_type;
        const bool photometric = params_.registration.pipeline.registration.photometric.enable;
        sycl_utils::events knn_ev;
        bool knn_ready = false;
        auto ensure_knn = [&]() {
            if (!knn_ready) {
                knn_ev = submap_tree_->knn_search_async(*submap_pc_ptr_, params_.covariance_estimation.neighbor_num,
                                                        knn_result_);
                knn_ready = true;
            }
        };

        sycl_utils::events grad_ev;
        if (photometric) {
            if (submap_pc_ptr_->has_rgb()) {
                ensure_knn();
                grad_ev +=
                    algorithms::color_gradient::compute_color_gradients_async(*submap_pc_ptr_, knn_result_, knn_ev.evs);
            } else if (submap_pc_ptr_->has_intensity()) {
                ensure_knn();
                grad_ev += algorithms::intensity_gradient::compute_intensity_gradients_async(*submap_pc_ptr_,
                                                                                             knn_result_, knn_ev.evs);
            }
        }

        sycl_utils::events cov_ev;
        {
            const bool need_cov = reg_type == algorithms::registration::RegType::GICP ||
                                  reg_type == algorithms::registration::RegType::POINT_TO_DISTRIBUTION ||
                                  reg_type == algorithms::registration::RegType::GENZ ||
                                  params_.registration.pipeline.registration.rotation_constraint.enable;
            const bool need_normal = reg_type == algorithms::registration::RegType::POINT_TO_PLANE ||
                                     reg_type == algorithms::registration::RegType::GENZ || photometric;
            const bool has_cov = submap_pc_ptr_->has_cov();

            if (need_normal) {
                ensure_knn();
                if (has_cov) {
                    cov_ev +=
                        algorithms::covariance::compute_normals_from_covariances_async(*submap_pc_ptr_, knn_ev.evs);
                } else {
                    cov_ev += algorithms::covariance::compute_normals_async(knn_result_, *submap_pc_ptr_, knn_ev.evs);
                }
            }
            if (need_cov && !has_cov) {
                ensure_knn();
                cov_ev += algorithms::covariance::compute_covariances_async(knn_result_, *submap_pc_ptr_, knn_ev.evs);
            }
            if (photometric && !need_normal) {
                if (has_cov) {
                    cov_ev +=
                        algorithms::covariance::compute_normals_from_covariances_async(*submap_pc_ptr_, cov_ev.evs);
                } else {
                    ensure_knn();
                    cov_ev += algorithms::covariance::compute_normals_async(knn_result_, *submap_pc_ptr_, knn_ev.evs);
                }
            }
        }
        if (knn_ready) knn_ev.wait_and_throw();
        grad_ev.wait_and_throw();
        cov_ev.wait_and_throw();
    }

    bool submapping(const algorithms::registration::RegistrationResult& result, double timestamp) {
        const auto submap_type = params_.submap.map_type;
        if (submap_type == lidar_odometry::SubmapMapType::OCCUPANCY_GRID_MAP) {
            build_submap(preprocessed_pc_, result.T);
            return true;
        }

        // Keyframe-based voxel map
        if (params_.submap.keyframe.inlier_ratio_threshold > 0.0f && preprocessed_pc_->size() > 0) {
            const float ratio = static_cast<float>(result.inlier) / static_cast<float>(preprocessed_pc_->size());
            if (ratio <= params_.submap.keyframe.inlier_ratio_threshold) return false;
        }

        const auto delta = last_keyframe_pose_.inverse() * result.T;
        const float dist = delta.translation().norm();
        const float angle = std::fabs(Eigen::AngleAxisf(delta.rotation()).angle()) * (180.0f / M_PIf);
        const double dt_kf =
            last_keyframe_time_ > 0.0 ? timestamp - last_keyframe_time_ : std::numeric_limits<double>::max();
        const bool is_kf = dist >= params_.submap.keyframe.distance_threshold ||
                           angle >= params_.submap.keyframe.angle_threshold_degrees ||
                           dt_kf >= params_.submap.keyframe.time_threshold_seconds;
        if (is_kf) {
            last_keyframe_pose_ = result.T;
            last_keyframe_time_ = timestamp;
            keyframe_poses_.push_back(result.T);
            build_submap(preprocessed_pc_, result.T);
            return true;
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Processing-time bookkeeping
    // -------------------------------------------------------------------------
    void clear_current_processing_time() {
        current_processing_time_.clear();
        for (const auto& [k, v] : pn_map_) current_processing_time_[v] = 0.0;
    }
    void clear_total_processing_times() {
        total_processing_times_.clear();
        for (const auto& [k, v] : pn_map_) total_processing_times_[v] = {};
    }
    void add_delta_time(ProcessName name, double dt) {
        total_processing_times_[pn_map_.at(name)].push_back(dt);
        current_processing_time_[pn_map_.at(name)] = dt;
    }
};

}  // namespace lidar_inertial_odometry
}  // namespace pipeline
}  // namespace sycl_points
