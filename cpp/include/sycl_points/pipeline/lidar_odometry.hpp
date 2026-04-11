#pragma once

#include <deque>
#include <map>
#include <mutex>

#include "sycl_points/algorithms/deskew/imu_deskew.hpp"
#include "sycl_points/algorithms/deskew/relative_pose_deskew.hpp"
#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/feature/photometric_gradient.hpp"
#include "sycl_points/algorithms/filter/intensity_correction.hpp"
#include "sycl_points/algorithms/filter/polar_downsampling.hpp"
#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/algorithms/filter/voxel_downsampling.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/mapping/occupancy_grid_map.hpp"
#include "sycl_points/algorithms/mapping/voxel_hash_map.hpp"
#include "sycl_points/algorithms/registration/registration_pipeline.hpp"
#include "sycl_points/imu/imu_preintegration.hpp"
#include "sycl_points/pipeline/lidar_odometry_params.hpp"
#include "sycl_points/utils/time_utils.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_odometry {
using LidarOdometryParams = lidar_odometry::Parameters;

class LiDAROdometryPipeline {
public:
    using Ptr = std::shared_ptr<LiDAROdometryPipeline>;
    using ConstPtr = std::shared_ptr<const LiDAROdometryPipeline>;

    enum class ResultType : std::int8_t {
        success = 0,  //
        first_frame,
        error = 100,
        old_timestamp,
        small_number_of_points
    };

    LiDAROdometryPipeline(const LidarOdometryParams& params) {
        this->params_ = params;
        this->initialize();
    }

    auto get_device_queue() const { return this->queue_ptr_; }

    const auto& get_error_message() const { return this->error_message_; }
    const auto& get_current_processing_time() const { return this->current_processing_time_; }
    const auto& get_total_processing_times() const { return this->total_processing_times_; }

    const auto& get_odom() const { return this->odom_; }
    const auto& get_prev_odom() const { return this->prev_odom_; }
    const auto& get_last_keyframe_pose() const { return this->last_keyframe_pose_; }
    const auto& get_keyframe_poses() const { return this->keyframe_poses_; }

    const PointCloudShared& get_preprocessed_point_cloud() const { return *this->preprocessed_pc_; }
    const PointCloudShared& get_submap_point_cloud() const { return *this->submap_pc_ptr_; }
    const PointCloudShared& get_keyframe_point_cloud() const { return *this->keyframe_pc_; }
    const PointCloudShared* get_registration_input_point_cloud() const {
        return this->registration_pipeline_->get_registration_input_point_cloud();
    }

    const auto& get_registration_result() const { return *this->reg_result_; }

    /// @brief Feed a single IMU measurement into the buffer and preintegrator.
    ///        Out-of-order or duplicate timestamps are silently dropped.
    ///        No-op when IMU is disabled (params_.imu.enable == false).
    ///        Buffering always runs when enabled; preintegration runs only if initialized.
    ///        Thread-safe: may be called concurrently with process().
    void add_imu_measurement(const imu::IMUMeasurement& meas) {
        if (!this->params_.imu.enable) return;

        std::lock_guard<std::mutex> lock(imu_mutex_);

        // Drop out-of-order / duplicate timestamps
        if (!this->imu_buffer_.empty() && meas.timestamp <= this->imu_buffer_.back().timestamp) {
            return;
        }

        // Add to buffer and trim entries older than buffer_duration_sec
        const double latest_timestamp = meas.timestamp;
        this->imu_buffer_.push_back(meas);
        while (latest_timestamp - this->imu_buffer_.front().timestamp > this->params_.imu.buffer_duration_sec) {
            this->imu_buffer_.pop_front();
        }

        if (this->imu_preintegration_) {
            this->imu_preintegration_->integrate(meas);
        }
    }

    /// @brief Return a snapshot of the current IMU buffer (for deskewing etc.).
    ///        Thread-safe.
    std::deque<imu::IMUMeasurement, Eigen::aligned_allocator<imu::IMUMeasurement>> get_imu_buffer() const {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        return this->imu_buffer_;
    }

    ResultType process(const PointCloudShared::Ptr scan, double timestamp) {
        this->error_message_.clear();

        if (this->last_frame_time_ > 0.0) {
            const float dt = static_cast<float>(timestamp - this->last_frame_time_);
            if (dt > 0.0f) {
                this->dt_ = dt;
            } else {
                this->error_message_ = "old timestamp";
                return ResultType::old_timestamp;
            }
        }
        this->clear_current_processing_time();

        // preprocess
        double dt_preprocessing = 0.0;
        {
            time_utils::measure_execution([&]() { this->preprocess(scan); }, dt_preprocessing);
        }

        // compute covariances
        {
            double dt_covariance = 0.0;
            time_utils::measure_execution([&]() { compute_covariances(); }, dt_covariance);
            this->add_delta_time(ProcessName::compute_covariances, dt_covariance);
        }

        // angle incidence filter
        {
            double dt_angle_incidence_filter = 0.0;
            time_utils::measure_execution([&]() { this->angle_incidence_filter(this->preprocessed_pc_); },
                                          dt_angle_incidence_filter);
            dt_preprocessing += dt_angle_incidence_filter;
            this->add_delta_time(ProcessName::preprocessing, dt_preprocessing);
        }

        // check point cloud size
        if (this->preprocessed_pc_->size() <= this->params_.registration.min_num_points) {
            this->error_message_ = "point cloud size is too small";
            return ResultType::small_number_of_points;
        }

        // first frame processing
        if (this->is_first_frame_) {
            this->build_submap(this->preprocessed_pc_, this->params_.pose.initial);

            this->is_first_frame_ = false;
            this->last_keyframe_time_ = timestamp;
            this->last_frame_time_ = timestamp;

            // Reset IMU integrator so the next window starts from the initial pose.
            if (this->imu_preintegration_) {
                const Eigen::Matrix3f R_world_imu =
                    this->params_.pose.initial.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
                std::lock_guard<std::mutex> lock(imu_mutex_);
                this->imu_prev_Delta_R_ = Eigen::Matrix3f::Identity();
                this->imu_preintegration_->reset(this->params_.imu.bias, R_world_imu, Eigen::Vector3f::Zero());
            }

            return ResultType::first_frame;
        }

        // Registration
        {
            double dt_registration = 0.0;
            *this->reg_result_ = time_utils::measure_execution([&]() { return registration(); }, dt_registration);
            this->add_delta_time(ProcessName::registration, dt_registration);
        }

        // Submapping
        {
            double dt_build_submap = 0.0;
            time_utils::measure_execution([&]() { return submapping(*this->reg_result_, timestamp); }, dt_build_submap);
            this->add_delta_time(ProcessName::build_submap, dt_build_submap);
        }

        // update Velocity and Odometry
        {
            this->prev_odom_ = this->odom_;
            this->odom_ = this->reg_result_->T;
            this->last_frame_time_ = timestamp;

            const auto delta_pose = this->prev_odom_.inverse() * this->odom_;
            const Eigen::AngleAxisf delta_angle_axis(delta_pose.rotation());

            this->linear_velocity_ = delta_pose.translation() / this->dt_;
            this->angular_velocity_ = Eigen::AngleAxisf(delta_angle_axis.angle() / this->dt_, delta_angle_axis.axis());

            this->registrated_ = true;
        }
        return ResultType::success;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    sycl_utils::DeviceQueue::Ptr queue_ptr_ = nullptr;

    PointCloudShared::Ptr preprocessed_pc_ = nullptr;  // Sensor coordinate
    PointCloudShared::Ptr keyframe_pc_ = nullptr;      // Sensor coordinate
    PointCloudShared::Ptr submap_pc_ptr_ = nullptr;    // Odom/World coordinate
    PointCloudShared::Ptr submap_pc_tmp_ = nullptr;    // Odom/World coordinate
    bool is_first_frame_ = true;

    algorithms::mapping::VoxelHashMap::Ptr submap_voxel_ = nullptr;
    algorithms::mapping::OccupancyGridMap::Ptr occupancy_grid_ = nullptr;
    algorithms::knn::KDTree::Ptr submap_tree_ = nullptr;
    algorithms::knn::KNNResult knn_result_;
    algorithms::knn::KNNResult knn_result_grad_;
    shared_vector_ptr<float> icp_weights_ = nullptr;

    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_ = nullptr;
    algorithms::filter::VoxelGrid::Ptr voxel_filter_ = nullptr;
    algorithms::filter::PolarGrid::Ptr polar_filter_ = nullptr;
    algorithms::registration::RegistrationPipeline::Ptr registration_pipeline_ = nullptr;

    bool registrated_ = false;
    algorithms::registration::RegistrationResult::Ptr reg_result_ = nullptr;

    Eigen::Vector3f linear_velocity_;       // [m/s] in previous LiDAR body frame
    Eigen::Matrix3f imu_prev_Delta_R_;      // Delta_R from the previous IMU integration window
    Eigen::AngleAxisf angular_velocity_;    // [rad/s]
    Eigen::Isometry3f prev_odom_;           // prev T_odom_to_lidar
    Eigen::Isometry3f odom_;                // current T_odom_to_lidar
    Eigen::Isometry3f last_keyframe_pose_;  // keyframe T_odom_to_lidar
    std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> keyframe_poses_;

    double last_keyframe_time_;      // [s]
    double last_frame_time_ = -1.0;  // [s]
    float dt_ = -1.0f;               // [s]

    Parameters params_;

    imu::IMUPreintegration::Ptr imu_preintegration_ = nullptr;
    std::deque<imu::IMUMeasurement, Eigen::aligned_allocator<imu::IMUMeasurement>> imu_buffer_;
    mutable std::mutex imu_mutex_;  ///< Guards imu_preintegration_ and imu_buffer_.

    std::string error_message_;

    enum class ProcessName {
        preprocessing = 0,
        compute_covariances,
        registration,
        build_submap,
    };
    const std::map<ProcessName, std::string> pn_map_ = {
        {ProcessName::preprocessing, "1. preprocessing"},
        {ProcessName::compute_covariances, "2. compute covariances"},
        {ProcessName::registration, "3. registration"},
        {ProcessName::build_submap, "4. build submap"},
    };

    std::map<std::string, double> current_processing_time_;
    std::map<std::string, std::vector<double>> total_processing_times_;
    void clear_current_processing_time() {
        this->current_processing_time_.clear();
        this->current_processing_time_[pn_map_.at(ProcessName::preprocessing)] = 0.0;
        this->current_processing_time_[pn_map_.at(ProcessName::compute_covariances)] = 0.0;
        this->current_processing_time_[pn_map_.at(ProcessName::registration)] = 0.0;
        this->current_processing_time_[pn_map_.at(ProcessName::build_submap)] = 0.0;
    }
    void clear_total_processing_times() {
        this->total_processing_times_.clear();
        this->total_processing_times_[pn_map_.at(ProcessName::preprocessing)] = {};
        this->total_processing_times_[pn_map_.at(ProcessName::compute_covariances)] = {};
        this->total_processing_times_[pn_map_.at(ProcessName::registration)] = {};
        this->total_processing_times_[pn_map_.at(ProcessName::build_submap)] = {};
    }
    void add_delta_time(ProcessName name, double dt) {
        this->total_processing_times_[pn_map_.at(name)].push_back(dt);
        this->current_processing_time_[pn_map_.at(name)] = dt;
    }

    bool is_imu_deskew_enabled() const { return this->params_.imu.enable && this->params_.imu.deskew.enable; }

    void initialize() {
        // SYCL queue
        {
            // const auto device_selector = sycl_utils::device_selector::default_selector_v;
            // sycl::device dev(device_selector);
            const auto dev =
                sycl_utils::device_selector::select_device(this->params_.device.vendor, this->params_.device.type);
            this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);
            this->icp_weights_ = std::make_shared<shared_vector<float>>(*this->queue_ptr_->ptr);
        }

        // initialize buffer
        {
            this->preprocessed_pc_.reset(new PointCloudShared(*this->queue_ptr_));
            this->keyframe_pc_.reset(new PointCloudShared(*this->queue_ptr_));
            this->submap_pc_ptr_.reset(new PointCloudShared(*this->queue_ptr_));
            this->submap_pc_tmp_.reset(new PointCloudShared(*this->queue_ptr_));
        }

        // set Initial pose
        {
            this->odom_ = this->params_.pose.initial;
            this->prev_odom_ = this->params_.pose.initial;

            this->linear_velocity_ = Eigen::Vector3f::Zero();
            this->angular_velocity_ = Eigen::AngleAxisf::Identity();
        }

        // initialize keyframe
        {
            this->last_keyframe_pose_ = this->params_.pose.initial;
            this->last_keyframe_time_ = -1.0;
            this->keyframe_poses_.clear();
            this->keyframe_poses_.push_back(this->last_keyframe_pose_);
        }

        // Point cloud processor
        {
            this->preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(*this->queue_ptr_);
            if (this->params_.scan.downsampling.voxel.enable) {
                this->voxel_filter_ = std::make_shared<algorithms::filter::VoxelGrid>(
                    *this->queue_ptr_, this->params_.scan.downsampling.voxel.size);
            }
            if (this->params_.scan.downsampling.polar.enable) {
                const auto coord_system =
                    algorithms::coordinate_system_from_string(this->params_.scan.downsampling.polar.coord_system);
                this->polar_filter_ = std::make_shared<algorithms::filter::PolarGrid>(
                    *this->queue_ptr_, this->params_.scan.downsampling.polar.distance_size,
                    this->params_.scan.downsampling.polar.elevation_size,
                    this->params_.scan.downsampling.polar.azimuth_size, coord_system);
            }
        }

        // Submapping
        {
            const auto submap_type = this->params_.submap.map_type;
            if (submap_type == SubmapMapType::OCCUPANCY_GRID_MAP) {
                this->occupancy_grid_ = std::make_shared<algorithms::mapping::OccupancyGridMap>(
                    *this->queue_ptr_, this->params_.submap.voxel_size);

                this->occupancy_grid_->set_log_odds_hit(this->params_.submap.occupancy_grid_map.log_odds_hit);
                this->occupancy_grid_->set_log_odds_miss(this->params_.submap.occupancy_grid_map.log_odds_miss);
                this->occupancy_grid_->set_log_odds_limits(this->params_.submap.occupancy_grid_map.log_odds_limits_min,
                                                           this->params_.submap.occupancy_grid_map.log_odds_limits_max);
                this->occupancy_grid_->set_occupancy_threshold(
                    this->params_.submap.occupancy_grid_map.occupied_threshold);
                this->occupancy_grid_->set_free_space_updates_enabled(
                    this->params_.submap.occupancy_grid_map.enable_free_space_updates);
                this->occupancy_grid_->set_voxel_pruning_enabled(
                    this->params_.submap.occupancy_grid_map.enable_pruning);
                this->occupancy_grid_->set_stale_frame_threshold(
                    this->params_.submap.occupancy_grid_map.stale_frame_threshold);
                this->occupancy_grid_->set_covariance_aggregation_mode(
                    this->params_.submap.covariance_aggregation_mode);
            } else {
                this->submap_voxel_ = std::make_shared<algorithms::mapping::VoxelHashMap>(
                    *this->queue_ptr_, this->params_.submap.voxel_size);
                this->submap_voxel_->set_covariance_aggregation_mode(this->params_.submap.covariance_aggregation_mode);
            }
        }
        // Registration
        {
            auto reg_pipeline_params = this->params_.registration.pipeline;
            if (this->is_imu_deskew_enabled() && reg_pipeline_params.velocity_update.enable) {
                std::cerr << "[LiDAR Odometry] VelocityUpdate is disabled because IMU deskew is enabled." << std::endl;
                reg_pipeline_params.velocity_update.enable = false;
            }
            this->registration_pipeline_ = std::make_shared<algorithms::registration::RegistrationPipeline>(
                *this->queue_ptr_, reg_pipeline_params);
            this->reg_result_ = std::make_shared<algorithms::registration::RegistrationResult>();
            this->registrated_ = false;
        }
        // utilities
        {
            this->clear_total_processing_times();
        }

        // IMU preintegration (optional)
        if (this->params_.imu.enable) {
            this->imu_preintegration_ = std::make_shared<imu::IMUPreintegration>(this->params_.imu.preintegration);
            const Eigen::Matrix3f R_world_imu =
                this->params_.pose.initial.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
            this->imu_preintegration_->reset(this->params_.imu.bias, R_world_imu);
        }
    }

    void preprocess(const PointCloudShared::Ptr scan) {
        // Process Order:
        //   IMU deskew -> box filter -> polar grid -> voxel grid -> random sampling -> intensity correct

        // IMU deskew: pre-processing step applied before downsampling and ICP.
        // Brings all points into the sensor frame at scan-start time using
        // per-point IMU integration with gravity compensation.
        if (this->is_imu_deskew_enabled()) {
            auto imu_buf_snapshot = this->get_imu_buffer();
            const double scan_start_sec = scan->start_time_ms * 1e-3;
            // R_world_imu = R_world_lidar * R_lidar_imu
            const Eigen::Matrix3f R_world_imu = this->odom_.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
            algorithms::deskew::deskew_point_cloud_imu(*scan, *scan,  // in-place
                                                       imu_buf_snapshot, scan_start_sec,
                                                       this->params_.imu.T_imu_to_lidar, this->params_.imu.bias,
                                                       this->params_.imu.preintegration, R_world_imu);
            // On failure (insufficient IMU coverage etc.) processing continues without deskewing.
        }

        // Box filter
        if (this->params_.scan.preprocess.box_filter.enable) {
            this->preprocess_filter_->box_filter(*scan, this->params_.scan.preprocess.box_filter.min,
                                                 this->params_.scan.preprocess.box_filter.max);
        }

        // Grid downsampling
        {
            auto input_ptr = scan;
            auto output_ptr = this->preprocessed_pc_;
            bool grid_downsampling = false;
            if (this->params_.scan.downsampling.polar.enable) {
                grid_downsampling = true;
                this->polar_filter_->downsampling(*input_ptr, *output_ptr);
                input_ptr = this->preprocessed_pc_;
                output_ptr = this->preprocessed_pc_;
            }
            if (this->params_.scan.downsampling.voxel.enable) {
                grid_downsampling = true;
                this->voxel_filter_->downsampling(*input_ptr, *output_ptr);
                input_ptr = this->preprocessed_pc_;
                output_ptr = this->preprocessed_pc_;
            }
            if (!grid_downsampling) {
                *this->preprocessed_pc_ = *scan;  // copy
            }
        }

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
        if (this->params_.registration.pipeline.registration.reg_type == algorithms::registration::RegType::GICP ||
            this->params_.registration.pipeline.registration.rotation_constraint.enable ||
            this->params_.scan.preprocess.angle_incidence_filter.enable) {
            // build KDTree
            const auto src_tree = algorithms::knn::KDTree::build(*this->queue_ptr_, *this->preprocessed_pc_);
            auto events = src_tree->knn_search_async(
                *this->preprocessed_pc_, this->params_.covariance_estimation.neighbor_num, this->knn_result_);
            if (this->params_.covariance_estimation.m_estimation.enable) {
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

    /// Predict initial pose by applying the previous motion model
    Eigen::Isometry3f adaptive_motion_prediction() {
        float rot_factor = this->params_.motion_prediction.static_factor;
        float trans_factor = this->params_.motion_prediction.static_factor;

        if (this->registrated_ && this->reg_result_->inlier > 0) {
            if (this->params_.motion_prediction.adaptive.rotation.enable) {
                // Calculates the degeneracy score from the minimum eigenvalue of the Hessian in the registration result
                // of the previous frame.
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_rot(this->reg_result_->H_raw.block<3, 3>(0, 0));
                if (solver_rot.info() == Eigen::Success) {
                    const float low = this->params_.motion_prediction.adaptive.rotation.min_eigenvalue_low;
                    const float high = this->params_.motion_prediction.adaptive.rotation.min_eigenvalue_high;
                    const float max_factor = this->params_.motion_prediction.adaptive.rotation.factor_max;
                    const float min_factor = this->params_.motion_prediction.adaptive.rotation.factor_min;

                    const float min_eig_ratio = solver_rot.eigenvalues().minCoeff() / this->reg_result_->inlier;

                    // score == 1.0: Degenerate, score == 0.0: Non-degenerate
                    const float score = std::clamp((min_eig_ratio - low) / (high - low), 0.0f, 1.0f);

                    // Derive the coefficient from the degeneracy score.
                    rot_factor = max_factor * (1.0f - score) + min_factor * score;

                    if (this->params_.motion_prediction.verbose) {
                        std::cout << "[motion predictor] rot: factor=" << rot_factor << ", eigen value=["
                                  << solver_rot.eigenvalues().transpose() / this->reg_result_->inlier << "]"
                                  << std::endl;
                    }
                }
            }

            if (this->params_.motion_prediction.adaptive.translation.enable) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_trans(this->reg_result_->H_raw.block<3, 3>(3, 3));
                if (solver_trans.info() == Eigen::Success) {
                    const float low = this->params_.motion_prediction.adaptive.translation.min_eigenvalue_low;
                    const float high = this->params_.motion_prediction.adaptive.translation.min_eigenvalue_high;
                    const float max_factor = this->params_.motion_prediction.adaptive.translation.factor_max;
                    const float min_factor = this->params_.motion_prediction.adaptive.translation.factor_min;

                    const float min_eig_ratio = solver_trans.eigenvalues().minCoeff() / this->reg_result_->inlier;
                    const float score = std::clamp((min_eig_ratio - low) / (high - low), 0.0f, 1.0f);
                    trans_factor = max_factor * (1.0f - score) + min_factor * score;
                }
                if (this->params_.motion_prediction.verbose) {
                    std::cout << "[motion predictor] trans: factor=" << trans_factor << ", eigen value=["
                              << solver_trans.eigenvalues().transpose() / this->reg_result_->inlier << "]" << std::endl;
                }
            }
        }

        // Computes the displacement from the previous frame using the velocity derived from previous odometry data,
        // then multiplies this by the coefficient to predict the current position.
        const Eigen::Vector3f delta_trans = this->linear_velocity_ * this->dt_;
        const Eigen::AngleAxisf delta_angle_axis(this->angular_velocity_.angle() * this->dt_,
                                                 this->angular_velocity_.axis());

        const Eigen::Vector3f predicted_trans =
            this->odom_.translation() + this->odom_.rotation() * (delta_trans * trans_factor);
        const Eigen::Quaternionf predicted_rot =
            Eigen::AngleAxisf(delta_angle_axis.angle() * rot_factor, delta_angle_axis.axis()) *
            Eigen::Quaternionf(this->odom_.rotation());

        Eigen::Isometry3f init_T = Eigen::Isometry3f::Identity();
        init_T.translation() = predicted_trans;
        init_T.rotate(predicted_rot.normalized());
        return init_T;
    }

    /// @brief Predict the initial ICP pose using IMU preintegration.
    /// @return Absolute predicted pose T_odom_to_lidar_curr (Isometry3f).
    Eigen::Isometry3f imu_motion_prediction() {
        // T_imu_rel: relative pose in IMU body frame, with gravity and initial velocity
        // already accounted for inside predict_relative_transform().
        const TransformMatrix T_imu_rel = this->imu_preintegration_->predict_relative_transform(this->params_.imu.bias);

        // Convert to LiDAR-frame relative transform:
        // T_lidar_rel = T_imu_to_lidar * T_imu_rel * T_imu_to_lidar^{-1}
        const Eigen::Isometry3f& T_i2l = this->params_.imu.T_imu_to_lidar;
        Eigen::Isometry3f T_imu_rel_iso = Eigen::Isometry3f::Identity();
        T_imu_rel_iso.linear() = T_imu_rel.block<3, 3>(0, 0);
        T_imu_rel_iso.translation() = T_imu_rel.block<3, 1>(0, 3);

        const Eigen::Isometry3f T_lidar_rel = T_i2l * T_imu_rel_iso * T_i2l.inverse();

        return this->odom_ * T_lidar_rel;
    }

    algorithms::registration::RegistrationResult registration() {
        // Snapshot the IMU-based initial guess under the lock, then immediately reset the
        // integrator so the next window starts from this scan's arrival time.  Resetting
        // here (before ICP) ensures:
        //   1. The IMU window covers exactly [t_{k-1}, t_k] with no gap from ICP latency.
        //   2. The velocity used is the window-start velocity (previous frame's result),
        //      not the window-end velocity that would only be available after ICP.
        Eigen::Isometry3f init_T;
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            if (this->imu_preintegration_) {
                if (this->imu_preintegration_->get_dt_total() > 0.0) {
                    init_T = this->imu_motion_prediction();
                } else {
                    init_T = this->adaptive_motion_prediction();
                }
                const Eigen::Matrix3f R_world_imu =
                    this->odom_.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
                // Rotate linear_velocity_ (in previous LiDAR body frame at t_{k-2}) by the
                // IMU-derived Delta_R from the previous window to correct its direction to t_{k-1}.
                // The LiDAR-derived velocity captures the average direction over [t_{k-2}, t_{k-1}],
                // while Delta_R (integrated from gyroscope at 100-1000 Hz) gives the actual
                // rotation that occurred during that window, bringing the direction up to t_{k-1}.
                //
                // Derivation (all expressed in their respective body frames):
                //   v_imu_{k-2} = T_lidar_to_imu * v_lidar_{k-2}   (LiDAR → IMU frame)
                //   v_imu_{k-1} = Delta_R * v_imu_{k-2}             (rotate to t_{k-1} body frame)
                //   v_world     = R_world_imu_{k-1} * v_imu_{k-1}   (to world frame)
                const Eigen::Matrix3f T_lidar_to_imu = this->params_.imu.T_imu_to_lidar.rotation().transpose();
                const Eigen::Vector3f v_world =
                    R_world_imu * this->imu_prev_Delta_R_ * T_lidar_to_imu * this->linear_velocity_;
                // Save Delta_R of this window for the next frame's velocity correction.
                this->imu_prev_Delta_R_ = this->imu_preintegration_->get_raw().Delta_R;
                this->imu_preintegration_->reset(this->params_.imu.bias, R_world_imu, v_world);
            } else {
                init_T = this->adaptive_motion_prediction();
            }
        }

        algorithms::registration::Registration::ExecutionOptions options;
        options.dt = this->dt_;
        options.prev_pose = this->odom_.matrix();

        return this->registration_pipeline_->align(*this->preprocessed_pc_, *this->submap_pc_ptr_, *this->submap_tree_,
                                                   init_T.matrix(), options);
    }

    void build_submap(const PointCloudShared::Ptr& cloud, const Eigen::Isometry3f& current_pose) {
        // sampling
        {
            // If velocity update is disabled, get the registration input point cloud.
            auto reg_pc = this->registration_pipeline_->get_deskewed_point_cloud();
            if (reg_pc) {
                const size_t total_samples = this->params_.submap.point_random_sampling_num;
                if (reg_pc->size() <= total_samples) {
                    *this->keyframe_pc_ = *reg_pc;
                } else {
                    // Robust ICP weighted mixed random sampling
                    const auto robust_auto_scale = this->params_.registration.pipeline.robust.auto_scale;
                    const float robust_scale =
                        robust_auto_scale ? this->params_.registration.pipeline.robust.min_scale
                                          : this->params_.registration.pipeline.registration.robust.default_scale;
                    this->registration_pipeline_->compute_icp_robust_weights(*this->submap_pc_ptr_, *this->submap_tree_,
                                                                             current_pose.matrix(), robust_scale,
                                                                             *this->icp_weights_);
                    this->preprocess_filter_->mixed_random_sampling(*reg_pc, *this->keyframe_pc_, *this->icp_weights_,
                                                                    total_samples,
                                                                    this->params_.submap.weighted_sampling_ratio);
                }
            } else {
                // uniform random sampling
                this->preprocess_filter_->random_sampling(*cloud, *this->keyframe_pc_,
                                                          this->params_.submap.point_random_sampling_num);
            }
        }

        // add to grid map
        const auto submap_type = this->params_.submap.map_type;
        if (submap_type == SubmapMapType::OCCUPANCY_GRID_MAP) {
            this->occupancy_grid_->add_point_cloud(*this->keyframe_pc_, current_pose);
            this->occupancy_grid_->extract_occupied_points(*this->submap_pc_tmp_, current_pose,
                                                           this->params_.submap.max_distance_range);
        } else {
            this->submap_voxel_->add_point_cloud(*this->keyframe_pc_, current_pose);
            this->submap_voxel_->downsampling(*this->submap_pc_tmp_, current_pose.translation(),
                                              this->params_.submap.max_distance_range);
        }

        if (this->is_first_frame_) {
            // transform
            *this->submap_pc_ptr_ = sycl_points::algorithms::transform::transform_copy(*cloud, current_pose.matrix());
        } else if (this->submap_pc_tmp_->size() >= this->params_.registration.min_num_points) {
            // copy pointer
            this->submap_pc_ptr_ = this->submap_pc_tmp_;
        }

        // Build target search structure for registration. Neighbor queries are launched lazily only when needed.
        this->submap_tree_ = algorithms::knn::KDTree::build(*this->queue_ptr_, *this->submap_pc_ptr_);

        const auto reg_type = this->params_.registration.pipeline.registration.reg_type;
        const bool photometric_enabled = this->params_.registration.pipeline.registration.photometric.enable;
        sycl_utils::events knn_events;
        bool knn_ready = false;
        auto ensure_knn = [&]() {
            if (!knn_ready) {
                knn_events = this->submap_tree_->knn_search_async(
                    *this->submap_pc_ptr_, this->params_.covariance_estimation.neighbor_num, this->knn_result_);
                knn_ready = true;
            }
        };

        // compute grad
        sycl_utils::events grad_events;
        if (photometric_enabled) {
            if (this->submap_pc_ptr_->has_rgb()) {
                ensure_knn();
                grad_events += algorithms::color_gradient::compute_color_gradients_async(
                    *this->submap_pc_ptr_, this->knn_result_, knn_events.evs);
            } else if (this->submap_pc_ptr_->has_intensity()) {
                ensure_knn();
                grad_events += algorithms::intensity_gradient::compute_intensity_gradients_async(
                    *this->submap_pc_ptr_, this->knn_result_, knn_events.evs);
            }
        }

        // compute covariances and normals
        sycl_utils::events cov_events;
        {
            const bool need_covariances = reg_type == algorithms::registration::RegType::GICP ||
                                          reg_type == algorithms::registration::RegType::POINT_TO_DISTRIBUTION ||
                                          reg_type == algorithms::registration::RegType::GENZ ||
                                          this->params_.registration.pipeline.registration.rotation_constraint.enable;
            const bool need_normals = (reg_type == algorithms::registration::RegType::POINT_TO_PLANE ||
                                       reg_type == algorithms::registration::RegType::GENZ ||  //
                                       photometric_enabled);

            const bool submap_has_cov = this->submap_pc_ptr_->has_cov();
            bool normals_are_ready = false;
            bool covariances_are_ready = submap_has_cov;
            if (need_normals) {
                normals_are_ready = true;
                if (submap_has_cov) {
                    ensure_knn();
                    cov_events += algorithms::covariance::compute_normals_from_covariances_async(*this->submap_pc_ptr_,
                                                                                                 knn_events.evs);
                } else {
                    ensure_knn();
                    cov_events += algorithms::covariance::compute_normals_async(this->knn_result_,
                                                                                *this->submap_pc_ptr_, knn_events.evs);
                }
            }
            if (need_covariances && !submap_has_cov) {
                covariances_are_ready = true;
                ensure_knn();
                cov_events += algorithms::covariance::compute_covariances_async(this->knn_result_,
                                                                                *this->submap_pc_ptr_, knn_events.evs);
            }

            if (photometric_enabled && !normals_are_ready) {
                if (covariances_are_ready) {
                    cov_events += algorithms::covariance::compute_normals_from_covariances_async(*this->submap_pc_ptr_,
                                                                                                 cov_events.evs);
                } else {
                    ensure_knn();
                    cov_events += algorithms::covariance::compute_normals_async(this->knn_result_,
                                                                                *this->submap_pc_ptr_, knn_events.evs);
                }
            }
        }
        if (knn_ready) {
            knn_events.wait_and_throw();
        }
        grad_events.wait_and_throw();
        cov_events.wait_and_throw();
    }

    bool submapping(const algorithms::registration::RegistrationResult& reg_result, double timestamp) {
        if (this->params_.registration.pipeline.velocity_update.enable &&  //
            !this->is_imu_deskew_enabled()) {
            algorithms::deskew::deskew_point_cloud_constant_velocity(*this->preprocessed_pc_, *this->preprocessed_pc_,
                                                                     this->odom_, reg_result.T);
        }

        // check inlier ratio for registration success or not.
        const auto* registration_input_pc = this->registration_pipeline_->get_registration_input_point_cloud();
        if (registration_input_pc == nullptr || registration_input_pc->size() == 0) {
            return false;
        }
        if (this->params_.submap.keyframe.inlier_ratio_threshold > 0.0f) {
            const float inlier_ratio =
                static_cast<float>(reg_result.inlier) / static_cast<float>(registration_input_pc->size());
            if (inlier_ratio <= this->params_.submap.keyframe.inlier_ratio_threshold) {
                // registration is failed
                return false;
            }
        }

        const auto submap_type = this->params_.submap.map_type;
        if (submap_type == SubmapMapType::OCCUPANCY_GRID_MAP) {
            /* for occupancy grid map */
            // add point every frame
            build_submap(this->preprocessed_pc_, reg_result.T);
            return true;
        } else {
            /* for voxel grid map (keyframe base) */
            // calculate delta pose
            const auto delta_pose = this->last_keyframe_pose_.inverse() * reg_result.T;

            // calculate moving distance and angle
            const auto distance = delta_pose.translation().norm();
            const auto angle = std::fabs(Eigen::AngleAxisf(delta_pose.rotation()).angle()) * (180.0f / M_PIf);

            // calculate delta time
            const auto delta_time = this->last_keyframe_time_ > 0.0 ? timestamp - this->last_keyframe_time_
                                                                    : std::numeric_limits<double>::max();

            const bool is_keyframe = distance >= this->params_.submap.keyframe.distance_threshold ||
                                     angle >= this->params_.submap.keyframe.angle_threshold_degrees ||
                                     delta_time >= this->params_.submap.keyframe.time_threshold_seconds;
            // update submap
            if (is_keyframe) {
                this->last_keyframe_pose_ = reg_result.T;
                this->last_keyframe_time_ = timestamp;
                this->keyframe_poses_.push_back(reg_result.T);

                build_submap(this->preprocessed_pc_, reg_result.T);
                return true;
            }
            return false;
        }
    }
};

}  // namespace lidar_odometry
}  // namespace pipeline
}  // namespace sycl_points
