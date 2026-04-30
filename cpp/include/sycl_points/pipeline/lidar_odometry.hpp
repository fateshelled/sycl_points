#pragma once

#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include "sycl_points/algorithms/deskew/relative_pose_deskew.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/algorithms/imu/imu_velocity_corrector.hpp"
#include "sycl_points/algorithms/registration/registration_pipeline.hpp"
#include "sycl_points/pipeline/adaptive_motion_predictor.hpp"
#include "sycl_points/pipeline/lidar_odometry_params.hpp"
#include "sycl_points/pipeline/pointcloud_processing.hpp"
#include "sycl_points/pipeline/submapping.hpp"
#include "sycl_points/points/point_cloud.hpp"
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
    const auto& get_last_keyframe_pose() const { return this->submap_->get_last_keyframe_pose(); }
    const auto& get_keyframe_poses() const { return this->submap_->get_keyframe_poses(); }

    const PointCloudShared& get_preprocessed_point_cloud() const { return *this->preprocessed_pc_; }
    const PointCloudShared& get_submap_point_cloud() const { return this->submap_->get_submap_point_cloud(); }
    const PointCloudShared& get_keyframe_point_cloud() const { return this->submap_->get_keyframe_point_cloud(); }
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

        // Drop invalid data
        if (!meas.accel.allFinite() || !meas.gyro.allFinite()) {
            return;
        }

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
    }

    /// @brief Return a snapshot of the current IMU buffer (for deskewing etc.).
    ///        Thread-safe.
    std::deque<imu::IMUMeasurement> get_imu_buffer() const {
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
            try {
                time_utils::measure_execution([&]() { this->preprocess(scan); }, dt_preprocessing);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("preprocess: ") + e.what();
                std::cerr << "[LiDAR Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
        }

        // compute covariances
        {
            double dt_covariance = 0.0;
            try {
                time_utils::measure_execution([&]() { compute_covariances(); }, dt_covariance);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("compute_covariances: ") + e.what();
                std::cerr << "[LiDAR Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::compute_covariances, dt_covariance);
        }

        // refine filter (angle incidence filter + intensity zscore)
        {
            double dt_refine_filter = 0.0;
            try {
                time_utils::measure_execution([&]() { this->refine_filter(this->preprocessed_pc_); }, dt_refine_filter);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("refine_filter: ") + e.what();
                std::cerr << "[LiDAR Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            dt_preprocessing += dt_refine_filter;
            this->add_delta_time(ProcessName::preprocessing, dt_preprocessing);
        }

        // check point cloud size
        if (this->preprocessed_pc_->size() <= this->params_.registration.min_num_points) {
            this->error_message_ = "point cloud size is too small";
            return ResultType::small_number_of_points;
        }

        // first frame processing
        if (this->is_first_frame_) {
            try {
                this->submap_->add_first_frame(*this->preprocessed_pc_, timestamp);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("build_submap (first frame): ") + e.what();
                std::cerr << "[LiDAR Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }

            this->is_first_frame_ = false;
            this->last_frame_time_ = timestamp;

            // Reset IMU integrator so the next window starts from the initial pose.
            if (this->imu_preintegration_) {
                const Eigen::Matrix3f R_world_imu =
                    this->params_.pose.initial.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
                std::lock_guard<std::mutex> lock(imu_mutex_);
                this->imu_preintegration_->reset(this->params_.imu.bias, R_world_imu, Eigen::Vector3f::Zero());
                this->last_imu_reset_timestamp_ = timestamp;
            }

            return ResultType::first_frame;
        }

        // Integrate IMU buffer for the current window [last_reset, timestamp].
        if (this->imu_preintegration_) {
            this->imu_batch_.clear();
            {
                std::lock_guard<std::mutex> lock(imu_mutex_);
                this->imu_batch_.reserve(this->imu_buffer_.size());
                for (const auto& meas : this->imu_buffer_) {
                    if (meas.timestamp <= this->last_imu_reset_timestamp_) continue;
                    if (meas.timestamp > timestamp) break;
                    this->imu_batch_.push_back(meas);
                }
            }
            this->imu_preintegration_->integrate_batch(this->imu_batch_);
        }

        // Registration
        {
            double dt_registration = 0.0;
            try {
                *this->reg_result_ = time_utils::measure_execution([&]() { return registration(); }, dt_registration);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("registration: ") + e.what();
                std::cerr << "[LiDAR Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::registration, dt_registration);
        }
        this->last_imu_reset_timestamp_ = timestamp;

        // Submapping
        {
            double dt_build_submap = 0.0;
            try {
                time_utils::measure_execution([&]() { return submapping(*this->reg_result_, timestamp); },
                                              dt_build_submap);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("submapping: ") + e.what();
                std::cerr << "[LiDAR Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
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

            if (this->imu_preintegration_) {
                const Eigen::Matrix3f R_world_imu_prev =
                    this->prev_odom_.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
                this->imu_velocity_corrector_.update(this->odom_.translation() - this->prev_odom_.translation(),
                                                     R_world_imu_prev, this->params_.imu.preintegration.gravity);
            }

            this->registrated_ = true;
        }
        return ResultType::success;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    sycl_utils::DeviceQueue::Ptr queue_ptr_ = nullptr;

    PointCloudShared::Ptr preprocessed_pc_ = nullptr;  // Sensor coordinate
    bool is_first_frame_ = true;

    algorithms::knn::KNNResult knn_result_;
    shared_vector_ptr<float> icp_weights_ = nullptr;

    pointcloud_processing::PCProcessor::Ptr pc_processor_ = nullptr;
    algorithms::registration::RegistrationPipeline::Ptr registration_pipeline_ = nullptr;

    bool registrated_ = false;
    algorithms::registration::RegistrationResult::Ptr reg_result_ = nullptr;

    Eigen::Vector3f linear_velocity_;     // [m/s] in previous LiDAR body frame
    Eigen::AngleAxisf angular_velocity_;  // [rad/s]
    Eigen::Isometry3f prev_odom_;         // prev T_odom_to_lidar
    Eigen::Isometry3f odom_;              // current T_odom_to_lidar

    submapping::Submap::Ptr submap_ = nullptr;

    double last_frame_time_ = -1.0;  // [s]
    float dt_ = -1.0f;               // [s]

    Parameters params_;

    AdaptiveMotionPredictor::Ptr motion_predictor_ = nullptr;

    imu::IMUPreintegration::Ptr imu_preintegration_ = nullptr;
    imu::IMUVelocityCorrector imu_velocity_corrector_;
    std::deque<imu::IMUMeasurement> imu_buffer_;
    mutable std::mutex imu_mutex_;            ///< Guards imu_buffer_ (written by IMU callback, read by LiDAR callback).
    double last_imu_reset_timestamp_ = -1.0;  ///< LiDAR timestamp of the last IMU reset.
    std::vector<imu::IMUMeasurement> imu_batch_;  ///< Reusable buffer for per-frame IMU snapshots.

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
            this->preprocessed_pc_ = std::make_shared<PointCloudShared>(*this->queue_ptr_);
        }

        // set Initial pose
        {
            this->odom_ = this->params_.pose.initial;
            this->prev_odom_ = this->params_.pose.initial;

            this->linear_velocity_ = Eigen::Vector3f::Zero();
            this->angular_velocity_ = Eigen::AngleAxisf::Identity();
        }

        // Point cloud processor
        {
            this->pc_processor_ = std::make_shared<pointcloud_processing::PCProcessor>(
                *this->queue_ptr_, this->params_.scan, this->params_.covariance_estimation, this->params_.imu);
        }

        // Submapping
        {
            this->submap_ = std::make_shared<submapping::Submap>(*this->queue_ptr_, this->params_);
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

        // Motion predictor
        {
            this->motion_predictor_ = std::make_shared<AdaptiveMotionPredictor>(this->params_.motion_prediction);
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
        if (this->is_imu_deskew_enabled()) {
            auto imu_buf_snapshot = this->get_imu_buffer();
            this->pc_processor_->deskew_with_imu(*scan, *scan, imu_buf_snapshot, this->odom_);
        }
        this->pc_processor_->prefilter(*scan, *this->preprocessed_pc_);
    }

    void refine_filter(const PointCloudShared::Ptr scan) {
        this->pc_processor_->refine_filter(*scan, this->knn_result_);
    }

    void compute_covariances() {
        const bool needs_covs =
            (this->params_.registration.pipeline.registration.reg_type == algorithms::registration::RegType::GICP ||
             this->params_.registration.pipeline.registration.rotation_constraint.enable ||
             this->params_.scan.preprocess.angle_incidence_filter.enable);
        const bool needs_zscore = this->params_.scan.intensity_zscore.enable && this->preprocessed_pc_->has_intensity();

        if (!needs_covs && !needs_zscore) return;

        this->knn_result_ = this->pc_processor_->compute_covariances(*this->preprocessed_pc_);
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
        Eigen::Isometry3f init_T;
        if (this->imu_preintegration_) {
            if (this->imu_preintegration_->get_dt_total() > 0.0) {
                init_T = this->imu_motion_prediction();
            } else {
                init_T = this->motion_predictor_->predict(this->linear_velocity_, this->angular_velocity_, this->odom_,
                                                          this->dt_, this->reg_result_, this->registrated_);
            }
            const Eigen::Matrix3f R_world_imu = this->odom_.rotation() * this->params_.imu.T_imu_to_lidar.rotation();

            // linear_velocity_ is in the prev_odom_ body frame, so use prev_odom_.rotation() to convert to world frame.
            const Eigen::Vector3f v_reset =
                this->imu_velocity_corrector_.get_reset_velocity(*this->imu_preintegration_, this->params_.imu.bias,
                                                                 this->prev_odom_.rotation() * this->linear_velocity_);
            this->imu_preintegration_->reset(this->params_.imu.bias, R_world_imu, v_reset);
        } else {
            init_T = this->motion_predictor_->predict(this->linear_velocity_, this->angular_velocity_, this->odom_,
                                                      this->dt_, this->reg_result_, this->registrated_);
        }

        algorithms::registration::Registration::ExecutionOptions options;
        options.dt = this->dt_;
        options.prev_pose = this->odom_.matrix();

        return this->registration_pipeline_->align(*this->preprocessed_pc_, this->submap_->get_submap_point_cloud(),
                                                   this->submap_->get_submap_kdtree(), init_T.matrix(), options);
    }

    bool submapping(const algorithms::registration::RegistrationResult& reg_result, double timestamp) {
        // If velocity update is disabled, get the registration input point cloud.
        auto reg_pc_ptr = this->registration_pipeline_->get_deskewed_point_cloud();
        bool computed_icp_weights = false;
        if (reg_pc_ptr) {
            const size_t total_samples = this->params_.submap.point_random_sampling_num;
            if (reg_pc_ptr->size() > total_samples) {
                // Robust ICP weighted mixed random sampling
                const auto robust_auto_scale = this->params_.registration.pipeline.robust.auto_scale;
                const float robust_scale = robust_auto_scale
                                               ? this->params_.registration.pipeline.robust.min_scale
                                               : this->params_.registration.pipeline.registration.robust.default_scale;
                this->registration_pipeline_->compute_icp_robust_weights(
                    this->submap_->get_submap_point_cloud(), this->submap_->get_submap_kdtree(), reg_result.T.matrix(),
                    robust_scale, *this->icp_weights_);
                computed_icp_weights = true;
            }
            std::swap(this->preprocessed_pc_, reg_pc_ptr);
        } else {
            if (this->params_.registration.pipeline.velocity_update.enable &&  //
                !this->is_imu_deskew_enabled()) {
                algorithms::deskew::deskew_point_cloud_constant_velocity(
                    *this->preprocessed_pc_, *this->preprocessed_pc_, this->odom_, reg_result.T, this->dt_);
            }
        }
        const float inlier_ratio = this->registration_pipeline_->get_inlier_ratio(reg_result);

        if (computed_icp_weights) {
            return this->submap_->add_frame(*this->preprocessed_pc_, reg_result, inlier_ratio, timestamp,
                                            this->icp_weights_);
        }
        return this->submap_->add_frame(*this->preprocessed_pc_, reg_result, inlier_ratio, timestamp);
    }
};

}  // namespace lidar_odometry
}  // namespace pipeline
}  // namespace sycl_points
