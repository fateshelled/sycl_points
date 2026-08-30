#pragma once

#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "sycl_points/algorithms/graph/graph_optimization.hpp"
#include "sycl_points/algorithms/imu/imu_initial_alignment.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/algorithms/imu/imu_velocity_corrector.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/pipeline/lidar_odometry_params.hpp"
#include "sycl_points/pipeline/motion_predictor.hpp"
#include "sycl_points/pipeline/pointcloud_processing.hpp"
#include "sycl_points/pipeline/submapping.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/time_utils.hpp"

namespace sycl_points {
namespace pipeline {
namespace graph_odometry {
using GraphOdometryParams = lidar_odometry::Parameters;

/// @brief LiDAR odometry pipeline backed by the sliding-window graph optimizer
///        (local BA) instead of the single-frame registration pipeline.
///
/// This is intentionally a parallel of lidar_odometry::LiDAROdometryPipeline:
/// the preprocessing / covariance / submapping / IMU building blocks are shared,
/// but the per-frame pose estimate comes from algorithms::graph::GraphOptimization
/// (unary GICP vs. a fixed submap + binary GICP between window nodes, with
/// Schur-complement marginalization). The original lidar_odometry pipeline is
/// left untouched.
class GraphOdometryPipeline {
public:
    using Ptr = std::shared_ptr<GraphOdometryPipeline>;
    using ConstPtr = std::shared_ptr<const GraphOdometryPipeline>;

    enum class ResultType : std::int8_t {
        success = 0,
        first_frame,
        waiting_initial_alignment,
        error = 100,
        old_timestamp,
        small_number_of_points
    };

    explicit GraphOdometryPipeline(const GraphOdometryParams& params) {
        this->params_ = params;
        if (this->params_.imu.enable && this->params_.imu.initial_alignment.enable) {
            const double need = static_cast<double>(this->params_.imu.initial_alignment.required_duration_sec) + 0.2;
            if (this->params_.imu.buffer_duration_sec < need) {
                this->params_.imu.buffer_duration_sec = need;
            }
        }
        this->initialize();
    }

    auto get_device_queue() const { return this->queue_ptr_; }
    const auto& get_error_message() const { return this->error_message_; }
    const auto& get_odom() const { return this->odom_; }
    const auto& get_prev_odom() const { return this->prev_odom_; }
    const auto& get_last_keyframe_pose() const { return this->submap_->get_last_keyframe_pose(); }
    const auto& get_keyframe_poses() const { return this->submap_->get_keyframe_poses(); }
    const PointCloudShared& get_preprocessed_point_cloud() const { return *this->preprocessed_pc_; }
    const PointCloudShared& get_submap_point_cloud() const { return this->submap_->get_submap_point_cloud(); }
    const PointCloudShared* get_registration_input_point_cloud() const { return this->preprocessed_pc_.get(); }
    const auto& get_graph_window() const { return this->graph_opt_->window(); }
    std::map<std::string, double> get_current_processing_time() const { return this->current_processing_time_; }
    std::map<std::string, std::vector<double>> get_total_processing_times() const {
        return this->total_processing_times_;
    }

    void add_imu_measurement(const imu::IMUMeasurement& meas) {
        if (!this->params_.imu.enable) return;
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (!meas.accel.allFinite() || !meas.gyro.allFinite()) return;
        if (!this->imu_buffer_.empty() && meas.timestamp <= this->imu_buffer_.back().timestamp) return;
        this->imu_buffer_.push_back(meas);
        while (meas.timestamp - this->imu_buffer_.front().timestamp > this->params_.imu.buffer_duration_sec) {
            this->imu_buffer_.pop_front();
        }
    }

    std::deque<imu::IMUMeasurement> get_imu_buffer() const {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        return this->imu_buffer_;
    }

    ResultType process(const PointCloudShared::Ptr scan, double timestamp) {
        this->error_message_.clear();

        if (this->is_first_frame_ && this->alignment_estimator_ && this->alignment_estimator_->enabled() &&
            !this->alignment_estimator_->is_done()) {
            const auto out = this->alignment_estimator_->try_align(timestamp, this->get_imu_buffer(), this->imu_bias_);
            if (out.status != imu::InitialAlignmentEstimator::Status::success) {
                this->error_message_ = std::string("initial_alignment: ") + out.error_message;
                return ResultType::waiting_initial_alignment;
            }
            this->apply_initial_alignment(out);
        }

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
        {
            double dt = 0.0;
            try {
                time_utils::measure_execution([&]() { this->preprocess(scan); }, dt);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("preprocess: ") + e.what();
                std::cerr << "[Graph Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::preprocessing, dt);
        }

        // compute covariances
        {
            double dt = 0.0;
            try {
                time_utils::measure_execution([&]() { compute_covariances(); }, dt);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("compute_covariances: ") + e.what();
                std::cerr << "[Graph Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::compute_covariances, dt);
        }

        // refine filter
        {
            double dt = 0.0;
            try {
                time_utils::measure_execution([&]() { this->refine_filter(this->preprocessed_pc_); }, dt);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("refine_filter: ") + e.what();
                std::cerr << "[Graph Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::refine_filter, dt);
        }

        if (this->preprocessed_pc_->size() <= this->params_.registration.min_num_points) {
            this->error_message_ = "point cloud size is too small";
            return ResultType::small_number_of_points;
        }

        // first frame
        if (this->is_first_frame_) {
            try {
                this->submap_->add_first_frame(*this->preprocessed_pc_, timestamp, this->odom_);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("build_submap (first frame): ") + e.what();
                std::cerr << "[Graph Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->is_first_frame_ = false;
            this->last_frame_time_ = timestamp;
            if (this->imu_preintegration_) {
                const Eigen::Matrix3f R_world_imu =
                    this->odom_.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
                std::lock_guard<std::mutex> lock(imu_mutex_);
                this->imu_preintegration_->reset(this->imu_bias_, Eigen::Matrix<float, 15, 15>::Zero(), R_world_imu);
                this->imu_R_world_at_reset_ = R_world_imu;
                this->imu_v_world_at_reset_ = Eigen::Vector3f::Zero();
                this->last_imu_reset_timestamp_ = timestamp;
            }
            return ResultType::first_frame;
        }

        // IMU preintegration for the current window
        if (this->imu_preintegration_) {
            this->imu_batch_.clear();
            {
                std::lock_guard<std::mutex> lock(imu_mutex_);
                this->imu_batch_.reserve(this->imu_buffer_.size());
                imu::build_measurement_window(this->imu_buffer_, this->last_imu_reset_timestamp_, timestamp,
                                              this->imu_batch_);
            }
            constexpr double kTimestampToleranceSec = 1e-6;
            this->imu_window_complete_ =
                this->imu_batch_.size() >= 2 &&
                std::abs(this->imu_batch_.front().timestamp - this->last_imu_reset_timestamp_) <=
                    kTimestampToleranceSec &&
                std::abs(this->imu_batch_.back().timestamp - timestamp) <= kTimestampToleranceSec;
            this->imu_preintegration_->integrate_batch(this->imu_batch_);
        }

        // Graph optimization (local BA)
        algorithms::graph::GraphOptimization::FrameResult frame_result;
        {
            double dt = 0.0;
            try {
                frame_result = time_utils::measure_execution(
                    [&]() {
                        Eigen::Vector3f v_reset = Eigen::Vector3f::Zero();
                        const bool has_imu_prediction = this->imu_preintegration_ && this->imu_window_complete_ &&
                                                        this->imu_preintegration_->get_dt_total() > 0.0;

                        lidar_odometry::MotionPredictionCandidates candidates;
                        if (has_imu_prediction) {
                            const Eigen::Matrix3f delta_R_imu =
                                this->imu_preintegration_->get_corrected(this->imu_bias_).Delta_R;
                            const Eigen::Matrix3f& R_i2l = this->params_.imu.T_imu_to_lidar.rotation();
                            candidates.gyro_delta_rotation_lidar = R_i2l * delta_R_imu * R_i2l.transpose();
                            if (this->params_.motion_prediction.mode ==
                                lidar_odometry::MotionPredictionMode::IMU_SE3) {
                                candidates.imu_se3_pose = this->imu_motion_prediction();
                            }
                        }

                        const Eigen::Isometry3f init_T = this->motion_predictor_->predict(
                            this->linear_velocity_, this->angular_velocity_, this->odom_, this->dt_,
                            this->reg_result_, this->registrated_, candidates);

                        if (this->imu_preintegration_ &&
                            this->params_.motion_prediction.mode ==
                                lidar_odometry::MotionPredictionMode::IMU_SE3) {
                            v_reset = this->imu_velocity_corrector_.get_reset_velocity(
                                *this->imu_preintegration_, this->imu_bias_,
                                this->prev_odom_.rotation() * this->linear_velocity_);
                        }

                        // Deep-copy the source cloud: preprocessed_pc_ is reused next frame,
                        // so the window must own an immutable snapshot (and its kNN).
                        auto source_cloud = std::make_shared<PointCloudShared>(*this->preprocessed_pc_);
                        auto source_knn = algorithms::knn::KDTree::build(*this->queue_ptr_, *source_cloud);

                        // Snapshot the submap: Submap replaces its cloud/kd-tree on every
                        // add_frame(), so aliasing into it dangles. Build a fresh kd-tree on
                        // the copy so the target stays stable.
                        auto submap_cloud = std::make_shared<PointCloudShared>(this->submap_->get_submap_point_cloud());
                        std::shared_ptr<const algorithms::knn::KNNBase> submap_knn =
                            algorithms::knn::KDTree::build(*this->queue_ptr_, *submap_cloud);

                        auto result = this->graph_opt_->process_frame(
                            source_cloud, submap_cloud, submap_knn, source_knn, init_T,
                            timestamp, this->reg_params_);

                        if (this->imu_preintegration_) {
                            this->imu_R_world_at_reset_ =
                                result.current_pose.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
                            this->imu_v_world_at_reset_ = v_reset;
                            this->imu_preintegration_->reset(this->imu_bias_,
                                                             Eigen::Matrix<float, 15, 15>::Zero(),
                                                             this->imu_R_world_at_reset_);
                        }
                        return result;
                    },
                    dt);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("graph_optimize: ") + e.what();
                std::cerr << "[Graph Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::graph_optimization, dt);
        }
        this->last_imu_reset_timestamp_ = timestamp;

        // Submapping (mirrors lidar_odometry; uses the graph estimate as the pose)
        {
            double dt = 0.0;
            try {
                time_utils::measure_execution([&]() { return this->submapping(frame_result, timestamp); }, dt);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("submapping: ") + e.what();
                std::cerr << "[Graph Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::build_submap, dt);
        }

        // update odometry / velocity
        {
            this->prev_odom_ = this->odom_;
            this->odom_ = frame_result.current_pose;
            this->last_frame_time_ = timestamp;

            const auto delta_pose = this->prev_odom_.inverse() * this->odom_;
            const Eigen::AngleAxisf delta_angle_axis(delta_pose.rotation());
            this->linear_velocity_ = delta_pose.translation() / this->dt_;
            this->angular_velocity_ = Eigen::AngleAxisf(delta_angle_axis.angle() / this->dt_, delta_angle_axis.axis());

            if (this->imu_preintegration_ && this->params_.motion_prediction.mode == lidar_odometry::MotionPredictionMode::IMU_SE3) {
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
    PointCloudShared::Ptr preprocessed_pc_ = nullptr;
    bool is_first_frame_ = true;
    pointcloud_processing::ProcessingContext processing_ctx_;
    pointcloud_processing::PCProcessor::Ptr pc_processor_ = nullptr;
    submapping::Submap::Ptr submap_ = nullptr;
    std::shared_ptr<algorithms::graph::GraphOptimization> graph_opt_ = nullptr;
    algorithms::registration::RegistrationParams reg_params_;

    bool registrated_ = false;
    algorithms::registration::RegistrationResult::Ptr reg_result_ = nullptr;
    Eigen::Vector3f linear_velocity_;
    Eigen::AngleAxisf angular_velocity_;
    Eigen::Isometry3f prev_odom_;
    Eigen::Isometry3f odom_;
    double last_frame_time_ = -1.0;
    float dt_ = -1.0f;
    GraphOdometryParams params_;
    lidar_odometry::MotionPredictor::Ptr motion_predictor_ = nullptr;

    imu::IMUPreintegration::Ptr imu_preintegration_ = nullptr;
    imu::IMUVelocityCorrector imu_velocity_corrector_;
    imu::IMUBias imu_bias_;
    imu::InitialAlignmentEstimator::Ptr alignment_estimator_ = nullptr;
    std::deque<imu::IMUMeasurement> imu_buffer_;
    mutable std::mutex imu_mutex_;
    double last_imu_reset_timestamp_ = -1.0;
    Eigen::Matrix3f imu_R_world_at_reset_ = Eigen::Matrix3f::Identity();
    Eigen::Vector3f imu_v_world_at_reset_ = Eigen::Vector3f::Zero();
    std::vector<imu::IMUMeasurement> imu_batch_;
    bool imu_window_complete_ = false;

    std::string error_message_;
    enum class ProcessName { preprocessing = 0, compute_covariances, refine_filter, graph_optimization, build_submap };
    const std::map<ProcessName, std::string> pn_map_ = {
        {ProcessName::preprocessing, "1. preprocessing"},
        {ProcessName::compute_covariances, "2. compute covariances"},
        {ProcessName::refine_filter, "3. refine filter"},
        {ProcessName::graph_optimization, "4. graph optimization"},
        {ProcessName::build_submap, "5. build submap"},
    };
    std::map<std::string, double> current_processing_time_;
    std::map<std::string, std::vector<double>> total_processing_times_;

    void clear_current_processing_time() {
        for (const auto& [k, v] : pn_map_) {
            this->current_processing_time_[v] = 0.0;
        }
    }
    void clear_total_processing_times() {
        for (const auto& [k, v] : pn_map_) {
            this->total_processing_times_[v] = {};
        }
    }
    void add_delta_time(ProcessName name, double dt) {
        this->total_processing_times_[pn_map_.at(name)].push_back(dt);
        this->current_processing_time_[pn_map_.at(name)] = dt;
    }

    bool is_imu_deskew_enabled() const { return this->params_.imu.enable && this->params_.imu.deskew.enable; }

    void initialize() {
        {
            const auto dev =
                sycl_utils::device_selector::select_device(this->params_.device.vendor, this->params_.device.type);
            this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);
        }
        this->preprocessed_pc_ = std::make_shared<PointCloudShared>(*this->queue_ptr_);
        this->odom_ = this->params_.pose.initial;
        this->prev_odom_ = this->params_.pose.initial;
        this->linear_velocity_ = Eigen::Vector3f::Zero();
        this->angular_velocity_ = Eigen::AngleAxisf::Identity();

        this->pc_processor_ = std::make_shared<pointcloud_processing::PCProcessor>(
            *this->queue_ptr_, this->params_.scan, this->params_.covariance_estimation, this->params_.imu);

        this->submap_ = std::make_shared<submapping::Submap>(*this->queue_ptr_, this->params_);

        // Registration parameters for the graph factors.
        this->reg_params_ = algorithms::registration::RegistrationParams(this->params_.registration.factor,
                                                                        this->params_.lo.registration.optimization);

        // Graph optimizer (sliding window local BA).
        const size_t window_size = 5;
        this->graph_opt_ = std::make_shared<algorithms::graph::GraphOptimization>(*this->queue_ptr_, algorithms::graph::GraphSolverParams(),
                                                                     window_size);

        this->clear_total_processing_times();
        this->motion_predictor_ = std::make_shared<lidar_odometry::MotionPredictor>(this->params_.motion_prediction);
        this->imu_bias_ = this->params_.imu.bias;
        if (this->params_.imu.enable && this->params_.motion_prediction.mode != lidar_odometry::MotionPredictionMode::LIDAR_CV) {
            this->imu_preintegration_ = std::make_shared<imu::IMUPreintegration>(this->params_.imu.preintegration);
            const Eigen::Matrix3f R_world_imu =
                this->params_.pose.initial.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
            this->imu_preintegration_->reset(this->imu_bias_, Eigen::Matrix<float, 15, 15>::Zero(), R_world_imu);
            this->imu_R_world_at_reset_ = R_world_imu;
            this->imu_v_world_at_reset_ = Eigen::Vector3f::Zero();
        }
        if (this->params_.imu.enable) {
            this->alignment_estimator_ = std::make_shared<imu::InitialAlignmentEstimator>(
                this->params_.imu.initial_alignment, this->params_.imu.preintegration.gravity,
                this->params_.imu.T_imu_to_lidar);
        }
        this->reg_result_ = std::make_shared<algorithms::registration::RegistrationResult>();
    }

    void apply_initial_alignment(const imu::InitialAlignmentEstimator::Output& out) {
        const float yaw_user = imu::detail::yaw_from_rotation(this->params_.pose.initial.rotation());
        const Eigen::Matrix3f R_odom_lidar =
            Eigen::AngleAxisf(yaw_user, Eigen::Vector3f::UnitZ()).toRotationMatrix() * out.R_gravity_lidar;
        this->odom_.linear() = R_odom_lidar;
        this->prev_odom_.linear() = R_odom_lidar;
        this->imu_bias_.gyro_bias = out.gyro_bias;
    }

    void preprocess(const PointCloudShared::Ptr scan) {
        if (this->is_imu_deskew_enabled()) {
            auto imu_buf_snapshot = this->get_imu_buffer();
            this->pc_processor_->deskew_with_imu(*scan, *scan, imu_buf_snapshot, this->odom_);
        }
        this->pc_processor_->prefilter(*scan, *this->preprocessed_pc_);
    }

    void refine_filter(const PointCloudShared::Ptr scan) {
        this->pc_processor_->refine_filter(*scan, this->processing_ctx_);
    }

    void compute_covariances() {
        const bool needs_covs =
            (this->params_.registration.factor.reg_type == algorithms::registration::RegType::GICP ||
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

    Eigen::Isometry3f imu_motion_prediction() {
        const TransformMatrix T_imu_rel = this->imu_preintegration_->predict_relative_transform(
            this->imu_R_world_at_reset_, this->imu_v_world_at_reset_, this->imu_bias_);
        const Eigen::Isometry3f& T_i2l = this->params_.imu.T_imu_to_lidar;
        Eigen::Isometry3f T_imu_rel_iso = Eigen::Isometry3f::Identity();
        T_imu_rel_iso.linear() = T_imu_rel.block<3, 3>(0, 0);
        T_imu_rel_iso.translation() = T_imu_rel.block<3, 1>(0, 3);
        const Eigen::Isometry3f T_lidar_rel = T_i2l * T_imu_rel_iso * T_i2l.inverse();
        return this->odom_ * T_lidar_rel;
    }

    void submapping(const algorithms::graph::GraphOptimization::FrameResult& frame_result, double timestamp) {
        *this->reg_result_ = algorithms::registration::RegistrationResult();
        this->reg_result_->T = frame_result.current_pose;
        this->reg_result_->converged = frame_result.converged;
        this->reg_result_->iterations = frame_result.iterations;
        this->reg_result_->error = frame_result.error;

        this->submap_->add_frame(*this->preprocessed_pc_, *this->reg_result_, 1.0f, timestamp, nullptr);
    }
};

}  // namespace graph_odometry
}  // namespace pipeline
}  // namespace sycl_points
