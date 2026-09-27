#pragma once
#include <atomic>

#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/algorithms/graph/graph_optimization.hpp"
#include "sycl_points/algorithms/imu/imu_initial_alignment.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/pipeline/lidar_odometry_params.hpp"
#include "sycl_points/pipeline/detail/keyframe_imu_history.hpp"
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
        imu_only,
        error = 100,
        old_timestamp,
        small_number_of_points,
        insufficient_imu_coverage
    };

    enum class IMUCoverage : std::int8_t {
        ready,
        recovery_required,
        waiting_for_future,
        start_expired,
    };

    explicit GraphOdometryPipeline(const GraphOdometryParams& params) {
        this->params_ = params;
        // Tightly-coupled graph LIO: IMU is mandatory. The LiDAR-only graph mode
        // is intentionally removed (every keyframe node carries velocity and
        // biases and every keyframe pair carries a preintegration edge).
        if (!this->params_.imu.enable) {
            throw std::invalid_argument(
                "[Graph Odometry] tightly-coupled graph LIO requires imu/enable=true");
        }
        if (!this->params_.imu.initial_alignment.enable || !this->params_.imu.deskew.enable ||
            this->params_.motion_prediction.mode != lidar_odometry::MotionPredictionMode::IMU_SE3) {
            throw std::invalid_argument(
                "[Graph Odometry] graph LIO requires initial alignment, IMU deskew, and IMU_SE3 prediction");
        }
        const auto& lever_arm = this->params_.imu.T_imu_to_lidar.translation();
        if (!lever_arm.allFinite() || lever_arm.norm() > 1e-5f) {
            throw std::invalid_argument(
                "[Graph Odometry] T_imu_to_lidar translation must be zero: the IMU graph factor ignores the lever arm");
        }
        const auto& p = this->params_.imu.preintegration;
        if (!std::isfinite(p.gyro_noise_density) || p.gyro_noise_density <= 0.0f ||
            !std::isfinite(p.accel_noise_density) || p.accel_noise_density <= 0.0f ||
            !std::isfinite(p.gyro_bias_rw_density) || p.gyro_bias_rw_density <= 0.0f ||
            !std::isfinite(p.accel_bias_rw_density) || p.accel_bias_rw_density <= 0.0f) {
            throw std::invalid_argument("[Graph Odometry] IMU noise densities must be finite and positive");
        }
        const auto& lio = this->params_.graph.lio;
        if (!std::isfinite(lio.keyframe_imu_history_duration_sec) ||
            lio.keyframe_imu_history_duration_sec <= 0.0 ||
            lio.keyframe_imu_history_max_samples < 2 ||
            !std::isfinite(lio.root_prior_sigma_velocity) ||
            !std::isfinite(lio.root_prior_sigma_pose) || lio.root_prior_sigma_pose <= 0.0f ||
            lio.root_prior_sigma_velocity <= 0.0f ||
            !std::isfinite(lio.root_prior_sigma_accel_bias) ||
            lio.root_prior_sigma_accel_bias <= 0.0f ||
            !std::isfinite(lio.root_prior_sigma_gyro_bias) ||
            lio.root_prior_sigma_gyro_bias <= 0.0f) {
            throw std::invalid_argument("[Graph Odometry] graph LIO limits and prior sigmas must be positive");
        }
        this->keyframe_imu_history_ = detail::KeyframeImuHistory(
            {lio.keyframe_imu_history_duration_sec, lio.keyframe_imu_history_max_samples});
        if (this->params_.imu.initial_alignment.enable) {
            const double need = static_cast<double>(this->params_.imu.initial_alignment.required_duration_sec) + 0.2;
            if (this->params_.imu.buffer_duration_sec < need) {
                this->params_.imu.buffer_duration_sec = need;
            }
        }
        this->initialize();
    }

    auto get_device_queue() const { return this->queue_ptr_; }
    const auto& get_error_message() const { return this->error_message_; }
    bool has_fatal_submap_error() const { return this->fatal_submap_error_.load(); }
    const auto& get_odom() const { return this->odom_; }
    const auto& get_prev_odom() const { return this->prev_odom_; }
    const auto& get_last_keyframe_pose() const { return this->last_keyframe_pose_; }
    const auto& get_keyframe_poses() const { return this->keyframe_poses_; }
    const PointCloudShared& get_preprocessed_point_cloud() const { return *this->preprocessed_pc_; }
    const PointCloudShared& get_submap_point_cloud() const { return this->submap_->get_submap_point_cloud(); }
    /// @brief The exact point cloud the latest frame's tip factor consumed
    ///        (sampled / final-deskewed, i.e. FrameResult::tip_cloud). Null
    ///        before the first successful graph frame. Discarded frames leave
    ///        the latest committed factor input unchanged.
    const PointCloudShared* get_registration_input_point_cloud() const {
        return this->last_factor_input_.get();
    }
    const auto& get_graph_window() const { return this->graph_opt_->window(); }
    /// @brief Cumulative marginalization outcome counters (success / failure /
    ///        force-drop / max lambda) for observability on long runs.
    const auto& get_marginalization_diagnostics() const {
        return this->graph_opt_->window().marginalization_diagnostics();
    }
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
        while (meas.timestamp - this->imu_buffer_.front().timestamp >
               this->params_.imu.buffer_duration_sec) {
            const double pinned_start = this->imu_buffer_pin_start_.load();
            if (pinned_start >= 0.0 && this->imu_buffer_.size() >= 2 &&
                this->imu_buffer_[1].timestamp > pinned_start) {
                break;
            }
            this->imu_buffer_.pop_front();
        }
        this->keyframe_imu_history_.append(meas);
    }

    std::deque<imu::IMUMeasurement> get_imu_buffer() const {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        return this->imu_buffer_;
    }

    IMUCoverage get_imu_coverage(double start_timestamp, double end_timestamp) const {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        return classify_imu_coverage_locked(start_timestamp, end_timestamp);
    }

    IMUCoverage get_frame_imu_coverage(double frame_timestamp) const {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (is_first_frame_) {
            if (this->imu_buffer_.empty()) return IMUCoverage::waiting_for_future;
            if (this->imu_buffer_.front().timestamp > frame_timestamp) return IMUCoverage::start_expired;
            return this->imu_buffer_.back().timestamp < frame_timestamp
                       ? IMUCoverage::waiting_for_future : IMUCoverage::ready;
        }
        IMUCoverage coverage = classify_imu_coverage_locked(state_timestamp_, frame_timestamp);
        const auto edge_coverage = this->keyframe_imu_history_.coverage(frame_timestamp);
        if (edge_coverage == detail::KeyframeImuHistory::Coverage::waiting_for_future) {
            coverage = combine_imu_coverage(coverage, IMUCoverage::waiting_for_future);
        } else if (edge_coverage == detail::KeyframeImuHistory::Coverage::recovery_required) {
            coverage = combine_imu_coverage(coverage, IMUCoverage::recovery_required);
        }
        return coverage;
    }

    static IMUCoverage combine_imu_coverage(IMUCoverage lhs, IMUCoverage rhs) {
        if (lhs == IMUCoverage::start_expired || rhs == IMUCoverage::start_expired) {
            return IMUCoverage::start_expired;
        }
        if (lhs == IMUCoverage::waiting_for_future || rhs == IMUCoverage::waiting_for_future) {
            return IMUCoverage::waiting_for_future;
        }
        if (lhs == IMUCoverage::recovery_required || rhs == IMUCoverage::recovery_required) {
            return IMUCoverage::recovery_required;
        }
        return IMUCoverage::ready;
    }

    ResultType process(const PointCloudShared::Ptr scan, double timestamp) {
        if (this->fatal_submap_error_.load()) return ResultType::error;
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

        if (this->last_frame_time_ >= 0.0) {
            const float dt = static_cast<float>(timestamp - this->last_frame_time_);
            if (dt <= 0.0f) {
                this->error_message_ = "old timestamp";
                return ResultType::old_timestamp;
            }
        }
        double pin_start = this->is_first_frame_ ? timestamp : this->state_timestamp_.load();
        if (scan && scan->has_timestamps()) {
            const double scan_start = scan->start_time_ms * 1e-3;
            pin_start = pin_start < 0.0 ? scan_start : std::min(pin_start, scan_start);
        }
        ImuBufferPin pin(this->imu_buffer_pin_start_, pin_start);

        const IMUCoverage frame_coverage = this->get_frame_imu_coverage(timestamp);
        if (frame_coverage != IMUCoverage::ready &&
            frame_coverage != IMUCoverage::recovery_required) {
            this->error_message_ = "IMU measurements do not bracket the graph LIO frame interval";
            return ResultType::insufficient_imu_coverage;
        }
        OutputCloudTransaction output_cloud(this->preprocessed_pc_, *this->queue_ptr_);
        auto working_scan = std::make_shared<PointCloudShared>(*scan);
        this->clear_current_processing_time();

        // preprocess
        {
            double dt = 0.0;
            try {
                time_utils::measure_execution([&]() { this->preprocess(working_scan); }, dt);
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

        const bool insufficient_points =
            this->preprocessed_pc_->size() <= this->params_.graph.registration.min_num_points;
        if (this->is_first_frame_ && insufficient_points) {
            this->error_message_ = "point cloud size is too small";
            return ResultType::small_number_of_points;
        }

        // first frame
        if (this->is_first_frame_) {
            try {
                auto prepared_first = this->submap_->prepare_first_frame(
                    *this->preprocessed_pc_, timestamp, this->odom_);
                auto root_cloud = std::make_shared<PointCloudShared>(*this->preprocessed_pc_);
                auto root_knn = algorithms::knn::KDTree::build(*this->queue_ptr_, *root_cloud);
                algorithms::graph::NodeState root_state;
                root_state.pose = this->odom_;
                root_state.velocity = Eigen::Vector3f::Zero();
                root_state.accel_bias = this->imu_bias_.accel_bias;
                root_state.gyro_bias = this->imu_bias_.gyro_bias;
                algorithms::graph::GraphOptimization::ImuEdgeContext root_prior;
                root_prior.root_prior_sigma_pose = this->params_.graph.lio.root_prior_sigma_pose;
                root_prior.nav_prior_sigma_velocity = this->params_.graph.lio.root_prior_sigma_velocity;
                root_prior.nav_prior_sigma_accel_bias =
                    this->params_.graph.lio.root_prior_sigma_accel_bias;
                root_prior.nav_prior_sigma_gyro_bias =
                    this->params_.graph.lio.root_prior_sigma_gyro_bias;

                decltype(keyframe_poses_) bootstrap_poses;
                bootstrap_poses.push_back(root_state.pose);
                const auto& lio = this->params_.graph.lio;
                detail::KeyframeImuHistory bootstrap_imu_history(
                    {lio.keyframe_imu_history_duration_sec, lio.keyframe_imu_history_max_samples});
                static_assert(std::is_nothrow_move_assignable_v<detail::KeyframeImuHistory>);
                // Hold the IMU lock through publication so no samples arrive
                // between building the candidate and installing it.
                {
                    std::lock_guard<std::mutex> lock(imu_mutex_);
                    if (!bootstrap_imu_history.reset(timestamp, this->imu_buffer_)) {
                        this->error_message_ = "cannot establish bootstrap IMU history boundary";
                        return ResultType::insufficient_imu_coverage;
                    }
                    this->imu_edge_source_id_ = this->graph_opt_->add_bootstrap_node(
                        this->odom_, timestamp, root_cloud, root_knn, root_state, root_prior);
                    this->keyframe_imu_history_ = std::move(bootstrap_imu_history);
                }
                this->submap_->commit_freeze_keyframe_to_submap(std::move(prepared_first));
                this->nav_state_ = root_state;
                this->state_timestamp_ = timestamp;
                this->last_keyframe_pose_ = root_state.pose;
                this->keyframe_poses_.swap(bootstrap_poses);
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
            }
            output_cloud.commit();
            return ResultType::first_frame;
        }

        // Candidate integrations are rebuilt from committed boundaries for every
        // attempted frame. A failed frame therefore cannot leave samples in a
        // persistent accumulator and cannot cause duplicate integration later.
        this->imu_preintegration_->reset(
            this->current_bias(), Eigen::Matrix<float, 15, 15>::Zero(),
            this->nav_state_.pose.rotation() * this->params_.imu.T_imu_to_lidar.rotation());
        this->imu_batch_.clear();
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            imu::build_measurement_window(this->imu_buffer_, this->state_timestamp_, timestamp,
                                          this->imu_batch_);
        }
        this->imu_preintegration_->integrate_batch(this->imu_batch_);
        this->imu_window_complete_ = this->imu_preintegration_->get_dt_total() > 0.0;

        // The keyframe edge owns raw IMU history independently from the shorter
        // prediction/deskew buffer, so dropped tips and IMU-only frames cannot
        // expire its source boundary.
        bool recover_imu_edge = false;
        bool force_keyframe_for_history = false;
        bool imu_edge_window_complete = false;
        std::vector<imu::IMUMeasurement> edge_batch;
        if (this->imu_edge_source_id_ != algorithms::graph::INVALID_NODE_ID) {
            {
                std::lock_guard<std::mutex> lock(imu_mutex_);
                recover_imu_edge = this->keyframe_imu_history_.overflowed();
                if (!recover_imu_edge) {
                    imu_edge_window_complete =
                        this->keyframe_imu_history_.build_window(timestamp, edge_batch);
                    force_keyframe_for_history = imu_edge_window_complete &&
                                                 this->keyframe_imu_history_.should_force_keyframe(timestamp);
                }
            }
            auto source = this->graph_opt_->window().get_node(this->imu_edge_source_id_);
            if (imu_edge_window_complete && source) {
                imu::IMUBias source_bias;
                source_bias.accel_bias = source->accel_bias;
                source_bias.gyro_bias = source->gyro_bias;
                const Eigen::Matrix3f R_world_imu =
                    source->pose.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
                this->imu_edge_preintegration_->reset(
                    source_bias, Eigen::Matrix<float, 15, 15>::Zero(), R_world_imu);
                this->imu_edge_preintegration_->integrate_batch(edge_batch);
            }
        }

        if (!this->imu_window_complete_ || (!recover_imu_edge && !imu_edge_window_complete)) {
            this->error_message_ = "IMU measurements do not bracket the graph LIO frame interval";
            return ResultType::insufficient_imu_coverage;
        }

        const algorithms::graph::NodeState predicted_state = this->predict_nav_state();
        if (insufficient_points) {
            const ResultType result = this->process_imu_only(predicted_state, timestamp);
            if (result == ResultType::imu_only) output_cloud.commit();
            return result;
        }

        // Graph optimization (local BA)
        algorithms::graph::GraphOptimization::FrameResult frame_result;
        const auto graph_checkpoint = this->graph_opt_->checkpoint();
        {
            double dt = 0.0;
            try {
                frame_result = time_utils::measure_execution(
                    [&]() {
                        const Eigen::Isometry3f init_T = predicted_state.pose;

                        // Deep-copy and sample the graph factor input once.
                        auto source_cloud = std::make_shared<PointCloudShared>(*this->preprocessed_pc_);
                        const auto& rs = this->params_.graph.registration.random_sampling;
                        if (rs.enable && source_cloud->size() > rs.num) {
                            if (!this->factor_input_filter_) {
                                this->factor_input_filter_ =
                                    std::make_shared<algorithms::filter::PreprocessFilter>(*this->queue_ptr_);
                            }
                            auto sampled = std::make_shared<PointCloudShared>(source_cloud->queue);
                            if (rs.use_intensities && source_cloud->has_intensity()) {
                                this->factor_input_filter_->mixed_random_sampling(
                                    *source_cloud, *sampled, *source_cloud->intensities, rs.num,
                                    rs.weighted_ratio);
                            } else {
                                this->factor_input_filter_->random_sampling(*source_cloud, *sampled, rs.num);
                            }
                            source_cloud = sampled;
                        }

                        auto source_knn = algorithms::knn::KDTree::build(*this->queue_ptr_, *source_cloud);

                        // Submap generations are immutable snapshots handed to the graph.
                        // In voxel mode the submap only changes when a keyframe is merged,
                        // so the copy + KDTree rebuild runs at keyframe rate, not scan rate.
                        if (this->submap_dirty_) {
                            this->submap_gen_cloud_ = std::make_shared<PointCloudShared>(this->submap_->get_submap_point_cloud());
                            this->submap_gen_knn_ =
                                algorithms::knn::KDTree::build(*this->queue_ptr_, *this->submap_gen_cloud_);
                            this->submap_dirty_ = false;
                        }

                        algorithms::graph::GraphOptimization::ImuEdgeContext imu_edge;
                        imu_edge.enable = imu_edge_window_complete &&
                                          this->imu_edge_source_id_ != algorithms::graph::INVALID_NODE_ID;
                        imu_edge.source_id = this->imu_edge_source_id_;
                        imu_edge.preintegration = this->imu_edge_preintegration_;
                        imu_edge.T_imu_to_lidar = this->params_.imu.T_imu_to_lidar;
                        imu_edge.gravity = this->params_.imu.preintegration.gravity;
                        imu_edge.tip_velocity = predicted_state.velocity;
                        imu_edge.tip_accel_bias = predicted_state.accel_bias;
                        imu_edge.tip_gyro_bias = predicted_state.gyro_bias;
                        imu_edge.add_nav_prior = recover_imu_edge;
                        imu_edge.force_keyframe = recover_imu_edge || force_keyframe_for_history;
                        imu_edge.nav_prior_sigma_velocity =
                            this->params_.graph.lio.root_prior_sigma_velocity;
                        imu_edge.nav_prior_sigma_accel_bias =
                            this->params_.graph.lio.root_prior_sigma_accel_bias;
                        imu_edge.nav_prior_sigma_gyro_bias =
                            this->params_.graph.lio.root_prior_sigma_gyro_bias;

                        auto result = this->graph_opt_->process_frame(
                            source_cloud, this->submap_gen_cloud_, this->submap_gen_knn_, source_knn,
                            init_T, timestamp, this->reg_params_,
                            algorithms::graph::GraphOptimization::VelocityUpdateContext(), imu_edge);

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
        // A solver failure (non-finite system / decomposition / unstable step)
        // must not reach the map or odometry. GraphOptimization restores the
        // complete pre-frame topology, node state, factor runtime, and ID state.
        if (!frame_result.solver_valid()) {
            this->error_message_ =
                "graph_optimize: solver returned an invalid state; frame discarded";
            std::cerr << "[Graph Odometry] " << this->error_message_ << std::endl;
            return ResultType::error;
        }

        // Tight-coupling lifecycle: the optimized tip carries the new velocity
        // and bias estimate; a promoted keyframe becomes the next preintegration
        // source, while a dropped tip keeps the source and accumulates onward.
        if (!frame_result.has_current_state) {
            this->error_message_ = "graph_optimize: valid solve did not return an optimized tip state";
            return ResultType::error;
        }
        const algorithms::graph::NodeState optimized_state = frame_result.current_state;
        std::optional<detail::KeyframeImuHistory> rebased_imu_history;
        if (frame_result.keyframe) {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            auto candidate = this->keyframe_imu_history_;
            if (!candidate.reset(timestamp, this->imu_buffer_)) {
                this->graph_opt_->restore(graph_checkpoint);
                this->error_message_ = "cannot rebase keyframe IMU history";
                return ResultType::insufficient_imu_coverage;
            }
            rebased_imu_history = std::move(candidate);
        }
        // Once marginalized, an evicted scan must enter the fixed submap.
        // An in-place insertion cannot be rolled back after a partial failure.
        {
            double dt = 0.0;
            try {
                time_utils::measure_execution([&]() {
                    if (frame_result.freeze_ready && frame_result.frozen_cloud &&
                        frame_result.frozen_cloud->size() > 0) {
                        // The frozen scan uses uniform sampling, not the robust
                        // weights from its earlier registration.
                        this->submap_->freeze_keyframe_to_submap(
                            *frame_result.frozen_cloud, frame_result.frozen_pose);
                        this->submap_dirty_ = true;
                    }
                }, dt);
            } catch (const std::exception& e) {
                this->error_message_ = std::string("fatal submapping failure; restart required: ") + e.what();
                this->fatal_submap_error_.store(true);
                std::cerr << "[Graph Odometry] " << this->error_message_ << std::endl;
                return ResultType::error;
            }
            this->add_delta_time(ProcessName::build_submap, dt);
        }
        this->last_factor_input_ = frame_result.tip_cloud;
        *this->reg_result_ = frame_result.tip_registration;
        this->reg_result_->T = frame_result.current_pose;
        this->reg_result_->converged = frame_result.converged;
        this->reg_result_->iterations = frame_result.iterations;

        // update odometry / velocity
        {
            this->nav_state_ = optimized_state;
            this->state_timestamp_ = timestamp;
            this->imu_bias_.accel_bias = this->nav_state_.accel_bias;
            this->imu_bias_.gyro_bias = this->nav_state_.gyro_bias;
            if (frame_result.keyframe) {
                if (auto tip = this->graph_opt_->window().get_node(frame_result.current_node_id)) {
                    this->imu_edge_source_id_ = tip->id;
                    std::lock_guard<std::mutex> lock(imu_mutex_);
                    for (const auto& measurement : this->imu_buffer_) {
                        if (measurement.timestamp > rebased_imu_history->latest_timestamp()) {
                            rebased_imu_history->append(measurement);
                        }
                    }
                    this->keyframe_imu_history_ = std::move(*rebased_imu_history);
                    if (recover_imu_edge) ++this->imu_edge_recovery_count_;
                    this->last_keyframe_pose_ = optimized_state.pose;
                    this->keyframe_poses_.push_back(optimized_state.pose);
                    if (this->keyframe_poses_.size() > this->params_.graph.window_size) {
                        this->keyframe_poses_.erase(this->keyframe_poses_.begin());
                    }
                }
            }
            this->prev_odom_ = this->odom_;
            this->odom_ = this->nav_state_.pose;
            this->last_frame_time_ = timestamp;

        }
        output_cloud.commit();
        return ResultType::success;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    sycl_utils::DeviceQueue::Ptr queue_ptr_ = nullptr;
    PointCloudShared::Ptr preprocessed_pc_ = nullptr;
    /// @brief The exact factor input of the latest processed frame (the tip
    ///        node's sampled / final-deskewed cloud), published through
    ///        get_registration_input_point_cloud().
    PointCloudShared::Ptr last_factor_input_ = nullptr;
    bool is_first_frame_ = true;
    pointcloud_processing::ProcessingContext processing_ctx_;
    pointcloud_processing::PCProcessor::Ptr pc_processor_ = nullptr;
    submapping::Submap::Ptr submap_ = nullptr;
    /// @brief Lazy PreprocessFilter for graph/factor/random_sampling (LO analog:
    ///        RegistrationPipeline's internal filter).
    algorithms::filter::PreprocessFilter::Ptr factor_input_filter_ = nullptr;
    std::shared_ptr<algorithms::graph::GraphOptimization> graph_opt_ = nullptr;
    // Current immutable submap generation (cloud + kNN) shared by the whole graph.
    std::shared_ptr<PointCloudShared> submap_gen_cloud_ = nullptr;
    std::shared_ptr<const algorithms::knn::KNNBase> submap_gen_knn_ = nullptr;
    bool submap_dirty_ = true;
    algorithms::registration::RegistrationParams reg_params_;

    algorithms::registration::RegistrationResult::Ptr reg_result_ = nullptr;
    Eigen::Isometry3f prev_odom_;
    Eigen::Isometry3f odom_;
    Eigen::Isometry3f last_keyframe_pose_ = Eigen::Isometry3f::Identity();
    std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> keyframe_poses_;
    double last_frame_time_ = -1.0;
    GraphOdometryParams params_;

    imu::IMUPreintegration::Ptr imu_preintegration_ = nullptr;
    imu::IMUBias imu_bias_;
    algorithms::graph::NodeState nav_state_;
    std::atomic<double> state_timestamp_{-1.0};
    std::atomic<double> imu_buffer_pin_start_{-1.0};
    imu::InitialAlignmentEstimator::Ptr alignment_estimator_ = nullptr;
    std::deque<imu::IMUMeasurement> imu_buffer_;
    mutable std::mutex imu_mutex_;
    std::vector<imu::IMUMeasurement> imu_batch_;
    bool imu_window_complete_ = false;

    // Keyframe-to-keyframe preintegration for the graph IMU edge. Reset when a
    // node is promoted to a persistent keyframe; accumulates across dropped
    // tips so the edge always spans the last kept keyframe -> current tip.
    imu::IMUPreintegration::Ptr imu_edge_preintegration_ = nullptr;
    algorithms::graph::NodeId imu_edge_source_id_ = algorithms::graph::INVALID_NODE_ID;
    detail::KeyframeImuHistory keyframe_imu_history_{{}};
    uint64_t imu_edge_recovery_count_ = 0;

    std::string error_message_;
    std::atomic<bool> fatal_submap_error_{false};
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

    class ImuBufferPin {
    public:
        ImuBufferPin(std::atomic<double>& pin, double timestamp) : pin_(pin) { pin_.store(timestamp); }
        ~ImuBufferPin() { pin_.store(-1.0); }

        ImuBufferPin(const ImuBufferPin&) = delete;
        ImuBufferPin& operator=(const ImuBufferPin&) = delete;

    private:
        std::atomic<double>& pin_;
    };

    class OutputCloudTransaction {
    public:
        OutputCloudTransaction(PointCloudShared::Ptr& output,
                               const sycl_utils::DeviceQueue& queue)
            : output_(output), previous_(output) {
            output_ = std::make_shared<PointCloudShared>(queue);
        }
        ~OutputCloudTransaction() {
            if (!committed_) output_ = previous_;
        }

        void commit() noexcept { committed_ = true; }
        OutputCloudTransaction(const OutputCloudTransaction&) = delete;
        OutputCloudTransaction& operator=(const OutputCloudTransaction&) = delete;

    private:
        PointCloudShared::Ptr& output_;
        PointCloudShared::Ptr previous_;
        bool committed_ = false;
    };

    IMUCoverage classify_imu_coverage_locked(double start_timestamp, double end_timestamp) const {
        if (start_timestamp < 0.0 || this->imu_buffer_.empty()) {
            return IMUCoverage::waiting_for_future;
        }
        if (end_timestamp <= start_timestamp) return IMUCoverage::start_expired;
        if (this->imu_buffer_.front().timestamp > start_timestamp) return IMUCoverage::start_expired;
        if (this->imu_buffer_.back().timestamp < end_timestamp) return IMUCoverage::waiting_for_future;
        return IMUCoverage::ready;
    }

    imu::IMUBias current_bias() const {
        imu::IMUBias bias;
        bias.accel_bias = this->nav_state_.accel_bias;
        bias.gyro_bias = this->nav_state_.gyro_bias;
        return bias;
    }

    algorithms::graph::NodeState predict_nav_state() const {
        const imu::IMUBias bias = this->current_bias();
        const Eigen::Isometry3f T_world_imu_i =
            this->nav_state_.pose * this->params_.imu.T_imu_to_lidar;
        const TransformMatrix predicted_imu = this->imu_preintegration_->predict_transform(
            T_world_imu_i.matrix(), this->nav_state_.velocity, bias);
        Eigen::Isometry3f T_world_imu_j = Eigen::Isometry3f::Identity();
        T_world_imu_j.matrix() = predicted_imu;

        algorithms::graph::NodeState predicted = this->nav_state_;
        predicted.pose = T_world_imu_j * this->params_.imu.T_imu_to_lidar.inverse();
        const auto corrected = this->imu_preintegration_->get_corrected(bias);
        predicted.velocity = this->nav_state_.velocity +
                             this->params_.imu.preintegration.gravity *
                                 static_cast<float>(corrected.dt_total) +
                             T_world_imu_i.rotation() * corrected.Delta_v;
        return predicted;
    }

    ResultType process_imu_only(const algorithms::graph::NodeState& predicted, double timestamp) {
        if (!predicted.pose.matrix().allFinite() || !predicted.velocity.allFinite()) {
            this->error_message_ = "IMU-only propagation produced a non-finite state";
            return ResultType::error;
        }
        this->prev_odom_ = this->odom_;
        this->nav_state_ = predicted;
        this->state_timestamp_ = timestamp;
        this->odom_ = predicted.pose;
        this->last_frame_time_ = timestamp;
        this->error_message_ = "point cloud size is too small; propagated with IMU only";
        return ResultType::imu_only;
    }

    void initialize() {
        {
            const auto dev =
                sycl_utils::device_selector::select_device(this->params_.device.vendor, this->params_.device.type);
            this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);
        }
        this->preprocessed_pc_ = std::make_shared<PointCloudShared>(*this->queue_ptr_);
        this->odom_ = this->params_.pose.initial;
        this->prev_odom_ = this->params_.pose.initial;

        this->pc_processor_ = std::make_shared<pointcloud_processing::PCProcessor>(
            *this->queue_ptr_, this->params_.scan, this->params_.covariance_estimation, this->params_.imu);

        // Registration parameters for the graph factors (decoupled from the single-frame
        // align path: graph/factor/* instead of registration/*). The graph factors use
        // compute_linearized_result, so the solver optimization params are unused.
        this->reg_params_ = algorithms::registration::RegistrationParams(this->params_.graph.registration.factor);
        // Graph factors use graph/robust/* (loss type + fixed default scale), not the
        // shared LO registration/robust/* auto-scale schedule.
        this->reg_params_.robust.type = this->params_.graph.robust_type;
        this->reg_params_.robust.default_scale = this->params_.graph.robust_default_scale;
        this->submap_ = std::make_shared<submapping::Submap>(
            *this->queue_ptr_, this->params_, this->params_.graph.registration.factor,
            this->params_.graph.registration.min_num_points);
        // Graph optimizer (sliding window local BA). The internal keyframe gate
        // reuses the Submap keyframe thresholds for retention; map insertion is a
        // separate lifecycle event done on eviction (see submapping()).
        algorithms::graph::GraphSolverParams solver_params;
        solver_params.optimization_method = this->params_.graph.optimization_method;
        solver_params.lm = this->params_.graph.lm;
        solver_params.max_iterations = this->params_.graph.solver_iterations;
        solver_params.convergence_translation = this->params_.graph.convergence_translation;
        solver_params.convergence_rotation = this->params_.graph.convergence_rotation;
        solver_params.convergence_velocity = this->params_.graph.convergence_velocity;
        solver_params.convergence_bias = this->params_.graph.convergence_bias;
        solver_params.max_step_velocity = this->params_.graph.max_step_velocity;
        solver_params.max_step_bias = this->params_.graph.max_step_bias;
        solver_params.relinearize_translation_thresh = this->params_.graph.relinearize_translation_thresh;
        solver_params.relinearize_rotation_thresh = this->params_.graph.relinearize_rotation_thresh;
        solver_params.solver_damping_lambda = this->params_.graph.solver_damping_lambda;
        solver_params.marginalization_lambda = this->params_.graph.marginalization_lambda;
        solver_params.degenerate_regularization.enable =
            this->params_.graph.degenerate_regularization.enable;
        solver_params.degenerate_regularization.eigenvalue_threshold =
            this->params_.graph.degenerate_regularization.eigenvalue_threshold;
        solver_params.degenerate_regularization.strength =
            this->params_.graph.degenerate_regularization.strength;
        solver_params.degenerate_regularization.representative_length =
            this->params_.graph.degenerate_regularization.representative_length;
        solver_params.degenerate_regularization.pseudo_inverse_relative_cutoff =
            this->params_.graph.degenerate_regularization.pseudo_inverse_relative_cutoff;
        solver_params.degenerate_regularization.pseudo_inverse_absolute_cutoff =
            this->params_.graph.degenerate_regularization.pseudo_inverse_absolute_cutoff;
        solver_params.robust.enable = this->params_.graph.robust_enable;
        solver_params.robust.init_scale = this->params_.graph.robust_init_scale;
        solver_params.robust.min_scale = this->params_.graph.robust_min_scale;
        solver_params.robust.levels = this->params_.graph.robust_levels;
        solver_params.robust.iters_per_level = this->params_.graph.robust_iters_per_level;
        solver_params.robust.relinearize_per_rung = this->params_.graph.robust_relinearize_per_rung;
        // Per-iteration solver logs on the graph path (LO analog: RegistrationFactorParams::verbose).
        solver_params.verbose = this->params_.graph.registration.factor.verbose;

        algorithms::graph::GraphOptimization::Options gopts;
        gopts.gate.enabled = true;
        // Retention (graph keyframe promotion) is now internal to
        // GraphOptimization's gate: submap insertion no longer doubles as the
        // retention decision (map insertion happens on eviction, see
        // submapping()).
        gopts.gate.external_decision = false;
        gopts.gate.min_translation = this->params_.submap.keyframe.distance_threshold;
        gopts.gate.min_rotation = this->params_.submap.keyframe.angle_threshold_degrees *
                                  (std::numbers::pi_v<float> / 180.0f);
        gopts.gate.min_time_seconds = this->params_.submap.keyframe.time_threshold_seconds;
        gopts.gate.min_inlier_ratio = this->params_.submap.keyframe.inlier_ratio_threshold;
        gopts.relative_pose.sigma_rotation = this->params_.graph.chain_sigma_rotation;
        gopts.relative_pose.sigma_translation = this->params_.graph.chain_sigma_translation;
        this->graph_opt_ = std::make_shared<algorithms::graph::GraphOptimization>(
            *this->queue_ptr_, solver_params, this->params_.graph.window_size, gopts);

        this->clear_total_processing_times();
        this->imu_bias_ = this->params_.imu.bias;
        const Eigen::Matrix3f R_world_imu =
            this->params_.pose.initial.rotation() * this->params_.imu.T_imu_to_lidar.rotation();
        // Prediction preintegration is rebuilt from the committed navigation
        // state for every attempted frame.
        this->imu_preintegration_ = std::make_shared<imu::IMUPreintegration>(this->params_.imu.preintegration);
        this->imu_preintegration_->reset(this->imu_bias_, Eigen::Matrix<float, 15, 15>::Zero(), R_world_imu);
        // Edge preintegration is rebuilt over the retained keyframe -> tip
        // interval for each attempted frame.
        this->imu_edge_preintegration_ = std::make_shared<imu::IMUPreintegration>(this->params_.imu.preintegration);
        this->imu_edge_preintegration_->reset(this->imu_bias_, Eigen::Matrix<float, 15, 15>::Zero(), R_world_imu);
        this->imu_edge_source_id_ = algorithms::graph::INVALID_NODE_ID;
        this->alignment_estimator_ = std::make_shared<imu::InitialAlignmentEstimator>(
            this->params_.imu.initial_alignment, this->params_.imu.preintegration.gravity,
            this->params_.imu.T_imu_to_lidar);
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
            algorithms::deskew::IMUDeskewStatus status;
            const imu::IMUBias bias = this->is_first_frame_ ? this->imu_bias_ : this->current_bias();
            const Eigen::Vector3f velocity =
                this->is_first_frame_ ? Eigen::Vector3f::Zero() : this->nav_state_.velocity;
            if (!this->pc_processor_->deskew_with_imu(*scan, *scan, imu_buf_snapshot, this->odom_,
                                                       bias, velocity, &status)) {
                throw std::runtime_error("IMU deskew requires complete timestamped IMU coverage");
            }
        }
        this->pc_processor_->prefilter(*scan, *this->preprocessed_pc_);
    }

    void refine_filter(const PointCloudShared::Ptr scan) {
        this->pc_processor_->refine_filter(*scan, this->processing_ctx_);
    }

    void compute_covariances() {
        // Feature preparation must follow the factor type's ACTUAL data needs:
        // GICP needs both covariances, POINT_TO_DISTRIBUTION needs target
        // covariances (binary kernel Omega = C_tgt,w^-1) and POINT_TO_PLANE
        // needs target normals, which are extracted from covariances when a
        // node cloud becomes a binary factor target. Without this, a P2D /
        // P2Plane run would silently degrade its binary edges to
        // point-to-point information (the kernel's missing-covariance
        // fallback), so the pipeline computes scan covariances for those
        // types too. The LO single-frame align path has its own condition.
        using RT = algorithms::registration::RegType;
        const RT graph_reg_type = this->params_.graph.registration.factor.reg_type;
        const bool needs_covs =
            (graph_reg_type == RT::GICP || graph_reg_type == RT::POINT_TO_DISTRIBUTION ||
             graph_reg_type == RT::POINT_TO_PLANE ||
             this->params_.graph.registration.factor.rotation_constraint.enable ||
             this->params_.scan.preprocess.angle_incidence_filter.enable);
        const bool needs_gaussian =
            this->params_.scan.intensity_gaussian.enable && this->preprocessed_pc_->has_intensity();
        const bool needs_local_mean_norm =
            this->params_.scan.intensity_local_mean_norm.enable && this->preprocessed_pc_->has_intensity();
        if (!needs_covs && !needs_gaussian && !needs_local_mean_norm) return;
        this->processing_ctx_ = this->pc_processor_->prepare_context(*this->preprocessed_pc_);
        this->pc_processor_->compute_covariances(*this->preprocessed_pc_, this->processing_ctx_);
    }

};

}  // namespace graph_odometry
}  // namespace pipeline
}  // namespace sycl_points
