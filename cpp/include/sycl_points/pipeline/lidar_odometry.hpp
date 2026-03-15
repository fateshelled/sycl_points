#pragma once

#include <map>

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
    PointCloudShared::Ptr submap_pc_ptr_ = nullptr;    // World coordinate
    PointCloudShared::Ptr submap_pc_tmp_ = nullptr;    // World coordinate
    bool is_first_frame_ = true;

    algorithms::mapping::VoxelHashMap::Ptr submap_voxel_ = nullptr;
    algorithms::mapping::OccupancyGridMap::Ptr occupancy_grid_ = nullptr;
    algorithms::knn::KDTree::Ptr submap_tree_ = nullptr;
    algorithms::knn::KNNResult knn_result_;
    algorithms::knn::KNNResult knn_result_grad_;

    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_ = nullptr;
    algorithms::filter::VoxelGrid::Ptr voxel_filter_ = nullptr;
    algorithms::filter::PolarGrid::Ptr polar_filter_ = nullptr;
    algorithms::registration::RegistrationPipeline::Ptr registration_pipeline_ = nullptr;

    bool registrated_ = false;
    algorithms::registration::RegistrationResult::Ptr reg_result_ = nullptr;

    Eigen::Vector3f linear_velocity_;     // [m/s]
    Eigen::AngleAxisf angular_velocity_;  // [rad/s]
    Eigen::Isometry3f prev_odom_;
    Eigen::Isometry3f odom_;
    Eigen::Isometry3f last_keyframe_pose_;
    std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> keyframe_poses_;

    double last_keyframe_time_;      // [s]
    double last_frame_time_ = -1.0;  // [s]
    float dt_ = -1.0f;               // [s]

    Parameters params_;

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

    void initialize() {
        // SYCL queue
        {
            // const auto device_selector = sycl_utils::device_selector::default_selector_v;
            // sycl::device dev(device_selector);
            const auto dev =
                sycl_utils::device_selector::select_device(this->params_.device.vendor, this->params_.device.type);
            this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);
        }

        // initialize buffer
        {
            this->preprocessed_pc_.reset(new PointCloudShared(*this->queue_ptr_));
            this->keyframe_pc_.reset(new PointCloudShared(*this->queue_ptr_));
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
            this->registration_pipeline_ = std::make_shared<algorithms::registration::RegistrationPipeline>(
                *this->queue_ptr_, this->params_.registration.pipeline);
            this->reg_result_ = std::make_shared<algorithms::registration::RegistrationResult>();
            this->registrated_ = false;
        }
        // utilities
        {
            this->clear_total_processing_times();
        }
    }

    void preprocess(const PointCloudShared::Ptr scan) {
        // box filter -> polar grid -> voxel grid
        if (this->params_.scan.preprocess.box_filter.enable) {
            this->preprocess_filter_->box_filter(*scan, this->params_.scan.preprocess.box_filter.min,
                                                 this->params_.scan.preprocess.box_filter.max);
        }

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

    algorithms::registration::RegistrationResult registration() {
        const Eigen::Isometry3f init_T = this->adaptive_motion_prediction();

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
                // weighted random sampling
                const auto icp_weights = this->registration_pipeline_->compute_icp_robust_weights(
                    *this->submap_pc_ptr_, *this->submap_tree_, current_pose.matrix());
                this->preprocess_filter_->weighted_random_sampling(*reg_pc, *this->keyframe_pc_, icp_weights,
                                                                   this->params_.submap.point_random_sampling_num);
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
            // deep copy
            this->submap_pc_ptr_.reset(new PointCloudShared(*cloud));
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
            const bool need_normals = reg_type == algorithms::registration::RegType::POINT_TO_PLANE;

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

            if (reg_type == algorithms::registration::RegType::GENZ) {
                normals_are_ready = true;
                cov_events += algorithms::covariance::compute_normals_from_covariances_async(*this->submap_pc_ptr_,
                                                                                             cov_events.evs);
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
        if (this->params_.registration.pipeline.velocity_update.enable) {
            algorithms::deskew::deskew_point_cloud_constant_velocity(*this->preprocessed_pc_, *this->preprocessed_pc_,
                                                                     this->odom_, reg_result.T, this->dt_);
        }

        // check inlier ratio for registration success or not.
        const auto* registration_input_pc = this->registration_pipeline_->get_registration_input_point_cloud();
        if (registration_input_pc == nullptr || registration_input_pc->size() == 0) {
            return false;
        }
        const float inlier_ratio =
            static_cast<float>(reg_result.inlier) / static_cast<float>(registration_input_pc->size());
        if (inlier_ratio <= this->params_.submap.keyframe.inlier_ratio_threshold) {
            // registration is failed
            return false;
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
