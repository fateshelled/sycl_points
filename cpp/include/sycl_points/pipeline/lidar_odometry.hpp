#pragma once

#include <map>
#include <sycl_points/algorithms/color_gradient.hpp>
#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/algorithms/deskew/relative_pose_deskew.hpp>
#include <sycl_points/algorithms/intensity_correction.hpp>
#include <sycl_points/algorithms/knn/kdtree.hpp>
#include <sycl_points/algorithms/mapping/occupancy_grid_map.hpp>
#include <sycl_points/algorithms/mapping/voxel_hash_map.hpp>
#include <sycl_points/algorithms/polar_downsampling.hpp>
#include <sycl_points/algorithms/preprocess_filter.hpp>
#include <sycl_points/algorithms/registration/registration.hpp>
#include <sycl_points/algorithms/voxel_downsampling.hpp>
#include <sycl_points/pipeline/lidar_odometry_params.hpp>
#include <sycl_points/utils/time_utils.hpp>

namespace sycl_points {
namespace pipeline {
namespace lidar_odometry {
using LidarOdometryParams = lidar_odometry::Parameters;

class LiDAROdometry {
public:
    using Ptr = std::shared_ptr<LiDAROdometry>;
    using ConstPtr = std::shared_ptr<const LiDAROdometry>;

    enum class ResultType : std::int8_t {
        success = 0,  //
        first_frame,
        error = 100,
        old_timestamp,
        small_number_of_points
    };

    LiDAROdometry(const LidarOdometryParams& params) {
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
        {
            double dt_preprocessing = 0.0;
            time_utils::measure_execution([&]() { this->preprocess(scan); }, dt_preprocessing);
            this->add_delta_time(ProcessName::preprocessing, dt_preprocessing);
        }

        // check point cloud size
        if (this->preprocessed_pc_->size() <= this->params_.registration_min_num_points) {
            this->error_message_ = "point cloud size is too small";
            return ResultType::small_number_of_points;
        }

        // compute covariances
        {
            double dt_covariance = 0.0;
            time_utils::measure_execution([&]() { compute_covariances(); }, dt_covariance);
            this->add_delta_time(ProcessName::compute_covariances, dt_covariance);
        }

        // first frame processing
        if (this->is_first_frame_) {
            this->build_submap(this->preprocessed_pc_, this->params_.initial_pose);

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

    PointCloudShared::Ptr preprocessed_pc_ = nullptr;
    PointCloudShared::Ptr registration_input_pc_ = nullptr;
    PointCloudShared::Ptr keyframe_pc_ = nullptr;
    PointCloudShared::Ptr submap_pc_ptr_ = nullptr;
    PointCloudShared::Ptr submap_pc_tmp_ = nullptr;
    bool is_first_frame_ = true;

    algorithms::mapping::VoxelHashMap::Ptr submap_voxel_ = nullptr;
    algorithms::mapping::OccupancyGridMap::Ptr occupancy_grid_ = nullptr;
    algorithms::knn::KDTree::Ptr submap_tree_ = nullptr;
    algorithms::knn::KNNResult knn_result_;
    algorithms::knn::KNNResult knn_result_grad_;

    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_ = nullptr;
    algorithms::filter::VoxelGrid::Ptr voxel_filter_ = nullptr;
    algorithms::filter::PolarGrid::Ptr polar_filter_ = nullptr;
    algorithms::registration::Registration::Ptr registration_ = nullptr;

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
            const auto dev = sycl_utils::device_selector::select_device(this->params_.sycl_device_vendor,
                                                                        this->params_.sycl_device_type);
            this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);
        }

        // initialize buffer
        {
            this->preprocessed_pc_.reset(new PointCloudShared(*this->queue_ptr_));
            this->registration_input_pc_.reset(new PointCloudShared(*this->queue_ptr_));
            this->keyframe_pc_.reset(new PointCloudShared(*this->queue_ptr_));
            this->submap_pc_tmp_.reset(new PointCloudShared(*this->queue_ptr_));
        }

        // set Initial pose
        {
            this->odom_ = this->params_.initial_pose;
            this->prev_odom_ = this->params_.initial_pose;

            this->linear_velocity_ = Eigen::Vector3f::Zero();
            this->angular_velocity_ = Eigen::AngleAxisf::Identity();
        }

        // initialize keyframe
        {
            this->last_keyframe_pose_ = this->params_.initial_pose;
            this->last_keyframe_time_ = -1.0;
            this->keyframe_poses_.clear();
            this->keyframe_poses_.push_back(this->last_keyframe_pose_);
        }

        // Point cloud processor
        {
            this->preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(*this->queue_ptr_);
            if (this->params_.scan_downsampling_voxel_enable) {
                this->voxel_filter_ = std::make_shared<algorithms::filter::VoxelGrid>(
                    *this->queue_ptr_, this->params_.scan_downsampling_voxel_size);
            }
            if (this->params_.scan_downsampling_polar_enable) {
                const auto coord_system =
                    algorithms::coordinate_system_from_string(this->params_.scan_downsampling_polar_coord_system);
                this->polar_filter_ = std::make_shared<algorithms::filter::PolarGrid>(
                    *this->queue_ptr_, this->params_.scan_downsampling_polar_distance_size,
                    this->params_.scan_downsampling_polar_elevation_size,
                    this->params_.scan_downsampling_polar_azimuth_size, coord_system);
            }
        }

        // Submapping
        {
            if (this->params_.occupancy_grid_map_enable) {
                this->occupancy_grid_ = std::make_shared<algorithms::mapping::OccupancyGridMap>(
                    *this->queue_ptr_, this->params_.submap_voxel_size);

                this->occupancy_grid_->set_log_odds_hit(this->params_.occupancy_grid_map_log_odds_hit);
                this->occupancy_grid_->set_log_odds_miss(this->params_.occupancy_grid_map_log_odds_miss);
                this->occupancy_grid_->set_log_odds_limits(this->params_.occupancy_grid_map_log_odds_limits_min,
                                                           this->params_.occupancy_grid_map_log_odds_limits_max);
                this->occupancy_grid_->set_occupancy_threshold(this->params_.occupancy_grid_map_occupied_threshold);
                this->occupancy_grid_->set_voxel_pruning_enabled(this->params_.occupancy_grid_map_enable_pruning);
                this->occupancy_grid_->set_stale_frame_threshold(
                    this->params_.occupancy_grid_map_stale_frame_threshold);
            } else {
                this->submap_voxel_ = std::make_shared<algorithms::mapping::VoxelHashMap>(
                    *this->queue_ptr_, this->params_.submap_voxel_size);
            }
        }
        // Registration
        {
            this->registration_ =
                std::make_shared<algorithms::registration::Registration>(*this->queue_ptr_, this->params_.reg_params);
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
        if (this->params_.scan_preprocess_box_filter_enable) {
            this->preprocess_filter_->box_filter(*scan, this->params_.scan_preprocess_box_filter_min,
                                                 this->params_.scan_preprocess_box_filter_max);
        }
        if (this->params_.scan_downsampling_polar_enable) {
            this->polar_filter_->downsampling(*scan, *this->preprocessed_pc_);
            if (this->params_.scan_downsampling_voxel_enable) {
                this->voxel_filter_->downsampling(*this->preprocessed_pc_, *this->preprocessed_pc_);
            }
        } else {
            if (this->params_.scan_downsampling_voxel_enable) {
                this->voxel_filter_->downsampling(*scan, *this->preprocessed_pc_);
            } else {
                *this->preprocessed_pc_ = *scan;  // copy
            }
        }

        if (this->params_.scan_downsampling_random_enable) {
            preprocess_filter_->random_sampling(*this->preprocessed_pc_, *this->preprocessed_pc_,
                                                this->params_.scan_downsampling_random_num);
        }
        if (this->params_.scan_intensity_correction_enable) {
            algorithms::intensity_correction::correct_intensity(*this->preprocessed_pc_,
                                                                this->params_.scan_intensity_correction_exp);
        }
    }

    void compute_covariances() {
        if (this->params_.reg_params.reg_type == algorithms::registration::RegType::GICP) {
            // build KDTree
            const auto src_tree = algorithms::knn::KDTree::build(*this->queue_ptr_, *this->preprocessed_pc_);
            auto events = src_tree->knn_search_async(*this->preprocessed_pc_,
                                                     this->params_.scan_covariance_neighbor_num, this->knn_result_);
            events += algorithms::covariance::compute_covariances_async(this->knn_result_, *this->preprocessed_pc_,
                                                                        events.evs);
            // events += algorithms::covariance::covariance_update_plane_async(*this->preprocessed_pc_,
            // events.evs);
            events += algorithms::covariance::covariance_normalize_async(*this->preprocessed_pc_, events.evs);
            events.wait_and_throw();
        }
    }

    /// Predict initial pose by applying the previous motion model
    Eigen::Isometry3f adaptive_motion_prediction() {
        const float rot_factor = this->params_.motion_prediction_static_factor;
        float trans_factor = this->params_.motion_prediction_static_factor;

        if (this->params_.motion_prediction_adaptive_enable) {
            if (this->registrated_ && this->reg_result_->inlier > 0) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_trans(this->reg_result_->H.block<3, 3>(3, 3));
                if (solver_trans.info() == Eigen::Success) {
                    const float low = this->params_.motion_prediction_adaptive_eigen_low;
                    const float high = this->params_.motion_prediction_adaptive_eigen_high;
                    const float max_factor = this->params_.motion_prediction_adaptive_trans_factor_max;
                    const float min_factor = this->params_.motion_prediction_adaptive_trans_factor_min;

                    const float min_eig_ratio = solver_trans.eigenvalues().minCoeff() / this->reg_result_->inlier;
                    const float score = std::clamp((min_eig_ratio - low) / (high - low), 0.0f, 1.0f);
                    trans_factor = max_factor * (1.0f - score) + min_factor * score;
                }
            }
        }

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

        if (this->params_.registration_random_sampling_enable) {
            this->preprocess_filter_->random_sampling(*this->preprocessed_pc_, *this->registration_input_pc_,
                                                      this->params_.registration_random_sampling_num);
        } else {
            *this->registration_input_pc_ = *this->preprocessed_pc_;
        }

        algorithms::registration::RegistrationResult result;
        if (this->params_.registration_velocity_update_enable) {
            result = this->registration_->align_velocity_update(
                *this->registration_input_pc_, *this->submap_pc_ptr_, *this->submap_tree_, init_T.matrix(), this->dt_,
                this->params_.registration_velocity_update_iter, this->odom_.matrix());
        } else {
            result = this->registration_->align(*this->registration_input_pc_, *this->submap_pc_ptr_,
                                                *this->submap_tree_, init_T.matrix());
        }
        return result;
    }

    void build_submap(const PointCloudShared::Ptr& pc, const Eigen::Isometry3f& current_pose) {
        // random sampling
        this->preprocess_filter_->random_sampling(*pc, *this->keyframe_pc_,
                                                  this->params_.submap_point_random_sampling_num);

        // add to grid map
        if (this->params_.occupancy_grid_map_enable) {
            this->occupancy_grid_->add_point_cloud(*this->keyframe_pc_, current_pose);
            this->occupancy_grid_->extract_occupied_points(*this->submap_pc_tmp_, current_pose,
                                                           this->params_.submap_max_distance_range);
        } else {
            this->submap_voxel_->add_point_cloud(*this->keyframe_pc_, current_pose);
            this->submap_voxel_->downsampling(*this->submap_pc_tmp_, current_pose.translation(),
                                              this->params_.submap_max_distance_range);
        }

        if (this->is_first_frame_) {
            // deep copy
            this->submap_pc_ptr_.reset(new PointCloudShared(*this->queue_ptr_, *pc));
        } else if (this->submap_pc_tmp_->size() >= this->params_.registration_min_num_points) {
            // copy pointer
            this->submap_pc_ptr_ = this->submap_pc_tmp_;
        }

        // KNN search
        this->submap_tree_ = algorithms::knn::KDTree::build(*this->queue_ptr_, *this->submap_pc_ptr_);
        auto knn_events = this->submap_tree_->knn_search_async(
            *this->submap_pc_ptr_, this->params_.submap_covariance_neighbor_num, this->knn_result_);

        // compute grad
        sycl_utils::events grad_events;
        if (this->params_.reg_params.photometric.enable) {
            if (this->submap_pc_ptr_->has_rgb()) {
                if (this->params_.submap_covariance_neighbor_num != this->params_.submap_color_gradient_neighbor_num) {
                    grad_events += this->submap_tree_->knn_search_async(
                        *this->submap_pc_ptr_, this->params_.submap_color_gradient_neighbor_num,
                        this->knn_result_grad_);
                    grad_events += algorithms::color_gradient::compute_color_gradients_async(
                        *this->submap_pc_ptr_, this->knn_result_grad_, grad_events.evs);
                } else {
                    grad_events += algorithms::color_gradient::compute_color_gradients_async(
                        *this->submap_pc_ptr_, this->knn_result_, knn_events.evs);
                }
            } else if (this->submap_pc_ptr_->has_intensity()) {
                if (this->params_.submap_covariance_neighbor_num != this->params_.submap_color_gradient_neighbor_num) {
                    grad_events += this->submap_tree_->knn_search_async(
                        *this->submap_pc_ptr_, this->params_.submap_color_gradient_neighbor_num,
                        this->knn_result_grad_);
                    grad_events += algorithms::intensity_gradient::compute_intensity_gradients_async(
                        *this->submap_pc_ptr_, this->knn_result_grad_, grad_events.evs);
                } else {
                    grad_events += algorithms::intensity_gradient::compute_intensity_gradients_async(
                        *this->submap_pc_ptr_, this->knn_result_, knn_events.evs);
                }
            }
        }

        // compute covariances and normals
        sycl_utils::events cov_events;
        {
            bool compute_normal = false;
            bool compute_cov = false;
            if (this->params_.reg_params.reg_type != algorithms::registration::RegType::POINT_TO_POINT) {
                if (this->params_.reg_params.reg_type == algorithms::registration::RegType::POINT_TO_PLANE) {
                    compute_normal = true;
                    cov_events += algorithms::covariance::compute_normals_async(this->knn_result_,
                                                                                *this->submap_pc_ptr_, knn_events.evs);
                } else if (this->params_.reg_params.reg_type == algorithms::registration::RegType::GICP ||
                           this->params_.reg_params.reg_type == algorithms::registration::RegType::GENZ) {
                    compute_cov = true;
                    cov_events += algorithms::covariance::compute_covariances_async(
                        this->knn_result_, *this->submap_pc_ptr_, knn_events.evs);
                }

                if (this->params_.reg_params.reg_type == algorithms::registration::RegType::GICP) {
                    cov_events +=
                        algorithms::covariance::covariance_update_plane_async(*this->submap_pc_ptr_, cov_events.evs);
                    // cov_events += algorithms::covariance::covariance_normalize_async(*this->submap_pc_ptr_,
                    // cov_events.evs);
                } else if (this->params_.reg_params.reg_type == algorithms::registration::RegType::GENZ) {
                    compute_normal = true;
                    cov_events += algorithms::covariance::compute_normals_from_covariances_async(*this->submap_pc_ptr_,
                                                                                                 cov_events.evs);
                }
            }
            if (this->params_.reg_params.photometric.enable && !compute_normal) {
                if (compute_cov) {
                    cov_events += algorithms::covariance::compute_normals_from_covariances_async(*this->submap_pc_ptr_,
                                                                                                 cov_events.evs);
                } else {
                    cov_events += algorithms::covariance::compute_normals_async(this->knn_result_,
                                                                                *this->submap_pc_ptr_, knn_events.evs);
                }
            }
        }
        knn_events.wait_and_throw();
        grad_events.wait_and_throw();
        cov_events.wait_and_throw();
    }

    bool submapping(const algorithms::registration::RegistrationResult& reg_result, double timestamp) {
        if (this->params_.registration_velocity_update_enable) {
            algorithms::deskew::deskew_point_cloud_constant_velocity(*this->preprocessed_pc_, *this->preprocessed_pc_,
                                                                     this->odom_, reg_result.T, this->dt_);
        }

        // check inlier ratio for registration success or not.
        if (this->registration_input_pc_->size() == 0) {
            return false;
        }
        const float inlier_ratio =
            static_cast<float>(reg_result.inlier) / static_cast<float>(this->registration_input_pc_->size());
        if (inlier_ratio <= this->params_.keyframe_inlier_ratio_threshold) {
            return false;
        }

        // for occupancy grid map
        if (this->params_.occupancy_grid_map_enable) {
            // add point every frame
            build_submap(this->preprocessed_pc_, reg_result.T);
            return true;
        }

        // for voxel grid map (keyframe base)
        {
            // calculate delta pose
            const auto delta_pose = this->last_keyframe_pose_.inverse() * reg_result.T;

            // calculate moving distance and angle
            const auto distance = delta_pose.translation().norm();
            const auto angle = std::fabs(Eigen::AngleAxisf(delta_pose.rotation()).angle()) * (180.0f / M_PIf);

            // calculate delta time
            const auto delta_time = this->last_keyframe_time_ > 0.0 ? timestamp - this->last_keyframe_time_
                                                                    : std::numeric_limits<double>::max();

            // update submap
            if (distance >= this->params_.keyframe_distance_threshold ||
                angle >= this->params_.keyframe_angle_threshold_degrees ||
                delta_time >= this->params_.keyframe_time_threshold_seconds) {
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
