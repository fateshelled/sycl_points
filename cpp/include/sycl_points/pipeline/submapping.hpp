#pragma once

#include <Eigen/Geometry>
#include <sycl_points/algorithms/registration/registration_params.hpp>

#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/feature/photometric_gradient.hpp"
#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/mapping/occupancy_grid_map.hpp"
#include "sycl_points/algorithms/mapping/voxel_hash_map.hpp"
#include "sycl_points/pipeline/lidar_odometry_params.hpp"

namespace sycl_points {
namespace pipeline {
namespace submapping {
class Submap {
public:
    using Ptr = std::shared_ptr<Submap>;
    using ConstPtr = std::shared_ptr<const Submap>;
    using LidarOdometryParams = lidar_odometry::Parameters;
    using SubmapMapType = lidar_odometry::SubmapMapType;

    const auto& get_last_keyframe_pose() const { return this->last_keyframe_pose_; }
    const auto& get_keyframe_poses() const { return this->keyframe_poses_; }
    const auto& get_submap_kdtree() const { return *this->submap_tree_; }

    const PointCloudShared& get_submap_point_cloud() const { return *this->submap_pc_ptr_; }
    const PointCloudShared& get_keyframe_point_cloud() const { return *this->keyframe_pc_; }

    Submap(const sycl_utils::DeviceQueue queue, const LidarOdometryParams& params) : queue_(queue) {
        this->keyframe_pc_.reset(new PointCloudShared(this->queue_));
        this->submap_pc_ptr_.reset(new PointCloudShared(this->queue_));
        this->submap_pc_tmp_.reset(new PointCloudShared(this->queue_));

        this->submap_params_ = params.submap;
        this->cov_params_ = params.covariance_estimation;
        this->reg_params_ = params.registration;

        // initialize keyframe
        {
            const auto initial_pose = params.pose.initial;
            this->last_keyframe_pose_ = initial_pose;
            this->last_keyframe_time_ = -1.0;
            this->keyframe_poses_.clear();
            this->keyframe_poses_.push_back(initial_pose);
        }

        this->preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(this->queue_);

        // initialize submap
        {
            const auto submap_type = this->submap_params_.map_type;
            if (submap_type == SubmapMapType::OCCUPANCY_GRID_MAP) {
                this->occupancy_grid_ = std::make_shared<algorithms::mapping::OccupancyGridMap>(
                    this->queue_, this->submap_params_.voxel_size);

                this->occupancy_grid_->set_log_odds_hit(this->submap_params_.occupancy_grid_map.log_odds_hit);
                this->occupancy_grid_->set_log_odds_miss(this->submap_params_.occupancy_grid_map.log_odds_miss);
                this->occupancy_grid_->set_log_odds_limits(this->submap_params_.occupancy_grid_map.log_odds_limits_min,
                                                           this->submap_params_.occupancy_grid_map.log_odds_limits_max);
                this->occupancy_grid_->set_occupancy_threshold(
                    this->submap_params_.occupancy_grid_map.occupied_threshold);
                this->occupancy_grid_->set_free_space_updates_enabled(
                    this->submap_params_.occupancy_grid_map.enable_free_space_updates);
                this->occupancy_grid_->set_voxel_pruning_enabled(
                    this->submap_params_.occupancy_grid_map.enable_pruning);
                this->occupancy_grid_->set_stale_frame_threshold(
                    this->submap_params_.occupancy_grid_map.stale_frame_threshold);
                this->occupancy_grid_->set_covariance_aggregation_mode(
                    this->submap_params_.covariance_aggregation_mode);
            } else {
                this->submap_voxel_ =
                    std::make_shared<algorithms::mapping::VoxelHashMap>(this->queue_, this->submap_params_.voxel_size);
                this->submap_voxel_->set_covariance_aggregation_mode(this->submap_params_.covariance_aggregation_mode);
            }
        }
    }

    void add_first_frame(const PointCloudShared& cloud, double timestamp) {
        this->build_submap(cloud, this->last_keyframe_pose_, true);
        this->last_keyframe_time_ = timestamp;
    }

    bool add_frame(const PointCloudShared& preprocessed_cloud,
                   const algorithms::registration::RegistrationResult& reg_result, float inlier_ratio, double timestamp,
                   shared_vector_ptr<float> random_sampling_weights = nullptr) {
        // check inlier ratio for registration success or not.
        if (this->submap_params_.keyframe.inlier_ratio_threshold > 0.0f &&
            inlier_ratio <= this->submap_params_.keyframe.inlier_ratio_threshold) {
            // registration is failed
            return false;
        }

        const auto submap_type = this->submap_params_.map_type;
        if (submap_type == SubmapMapType::OCCUPANCY_GRID_MAP) {
            this->build_submap(preprocessed_cloud, reg_result.T, false, random_sampling_weights);
            return true;
        } else {
            if (this->is_keyframe(reg_result, timestamp)) {
                this->last_keyframe_pose_ = reg_result.T;
                this->last_keyframe_time_ = timestamp;
                this->keyframe_poses_.push_back(reg_result.T);

                this->build_submap(preprocessed_cloud, reg_result.T, false, random_sampling_weights);
                return true;
            }
        }
        return false;
    }

private:
    sycl_points::sycl_utils::DeviceQueue queue_;

    LidarOdometryParams::Submap submap_params_;
    LidarOdometryParams::CovarianceEstimation cov_params_;
    LidarOdometryParams::Registration reg_params_;

    algorithms::knn::KNNResult knn_result_;
    algorithms::knn::KNNResult knn_result_grad_;

    double last_keyframe_time_;             // [s]
    Eigen::Isometry3f last_keyframe_pose_;  // keyframe T_odom_to_lidar
    std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> keyframe_poses_;

    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_ = nullptr;
    algorithms::mapping::VoxelHashMap::Ptr submap_voxel_ = nullptr;
    algorithms::mapping::OccupancyGridMap::Ptr occupancy_grid_ = nullptr;
    algorithms::knn::KDTree::Ptr submap_tree_ = nullptr;
    PointCloudShared::Ptr keyframe_pc_ = nullptr;    // Sensor coordinate
    PointCloudShared::Ptr submap_pc_ptr_ = nullptr;  // Odom/World coordinate
    PointCloudShared::Ptr submap_pc_tmp_ = nullptr;  // Odom/World coordinate

    bool is_keyframe(const algorithms::registration::RegistrationResult& reg_result, double timestamp) {
        // calculate delta pose
        const auto delta_pose = this->last_keyframe_pose_.inverse() * reg_result.T;

        // calculate moving distance and angle
        const auto distance = delta_pose.translation().norm();
        const auto angle = std::fabs(Eigen::AngleAxisf(delta_pose.rotation()).angle()) * (180.0f / M_PIf);

        // calculate delta time
        const auto delta_time = this->last_keyframe_time_ > 0.0 ? timestamp - this->last_keyframe_time_
                                                                : std::numeric_limits<double>::max();

        const bool is_keyframe = distance >= this->submap_params_.keyframe.distance_threshold ||
                                 angle >= this->submap_params_.keyframe.angle_threshold_degrees ||
                                 delta_time >= this->submap_params_.keyframe.time_threshold_seconds;
        return is_keyframe;
    }

    void build_submap(const PointCloudShared& cloud, const Eigen::Isometry3f& current_pose, bool is_first_frame,
                      shared_vector_ptr<float> random_sampling_weights = nullptr) {
        if (random_sampling_weights &&
            random_sampling_weights->size() == cloud.size()) {  // weighted/uniform mixed random sampling
            this->preprocess_filter_->mixed_random_sampling(cloud, *this->keyframe_pc_, *random_sampling_weights,
                                                            this->submap_params_.point_random_sampling_num,
                                                            this->submap_params_.weighted_sampling_ratio);
        } else {
            // uniform random sampling
            this->preprocess_filter_->random_sampling(cloud, *this->keyframe_pc_,
                                                      this->submap_params_.point_random_sampling_num);
        }

        // add to grid map
        const auto submap_type = this->submap_params_.map_type;
        if (submap_type == SubmapMapType::OCCUPANCY_GRID_MAP) {
            this->occupancy_grid_->add_point_cloud(*this->keyframe_pc_, current_pose);
            this->occupancy_grid_->extract_occupied_points(*this->submap_pc_tmp_, current_pose,
                                                           this->submap_params_.max_distance_range);
        } else {
            this->submap_voxel_->add_point_cloud(*this->keyframe_pc_, current_pose);
            this->submap_voxel_->downsampling(*this->submap_pc_tmp_, current_pose.translation(),
                                              this->submap_params_.max_distance_range);
        }

        if (is_first_frame) {
            // transform
            *this->submap_pc_ptr_ = sycl_points::algorithms::transform::transform_copy(cloud, current_pose.matrix());
        } else if (this->submap_pc_tmp_->size() >= this->reg_params_.min_num_points) {
            // copy pointer
            this->submap_pc_ptr_ = this->submap_pc_tmp_;
        }

        // Build target search structure for registration. Neighbor queries are launched lazily only when needed.
        this->submap_tree_ = algorithms::knn::KDTree::build(this->queue_, *this->submap_pc_ptr_);

        // compute gradient and covariances
        sycl_utils::events knn_events;
        bool knn_ready = false;
        compute_grad(knn_ready, knn_events);
        compute_covariances(knn_ready, knn_events);

        if (knn_ready) {
            knn_events.wait_and_throw();
        }
    }

    void compute_grad(bool& knn_ready, sycl_utils::events& knn_events) {
        auto ensure_knn = [&]() {
            if (!knn_ready) {
                knn_events = this->submap_tree_->knn_search_async(*this->submap_pc_ptr_, this->cov_params_.neighbor_num,
                                                                  this->knn_result_);
                knn_ready = true;
            }
        };

        sycl_utils::events grad_events;
        const bool photometric_enabled = this->reg_params_.pipeline.registration.photometric.enable;
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
    }

    void compute_covariances(bool& knn_ready, sycl_utils::events& knn_events) {
        auto ensure_knn = [&]() {
            if (!knn_ready) {
                knn_events = this->submap_tree_->knn_search_async(*this->submap_pc_ptr_, this->cov_params_.neighbor_num,
                                                                  this->knn_result_);
                knn_ready = true;
            }
        };

        // compute covariances and normals
        sycl_utils::events cov_events;
        const auto reg_type = this->reg_params_.pipeline.registration.reg_type;
        const bool photometric_enabled = this->reg_params_.pipeline.registration.photometric.enable;
        {
            const bool need_covariances = reg_type == algorithms::registration::RegType::GICP ||
                                          reg_type == algorithms::registration::RegType::POINT_TO_DISTRIBUTION ||
                                          reg_type == algorithms::registration::RegType::GENZ ||
                                          this->reg_params_.pipeline.registration.rotation_constraint.enable;
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
        cov_events.wait_and_throw();
    }
};
}  // namespace submapping
}  // namespace pipeline
}  // namespace sycl_points
