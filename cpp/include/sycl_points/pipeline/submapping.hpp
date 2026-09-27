#pragma once

#include <Eigen/Geometry>
#include <numbers>
#include <utility>

#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/mapping/occupancy_grid_map.hpp"
#include "sycl_points/algorithms/mapping/voxel_hash_map.hpp"
#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/pipeline/odometry_common_params.hpp"

namespace sycl_points {
namespace pipeline {
namespace submapping {
class Submap {
public:
    using Ptr = std::shared_ptr<Submap>;
    using ConstPtr = std::shared_ptr<const Submap>;
    using OdometryCommonParams = odometry::CommonParameters;
    using SubmapMapType = odometry::SubmapMapType;

    class PreparedFreeze {
    public:
        PreparedFreeze(PreparedFreeze&&) noexcept = default;
        PreparedFreeze& operator=(PreparedFreeze&&) noexcept = default;

        PreparedFreeze(const PreparedFreeze&) = delete;
        PreparedFreeze& operator=(const PreparedFreeze&) = delete;

    private:
        friend class Submap;
        PreparedFreeze() = default;

        algorithms::mapping::VoxelHashMap::Ptr submap_voxel;
        algorithms::mapping::OccupancyGridMap::Ptr occupancy_grid;
        algorithms::knn::KDTree::Ptr submap_tree;
        PointCloudShared::Ptr last_keyframe_pc;
        PointCloudShared::Ptr submap_pc;
        PointCloudShared::Ptr submap_pc_tmp;
        algorithms::knn::KNNResult knn_result;
        bool update_keyframe_metadata = false;
        double last_keyframe_time = -1.0;
        Eigen::Isometry3f last_keyframe_pose = Eigen::Isometry3f::Identity();
        std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> keyframe_poses;
    };

    const auto& get_last_keyframe_pose() const { return this->last_keyframe_pose_; }
    const auto& get_keyframe_poses() const { return this->keyframe_poses_; }
    const auto& get_submap_kdtree() const { return *this->submap_tree_; }

    const PointCloudShared& get_submap_point_cloud() const { return *this->submap_pc_ptr_; }
    const PointCloudShared& get_last_keyframe_point_cloud() const { return *this->last_keyframe_pc_; }

    Submap(const sycl_utils::DeviceQueue& queue, const OdometryCommonParams& params)
        : Submap(queue, params, params.registration.factor, params.registration.min_num_points) {}

    /// @brief Construct a submap with an explicit registration contract.
    ///
    /// GraphOdometry has its own factor namespace, so covariance/normal
    /// requirements and minimum target size must not silently fall back to the
    /// single-frame LO registration defaults.
    Submap(const sycl_utils::DeviceQueue& queue, const OdometryCommonParams& params,
           const algorithms::registration::RegistrationFactorParams& factor_params,
           size_t min_num_points)
        : queue_(queue) {
        this->last_keyframe_pc_ = std::make_shared<PointCloudShared>(this->queue_);
        this->submap_pc_ptr_ = std::make_shared<PointCloudShared>(this->queue_);
        this->submap_pc_tmp_ = std::make_shared<PointCloudShared>(this->queue_);

        this->submap_params_ = params.submap;
        this->cov_params_ = params.covariance_estimation;
        this->reg_params_ = params.registration;
        this->reg_params_.factor = factor_params;
        this->reg_params_.min_num_points = min_num_points;

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
            } else {
                this->submap_voxel_ =
                    std::make_shared<algorithms::mapping::VoxelHashMap>(this->queue_, this->submap_params_.voxel_size);
            }
        }
    }

    /// @brief Build the very first keyframe of the submap.
    /// @param current_pose  Pose at which the first scan is anchored.  Must match the
    ///                      pipeline's current odom_ (T_odom_to_lidar in the odom/world
    ///                      frame) so that subsequent frames added via add_frame() share
    ///                      the same reference frame.  If alignment fired before the first
    ///                      frame, this is the gravity-corrected pose, not the
    ///                      constructor-time default.
    void add_first_frame(const PointCloudShared& cloud, double timestamp, const Eigen::Isometry3f& current_pose) {
        auto prepared = this->prepare_first_frame(cloud, timestamp, current_pose);
        this->commit_freeze_keyframe_to_submap(std::move(prepared));
    }

    [[nodiscard]] PreparedFreeze prepare_first_frame(
        const PointCloudShared& cloud, double timestamp,
        const Eigen::Isometry3f& current_pose) const {
        auto prepared = this->prepare_freeze(cloud, current_pose);
        prepared.update_keyframe_metadata = true;
        prepared.last_keyframe_time = timestamp;
        prepared.last_keyframe_pose = current_pose;
        prepared.keyframe_poses = this->keyframe_poses_;
        if (prepared.keyframe_poses.empty()) {
            prepared.keyframe_poses.push_back(current_pose);
        } else {
            prepared.keyframe_poses.front() = current_pose;
        }
        return prepared;
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
    [[nodiscard]] PreparedFreeze prepare_freeze(
        const PointCloudShared& cloud, const Eigen::Isometry3f& optimized_pose) const {
        PreparedFreeze prepared;
        prepared.last_keyframe_pc = std::make_shared<PointCloudShared>(this->queue_);
        prepared.submap_pc_tmp = std::make_shared<PointCloudShared>(this->queue_);

        this->preprocess_filter_->random_sampling(cloud, *prepared.last_keyframe_pc,
                                                  this->submap_params_.point_random_sampling_num);

        if (this->submap_params_.map_type == SubmapMapType::OCCUPANCY_GRID_MAP) {
            prepared.occupancy_grid = this->occupancy_grid_->clone();
            prepared.occupancy_grid->add_point_cloud(*prepared.last_keyframe_pc, optimized_pose);
            prepared.occupancy_grid->extract_occupied_points(*prepared.submap_pc_tmp, optimized_pose,
                                                             this->submap_params_.max_distance_range);
        } else {
            prepared.submap_voxel = this->submap_voxel_->clone();
            prepared.submap_voxel->add_point_cloud(*prepared.last_keyframe_pc, optimized_pose);
            prepared.submap_voxel->downsampling(*prepared.submap_pc_tmp, optimized_pose.translation(),
                                                this->submap_params_.max_distance_range);
        }

        prepared.submap_pc = std::make_shared<PointCloudShared>(
            sycl_points::algorithms::transform::transform_copy(cloud, optimized_pose.matrix()));

        prepared.submap_tree = algorithms::knn::KDTree::build(this->queue_, *prepared.submap_pc);
        this->compute_covariances(*prepared.submap_pc, *prepared.submap_tree, prepared.knn_result);
        return prepared;
    }

public:

    /// @brief Atomically publish a prepared replacement state using only non-throwing swaps.
    void commit_freeze_keyframe_to_submap(PreparedFreeze prepared) noexcept {
        this->submap_voxel_.swap(prepared.submap_voxel);
        this->occupancy_grid_.swap(prepared.occupancy_grid);
        this->submap_tree_.swap(prepared.submap_tree);
        this->last_keyframe_pc_.swap(prepared.last_keyframe_pc);
        this->submap_pc_ptr_.swap(prepared.submap_pc);
        this->submap_pc_tmp_.swap(prepared.submap_pc_tmp);
        this->knn_result_.indices.swap(prepared.knn_result.indices);
        this->knn_result_.distances.swap(prepared.knn_result.distances);
        std::swap(this->knn_result_.query_size, prepared.knn_result.query_size);
        std::swap(this->knn_result_.k, prepared.knn_result.k);
        if (prepared.update_keyframe_metadata) {
            std::swap(this->last_keyframe_pose_, prepared.last_keyframe_pose);
            std::swap(this->last_keyframe_time_, prepared.last_keyframe_time);
            this->keyframe_poses_.swap(prepared.keyframe_poses);
        }
    }

    /// @brief Insert an evicted graph keyframe at its final optimized pose.
    ///        Like LO/LIO, this updates the map in place; callers must stop
    ///        processing if an exception leaves the map partially updated.
    void freeze_keyframe_to_submap(const PointCloudShared& cloud, const Eigen::Isometry3f& optimized_pose,
                                    shared_vector_ptr<float> random_sampling_weights = nullptr) {
        this->build_submap(cloud, optimized_pose, false, random_sampling_weights);
    }

private:
    sycl_points::sycl_utils::DeviceQueue queue_;

    OdometryCommonParams::Submap submap_params_;
    OdometryCommonParams::CovarianceEstimation cov_params_;
    OdometryCommonParams::Registration reg_params_;

    algorithms::knn::KNNResult knn_result_;

    double last_keyframe_time_;             // [s]
    Eigen::Isometry3f last_keyframe_pose_;  // keyframe T_odom_to_lidar
    std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> keyframe_poses_;

    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_ = nullptr;
    algorithms::mapping::VoxelHashMap::Ptr submap_voxel_ = nullptr;
    algorithms::mapping::OccupancyGridMap::Ptr occupancy_grid_ = nullptr;
    algorithms::knn::KDTree::Ptr submap_tree_ = nullptr;
    PointCloudShared::Ptr last_keyframe_pc_ = nullptr;  // Sensor coordinate
    PointCloudShared::Ptr submap_pc_ptr_ = nullptr;     // Odom/World coordinate
    PointCloudShared::Ptr submap_pc_tmp_ = nullptr;     // Odom/World coordinate

    bool is_keyframe(const algorithms::registration::RegistrationResult& reg_result, double timestamp) {        // calculate delta pose
        const auto delta_pose = this->last_keyframe_pose_.inverse() * reg_result.T;

        // calculate moving distance and angle
        const auto distance = delta_pose.translation().norm();
        const auto angle =
            std::fabs(Eigen::AngleAxisf(delta_pose.rotation()).angle()) * (180.0f / std::numbers::pi_v<float>);

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
            this->preprocess_filter_->mixed_random_sampling(cloud, *this->last_keyframe_pc_, *random_sampling_weights,
                                                            this->submap_params_.point_random_sampling_num,
                                                            this->submap_params_.weighted_sampling_ratio);
        } else {
            // uniform random sampling
            this->preprocess_filter_->random_sampling(cloud, *this->last_keyframe_pc_,
                                                      this->submap_params_.point_random_sampling_num);
        }

        // add to grid map
        const auto submap_type = this->submap_params_.map_type;
        if (submap_type == SubmapMapType::OCCUPANCY_GRID_MAP) {
            this->occupancy_grid_->add_point_cloud(*this->last_keyframe_pc_, current_pose);
            this->occupancy_grid_->extract_occupied_points(*this->submap_pc_tmp_, current_pose,
                                                           this->submap_params_.max_distance_range);
        } else {
            this->submap_voxel_->add_point_cloud(*this->last_keyframe_pc_, current_pose);
            this->submap_voxel_->downsampling(*this->submap_pc_tmp_, current_pose.translation(),
                                              this->submap_params_.max_distance_range);
        }

        if (is_first_frame) {
            // transform
            *this->submap_pc_ptr_ = sycl_points::algorithms::transform::transform_copy(cloud, current_pose.matrix());
        } else if (this->submap_pc_tmp_->size() >= this->reg_params_.min_num_points) {
            // swap pointer
            std::swap(this->submap_pc_ptr_, this->submap_pc_tmp_);
        }

        // Build target search structure for registration. Neighbor queries are launched lazily only when needed.
        this->submap_tree_ = algorithms::knn::KDTree::build(this->queue_, *this->submap_pc_ptr_);

        // compute covariances
        compute_covariances(*this->submap_pc_ptr_, *this->submap_tree_, this->knn_result_);
    }

    void compute_covariances(PointCloudShared& submap_pc, algorithms::knn::KDTree& submap_tree,
                             algorithms::knn::KNNResult& knn_result) const {
        bool knn_ready = false;
        sycl_utils::events knn_events;
        auto ensure_knn = [&]() {
            if (!knn_ready) {
                knn_events = submap_tree.knn_search_async(submap_pc, this->cov_params_.neighbor_num, knn_result);
                knn_ready = true;
            }
        };

        // compute covariances and normals
        sycl_utils::events cov_events;
        const auto reg_type = this->reg_params_.factor.reg_type;
        {
            const bool need_covariances = reg_type == algorithms::registration::RegType::GICP ||
                                          reg_type == algorithms::registration::RegType::POINT_TO_DISTRIBUTION ||
                                          reg_type == algorithms::registration::RegType::GENZ ||
                                          this->reg_params_.factor.rotation_constraint.enable;
            const bool need_normals = (reg_type == algorithms::registration::RegType::POINT_TO_PLANE ||
                                       reg_type == algorithms::registration::RegType::GENZ);

            const bool submap_has_cov = submap_pc.has_cov();
            bool normals_are_ready = false;
            bool covariances_are_ready = submap_has_cov;
            if (need_normals) {
                normals_are_ready = true;
                if (submap_has_cov) {
                    ensure_knn();
                    cov_events += algorithms::covariance::extract_normals_async(submap_pc, knn_events.evs);
                } else {
                    ensure_knn();
                    cov_events += algorithms::covariance::estimate_normals_async(knn_result, submap_pc, knn_events.evs);
                }
            }
            if (need_covariances && !submap_has_cov) {
                covariances_are_ready = true;
                ensure_knn();
                cov_events +=
                    algorithms::covariance::estimate_async(knn_result, submap_pc, knn_events.evs);
            }
        }
        cov_events.wait_and_throw();
    }
};
}  // namespace submapping
}  // namespace pipeline
}  // namespace sycl_points
