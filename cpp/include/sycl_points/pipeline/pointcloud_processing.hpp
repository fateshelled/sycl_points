#pragma once

#include <Eigen/Geometry>

#include "sycl_points/algorithms/deskew/imu_deskew.hpp"
#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/filter/intensity_correction.hpp"
#include "sycl_points/algorithms/filter/intensity_gaussian.hpp"
#include "sycl_points/algorithms/filter/intensity_local_mean_norm.hpp"
#include "sycl_points/algorithms/filter/polar_downsampling.hpp"
#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/algorithms/filter/voxel_downsampling.hpp"
#include "sycl_points/algorithms/imu/imu_preintegration.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/pipeline/lidar_odometry_params.hpp"
#include "sycl_points/points/point_cloud.hpp"

namespace sycl_points {
namespace pipeline {
namespace pointcloud_processing {

/// @brief Per-frame context derived from a scan's KDTree and KNN search results.
///        Built by PCProcessor::prepare_context() and consumed by compute_covariances()
///        and refine_filter().
struct ProcessingContext {
    algorithms::knn::KDTree::Ptr tree;
    algorithms::knn::KNNResult knn_result;
};

class PCProcessor {
public:
    using Ptr = std::shared_ptr<PCProcessor>;
    using ConstPtr = std::shared_ptr<const PCProcessor>;

    PCProcessor(const sycl_utils::DeviceQueue& q, const lidar_odometry::Parameters::Scan& scan_params,
                const lidar_odometry::Parameters::CovarianceEstimation& covs_params,
                const lidar_odometry::Parameters::IMU& imu_params)
        : queue_(q), scan_params_(scan_params), covs_params_(covs_params), imu_params_(imu_params) {
        this->initialize();
    }

    template <imu::imu_measurement_range Range>
    void deskew_with_imu(const PointCloudShared& src, PointCloudShared& dst, const Range& imu_buffer,
                         const Eigen::Isometry3f& current_pose) const {
        this->deskew_with_imu_impl(src, dst, imu_buffer, current_pose, this->imu_params_.bias, Eigen::Vector3f::Zero());
    }

    template <imu::imu_measurement_range Range>
    void deskew_with_imu(const PointCloudShared& src, PointCloudShared& dst, const Range& imu_buffer,
                         const Eigen::Isometry3f& current_pose, const imu::IMUBias& bias,
                         const Eigen::Vector3f& v_world_body_i = Eigen::Vector3f::Zero()) const {
        this->deskew_with_imu_impl(src, dst, imu_buffer, current_pose, bias, v_world_body_i);
    }

    void prefilter(const PointCloudShared& src, PointCloudShared& dst) const { this->prefilter_impl(src, dst); }

    void random_sampling(const PointCloudShared& src, PointCloudShared& dst, size_t num) const {
        this->preprocess_filter_->random_sampling(src, dst, num);
    }

    /// @brief Build a KDTree from scan. Must be called before compute_covariances() and refine_filter().
    ProcessingContext prepare_context(const PointCloudShared& scan) const {
        ProcessingContext ctx;
        ctx.tree = algorithms::knn::KDTree::build(this->queue_, scan);
        return ctx;
    }

    /// @brief Estimate covariances using the KDTree in ctx. Populates ctx.knn_result.
    void compute_covariances(PointCloudShared& scan, ProcessingContext& ctx) const {
        this->compute_covariances_impl(scan, ctx);
    }

    /// @brief Apply post-KNN filters (angle incidence, Gaussian intensity smoothing, local-mean normalization).
    void refine_filter(PointCloudShared& scan, const ProcessingContext& ctx) const {
        this->refine_filter_impl(scan, ctx);
    }

private:
    sycl_utils::DeviceQueue queue_;
    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_ = nullptr;
    algorithms::filter::VoxelGrid::Ptr voxel_filter_ = nullptr;
    algorithms::filter::PolarGrid::Ptr polar_filter_ = nullptr;
    lidar_odometry::Parameters::Scan scan_params_;
    lidar_odometry::Parameters::CovarianceEstimation covs_params_;
    lidar_odometry::Parameters::IMU imu_params_;

    void initialize() {
        this->preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(this->queue_);
        if (this->scan_params_.downsampling.voxel.enable) {
            this->voxel_filter_ = std::make_shared<algorithms::filter::VoxelGrid>(
                this->queue_, this->scan_params_.downsampling.voxel.size);
        }
        if (this->scan_params_.downsampling.polar.enable) {
            const auto coord_system =
                algorithms::coordinate_system_from_string(this->scan_params_.downsampling.polar.coord_system);
            this->polar_filter_ = std::make_shared<algorithms::filter::PolarGrid>(
                this->queue_, this->scan_params_.downsampling.polar.distance_size,
                this->scan_params_.downsampling.polar.elevation_size,
                this->scan_params_.downsampling.polar.azimuth_size, coord_system);
        }
    }

    template <imu::imu_measurement_range Range>
    void deskew_with_imu_impl(const PointCloudShared& src, PointCloudShared& dst, const Range& imu_buffer,
                              const Eigen::Isometry3f& current_pose, const imu::IMUBias& bias,
                              const Eigen::Vector3f& v_world_body_i) const {
        const double scan_start_sec = src.start_time_ms * 1e-3;
        const Eigen::Matrix3f R_world_imu = current_pose.rotation() * this->imu_params_.T_imu_to_lidar.rotation();
        algorithms::deskew::deskew_point_cloud_imu(src, dst, imu_buffer, scan_start_sec,
                                                   this->imu_params_.T_imu_to_lidar, bias,
                                                   this->imu_params_.preintegration, R_world_imu, v_world_body_i);
    }

    void prefilter_impl(const PointCloudShared& src, PointCloudShared& dst) const {
        // Process Order:
        //   box filter -> polar grid -> voxel grid -> random sampling -> intensity correct
        //
        // `input` tracks where the current data lives. Each step writes to dst and advances
        // input to &dst. PolarGrid / VoxelGrid support in-place (input == &dst).
        const PointCloudShared* input = &src;

        if (this->scan_params_.preprocess.box_filter.enable) {
            this->preprocess_filter_->box_filter(src, dst, this->scan_params_.preprocess.box_filter.min,
                                                 this->scan_params_.preprocess.box_filter.max);
            input = &dst;
        }
        if (this->scan_params_.downsampling.polar.enable) {
            this->polar_filter_->downsampling(*input, dst);
            input = &dst;
        }
        if (this->scan_params_.downsampling.voxel.enable) {
            this->voxel_filter_->downsampling(*input, dst);
            input = &dst;
        }
        if (input != &dst) {
            dst = src;
        }

        if (this->scan_params_.downsampling.random.enable) {
            this->preprocess_filter_->random_sampling(dst, this->scan_params_.downsampling.random.num);
        }
    }

    void compute_covariances_impl(PointCloudShared& scan, ProcessingContext& ctx) const {
        auto events = ctx.tree->knn_search_async(scan, this->covs_params_.neighbor_num, ctx.knn_result);

        if (this->covs_params_.m_estimation.enable) {
            events += algorithms::covariance::estimate_robust_async(
                ctx.knn_result, scan, this->covs_params_.m_estimation.type, this->covs_params_.m_estimation.mad_scale,
                this->covs_params_.m_estimation.min_robust_scale, this->covs_params_.m_estimation.max_iterations,
                events.evs);
        } else {
            events += algorithms::covariance::estimate_async(ctx.knn_result, scan, events.evs);
        }
        events.wait_and_throw();
    }

    void refine_filter_impl(PointCloudShared& scan, const ProcessingContext& ctx) const {
        if (this->scan_params_.preprocess.angle_incidence_filter.enable) {
            this->preprocess_filter_->angle_incidence_filter(
                scan, scan, this->scan_params_.preprocess.angle_incidence_filter.min_angle,
                this->scan_params_.preprocess.angle_incidence_filter.max_angle);
        }

        if (this->scan_params_.intensity_correction.enable && !this->scan_params_.enhanced_reflectivity.enable &&
            scan.has_intensity()) {
            algorithms::intensity_correction::correct_intensity(        //
                scan,                                                   //
                this->scan_params_.intensity_correction.exp,            //
                this->scan_params_.intensity_correction.scale,          //
                this->scan_params_.intensity_correction.min_intensity,  //
                this->scan_params_.intensity_correction.max_intensity,  //
                this->scan_params_.intensity_correction.ref_distance,   //
                this->scan_params_.intensity_correction.angle_exponent);
        }

        if (this->scan_params_.intensity_gaussian.enable && scan.has_intensity()) {
            const auto& gp = this->scan_params_.intensity_gaussian;
            // Reuse covariance KNN result if it has enough neighbors, avoiding a redundant search.
            // When reusing, far neighbors contribute negligibly due to exponential weight decay.
            if (gp.neighbor_num <= ctx.knn_result.k) {
                algorithms::intensity_gaussian::smooth_intensity(scan, ctx.knn_result, gp.sigma_azimuth,
                                                                 gp.sigma_elevation, gp.sigma_range, gp.neighbor_num);
            } else {
                const auto gaussian_knn = ctx.tree->knn_search(scan, gp.neighbor_num);
                algorithms::intensity_gaussian::smooth_intensity(scan, gaussian_knn, gp.sigma_azimuth,
                                                                 gp.sigma_elevation, gp.sigma_range);
            }
        }

        if (this->scan_params_.intensity_local_mean_norm.enable && scan.has_intensity()) {
            const auto& lp = this->scan_params_.intensity_local_mean_norm;
            if (lp.neighbor_num <= ctx.knn_result.k) {
                algorithms::intensity_local_mean_norm::normalize(scan, ctx.knn_result, lp.sigma_azimuth,
                                                                 lp.sigma_elevation, lp.sigma_range, lp.mean_min,
                                                                 lp.neighbor_num);
            } else {
                const auto local_knn = ctx.tree->knn_search(scan, lp.neighbor_num);
                algorithms::intensity_local_mean_norm::normalize(scan, local_knn, lp.sigma_azimuth, lp.sigma_elevation,
                                                                 lp.sigma_range, lp.mean_min);
            }
        }
    }
};

}  // namespace pointcloud_processing
}  // namespace pipeline
}  // namespace sycl_points
