#pragma once

#include <Eigen/Core>
#include <array>
#include <iostream>
#include <stdexcept>

#include "sycl_points/algorithms/common/prefix_sum.hpp"
#include "sycl_points/algorithms/common/transform.hpp"
#include "sycl_points/algorithms/common/voxel_constants.hpp"
#include "sycl_points/algorithms/common/workgroup_utils.hpp"
#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/algorithms/mapping/covariance_aggregation_mode.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace mapping {

// Reuse the voxel hashing utilities defined for filtering algorithms.
namespace kernel = sycl_points::algorithms::filter::kernel;

class VoxelHashMap : public knn::NearestNeighborBase {
public:
    using Ptr = std::shared_ptr<VoxelHashMap>;

    /// @brief Constructor
    /// @param queue SYCL queue
    /// @param voxel_size voxel size
    VoxelHashMap(const sycl_utils::DeviceQueue& queue, const float voxel_size) : queue_(queue) {
        this->set_voxel_size(voxel_size);
        this->allocate_storage(this->capacity_);
        this->prefix_sum_ = std::make_shared<common::PrefixSum>(this->queue_);
        this->valid_flags_ptr_ = std::make_shared<shared_vector<uint8_t>>(*this->queue_.ptr);
        this->clear();
        this->wg_size_add_point_cloud_ = this->compute_wg_size_add_point_cloud();
    }

    /// @brief Set voxel size
    /// @param size voxel size
    void set_voxel_size(const float voxel_size) {
        if (voxel_size <= 0.0f) {
            throw std::invalid_argument("voxel_size must be positive.");
        }
        this->voxel_size_ = voxel_size;
        // Keep the cached reciprocal consistent for hashing operations.
        this->voxel_size_inv_ = 1.0f / voxel_size;
    }
    /// @brief Get voxel size
    /// @param voxel_size voxel size
    float get_voxel_size() const { return this->voxel_size_; }

    /// @brief
    /// @param max_staleness
    void set_max_staleness(const uint32_t max_staleness) { this->max_staleness_ = max_staleness; }
    /// @brief
    /// @return
    uint32_t get_max_staleness() const { return this->max_staleness_; }

    /// @brief
    /// @param remove_old_data_cycle
    void set_remove_old_data_cycle(const uint32_t remove_old_data_cycle) {
        this->remove_old_data_cycle_ = remove_old_data_cycle;
    }
    /// @brief
    /// @return
    uint32_t get_remove_old_data_cycle() const { return this->remove_old_data_cycle_; }

    /// @brief
    /// @param rehash_threshold
    void set_rehash_threshold(const float rehash_threshold) { this->rehash_threshold_ = rehash_threshold; }
    /// @brief
    /// @return
    float get_rehash_threshold() const { return this->rehash_threshold_; }

    /// @brief Set minimum number of points required to keep a voxel in the output
    /// @param min_num_point minimum number of accumulated points
    void set_min_num_point(const uint32_t min_num_point) { this->min_num_point_ = min_num_point; }
    /// @brief Get minimum number of points required to keep a voxel in the output
    /// @return minimum number of accumulated points
    uint32_t get_min_num_point() const { return this->min_num_point_; }

    void set_covariance_aggregation_mode(const CovarianceAggregationMode mode) {
        this->covariance_aggregation_mode_ = mode;
    }
    CovarianceAggregationMode get_covariance_aggregation_mode() const { return this->covariance_aggregation_mode_; }

    /// @brief Reset the map data.
    void clear() {
        this->capacity_ = kCapacityCandidates[0];
        this->voxel_num_ = 0;
        this->staleness_counter_ = 0;
        this->has_cov_data_ = false;
        this->has_rgb_data_ = false;
        this->has_intensity_data_ = false;

        this->key_ptr_->resize(this->capacity_);
        this->core_data_ptr_->resize(this->capacity_);
        this->covariance_data_ptr_->resize(this->capacity_);
        this->color_data_ptr_->resize(this->capacity_);
        this->intensity_data_ptr_->resize(this->capacity_);
        this->last_update_ptr_->resize(this->capacity_);

        // Reset the hash table content before the next integration round.
        sycl_utils::events evs;
        evs += this->queue_.ptr->fill<uint64_t>(this->key_ptr_->data(), VoxelConstants::invalid_coord,
                                                this->key_ptr_->size());
        evs += this->queue_.ptr->fill<VoxelCoreData>(this->core_data_ptr_->data(), VoxelCoreData{},
                                                     this->core_data_ptr_->size());
        evs += this->queue_.ptr->fill<VoxelCovarianceData>(this->covariance_data_ptr_->data(), VoxelCovarianceData{},
                                                           this->covariance_data_ptr_->size());
        evs += this->queue_.ptr->fill<VoxelColorData>(this->color_data_ptr_->data(), VoxelColorData{},
                                                      this->color_data_ptr_->size());
        evs += this->queue_.ptr->fill<VoxelIntensityData>(this->intensity_data_ptr_->data(), VoxelIntensityData{},
                                                          this->intensity_data_ptr_->size());
        evs += this->queue_.ptr->fill<uint32_t>(this->last_update_ptr_->data(), 0U, this->last_update_ptr_->size());
        evs.wait_and_throw();
    }

    /// @brief add PointCloud to voxel map
    /// @param cloud Point cloud in the sensor frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    void add_point_cloud(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) {
        const size_t N = cloud.size();

        // rehash
        if (this->rehash_threshold_ < (float)this->voxel_num_ / (float)this->capacity_) {
            const size_t next_capacity = this->get_next_capacity_value();
            if (next_capacity > this->capacity_) {
                this->rehash(next_capacity);
            }
        }

        if (N > 0) {
            // add PointCloud to voxel map
            this->add_point_cloud_impl(cloud, sensor_pose);
        }

        // remove old data
        if (this->remove_old_data_cycle_ > 0 && (this->staleness_counter_ % this->remove_old_data_cycle_) == 0) {
            this->remove_old_data();
        }

        // increment counter
        ++this->staleness_counter_;
    }

    /// @brief Export the aggregated voxels within the provided bounding range.
    /// @param result Point cloud container storing the filtered voxels.
    /// @param center Center of the query bounding box in meters.
    /// @param distance Half-extent of the axis-aligned bounding box in meters.
    void downsampling(PointCloudShared& result, const Eigen::Vector3f& center, const float distance = 100.0f) {
        if (this->voxel_num_ == 0) {
            result.clear();
            return;
        }

        const size_t allocation_size = this->voxel_num_;

        result.resize_points(allocation_size);

        Covariance* cov_output = nullptr;
        if (this->has_cov_data_) {
            result.resize_covs(allocation_size);
            cov_output = result.covs_ptr();
        } else {
            result.resize_covs(0);
        }

        RGBType* rgb_output = nullptr;
        if (this->has_rgb_data_) {
            // Allocate RGB container when aggregated color data is available.
            result.resize_rgb(allocation_size);
            rgb_output = result.rgb_ptr();
        } else {
            result.resize_rgb(0);
        }

        float* intensity_output = nullptr;
        if (this->has_intensity_data_) {
            // Allocate intensity container when aggregated intensity data is available.
            result.resize_intensities(allocation_size);
            intensity_output = result.intensities_ptr();
        } else {
            result.resize_intensities(0);
        }

        const size_t final_voxel_num =
            this->downsampling_impl(*result.points, center, distance, cov_output, rgb_output, intensity_output);
        result.resize_points(final_voxel_num);
        result.resize_covs(this->has_cov_data_ ? final_voxel_num : 0);
        result.resize_rgb(this->has_rgb_data_ ? final_voxel_num : 0);
        result.resize_intensities(this->has_intensity_data_ ? final_voxel_num : 0);
    }

    /// @brief Compute the overlap ratio between the map and an input point cloud.
    /// @param cloud Point cloud in the sensor frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    /// @return Ratio of points that overlap with existing voxels in the map.
    float compute_overlap_ratio(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) const {
        if (!cloud.points || cloud.points->empty() || this->voxel_num_ == 0) {
            return 0.0f;
        }

        const size_t N = cloud.size();
        shared_vector<uint32_t> overlap_counter(1, 0U, *this->queue_.ptr);

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            auto overlap_reduction = sycl::reduction(overlap_counter.data(), sycl::plus<uint32_t>());
            const auto trans = eigen_utils::to_sycl_vec(sensor_pose.matrix());

            const auto point_ptr = cloud.points_ptr();
            const auto key_ptr = this->key_ptr_->data();
            const auto core_ptr = this->core_data_ptr_->data();
            const float voxel_size_inv = this->voxel_size_inv_;
            const size_t max_probe = this->max_probe_length_;
            const size_t capacity = this->capacity_;
            const uint32_t min_num_point = this->min_num_point_;

            h.parallel_for(sycl::range<1>(N), overlap_reduction, [=](sycl::id<1> idx, auto& overlap_sum) {
                const size_t i = idx[0];
                const PointType local_point = point_ptr[i];
                PointType world_point;
                // Transform the input point into the map frame before voxel hashing.
                transform::kernel::transform_point(local_point, world_point, trans);
                const uint64_t voxel_hash = filter::kernel::compute_voxel_bit(world_point, voxel_size_inv);
                if (voxel_hash == VoxelConstants::invalid_coord) {
                    return;
                }

                for (size_t probe = 0; probe < max_probe; ++probe) {
                    const size_t slot = compute_slot_id(voxel_hash, probe, capacity);
                    const uint64_t stored_key = key_ptr[slot];
                    if (stored_key == voxel_hash) {
                        // Count as overlap only when enough samples were accumulated in the voxel.
                        const VoxelCoreData& voxel_core = core_ptr[slot];
                        if (voxel_core.count >= min_num_point) {
                            overlap_sum += 1U;
                        }
                        return;
                    }
                    if (stored_key == VoxelConstants::invalid_coord) {
                        return;
                    }
                }
            });
        });

        event.wait_and_throw();

        return static_cast<float>(overlap_counter.at(0)) / static_cast<float>(N);
    }

    void remove_old_data() { this->remove_old_data_impl(); }

    /// @brief Async nearest neighbor search using voxel neighborhood lookup.
    /// @tparam NUM_NEIGHBOR_VOXELS Number of neighboring voxels to search: 7 (faces), 19 (faces+edges), or 27 (all)
    /// @param queries Query point cloud
    /// @param result KNNResult with k=1: indices are hash-table slot indices (-1 if not found),
    ///               distances are squared Euclidean distances to the voxel centroid
    /// @param depends SYCL event dependencies
    /// @param transT Transform matrix applied to query points before search
    /// @return SYCL events
    template <size_t NUM_NEIGHBOR_VOXELS = 27>
    sycl_utils::events nearest_neighbor_search_async(
        const PointCloudShared& queries, knn::KNNResult& result, const std::vector<sycl::event>& depends = {},
        const TransformMatrix& transT = TransformMatrix::Identity()) const {
        static_assert(NUM_NEIGHBOR_VOXELS == 7 || NUM_NEIGHBOR_VOXELS == 19 || NUM_NEIGHBOR_VOXELS == 27,
                      "NUM_NEIGHBOR_VOXELS must be 7, 19, or 27");
        return nearest_neighbor_search_impl<NUM_NEIGHBOR_VOXELS>(queries, result, depends, transT);
    }

    /// @brief Nearest neighbor search (synchronous wrapper).
    /// @tparam NUM_NEIGHBOR_VOXELS Number of neighboring voxels to search: 7 (faces), 19 (faces+edges), or 27 (all)
    /// @param queries Query point cloud
    /// @param result KNNResult with k=1: indices are hash-table slot indices (-1 if not found),
    ///               distances are squared Euclidean distances to the voxel centroid
    /// @param depends SYCL event dependencies
    /// @param transT Transform matrix applied to query points before search
    template <size_t NUM_NEIGHBOR_VOXELS = 27>
    void nearest_neighbor_search(const PointCloudShared& queries, knn::KNNResult& result,
                                 const std::vector<sycl::event>& depends = {},
                                 const TransformMatrix& transT = TransformMatrix::Identity()) const {
        nearest_neighbor_search_async<NUM_NEIGHBOR_VOXELS>(queries, result, depends, transT).wait_and_throw();
    }

    /// @brief Set the number of neighboring voxels used by the non-template nearest_neighbor_search_async override.
    /// @param n Must be 7 (faces), 19 (faces+edges), or 27 (all)
    void set_num_neighbor_voxels(size_t n) {
        if (n != 7 && n != 19 && n != 27) {
            throw std::invalid_argument("num_neighbor_voxels must be 7, 19, or 27");
        }
        num_neighbor_voxels_ = n;
    }

    /// @brief Get the current num_neighbor_voxels setting.
    size_t get_num_neighbor_voxels() const { return num_neighbor_voxels_; }

    // NearestNeighborBase override: dispatches to the template variant selected by num_neighbor_voxels_.
    sycl_utils::events nearest_neighbor_search_async(
        const PointCloudShared& queries, knn::KNNResult& result,
        const std::vector<sycl::event>& depends = std::vector<sycl::event>(),
        const TransformMatrix& transT = TransformMatrix::Identity()) const override {
        switch (num_neighbor_voxels_) {
            case 7:
                return nearest_neighbor_search_async<7>(queries, result, depends, transT);
            case 19:
                return nearest_neighbor_search_async<19>(queries, result, depends, transT);
            default:
                return nearest_neighbor_search_async<27>(queries, result, depends, transT);
        }
    }

    // Prevent name hiding of the base class non-template nearest_neighbor_search.
    using knn::NearestNeighborBase::nearest_neighbor_search;

private:
    using atomic_ref_float = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint32_t = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    using atomic_ref_uint64_t = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;

    /// @brief Core voxel data containing position information (16 bytes)
    struct VoxelCoreData {
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        uint32_t count = 0U;
    };
    static_assert(sizeof(VoxelCoreData) == 16, "VoxelCoreData must be 16 bytes for optimal memory layout");

    /// @brief Color data for RGB information (16 bytes)
    struct VoxelColorData {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_a = 0.0f;
    };
    static_assert(sizeof(VoxelColorData) == 16, "VoxelColorData must be 16 bytes for optimal memory layout");

    /// @brief Covariance data stored as the upper triangular 3x3 block (24 bytes)
    struct VoxelCovarianceData {
        float sum_xx = 0.0f;
        float sum_xy = 0.0f;
        float sum_xz = 0.0f;
        float sum_yy = 0.0f;
        float sum_yz = 0.0f;
        float sum_zz = 0.0f;
    };
    static_assert(sizeof(VoxelCovarianceData) == 24, "VoxelCovarianceData must be 24 bytes");

    /// @brief Intensity data for reflectivity information (4 bytes)
    struct VoxelIntensityData {
        float sum_intensity = 0.0f;
    };
    static_assert(sizeof(VoxelIntensityData) == 4, "VoxelIntensityData must be 4 bytes for optimal memory layout");

    /// @brief Accumulator types for local workgroup reduction.
    /// @note These are type aliases to the corresponding Data structs since they share
    ///       identical field layouts. The semantic difference (persistent storage vs
    ///       temporary accumulation) is preserved through naming and usage context.
    using VoxelCoreAccumulator = VoxelCoreData;
    using VoxelCovarianceAccumulator = VoxelCovarianceData;
    using VoxelColorAccumulator = VoxelColorData;
    using VoxelIntensityAccumulator = VoxelIntensityData;

    struct VoxelLocalData {
        uint64_t voxel_idx = VoxelConstants::invalid_coord;
        VoxelCoreAccumulator core_acc;
        VoxelCovarianceAccumulator covariance_acc;
        VoxelColorAccumulator color_acc;
        VoxelIntensityAccumulator intensity_acc;
    };

    SYCL_EXTERNAL static void atomic_add_voxel_data(const VoxelCoreAccumulator& core_src,
                                                    const VoxelCovarianceAccumulator& covariance_src,
                                                    const VoxelColorAccumulator& color_src,
                                                    const VoxelIntensityAccumulator& intensity_src,
                                                    VoxelCoreData& core_dst, VoxelCovarianceData& covariance_dst,
                                                    VoxelColorData& color_dst, VoxelIntensityData& intensity_dst,
                                                    bool has_cov, bool has_rgb, bool has_intensity) {
        // Core data - position accumulation
        atomic_ref_float(core_dst.sum_x).fetch_add(core_src.sum_x);
        atomic_ref_float(core_dst.sum_y).fetch_add(core_src.sum_y);
        atomic_ref_float(core_dst.sum_z).fetch_add(core_src.sum_z);
        atomic_ref_uint32_t(core_dst.count).fetch_add(core_src.count);

        if (has_cov) {
            atomic_ref_float(covariance_dst.sum_xx).fetch_add(covariance_src.sum_xx);
            atomic_ref_float(covariance_dst.sum_xy).fetch_add(covariance_src.sum_xy);
            atomic_ref_float(covariance_dst.sum_xz).fetch_add(covariance_src.sum_xz);
            atomic_ref_float(covariance_dst.sum_yy).fetch_add(covariance_src.sum_yy);
            atomic_ref_float(covariance_dst.sum_yz).fetch_add(covariance_src.sum_yz);
            atomic_ref_float(covariance_dst.sum_zz).fetch_add(covariance_src.sum_zz);
        }

        // Color data (only if present)
        if (has_rgb) {
            atomic_ref_float(color_dst.sum_r).fetch_add(color_src.sum_r);
            atomic_ref_float(color_dst.sum_g).fetch_add(color_src.sum_g);
            atomic_ref_float(color_dst.sum_b).fetch_add(color_src.sum_b);
            atomic_ref_float(color_dst.sum_a).fetch_add(color_src.sum_a);
        }

        // Intensity data (only if present)
        if (has_intensity) {
            atomic_ref_float(intensity_dst.sum_intensity).fetch_add(intensity_src.sum_intensity);
        }
    }

    SYCL_EXTERNAL static void atomic_store_timestamp(uint32_t old_timestamp, uint32_t& new_timestamp) {
        // update
        atomic_ref_uint32_t(new_timestamp).store(old_timestamp);
    }

    SYCL_EXTERNAL static void compute_averaged_attributes(
        const VoxelCoreData& core, const VoxelCovarianceData& covariance, const VoxelColorData& color,
        const VoxelIntensityData& intensity, size_t output_idx, PointType* pt_output, Covariance* cov_output,
        RGBType* rgb_output, float* intensity_output, CovarianceAggregationMode covariance_mode,
        uint32_t min_num_point = 1) {
        if (core.count >= min_num_point) {
            const float inv_count = 1.0f / static_cast<float>(core.count);
            pt_output[output_idx].x() = core.sum_x * inv_count;
            pt_output[output_idx].y() = core.sum_y * inv_count;
            pt_output[output_idx].z() = core.sum_z * inv_count;
            pt_output[output_idx].w() = 1.0f;
            if (cov_output) {
                cov_output[output_idx].setZero();
                Eigen::Matrix3f cov3 = Eigen::Matrix3f::Zero();
                cov3(0, 0) = covariance.sum_xx * inv_count;
                cov3(0, 1) = covariance.sum_xy * inv_count;
                cov3(1, 0) = cov3(0, 1);
                cov3(0, 2) = covariance.sum_xz * inv_count;
                cov3(2, 0) = cov3(0, 2);
                cov3(1, 1) = covariance.sum_yy * inv_count;
                cov3(1, 2) = covariance.sum_yz * inv_count;
                cov3(2, 1) = cov3(1, 2);
                cov3(2, 2) = covariance.sum_zz * inv_count;
                if (covariance_mode == CovarianceAggregationMode::LOG_EUCLIDEAN) {
                    cov3 = eigen_utils::exp_spd_3x3(cov3);
                }
                cov_output[output_idx].block<3, 3>(0, 0) = cov3;
            }
            if (rgb_output) {
                rgb_output[output_idx].x() = color.sum_r * inv_count;
                rgb_output[output_idx].y() = color.sum_g * inv_count;
                rgb_output[output_idx].z() = color.sum_b * inv_count;
                rgb_output[output_idx].w() = color.sum_a * inv_count;
            }
            if (intensity_output) {
                intensity_output[output_idx] = intensity.sum_intensity * inv_count;
            }
        } else {
            pt_output[output_idx].setZero();
            if (cov_output) {
                cov_output[output_idx].setZero();
            }
            if (rgb_output) {
                rgb_output[output_idx].setZero();
            }
            if (intensity_output) {
                intensity_output[output_idx] = 0.0f;
            }
        }
    }

    SYCL_EXTERNAL static bool centroid_inside_bbox(const VoxelCoreData& core, float min_x, float min_y, float min_z,
                                                   float max_x, float max_y, float max_z) {
        if (core.count == 0U) {
            return false;
        }

        const float inv_count = 1.0f / static_cast<float>(core.count);
        const float centroid_x = core.sum_x * inv_count;
        const float centroid_y = core.sum_y * inv_count;
        const float centroid_z = core.sum_z * inv_count;

        return (centroid_x >= min_x && centroid_x <= max_x) && (centroid_y >= min_y && centroid_y <= max_y) &&
               (centroid_z >= min_z && centroid_z <= max_z);
    }

    SYCL_EXTERNAL static bool should_include_voxel(uint64_t key, const VoxelCoreData& core, uint32_t min_num_point,
                                                   float min_x, float min_y, float min_z, float max_x, float max_y,
                                                   float max_z) {
        if (key == VoxelConstants::invalid_coord || core.count < min_num_point) {
            return false;
        }

        return centroid_inside_bbox(core, min_x, min_y, min_z, max_x, max_y, max_z);
    }

    SYCL_EXTERNAL static void rotate_covariance_upper_triangle(const Covariance& cov,
                                                               const std::array<sycl::vec<float, 4>, 4>& trans,
                                                               VoxelCovarianceAccumulator& output) {
        const float cxx = cov(0, 0);
        const float cxy = cov(0, 1);
        const float cxz = cov(0, 2);
        const float cyy = cov(1, 1);
        const float cyz = cov(1, 2);
        const float czz = cov(2, 2);

        const float r00 = trans[0].x();
        const float r01 = trans[0].y();
        const float r02 = trans[0].z();
        const float r10 = trans[1].x();
        const float r11 = trans[1].y();
        const float r12 = trans[1].z();
        const float r20 = trans[2].x();
        const float r21 = trans[2].y();
        const float r22 = trans[2].z();

        const float a00 = sycl::fma(r02, cxz, sycl::fma(r01, cxy, r00 * cxx));
        const float a01 = sycl::fma(r02, cyz, sycl::fma(r01, cyy, r00 * cxy));
        const float a02 = sycl::fma(r02, czz, sycl::fma(r01, cyz, r00 * cxz));
        const float a10 = sycl::fma(r12, cxz, sycl::fma(r11, cxy, r10 * cxx));
        const float a11 = sycl::fma(r12, cyz, sycl::fma(r11, cyy, r10 * cxy));
        const float a12 = sycl::fma(r12, czz, sycl::fma(r11, cyz, r10 * cxz));
        const float a20 = sycl::fma(r22, cxz, sycl::fma(r21, cxy, r20 * cxx));
        const float a21 = sycl::fma(r22, cyz, sycl::fma(r21, cyy, r20 * cxy));
        const float a22 = sycl::fma(r22, czz, sycl::fma(r21, cyz, r20 * cxz));

        output.sum_xx = sycl::fma(a02, r02, sycl::fma(a01, r01, a00 * r00));
        output.sum_xy = sycl::fma(a02, r12, sycl::fma(a01, r11, a00 * r10));
        output.sum_xz = sycl::fma(a02, r22, sycl::fma(a01, r21, a00 * r20));
        output.sum_yy = sycl::fma(a12, r12, sycl::fma(a11, r11, a10 * r10));
        output.sum_yz = sycl::fma(a12, r22, sycl::fma(a11, r21, a10 * r20));
        output.sum_zz = sycl::fma(a22, r22, sycl::fma(a21, r21, a20 * r20));
    }

    SYCL_EXTERNAL static void encode_covariance_for_aggregation(VoxelCovarianceAccumulator& covariance,
                                                                CovarianceAggregationMode mode) {
        if (mode != CovarianceAggregationMode::LOG_EUCLIDEAN) {
            return;
        }

        Eigen::Matrix3f cov3 = Eigen::Matrix3f::Zero();
        cov3(0, 0) = covariance.sum_xx;
        cov3(0, 1) = covariance.sum_xy;
        cov3(1, 0) = covariance.sum_xy;
        cov3(0, 2) = covariance.sum_xz;
        cov3(2, 0) = covariance.sum_xz;
        cov3(1, 1) = covariance.sum_yy;
        cov3(1, 2) = covariance.sum_yz;
        cov3(2, 1) = covariance.sum_yz;
        cov3(2, 2) = covariance.sum_zz;
        const Eigen::Matrix3f log_cov3 = eigen_utils::log_spd_3x3(cov3);
        covariance.sum_xx = log_cov3(0, 0);
        covariance.sum_xy = log_cov3(0, 1);
        covariance.sum_xz = log_cov3(0, 2);
        covariance.sum_yy = log_cov3(1, 1);
        covariance.sum_yz = log_cov3(1, 2);
        covariance.sum_zz = log_cov3(2, 2);
    }

    sycl_utils::DeviceQueue queue_;
    float voxel_size_ = 0.0f;
    float voxel_size_inv_ = 0.0f;
    inline static constexpr std::array<size_t, 11> kCapacityCandidates = {
        30029, 60013, 120011, 240007, 480013, 960017, 1920001, 3840007, 7680017, 15360013, 30720007};  // prime number
    size_t capacity_ = kCapacityCandidates[0];

    shared_vector_ptr<uint64_t> key_ptr_ = nullptr;
    shared_vector_ptr<VoxelCoreData> core_data_ptr_ = nullptr;
    shared_vector_ptr<VoxelCovarianceData> covariance_data_ptr_ = nullptr;
    shared_vector_ptr<VoxelColorData> color_data_ptr_ = nullptr;
    shared_vector_ptr<VoxelIntensityData> intensity_data_ptr_ = nullptr;
    shared_vector_ptr<uint32_t> last_update_ptr_ = nullptr;
    shared_vector_ptr<uint8_t> valid_flags_ptr_ = nullptr;
    common::PrefixSum::Ptr prefix_sum_ = nullptr;

    uint32_t staleness_counter_ = 0;
    uint32_t max_staleness_ = 100;
    uint32_t remove_old_data_cycle_ = 10;

    const size_t max_probe_length_ = 100;

    float rehash_threshold_ = 0.7f;

    size_t wg_size_add_point_cloud_ = 128UL;

    size_t num_neighbor_voxels_ = 27;  ///< Neighbor voxel count used by non-template override (7, 19, or 27)
    size_t voxel_num_ = 0;
    bool has_cov_data_ = false;
    bool has_rgb_data_ = false;
    bool has_intensity_data_ = false;
    uint32_t min_num_point_ = 1U;
    CovarianceAggregationMode covariance_aggregation_mode_ = CovarianceAggregationMode::ARITHMETIC;

    void update_voxel_num_and_flags(size_t new_voxel_num) {
        this->voxel_num_ = new_voxel_num;
        if (this->voxel_num_ == 0) {
            this->has_cov_data_ = false;
            this->has_rgb_data_ = false;
            this->has_intensity_data_ = false;
        }
    }

    void allocate_storage(size_t new_capacity) {
        this->key_ptr_ =
            std::make_shared<shared_vector<uint64_t>>(new_capacity, VoxelConstants::invalid_coord, *this->queue_.ptr);
        this->core_data_ptr_ =
            std::make_shared<shared_vector<VoxelCoreData>>(new_capacity, VoxelCoreData{}, *this->queue_.ptr);
        this->covariance_data_ptr_ = std::make_shared<shared_vector<VoxelCovarianceData>>(
            new_capacity, VoxelCovarianceData{}, *this->queue_.ptr);
        this->color_data_ptr_ =
            std::make_shared<shared_vector<VoxelColorData>>(new_capacity, VoxelColorData{}, *this->queue_.ptr);
        this->intensity_data_ptr_ =
            std::make_shared<shared_vector<VoxelIntensityData>>(new_capacity, VoxelIntensityData{}, *this->queue_.ptr);
        this->last_update_ptr_ = std::make_shared<shared_vector<uint32_t>>(new_capacity, 0U, *this->queue_.ptr);

        this->capacity_ = new_capacity;
    }

    size_t get_next_capacity_value() const {
        // Select the next pre-defined capacity to keep probing statistics stable.
        for (const auto candidate : kCapacityCandidates) {
            if (candidate > this->capacity_) {
                return candidate;
            }
        }
        std::cout << "[Caution] VoxelHashMap reached the maximum predefined capacity (" << this->capacity_
                  << "). Further growth is not available." << std::endl;
        return this->capacity_;
    }

    size_t compute_wg_size_add_point_cloud() const {
        const size_t max_work_group_size =
            this->queue_.get_device().get_info<sycl::info::device::max_work_group_size>();
        const size_t compute_units = this->queue_.get_device().get_info<sycl::info::device::max_compute_units>();
        if (this->queue_.is_nvidia()) {
            // NVIDIA:
            return std::min(max_work_group_size, 64UL);
        } else if (this->queue_.is_intel() && this->queue_.is_gpu()) {
            // Intel iGPU:
            return std::min(max_work_group_size, compute_units * 8UL);
        } else if (this->queue_.is_cpu()) {
            // CPU:
            return std::min(max_work_group_size, compute_units * 100UL);
        }
        return 128UL;
    }

    size_t compute_local_size_for_add_point_cloud(bool has_cov) const {
        if (!has_cov) {
            return this->wg_size_add_point_cloud_;
        }
        if (this->queue_.is_nvidia()) {
            return std::min(this->wg_size_add_point_cloud_, 32UL);
        }
        return std::max<size_t>(1UL, this->wg_size_add_point_cloud_ / 2UL);
    }

    template <typename Func>
    SYCL_EXTERNAL static void global_reduction(const VoxelLocalData& data, uint64_t* key_ptr, VoxelCoreData* core_ptr,
                                               VoxelCovarianceData* covariance_ptr, VoxelColorData* color_ptr,
                                               VoxelIntensityData* intensity_ptr, uint32_t current,
                                               uint32_t* last_update_ptr, size_t max_probe, size_t capacity,
                                               Func voxel_num_counter, bool has_cov, bool has_rgb, bool has_intensity) {
        const uint64_t voxel_hash = data.voxel_idx;
        if (voxel_hash == VoxelConstants::invalid_coord) return;

        for (size_t j = 0; j < max_probe; ++j) {
            const size_t slot_idx = compute_slot_id(voxel_hash, j, capacity);

            uint64_t expected = VoxelConstants::invalid_coord;
            if (atomic_ref_uint64_t(key_ptr[slot_idx]).compare_exchange_strong(expected, voxel_hash)) {
                // count up num of voxel
                voxel_num_counter(1U);

                atomic_add_voxel_data(data.core_acc, data.covariance_acc, data.color_acc, data.intensity_acc,
                                      core_ptr[slot_idx], covariance_ptr[slot_idx], color_ptr[slot_idx],
                                      intensity_ptr[slot_idx], has_cov, has_rgb, has_intensity);
                atomic_store_timestamp(current, last_update_ptr[slot_idx]);
                break;

            } else if (expected == voxel_hash) {
                atomic_add_voxel_data(data.core_acc, data.covariance_acc, data.color_acc, data.intensity_acc,
                                      core_ptr[slot_idx], covariance_ptr[slot_idx], color_ptr[slot_idx],
                                      intensity_ptr[slot_idx], has_cov, has_rgb, has_intensity);
                atomic_store_timestamp(current, last_update_ptr[slot_idx]);
                break;
            }
        }
    }

    SYCL_EXTERNAL static uint64_t hash2(uint64_t voxel_hash, size_t capacity) {
        return (capacity - 2) - (voxel_hash % (capacity - 2));
    }
    SYCL_EXTERNAL static size_t compute_slot_id(uint64_t voxel_hash, size_t probe, size_t capacity) {
        return (voxel_hash + probe * hash2(voxel_hash, capacity)) % capacity;
    }

    void add_point_cloud_impl(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) {
        const size_t N = cloud.size();
        if (N == 0) return;

        const bool has_rgb = cloud.has_rgb();
        const bool has_intensity = cloud.has_intensity();
        const bool has_cov = cloud.has_cov();
        this->has_cov_data_ |= has_cov;
        this->has_rgb_data_ |= has_rgb;
        this->has_intensity_data_ |= has_intensity;

        // add to voxel hash map
        shared_vector<uint32_t> voxel_num_vec(1, this->voxel_num_, *this->queue_.ptr);

        auto reduction_event = this->queue_.ptr->submit([&](sycl::handler& h) {
            // Use the configured work-group size as the kernel's local size.
            const size_t local_size = this->compute_local_size_for_add_point_cloud(has_cov);
            const size_t num_work_groups = (N + local_size - 1) / local_size;
            const size_t global_size = num_work_groups * local_size;

            // Allocate local memory for work group operations
            const auto local_voxel_data = sycl::local_accessor<VoxelLocalData>(local_size, h);
            const auto trans = eigen_utils::to_sycl_vec(sensor_pose.matrix());

            size_t power_of_2 = 1;
            while (power_of_2 < local_size) {
                power_of_2 *= 2;
            }

            // memory ptr
            const auto key_ptr = this->key_ptr_->data();
            const auto core_ptr = this->core_data_ptr_->data();
            const auto covariance_ptr = this->covariance_data_ptr_->data();
            const auto color_ptr = this->color_data_ptr_->data();
            const auto intensity_data_ptr = this->intensity_data_ptr_->data();
            const auto last_update_ptr = this->last_update_ptr_->data();

            const auto point_ptr = cloud.points_ptr();
            const auto cov_ptr = has_cov ? cloud.covs_ptr() : static_cast<Covariance*>(nullptr);
            const auto rgb_ptr = has_rgb ? cloud.rgb_ptr() : static_cast<RGBType*>(nullptr);
            const auto intensity_ptr = has_intensity ? cloud.intensities_ptr() : static_cast<float*>(nullptr);

            const auto vs_inv = this->voxel_size_inv_;
            const auto cp = this->capacity_;
            const auto current = this->staleness_counter_;
            const auto max_probe = this->max_probe_length_;
            const auto covariance_mode = this->covariance_aggregation_mode_;

            auto load_entry = [=](VoxelLocalData& entry, const size_t idx) {
                const PointType local_point = point_ptr[idx];
                PointType world_point;
                transform::kernel::transform_point(local_point, world_point, trans);

                const auto voxel_hash = kernel::compute_voxel_bit(world_point, vs_inv);

                entry.voxel_idx = voxel_hash;
                entry.core_acc.sum_x = world_point.x();
                entry.core_acc.sum_y = world_point.y();
                entry.core_acc.sum_z = world_point.z();
                entry.core_acc.count = 1U;

                entry.covariance_acc.sum_xx = 0.0f;
                entry.covariance_acc.sum_xy = 0.0f;
                entry.covariance_acc.sum_xz = 0.0f;
                entry.covariance_acc.sum_yy = 0.0f;
                entry.covariance_acc.sum_yz = 0.0f;
                entry.covariance_acc.sum_zz = 0.0f;

                entry.color_acc.sum_r = 0.0f;
                entry.color_acc.sum_g = 0.0f;
                entry.color_acc.sum_b = 0.0f;
                entry.color_acc.sum_a = 0.0f;

                entry.intensity_acc.sum_intensity = 0.0f;

                if (has_cov && cov_ptr) {
                    rotate_covariance_upper_triangle(cov_ptr[idx], trans, entry.covariance_acc);
                    encode_covariance_for_aggregation(entry.covariance_acc, covariance_mode);
                }

                if (has_rgb && rgb_ptr) {
                    const auto color = rgb_ptr[idx];
                    entry.color_acc.sum_r = color.x();
                    entry.color_acc.sum_g = color.y();
                    entry.color_acc.sum_b = color.z();
                    entry.color_acc.sum_a = color.w();
                }

                if (has_intensity && intensity_ptr) {
                    entry.intensity_acc.sum_intensity = intensity_ptr[idx];
                }
            };

            auto combine_entry = [=](VoxelLocalData& dst, const VoxelLocalData& src) {
                dst.core_acc.sum_x += src.core_acc.sum_x;
                dst.core_acc.sum_y += src.core_acc.sum_y;
                dst.core_acc.sum_z += src.core_acc.sum_z;
                dst.core_acc.count += src.core_acc.count;
                if (has_cov) {
                    dst.covariance_acc.sum_xx += src.covariance_acc.sum_xx;
                    dst.covariance_acc.sum_xy += src.covariance_acc.sum_xy;
                    dst.covariance_acc.sum_xz += src.covariance_acc.sum_xz;
                    dst.covariance_acc.sum_yy += src.covariance_acc.sum_yy;
                    dst.covariance_acc.sum_yz += src.covariance_acc.sum_yz;
                    dst.covariance_acc.sum_zz += src.covariance_acc.sum_zz;
                }
                if (has_rgb) {
                    dst.color_acc.sum_r += src.color_acc.sum_r;
                    dst.color_acc.sum_g += src.color_acc.sum_g;
                    dst.color_acc.sum_b += src.color_acc.sum_b;
                    dst.color_acc.sum_a += src.color_acc.sum_a;
                }
                if (has_intensity) {
                    dst.intensity_acc.sum_intensity += src.intensity_acc.sum_intensity;
                }
            };

            auto reset_entry = [](VoxelLocalData& entry) {
                entry.voxel_idx = VoxelConstants::invalid_coord;
                entry.core_acc = VoxelCoreAccumulator{};
                entry.covariance_acc = VoxelCovarianceAccumulator{};
                entry.color_acc = VoxelColorAccumulator{};
                entry.intensity_acc = VoxelIntensityAccumulator{};
            };

            // Configure key accessors and comparators for the shared reduction helpers.
            auto key_of_entry = [](const VoxelLocalData& entry) { return entry.voxel_idx; };
            auto compare_keys = [](uint64_t lhs, uint64_t rhs) { return lhs < rhs; };
            auto equal_keys = [](uint64_t lhs, uint64_t rhs) { return lhs == rhs; };

            auto range = sycl::nd_range<1>(global_size, local_size);

            if (this->queue_.is_nvidia()) {
                auto voxel_num = sycl::reduction(voxel_num_vec.data(), sycl::plus<uint32_t>());

                h.parallel_for(range, voxel_num, [=](sycl::nd_item<1> item, auto& voxel_num_arg) {
                    const size_t global_id = item.get_global_id(0);
                    const size_t local_id = item.get_local_id(0);

                    // Reduction on workgroup
                    common::local_reduction<true, VoxelLocalData>(
                        local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(), N, local_size, power_of_2,
                        item, load_entry, combine_entry, reset_entry, VoxelConstants::invalid_coord, key_of_entry,
                        compare_keys, equal_keys);

                    if (global_id >= N) return;

                    // Reduction on global memory
                    global_reduction(
                        local_voxel_data[local_id], key_ptr, core_ptr, covariance_ptr, color_ptr, intensity_data_ptr,
                        current, last_update_ptr, max_probe, cp, [&](uint32_t num) { voxel_num_arg += num; }, has_cov,
                        has_rgb, has_intensity);
                });
            } else {
                auto voxel_num_ptr = voxel_num_vec.data();

                h.parallel_for(range, [=](sycl::nd_item<1> item) {
                    const size_t global_id = item.get_global_id(0);
                    const size_t local_id = item.get_local_id(0);

                    // Reduction on workgroup
                    common::local_reduction<false, VoxelLocalData>(
                        local_voxel_data.get_multi_ptr<sycl::access::decorated::no>().get(), N, local_size, power_of_2,
                        item, load_entry, combine_entry, reset_entry, VoxelConstants::invalid_coord, key_of_entry,
                        compare_keys, equal_keys);

                    if (global_id >= N) return;

                    // Reduction on global memory
                    global_reduction(
                        local_voxel_data[local_id], key_ptr, core_ptr, covariance_ptr, color_ptr, intensity_data_ptr,
                        current, last_update_ptr, max_probe, cp,
                        [&](uint32_t num) { atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(num); }, has_cov, has_rgb,
                        has_intensity);
                });
            }
        });
        reduction_event.wait_and_throw();
        this->voxel_num_ = static_cast<size_t>(voxel_num_vec.at(0));
    }

    void remove_old_data_impl() {
        if (this->staleness_counter_ <= this->max_staleness_) return;

        shared_vector<uint32_t> voxel_num_vec(1, 0, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t N = this->capacity_;
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                // memory ptr
                const auto key_ptr = this->key_ptr_->data();
                const auto core_ptr = this->core_data_ptr_->data();
                const auto covariance_ptr = this->covariance_data_ptr_->data();
                const auto color_ptr = this->color_data_ptr_->data();
                const auto intensity_ptr = this->intensity_data_ptr_->data();
                const auto last_update_ptr = this->last_update_ptr_->data();
                auto clear_function = [](uint64_t& key, VoxelCoreData& core, VoxelCovarianceData& covariance,
                                         VoxelColorData& color, VoxelIntensityData& intensity, uint32_t& last_update) {
                    key = VoxelConstants::invalid_coord;
                    core = VoxelCoreData{};
                    covariance = VoxelCovarianceData{};
                    color = VoxelColorData{};
                    intensity = VoxelIntensityData{};
                    last_update = 0;
                };

                const uint32_t remove_staleness = this->staleness_counter_ - this->max_staleness_;
                auto range = sycl::nd_range<1>(global_size, work_group_size);
                const auto voxel_num_ptr = voxel_num_vec.data();

                h.parallel_for(range, [=](sycl::nd_item<1> item) {
                    const uint32_t i = item.get_global_id(0);
                    if (i >= N) return;

                    const auto voxel_hash = key_ptr[i];
                    if (voxel_hash == VoxelConstants::invalid_coord) return;

                    const auto last_update = last_update_ptr[i];
                    if (last_update >= remove_staleness) {
                        // count up num of voxel
                        atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(1U);
                        return;
                    }
                    clear_function(key_ptr[i], core_ptr[i], covariance_ptr[i], color_ptr[i], intensity_ptr[i],
                                   last_update_ptr[i]);
                });
            })
            .wait_and_throw();
        this->update_voxel_num_and_flags(static_cast<size_t>(voxel_num_vec.at(0)));
    }

    void rehash(size_t new_capacity) {
        if (this->capacity_ >= new_capacity) return;

        const auto old_capacity = this->capacity_;

        // old pointer
        auto old_key_ptr = this->key_ptr_;
        auto old_core_ptr = this->core_data_ptr_;
        auto old_covariance_ptr = this->covariance_data_ptr_;
        auto old_color_ptr = this->color_data_ptr_;
        auto old_intensity_ptr = this->intensity_data_ptr_;
        auto old_last_update_ptr = this->last_update_ptr_;

        // make new
        this->allocate_storage(new_capacity);

        shared_vector<uint32_t> voxel_num_vec(1, 0, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t N = old_capacity;
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                // memory ptr
                const auto old_key = old_key_ptr->data();
                const auto old_core = old_core_ptr->data();
                const auto old_covariance = old_covariance_ptr->data();
                const auto old_color = old_color_ptr->data();
                const auto old_intensity = old_intensity_ptr->data();
                const auto old_last_update = old_last_update_ptr->data();
                const auto new_key = this->key_ptr_->data();
                const auto new_core = this->core_data_ptr_->data();
                const auto new_covariance = this->covariance_data_ptr_->data();
                const auto new_color = this->color_data_ptr_->data();
                const auto new_intensity = this->intensity_data_ptr_->data();
                const auto new_last_update = this->last_update_ptr_->data();

                const auto new_cp = new_capacity;
                const auto max_probe = this->max_probe_length_;
                const auto has_cov = this->has_cov_data_;
                const auto has_rgb = this->has_rgb_data_;
                const auto has_intensity = this->has_intensity_data_;
                auto range = sycl::nd_range<1>(global_size, work_group_size);

                auto voxel_num_ptr = voxel_num_vec.data();

                h.parallel_for(range, [=](sycl::nd_item<1> item) {
                    const uint32_t i = item.get_global_id(0);
                    if (i >= N) return;

                    const uint64_t key = old_key[i];
                    if (key == VoxelConstants::invalid_coord) return;

                    VoxelLocalData data;
                    data.voxel_idx = key;
                    data.core_acc.sum_x = old_core[i].sum_x;
                    data.core_acc.sum_y = old_core[i].sum_y;
                    data.core_acc.sum_z = old_core[i].sum_z;
                    data.core_acc.count = old_core[i].count;
                    if (has_cov) {
                        data.covariance_acc.sum_xx = old_covariance[i].sum_xx;
                        data.covariance_acc.sum_xy = old_covariance[i].sum_xy;
                        data.covariance_acc.sum_xz = old_covariance[i].sum_xz;
                        data.covariance_acc.sum_yy = old_covariance[i].sum_yy;
                        data.covariance_acc.sum_yz = old_covariance[i].sum_yz;
                        data.covariance_acc.sum_zz = old_covariance[i].sum_zz;
                    }
                    if (has_rgb) {
                        data.color_acc.sum_r = old_color[i].sum_r;
                        data.color_acc.sum_g = old_color[i].sum_g;
                        data.color_acc.sum_b = old_color[i].sum_b;
                        data.color_acc.sum_a = old_color[i].sum_a;
                    }
                    if (has_intensity) {
                        data.intensity_acc.sum_intensity = old_intensity[i].sum_intensity;
                    }

                    global_reduction(
                        data, new_key, new_core, new_covariance, new_color, new_intensity, old_last_update[i],
                        new_last_update, max_probe, new_cp,
                        [&](uint32_t num) { atomic_ref_uint32_t(voxel_num_ptr[0]).fetch_add(num); }, has_cov, has_rgb,
                        has_intensity);
                });
            })
            .wait_and_throw();
        this->update_voxel_num_and_flags(static_cast<size_t>(voxel_num_vec.at(0)));
    }

    size_t downsampling_impl(PointContainerShared& result, const Eigen::Vector3f& center, const float distance,
                             Covariance* cov_output_ptr = nullptr, RGBType* rgb_output_ptr = nullptr,
                             float* intensity_output_ptr = nullptr) {
        // Compute the axis-aligned bounding box around the requested query center.
        const float bbox_min_x = center.x() - distance;
        const float bbox_min_y = center.y() - distance;
        const float bbox_min_z = center.z() - distance;
        const float bbox_max_x = center.x() + distance;
        const float bbox_max_y = center.y() + distance;
        const float bbox_max_z = center.z() + distance;

        const bool is_nvidia = this->queue_.is_nvidia();
        size_t filtered_voxel_count = 0;

        if (is_nvidia) {
            // compute valid flags
            if (this->valid_flags_ptr_->size() < this->capacity_) {
                this->valid_flags_ptr_->resize(this->capacity_);
            }
            this->queue_.ptr
                ->submit([&](sycl::handler& h) {
                    const size_t cp = this->capacity_;
                    const size_t work_group_size = this->queue_.get_work_group_size();
                    const size_t global_size = this->queue_.get_global_size(cp);

                    // memory ptr
                    const auto valid_flags = this->valid_flags_ptr_->data();
                    const auto key_ptr = this->key_ptr_->data();
                    const auto core_ptr = this->core_data_ptr_->data();
                    const auto min_num_point = this->min_num_point_;

                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t global_id = item.get_global_id(0);
                        if (global_id >= cp) return;

                        const auto key = key_ptr[global_id];
                        const auto core = core_ptr[global_id];

                        if (!should_include_voxel(key, core, min_num_point, bbox_min_x, bbox_min_y, bbox_min_z,
                                                  bbox_max_x, bbox_max_y, bbox_max_z)) {
                            valid_flags[global_id] = 0U;
                            return;
                        }

                        valid_flags[global_id] = 1U;
                    });
                })
                .wait_and_throw();
            // compute prefix sum
            filtered_voxel_count = this->prefix_sum_->compute(*this->valid_flags_ptr_);
        }

        // voxel hash map to point cloud
        result.resize(this->voxel_num_);
        shared_vector<uint32_t> point_num_vec(1, 0, *this->queue_.ptr);

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const auto cp = this->capacity_;
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(cp);

                // memory ptr
                const auto core_ptr = this->core_data_ptr_->data();
                const auto covariance_ptr = this->covariance_data_ptr_->data();
                const auto color_ptr = this->color_data_ptr_->data();
                const auto intensity_data_ptr = this->intensity_data_ptr_->data();
                const auto result_ptr = result.data();
                const auto cov_output = cov_output_ptr;
                // Optional output arrays for aggregated RGB colors and intensity values.
                const auto rgb_output = rgb_output_ptr;
                const auto intensity_output = intensity_output_ptr;
                const auto covariance_mode = this->covariance_aggregation_mode_;

                if (is_nvidia) {
                    const auto flag_ptr = this->valid_flags_ptr_->data();
                    const auto prefix_sum_ptr = this->prefix_sum_->get_prefix_sum().data();
                    const auto min_num_point = this->min_num_point_;

                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t i = item.get_global_id(0);
                        if (i >= cp) return;

                        if (flag_ptr[i] == 1) {
                            const size_t output_idx = prefix_sum_ptr[i] - 1;
                            const auto core = core_ptr[i];
                            const auto covariance = covariance_ptr[i];
                            const auto color = color_ptr[i];
                            const auto intensity = intensity_data_ptr[i];

                            compute_averaged_attributes(core, covariance, color, intensity, output_idx, result_ptr,
                                                        cov_output, rgb_output, intensity_output, covariance_mode,
                                                        min_num_point);
                        }
                    });

                } else {
                    const auto key_ptr = this->key_ptr_->data();

                    const auto point_num_ptr = point_num_vec.data();
                    const auto min_num_point = this->min_num_point_;
                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const auto i = item.get_global_id(0);
                        if (i >= cp) return;

                        const auto key = key_ptr[i];
                        const auto core = core_ptr[i];

                        if (!should_include_voxel(key, core, min_num_point, bbox_min_x, bbox_min_y, bbox_min_z,
                                                  bbox_max_x, bbox_max_y, bbox_max_z)) {
                            return;
                        }

                        const auto output_idx = atomic_ref_uint32_t(point_num_ptr[0]).fetch_add(1U);

                        const auto covariance = covariance_ptr[i];
                        const auto color = color_ptr[i];
                        const auto intensity = intensity_data_ptr[i];

                        compute_averaged_attributes(core, covariance, color, intensity, output_idx, result_ptr,
                                                    cov_output, rgb_output, intensity_output, covariance_mode,
                                                    min_num_point);
                    });
                }
            })
            .wait_and_throw();

        if (!is_nvidia) {
            filtered_voxel_count = static_cast<size_t>(point_num_vec.at(0));
        }

        return filtered_voxel_count;
    }

    /// @brief Encode offset-adjusted voxel coordinates into a 64-bit hash key.
    SYCL_EXTERNAL static inline uint64_t encode_voxel_coords(const int64_t cx, const int64_t cy, const int64_t cz) {
        return (static_cast<uint64_t>(cx & VoxelConstants::coord_bit_mask)
                << (VoxelConstants::coord_bit_size * CartesianCoordComponent::X)) |
               (static_cast<uint64_t>(cy & VoxelConstants::coord_bit_mask)
                << (VoxelConstants::coord_bit_size * CartesianCoordComponent::Y)) |
               (static_cast<uint64_t>(cz & VoxelConstants::coord_bit_mask)
                << (VoxelConstants::coord_bit_size * CartesianCoordComponent::Z));
    }

    /// @brief SYCL kernel implementation of nearest neighbor search over neighboring voxels.
    template <size_t NUM_NEIGHBOR_VOXELS>
    sycl_utils::events nearest_neighbor_search_impl(const PointCloudShared& queries, knn::KNNResult& result,
                                                    const std::vector<sycl::event>& depends,
                                                    const TransformMatrix& transT = TransformMatrix::Identity()) const {
        const size_t query_size = queries.size();
        result.allocate(this->queue_, query_size, 1);  // k = 1

        if (query_size == 0 || this->voxel_num_ == 0) {
            return {};
        }

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            h.depends_on(depends);

            const auto query_ptr = queries.points_ptr();
            const auto key_ptr = this->key_ptr_->data();
            const auto core_ptr = this->core_data_ptr_->data();
            const float vs_inv = this->voxel_size_inv_;
            const size_t capacity = this->capacity_;
            const size_t max_probe = this->max_probe_length_;
            const uint32_t min_pts = this->min_num_point_;
            auto out_indices = result.indices->data();
            auto out_distances = result.distances->data();
            const auto trans_vec = eigen_utils::to_sycl_vec(transT);

            h.parallel_for(sycl::range<1>(query_size), [=](sycl::id<1> id) {
                const size_t qi = id[0];
                PointType q;
                transform::kernel::transform_point(query_ptr[qi], q, trans_vec);

                // Compute query voxel coordinates (offset-adjusted).
                const int64_t qx = static_cast<int64_t>(sycl::floor(q.x() * vs_inv)) + VoxelConstants::coord_offset;
                const int64_t qy = static_cast<int64_t>(sycl::floor(q.y() * vs_inv)) + VoxelConstants::coord_offset;
                const int64_t qz = static_cast<int64_t>(sycl::floor(q.z() * vs_inv)) + VoxelConstants::coord_offset;

                float best_dist = std::numeric_limits<float>::max();
                int32_t best_slot = -1;

                // Iterate all 27 combinations of dx/dy/dz in {-1, 0, 1}.
                for (int8_t dx = -1; dx <= 1; ++dx) {
                    for (int8_t dy = -1; dy <= 1; ++dy) {
                        for (int8_t dz = -1; dz <= 1; ++dz) {
                            // Filter based on the requested neighbor pattern.
                            const int8_t sum_abs = (dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy) + (dz < 0 ? -dz : dz);
                            if constexpr (NUM_NEIGHBOR_VOXELS == 7) {
                                if (sum_abs > 1) continue;  // Keep only face neighbors (Manhattan dist <= 1)
                            }
                            if constexpr (NUM_NEIGHBOR_VOXELS == 19) {
                                if (sum_abs > 2) continue;  // Keep face + edge neighbors (Manhattan dist <= 2)
                            }
                            // 27: no filter, all 3x3x3 neighbors included

                            const int64_t nx = qx + static_cast<int64_t>(dx);
                            const int64_t ny = qy + static_cast<int64_t>(dy);
                            const int64_t nz = qz + static_cast<int64_t>(dz);

                            // Skip out-of-range coordinates.
                            if (nx < 0 || nx > VoxelConstants::coord_bit_mask) continue;
                            if (ny < 0 || ny > VoxelConstants::coord_bit_mask) continue;
                            if (nz < 0 || nz > VoxelConstants::coord_bit_mask) continue;

                            const uint64_t neighbor_key = encode_voxel_coords(nx, ny, nz);

                            // Probe hash table for this neighbor voxel.
                            for (size_t probe = 0; probe < max_probe; ++probe) {
                                const size_t slot = compute_slot_id(neighbor_key, probe, capacity);
                                const uint64_t stored = key_ptr[slot];

                                if (stored == neighbor_key) {
                                    const VoxelCoreData& core = core_ptr[slot];
                                    if (core.count >= min_pts) {
                                        // Compute squared distance from query to voxel centroid.
                                        const float inv_c = 1.0f / static_cast<float>(core.count);
                                        const float cx = core.sum_x * inv_c;
                                        const float cy = core.sum_y * inv_c;
                                        const float cz = core.sum_z * inv_c;
                                        const float ex = q.x() - cx;
                                        const float ey = q.y() - cy;
                                        const float ez = q.z() - cz;
                                        const float dist_sq = ex * ex + ey * ey + ez * ez;
                                        if (dist_sq < best_dist) {
                                            best_dist = dist_sq;
                                            best_slot = static_cast<int32_t>(slot);
                                        }
                                    }
                                    break;
                                }
                                if (stored == VoxelConstants::invalid_coord) break;
                            }
                        }
                    }
                }

                out_indices[qi] = best_slot;
                out_distances[qi] = best_dist;
            });
        });
        sycl_utils::events out;
        out.push_back(event);
        return out;
    }
};

}  // namespace mapping
}  // namespace algorithms
}  // namespace sycl_points
