#pragma once

#include <numeric>
#include <random>

#include "sycl_points/algorithms/common/filter_by_flags.hpp"
#include "sycl_points/algorithms/common/prefix_sum.hpp"
#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {

namespace kernel {

SYCL_EXTERNAL inline bool is_finite(const PointType& pt) {
    return std::isfinite(pt[0]) && std::isfinite(pt[1]) && std::isfinite(pt[2]) && std::isfinite(pt[3]);
}

SYCL_EXTERNAL inline void box_filter(const PointType& pt, uint8_t& flag, float min_distance, float max_distance) {
    const float linf_dist = sycl::max(sycl::fabs(pt.x()), sycl::max(sycl::fabs(pt.y()), sycl::fabs(pt.z())));

    if (linf_dist < min_distance || linf_dist > max_distance) {
        flag = REMOVE_FLAG;
    }
}

}  // namespace kernel

/// @brief Preprocessing filter for point cloud data
class PreprocessFilter {
public:
    using Ptr = std::shared_ptr<PreprocessFilter>;

    /// @brief Constructor
    /// @param queue SYCL queue
    PreprocessFilter(const sycl_utils::DeviceQueue& queue) : queue_(queue) {
        this->filter_ = std::make_shared<FilterByFlags>(this->queue_);
        this->flags_ = std::make_shared<shared_vector<uint8_t>>(*this->queue_.ptr);
        this->dist_sq_ = std::make_shared<shared_vector<float>>(*this->queue_.ptr);

        this->mt_.seed(1234);  // Default seed for reproducibility
    }

    /// @brief Sets the seed for the random number generator
    /// @param seed Seed value to initialize the Mersenne Twister random generator
    void set_random_seed(uint_fast32_t seed) { this->mt_.seed(seed); }

    /// @brief L∞ distance (chebyshev distance) to the point cloud
    /// @param data Point cloud to be filtered (modified in-place)
    /// @param min_distance Minimum distance threshold (points closer than this are removed)
    /// @param max_distance Maximum distance threshold (points farther than this are removed)
    void box_filter(PointCloudShared& data, float min_distance = 1.0f,
                    float max_distance = std::numeric_limits<float>::max()) {
        this->box_filter_impl(data, data, min_distance, max_distance);
    }

    /// @brief L∞ distance (chebyshev distance) to the point cloud
    /// @param source Source point cloud to be filtered
    /// @param output Output point cloud
    /// @param min_distance Minimum distance threshold (points closer than this are removed)
    /// @param max_distance Maximum distance threshold (points farther than this are removed)
    void box_filter(const PointCloudShared& source, PointCloudShared& output, float min_distance = 1.0f,
                    float max_distance = std::numeric_limits<float>::max()) {
        this->box_filter_impl(source, output, min_distance, max_distance);
    }

    /// @brief Randomly samples a specified number of points from the point cloud
    /// @param data Point cloud to be sampled (modified in-place)
    /// @param sampling_num Number of points to retain after sampling
    void random_sampling(PointCloudShared& data, size_t sampling_num) {
        this->random_sampling_impl(data, data, sampling_num);
    }

    /// @brief Randomly samples a specified number of points from the point cloud
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    void random_sampling(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        this->random_sampling_impl(source, output, sampling_num);
    }

    /// @brief Farthest Point Sampling (FPS)
    /// @param source Source point cloud to be sampled (modified in-place)
    /// @param sampling_num Number of points to retain after sampling
    void farthest_point_sampling(PointCloudShared& data, size_t sampling_num) {
        this->farthest_point_sampling_impl(data, data, sampling_num);
    }

    /// @brief Farthest Point Sampling (FPS)
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    void farthest_point_sampling(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        this->farthest_point_sampling_impl(source, output, sampling_num);
    }

    /// @brief Normal Histogram Bucket Sampling
    ///
    /// Divides the unit hemisphere into longitude×latitude bins and samples equally from each bin,
    /// ensuring the selected points cover all normal directions for GICP constraint coverage.
    /// Requires pre-computed normals or covariance matrices.
    ///
    /// @param data Point cloud to be sampled (modified in-place)
    /// @param sampling_num Number of points to retain after sampling
    /// @param longitude_bins Number of bins along the longitude axis (default: 8)
    /// @param latitude_bins Number of bins along the latitude axis, covering [0, π/2] (default: 4)
    void normal_histogram_sampling(PointCloudShared& data, size_t sampling_num, size_t longitude_bins = 8,
                                   size_t latitude_bins = 4) {
        this->normal_histogram_sampling_impl(data, data, sampling_num, longitude_bins, latitude_bins);
    }

    /// @brief Normal Histogram Bucket Sampling
    ///
    /// Divides the unit hemisphere into longitude×latitude bins and samples equally from each bin,
    /// ensuring the selected points cover all normal directions for GICP constraint coverage.
    /// Requires pre-computed normals or covariance matrices.
    ///
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    /// @param longitude_bins Number of bins along the longitude axis (default: 8)
    /// @param latitude_bins Number of bins along the latitude axis, covering [0, π/2] (default: 4)
    void normal_histogram_sampling(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num,
                                   size_t longitude_bins = 8, size_t latitude_bins = 4) {
        this->normal_histogram_sampling_impl(source, output, sampling_num, longitude_bins, latitude_bins);
    }

    /// @brief Spherical Fibonacci Grid Sampling
    ///
    /// Generates sampling_num uniformly distributed target directions on S² using the golden-ratio
    /// Fibonacci lattice, then selects the point whose normal direction is closest (by absolute dot
    /// product) to each target. This provides mathematically near-optimal sphere coverage and
    /// guarantees constraints in all directions for GICP.
    /// Requires pre-computed normals or covariance matrices.
    ///
    /// Time complexity: O(N × sampling_num) on CPU.
    ///
    /// @param data Point cloud to be sampled (modified in-place)
    /// @param sampling_num Number of points to retain after sampling
    void spherical_fibonacci_sampling(PointCloudShared& data, size_t sampling_num) {
        this->spherical_fibonacci_sampling_impl(data, data, sampling_num);
    }

    /// @brief Spherical Fibonacci Grid Sampling
    ///
    /// Generates sampling_num uniformly distributed target directions on S² using the golden-ratio
    /// Fibonacci lattice, then selects the point whose normal direction is closest (by absolute dot
    /// product) to each target. This provides mathematically near-optimal sphere coverage and
    /// guarantees constraints in all directions for GICP.
    /// Requires pre-computed normals or covariance matrices.
    ///
    /// Time complexity: O(N × sampling_num) on CPU.
    ///
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    void spherical_fibonacci_sampling(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        this->spherical_fibonacci_sampling_impl(source, output, sampling_num);
    }

    /// @brief Filters points by incidence angle between point position and surface normal.
    /// @param data Point cloud to be filtered (modified in-place).
    /// @param min_angle Minimum allowable incidence angle in radians.
    /// @param max_angle Maximum allowable incidence angle in radians.
    /// @note Requires per-point normals or covariance matrices to compute normals; uses absolute incidence angle
    /// (normal direction ignored).
    void angle_incidence_filter(PointCloudShared& data, float min_angle, float max_angle) {
        this->angle_incidence_filter_impl(data, data, min_angle, max_angle);
    }

    /// @brief Filters points by incidence angle between point position and surface normal.
    /// @param source Source point cloud to be filtered.
    /// @param output Output point cloud.
    /// @param min_angle Minimum allowable incidence angle in radians.
    /// @param max_angle Maximum allowable incidence angle in radians.
    /// @note Requires per-point normals or covariance matrices to compute normals; uses absolute incidence angle
    /// (normal direction ignored).
    void angle_incidence_filter(const PointCloudShared& source, PointCloudShared& output, float min_angle,
                                float max_angle) {
        this->angle_incidence_filter_impl(source, output, min_angle, max_angle);
    }

private:
    sycl_utils::DeviceQueue queue_;
    FilterByFlags::Ptr filter_;
    shared_vector_ptr<uint8_t> flags_;
    shared_vector_ptr<float> dist_sq_ = nullptr;  // for FPS

    std::mt19937 mt_;

    /// @brief Initializes the flags vector with a specified value
    /// @param data_size Size needed for the flags vector
    /// @param initial_flag Initial value to fill the flags with (INCLUDE_FLAG or REMOVE_FLAG)
    sycl_utils::events initialize_flags(size_t data_size, uint8_t initial_flag = INCLUDE_FLAG) {
        if (this->flags_->size() < data_size) {
            this->flags_->resize(data_size);
        }
        sycl_utils::events events;
        events += this->queue_.ptr->fill(this->flags_->data(), initial_flag, data_size);
        return events;
    }

    /// @brief Applies filtering based on the current flags
    /// @param data Point cloud to be filtered (modified in-place)
    void filter_by_flags(const PointCloudShared& source, PointCloudShared& output) {
        if (source.has_cov()) {
            this->filter_->filter_by_flags(*source.covs, *output.covs, *this->flags_);
        }
        if (source.has_normal()) {
            this->filter_->filter_by_flags(*source.normals, *output.normals, *this->flags_);
        }
        if (source.has_rgb()) {
            this->filter_->filter_by_flags(*source.rgb, *output.rgb, *this->flags_);
        }
        if (source.has_intensity()) {
            this->filter_->filter_by_flags(*source.intensities, *output.intensities, *this->flags_);
        }
        if (source.has_timestamps()) {
            this->filter_->filter_by_flags(*source.timestamp_offsets, *output.timestamp_offsets, *this->flags_);
        }
        this->filter_->filter_by_flags(*source.points, *output.points, *this->flags_);
    }

    /// @brief L∞ distance (chebyshev distance) to the point cloud
    /// @param source Source point cloud to be filtered
    /// @param output Output point cloud
    /// @param min_distance Minimum distance threshold (points closer than this are removed)
    /// @param max_distance Maximum distance threshold (points farther than this are removed)
    void box_filter_impl(const PointCloudShared& source, PointCloudShared& output, float min_distance = 1.0f,
                         float max_distance = std::numeric_limits<float>::max()) {
        const size_t N = source.size();
        if (N == 0) return;

        this->initialize_flags(N).wait_and_throw();

        // mem_advise set to device
        {
            this->queue_.set_accessed_by_device(this->flags_->data(), N);
            this->queue_.set_accessed_by_device(source.points_ptr(), N);
        }

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t work_group_size = this->queue_.get_work_group_size();
            const size_t global_size = this->queue_.get_global_size(N);
            // memory ptr
            const auto point_ptr = source.points_ptr();
            const auto flag_ptr = this->flags_->data();
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                if (!kernel::is_finite(point_ptr[i])) {
                    flag_ptr[i] = REMOVE_FLAG;
                    return;
                }
                kernel::box_filter(point_ptr[i], flag_ptr[i], min_distance, max_distance);
            });
        });
        event.wait_and_throw();

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(this->flags_->data(), N);
            this->queue_.clear_accessed_by_device(source.points_ptr(), N);
        }

        this->filter_by_flags(source, output);
    }

    /// @brief Randomly samples a specified number of points from the point cloud
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    void random_sampling_impl(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        const size_t N = source.size();
        if (N == 0) return;
        if (N <= sampling_num) return;

        this->initialize_flags(N, REMOVE_FLAG).wait_and_throw();

        // mem_advise to host
        this->queue_.set_accessed_by_host(this->flags_->data(), N);

        // Generate indices and perform Fisher-Yates shuffle for the first sampling_num elements
        std::vector<size_t> indices(N);
        {
            std::iota(indices.begin(), indices.end(), 0);
            for (size_t i = 0; i < sampling_num; ++i) {
                std::uniform_int_distribution<size_t> dist(i, N - 1);
                const size_t j = dist(this->mt_);
                std::swap(indices[i], indices[j]);
            }
        }

        // Mark the selected indices for inclusion
        for (size_t i = 0; i < sampling_num; ++i) {
            (*this->flags_)[indices[i]] = INCLUDE_FLAG;
        }

        // mem_advise clear
        this->queue_.clear_accessed_by_host(this->flags_->data(), N);

        this->filter_by_flags(source, output);
    }

    /// @brief Normal Histogram Bucket Sampling implementation
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    /// @param longitude_bins Number of longitude bins
    /// @param latitude_bins Number of latitude bins
    void normal_histogram_sampling_impl(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num,
                                        size_t longitude_bins, size_t latitude_bins) {
        const size_t N = source.size();
        if (N == 0) {
            output = PointCloudShared(output.queue);
            return;
        }
        if (N <= sampling_num) {
            // Keep output behavior consistent between in-place and out-of-place overloads.
            output = PointCloudShared(output.queue, source);
            return;
        }

        if (longitude_bins == 0 || latitude_bins == 0) {
            throw std::invalid_argument(
                "[PreprocessFilter::normal_histogram_sampling] longitude_bins and latitude_bins must be greater "
                "than zero.");
        }

        if (longitude_bins > std::numeric_limits<size_t>::max() / latitude_bins) {
            throw std::overflow_error(
                "[PreprocessFilter::normal_histogram_sampling] Bin count overflow: longitude_bins * latitude_bins "
                "is too large.");
        }

        if (!source.has_normal() && !source.has_cov()) {
            throw std::runtime_error(
                "[PreprocessFilter::normal_histogram_sampling] Normal vectors or covariance matrices must be "
                "pre-computed.");
        }

        this->initialize_flags(N, REMOVE_FLAG).wait_and_throw();

        // Access data on host
        this->queue_.set_accessed_by_host(this->flags_->data(), N);
        if (source.has_normal()) {
            this->queue_.set_accessed_by_host(source.normals_ptr(), N);
        } else {
            // compute_normal_from_covariance requires both point position and covariance
            this->queue_.set_accessed_by_host(source.points_ptr(), N);
            this->queue_.set_accessed_by_host(source.covs_ptr(), N);
        }

        const size_t num_bins = longitude_bins * latitude_bins;
        std::vector<std::vector<size_t>> bins(num_bins);

        // Assign each point to a bin based on its normal direction
        for (size_t i = 0; i < N; ++i) {
            Normal normal;
            if (source.has_normal()) {
                normal = source.normals_ptr()[i];
            } else {
                algorithms::covariance::kernel::compute_normal_from_covariance(source.points_ptr()[i],
                                                                               source.covs_ptr()[i], normal);
            }

            float nx = normal.x(), ny = normal.y(), nz = normal.z();
            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len < 1e-6f) continue;
            nx /= len;
            ny /= len;
            nz /= len;

            // Half-sphere symmetry: fold to upper hemisphere (nz >= 0)
            if (nz < 0.0f) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }

            // Longitude bin: atan2(ny, nx) in [-π, π] → shifted to [0, 2π)
            float lon = std::atan2(ny, nx);
            if (lon < 0.0f) lon += 2.0f * eigen_utils::PI;
            const size_t lon_bin =
                std::min(static_cast<size_t>(lon / (2.0f * eigen_utils::PI) * longitude_bins), longitude_bins - 1);

            // Latitude bin: acos(nz) in [0, π/2] → [0, latitude_bins)
            const float lat = std::acos(std::min(nz, 1.0f));
            const size_t lat_bin =
                std::min(static_cast<size_t>(lat / (eigen_utils::PI * 0.5f) * latitude_bins), latitude_bins - 1);

            bins[lon_bin + lat_bin * longitude_bins].push_back(i);
        }

        // Shuffle each bin for randomness within the bin
        for (auto& bin : bins) {
            std::shuffle(bin.begin(), bin.end(), this->mt_);
        }

        // Round-robin sampling across bins until sampling_num points are selected
        size_t selected = 0;
        std::vector<size_t> bin_offsets(num_bins, 0);
        bool any_progress = true;
        while (selected < sampling_num && any_progress) {
            any_progress = false;
            for (size_t b = 0; b < num_bins && selected < sampling_num; ++b) {
                if (bin_offsets[b] < bins[b].size()) {
                    (*this->flags_)[bins[b][bin_offsets[b]]] = INCLUDE_FLAG;
                    ++bin_offsets[b];
                    ++selected;
                    any_progress = true;
                }
            }
        }

        // Clear memory advise
        this->queue_.clear_accessed_by_host(this->flags_->data(), N);
        if (source.has_normal()) {
            this->queue_.clear_accessed_by_host(source.normals_ptr(), N);
        } else {
            this->queue_.clear_accessed_by_host(source.points_ptr(), N);
            this->queue_.clear_accessed_by_host(source.covs_ptr(), N);
        }

        this->filter_by_flags(source, output);
    }

    /// @brief Spherical Fibonacci Grid Sampling implementation
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    void spherical_fibonacci_sampling_impl(const PointCloudShared& source, PointCloudShared& output,
                                           size_t sampling_num) {
        const size_t N = source.size();
        if (N == 0) {
            output = PointCloudShared(output.queue);
            return;
        }
        if (N <= sampling_num) {
            // Keep output behavior consistent between in-place and out-of-place overloads.
            output = PointCloudShared(output.queue, source);
            return;
        }

        if (!source.has_normal() && !source.has_cov()) {
            throw std::runtime_error(
                "[PreprocessFilter::spherical_fibonacci_sampling] Normal vectors or covariance matrices must be "
                "pre-computed.");
        }

        // Generate uniformly distributed target directions on S² via the Fibonacci lattice.
        // z_i = 1 - (2i+1)/N places points from north to south pole.
        // theta_i = 2π*i/φ (φ = golden ratio) distributes longitude uniformly without repetition.
        const float golden_ratio = (1.0f + std::sqrt(5.0f)) * 0.5f;
        std::vector<Eigen::Vector3f> targets(sampling_num);
        for (size_t i = 0; i < sampling_num; ++i) {
            const float z = 1.0f - (2.0f * static_cast<float>(i) + 1.0f) / static_cast<float>(sampling_num);
            const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
            const float theta = 2.0f * eigen_utils::PI * static_cast<float>(i) / golden_ratio;
            targets[i] = Eigen::Vector3f(r * std::cos(theta), r * std::sin(theta), z);
        }

        // Extract and normalize all normals on the host
        if (source.has_normal()) {
            this->queue_.set_accessed_by_host(source.normals_ptr(), N);
        } else {
            this->queue_.set_accessed_by_host(source.points_ptr(), N);
            this->queue_.set_accessed_by_host(source.covs_ptr(), N);
        }

        std::vector<Eigen::Vector3f> normals(N);
        for (size_t i = 0; i < N; ++i) {
            Normal normal;
            if (source.has_normal()) {
                normal = source.normals_ptr()[i];
            } else {
                algorithms::covariance::kernel::compute_normal_from_covariance(source.points_ptr()[i],
                                                                               source.covs_ptr()[i], normal);
            }
            const float nx = normal.x(), ny = normal.y(), nz = normal.z();
            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            normals[i] = (len > 1e-6f) ? Eigen::Vector3f(nx / len, ny / len, nz / len) : Eigen::Vector3f::Zero();
        }

        if (source.has_normal()) {
            this->queue_.clear_accessed_by_host(source.normals_ptr(), N);
        } else {
            this->queue_.clear_accessed_by_host(source.points_ptr(), N);
            this->queue_.clear_accessed_by_host(source.covs_ptr(), N);
        }

        // For each Fibonacci target direction, select the closest unselected normal (half-sphere symmetric)
        this->initialize_flags(N, REMOVE_FLAG).wait_and_throw();
        this->queue_.set_accessed_by_host(this->flags_->data(), N);

        std::vector<bool> used(N, false);
        for (const auto& target : targets) {
            float best_abs_dot = -1.0f;
            size_t best_idx = N;  // invalid sentinel

            for (size_t i = 0; i < N; ++i) {
                if (used[i]) continue;
                // |dot| accounts for half-sphere symmetry: n and -n are equivalent
                const float abs_dot = std::abs(normals[i].dot(target));
                if (abs_dot > best_abs_dot) {
                    best_abs_dot = abs_dot;
                    best_idx = i;
                }
            }

            if (best_idx < N) {
                (*this->flags_)[best_idx] = INCLUDE_FLAG;
                used[best_idx] = true;
            }
        }

        this->queue_.clear_accessed_by_host(this->flags_->data(), N);

        this->filter_by_flags(source, output);
    }

    /// @brief Farthest Point Sampling (FPS)
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    void farthest_point_sampling_impl(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        const size_t N = source.size();
        if (N == 0) return;
        if (N <= sampling_num) return;

        if (this->dist_sq_->size() < N) {
            this->dist_sq_->resize(N);
        }

        // mem_advise set to device
        {
            this->queue_.set_accessed_by_device(source.points_ptr(), N);
            this->queue_.set_accessed_by_device(this->dist_sq_->data(), N);
        }

        // initialize
        sycl_utils::events init_events;
        {
            init_events += this->initialize_flags(N, REMOVE_FLAG);
            init_events += this->queue_.ptr->fill(this->dist_sq_->data(), std::numeric_limits<float>::max(), N);
        }

        init_events.wait_and_throw();

        // ramdom select initial point
        std::uniform_int_distribution<size_t> dist(0, N - 1);
        size_t selected_idx = dist(this->mt_);
        this->flags_->at(selected_idx) = INCLUDE_FLAG;

        for (size_t iter = 1; iter < sampling_num; ++iter) {
            // compute distance
            this->queue_.ptr
                ->submit([&](sycl::handler& h) {
                    const size_t work_group_size = this->queue_.get_work_group_size();
                    const size_t global_size = this->queue_.get_global_size(N);
                    // memory ptr
                    const auto point_ptr = source.points_ptr();
                    const auto flag_ptr = this->flags_->data();
                    const auto dist_sq_ptr = this->dist_sq_->data();

                    const auto selected_idx_capture = selected_idx;

                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t gid = item.get_global_id(0);
                        if (gid >= N) return;

                        const float dist_sq = eigen_utils::frobenius_norm_squared<4>(
                            eigen_utils::subtract<4, 1>(point_ptr[gid], point_ptr[selected_idx_capture]));

                        dist_sq_ptr[gid] = sycl::min(dist_sq_ptr[gid], dist_sq);
                    });
                })
                .wait_and_throw();

            // find farthest point
            {
#if __cplusplus >= 202002L
                const auto max_elem =
                    std::max_element(std::execution::unseq, this->dist_sq_->data(), this->dist_sq_->data() + N);
#else
                const auto max_elem = std::max_element(this->dist_sq_->data(), this->dist_sq_->data() + N);
#endif
                const size_t max_elem_idx = std::distance(this->dist_sq_->data(), max_elem);

                // update index
                selected_idx = max_elem_idx;
                this->flags_->at(max_elem_idx) = INCLUDE_FLAG;
            }
        }

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(source.points_ptr(), N);
            this->queue_.clear_accessed_by_device(this->dist_sq_->data(), N);
        }

        this->filter_by_flags(source, output);
    }

    /// @brief Removes points whose (absolute) incidence angle with the surface normal is outside [min_angle,
    /// max_angle].
    /// @param source Source point cloud to be sampled.
    /// @param output Output point cloud.
    /// @param min_angle Minimum allowable incidence angle in radians.
    /// @param max_angle Maximum allowable incidence angle in radians.
    void angle_incidence_filter_impl(const PointCloudShared& source, PointCloudShared& output, float min_angle,
                                     float max_angle) {
        const size_t N = source.size();
        if (N == 0) return;

        if (!source.has_normal() && !source.has_cov()) {
            throw std::runtime_error(
                "[PreprocessFilter::angle_incidence_filter] Normal vector or covariance matrices must be "
                "pre-computed.");
        }
        if (min_angle < 0.0f || max_angle > M_PIf * 0.5f || min_angle >= max_angle) {
            throw std::invalid_argument("[PreprocessFilter::angle_incidence_filter] Invalid angle range");
        }

        this->initialize_flags(N).wait_and_throw();

        // mem_advise set to device
        {
            this->queue_.set_accessed_by_device(this->flags_->data(), N);
            this->queue_.set_accessed_by_device(source.points_ptr(), N);
            if (source.has_normal()) {
                this->queue_.set_accessed_by_device(source.normals_ptr(), N);
            } else {
                this->queue_.set_accessed_by_device(source.covs_ptr(), N);
            }
        }

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t work_group_size = this->queue_.get_work_group_size();
            const size_t global_size = this->queue_.get_global_size(N);
            // memory ptr
            const auto point_ptr = source.points_ptr();
            const auto cov_ptr = source.covs_ptr();
            const auto normal_ptr = source.normals_ptr();
            const auto flag_ptr = this->flags_->data();
            const auto max_cos = std::cos(min_angle);
            const auto min_cos = std::cos(max_angle);

            auto compute_flag = [=](const PointType& pt, const Normal& normal, uint8_t& flag) {
                const float dot = eigen_utils::dot<3>(pt.head<3>(), normal.head<3>());
                const float denom =
                    eigen_utils::frobenius_norm<3>(pt.head<3>()) * eigen_utils::frobenius_norm<3>(normal.head<3>());

                if (denom <= 1e-6f) {
                    flag = REMOVE_FLAG;
                    return;
                }
                const float abs_cos = sycl::fabs(dot / denom);
                if (abs_cos < min_cos || abs_cos > max_cos) {
                    flag = REMOVE_FLAG;
                }
            };

            if (source.has_normal()) {
                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_id(0);
                    if (i >= N) return;
                    if (!kernel::is_finite(point_ptr[i])) {
                        flag_ptr[i] = REMOVE_FLAG;
                        return;
                    }
                    const auto pt = point_ptr[i];
                    const auto normal = normal_ptr[i];
                    compute_flag(pt, normal, flag_ptr[i]);
                });
            } else {
                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_id(0);
                    if (i >= N) return;
                    if (!kernel::is_finite(point_ptr[i])) {
                        flag_ptr[i] = REMOVE_FLAG;
                        return;
                    }
                    const auto pt = point_ptr[i];
                    const auto cov = cov_ptr[i];
                    Normal normal;
                    algorithms::covariance::kernel::compute_normal_from_covariance(pt, cov, normal);

                    compute_flag(pt, normal, flag_ptr[i]);
                });
            }
        });
        event.wait_and_throw();

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(this->flags_->data(), N);
            this->queue_.clear_accessed_by_device(source.points_ptr(), N);
            if (source.has_normal()) {
                this->queue_.clear_accessed_by_device(source.normals_ptr(), N);
            } else {
                this->queue_.clear_accessed_by_device(source.covs_ptr(), N);
            }
        }

        this->filter_by_flags(source, output);
    }
};

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
