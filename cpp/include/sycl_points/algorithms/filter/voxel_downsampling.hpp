#pragma once

#include <algorithm>
#include <vector>

#include "sycl_points/algorithms/common/voxel_constants.hpp"
#include "sycl_points/points/point_cloud.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {

/// @brief Voxel grid downsampling with SYCL implementation
class VoxelGrid {
public:
    using Ptr = std::shared_ptr<VoxelGrid>;

    /// @brief Constructor
    /// @param queue SYCL queue
    /// @param voxel_size voxel size
    VoxelGrid(const sycl_points::sycl_utils::DeviceQueue& queue, const float voxel_size)
        : queue_(queue), voxel_size_(voxel_size) {
        if (voxel_size <= 0.0f) {
            throw std::invalid_argument("voxel_size must be positive");
        }
        this->bit_ptr_ = std::make_shared<shared_vector<uint64_t>>(0, *this->queue_.ptr);
        this->voxel_size_inv_ = 1.0f / this->voxel_size_;
        this->min_voxel_count_ = 1;
    }

    /// @brief Set voxel size
    /// @param voxel_size voxel size
    void set_voxel_size(const float voxel_size) {
        if (voxel_size <= 0.0f) {
            throw std::invalid_argument("voxel_size must be positive");
        }
        this->voxel_size_ = voxel_size;
        this->voxel_size_inv_ = 1.0f / this->voxel_size_;
    }

    /// @brief Get voxel size
    /// @param voxel_size voxel size
    float get_voxel_size() const { return this->voxel_size_; }

    void set_min_voxel_count(const size_t min_voxel_count) { this->min_voxel_count_ = min_voxel_count; }

    /// @brief Voxel downsampling
    /// @param points Point Cloud
    /// @param result Voxel downsampled
    void downsampling(const PointContainerShared& points, PointContainerShared& result) {
        const size_t N = points.size();
        if (N == 0) {
            result.resize(0);
            return;
        }
        // Compute voxel keys on device and aggregate sorted groups on host.
        const auto sorted_indices = this->compute_sorted_voxel_indices(points);
        this->sorted_voxel_indices_to_cloud(points, sorted_indices, result);
    }

    /// @brief Voxel downsampling
    /// @param points Point Cloud
    /// @param result Voxel downsampled
    void downsampling(const PointCloudShared& cloud, PointCloudShared& result) {
        const size_t N = cloud.size();
        if (N == 0) {
            result.resize_points(0);
            return;
        }
        const auto start_time_ms = cloud.start_time_ms;
        const auto end_time_ms = cloud.end_time_ms;
        // Compute voxel keys on device and aggregate sorted groups on host.
        const auto sorted_indices = this->compute_sorted_voxel_indices(*cloud.points);
        this->sorted_voxel_indices_to_cloud(cloud, sorted_indices, result);
        if (cloud.has_timestamps()) {
            result.start_time_ms = start_time_ms;
            result.end_time_ms = end_time_ms;
        }
    }

private:
    sycl_points::sycl_utils::DeviceQueue queue_;
    float voxel_size_;
    float voxel_size_inv_;
    size_t min_voxel_count_;

    shared_vector_ptr<uint64_t> bit_ptr_ = nullptr;

    void compute_voxel_bit(const PointContainerShared& points) {
        const size_t N = points.size();

        // mem_advise set to device
        {
            this->queue_.set_accessed_by_device(this->bit_ptr_->data(), N);
            this->queue_.set_accessed_by_device(points.data(), N);
        }

        const size_t work_group_size = this->queue_.get_work_group_size();
        const size_t global_size = this->queue_.get_global_size(N);
        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            // memory ptr
            const auto point_ptr = points.data();
            const auto bit_ptr = this->bit_ptr_->data();
            const auto voxel_size_inv = this->voxel_size_inv_;
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const uint32_t i = item.get_global_id(0);
                if (i >= N) return;

                bit_ptr[i] = kernel::compute_voxel_bit(point_ptr[i], voxel_size_inv);
            });
        });
        event.wait_and_throw();

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(this->bit_ptr_->data(), N);
            this->queue_.clear_accessed_by_device(points.data(), N);
        }
    }

    std::vector<size_t> compute_sorted_voxel_indices(const PointContainerShared& points) {
        const size_t N = points.size();
        if (this->bit_ptr_->size() < N) {
            this->bit_ptr_->resize(N);
        }

        // compute bit on device
        this->compute_voxel_bit(points);

        // mem_advise set to host
        {
            this->queue_.set_accessed_by_host(this->bit_ptr_->data(), N);
        }

        // Collect valid points and sort by voxel key.
        std::vector<size_t> sorted_indices;
        sorted_indices.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            if ((*this->bit_ptr_)[i] != VoxelConstants::invalid_coord) {
                sorted_indices.push_back(i);
            }
        }

        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t lhs, size_t rhs) {
            return (*this->bit_ptr_)[lhs] < (*this->bit_ptr_)[rhs];
        });

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_host(this->bit_ptr_->data(), N);
        }

        return sorted_indices;
    }

    void sorted_voxel_indices_to_cloud(const PointContainerShared& points, const std::vector<size_t>& sorted_indices,
                                       PointContainerShared& result) const {
        const size_t N = sorted_indices.size();
        result.clear();
        result.reserve(N);
        const float min_voxel_count = static_cast<float>(this->min_voxel_count_);

        // mem_advise set to host
        this->queue_.set_accessed_by_host(this->bit_ptr_->data(), this->bit_ptr_->size());

        size_t group_begin = 0;
        while (group_begin < N) {
            const auto key = (*this->bit_ptr_)[sorted_indices[group_begin]];
            PointType point_sum = PointType::Zero();

            size_t group_end = group_begin;
            while (group_end < N && (*this->bit_ptr_)[sorted_indices[group_end]] == key) {
                point_sum += points[sorted_indices[group_end]];
                ++group_end;
            }

            const auto point_count = point_sum.w();
            if (point_count >= min_voxel_count) {
                result.push_back(point_sum / point_count);
            }
            group_begin = group_end;
        }

        // mem_advise clear
        this->queue_.clear_accessed_by_host(this->bit_ptr_->data(), this->bit_ptr_->size());
    }

    void sorted_voxel_indices_to_cloud(const PointCloudShared& cloud, const std::vector<size_t>& sorted_indices,
                                       PointCloudShared& result) const {
        const size_t N = sorted_indices.size();
        const bool has_rgb = cloud.has_rgb();
        const bool has_intensity = cloud.has_intensity();
        const bool has_timestamp = cloud.has_timestamps();
        result.clear();
        result.reserve_points(N);
        if (has_rgb) {
            result.reserve_rgb(N);
        }
        if (has_intensity) {
            result.reserve_intensities(N);
        }
        if (has_timestamp) {
            result.reserve_timestamps(N);
        }

        // mem_advise set to host
        this->queue_.set_accessed_by_host(this->bit_ptr_->data(), this->bit_ptr_->size());

        const float min_voxel_count = static_cast<float>(this->min_voxel_count_);
        size_t group_begin = 0;
        while (group_begin < N) {
            const auto key = (*this->bit_ptr_)[sorted_indices[group_begin]];
            PointType point_sum = PointType::Zero();
            RGBType rgb_sum = RGBType::Zero();
            float intensity_sum = 0.0f;
            float timestamp_sum = 0.0f;

            size_t group_end = group_begin;
            while (group_end < N && (*this->bit_ptr_)[sorted_indices[group_end]] == key) {
                const size_t idx = sorted_indices[group_end];
                point_sum += (*cloud.points)[idx];
                if (has_rgb) {
                    rgb_sum += (*cloud.rgb)[idx];
                }
                if (has_intensity) {
                    intensity_sum += (*cloud.intensities)[idx];
                }
                if (has_timestamp) {
                    timestamp_sum += (*cloud.timestamp_offsets)[idx];
                }
                ++group_end;
            }

            const auto point_count = point_sum.w();
            if (point_count >= min_voxel_count) {
                result.points->emplace_back(point_sum / point_count);
                if (has_rgb) {
                    result.rgb->emplace_back(rgb_sum / point_count);
                }
                if (has_intensity) {
                    result.intensities->emplace_back(intensity_sum / point_count);
                }
                if (has_timestamp) {
                    result.timestamp_offsets->emplace_back(timestamp_sum / point_count);
                }
            }
            group_begin = group_end;
        }

        // mem_advise clear
        this->queue_.clear_accessed_by_host(this->bit_ptr_->data(), this->bit_ptr_->size());
    }
};

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
