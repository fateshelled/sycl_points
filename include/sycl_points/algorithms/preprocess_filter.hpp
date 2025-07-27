#pragma once

#include <execution>
#include <numeric>
#include <random>
#include <sycl_points/algorithms/common/prefix_sum.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace filter {

constexpr uint8_t REMOVE_FLAG = 0;
constexpr uint8_t INCLUDE_FLAG = 1;

namespace kernel {

SYCL_EXTERNAL inline bool is_finite(const PointType& pt) {
    return std::isfinite(pt[0]) && std::isfinite(pt[1]) && std::isfinite(pt[2]) && std::isfinite(pt[3]);
}

SYCL_EXTERNAL inline void box_filter(const PointType& pt, uint8_t& flag, float min_distance, float max_distance) {
#pragma unroll 3
    for (size_t j = 0; j < 3; ++j) {
        const auto val = std::fabs(pt[j]);
        if (val < min_distance || val > max_distance) {
            flag = REMOVE_FLAG;
            return;
        }
    }
}

}  // namespace kernel

/// @brief Filter class for processing data based on flags
class FilterByFlags {
public:
    /// @brief Constructor
    /// @param queue SYCL queue
    FilterByFlags(const sycl_utils::DeviceQueue& queue) : queue_(queue) {
        this->points_copy_ptr_ = std::make_shared<PointContainerShared>(*this->queue_.ptr);
        this->covs_copy_ptr_ = std::make_shared<CovarianceContainerShared>(*this->queue_.ptr);
        this->prefix_sum_ = std::make_shared<common::PrefixSum>(this->queue_);
    }

    /// @brief Filter data synchronously on host
    /// @tparam T Data type (PointType or Covariance)
    /// @tparam AllocSize Optional allocator size
    /// @param data Data to be filtered
    /// @param flags Flags indicating which elements to keep (INCLUDE_FLAG) or remove
    template <typename T, size_t AllocSize = 0>
    void filter_by_flags(shared_vector<T, AllocSize>& data, const shared_vector<uint8_t>& flags) const {
        const size_t N = data.size();
        if (N == 0) return;

        // mem_advise to host
        {
            this->queue_.set_accessed_by_host(data.data(), N);
            this->queue_.set_accessed_by_host(flags.data(), N);
        }

        // Filter data on host (compact elements with INCLUDE_FLAG)
        size_t new_size = 0;
        for (size_t i = 0; i < N; ++i) {
            if (flags[i] == INCLUDE_FLAG) {
                data[new_size] = data[i];
                ++new_size;
            }
        }
        // mem_advise clear
        {
            this->queue_.clear_accessed_by_host(data.data(), N);
            this->queue_.clear_accessed_by_host(flags.data(), N);
        }
        data.resize(new_size);
    }

    /// @brief Filter data asynchronously on device
    /// @tparam T Data type (must be PointType or Covariance)
    /// @tparam AllocSize Optional allocator size
    /// @param data Data to be filtered
    /// @param flags Flags indicating which elements to keep (INCLUDE_FLAG) or remove
    /// @return Events representing the asynchronous operations
    template <typename T, size_t AllocSize = 0>
    void filter_by_flags_on_device(shared_vector<T, AllocSize>& data, const shared_vector<uint8_t>& flags) {
        static_assert(std::is_same<T, PointType>::value || std::is_same<T, Covariance>::value,
                      "T is not supported type.");

        const size_t N = data.size();
        if (N == 0) return;

        // Copy original data to preserve it during filtering
        auto copy_event = this->copy_data_async(data);

        // Calculate inclusive prefix sum of flags
        const size_t new_size = this->prefix_sum_->compute(flags);

        // Get new size from last element of prefix sum (total count of INCLUDE_FLAG elements)
        const size_t original_size = data.size();

        // Wait for copy to complete before resizing
        copy_event.wait();

        // Apply filter using prefix sum for destination indices
        this->apply_filter<T, AllocSize>(data, flags, this->prefix_sum_->get_prefix_sum(), original_size, new_size);
        data.resize(new_size);
    }

private:
    sycl_utils::DeviceQueue queue_;                             // SYCL queue
    std::shared_ptr<PointContainerShared> points_copy_ptr_;     // Copy of point data
    std::shared_ptr<CovarianceContainerShared> covs_copy_ptr_;  // Copy of covariance data
    common::PrefixSum::Ptr prefix_sum_;

    /// @brief Copy input data to temporary storage
    /// @tparam T Data type
    /// @tparam AllocSize Optional allocator size
    /// @param data Data to copy
    /// @return Events representing the copy operation
    template <typename T, size_t AllocSize = 0>
    sycl_utils::events copy_data_async(shared_vector<T, AllocSize>& data) {
        const size_t N = data.size();
        sycl_utils::events events;

        if constexpr (std::is_same<T, PointType>::value) {
            if (this->points_copy_ptr_->size() < N) {
                this->points_copy_ptr_->resize(N);
            }
            events += this->queue_.ptr->memcpy(this->points_copy_ptr_->data(), data.data(), N * sizeof(T));
        } else if constexpr (std::is_same<T, Covariance>::value) {
            if (this->covs_copy_ptr_->size() < N) {
                this->covs_copy_ptr_->resize(N);
            }
            events += this->queue_.ptr->memcpy(this->covs_copy_ptr_->data(), data.data(), N * sizeof(T));
        } else {
            throw std::runtime_error("Not supported type T");
        }
        return events;
    }

    /// @brief Apply filter using prefix sum results
    /// @tparam T Data type
    /// @tparam AllocSize Optional allocator size
    /// @param data Output data buffer (already resized)
    /// @param flags Flags indicating which elements to keep
    template <typename T, size_t AllocSize = 0>
    void apply_filter(shared_vector<T, AllocSize>& data, const shared_vector<uint8_t>& flags,
                      const shared_vector<uint32_t>& prefix_sum, size_t original_size, size_t new_size) {
        // mem_advise to device
        {
            this->queue_.set_accessed_by_device(flags.data(), original_size);
            this->queue_.set_accessed_by_device(prefix_sum.data(), original_size);
        }

        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(original_size);

                // memory ptr
                T* data_ptr = data.data();
                T* copy_ptr;
                if constexpr (std::is_same<T, PointType>::value) {
                    copy_ptr = this->points_copy_ptr_->data();
                } else if constexpr (std::is_same<T, Covariance>::value) {
                    copy_ptr = this->covs_copy_ptr_->data();
                } else {
                    throw std::runtime_error("Invalid Type [T]");
                }

                const auto flag_ptr = flags.data();
                const auto prefix_sum_ptr = prefix_sum.data();

                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_id(0);
                    if (i >= original_size) return;

                    // Only process elements with INCLUDE_FLAG
                    if (flag_ptr[i] == INCLUDE_FLAG) {
                        const size_t dest_index = prefix_sum_ptr[i] - 1;
                        if constexpr (std::is_same<T, PointType>::value) {
                            const PointType pt = copy_ptr[i];
                            eigen_utils::copy<4, 1>(pt, data_ptr[dest_index]);
                        } else if constexpr (std::is_same<T, Covariance>::value) {
                            const Covariance cov = copy_ptr[i];
                            eigen_utils::copy<4, 4>(cov, data_ptr[dest_index]);
                        }
                    }
                });
            })
            .wait();

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(flags.data(), original_size);
            this->queue_.clear_accessed_by_device(prefix_sum.data(), original_size);
        }
    }
};

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
        this->selected_idx_ = std::make_shared<shared_vector<uint32_t>>(*this->queue_.ptr);

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
        const size_t N = data.size();
        if (N == 0) return;

        this->initialize_flags(N).wait();

        // mem_advise set to device
        {
            this->queue_.set_accessed_by_device(this->flags_->data(), N);
            this->queue_.set_accessed_by_device(data.points_ptr(), N);
        }

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t work_group_size = this->queue_.get_work_group_size();
            const size_t global_size = this->queue_.get_global_size(N);
            // memory ptr
            const auto point_ptr = data.points_ptr();
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
        event.wait();

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(this->flags_->data(), N);
            this->queue_.clear_accessed_by_device(data.points_ptr(), N);
        }

        this->filter_by_flags(data);
    }

    /// @brief Randomly samples a specified number of points from the point cloud
    /// @param data Point cloud to be sampled (modified in-place)
    /// @param sampling_num Number of points to retain after sampling
    void random_sampling(PointCloudShared& data, size_t sampling_num) {
        const size_t N = data.size();
        if (N == 0) return;
        if (N <= sampling_num) return;

        this->initialize_flags(N, REMOVE_FLAG).wait();

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

        this->filter_by_flags(data);
    }

    /// @brief
    /// @param data
    /// @param sampling_num
    void farthest_point_sampling(PointCloudShared& data, size_t sampling_num) {
        const size_t N = data.size();
        if (N == 0) return;
        if (N <= sampling_num) return;

        // mem_advise set to device
        {
            this->queue_.set_accessed_by_device(data.points_ptr(), N);
        }

        // initialize
        sycl_utils::events init_events;
        {
            init_events += this->initialize_flags(N, REMOVE_FLAG);

            if (this->selected_idx_->size() < sampling_num) {
                this->selected_idx_->resize(sampling_num);
            }
            if (this->dist_sq_->size() < N) {
                this->dist_sq_->resize(N);
            }
            init_events +=
                this->queue_.ptr->fill(this->selected_idx_->data(), std::numeric_limits<uint32_t>::max(), sampling_num);
            init_events += this->queue_.ptr->fill(this->dist_sq_->data(), std::numeric_limits<float>::max(), N);
        }

        init_events.wait();

        // ramdom select initial point
        std::uniform_int_distribution<size_t> dist(0, N - 1);
        const size_t initial_idx = dist(this->mt_);
        this->selected_idx_->at(0) = initial_idx;
        this->flags_->at(initial_idx) = INCLUDE_FLAG;

        for (size_t iter = 1; iter < sampling_num; ++iter) {
            // compute distance
            this->queue_.ptr
                ->submit([&](sycl::handler& h) {
                    const size_t work_group_size = this->queue_.get_work_group_size();
                    const size_t global_size = this->queue_.get_global_size(N);
                    // memory ptr
                    const auto point_ptr = data.points_ptr();
                    const auto flag_ptr = this->flags_->data();
                    const auto dist_sq_ptr = this->dist_sq_->data();
                    const auto selected_idx_ptr = this->selected_idx_->data();

                    const auto i = iter;

                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t gid = item.get_global_id(0);
                        if (gid >= N) return;

                        const size_t selected_idx = selected_idx_ptr[i - 1];
                        const float dist_sq = (selected_idx == gid)
                                                  ? 0.0f
                                                  : eigen_utils::frobenius_norm_squared<4>(eigen_utils::subtract<4, 1>(
                                                        point_ptr[gid], point_ptr[selected_idx]));

                        dist_sq_ptr[gid] = std::min(dist_sq_ptr[gid], dist_sq);
                    });
                })
                .wait();

            // find farthest point
            this->queue_.ptr
                ->submit([&](sycl::handler& h) {
                    // memory ptr
                    const auto point_ptr = data.points_ptr();
                    const auto flag_ptr = this->flags_->data();
                    const auto dist_sq_ptr = this->dist_sq_->data();
                    const auto selected_idx_ptr = this->selected_idx_->data();

                    const auto i = iter;

                    h.host_task([=]() {
#if __cplusplus >= 202002L
                        const auto max_elem = std::max_element(std::execution::unseq, dist_sq_ptr, dist_sq_ptr + N);
#else
                        const auto max_elem = std::max_element(dist_sq_ptr, dist_sq_ptr + N);
#endif
                        const size_t max_idx = std::distance(dist_sq_ptr, max_elem);
                        selected_idx_ptr[i] = max_idx;
                        flag_ptr[max_idx] = INCLUDE_FLAG;
                    });
                })
                .wait();
        }
        this->filter_by_flags(data);

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(data.points_ptr(), N);
        }
    }

private:
    sycl_utils::DeviceQueue queue_;
    std::shared_ptr<FilterByFlags> filter_;
    shared_vector_ptr<uint8_t> flags_;
    shared_vector_ptr<uint32_t> selected_idx_;  // for FPS
    shared_vector_ptr<float> dist_sq_;          // for FPS
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
    void filter_by_flags(PointCloudShared& data) {
        if (data.has_cov()) {
            this->filter_->filter_by_flags(*data.covs, *this->flags_);
        }
        if (data.has_normal()) {
            this->filter_->filter_by_flags(*data.normals, *this->flags_);
        }
        this->filter_->filter_by_flags(*data.points, *this->flags_);
    }
};

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
