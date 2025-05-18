#pragma once

#include <execution>
#include <numeric>
#include <random>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace filter {

constexpr uint8_t REMOVE_FLAG = 0;
constexpr uint8_t INCLUDE_FLAG = 1;

namespace kernel {

SYCL_EXTERNAL inline bool is_finite(const PointType& pt) {
    return std::isfinite(pt[0]) && std::isfinite(pt[3]) && std::isfinite(pt[2]) && std::isfinite(pt[3]);
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
        this->points_copy_ptr_ = std::make_shared<sycl_points::PointContainerShared>(*this->queue_.ptr);
        this->covs_copy_ptr_ = std::make_shared<sycl_points::CovarianceContainerShared>(*this->queue_.ptr);
        this->prefix_sum_ptr_ = std::make_shared<shared_vector<uint32_t>>(*this->queue_.ptr);
        this->group_sums_ptr_ = std::make_shared<shared_vector<uint32_t>>(*this->queue_.ptr);
        this->group_prefix_ptr_ = std::make_shared<shared_vector<uint32_t>>(*this->queue_.ptr);
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
        data.resize(new_size);
        // mem_advise clear
        {
            this->queue_.clear_accessed_by_host(data.data(), N);
            this->queue_.clear_accessed_by_host(flags.data(), N);
        }
    }

    /// @brief Filter data asynchronously on device
    /// @tparam T Data type (must be PointType or Covariance)
    /// @tparam AllocSize Optional allocator size
    /// @param data Data to be filtered
    /// @param flags Flags indicating which elements to keep (INCLUDE_FLAG) or remove
    /// @return Events representing the asynchronous operations
    template <typename T, size_t AllocSize = 0>
    sycl_utils::events filter_by_flags_async(shared_vector<T, AllocSize>& data, const shared_vector<uint8_t>& flags) {
        static_assert(std::is_same<T, PointType>::value || std::is_same<T, Covariance>::value,
                      "T is not supported type.");

        sycl_utils::events events;

        const size_t N = data.size();
        if (N == 0) return events;

        // Copy original data to preserve it during filtering
        auto copy_event = this->copy_data(data);

        // Calculate inclusive prefix sum of flags
        this->prefix_sum_flags(flags);

        // Get new size from last element of prefix sum (total count of INCLUDE_FLAG elements)
        const size_t new_size = this->prefix_sum_ptr_->at(N - 1);

        // Wait for copy to complete before resizing
        copy_event.wait();
        data.resize(new_size);

        // Apply filter using prefix sum for destination indices
        events += this->filter<T, AllocSize>(data, flags);
        return events;
    }

private:
    sycl_utils::DeviceQueue queue_;                                          // SYCL queue
    std::shared_ptr<sycl_points::PointContainerShared> points_copy_ptr_;     // Copy of point data
    std::shared_ptr<sycl_points::CovarianceContainerShared> covs_copy_ptr_;  // Copy of covariance data
    shared_vector_ptr<uint32_t> prefix_sum_ptr_;                             // Prefix sum results
    shared_vector_ptr<uint32_t> group_sums_ptr_;                             // Group sums for prefix calculation
    shared_vector_ptr<uint32_t> group_prefix_ptr_;                           // Group prefix sums

    /// @brief Copy input data to temporary storage
    /// @tparam T Data type
    /// @tparam AllocSize Optional allocator size
    /// @param data Data to copy
    /// @return Events representing the copy operation
    template <typename T, size_t AllocSize = 0>
    sycl_utils::events copy_data(shared_vector<T, AllocSize>& data) {
        const size_t N = data.size();
        sycl_utils::events events;

        if constexpr (std::is_same<T, PointType>::value) {
            if (this->points_copy_ptr_->size() < N) {
                this->points_copy_ptr_->resize(N);
            }
            this->queue_.set_accessed_by_device(this->points_copy_ptr_->data(), N);
            events += this->queue_.ptr->memcpy(this->points_copy_ptr_->data(), data.data(), N * sizeof(T));
        } else if constexpr (std::is_same<T, Covariance>::value) {
            if (this->covs_copy_ptr_->size() < N) {
                this->covs_copy_ptr_->resize(N);
            }
            this->queue_.set_accessed_by_device(this->covs_copy_ptr_->data(), N);
            events += this->queue_.ptr->memcpy(this->covs_copy_ptr_->data(), data.data(), N * sizeof(T));
        } else {
            throw std::runtime_error("Not supported type T");
        }
        return events;
    }

    /// @brief Calculate inclusive prefix sum of flags
    /// @param flags Input flags (REMOVE_FLAG or INCLUDE_FLAG)
    void prefix_sum_flags(const shared_vector<uint8_t>& flags) {
        const size_t N = flags.size();
        if (N == 0) return;

        if (this->prefix_sum_ptr_->size() < N) {
            this->prefix_sum_ptr_->resize(N);
        }

        const size_t wg_size = this->queue_.get_work_group_size();
        const size_t num_groups = (N + wg_size - 1) / wg_size;

        // Resize and initialize group sum buffers
        if (group_sums_ptr_->size() < num_groups) {
            this->group_sums_ptr_->resize(num_groups);
            this->group_prefix_ptr_->resize(num_groups);
        }
        // mem_advise to device
        {
            this->queue_.set_accessed_by_device(this->prefix_sum_ptr_->data(), N);
            this->queue_.set_accessed_by_device(this->group_sums_ptr_->data(), num_groups);
            this->queue_.set_accessed_by_device(this->group_prefix_ptr_->data(), num_groups);
        }

        // Initialize buffers to zero
        sycl_utils::events init_event;
        {
            init_event += this->queue_.ptr->fill(this->group_sums_ptr_->data(), (uint32_t)0, num_groups);
            init_event += this->queue_.ptr->fill(this->group_prefix_ptr_->data(), (uint32_t)0, num_groups);
        }
        init_event.wait();

        // Step 1: Inclusive scan within each work group
        {
            auto event1 = this->queue_.ptr->submit([&](sycl::handler& h) {
                const auto flags_ptr = flags.data();
                auto prefix_sum_ptr = this->prefix_sum_ptr_->data();
                auto group_sums_ptr = this->group_sums_ptr_->data();
                const auto N_capture = N;
                const auto wg_size_capture = wg_size;

                h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                    const size_t gid = item.get_global_id(0);
                    const size_t lid = item.get_local_id(0);
                    // Calculate group ID
                    const size_t group_id = gid / wg_size_capture;
                    auto group = item.get_group();

                    // Each thread loads a value (0 if out of range)
                    uint32_t value = (gid < N_capture) ? static_cast<uint32_t>(flags_ptr[gid]) : 0;

                    // Inclusive scan within the work group
                    uint32_t scan_result = sycl::inclusive_scan_over_group(group, value, sycl::plus<uint32_t>());

                    // Write results to global memory
                    if (gid < N_capture) {
                        prefix_sum_ptr[gid] = scan_result;
                    }

                    // Only the last element of each work group saves the sum
                    if (lid == wg_size_capture - 1 || gid == N_capture - 1) {
                        group_sums_ptr[group_id] = scan_result;
                    }
                });
            });
            event1.wait();
        }

        // Complete if only one group
        if (num_groups == 1) return;

        // Step 2: Perform an exclusive scan on group sums
        {
            // 1. Shift group_sum to create exclusive scan input
            auto event2a = this->queue_.ptr->submit([&](sycl::handler& h) {
                auto group_sums_ptr = this->group_sums_ptr_->data();
                auto group_prefix_ptr = this->group_prefix_ptr_->data();
                const auto num_groups_capture = num_groups;

                h.parallel_for(sycl::range<1>(num_groups_capture), [=](sycl::id<1> idx) {
                    const size_t i = idx[0];
                    if (i == 0) {
                        group_prefix_ptr[i] = 0;  // First element is zero
                    } else {
                        group_prefix_ptr[i] = group_sums_ptr[i - 1];  // Shift
                    }
                });
            });
            event2a.wait_and_throw();

            // 2. Sequential scan for stability
            auto event2b = this->queue_.ptr->submit([&](sycl::handler& h) {
                auto group_prefix_ptr = this->group_prefix_ptr_->data();
                const auto num_groups_capture = num_groups;

                h.single_task([=]() {
                    for (size_t i = 1; i < num_groups_capture; ++i) {
                        group_prefix_ptr[i] += group_prefix_ptr[i - 1];
                    }
                });
            });
            event2b.wait_and_throw();

            // Copy result back to group_sums for use in step 3
            this->queue_.ptr
                ->memcpy(this->group_sums_ptr_->data(), this->group_prefix_ptr_->data(), num_groups * sizeof(uint32_t))
                .wait_and_throw();
        }

        // Step 3: Add the appropriate group offset to each element
        {
            auto event3 = this->queue_.ptr->submit([&](sycl::handler& h) {
                auto prefix_sum_ptr = this->prefix_sum_ptr_->data();
                auto group_sums_ptr = this->group_sums_ptr_->data();
                const auto N_capture = N;
                const auto wg_size_capture = wg_size;

                h.parallel_for(sycl::range<1>(N_capture), [=](sycl::id<1> idx) {
                    const size_t i = idx[0];
                    const size_t group_id = i / wg_size_capture;

                    // Add the appropriate group offset
                    if (group_id > 0) {
                        prefix_sum_ptr[i] += group_sums_ptr[group_id];
                    }
                });
            });
            event3.wait();
        }
        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(this->prefix_sum_ptr_->data(), N);
            this->queue_.clear_accessed_by_device(this->group_sums_ptr_->data(), num_groups);
            this->queue_.clear_accessed_by_device(this->group_prefix_ptr_->data(), num_groups);
        }
    }

    /// @brief Apply filter using prefix sum results
    /// @tparam T Data type
    /// @tparam AllocSize Optional allocator size
    /// @param data Output data buffer (already resized)
    /// @param flags Flags indicating which elements to keep
    /// @return Event representing the filter operation
    template <typename T, size_t AllocSize = 0>
    sycl::event filter(shared_vector<T, AllocSize>& data, const shared_vector<uint8_t>& flags) {
        const size_t N = flags.size();

        // mem_advise to device
        {
            this->queue_.set_accessed_by_device(flags.data(), N);
            this->queue_.set_accessed_by_device(this->prefix_sum_ptr_->data(), N);
        }

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t work_group_size = this->queue_.get_work_group_size();
            const size_t global_size = this->queue_.get_global_size(N);

            // memory ptr
            T* data_ptr = data.data();
            T* copy_ptr;
            if constexpr (std::is_same<T, PointType>::value) {
                copy_ptr = this->points_copy_ptr_->data();
            } else if constexpr (std::is_same<T, Covariance>::value) {
                copy_ptr = this->covs_copy_ptr_->data();
            } else {
                std::runtime_error("Invalid Type [T]");
            }

            const auto flag_ptr = flags.data();
            const auto prefix_sum_ptr = this->prefix_sum_ptr_->data();

            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;

                // Only process elements with INCLUDE_FLAG
                if (flag_ptr[i] == INCLUDE_FLAG) {
                    if constexpr (std::is_same<T, PointType>::value) {
                        const PointType pt = copy_ptr[i];
                        eigen_utils::copy<4, 1>(pt, data_ptr[prefix_sum_ptr[i]]);
                    } else if constexpr (std::is_same<T, Covariance>::value) {
                        const Covariance cov = copy_ptr[i];
                        eigen_utils::copy<4, 4>(cov, data_ptr[prefix_sum_ptr[i]]);
                    }
                }
            });
        });
        return event;
    }
};

/// @brief Preprocessing filter for point cloud data
class PreprocessFilter {
public:
    /// @brief Constructor
    /// @param queue SYCL queue
    PreprocessFilter(const sycl_utils::DeviceQueue& queue) : queue_(queue) {
        filter_ = std::make_shared<FilterByFlags>(this->queue_);
        flags_ = std::make_shared<shared_vector<uint8_t>>(*this->queue_.ptr);

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

        this->initialize_flags(N);

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
            auto flag_ptr = this->flags_->data();
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
        if (N < sampling_num) return;

        this->initialize_flags(N, REMOVE_FLAG);

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

private:
    sycl_utils::DeviceQueue queue_;
    std::shared_ptr<FilterByFlags> filter_;
    shared_vector_ptr<uint8_t> flags_;
    std::mt19937 mt_;

    /// @brief Initializes the flags vector with a specified value
    /// @param data_size Size needed for the flags vector
    /// @param initial_flag Initial value to fill the flags with (INCLUDE_FLAG or REMOVE_FLAG)
    void initialize_flags(size_t data_size, uint8_t initial_flag = INCLUDE_FLAG) {
        if (this->flags_->size() < data_size) {
            this->flags_->resize(data_size);
        }
        this->queue_.ptr->fill(this->flags_->data(), initial_flag, data_size).wait();
    }

    /// @brief Applies filtering based on the current flags
    /// @param data Point cloud to be filtered (modified in-place)
    void filter_by_flags(PointCloudShared& data) {
        if (this->queue_.is_nvidia()) {
            auto events = filter_->filter_by_flags_async(*data.points, *this->flags_);
            if (data.has_cov()) {
                events += filter_->filter_by_flags_async(*data.covs, *this->flags_);
            }
            events.wait();
        } else {
            filter_->filter_by_flags(*data.points, *this->flags_);
            if (data.has_cov()) {
                filter_->filter_by_flags(*data.covs, *this->flags_);
            }
        }
    }
};

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points