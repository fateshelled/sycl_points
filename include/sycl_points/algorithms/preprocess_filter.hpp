#pragma once

#include <execution>
#include <numeric>
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

class FilterByFlags {
public:
    FilterByFlags(const sycl_utils::DeviceQueue& queue) : queue_(queue) {
        points_copy_ptr_ = std::make_shared<sycl_points::PointContainerShared>(*this->queue_.ptr);
        covs_copy_ptr_ = std::make_shared<sycl_points::CovarianceContainerShared>(*this->queue_.ptr);
        prefix_sum_ptr_ = std::make_shared<shared_vector<uint32_t>>(*this->queue_.ptr);
    }

    template <typename T, size_t AllocSize = 0>
    void filter_by_flags(shared_vector<T, AllocSize>& data, const shared_vector<uint8_t>& flags) const {
        const size_t N = data.size();
        if (N == 0) return;

        // mem_advise to host
        {
            this->queue_.set_accessed_by_host(data.data(), N);
            this->queue_.set_accessed_by_host(flags.data(), N);
        }
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

    template <typename T, size_t AllocSize = 0>
    sycl_utils::events filter_by_flags_async(shared_vector<T, AllocSize>& data, const shared_vector<uint8_t>& flags) {
        static_assert(std::is_same<T, PointType>::value || std::is_same<T, Covariance>::value,
                      "T is not supported type.");

        sycl_utils::events events;

        const size_t N = data.size();
        if (N == 0) return events;

        // copy original vector data
        sycl::event copy_event = this->copy_data(data);

        if (this->prefix_sum_ptr_->size() < N) {
            this->prefix_sum_ptr_->resize(N);
        }

        // calc prefix sum on host
        this->prefix_sum_flags(flags);

        const size_t new_size = this->prefix_sum_ptr_->at(N - 1);

        copy_event.wait();
        data.resize(new_size);

        // apply filter on device
        events += this->filter<T, AllocSize>(data, flags);
        return events;
    }

private:
    sycl_utils::DeviceQueue queue_;
    std::shared_ptr<sycl_points::PointContainerShared> points_copy_ptr_;
    std::shared_ptr<sycl_points::CovarianceContainerShared> covs_copy_ptr_;
    shared_vector_ptr<uint32_t> prefix_sum_ptr_;

    template <typename T, size_t AllocSize = 0>
    sycl::event copy_data(shared_vector<T, AllocSize>& data) {
        const size_t N = data.size();
        if constexpr (std::is_same<T, PointType>::value) {
            if (this->points_copy_ptr_->size() < N) {
                this->points_copy_ptr_->resize(N);
            }
            this->queue_.set_accessed_by_device(this->points_copy_ptr_->data(), N);
            return this->queue_.ptr->memcpy(this->points_copy_ptr_->data(), data.data(), N * sizeof(T));
        } else if constexpr (std::is_same<T, Covariance>::value) {
            if (this->covs_copy_ptr_->size() < N) {
                this->covs_copy_ptr_->resize(N);
            }
            this->queue_.set_accessed_by_device(this->covs_copy_ptr_->data(), N);
            return this->queue_.ptr->memcpy(this->covs_copy_ptr_->data(), data.data(), N * sizeof(T));
        }
        throw std::runtime_error("Not supported type T");
    }

    void prefix_sum_flags(const shared_vector<uint8_t>& flags) {
        const size_t N = flags.size();

        // mem_advise to host
        {
            this->queue_.set_accessed_by_host(flags.data(), N);
            this->queue_.set_accessed_by_host(this->prefix_sum_ptr_->data(), N);
        }
        // calc prefix sum
#if __cplusplus >= 202002L
        std::transform_inclusive_scan(std::execution::unseq, flags.begin(), flags.begin() + N,
                                      this->prefix_sum_ptr_->begin(), std::plus<uint32_t>(),
                                      [](uint8_t a) { return static_cast<uint32_t>(a); });
#else
        std::transform_inclusive_scan(flags.begin(), flags.begin() + N, this->prefix_sum_ptr_->begin(),
                                      std::plus<uint32_t>(), [](uint8_t a) { return static_cast<uint32_t>(a); });
#endif
        // mem_advise clear
        {
            this->queue_.clear_accessed_by_host(flags.data(), N);
            this->queue_.clear_accessed_by_host(this->prefix_sum_ptr_->data(), N);
        }
    }

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

/// @brief Preprocessing filter
class PreprocessFilter {
public:
    /// @brief Constructor
    /// @param queue SYCL queue
    PreprocessFilter(const sycl_utils::DeviceQueue& queue) : queue_(queue) {
        filter_ = std::make_shared<FilterByFlags>(this->queue_);
        flags_ = std::make_shared<shared_vector<uint8_t>>(*this->queue_.ptr);
    }

    /// @brief L∞ distance (chebyshev distance) filter.
    /// @param data Point Cloud
    /// @param min_distance min distance
    /// @param max_distance max distance
    void box_filter(PointCloudShared& data, float min_distance = 1.0f,
                    float max_distance = std::numeric_limits<float>::max()) {
        const size_t N = data.size();
        if (N == 0) return;

        if (this->flags_->size() < N) {
            this->flags_->resize(N);
        }
        // initialize flags
        this->queue_.ptr->fill(this->flags_->data(), INCLUDE_FLAG, N).wait();

        // mem_advise set to device
        this->queue_.set_accessed_by_device(this->flags_->data(), N);
        this->queue_.set_accessed_by_device(data.points_ptr(), N);

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t work_group_size = this->queue_.get_work_group_size();
            const size_t global_size = this->queue_.get_global_size(N);
            // memory ptr
            const auto point_ptr = data.points_ptr();
            auto flag_ptr = this->flags_->data();
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                if (flag_ptr[i] == REMOVE_FLAG) return;
                if (!kernel::is_finite(point_ptr[i])) {
                    flag_ptr[i] = REMOVE_FLAG;
                    return;
                }
                kernel::box_filter(point_ptr[i], flag_ptr[i], min_distance, max_distance);
            });
        });
        event.wait();
        this->queue_.clear_accessed_by_device(this->flags_->data(), N);
        this->queue_.clear_accessed_by_device(data.points_ptr(), N);

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

private:
    sycl_utils::DeviceQueue queue_;
    std::shared_ptr<FilterByFlags> filter_;
    shared_vector_ptr<uint8_t> flags_;
};
}  // namespace filter

}  // namespace algorithms
}  // namespace sycl_points