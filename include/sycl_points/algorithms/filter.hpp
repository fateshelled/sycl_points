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

SYCL_EXTERNAL inline void crop_box(const PointType& pt, uint8_t& flag, float min_distance, float max_distance) {
#pragma unroll 3
    for (size_t j = 0; j < 3; ++j) {
        const auto val = std::fabs(pt[j]);
        if (val < min_distance || val > max_distance) {
            flag = REMOVE_FLAG;
            return;
        }
    }
}

SYCL_EXTERNAL inline void copy_point(const PointType& src, PointType& dst) {
    dst(0) = src(0);
    dst(1) = src(1);
    dst(2) = src(2);
    dst(3) = src(3);
}

SYCL_EXTERNAL inline void copy_covariance(const Covariance& src, Covariance& dst) {
#pragma unroll 4
    for (size_t j = 0; j < 4; ++j) {
#pragma unroll 4
        for (size_t k = 0; k < 4; ++k) {
            dst(j, k) = src(j, k);
        }
    }
}
}  // namespace kernel

class FilterByFlags {
public:
    FilterByFlags(const std::shared_ptr<sycl::queue>& queue_ptr) : queue_ptr_(queue_ptr) {
        points_copy_ptr_ = std::make_shared<sycl_points::PointContainerShared>(*this->queue_ptr_);
        covs_copy_ptr_ = std::make_shared<sycl_points::CovarianceContainerShared>(*this->queue_ptr_);
        prefix_sum_ptr_ = std::make_shared<shared_vector<uint32_t, sizeof(uint32_t)>>(*this->queue_ptr_);
    }

    template <typename T, size_t AllocSize = 0>
    static void filter_by_flags(shared_vector<T, AllocSize>& data,
                                const shared_vector<uint8_t, sizeof(uint8_t)>& flags) {
        const size_t N = data.size();
        if (N == 0) return;

        size_t new_size = 0;
        for (size_t i = 0; i < N; ++i) {
            if (flags[i] == INCLUDE_FLAG) {
                data[new_size] = data[i];
                ++new_size;
            }
        }
        data.resize(new_size);
    }

    template <typename T, size_t AllocSize = 0>
    sycl_utils::events filter_by_flags_async(shared_vector<T, AllocSize>& data,
                                             const shared_vector<uint8_t, sizeof(uint8_t)>& flags) {
        static_assert(std::is_same<T, PointType>::value || std::is_same<T, Covariance>::value,
                      "T is not supported type.");

        sycl_utils::events events;

        const size_t N = data.size();
        if (N == 0) return events;

        // allocate memory
        sycl::event copy_event;
        if constexpr (std::is_same<T, PointType>::value) {
            if (this->points_copy_ptr_->size() < N) {
                this->points_copy_ptr_->resize(N);
            }
            copy_event = this->queue_ptr_->memcpy(this->points_copy_ptr_->data(), data.data(), N * sizeof(T));
        } else if constexpr (std::is_same<T, Covariance>::value) {
            if (this->covs_copy_ptr_->size() < N) {
                this->covs_copy_ptr_->resize(N);
            }
            copy_event = this->queue_ptr_->memcpy(this->covs_copy_ptr_->data(), data.data(), N * sizeof(T));
        }

        if (this->prefix_sum_ptr_->size() < N) {
            this->prefix_sum_ptr_->resize(N);
        }

        // calc prefix sum
        std::transform_inclusive_scan(
            std::execution::unseq, flags.begin(), flags.begin() + N, this->prefix_sum_ptr_->begin(),
            [](uint32_t a, uint32_t b) { return a + b; }, [](uint8_t a) { return static_cast<uint32_t>(a); });
        const size_t new_size = this->prefix_sum_ptr_->at(N - 1);

        copy_event.wait();
        data.resize(new_size);

        auto event = this->queue_ptr_->submit([&](sycl::handler& h) {
            const size_t work_group_size = sycl_utils::get_work_group_size(*this->queue_ptr_);
            const size_t global_size = sycl_utils::get_global_size(N, work_group_size);

            // memory ptr
            T* data_ptr = data.data();
            T* copy_ptr;
            if constexpr (std::is_same<T, PointType>::value) {
                copy_ptr = this->points_copy_ptr_->data();
            } else if constexpr (std::is_same<T, Covariance>::value) {
                copy_ptr = this->covs_copy_ptr_->data();
            }

            const auto flag_ptr = flags.data();
            const auto prefix_sum_ptr = this->prefix_sum_ptr_->data();
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                if (flag_ptr[i] == INCLUDE_FLAG) {
                    if constexpr (std::is_same<T, PointType>::value) {
                        const PointType pt = copy_ptr[i];
                        kernel::copy_point(pt, data_ptr[prefix_sum_ptr[i]]);
                    } else if constexpr (std::is_same<T, Covariance>::value) {
                        const Covariance cov = copy_ptr[i];
                        kernel::copy_covariance(cov, data_ptr[prefix_sum_ptr[i]]);
                    }
                }
            });
        });
        events += event;
        return events;
    }

private:
    std::shared_ptr<sycl::queue> queue_ptr_;
    std::shared_ptr<sycl_points::PointContainerShared> points_copy_ptr_;
    std::shared_ptr<sycl_points::CovarianceContainerShared> covs_copy_ptr_;
    std::shared_ptr<shared_vector<uint32_t, sizeof(uint32_t)>> prefix_sum_ptr_;
};

class PreprocessFilter {
public:
    PreprocessFilter(const std::shared_ptr<sycl::queue>& queue_ptr) : queue_ptr_(queue_ptr) {
        filter_ = std::make_shared<FilterByFlags>(this->queue_ptr_);
        flags_ = std::make_shared<shared_vector<uint8_t, sizeof(uint8_t)>>(*this->queue_ptr_);
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
        this->queue_ptr_->fill(this->flags_->data(), INCLUDE_FLAG, N).wait();

        auto event = this->queue_ptr_->submit([&](sycl::handler& h) {
            const size_t work_group_size = sycl_utils::get_work_group_size(*this->queue_ptr_);
            const size_t global_size = sycl_utils::get_global_size(N, work_group_size);
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
                kernel::crop_box(point_ptr[i], flag_ptr[i], min_distance, max_distance);
            });
        });
        event.wait();

        if (sycl_utils::is_nvidia(*this->queue_ptr_)) {
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
    std::shared_ptr<sycl::queue> queue_ptr_;
    std::shared_ptr<FilterByFlags> filter_;
    std::shared_ptr<shared_vector<uint8_t, sizeof(uint8_t)>> flags_;
};
}  // namespace filter

}  // namespace algorithms
}  // namespace sycl_points