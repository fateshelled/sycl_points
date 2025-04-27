#pragma once

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
    for (size_t j = 0; j < 3; ++j) {
        const auto val = std::fabs(pt[j]);
        if (val < min_distance || val > max_distance) {
            flag = REMOVE_FLAG;
            return;
        }
    }
}
}  // namespace kernel

template <typename T, typename Allocator>
void filter_by_flags(std::vector<T, Allocator>& data, const shared_vector<uint8_t, sizeof(uint8_t)>& flags) {
    const size_t N = data.size();
    size_t new_size = 0;
    for (size_t i = 0; i < N; ++i) {
        if (flags[i] == INCLUDE_FLAG) {
            data[new_size] = data[i];
            ++new_size;
        }
    }
    data.resize(new_size);
}

template <typename T>
sycl_utils::events filter_by_flags_async(const std::shared_ptr<sycl::queue>& queue_ptr,
                                         std::vector<T, shared_allocator<T, sizeof(T)>>& data,
                                         const shared_vector<uint8_t, sizeof(uint8_t)>& flags) {
    static_assert(std::is_same<T, PointType>::value || std::is_same<T, Covariance>::value, "T is not supported type.");

    sycl_utils::events events;

    const size_t N = data.size();
    if (N == 0) return events;

    std::vector<T, shared_allocator<T, sizeof(T)>> data_copy(N, *queue_ptr);
    auto copy_event = queue_ptr->memcpy(data_copy.data(), data.data(), N * sizeof(T));

    shared_vector<uint32_t, sizeof(uint32_t)> prefix_sum(N, 0, *queue_ptr);
    size_t new_size = 0;
    for (size_t i = 0; i < N; ++i) {
        prefix_sum[i] = new_size;
        new_size += flags[i];
    }

    copy_event.wait();
    data.resize(new_size);

    auto event = queue_ptr->submit([&](sycl::handler& h) {
        const size_t work_group_size = sycl_utils::get_work_group_size(*queue_ptr);
        const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;
        // memory ptr
        T* data_ptr = data.data();
        const T* data_copy_ptr = data_copy.data();
        const auto flag_ptr = flags.data();
        const auto prefix_sum_ptr = prefix_sum.data();
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;
            if (flag_ptr[i] == INCLUDE_FLAG) {
                if constexpr (std::is_same<T, PointType>::value) {
                    const PointType pt = data_copy_ptr[i];
                    data_ptr[prefix_sum_ptr[i]](0) = pt(0);
                    data_ptr[prefix_sum_ptr[i]](1) = pt(1);
                    data_ptr[prefix_sum_ptr[i]](2) = pt(2);
                    data_ptr[prefix_sum_ptr[i]](3) = pt(3);
                } else if constexpr (std::is_same<T, Covariance>::value) {
                    const Covariance cov = data_copy_ptr[i];
                    for (size_t j = 0; j < 3; ++j) {
                        for (size_t k = 0; k < 3; ++k) {
                            data_ptr[prefix_sum_ptr[i]](j, k) = cov(j, k);
                        }
                    }
                }
            }
        });
    });
    events += event;
    return events;
}

void crop_box(const std::shared_ptr<sycl::queue>& queue_ptr, PointContainerShared& data, float min_distance = 1.0f,
              float max_distance = std::numeric_limits<float>::max()) {
    const size_t N = data.size();
    if (N == 0) return;

    shared_vector<uint8_t, sizeof(uint8_t)> flags(N, INCLUDE_FLAG, *queue_ptr);

    auto event = queue_ptr->submit([&](sycl::handler& h) {
        const size_t work_group_size = sycl_utils::get_work_group_size(*queue_ptr);
        const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;
        // memory ptr
        const auto point_ptr = traits::point::const_data_ptr(data);
        auto flag_ptr = flags.data();
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;
            if (!kernel::is_finite(point_ptr[i])) {
                flag_ptr[i] = REMOVE_FLAG;
                return;
            }
            kernel::crop_box(point_ptr[i], flag_ptr[i], min_distance, max_distance);
        });
    });
    event.wait();

    if (sycl_utils::is_nvidia(*queue_ptr)) {
        filter_by_flags_async(queue_ptr, data, flags).wait();
    } else {
        filter_by_flags(data, flags);
    }
}

void crop_box(PointCloudShared& data, float min_distance = 1.0f,
              float max_distance = std::numeric_limits<float>::max()) {
    const size_t N = data.size();
    if (N == 0) return;

    shared_vector<uint8_t, sizeof(uint8_t)> flags(N, INCLUDE_FLAG, *data.queue_ptr);

    auto event = data.queue_ptr->submit([&](sycl::handler& h) {
        const size_t work_group_size = sycl_utils::get_work_group_size(*data.queue_ptr);
        const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;

        const auto point_ptr = data.points_ptr();
        auto flag_ptr = flags.data();
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;
            kernel::crop_box(point_ptr[i], flag_ptr[i], min_distance, max_distance);
        });
    });
    event.wait();

    if (sycl_utils::is_nvidia(*data.queue_ptr)) {
        auto events = filter_by_flags_async(data.queue_ptr, *data.points, flags);
        if (data.has_cov()) {
            events += filter_by_flags_async(data.queue_ptr, *data.covs, flags);
        }
        events.wait();
    } else {
        filter_by_flags(*data.points, flags);
        if (data.has_cov()) {
            filter_by_flags(*data.covs, flags);
        }
    }
}
}  // namespace filter

}  // namespace algorithms
}  // namespace sycl_points