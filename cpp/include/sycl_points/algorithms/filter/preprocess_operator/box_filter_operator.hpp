#pragma once

#include <limits>

#include "sycl_points/algorithms/filter/preprocess_operator/common.hpp"
#include "sycl_points/algorithms/filter/preprocess_operator/preprocess_operator_base.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {
namespace preprocess_operator {

class BoxFilterOperator : public PreprocessOperatorBase {
public:
    BoxFilterOperator(const sycl_utils::DeviceQueue& queue, shared_vector_ptr<uint8_t> flags,
                      InitializeFlagsFn initialize_flags, FilterByFlagsFn filter_by_flags)
        : PreprocessOperatorBase(queue, std::move(flags), std::move(initialize_flags), std::move(filter_by_flags)) {}

    void apply(const PointCloudShared& source, PointCloudShared& output, float min_distance = 1.0f,
               float max_distance = std::numeric_limits<float>::max()) const {
        const size_t N = source.size();
        if (N == 0) return;

        this->initialize_flags_(N, INCLUDE_FLAG).wait_and_throw();

        {
            this->queue_.set_accessed_by_device(this->flags_->data(), N);
            this->queue_.set_accessed_by_device(source.points_ptr(), N);
        }

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t work_group_size = this->queue_.get_work_group_size();
            const size_t global_size = this->queue_.get_global_size(N);
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

        {
            this->queue_.clear_accessed_by_device(this->flags_->data(), N);
            this->queue_.clear_accessed_by_device(source.points_ptr(), N);
        }

        this->filter_by_flags_(source, output);
    }
};

}  // namespace preprocess_operator
}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
