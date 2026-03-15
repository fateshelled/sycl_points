#pragma once

#include <algorithm>
#include <execution>
#include <limits>
#include <random>

#include "sycl_points/algorithms/filter/preprocess_operator/common.hpp"
#include "sycl_points/algorithms/filter/preprocess_operator/preprocess_operator_base.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {
namespace preprocess_operator {

class FarthestPointSamplingOperator : public PreprocessOperatorBase {
public:
    FarthestPointSamplingOperator(const sycl_utils::DeviceQueue& queue, shared_vector_ptr<uint8_t> flags,
                                  InitializeFlagsFn initialize_flags, FilterByFlagsFn filter_by_flags)
        : PreprocessOperatorBase(queue, std::move(flags), std::move(initialize_flags), std::move(filter_by_flags)),
          dist_sq_(std::make_shared<shared_vector<float>>(*this->queue_.ptr)),
          mt_(1234) {}

    void set_random_seed(uint_fast32_t seed) { this->mt_.seed(seed); }

    void apply(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        const size_t N = source.size();
        if (N <= sampling_num) {
            // Keep all points when the requested sample count covers the full input, including empty input.
            this->copy_source_to_output(source, output);
            return;
        }

        if (this->dist_sq_->size() < N) {
            this->dist_sq_->resize(N);
        }

        {
            this->queue_.set_accessed_by_device(source.points_ptr(), N);
            this->queue_.set_accessed_by_device(this->dist_sq_->data(), N);
        }

        sycl_utils::events init_events;
        init_events += this->initialize_flags_(N, REMOVE_FLAG);
        init_events += this->queue_.ptr->fill(this->dist_sq_->data(), std::numeric_limits<float>::max(), N);
        init_events.wait_and_throw();

        std::uniform_int_distribution<size_t> dist(0, N - 1);
        size_t selected_idx = dist(this->mt_);
        this->flags_->at(selected_idx) = INCLUDE_FLAG;

        for (size_t iter = 1; iter < sampling_num; ++iter) {
            this->queue_.ptr
                ->submit([&](sycl::handler& h) {
                    const size_t work_group_size = this->queue_.get_work_group_size();
                    const size_t global_size = this->queue_.get_global_size(N);
                    const auto point_ptr = source.points_ptr();
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

            const auto max_elem =
#if __cplusplus >= 202002L
                std::max_element(std::execution::unseq, this->dist_sq_->data(), this->dist_sq_->data() + N);
#else
                std::max_element(this->dist_sq_->data(), this->dist_sq_->data() + N);
#endif
            const size_t max_elem_idx = std::distance(this->dist_sq_->data(), max_elem);
            selected_idx = max_elem_idx;
            this->flags_->at(max_elem_idx) = INCLUDE_FLAG;
        }

        {
            this->queue_.clear_accessed_by_device(source.points_ptr(), N);
            this->queue_.clear_accessed_by_device(this->dist_sq_->data(), N);
        }

        this->filter_by_flags_(source, output);
    }

private:
    shared_vector_ptr<float> dist_sq_;
    std::mt19937 mt_;
};

}  // namespace preprocess_operator
}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
