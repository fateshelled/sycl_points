#pragma once

#include <numeric>
#include <random>
#include <vector>

#include "sycl_points/algorithms/filter/preprocess_operator/common.hpp"
#include "sycl_points/algorithms/filter/preprocess_operator/preprocess_operator_base.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {
namespace preprocess_operator {

class RandomSamplingOperator : public PreprocessOperatorBase {
public:
    RandomSamplingOperator(const sycl_utils::DeviceQueue& queue, shared_vector_ptr<uint8_t> flags,
                           InitializeFlagsFn initialize_flags, FilterByFlagsFn filter_by_flags)
        : PreprocessOperatorBase(queue, std::move(flags), std::move(initialize_flags), std::move(filter_by_flags)),
          mt_(1234) {}

    void set_random_seed(uint_fast32_t seed) { this->mt_.seed(seed); }

    void apply(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        const size_t N = source.size();
        if (N <= sampling_num) {
            // Keep all points when the requested sample count covers the full input, including empty input.
            this->copy_source_to_output(source, output);
            return;
        }

        this->initialize_flags_(N, REMOVE_FLAG).wait_and_throw();

        this->queue_.set_accessed_by_host(this->flags_->data(), N);

        std::vector<size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        for (size_t i = 0; i < sampling_num; ++i) {
            std::uniform_int_distribution<size_t> dist(i, N - 1);
            const size_t j = dist(this->mt_);
            std::swap(indices[i], indices[j]);
        }

        for (size_t i = 0; i < sampling_num; ++i) {
            (*this->flags_)[indices[i]] = INCLUDE_FLAG;
        }

        this->queue_.clear_accessed_by_host(this->flags_->data(), N);

        this->filter_by_flags_(source, output);
    }

private:
    std::mt19937 mt_;
};

}  // namespace preprocess_operator
}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
