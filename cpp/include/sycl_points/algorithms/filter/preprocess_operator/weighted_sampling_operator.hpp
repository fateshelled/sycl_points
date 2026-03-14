#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sycl_points/algorithms/filter/preprocess_operator/common.hpp"
#include "sycl_points/algorithms/filter/preprocess_operator/preprocess_operator_base.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {
namespace preprocess_operator {

class WeightedSamplingOperator : public PreprocessOperatorBase {
public:
    WeightedSamplingOperator(const sycl_utils::DeviceQueue& queue, shared_vector_ptr<uint8_t> flags,
                             InitializeFlagsFn initialize_flags, FilterByFlagsFn filter_by_flags)
        : PreprocessOperatorBase(queue, std::move(flags), std::move(initialize_flags), std::move(filter_by_flags)),
          mt_(1234) {}

    void set_random_seed(uint_fast32_t seed) { this->mt_.seed(seed); }

    void apply(const PointCloudShared& source, PointCloudShared& output, const shared_vector<float>& weights,
               size_t sampling_num) {
        const size_t N = source.size();
        if (N == 0) return;
        if (N <= sampling_num) return;

        if (weights.size() != N) {
            throw std::invalid_argument("[PreprocessFilter::weighted_random_sampling] weights size must match points");
        }

        size_t positive_weight_count = 0;
        for (size_t i = 0; i < N; ++i) {
            const float weight = weights[i];
            if (!std::isfinite(weight) || weight < 0.0f) {
                throw std::invalid_argument(
                    "[PreprocessFilter::weighted_random_sampling] weights must be finite and non-negative");
            }
            if (weight > 0.0f) {
                ++positive_weight_count;
            }
        }

        if (positive_weight_count == 0) {
            throw std::invalid_argument(
                "[PreprocessFilter::weighted_random_sampling] at least one weight must be positive");
        }
        if (sampling_num > positive_weight_count) {
            throw std::invalid_argument(
                "[PreprocessFilter::weighted_random_sampling] sampling_num exceeds positive-weight points");
        }

        this->initialize_flags_(N, REMOVE_FLAG).wait_and_throw();

        this->queue_.set_accessed_by_host(this->flags_->data(), N);

        using KeyIndexPair = std::pair<float, size_t>;
        std::priority_queue<KeyIndexPair, std::vector<KeyIndexPair>, std::greater<KeyIndexPair>> selected;
        std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), 1.0f);

        for (size_t i = 0; i < N; ++i) {
            const float weight = weights[i];
            if (weight <= 0.0f) continue;

            const float key = std::log(dist(this->mt_)) / weight;
            if (selected.size() < sampling_num) {
                selected.emplace(key, i);
                continue;
            }

            if (selected.top().first < key) {
                selected.pop();
                selected.emplace(key, i);
            }
        }

        while (!selected.empty()) {
            (*this->flags_)[selected.top().second] = INCLUDE_FLAG;
            selected.pop();
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
