#pragma once

#include <cmath>
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

class MixedRandomSamplingOperator : public PreprocessOperatorBase {
public:
    MixedRandomSamplingOperator(const sycl_utils::DeviceQueue& queue, shared_vector_ptr<uint8_t> flags,
                                InitializeFlagsFn initialize_flags, FilterByFlagsFn filter_by_flags)
        : PreprocessOperatorBase(queue, std::move(flags), std::move(initialize_flags), std::move(filter_by_flags)),
          mt_(1234) {}

    void set_random_seed(uint_fast32_t seed) { this->mt_.seed(seed); }

    void apply(const PointCloudShared& source, PointCloudShared& output, const shared_vector<float>& weights,
               size_t sampling_num, float weighted_ratio) {
        const size_t N = source.size();
        if (N <= sampling_num) {
            this->copy_source_to_output(source, output);
            return;
        }

        if (weights.size() != N) {
            throw std::invalid_argument("[PreprocessFilter::mixed_random_sampling] weights size must match points");
        }
        if (!std::isfinite(weighted_ratio) || weighted_ratio < 0.0f || weighted_ratio > 1.0f) {
            throw std::invalid_argument(
                "[PreprocessFilter::mixed_random_sampling] weighted_ratio must be within [0.0, 1.0]");
        }

        const size_t weighted_target =
            static_cast<size_t>(std::floor(static_cast<double>(sampling_num) * weighted_ratio));

        this->initialize_flags_(N, REMOVE_FLAG).wait_and_throw();
        this->queue_.set_accessed_by_host(this->flags_->data(), N);
        this->queue_.set_accessed_by_host(weights.data(), N);

        using KeyIndexPair = std::pair<float, size_t>;
        std::priority_queue<KeyIndexPair, std::vector<KeyIndexPair>, std::greater<KeyIndexPair>> selected;
        std::uniform_real_distribution<float> weighted_dist(std::numeric_limits<float>::min(), 1.0f);

        for (size_t i = 0; i < N; ++i) {
            const float weight = weights[i];
            if (!std::isfinite(weight) || weight < 0.0f) {
                this->queue_.clear_accessed_by_host(weights.data(), N);
                this->queue_.clear_accessed_by_host(this->flags_->data(), N);
                throw std::invalid_argument(
                    "[PreprocessFilter::mixed_random_sampling] weights must be finite and non-negative");
            }
            if (weight <= 0.0f) continue;

            const float key = std::log(weighted_dist(this->mt_)) / weight;
            if (selected.size() < weighted_target) {
                selected.emplace(key, i);
                continue;
            }

            if (!selected.empty() && selected.top().first < key) {
                selected.pop();
                selected.emplace(key, i);
            }
        }

        while (!selected.empty()) {
            (*this->flags_)[selected.top().second] = INCLUDE_FLAG;
            selected.pop();
        }

        std::vector<size_t> remaining_indices;
        remaining_indices.reserve(N);
        size_t selected_count = 0;
        for (size_t i = 0; i < N; ++i) {
            if ((*this->flags_)[i] == INCLUDE_FLAG) {
                ++selected_count;
            } else {
                remaining_indices.push_back(i);
            }
        }

        const size_t uniform_target = std::min(sampling_num - selected_count, remaining_indices.size());
        for (size_t i = 0; i < uniform_target; ++i) {
            std::uniform_int_distribution<size_t> uniform_dist(i, remaining_indices.size() - 1);
            const size_t j = uniform_dist(this->mt_);
            std::swap(remaining_indices[i], remaining_indices[j]);
            (*this->flags_)[remaining_indices[i]] = INCLUDE_FLAG;
        }

        this->queue_.clear_accessed_by_host(weights.data(), N);
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
