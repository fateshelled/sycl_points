#pragma once

#include <oneapi/dpl/random>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {
namespace algorithms {
namespace random {

template <typename T, size_t Alignment = 0>
sycl_utils::events shuffle_async(sycl::queue& queue, shared_vector<T, Alignment>& input, size_t start_idx, size_t end_idx, size_t seed) {
    sycl_utils::events events;
    events += queue.submit([&](sycl::handler& h) {
        const auto s = seed;
        const auto begin = start_idx;
        const auto end = std::min(end_idx, input.size() - 1);
        auto data_ptr = input.data();
        h.single_task([=]() {
            oneapi::dpl::minstd_rand engine(s);
            for (int32_t i = end; i > begin; --i) {
                oneapi::dpl::uniform_int_distribution<size_t> dist(0, i);
                const size_t j = dist(engine);

                std::swap(data_ptr[i], data_ptr[j]);
            }
        });
    });
    return events;
}

template <typename T, size_t Alignment = 0>
sycl_utils::events shuffle_async(sycl::queue& queue, shared_vector<T, Alignment>& input, size_t seed) {
    return shuffle_async(queue, input, 0, input.size() - 1, seed);
}
}  // namespace random
}  // namespace algorithms
}  // namespace sycl_points