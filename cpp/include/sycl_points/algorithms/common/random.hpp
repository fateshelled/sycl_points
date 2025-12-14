#pragma once

#include <oneapi/dpl/random>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {
namespace algorithms {
namespace random {

template <typename T>
sycl_utils::events shuffle_async(sycl::queue& queue, shared_vector<T>& input, size_t start_idx,
                                 size_t end_idx, size_t seed) {
    sycl_utils::events events;
    events += queue.submit([&](sycl::handler& h) {
        const auto s = seed;
        const auto begin = start_idx;
        const auto end = std::min(end_idx, input.size() - 1);
        auto data_ptr = input.data();
        h.single_task([=]() {
            oneapi::dpl::minstd_rand engine(s);
            for (auto i = begin; i < end; ++i) {
                oneapi::dpl::uniform_int_distribution<size_t> dist(i, end - 1);
                const auto j = dist(engine);
                std::swap(data_ptr[i], data_ptr[j]);
            }
        });
    });
    return events;
}

template <typename T>
sycl_utils::events shuffle_async(sycl::queue& queue, shared_vector<T>& input, size_t seed) {
    return shuffle_async(queue, input, 0, input.size() - 1, seed);
}
}  // namespace random
}  // namespace algorithms
}  // namespace sycl_points
