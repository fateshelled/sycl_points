#pragma once

#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace knn {

/// @brief Structure to store K nearest neighbors and their distances
struct KNNResult {
    using Ptr = std::shared_ptr<KNNResult>;

    shared_vector_ptr<int32_t> indices = nullptr;
    shared_vector_ptr<float> distances = nullptr;
    size_t query_size;
    size_t k;
    KNNResult() : query_size(0), k(0) {}

    void allocate(const sycl_utils::DeviceQueue& queue, size_t query_size = 0, size_t k = 0) {
        this->query_size = query_size;
        this->k = k;
        this->indices = std::make_shared<shared_vector<int32_t>>(query_size * k, -1, *queue.ptr);
        this->distances =
            std::make_shared<shared_vector<float>>(query_size * k, std::numeric_limits<float>::max(), *queue.ptr);
    }
    void resize(size_t query_size = 0, size_t k = 0) {
        this->query_size = query_size;
        this->k = k;
        this->indices->resize(query_size * k);
        this->distances->resize(query_size * k);
    }
};

}  // namespace knn

}  // namespace algorithms

}  // namespace sycl_points
