#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <sycl_points/algorithms/knn/knn.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace knn {

/// @brief kNN search by brute force
/// @param queue SYCL queue
/// @param queries query points
/// @param targets target points
/// @param k number of search nearrest neightbor
/// @return knn search result
inline KNNResult knn_search_bruteforce(const sycl_utils::DeviceQueue& queue, const PointCloudShared& queries,
                                       const PointCloudShared& targets, const size_t k) {
    if (k == 0) {
        throw std::runtime_error("`k` must be at least 1.");
    }

    if (k > MAX_SUPPORTED_K) {
        throw std::runtime_error("`k` must not exceed " + std::to_string(MAX_SUPPORTED_K) + ".");
    }

    constexpr size_t MAX_K = MAX_SUPPORTED_K;

    const size_t n = targets.points->size();  // Number of dataset points
    const size_t q = queries.points->size();  // Number of query points

    // Initialize result structure
    KNNResult result;
    result.allocate(queue, q, k);

    const size_t work_group_size = queue.get_work_group_size();
    const size_t global_size = queue.get_global_size(q);

    // memory ptr
    auto targets_ptr = (*targets.points).data();
    auto queries_ptr = (*queries.points).data();

    auto* distance_ptr = result.distances->data();
    auto* index_ptr = result.indices->data();

    // KNN search kernel BruteForce
    auto event = queue.ptr->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t queryIdx = item.get_global_id(0);
            if (queryIdx >= q) return;
            const auto query = queries_ptr[queryIdx];

            // Arrays to store K nearest points
            // Private arrays sized by the supported maximum.
            float kDistances[MAX_K];
            int32_t kIndices[MAX_K];

            // Initialize
            for (size_t i = 0; i < k; ++i) {
                kDistances[i] = std::numeric_limits<float>::max();
                kIndices[i] = -1;
            }

            // Calculate distances to all dataset points
            for (size_t j = 0; j < n; ++j) {
                // Calculate 3D distance
                const auto target = targets_ptr[j];
                const sycl::float4 diff = {query.x() - target.x(), query.y() - target.y(), query.z() - target.z(),
                                           0.0f};
                const float dist = sycl::dot(diff, diff);

                // Check if this point should be included in K nearest
                if (dist < kDistances[k - 1]) {
                    // Find insertion position
                    int32_t insertPos = k - 1;
                    while (insertPos > 0 && dist < kDistances[insertPos - 1]) {
                        kDistances[insertPos] = kDistances[insertPos - 1];
                        kIndices[insertPos] = kIndices[insertPos - 1];
                        --insertPos;
                    }

                    // Insert new point
                    kDistances[insertPos] = dist;
                    kIndices[insertPos] = j;
                }
            }

            // Write results to global memory
            for (size_t i = 0; i < k; i++) {
                distance_ptr[queryIdx * k + i] = kDistances[i];
                index_ptr[queryIdx * k + i] = kIndices[i];
            }
        });
    });
    event.wait();

    return result;
}

}  // namespace knn_search

}  // namespace algorithms

}  // namespace sycl_points
