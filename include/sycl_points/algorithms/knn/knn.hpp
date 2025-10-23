#pragma once

#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sycl_points/algorithms/common/filter_by_flags.hpp>
#include <sycl_points/algorithms/knn/result.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace knn {

/// @brief Maximum supported number of neighbors for kNN queries.
inline constexpr size_t MAX_SUPPORTED_K = 100;

/// @brief KNN search base class
class KNNBase {
public:
    /// @brief async kNN search
    /// @param queries query points
    /// @param k number of search nearrest neightbor
    /// @param result Search result
    /// @param depends depends sycl events
    /// @return knn search event
    virtual sycl_utils::events knn_search_async(
        const PointCloudShared& queries, const size_t k, KNNResult& result,
        const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const = 0;

    /// @brief kNN search
    /// @param queries query points
    /// @param k number of search nearrest neightbor
    /// @param depends depends sycl events
    /// @return knn search result
    KNNResult knn_search(const PointCloudShared& queries, const size_t k,
                         const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const {
        KNNResult result;
        knn_search_async(queries, k, result, depends).wait();
        return result;
    }

    /// @brief async nearest neighbor search
    /// @param queries query points
    /// @param result Search result
    /// @param depends depends sycl events
    /// @return knn search event
    sycl_utils::events nearest_neighbor_search_async(
        const PointCloudShared& queries, KNNResult& result,
        const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const {
        return knn_search_async(queries, 1, result, depends);
    }

    /// @brief nearest neighbor search
    /// @param queries query points
    /// @param result Search result
    /// @param depends depends sycl events
    /// @return knn search event
    void nearest_neighbor_search(const PointCloudShared& queries, KNNResult& result,
                                 const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const {
        nearest_neighbor_search_async(queries, result, depends).wait();
    }
};

}  // namespace knn

}  // namespace algorithms

}  // namespace sycl_points
