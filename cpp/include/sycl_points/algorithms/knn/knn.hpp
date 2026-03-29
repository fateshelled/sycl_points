#pragma once

#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>

#include "sycl_points/algorithms/common/filter_by_flags.hpp"
#include "sycl_points/algorithms/knn/result.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace knn {

/// @brief Nearest neighbor search base class
class NearestNeighborBase {
public:
    virtual ~NearestNeighborBase() = default;

    /// @brief async nearest neighbor search
    /// @param queries query points
    /// @param result Search result
    /// @param depends depends sycl events
    /// @param transT transform matrix applied to query points
    /// @return nearest neighbor search event
    virtual sycl_utils::events nearest_neighbor_search_async(
        const PointCloudShared& queries, KNNResult& result,
        const std::vector<sycl::event>& depends = std::vector<sycl::event>(),
        const TransformMatrix& transT = TransformMatrix::Identity()) const = 0;

    /// @brief nearest neighbor search
    /// @param queries query points
    /// @param result Search result
    /// @param depends depends sycl events
    /// @param transT transform matrix applied to query points
    void nearest_neighbor_search(const PointCloudShared& queries, KNNResult& result,
                                 const std::vector<sycl::event>& depends = std::vector<sycl::event>(),
                                 const TransformMatrix& transT = TransformMatrix::Identity()) const {
        nearest_neighbor_search_async(queries, result, depends, transT).wait_and_throw();
    }
};

/// @brief KNN search base class
class KNNBase : public NearestNeighborBase {
public:
    /// @brief async kNN search
    /// @param queries query points
    /// @param k number of nearest neighbors to search
    /// @param result Search result
    /// @param depends depends sycl events
    /// @param transT transform matrix applied to query points
    /// @return knn search event
    virtual sycl_utils::events knn_search_async(const PointCloudShared& queries, const size_t k, KNNResult& result,
                                                const std::vector<sycl::event>& depends = std::vector<sycl::event>(),
                                                const TransformMatrix& transT = TransformMatrix::Identity()) const = 0;

    /// @brief kNN search
    /// @param queries query points
    /// @param k number of nearest neighbors to search
    /// @param depends depends sycl events
    /// @param transT transform matrix applied to query points
    /// @return knn search result
    KNNResult knn_search(const PointCloudShared& queries, const size_t k,
                         const std::vector<sycl::event>& depends = std::vector<sycl::event>(),
                         const TransformMatrix& transT = TransformMatrix::Identity()) const {
        KNNResult result;
        knn_search_async(queries, k, result, depends, transT).wait_and_throw();
        return result;
    }

    /// @brief async nearest neighbor search (delegates to knn_search_async with k=1)
    sycl_utils::events nearest_neighbor_search_async(
        const PointCloudShared& queries, KNNResult& result,
        const std::vector<sycl::event>& depends = std::vector<sycl::event>(),
        const TransformMatrix& transT = TransformMatrix::Identity()) const override {
        return knn_search_async(queries, 1, result, depends, transT);
    }
};

}  // namespace knn

}  // namespace algorithms

}  // namespace sycl_points
