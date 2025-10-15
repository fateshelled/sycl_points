#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include <sycl_points/algorithms/knn_search.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace octree {

/// @brief Result container for radius or k-nearest neighbor queries executed on the Octree.
/// @details The interface mirrors the KNNResult structure so that the callers can reuse
///          existing post-processing utilities. Only the allocation routine is provided here;
///          the actual search logic will be implemented in a later change.
using KNNResult = knn_search::KNNResult;

/// @brief Octree data structure that will support parallel construction and neighbour search on SYCL devices.
class Octree {
public:
    using Ptr = std::shared_ptr<Octree>;

    /// @brief Construct an empty Octree instance that is ready to be built.
    /// @param queue Device queue used for all SYCL operations.
    /// @param resolution Leaf node resolution expressed in metres.
    /// @param max_points_per_node Maximum number of points per leaf node before subdivision.
    Octree(const sycl_utils::DeviceQueue& queue, float resolution, size_t max_points_per_node);

    /// @brief Build an Octree from the given point cloud on the specified queue.
    /// @param queue Device queue used for the construction kernels.
    /// @param points Input point cloud.
    /// @param resolution Leaf node resolution expressed in metres.
    /// @param max_points_per_node Maximum number of points per leaf node before subdivision.
    /// @return Shared pointer to the constructed Octree.
    static Ptr build(const sycl_utils::DeviceQueue& queue, const PointCloudShared& points, float resolution,
                     size_t max_points_per_node = 32);

    /// @brief Execute a k-nearest neighbour query for the supplied queries.
    /// @param queries Query point cloud.
    /// @param k Number of neighbours to gather.
    /// @return Result container that stores neighbour indices and squared distances.
    [[nodiscard]] KNNResult knn_search(const PointCloudShared& queries, size_t k) const;

    /// @brief Accessor for the resolution that was requested at build time.
    [[nodiscard]] float resolution() const { return resolution_; }

    /// @brief Accessor for the maximum number of points per node.
    [[nodiscard]] size_t max_points_per_node() const { return max_points_per_node_; }

    /// @brief Number of points stored in the Octree.
    [[nodiscard]] size_t size() const { return target_cloud_ ? target_cloud_->size() : 0; }

private:
    void copy_point_cloud(const PointCloudShared& points);
    void build_structure_async() const;

    const sycl_utils::DeviceQueue* queue_;
    PointCloudShared::Ptr target_cloud_;
    float resolution_;
    size_t max_points_per_node_;
};

inline Octree::Octree(const sycl_utils::DeviceQueue& queue, float resolution, size_t max_points_per_node)
    : queue_(&queue), target_cloud_(nullptr), resolution_(resolution), max_points_per_node_(max_points_per_node) {}

inline void Octree::copy_point_cloud(const PointCloudShared& points) {
    target_cloud_ = std::make_shared<PointCloudShared>(points);
}

inline void Octree::build_structure_async() const {
    const size_t point_count = size();
    if (queue_ == nullptr || !queue_->ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }
    const size_t global_size = std::max<size_t>(point_count, 1);
    auto event = queue_->ptr->submit([=](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(global_size), [=](sycl::id<1> idx) {
            (void)idx;
            // TODO: Build octree nodes here.
        });
    });
    event.wait();
}

inline Octree::Ptr Octree::build(const sycl_utils::DeviceQueue& queue, const PointCloudShared& points, float resolution,
                                 size_t max_points_per_node) {
    auto tree = std::make_shared<Octree>(queue, resolution, max_points_per_node);
    tree->copy_point_cloud(points);
    tree->build_structure_async();
    return tree;
}

inline KNNResult Octree::knn_search(const PointCloudShared& queries, size_t k) const {
    if (queue_ == nullptr || !queue_->ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }

    KNNResult result;
    const size_t query_size = queries.points->size();
    result.allocate(*queue_, query_size, k);

    auto indices_ptr = result.indices->data();
    auto distances_ptr = result.distances->data();

    const size_t global_size = std::max<size_t>(query_size, 1);
    auto event = queue_->ptr->submit([=](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(global_size), [=](sycl::id<1> idx) {
            const size_t query_idx = idx[0];
            if (query_idx >= query_size) {
                return;
            }
            for (size_t neighbour_idx = 0; neighbour_idx < k; ++neighbour_idx) {
                const size_t offset = query_idx * k + neighbour_idx;
                indices_ptr[offset] = -1;
                distances_ptr[offset] = std::numeric_limits<float>::infinity();
            }
        });
    });
    event.wait();

    return result;
}

}  // namespace octree

}  // namespace algorithms

}  // namespace sycl_points

