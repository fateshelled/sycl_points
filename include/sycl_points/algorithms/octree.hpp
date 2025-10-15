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
    void build_structure_async();

    const sycl_utils::DeviceQueue* queue_;
    PointCloudShared::Ptr target_cloud_;
    float resolution_;
    size_t max_points_per_node_;
    sycl::float3 bbox_min_;
    sycl::float3 bbox_max_;
};

inline Octree::Octree(const sycl_utils::DeviceQueue& queue, float resolution, size_t max_points_per_node)
    : queue_(&queue),
      target_cloud_(nullptr),
      resolution_(resolution),
      max_points_per_node_(max_points_per_node),
      bbox_min_(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()),
      bbox_max_(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::lowest()) {}

inline void Octree::copy_point_cloud(const PointCloudShared& points) {
    target_cloud_ = std::make_shared<PointCloudShared>(points);
}

inline void Octree::build_structure_async() {
    const size_t point_count = size();
    if (queue_ == nullptr || !queue_->ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }
    if (!target_cloud_ || !target_cloud_->points) {
        throw std::runtime_error("Octree target cloud is not initialised");
    }

    if (point_count == 0) {
        bbox_min_ = sycl::float3(0.0f, 0.0f, 0.0f);
        bbox_max_ = sycl::float3(0.0f, 0.0f, 0.0f);
        return;
    }

    auto points_ptr = target_cloud_->points->data();

    shared_allocator<float> alloc(*queue_->ptr);
    shared_vector<float> min_values(3, std::numeric_limits<float>::infinity(), alloc);
    shared_vector<float> max_values(3, std::numeric_limits<float>::lowest(), alloc);

    const size_t global_size = point_count;
    auto bbox_event = queue_->ptr->submit([=, &min_values, &max_values](sycl::handler& handler) {
        auto min_ptr = min_values.data();
        auto max_ptr = max_values.data();
        handler.parallel_for(sycl::range<1>(global_size), [=](sycl::id<1> idx) {
            const size_t point_idx = idx[0];
            if (point_idx >= point_count) {
                return;
            }
            const auto point = points_ptr[point_idx];
            for (size_t axis = 0; axis < 3; ++axis) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    min_atomic(min_ptr[axis]);
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    max_atomic(max_ptr[axis]);
                min_atomic.fetch_min(point[axis]);
                max_atomic.fetch_max(point[axis]);
            }
        });
    });
    bbox_event.wait();

    bbox_min_ = sycl::float3(min_values[0], min_values[1], min_values[2]);
    bbox_max_ = sycl::float3(max_values[0], max_values[1], max_values[2]);
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
    if (target_cloud_ == nullptr || target_cloud_->points == nullptr) {
        throw std::runtime_error("Octree target cloud is not initialised");
    }

    KNNResult result;
    const size_t query_size = queries.points->size();
    result.allocate(*queue_, query_size, k);

    auto indices_ptr = result.indices->data();
    auto distances_ptr = result.distances->data();
    const auto target_points_ptr = target_cloud_->points->data();
    const auto query_points_ptr = queries.points->data();
    const size_t target_size = target_cloud_->points->size();

    const size_t global_size = std::max<size_t>(query_size, 1);
    auto event = queue_->ptr->submit([=](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(global_size), [=](sycl::id<1> idx) {
            const size_t query_idx = idx[0];
            if (query_idx >= query_size) {
                return;
            }
            if (target_size == 0 || k == 0) {
                return;
            }

            const auto query_point = query_points_ptr[query_idx];

            for (size_t neighbour_idx = 0; neighbour_idx < k; ++neighbour_idx) {
                const size_t offset = query_idx * k + neighbour_idx;
                indices_ptr[offset] = -1;
                distances_ptr[offset] = std::numeric_limits<float>::infinity();
            }

            for (size_t target_idx = 0; target_idx < target_size; ++target_idx) {
                const auto target_point = target_points_ptr[target_idx];
                const float dx = query_point.x() - target_point.x();
                const float dy = query_point.y() - target_point.y();
                const float dz = query_point.z() - target_point.z();
                const float dist_sq = dx * dx + dy * dy + dz * dz;

                size_t worst_idx = 0;
                float worst_dist = distances_ptr[query_idx * k];
                for (size_t neighbour_idx = 1; neighbour_idx < k; ++neighbour_idx) {
                    const size_t offset = query_idx * k + neighbour_idx;
                    const float current_dist = distances_ptr[offset];
                    if (current_dist > worst_dist) {
                        worst_dist = current_dist;
                        worst_idx = neighbour_idx;
                    }
                }

                if (dist_sq < worst_dist) {
                    const size_t offset = query_idx * k + worst_idx;
                    distances_ptr[offset] = dist_sq;
                    indices_ptr[offset] = static_cast<int32_t>(target_idx);
                }
            }

            for (size_t i = 1; i < k; ++i) {
                const size_t current_offset = query_idx * k + i;
                const float current_dist = distances_ptr[current_offset];
                const int32_t current_index = indices_ptr[current_offset];
                size_t j = i;
                while (j > 0 && distances_ptr[query_idx * k + (j - 1)] > current_dist) {
                    distances_ptr[query_idx * k + j] = distances_ptr[query_idx * k + (j - 1)];
                    indices_ptr[query_idx * k + j] = indices_ptr[query_idx * k + (j - 1)];
                    --j;
                }
                distances_ptr[query_idx * k + j] = current_dist;
                indices_ptr[query_idx * k + j] = current_index;
            }
        });
    });
    event.wait();

    return result;
}

}  // namespace octree

}  // namespace algorithms

}  // namespace sycl_points

