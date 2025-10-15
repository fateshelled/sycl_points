#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

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

    struct Node {
        sycl::float3 min_bounds;
        sycl::float3 max_bounds;
        uint32_t start_index;
        uint32_t point_count;
        int32_t children[8];
        uint8_t is_leaf;
        uint8_t padding[3];
    };

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
    sycl::event build_structure_async();
    void finalize_structure();
    void build_octree();
    [[nodiscard]] static sycl::float3 axis_lengths(const sycl::float3& min_bounds, const sycl::float3& max_bounds);
    int32_t build_node_recursive(const sycl::float3& min_bounds, const sycl::float3& max_bounds, size_t start,
                                 size_t count, size_t depth, std::vector<Node>& host_nodes,
                                 std::vector<int32_t>& indices, std::vector<int32_t>& scratch) const;
    [[nodiscard]] static float distance_to_aabb(const sycl::float3& min_bounds, const sycl::float3& max_bounds,
                                                const sycl::float3& point);

    const sycl_utils::DeviceQueue* queue_;
    PointCloudShared::Ptr target_cloud_;
    float resolution_;
    size_t max_points_per_node_;
    sycl::float3 bbox_min_;
    sycl::float3 bbox_max_;
    shared_vector_ptr<sycl::float3> bbox_group_mins_ = nullptr;
    shared_vector_ptr<sycl::float3> bbox_group_maxs_ = nullptr;
    shared_vector<Node> nodes_;
    shared_vector<int32_t> point_indices_;
};

inline Octree::Octree(const sycl_utils::DeviceQueue& queue, float resolution, size_t max_points_per_node)
    : queue_(&queue),
      target_cloud_(nullptr),
      resolution_(resolution),
      max_points_per_node_(max_points_per_node),
      bbox_min_(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()),
      bbox_max_(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::lowest()),
      nodes_(shared_allocator<Node>(*queue.ptr)),
      point_indices_(shared_allocator<int32_t>(*queue.ptr)) {}

inline void Octree::copy_point_cloud(const PointCloudShared& points) {
    target_cloud_ = std::make_shared<PointCloudShared>(points);
}

inline sycl::event Octree::build_structure_async() {
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
        nodes_.clear();
        point_indices_.clear();
        return sycl::event{};
    }

    const size_t work_group_size = 128;
    const size_t group_count = (point_count + work_group_size - 1) / work_group_size;

    bbox_group_mins_ = std::make_shared<shared_vector<sycl::float3>>(group_count,
                                                                     sycl::float3(std::numeric_limits<float>::infinity(),
                                                                                  std::numeric_limits<float>::infinity(),
                                                                                  std::numeric_limits<float>::infinity()),
                                                                     *queue_->ptr);
    bbox_group_maxs_ = std::make_shared<shared_vector<sycl::float3>>(group_count,
                                                                     sycl::float3(std::numeric_limits<float>::lowest(),
                                                                                  std::numeric_limits<float>::lowest(),
                                                                                  std::numeric_limits<float>::lowest()),
                                                                     *queue_->ptr);

    auto points_ptr = target_cloud_->points->data();
    auto group_mins_ptr = bbox_group_mins_->data();
    auto group_maxs_ptr = bbox_group_maxs_->data();

    const size_t global_size = group_count * work_group_size;
    auto bbox_event = queue_->ptr->submit([=](sycl::handler& handler) {
        sycl::local_accessor<sycl::float3, 1> local_mins(sycl::range<1>(work_group_size), handler);
        sycl::local_accessor<sycl::float3, 1> local_maxs(sycl::range<1>(work_group_size), handler);

        handler.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)),
            [=](sycl::nd_item<1> item) {
                const size_t local_id = item.get_local_linear_id();
                const size_t group_id = item.get_group_linear_id();
                const size_t global_range = item.get_global_range(0);
                sycl::float3 local_min(std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::infinity());
                sycl::float3 local_max(std::numeric_limits<float>::lowest(),
                                       std::numeric_limits<float>::lowest(),
                                       std::numeric_limits<float>::lowest());

                for (size_t idx = item.get_global_linear_id(); idx < point_count; idx += global_range) {
                    const auto point = points_ptr[idx];
                    local_min.x() = sycl::min(local_min.x(), point.x());
                    local_min.y() = sycl::min(local_min.y(), point.y());
                    local_min.z() = sycl::min(local_min.z(), point.z());
                    local_max.x() = sycl::max(local_max.x(), point.x());
                    local_max.y() = sycl::max(local_max.y(), point.y());
                    local_max.z() = sycl::max(local_max.z(), point.z());
                }

                local_mins[local_id] = local_min;
                local_maxs[local_id] = local_max;
                item.barrier(sycl::access::fence_space::local_space);

                for (size_t stride = work_group_size / 2; stride > 0; stride >>= 1) {
                    if (local_id < stride) {
                        auto other_min = local_mins[local_id + stride];
                        auto other_max = local_maxs[local_id + stride];
                        auto current_min = local_mins[local_id];
                        auto current_max = local_maxs[local_id];
                        current_min.x() = sycl::min(current_min.x(), other_min.x());
                        current_min.y() = sycl::min(current_min.y(), other_min.y());
                        current_min.z() = sycl::min(current_min.z(), other_min.z());
                        current_max.x() = sycl::max(current_max.x(), other_max.x());
                        current_max.y() = sycl::max(current_max.y(), other_max.y());
                        current_max.z() = sycl::max(current_max.z(), other_max.z());
                        local_mins[local_id] = current_min;
                        local_maxs[local_id] = current_max;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (local_id == 0) {
                    group_mins_ptr[group_id] = local_mins[0];
                    group_maxs_ptr[group_id] = local_maxs[0];
                }
            });
    });

    return bbox_event;
}

inline void Octree::finalize_structure() {
    const size_t point_count = size();

    if (point_count == 0 || !bbox_group_mins_ || !bbox_group_maxs_) {
        bbox_group_mins_.reset();
        bbox_group_maxs_.reset();
        if (point_count == 0) {
            nodes_.clear();
            point_indices_.clear();
        }
        return;
    }

    sycl::float3 min_bounds(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity());
    sycl::float3 max_bounds(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                            std::numeric_limits<float>::lowest());

    for (const auto& partial_min : *bbox_group_mins_) {
        min_bounds.x() = std::min(min_bounds.x(), partial_min.x());
        min_bounds.y() = std::min(min_bounds.y(), partial_min.y());
        min_bounds.z() = std::min(min_bounds.z(), partial_min.z());
    }

    for (const auto& partial_max : *bbox_group_maxs_) {
        max_bounds.x() = std::max(max_bounds.x(), partial_max.x());
        max_bounds.y() = std::max(max_bounds.y(), partial_max.y());
        max_bounds.z() = std::max(max_bounds.z(), partial_max.z());
    }

    bbox_min_ = min_bounds;
    bbox_max_ = max_bounds;

    bbox_group_mins_.reset();
    bbox_group_maxs_.reset();

    build_octree();
}

inline sycl::float3 Octree::axis_lengths(const sycl::float3& min_bounds, const sycl::float3& max_bounds) {
    return sycl::float3(max_bounds.x() - min_bounds.x(), max_bounds.y() - min_bounds.y(),
                        max_bounds.z() - min_bounds.z());
}

inline void Octree::build_octree() {
    const size_t point_count = size();
    nodes_.clear();
    point_indices_.clear();

    if (point_count == 0) {
        return;
    }

    std::vector<Node> host_nodes;
    host_nodes.reserve(point_count);

    std::vector<int32_t> indices(point_count);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<int32_t> scratch(point_count);

    build_node_recursive(bbox_min_, bbox_max_, 0, point_count, 0, host_nodes, indices, scratch);

    shared_allocator<Node> node_alloc(*queue_->ptr);
    nodes_ = shared_vector<Node>(host_nodes.size(), Node{}, node_alloc);
    std::copy(host_nodes.begin(), host_nodes.end(), nodes_.begin());

    shared_allocator<int32_t> index_alloc(*queue_->ptr);
    point_indices_ = shared_vector<int32_t>(indices.size(), 0, index_alloc);
    std::copy(indices.begin(), indices.end(), point_indices_.begin());
}

inline int32_t Octree::build_node_recursive(const sycl::float3& min_bounds, const sycl::float3& max_bounds,
                                            size_t start, size_t count, size_t depth, std::vector<Node>& host_nodes,
                                            std::vector<int32_t>& indices, std::vector<int32_t>& scratch) const {
    Node node{};
    node.min_bounds = min_bounds;
    node.max_bounds = max_bounds;
    node.start_index = static_cast<uint32_t>(start);
    node.point_count = static_cast<uint32_t>(count);
    node.is_leaf = 1;
    std::fill(std::begin(node.children), std::end(node.children), -1);

    const int32_t node_index = static_cast<int32_t>(host_nodes.size());
    host_nodes.push_back(node);

    if (count == 0) {
        return node_index;
    }

    const auto lengths = axis_lengths(min_bounds, max_bounds);
    const float max_axis = std::max({lengths.x(), lengths.y(), lengths.z()});
    if (count <= max_points_per_node_ || max_axis <= resolution_ || depth >= 32) {
        return node_index;
    }

    const sycl::float3 center((min_bounds.x() + max_bounds.x()) * 0.5f, (min_bounds.y() + max_bounds.y()) * 0.5f,
                              (min_bounds.z() + max_bounds.z()) * 0.5f);

    std::array<size_t, 8> counts{};
    const auto* points_ptr = target_cloud_->points->data();
    for (size_t i = 0; i < count; ++i) {
        const auto point = points_ptr[indices[start + i]];
        int octant = 0;
        if (point.x() >= center.x()) {
            octant |= 1;
        }
        if (point.y() >= center.y()) {
            octant |= 2;
        }
        if (point.z() >= center.z()) {
            octant |= 4;
        }
        counts[octant]++;
    }

    size_t total = 0;
    std::array<size_t, 8> offsets{};
    for (size_t i = 0; i < counts.size(); ++i) {
        offsets[i] = total;
        total += counts[i];
    }

    auto write_offsets = offsets;
    for (size_t i = 0; i < count; ++i) {
        const int32_t point_index = indices[start + i];
        const auto point = points_ptr[point_index];
        int octant = 0;
        if (point.x() >= center.x()) {
            octant |= 1;
        }
        if (point.y() >= center.y()) {
            octant |= 2;
        }
        if (point.z() >= center.z()) {
            octant |= 4;
        }
        scratch[start + write_offsets[octant]++] = point_index;
    }

    for (size_t i = 0; i < count; ++i) {
        indices[start + i] = scratch[start + i];
    }

    host_nodes[node_index].is_leaf = 0;

    for (size_t child = 0; child < counts.size(); ++child) {
        const size_t child_count = counts[child];
        if (child_count == 0) {
            continue;
        }

        const size_t child_start = start + offsets[child];
        sycl::float3 child_min = min_bounds;
        sycl::float3 child_max = max_bounds;

        child_min.x() = (child & 1) ? center.x() : min_bounds.x();
        child_max.x() = (child & 1) ? max_bounds.x() : center.x();
        child_min.y() = (child & 2) ? center.y() : min_bounds.y();
        child_max.y() = (child & 2) ? max_bounds.y() : center.y();
        child_min.z() = (child & 4) ? center.z() : min_bounds.z();
        child_max.z() = (child & 4) ? max_bounds.z() : center.z();

        const int32_t child_index = build_node_recursive(child_min, child_max, child_start, child_count, depth + 1,
                                                         host_nodes, indices, scratch);
        host_nodes[node_index].children[child] = child_index;
    }

    if (std::all_of(std::begin(host_nodes[node_index].children), std::end(host_nodes[node_index].children),
                    [](int32_t idx) { return idx < 0; })) {
        host_nodes[node_index].is_leaf = 1;
    }

    return node_index;
}

inline float Octree::distance_to_aabb(const sycl::float3& min_bounds, const sycl::float3& max_bounds,
                                      const sycl::float3& point) {
    const float dx = (point.x() < min_bounds.x()) ? (min_bounds.x() - point.x())
                    : (point.x() > max_bounds.x()) ? (point.x() - max_bounds.x())
                                                   : 0.0f;
    const float dy = (point.y() < min_bounds.y()) ? (min_bounds.y() - point.y())
                    : (point.y() > max_bounds.y()) ? (point.y() - max_bounds.y())
                                                   : 0.0f;
    const float dz = (point.z() < min_bounds.z()) ? (min_bounds.z() - point.z())
                    : (point.z() > max_bounds.z()) ? (point.z() - max_bounds.z())
                                                   : 0.0f;
    return dx * dx + dy * dy + dz * dz;
}

inline Octree::Ptr Octree::build(const sycl_utils::DeviceQueue& queue, const PointCloudShared& points, float resolution,
                                 size_t max_points_per_node) {
    auto tree = std::make_shared<Octree>(queue, resolution, max_points_per_node);
    tree->copy_point_cloud(points);
    auto event = tree->build_structure_async();
    event.wait();
    tree->finalize_structure();
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
    const auto nodes_ptr = nodes_.data();
    const auto ordered_indices_ptr = point_indices_.data();
    const size_t node_count = nodes_.size();

    if (target_size > 0 && (node_count == 0 || point_indices_.empty())) {
        throw std::runtime_error("Octree structure has not been initialised");
    }

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
            const sycl::float3 query_point_vec(query_point.x(), query_point.y(), query_point.z());

            auto* query_indices = indices_ptr + query_idx * k;
            auto* query_distances = distances_ptr + query_idx * k;

            for (size_t neighbour_idx = 0; neighbour_idx < k; ++neighbour_idx) {
                query_indices[neighbour_idx] = -1;
                query_distances[neighbour_idx] = std::numeric_limits<float>::infinity();
            }

            constexpr size_t kMaxStackSize = 512;
            int32_t node_stack[kMaxStackSize];
            float node_stack_distance[kMaxStackSize];
            size_t stack_size = 0;
            size_t neighbour_count = 0;

            auto heap_swap = [&](size_t a, size_t b) {
                const float dist_tmp = query_distances[a];
                const int32_t idx_tmp = query_indices[a];
                query_distances[a] = query_distances[b];
                query_indices[a] = query_indices[b];
                query_distances[b] = dist_tmp;
                query_indices[b] = idx_tmp;
            };

            auto sift_up = [&](size_t idx) {
                while (idx > 0) {
                    const size_t parent = (idx - 1) / 2;
                    if (query_distances[parent] >= query_distances[idx]) {
                        break;
                    }
                    heap_swap(parent, idx);
                    idx = parent;
                }
            };

            auto sift_down = [&](size_t idx, size_t heap_size) {
                while (true) {
                    const size_t left = idx * 2 + 1;
                    if (left >= heap_size) {
                        break;
                    }
                    size_t largest = left;
                    const size_t right = left + 1;
                    if (right < heap_size && query_distances[right] > query_distances[largest]) {
                        largest = right;
                    }
                    if (query_distances[idx] >= query_distances[largest]) {
                        break;
                    }
                    heap_swap(idx, largest);
                    idx = largest;
                }
            };

            auto current_worst = [&]() {
                return neighbour_count < k ? std::numeric_limits<float>::infinity() : query_distances[0];
            };

            auto push_candidate = [&](float distance_sq, int32_t index) {
                if (neighbour_count < k) {
                    query_distances[neighbour_count] = distance_sq;
                    query_indices[neighbour_count] = index;
                    ++neighbour_count;
                    sift_up(neighbour_count - 1);
                } else if (distance_sq < query_distances[0]) {
                    query_distances[0] = distance_sq;
                    query_indices[0] = index;
                    sift_down(0, neighbour_count);
                }
            };

            auto push_node = [&](int32_t node_idx, float distance_sq) {
                if (stack_size < kMaxStackSize) {
                    node_stack[stack_size] = node_idx;
                    node_stack_distance[stack_size] = distance_sq;
                    ++stack_size;
                } else {
                    size_t worst_pos = 0;
                    float worst_dist = node_stack_distance[0];
                    for (size_t i = 1; i < stack_size; ++i) {
                        if (node_stack_distance[i] > worst_dist) {
                            worst_dist = node_stack_distance[i];
                            worst_pos = i;
                        }
                    }
                    if (distance_sq < worst_dist) {
                        node_stack[worst_pos] = node_idx;
                        node_stack_distance[worst_pos] = distance_sq;
                    }
                }
            };

            if (node_count == 0) {
                return;
            }

            push_node(0, distance_to_aabb(nodes_ptr[0].min_bounds, nodes_ptr[0].max_bounds, query_point_vec));

            while (stack_size > 0) {
                size_t best_pos = 0;
                float best_dist = node_stack_distance[0];
                for (size_t i = 1; i < stack_size; ++i) {
                    if (node_stack_distance[i] < best_dist) {
                        best_dist = node_stack_distance[i];
                        best_pos = i;
                    }
                }

                const int32_t current_node_idx = node_stack[best_pos];
                --stack_size;
                node_stack[best_pos] = node_stack[stack_size];
                node_stack_distance[best_pos] = node_stack_distance[stack_size];

                if (best_dist > current_worst()) {
                    continue;
                }

                const Node node = nodes_ptr[current_node_idx];
                if (node.is_leaf) {
                    for (size_t i = 0; i < node.point_count; ++i) {
                        const int32_t point_index = ordered_indices_ptr[node.start_index + i];
                        const auto target_point = target_points_ptr[point_index];
                        const float dx = query_point.x() - target_point.x();
                        const float dy = query_point.y() - target_point.y();
                        const float dz = query_point.z() - target_point.z();
                        const float dist_sq = dx * dx + dy * dy + dz * dz;
                        push_candidate(dist_sq, point_index);
                    }
                } else {
                    for (size_t child = 0; child < 8; ++child) {
                        const int32_t child_idx = node.children[child];
                        if (child_idx < 0) {
                            continue;
                        }
                        const float child_dist =
                            distance_to_aabb(nodes_ptr[child_idx].min_bounds, nodes_ptr[child_idx].max_bounds,
                                             query_point_vec);
                        if (child_dist <= current_worst()) {
                            push_node(child_idx, child_dist);
                        }
                    }
                }
            }

            size_t heap_size = neighbour_count;
            while (heap_size > 1) {
                const size_t last = heap_size - 1;
                heap_swap(0, last);
                --heap_size;
                sift_down(0, heap_size);
            }
        });
    });
    event.wait();

    return result;
}

}  // namespace octree

}  // namespace algorithms

}  // namespace sycl_points

