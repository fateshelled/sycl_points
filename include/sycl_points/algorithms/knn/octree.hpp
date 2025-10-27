#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <sycl_points/algorithms/knn/knn.hpp>
#include <sycl_points/algorithms/knn/result.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace sycl_points {

namespace algorithms {

namespace knn {

// These helpers are declared inline to keep their definitions in the header ODR-safe across translation units.
/// @brief Compute the edge lengths of an axis-aligned bounding box.
SYCL_EXTERNAL inline sycl::float3 axis_lengths(const sycl::float3& min_bounds, const sycl::float3& max_bounds);
/// @brief Compute the squared distance from @p point to the provided bounding box.
SYCL_EXTERNAL inline float distance_to_aabb(const sycl::float3& min_bounds, const sycl::float3& max_bounds,
                                            const sycl::float3& point);
/// @brief Helper that avoids repeated Eigen boilerplate when computing squared distances.
inline float squared_distance(const PointType& a, const PointType& b);

/// @brief Octree data structure that will support parallel construction and neighbour search on SYCL devices.
class Octree : public KNNBase {
public:
    using Ptr = std::shared_ptr<Octree>;

    /// @brief Compact representation of a node stored on the device.
    struct Node {
        Eigen::Vector3f min_bounds;
        Eigen::Vector3f max_bounds;
        uint32_t is_leaf;
        // Union stores either the metadata for leaf nodes or the child indices for internal nodes.
        union {
            struct {
                uint32_t start_index;
                uint32_t point_count;
                uint32_t padding[6];
            } leaf;
            int32_t children[8];
        } data;
        uint32_t padding;
    };

    static_assert(std::is_standard_layout_v<Node>, "Octree::Node must remain standard-layout");

    static_assert(sizeof(Node) == 64, "Octree::Node must remain 64 bytes");

    /// @brief Axis-aligned bounding box helper used during host-side operations.
    struct BoundingBox {
        sycl::float3 min_bounds;
        sycl::float3 max_bounds;

        [[nodiscard]] bool contains(const sycl::float3& point) const {
            return point.x() >= min_bounds.x() && point.x() <= max_bounds.x() && point.y() >= min_bounds.y() &&
                   point.y() <= max_bounds.y() && point.z() >= min_bounds.z() && point.z() <= max_bounds.z();
        }

        [[nodiscard]] bool contains(const PointType& point) const {
            return contains(sycl::float3(point.x(), point.y(), point.z()));
        }

        [[nodiscard]] bool intersects(const BoundingBox& other) const {
            return !(other.min_bounds.x() > max_bounds.x() || other.max_bounds.x() < min_bounds.x() ||
                     other.min_bounds.y() > max_bounds.y() || other.max_bounds.y() < min_bounds.y() ||
                     other.min_bounds.z() > max_bounds.z() || other.max_bounds.z() < min_bounds.z());
        }

        [[nodiscard]] bool fully_contains(const BoundingBox& other) const {
            return other.min_bounds.x() >= min_bounds.x() && other.max_bounds.x() <= max_bounds.x() &&
                   other.min_bounds.y() >= min_bounds.y() && other.max_bounds.y() <= max_bounds.y() &&
                   other.min_bounds.z() >= min_bounds.z() && other.max_bounds.z() <= max_bounds.z();
        }
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
    /// @tparam MAX_DEPTH Maximum number of nodes kept on the traversal stack.
    /// @tparam MAX_K Maximum number of neighbours that can be requested.
    /// @return Result container that stores neighbour indices and squared distances.
    sycl_utils::events knn_search_async(
        const PointCloudShared& queries, const size_t k, KNNResult& result,
        const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const override;

    [[nodiscard]] KNNResult knn_search(const PointCloudShared& queries, size_t k) const {
        return KNNBase::knn_search(queries, k);
    }

    /// @brief Accessor for the resolution that was requested at build time.
    [[nodiscard]] float resolution() const { return this->resolution_; }

    /// @brief Accessor for the maximum number of points per node.
    [[nodiscard]] size_t max_points_per_node() const { return this->max_points_per_node_; }

    /// @brief Number of points stored in the Octree.
    [[nodiscard]] size_t size() const { return this->total_point_count_; }

    /// @brief Insert a new point into the tree.
    void insert(const PointType& point);

    /// @brief Remove the first point matching the provided coordinates within a tolerance.
    /// @return True when a point was removed.
    bool remove(const PointType& point, float tolerance = 1e-5f);

    /// @brief Delete all points inside the provided bounding box.
    /// @return Number of points removed.
    size_t delete_box(const BoundingBox& region);

    /// @brief Gather all point identifiers that lie within the radius from the query point.
    [[nodiscard]] std::vector<int32_t> radius_search(const PointType& query, float radius) const;

    /// @brief Retrieve a snapshot of the points stored in the tree in canonical order.
    [[nodiscard]] std::vector<PointType> snapshot_points() const;

    /// @brief Retrieve the original identifiers for the snapshot points in the same order.
    [[nodiscard]] std::vector<int32_t> snapshot_ids() const;

private:
    /// @brief Host-side storage for a point and its persistent identifier.
    struct PointRecord {
        PointType point;
        int32_t id;
    };

    /// @brief Host-side representation of a node used for incremental updates.
    struct HostNode {
        sycl::float3 min_bounds;
        sycl::float3 max_bounds;
        std::array<int32_t, 8> children{};
        bool is_leaf{true};
        std::vector<PointRecord> points;
        size_t subtree_size{0};
        size_t start_index{0};

        HostNode() { children.fill(-1); }
    };

    /// @brief Work item stored in traversal stacks during neighbour searches.
    /// @details Encodes the node index and the squared distance used to prioritise traversal.
    struct NodeEntry {
        int32_t nodeIdx;  // node index
        float dist_sq;    // squared distance to splitting plane
    };

    /// @brief Rebuild the tree from a dense point cloud.
    void build_from_cloud(const PointCloudShared& points);
    /// @brief Allocate a new host-side node and trigger subdivision if necessary.
    int32_t create_host_node(const sycl::float3& min_bounds, const sycl::float3& max_bounds,
                             std::vector<PointRecord>&& points, size_t depth);
    /// @brief Split the provided leaf node when it exceeds capacity.
    void subdivide_leaf(int32_t node_index, size_t depth);
    /// @brief Recursive insertion helper that descends the tree to place a point.
    void insert_recursive(int32_t node_index, const PointType& point, int32_t id, size_t depth);
    /// @brief Recursive removal helper used by the public remove method.
    bool remove_recursive(int32_t node_index, const PointType& point, float tolerance_sq);
    /// @brief Recursive deletion helper that prunes subtrees intersecting the region.
    bool delete_box_recursive(int32_t node_index, const BoundingBox& region);
    /// @brief Expand the root bounds until the provided point lies inside them.
    void ensure_root_bounds(const PointType& point);
    /// @brief Compute the bounding box of a specific child octant.
    BoundingBox child_bounds(const HostNode& node, size_t child_index) const;
    /// @brief Refresh the cached subtree size after structural changes.
    void recompute_subtree_size(int32_t node_index);
    /// @brief Synchronise host-side data into device-visible buffers.
    void sync_device_buffers();
    /// @brief Const-qualified overload that forwards to the non-const synchronisation path.
    void sync_device_buffers() const;
    sycl_utils::DeviceQueue queue_;
    float resolution_;
    size_t max_points_per_node_;
    sycl::float3 bbox_min_;
    sycl::float3 bbox_max_;
    int32_t root_index_;
    size_t total_point_count_;
    int32_t next_point_id_;
    mutable std::vector<HostNode> host_nodes_;
    mutable shared_vector<Node> nodes_;
    mutable shared_vector<PointType> device_points_;
    mutable shared_vector<int32_t> device_point_ids_;
    mutable std::vector<int32_t> snapshot_ids_;
    mutable bool device_dirty_;

    template <size_t MAX_K, size_t MAX_DEPTH>
    sycl_utils::events knn_search_async_impl(
        const PointCloudShared& queries, size_t k, KNNResult& result,
        const std::vector<sycl::event>& depends) const;
};

/// @brief Initialise the octree with the provided execution queue and parameters.
/// @param queue Device queue used for all SYCL operations.
/// @param resolution Leaf node resolution expressed in metres.
/// @param max_points_per_node Maximum number of points per leaf node before subdivision.
inline Octree::Octree(const sycl_utils::DeviceQueue& queue, float resolution, size_t max_points_per_node)
    : queue_(queue),
      resolution_(resolution),
      max_points_per_node_(max_points_per_node),
      bbox_min_(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()),
      bbox_max_(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::lowest()),
      root_index_(-1),
      total_point_count_(0),
      next_point_id_(0),
      host_nodes_(),
      nodes_(shared_allocator<Node>(*queue_.ptr)),
      device_points_(shared_allocator<PointType>(*queue_.ptr)),
      device_point_ids_(shared_allocator<int32_t>(*queue_.ptr)),
      device_dirty_(true) {}

/// @brief Populate the octree from an entire point cloud.
/// @details The method resets any previously stored data and rebuilds the hierarchy from scratch.
/// @param points Input point cloud.
inline void Octree::build_from_cloud(const PointCloudShared& points) {
    if (!this->queue_.ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }

    // Reset cached host/device structures before rebuilding the tree.
    this->host_nodes_.clear();
    this->total_point_count_ = 0;
    this->root_index_ = -1;
    this->device_dirty_ = true;
    if (!points.points) {
        throw std::runtime_error("Point cloud is not initialised");
    }

    const size_t point_count = points.points->size();
    if (point_count == 0) {
        this->bbox_min_ = sycl::float3(0.0f, 0.0f, 0.0f);
        this->bbox_max_ = sycl::float3(0.0f, 0.0f, 0.0f);
        this->nodes_.clear();
        this->device_points_.clear();
        this->device_point_ids_.clear();
        return;
    }

    this->bbox_min_ = sycl::float3(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::infinity());
    this->bbox_max_ = sycl::float3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                                   std::numeric_limits<float>::lowest());

    std::vector<PointRecord> records(point_count);
    for (size_t i = 0; i < point_count; ++i) {
        const auto point = (*points.points)[i];
        this->bbox_min_.x() = std::min(this->bbox_min_.x(), point.x());
        this->bbox_min_.y() = std::min(this->bbox_min_.y(), point.y());
        this->bbox_min_.z() = std::min(this->bbox_min_.z(), point.z());
        this->bbox_max_.x() = std::max(this->bbox_max_.x(), point.x());
        this->bbox_max_.y() = std::max(this->bbox_max_.y(), point.y());
        this->bbox_max_.z() = std::max(this->bbox_max_.z(), point.z());

        records[i] = PointRecord{point, static_cast<int32_t>(i)};
    }

    const float epsilon = std::max(1e-5f, this->resolution_ * 0.5f);
    this->bbox_min_ -= sycl::float3(epsilon, epsilon, epsilon);
    this->bbox_max_ += sycl::float3(epsilon, epsilon, epsilon);

    this->root_index_ = this->create_host_node(this->bbox_min_, this->bbox_max_, std::move(records), 0);
    this->total_point_count_ = point_count;
    this->next_point_id_ = static_cast<int32_t>(point_count);
}

/// @brief Create a host node and recursively subdivide it when necessary.
/// @param min_bounds Minimum coordinates of the node's bounding box.
/// @param max_bounds Maximum coordinates of the node's bounding box.
/// @param points Points contained within the node's bounding box.
/// @param depth Current depth of the node in the tree.
/// @return Index of the created node inside @c host_nodes_.
inline int32_t Octree::create_host_node(const sycl::float3& min_bounds, const sycl::float3& max_bounds,
                                        std::vector<PointRecord>&& points, size_t depth) {
    HostNode node;
    node.min_bounds = min_bounds;
    node.max_bounds = max_bounds;
    node.points = std::move(points);
    node.subtree_size = node.points.size();

    const int32_t node_index = static_cast<int32_t>(this->host_nodes_.size());
    this->host_nodes_.push_back(std::move(node));

    this->subdivide_leaf(node_index, depth);
    return node_index;
}

/// @brief Split a leaf node into up to eight child nodes when it exceeds capacity.
/// @param node_index Index of the leaf node that may be subdivided.
/// @param depth Depth of the node to guard against excessive recursion.
inline void Octree::subdivide_leaf(int32_t node_index, size_t depth) {
    HostNode& node = this->host_nodes_[static_cast<size_t>(node_index)];
    if (!node.is_leaf) {
        return;
    }

    const auto lengths = axis_lengths(node.min_bounds, node.max_bounds);
    const float max_axis = std::max({lengths.x(), lengths.y(), lengths.z()});
    if (node.points.size() <= this->max_points_per_node_ || max_axis <= this->resolution_ || depth >= 32) {
        node.subtree_size = node.points.size();
        return;
    }

    const sycl::float3 min_bounds = node.min_bounds;
    const sycl::float3 max_bounds = node.max_bounds;
    const sycl::float3 center = 0.5f * (min_bounds + max_bounds);

    std::vector<PointRecord> points = std::move(node.points);
    std::array<std::vector<PointRecord>, 8> child_points;
    for (const auto& record : points) {
        int octant = 0;
        if (record.point.x() >= center.x()) {
            octant |= 1;
        }
        if (record.point.y() >= center.y()) {
            octant |= 2;
        }
        if (record.point.z() >= center.z()) {
            octant |= 4;
        }
        child_points[static_cast<size_t>(octant)].push_back(record);
    }

    node.is_leaf = false;
    node.subtree_size = 0;
    node.children.fill(-1);

    for (size_t child = 0; child < child_points.size(); ++child) {
        if (child_points[child].empty()) {
            continue;
        }

        BoundingBox bounds{};
        bounds.min_bounds = min_bounds;
        bounds.max_bounds = max_bounds;
        bounds.min_bounds.x() = (child & 1) ? center.x() : min_bounds.x();
        bounds.max_bounds.x() = (child & 1) ? max_bounds.x() : center.x();
        bounds.min_bounds.y() = (child & 2) ? center.y() : min_bounds.y();
        bounds.max_bounds.y() = (child & 2) ? max_bounds.y() : center.y();
        bounds.min_bounds.z() = (child & 4) ? center.z() : min_bounds.z();
        bounds.max_bounds.z() = (child & 4) ? max_bounds.z() : center.z();

        const int32_t child_index =
            this->create_host_node(bounds.min_bounds, bounds.max_bounds, std::move(child_points[child]), depth + 1);
        HostNode& updated_node = this->host_nodes_[static_cast<size_t>(node_index)];
        updated_node.children[child] = child_index;
        updated_node.subtree_size += this->host_nodes_[static_cast<size_t>(child_index)].subtree_size;
    }

    HostNode& updated_node = this->host_nodes_[static_cast<size_t>(node_index)];
    if (updated_node.subtree_size == 0) {
        updated_node.is_leaf = true;
        updated_node.children.fill(-1);
    }
}

/// @brief Insert a single point into the octree.
/// @param point The point to insert.
inline void Octree::insert(const PointType& point) {
    this->ensure_root_bounds(point);
    const int32_t id = this->next_point_id_++;
    this->insert_recursive(this->root_index_, point, id, 0);
    this->total_point_count_ += 1;
    this->device_dirty_ = true;
}

/// @brief Internal helper that performs the recursive insert traversal.
/// @param node_index Index of the current node.
/// @param point Point to be inserted.
/// @param id Stable identifier assigned to the point.
/// @param depth Current depth of the traversal for termination checks.
inline void Octree::insert_recursive(int32_t node_index, const PointType& point, int32_t id, size_t depth) {
    HostNode& node = this->host_nodes_[static_cast<size_t>(node_index)];
    if (node.is_leaf) {
        node.points.push_back(PointRecord{point, id});
        node.subtree_size = node.points.size();
        this->subdivide_leaf(node_index, depth);
        return;
    }

    int octant = 0;
    {
        const auto bounds = BoundingBox{node.min_bounds, node.max_bounds};
        const sycl::float3 center = 0.5f * (bounds.min_bounds + bounds.max_bounds);

        if (point.x() >= center.x()) {
            octant |= 1;
        }
        if (point.y() >= center.y()) {
            octant |= 2;
        }
        if (point.z() >= center.z()) {
            octant |= 4;
        }
    }

    const size_t child_index = static_cast<size_t>(octant);
    if (child_index >= node.children.size()) {
        throw std::out_of_range("Octree child index out of bounds");
    }
    const int32_t existing_child = node.children[child_index];
    if (existing_child < 0) {
        std::vector<PointRecord> new_points;
        new_points.push_back(PointRecord{point, id});
        const auto child_bounds_value = this->child_bounds(node, child_index);
        const int32_t new_child = this->create_host_node(child_bounds_value.min_bounds, child_bounds_value.max_bounds,
                                                         std::move(new_points), depth + 1);
        HostNode& refreshed_node = this->host_nodes_[static_cast<size_t>(node_index)];
        refreshed_node.children[child_index] = new_child;
    } else {
        this->insert_recursive(existing_child, point, id, depth + 1);
    }

    this->recompute_subtree_size(node_index);
}

/// @brief Remove the first point matching the query within the specified tolerance.
/// @param point Position of the target point.
/// @param tolerance Euclidean distance tolerance used for comparisons.
/// @return True when a matching point was found and removed.
inline bool Octree::remove(const PointType& point, float tolerance) {
    if (this->root_index_ < 0) {
        return false;
    }
    const float tolerance_sq = tolerance * tolerance;
    const bool removed = this->remove_recursive(this->root_index_, point, tolerance_sq);
    if (removed) {
        this->total_point_count_ -= 1;
        this->device_dirty_ = true;
    }
    return removed;
}

/// @brief Recursive removal routine that prunes empty nodes on the way back up.
/// @param node_index Index of the current node being inspected.
/// @param point Point to search for.
/// @param tolerance_sq Squared tolerance for comparisons to avoid repeated sqrt operations.
/// @return True when a point was erased in the subtree.
inline bool Octree::remove_recursive(int32_t node_index, const PointType& point, float tolerance_sq) {
    HostNode& node = this->host_nodes_[static_cast<size_t>(node_index)];
    bool removed = false;
    if (node.is_leaf) {
        auto& pts = node.points;
        auto it = std::find_if(pts.begin(), pts.end(), [&](const PointRecord& record) {
            return squared_distance(record.point, point) <= tolerance_sq;
        });
        if (it != pts.end()) {
            pts.erase(it);
            node.subtree_size = pts.size();
            removed = true;
        }
    } else {
        for (size_t child = 0; child < node.children.size(); ++child) {
            const int32_t child_index = node.children[child];
            if (child_index < 0) {
                continue;
            }
            if (!this->host_nodes_[static_cast<size_t>(child_index)].subtree_size) {
                continue;
            }
            const auto& child_node = this->host_nodes_[static_cast<size_t>(child_index)];
            const BoundingBox bounds{child_node.min_bounds, child_node.max_bounds};
            const float aabb_distance =
                distance_to_aabb(bounds.min_bounds, bounds.max_bounds, sycl::float3(point.x(), point.y(), point.z()));
            if (aabb_distance > tolerance_sq) {
                continue;
            }
            if (this->remove_recursive(child_index, point, tolerance_sq)) {
                removed = true;
            }
            if (this->host_nodes_[static_cast<size_t>(child_index)].subtree_size == 0) {
                node.children[child] = -1;
            }
            if (removed) {
                break;
            }
        }
        this->recompute_subtree_size(node_index);
    }
    return removed;
}

/// @brief Remove every point that lies inside the provided axis-aligned bounding box.
/// @param region Bounding region that will be emptied.
/// @return Number of points that were erased.
inline size_t Octree::delete_box(const BoundingBox& region) {
    if (this->root_index_ < 0) {
        return 0;
    }
    const size_t before = this->total_point_count_;
    if (this->delete_box_recursive(this->root_index_, region)) {
        // Root cleared completely.
    }
    if (this->root_index_ >= 0) {
        this->host_nodes_[static_cast<size_t>(this->root_index_)].subtree_size = this->total_point_count_;
    }
    const size_t removed = before - this->total_point_count_;
    if (removed > 0) {
        this->device_dirty_ = true;
    }
    return removed;
}

/// @brief Recursive helper for @ref delete_box that supports partial overlap.
/// @param node_index Index of the node currently processed.
/// @param region Bounding box defining the delete region.
/// @return True when the subtree becomes empty and can be pruned.
inline bool Octree::delete_box_recursive(int32_t node_index, const BoundingBox& region) {
    HostNode& node = this->host_nodes_[static_cast<size_t>(node_index)];
    const BoundingBox node_bounds{node.min_bounds, node.max_bounds};
    if (!region.intersects(node_bounds)) {
        return false;
    }

    if (region.fully_contains(node_bounds)) {
        this->total_point_count_ -= node.subtree_size;
        node.points.clear();
        node.points.shrink_to_fit();
        node.children.fill(-1);
        node.is_leaf = true;
        node.subtree_size = 0;
        return true;
    }

    if (node.is_leaf) {
        auto& pts = node.points;
        auto it = std::remove_if(pts.begin(), pts.end(),
                                 [&](const PointRecord& record) { return region.contains(record.point); });
        if (it != pts.end()) {
            this->total_point_count_ -= static_cast<size_t>(std::distance(it, pts.end()));
            pts.erase(it, pts.end());
            node.subtree_size = pts.size();
        }
        return node.subtree_size == 0;
    }

    for (size_t child = 0; child < node.children.size(); ++child) {
        const int32_t child_idx = node.children[child];
        if (child_idx < 0) {
            continue;
        }
        if (this->delete_box_recursive(child_idx, region)) {
            node.children[child] = -1;
        }
    }

    this->recompute_subtree_size(node_index);
    return node.subtree_size == 0;
}

/// @brief Collect identifiers of every point within the specified radius around a query point.
/// @param query Query position expressed in the same coordinate system as the tree.
/// @param radius Search radius in metres.
/// @return Vector containing identifiers of matching points.
inline std::vector<int32_t> Octree::radius_search(const PointType& query, float radius) const {
    this->sync_device_buffers();
    if (this->root_index_ < 0 || this->total_point_count_ == 0) {
        return {};
    }

    const float radius_sq = radius * radius;
    std::vector<int32_t> result;
    result.reserve(16);

    std::vector<int32_t> stack;
    stack.push_back(this->root_index_);

    while (!stack.empty()) {
        const int32_t node_index = stack.back();
        stack.pop_back();
        const HostNode& node = this->host_nodes_[static_cast<size_t>(node_index)];
        const BoundingBox bounds{node.min_bounds, node.max_bounds};
        const float dist_sq =
            distance_to_aabb(bounds.min_bounds, bounds.max_bounds, sycl::float3(query.x(), query.y(), query.z()));
        if (dist_sq > radius_sq) {
            continue;
        }

        if (node.is_leaf) {
            for (size_t i = 0; i < node.points.size(); ++i) {
                if (squared_distance(node.points[i].point, query) <= radius_sq) {
                    result.push_back(node.points[i].id);
                }
            }
        } else {
            for (const int32_t child : node.children) {
                if (child >= 0 && this->host_nodes_[static_cast<size_t>(child)].subtree_size > 0) {
                    stack.push_back(child);
                }
            }
        }
    }

    return result;
}

/// @brief Grow the root bounding box until it encloses the provided point.
/// @param point The point to be enclosed by the root bounding box.
inline void Octree::ensure_root_bounds(const PointType& point) {
    const sycl::float3 point_vec(point.x(), point.y(), point.z());
    if (this->root_index_ < 0) {
        this->bbox_min_ = point_vec - sycl::float3(this->resolution_, this->resolution_, this->resolution_);
        this->bbox_max_ = point_vec + sycl::float3(this->resolution_, this->resolution_, this->resolution_);
        std::vector<PointRecord> pts;
        this->root_index_ = this->create_host_node(this->bbox_min_, this->bbox_max_, std::move(pts), 0);
        return;
    }

    HostNode& root = this->host_nodes_[static_cast<size_t>(this->root_index_)];
    BoundingBox root_bounds{root.min_bounds, root.max_bounds};
    if (root_bounds.contains(point_vec)) {
        return;
    }

    const sycl::float3 new_min = sycl::min(root_bounds.min_bounds, point_vec);
    const sycl::float3 new_max = sycl::max(root_bounds.max_bounds, point_vec);

    HostNode new_root;
    new_root.min_bounds = new_min;
    new_root.max_bounds = new_max;
    new_root.is_leaf = false;
    new_root.subtree_size = root.subtree_size;
    new_root.start_index = 0;
    new_root.children.fill(-1);

    int octant = 0;
    {
        const sycl::float3 center = 0.5f * (new_min + new_max);
        const sycl::float3 old_center = 0.5f * (root.min_bounds + root.max_bounds);
        if (old_center.x() >= center.x()) {
            octant |= 1;
        }
        if (old_center.y() >= center.y()) {
            octant |= 2;
        }
        if (old_center.z() >= center.z()) {
            octant |= 4;
        }
    }

    const int32_t new_root_index = static_cast<int32_t>(this->host_nodes_.size());
    this->host_nodes_.push_back(new_root);
    this->host_nodes_[static_cast<size_t>(new_root_index)].children[static_cast<size_t>(octant)] = this->root_index_;
    this->root_index_ = new_root_index;
    this->host_nodes_[static_cast<size_t>(this->root_index_)].subtree_size = this->total_point_count_;
    this->bbox_min_ = new_min;
    this->bbox_max_ = new_max;
}

/// @brief Calculate the bounding box for a child octant of the provided node.
/// @param node Parent node whose child bounds are requested.
/// @param child_index Index of the child (0-7).
/// @return Bounding box describing the child octant.
inline Octree::BoundingBox Octree::child_bounds(const HostNode& node, size_t child_index) const {
    const sycl::float3 center = 0.5f * (node.min_bounds + node.max_bounds);

    BoundingBox result;
    result.min_bounds = node.min_bounds;
    result.max_bounds = node.max_bounds;

    result.min_bounds.x() = (child_index & 1) ? center.x() : node.min_bounds.x();
    result.max_bounds.x() = (child_index & 1) ? node.max_bounds.x() : center.x();
    result.min_bounds.y() = (child_index & 2) ? center.y() : node.min_bounds.y();
    result.max_bounds.y() = (child_index & 2) ? node.max_bounds.y() : center.y();
    result.min_bounds.z() = (child_index & 4) ? center.z() : node.min_bounds.z();
    result.max_bounds.z() = (child_index & 4) ? node.max_bounds.z() : center.z();

    return result;
}

/// @brief Recalculate the number of points contained inside the subtree rooted at @p node_index.
/// @param node_index Index of the subtree's root node.
inline void Octree::recompute_subtree_size(int32_t node_index) {
    HostNode& node = this->host_nodes_[static_cast<size_t>(node_index)];
    if (node.is_leaf) {
        node.subtree_size = node.points.size();
        return;
    }

    size_t total = 0;
    for (int32_t child : node.children) {
        if (child >= 0) {
            total += this->host_nodes_[static_cast<size_t>(child)].subtree_size;
        }
    }

    node.subtree_size = total;
    if (total == 0) {
        node.is_leaf = true;
        node.children.fill(-1);
        node.points.clear();
    }
}

/// @brief Non-const overload for device buffer synchronisation that calls the const version.
inline void Octree::sync_device_buffers() { const_cast<const Octree*>(this)->sync_device_buffers(); }

/// @brief Upload host nodes and leaf points to the device buffers on demand.
inline void Octree::sync_device_buffers() const {
    if (!this->device_dirty_) {
        return;
    }

    if (this->root_index_ < 0 || this->total_point_count_ == 0) {
        this->nodes_.clear();
        this->device_points_.clear();
        this->device_point_ids_.clear();
        this->snapshot_ids_.clear();
        this->device_dirty_ = false;
        return;
    }

    const size_t node_count = this->host_nodes_.size();
    const size_t point_count = this->total_point_count_;

    // Allocate fresh shared memory buffers for the current host-side tree snapshot.
    this->nodes_ = shared_vector<Node>(node_count, Node{}, *this->queue_.ptr);
    this->device_points_ = shared_vector<PointType>(point_count, PointType(), *this->queue_.ptr);
    this->device_point_ids_ = shared_vector<int32_t>(point_count, 0, *this->queue_.ptr);

    this->snapshot_ids_.clear();
    this->snapshot_ids_.reserve(point_count);

    size_t offset = 0;

    for (size_t idx = 0; idx < node_count; ++idx) {
        HostNode& host_node = this->host_nodes_[idx];
        Node device_node{};
        device_node.min_bounds =
            Eigen::Vector3f(host_node.min_bounds.x(), host_node.min_bounds.y(), host_node.min_bounds.z());
        device_node.max_bounds =
            Eigen::Vector3f(host_node.max_bounds.x(), host_node.max_bounds.y(), host_node.max_bounds.z());
        device_node.is_leaf = host_node.is_leaf ? 1u : 0u;
        if (host_node.is_leaf) {
            device_node.data.leaf.start_index = static_cast<uint32_t>(offset);
            device_node.data.leaf.point_count = static_cast<uint32_t>(host_node.points.size());
            host_node.start_index = offset;
            for (size_t i = 0; i < host_node.points.size(); ++i) {
                const auto& record = host_node.points[i];
                this->device_points_[offset + i] = record.point;
                this->device_point_ids_[offset + i] = record.id;
                this->snapshot_ids_.push_back(record.id);
            }
            offset += host_node.points.size();
        } else {
            std::copy(host_node.children.begin(), host_node.children.end(), std::begin(device_node.data.children));
            host_node.start_index = 0;
        }
        this->nodes_[idx] = device_node;
    }

    this->device_dirty_ = false;
}

/// @brief Return a host copy of the current point order used by the device buffers.
inline std::vector<PointType> Octree::snapshot_points() const {
    this->sync_device_buffers();
    return std::vector<PointType>(this->device_points_.begin(), this->device_points_.end());
}

/// @brief Return the stable identifiers associated with @ref snapshot_points.
inline std::vector<int32_t> Octree::snapshot_ids() const {
    this->sync_device_buffers();
    return this->snapshot_ids_;
}

/// @brief Compute the squared Euclidean distance between two 4D points.
/// @param a The first point.
/// @param b The second point.
/// @return The squared Euclidean distance.
inline float squared_distance(const PointType& a, const PointType& b) {
    const PointType diff = eigen_utils::subtract<4, 1>(a, b);
    return eigen_utils::dot<4>(diff, diff);
}

SYCL_EXTERNAL inline sycl::float3 axis_lengths(const sycl::float3& min_bounds, const sycl::float3& max_bounds) {
    return max_bounds - min_bounds;
}

/// @brief Generic implementation of @ref distance_to_aabb usable with different vector types.
/// @tparam Vec3 A 3D vector type for the bounding box (e.g., `sycl::float3` or `Eigen::Vector3f`).
/// @param min_bounds Minimum coordinates of the bounding box.
/// @param max_bounds Maximum coordinates of the bounding box.
/// @param point The point to measure distance from.
/// @return The squared distance to the bounding box.
template <typename Vec3>
SYCL_EXTERNAL inline float distance_to_aabb_generic(const Vec3& min_bounds, const Vec3& max_bounds,
                                                    const sycl::float3& point) {
    // Cache the query coordinates to avoid recomputing point.x()/y()/z().
    const float px = point.x();
    const float py = point.y();
    const float pz = point.z();
    // Clamp the query position against the bounding box extents along each axis.
    const float dx = sycl::fmax(0.0f, sycl::fmax(min_bounds.x() - px, px - max_bounds.x()));
    const float dy = sycl::fmax(0.0f, sycl::fmax(min_bounds.y() - py, py - max_bounds.y()));
    const float dz = sycl::fmax(0.0f, sycl::fmax(min_bounds.z() - pz, pz - max_bounds.z()));
    // Return the squared Euclidean distance from the query point to the box surface.
    return dx * dx + dy * dy + dz * dz;
}

/// @brief Compute the squared distance from a point to an axis-aligned bounding box.
/// @param min_bounds Minimum coordinates of the bounding box.
/// @param max_bounds Maximum coordinates of the bounding box.
/// @param point The point to measure distance from.
/// @return The squared distance from the point to the bounding box.
SYCL_EXTERNAL inline float distance_to_aabb(const sycl::float3& min_bounds, const sycl::float3& max_bounds,
                                            const sycl::float3& point) {
    return distance_to_aabb_generic(min_bounds, max_bounds, point);
}

/// @copydoc distance_to_aabb(const sycl::float3&, const sycl::float3&, const sycl::float3&)
SYCL_EXTERNAL inline float distance_to_aabb(const Eigen::Vector3f& min_bounds, const Eigen::Vector3f& max_bounds,
                                            const sycl::float3& point) {
    return distance_to_aabb_generic(min_bounds, max_bounds, point);
}

inline Octree::Ptr Octree::build(const sycl_utils::DeviceQueue& queue, const PointCloudShared& points, float resolution,
                                 size_t max_points_per_node) {
    auto tree = std::make_shared<Octree>(queue, resolution, max_points_per_node);
    if (!queue.ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }

    tree->build_from_cloud(points);
    tree->sync_device_buffers();
    return tree;
}

// Dispatch helper that selects an appropriate MAX_K bound at compile time.
inline sycl_utils::events Octree::knn_search_async(
    const PointCloudShared& queries, const size_t k, KNNResult& result,
    const std::vector<sycl::event>& depends) const {
    constexpr size_t MAX_STACK_DEPTH = 32;
    if (k == 0) {
        const size_t query_size = queries.points ? queries.points->size() : 0;
        if (result.indices == nullptr || result.distances == nullptr) {
            result.allocate(this->queue_, query_size, 0);
        } else {
            result.resize(query_size, 0);
        }
        return sycl_utils::events();
    }

    if (k == 1) {
        return knn_search_async_impl<1, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 10) {
        return knn_search_async_impl<10, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 20) {
        return knn_search_async_impl<20, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 30) {
        return knn_search_async_impl<30, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 40) {
        return knn_search_async_impl<40, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 50) {
        return knn_search_async_impl<50, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 100) {
        return knn_search_async_impl<100, MAX_STACK_DEPTH>(queries, k, result, depends);
    }

    throw std::runtime_error("Requested neighbour count exceeds the supported maximum");
}

template <size_t MAX_K, size_t MAX_DEPTH>
// Core implementation of the octree kNN search.
inline sycl_utils::events Octree::knn_search_async_impl(
    const PointCloudShared& queries, size_t k, KNNResult& result,
    const std::vector<sycl::event>& depends) const {
    static_assert(MAX_DEPTH > 0, "MAX_DEPTH must be greater than zero");
    static_assert(MAX_K > 0, "MAX_K must be greater than zero");

    if (!this->queue_.ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }
    if (!queries.points) {
        throw std::runtime_error("Query cloud is not initialised");
    }
    if (k > MAX_K) {
        throw std::runtime_error("Requested neighbour count exceeds the compile-time limit");
    }

    this->sync_device_buffers();

    const size_t target_size = this->total_point_count_;
    const size_t node_count = this->nodes_.size();
    const int32_t root_index = this->root_index_;

    const size_t query_size = queries.points->size();
    if (result.indices == nullptr || result.distances == nullptr) {
        result.allocate(this->queue_, query_size, k);
    } else {
        result.resize(query_size, k);
    }

    if (target_size > 0 && (node_count == 0 || this->device_points_.empty())) {
        throw std::runtime_error("Octree structure has not been initialized");
    }

    const auto depends_copy = depends;
    auto search_task = [=](sycl::handler& handler) {
        const size_t work_group_size = this->queue_.get_work_group_size();
        const size_t global_size = this->queue_.get_global_size(query_size);

        if (!depends_copy.empty()) {
            handler.depends_on(depends_copy);
        }

        auto indices_ptr = result.indices->data();
        auto distances_ptr = result.distances->data();
        const auto query_points_ptr = queries.points->data();
        const auto nodes_ptr = this->nodes_.data();
        const auto leaf_points_ptr = this->device_points_.data();
        const auto leaf_ids_ptr = this->device_point_ids_.data();

        handler.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t query_idx = item.get_global_id(0);

            if (query_idx >= query_size) {
                return;
            }
            if (target_size == 0) {
                return;
            }

            const auto query_point = query_points_ptr[query_idx];
            const sycl::float3 query_point_vec(query_point.x(), query_point.y(), query_point.z());

            NodeEntry bestK[MAX_K];
            std::fill(bestK, bestK + MAX_K, NodeEntry{-1, std::numeric_limits<float>::max()});

            NodeEntry stack[MAX_DEPTH];
            size_t stack_size = 0;
            size_t neighbour_count = 0;

            auto heap_swap = [&](size_t a, size_t b) { std::swap(bestK[a], bestK[b]); };

            auto sift_up = [&](size_t idx) {
                while (idx > 0) {
                    const size_t parent = (idx - 1) / 2;
                    if (bestK[parent].dist_sq >= bestK[idx].dist_sq) {
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
                    if (right < heap_size && bestK[right].dist_sq > bestK[largest].dist_sq) {
                        largest = right;
                    }
                    if (bestK[idx].dist_sq >= bestK[largest].dist_sq) {
                        break;
                    }
                    heap_swap(idx, largest);
                    idx = largest;
                }
            };

            auto current_worst = [&]() {
                return neighbour_count < k ? std::numeric_limits<float>::infinity() : bestK[0].dist_sq;
            };

            auto push_candidate = [&](float distance_sq, int32_t index) {
                if (neighbour_count < k) {
                    bestK[neighbour_count] = {index, distance_sq};
                    ++neighbour_count;
                    sift_up(neighbour_count - 1);
                } else if (distance_sq < bestK[0].dist_sq) {
                    bestK[0] = {index, distance_sq};
                    sift_down(0, neighbour_count);
                }
            };

            auto push_node_to_stack = [&](int32_t node_idx, float distance_sq) {
                if (stack_size < MAX_DEPTH) {
                    stack[stack_size++] = {node_idx, distance_sq};
                } else {
                    size_t worst_pos = 0;
                    float worst_dist = stack[0].dist_sq;
                    for (size_t i = 1; i < stack_size; ++i) {
                        if (stack[i].dist_sq > worst_dist) {
                            worst_dist = stack[i].dist_sq;
                            worst_pos = i;
                        }
                    }
                    if (distance_sq < worst_dist) {
                        stack[worst_pos] = {node_idx, distance_sq};
                    }
                }
            };

            if (node_count == 0 || root_index < 0) {
                return;
            }

            push_node_to_stack(root_index, distance_to_aabb(nodes_ptr[root_index].min_bounds,
                                                            nodes_ptr[root_index].max_bounds, query_point_vec));

            while (stack_size > 0) {
                size_t best_pos = 0;
                float best_dist = stack[0].dist_sq;
                for (size_t i = 1; i < stack_size; ++i) {
                    if (stack[i].dist_sq < best_dist) {
                        best_dist = stack[i].dist_sq;
                        best_pos = i;
                    }
                }

                const int32_t current_node_idx = stack[best_pos].nodeIdx;
                --stack_size;
                stack[best_pos] = stack[stack_size];

                if (best_dist > current_worst()) {
                    continue;
                }

                const Node node = nodes_ptr[current_node_idx];
                if (node.is_leaf) {
                    const uint32_t leaf_start = node.data.leaf.start_index;
                    const uint32_t leaf_count = node.data.leaf.point_count;
                    for (uint32_t i = 0; i < leaf_count; ++i) {
                        const auto target_point = leaf_points_ptr[leaf_start + i];
                        const int32_t point_id = leaf_ids_ptr[leaf_start + i];
                        const PointType diff = eigen_utils::subtract<4, 1>(query_point, target_point);
                        const float dist_sq = eigen_utils::dot<4>(diff, diff);
                        push_candidate(dist_sq, point_id);
                    }
                } else {
                    for (size_t child = 0; child < 8; ++child) {
                        const int32_t child_idx = node.data.children[child];
                        if (child_idx < 0) {
                            continue;
                        }
                        const float child_dist = distance_to_aabb(nodes_ptr[child_idx].min_bounds,
                                                                  nodes_ptr[child_idx].max_bounds, query_point_vec);
                        if (child_dist <= current_worst()) {
                            push_node_to_stack(child_idx, child_dist);
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

            for (size_t i = 0; i < k; ++i) {
                indices_ptr[query_idx * k + i] = bestK[i].nodeIdx;
                distances_ptr[query_idx * k + i] = bestK[i].dist_sq;
            }
        });
    };

    sycl_utils::events events;
    events += this->queue_.ptr->submit(search_task);
    return events;
}

}  // namespace knn

}  // namespace algorithms

}  // namespace sycl_points

namespace sycl_points {

namespace algorithms {

namespace knn {

/// @brief Float3 helper used by the linear octree implementation.
struct LinearFloat3 {
    float x;
    float y;
    float z;

    LinearFloat3() : x(0.0f), y(0.0f), z(0.0f) {}
    LinearFloat3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

/// @brief Linear octree node representation suitable for USM allocations.
struct LinearOctreeNode {
    LinearFloat3 aabb_min;
    LinearFloat3 aabb_max;
    int32_t child_base_index;
    int32_t point_start_index;
    int32_t point_count;

    bool isLeaf() const { return child_base_index == -1; }
};

/// @brief Linear octree variant tailored for static point clouds and USM traversal.
class LinearOctree : public KNNBase {
public:
    using Ptr = std::shared_ptr<LinearOctree>;

    LinearOctree(const sycl_utils::DeviceQueue& queue, float resolution, size_t max_points_per_node);
    LinearOctree(const LinearOctree&) = delete;
    LinearOctree& operator=(const LinearOctree&) = delete;
    LinearOctree(LinearOctree&& other) noexcept;
    LinearOctree& operator=(LinearOctree&& other) noexcept;
    ~LinearOctree();

    static Ptr build(const sycl_utils::DeviceQueue& queue, const PointCloudShared& points, float resolution,
                     size_t max_points_per_node = 32);

    [[nodiscard]] size_t size() const { return point_count_; }
    [[nodiscard]] float resolution() const { return resolution_; }
    [[nodiscard]] size_t max_points_per_node() const { return max_points_per_node_; }

    sycl_utils::events knn_search_async(
        const PointCloudShared& queries, size_t k, KNNResult& result,
        const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const override;

    [[nodiscard]] KNNResult knn_search(const PointCloudShared& queries, size_t k) const {
        return KNNBase::knn_search(queries, k);
    }

private:
    struct MortonPoint {
        LinearFloat3 position;
        uint64_t morton_code;
        int32_t id;
    };

    struct HostNode {
        LinearFloat3 min_bounds;
        LinearFloat3 max_bounds;
        std::array<int32_t, 8> children;
        int32_t point_start;
        int32_t point_count;
        bool is_leaf;

        HostNode() : min_bounds(), max_bounds(), point_start(0), point_count(0), is_leaf(true) {
            children.fill(-1);
        }
    };

    template <size_t MAX_K, size_t MAX_DEPTH>
    sycl_utils::events knn_search_async_impl(
        const PointCloudShared& queries, size_t k, KNNResult& result,
        const std::vector<sycl::event>& depends) const;

    void build_from_cloud(const PointCloudShared& points);
    void release_device_memory();

    int32_t build_host_subtree(int32_t start_index, int32_t end_index, const LinearFloat3& min_bounds,
                               const LinearFloat3& max_bounds, size_t depth, std::vector<HostNode>& host_nodes,
                               std::vector<LinearFloat3>& sorted_points, std::vector<LinearFloat3>& scratch_points,
                               std::vector<int32_t>& sorted_ids, std::vector<int32_t>& scratch_ids,
                               std::vector<uint64_t>& sorted_morton, std::vector<uint64_t>& scratch_morton);

    void linearise_tree(const std::vector<HostNode>& host_nodes, int32_t root_index,
                        std::vector<LinearOctreeNode>& linear_nodes) const;

    static LinearFloat3 compute_child_min(const LinearFloat3& parent_min, const LinearFloat3& parent_max,
                                          size_t child_index);
    static LinearFloat3 compute_child_max(const LinearFloat3& parent_min, const LinearFloat3& parent_max,
                                          size_t child_index);
    static LinearFloat3 point_from_type(const PointType& point);
    static LinearFloat3 make_safe_lengths(const LinearFloat3& min_bounds, const LinearFloat3& max_bounds);
    static float distance_to_aabb(const LinearFloat3& min_bounds, const LinearFloat3& max_bounds,
                                  const LinearFloat3& point);
    static float squared_distance(const LinearFloat3& a, const LinearFloat3& b);
    static uint64_t morton_encode(const LinearFloat3& point, const LinearFloat3& min_bounds,
                                  const LinearFloat3& max_bounds);

    sycl_utils::DeviceQueue queue_;
    float resolution_;
    size_t max_points_per_node_;
    LinearOctreeNode* nodes_usm_;
    LinearFloat3* points_usm_;
    int32_t* point_ids_usm_;
    size_t node_count_;
    size_t point_count_;
};

inline LinearFloat3 operator+(const LinearFloat3& lhs, const LinearFloat3& rhs) {
    return LinearFloat3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

inline LinearFloat3 operator-(const LinearFloat3& lhs, const LinearFloat3& rhs) {
    return LinearFloat3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

inline LinearFloat3 operator*(const LinearFloat3& lhs, float rhs) {
    return LinearFloat3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

inline LinearFloat3 operator*(float lhs, const LinearFloat3& rhs) {
    return rhs * lhs;
}

inline LinearFloat3 min_components(const LinearFloat3& lhs, const LinearFloat3& rhs) {
    return LinearFloat3(std::min(lhs.x, rhs.x), std::min(lhs.y, rhs.y), std::min(lhs.z, rhs.z));
}

inline LinearFloat3 max_components(const LinearFloat3& lhs, const LinearFloat3& rhs) {
    return LinearFloat3(std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y), std::max(lhs.z, rhs.z));
}

inline LinearOctree::LinearOctree(const sycl_utils::DeviceQueue& queue, float resolution, size_t max_points_per_node)
    : queue_(queue),
      resolution_(resolution),
      max_points_per_node_(max_points_per_node),
      nodes_usm_(nullptr),
      points_usm_(nullptr),
      point_ids_usm_(nullptr),
      node_count_(0),
      point_count_(0) {}

inline LinearOctree::LinearOctree(LinearOctree&& other) noexcept
    : queue_(other.queue_),
      resolution_(other.resolution_),
      max_points_per_node_(other.max_points_per_node_),
      nodes_usm_(other.nodes_usm_),
      points_usm_(other.points_usm_),
      point_ids_usm_(other.point_ids_usm_),
      node_count_(other.node_count_),
      point_count_(other.point_count_) {
    other.nodes_usm_ = nullptr;
    other.points_usm_ = nullptr;
    other.point_ids_usm_ = nullptr;
    other.node_count_ = 0;
    other.point_count_ = 0;
}

inline LinearOctree& LinearOctree::operator=(LinearOctree&& other) noexcept {
    if (this != &other) {
        release_device_memory();
        queue_ = other.queue_;
        resolution_ = other.resolution_;
        max_points_per_node_ = other.max_points_per_node_;
        nodes_usm_ = other.nodes_usm_;
        points_usm_ = other.points_usm_;
        point_ids_usm_ = other.point_ids_usm_;
        node_count_ = other.node_count_;
        point_count_ = other.point_count_;
        other.nodes_usm_ = nullptr;
        other.points_usm_ = nullptr;
        other.point_ids_usm_ = nullptr;
        other.node_count_ = 0;
        other.point_count_ = 0;
    }
    return *this;
}

inline LinearOctree::~LinearOctree() { release_device_memory(); }

inline void LinearOctree::release_device_memory() {
    if (queue_.ptr) {
        if (nodes_usm_ != nullptr) {
            sycl::free(nodes_usm_, *queue_.ptr);
            nodes_usm_ = nullptr;
        }
        if (points_usm_ != nullptr) {
            sycl::free(points_usm_, *queue_.ptr);
            points_usm_ = nullptr;
        }
        if (point_ids_usm_ != nullptr) {
            sycl::free(point_ids_usm_, *queue_.ptr);
            point_ids_usm_ = nullptr;
        }
    }
    node_count_ = 0;
    point_count_ = 0;
}

inline LinearFloat3 LinearOctree::point_from_type(const PointType& point) {
    return LinearFloat3(point.x(), point.y(), point.z());
}

inline LinearFloat3 LinearOctree::make_safe_lengths(const LinearFloat3& min_bounds, const LinearFloat3& max_bounds) {
    const float epsilon = 1e-6f;
    const float dx = std::max(max_bounds.x - min_bounds.x, epsilon);
    const float dy = std::max(max_bounds.y - min_bounds.y, epsilon);
    const float dz = std::max(max_bounds.z - min_bounds.z, epsilon);
    return LinearFloat3(dx, dy, dz);
}

inline float LinearOctree::distance_to_aabb(const LinearFloat3& min_bounds, const LinearFloat3& max_bounds,
                                            const LinearFloat3& point) {
    const float dx = std::max({0.0f, min_bounds.x - point.x, point.x - max_bounds.x});
    const float dy = std::max({0.0f, min_bounds.y - point.y, point.y - max_bounds.y});
    const float dz = std::max({0.0f, min_bounds.z - point.z, point.z - max_bounds.z});
    return dx * dx + dy * dy + dz * dz;
}

inline float LinearOctree::squared_distance(const LinearFloat3& a, const LinearFloat3& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

inline LinearFloat3 LinearOctree::compute_child_min(const LinearFloat3& parent_min, const LinearFloat3& parent_max,
                                                    size_t child_index) {
    const LinearFloat3 center = 0.5f * (parent_min + parent_max);
    LinearFloat3 child_min = parent_min;
    if (child_index & 1u) {
        child_min.x = center.x;
    }
    if (child_index & 2u) {
        child_min.y = center.y;
    }
    if (child_index & 4u) {
        child_min.z = center.z;
    }
    return child_min;
}

inline LinearFloat3 LinearOctree::compute_child_max(const LinearFloat3& parent_min, const LinearFloat3& parent_max,
                                                    size_t child_index) {
    const LinearFloat3 center = 0.5f * (parent_min + parent_max);
    LinearFloat3 child_max = parent_max;
    if ((child_index & 1u) == 0u) {
        child_max.x = center.x;
    }
    if ((child_index & 2u) == 0u) {
        child_max.y = center.y;
    }
    if ((child_index & 4u) == 0u) {
        child_max.z = center.z;
    }
    return child_max;
}

inline uint64_t LinearOctree::morton_encode(const LinearFloat3& point, const LinearFloat3& min_bounds,
                                            const LinearFloat3& max_bounds) {
    constexpr uint32_t MORTON_BITS = 21;
    constexpr uint32_t MAX_COORD = (1u << MORTON_BITS) - 1u;

    const LinearFloat3 lengths = make_safe_lengths(min_bounds, max_bounds);
    const float nx = (point.x - min_bounds.x) / lengths.x;
    const float ny = (point.y - min_bounds.y) / lengths.y;
    const float nz = (point.z - min_bounds.z) / lengths.z;

    const uint32_t ix = static_cast<uint32_t>(std::clamp(nx, 0.0f, 1.0f) * static_cast<float>(MAX_COORD));
    const uint32_t iy = static_cast<uint32_t>(std::clamp(ny, 0.0f, 1.0f) * static_cast<float>(MAX_COORD));
    const uint32_t iz = static_cast<uint32_t>(std::clamp(nz, 0.0f, 1.0f) * static_cast<float>(MAX_COORD));

    auto expand_bits = [](uint32_t value) {
        uint64_t v = value & 0x1fffff;
        v = (v | (v << 32)) & 0x1f00000000ffffULL;
        v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
        v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
        v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
        v = (v | (v << 2)) & 0x1249249249249249ULL;
        return v;
    };

    return (expand_bits(ix) << 2) | (expand_bits(iy) << 1) | expand_bits(iz);
}

inline void LinearOctree::build_from_cloud(const PointCloudShared& points) {
    if (!queue_.ptr) {
        throw std::runtime_error("LinearOctree queue is not initialised");
    }
    if (!points.points) {
        throw std::runtime_error("Point cloud is not initialised");
    }

    release_device_memory();

    const size_t count = points.points->size();
    point_count_ = count;
    if (count == 0) {
        return;
    }

    std::vector<MortonPoint> morton_points(count);
    LinearFloat3 bbox_min(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::infinity());
    LinearFloat3 bbox_max(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                          std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < count; ++i) {
        const PointType& pt = (*points.points)[i];
        const LinearFloat3 converted = point_from_type(pt);
        bbox_min = min_components(bbox_min, converted);
        bbox_max = max_components(bbox_max, converted);
        morton_points[i] = MortonPoint{converted, 0ULL, static_cast<int32_t>(i)};
    }

    const float epsilon = std::max(1e-5f, resolution_ * 0.5f);
    bbox_min = bbox_min - LinearFloat3(epsilon, epsilon, epsilon);
    bbox_max = bbox_max + LinearFloat3(epsilon, epsilon, epsilon);

    for (auto& entry : morton_points) {
        entry.morton_code = morton_encode(entry.position, bbox_min, bbox_max);
    }

    std::sort(morton_points.begin(), morton_points.end(),
              [](const MortonPoint& a, const MortonPoint& b) { return a.morton_code < b.morton_code; });

    std::vector<LinearFloat3> sorted_points(count);
    std::vector<int32_t> sorted_ids(count);
    std::vector<uint64_t> sorted_morton(count);
    for (size_t i = 0; i < count; ++i) {
        sorted_points[i] = morton_points[i].position;
        sorted_ids[i] = morton_points[i].id;
        sorted_morton[i] = morton_points[i].morton_code;
    }

    std::vector<LinearFloat3> scratch_points(count);
    std::vector<int32_t> scratch_ids(count);
    std::vector<uint64_t> scratch_morton(count);

    std::vector<HostNode> host_nodes;
    host_nodes.reserve(count * 2);

    const int32_t root_index = build_host_subtree(0, static_cast<int32_t>(count), bbox_min, bbox_max, 0, host_nodes,
                                                  sorted_points, scratch_points, sorted_ids, scratch_ids, sorted_morton,
                                                  scratch_morton);

    std::vector<LinearOctreeNode> linear_nodes;
    linear_nodes.reserve(host_nodes.size() * 2);
    linearise_tree(host_nodes, root_index, linear_nodes);

    node_count_ = linear_nodes.size();
    nodes_usm_ = sycl::malloc_shared<LinearOctreeNode>(node_count_, *queue_.ptr);
    points_usm_ = sycl::malloc_shared<LinearFloat3>(count, *queue_.ptr);
    point_ids_usm_ = sycl::malloc_shared<int32_t>(count, *queue_.ptr);

    std::copy(linear_nodes.begin(), linear_nodes.end(), nodes_usm_);
    std::copy(sorted_points.begin(), sorted_points.end(), points_usm_);
    std::copy(sorted_ids.begin(), sorted_ids.end(), point_ids_usm_);
}

inline int32_t LinearOctree::build_host_subtree(int32_t start_index, int32_t end_index, const LinearFloat3& min_bounds,
                                                const LinearFloat3& max_bounds, size_t depth,
                                                std::vector<HostNode>& host_nodes,
                                                std::vector<LinearFloat3>& sorted_points,
                                                std::vector<LinearFloat3>& scratch_points,
                                                std::vector<int32_t>& sorted_ids, std::vector<int32_t>& scratch_ids,
                                                std::vector<uint64_t>& sorted_morton,
                                                std::vector<uint64_t>& scratch_morton) {
    HostNode node;
    node.min_bounds = min_bounds;
    node.max_bounds = max_bounds;
    node.point_start = start_index;
    node.point_count = end_index - start_index;

    const int32_t node_index = static_cast<int32_t>(host_nodes.size());
    host_nodes.push_back(node);

    if (node.point_count <= 0) {
        return node_index;
    }

    const LinearFloat3 extents = max_bounds - min_bounds;
    const float max_axis = std::max({extents.x, extents.y, extents.z});
    const bool depth_limit = depth >= 32;

    if (node.point_count <= static_cast<int32_t>(max_points_per_node_) || max_axis <= resolution_ || depth_limit) {
        host_nodes[node_index].is_leaf = true;
        return node_index;
    }

    const LinearFloat3 center = 0.5f * (min_bounds + max_bounds);
    std::array<int32_t, 8> counts{};
    for (int32_t idx = start_index; idx < end_index; ++idx) {
        const LinearFloat3& p = sorted_points[static_cast<size_t>(idx)];
        size_t child = 0;
        if (p.x >= center.x) {
            child |= 1u;
        }
        if (p.y >= center.y) {
            child |= 2u;
        }
        if (p.z >= center.z) {
            child |= 4u;
        }
        counts[child] += 1;
    }

    size_t occupied_children = 0;
    for (const auto count : counts) {
        if (count > 0) {
            ++occupied_children;
        }
    }

    if (occupied_children <= 1) {
        host_nodes[node_index].is_leaf = true;
        return node_index;
    }

    std::array<int32_t, 8> offsets{};
    int32_t total = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        offsets[i] = total;
        total += counts[i];
    }

    for (int32_t idx = start_index; idx < end_index; ++idx) {
        const size_t local_idx = static_cast<size_t>(idx - start_index);
        const LinearFloat3& point = sorted_points[static_cast<size_t>(idx)];
        size_t child = 0;
        if (point.x >= center.x) {
            child |= 1u;
        }
        if (point.y >= center.y) {
            child |= 2u;
        }
        if (point.z >= center.z) {
            child |= 4u;
        }
        const int32_t dest = offsets[child]++;
        scratch_points[static_cast<size_t>(dest)] = point;
        scratch_ids[static_cast<size_t>(dest)] = sorted_ids[static_cast<size_t>(idx)];
        scratch_morton[static_cast<size_t>(dest)] = sorted_morton[static_cast<size_t>(idx)];
    }

    for (int32_t i = 0; i < total; ++i) {
        sorted_points[static_cast<size_t>(start_index + i)] = scratch_points[static_cast<size_t>(i)];
        sorted_ids[static_cast<size_t>(start_index + i)] = scratch_ids[static_cast<size_t>(i)];
        sorted_morton[static_cast<size_t>(start_index + i)] = scratch_morton[static_cast<size_t>(i)];
    }

    offsets[0] = 0;
    for (size_t i = 1; i < counts.size(); ++i) {
        offsets[i] = offsets[i - 1] + counts[i - 1];
    }

    host_nodes[node_index].is_leaf = false;
    for (size_t child = 0; child < counts.size(); ++child) {
        const int32_t child_count = counts[child];
        if (child_count == 0) {
            continue;
        }
        const int32_t child_start = start_index + offsets[child];
        const int32_t child_end = child_start + child_count;
        const LinearFloat3 child_min = compute_child_min(min_bounds, max_bounds, child);
        const LinearFloat3 child_max = compute_child_max(min_bounds, max_bounds, child);
        const int32_t child_index = build_host_subtree(child_start, child_end, child_min, child_max, depth + 1,
                                                       host_nodes, sorted_points, scratch_points, sorted_ids,
                                                       scratch_ids, sorted_morton, scratch_morton);
        host_nodes[node_index].children[child] = child_index;
    }

    return node_index;
}

inline void LinearOctree::linearise_tree(const std::vector<HostNode>& host_nodes, int32_t root_index,
                                         std::vector<LinearOctreeNode>& linear_nodes) const {
    if (root_index < 0 || host_nodes.empty()) {
        return;
    }

    std::vector<int32_t> host_to_linear(host_nodes.size(), -1);
    std::queue<int32_t> pending;

    linear_nodes.clear();
    linear_nodes.push_back(LinearOctreeNode{});
    host_to_linear[static_cast<size_t>(root_index)] = 0;
    pending.push(root_index);

    while (!pending.empty()) {
        const int32_t host_idx = pending.front();
        pending.pop();

        const HostNode& host_node = host_nodes[static_cast<size_t>(host_idx)];
        const int32_t linear_idx = host_to_linear[static_cast<size_t>(host_idx)];
        LinearOctreeNode& linear_node = linear_nodes[static_cast<size_t>(linear_idx)];

        linear_node.aabb_min = host_node.min_bounds;
        linear_node.aabb_max = host_node.max_bounds;
        linear_node.point_start_index = host_node.point_start;
        linear_node.point_count = host_node.point_count;

        if (host_node.is_leaf) {
            linear_node.child_base_index = -1;
            continue;
        }

        linear_node.child_base_index = static_cast<int32_t>(linear_nodes.size());
        for (size_t child = 0; child < host_node.children.size(); ++child) {
            LinearOctreeNode child_node{};
            child_node.aabb_min = compute_child_min(host_node.min_bounds, host_node.max_bounds, child);
            child_node.aabb_max = compute_child_max(host_node.min_bounds, host_node.max_bounds, child);
            child_node.child_base_index = -1;
            child_node.point_start_index = host_node.point_start;
            child_node.point_count = 0;

            const int32_t child_host_idx = host_node.children[child];
            if (child_host_idx >= 0) {
                host_to_linear[static_cast<size_t>(child_host_idx)] = static_cast<int32_t>(linear_nodes.size());
                pending.push(child_host_idx);
            }

            linear_nodes.push_back(child_node);
        }
    }
}

inline LinearOctree::Ptr LinearOctree::build(const sycl_utils::DeviceQueue& queue, const PointCloudShared& points,
                                             float resolution, size_t max_points_per_node) {
    auto tree = std::make_shared<LinearOctree>(queue, resolution, max_points_per_node);
    tree->build_from_cloud(points);
    return tree;
}

inline sycl_utils::events LinearOctree::knn_search_async(
    const PointCloudShared& queries, size_t k, KNNResult& result,
    const std::vector<sycl::event>& depends) const {
    constexpr size_t MAX_STACK_DEPTH = 32;
    if (k == 0) {
        const size_t query_size = queries.points ? queries.points->size() : 0;
        if (result.indices == nullptr || result.distances == nullptr) {
            result.allocate(queue_, query_size, 0);
        } else {
            result.resize(query_size, 0);
        }
        return sycl_utils::events();
    }

    if (k == 1) {
        return knn_search_async_impl<1, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 10) {
        return knn_search_async_impl<10, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 20) {
        return knn_search_async_impl<20, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 30) {
        return knn_search_async_impl<30, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 40) {
        return knn_search_async_impl<40, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 50) {
        return knn_search_async_impl<50, MAX_STACK_DEPTH>(queries, k, result, depends);
    } else if (k <= 100) {
        return knn_search_async_impl<100, MAX_STACK_DEPTH>(queries, k, result, depends);
    }

    throw std::runtime_error("Requested neighbour count exceeds the supported maximum");
}

template <size_t MAX_K, size_t MAX_DEPTH>
inline sycl_utils::events LinearOctree::knn_search_async_impl(
    const PointCloudShared& queries, size_t k, KNNResult& result,
    const std::vector<sycl::event>& depends) const {
    if (!queue_.ptr) {
        throw std::runtime_error("LinearOctree queue is not initialised");
    }
    if (!queries.points) {
        throw std::runtime_error("Query cloud is not initialised");
    }
    if (k > MAX_K) {
        throw std::runtime_error("Requested neighbour count exceeds the compile-time limit");
    }

    const size_t query_size = queries.points->size();
    if (result.indices == nullptr || result.distances == nullptr) {
        result.allocate(queue_, query_size, k);
    } else {
        result.resize(query_size, k);
    }

    if (point_count_ > 0 && (nodes_usm_ == nullptr || points_usm_ == nullptr || point_ids_usm_ == nullptr)) {
        throw std::runtime_error("LinearOctree structure has not been initialised");
    }

    const auto depends_copy = depends;
    auto search_task = [=](sycl::handler& handler) {
        const size_t work_group_size = queue_.get_work_group_size();
        const size_t global_size = queue_.get_global_size(query_size);

        if (!depends_copy.empty()) {
            handler.depends_on(depends_copy);
        }

        auto indices_ptr = result.indices->data();
        auto distances_ptr = result.distances->data();
        const auto query_points_ptr = queries.points->data();
        const auto nodes_ptr = nodes_usm_;
        const auto leaf_points_ptr = points_usm_;
        const auto leaf_ids_ptr = point_ids_usm_;
        const size_t node_count = node_count_;
        const size_t target_size = point_count_;

        handler.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t query_idx = item.get_global_id(0);

            if (query_idx >= query_size || target_size == 0) {
                return;
            }

            const PointType query_point = query_points_ptr[query_idx];
            const LinearFloat3 query_vec(query_point.x(), query_point.y(), query_point.z());

            struct Candidate {
                int32_t index;
                float dist_sq;
            };

            Candidate best_k[MAX_K];
            for (size_t i = 0; i < MAX_K; ++i) {
                best_k[i] = Candidate{-1, std::numeric_limits<float>::max()};
            }

            Candidate stack[MAX_DEPTH];
            size_t stack_size = 0;
            size_t neighbour_count = 0;

            auto heap_swap = [&](size_t a, size_t b) { std::swap(best_k[a], best_k[b]); };

            auto sift_up = [&](size_t idx) {
                while (idx > 0) {
                    const size_t parent = (idx - 1) / 2;
                    if (best_k[parent].dist_sq >= best_k[idx].dist_sq) {
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
                    if (right < heap_size && best_k[right].dist_sq > best_k[largest].dist_sq) {
                        largest = right;
                    }
                    if (best_k[idx].dist_sq >= best_k[largest].dist_sq) {
                        break;
                    }
                    heap_swap(idx, largest);
                    idx = largest;
                }
            };

            auto current_worst = [&]() {
                return neighbour_count < k ? std::numeric_limits<float>::infinity() : best_k[0].dist_sq;
            };

            auto push_candidate = [&](float distance_sq, int32_t index) {
                if (neighbour_count < k) {
                    best_k[neighbour_count] = Candidate{index, distance_sq};
                    ++neighbour_count;
                    sift_up(neighbour_count - 1);
                } else if (distance_sq < best_k[0].dist_sq) {
                    best_k[0] = Candidate{index, distance_sq};
                    sift_down(0, neighbour_count);
                }
            };

            auto push_node_to_stack = [&](int32_t node_idx, float distance_sq) {
                if (stack_size < MAX_DEPTH) {
                    stack[stack_size++] = Candidate{node_idx, distance_sq};
                } else {
                    size_t worst_pos = 0;
                    float worst_dist = stack[0].dist_sq;
                    for (size_t i = 1; i < stack_size; ++i) {
                        if (stack[i].dist_sq > worst_dist) {
                            worst_dist = stack[i].dist_sq;
                            worst_pos = i;
                        }
                    }
                    if (distance_sq < worst_dist) {
                        stack[worst_pos] = Candidate{node_idx, distance_sq};
                    }
                }
            };

            if (node_count == 0) {
                return;
            }

            push_node_to_stack(0, distance_to_aabb(nodes_ptr[0].aabb_min, nodes_ptr[0].aabb_max, query_vec));

            while (stack_size > 0) {
                size_t best_pos = 0;
                float best_dist = stack[0].dist_sq;
                for (size_t i = 1; i < stack_size; ++i) {
                    if (stack[i].dist_sq < best_dist) {
                        best_dist = stack[i].dist_sq;
                        best_pos = i;
                    }
                }

                const int32_t current_node_idx = stack[best_pos].index;
                --stack_size;
                stack[best_pos] = stack[stack_size];

                if (best_dist > current_worst()) {
                    continue;
                }

                const LinearOctreeNode node = nodes_ptr[current_node_idx];
                if (node.isLeaf()) {
                    for (int32_t i = 0; i < node.point_count; ++i) {
                        const int32_t point_idx = node.point_start_index + i;
                        const LinearFloat3 target_point = leaf_points_ptr[point_idx];
                        const int32_t target_id = leaf_ids_ptr[point_idx];
                        const float dist_sq = squared_distance(query_vec, target_point);
                        push_candidate(dist_sq, target_id);
                    }
                } else {
                    for (size_t child = 0; child < 8; ++child) {
                        const int32_t child_idx = node.child_base_index + static_cast<int32_t>(child);
                        if (child_idx < 0 || static_cast<size_t>(child_idx) >= node_count) {
                            continue;
                        }
                        const LinearOctreeNode child_node = nodes_ptr[child_idx];
                        const float child_dist = distance_to_aabb(child_node.aabb_min, child_node.aabb_max, query_vec);
                        if (child_dist <= current_worst()) {
                            push_node_to_stack(child_idx, child_dist);
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

            for (size_t i = 0; i < k; ++i) {
                indices_ptr[query_idx * k + i] = best_k[i].index;
                distances_ptr[query_idx * k + i] = best_k[i].dist_sq;
            }
        });
    };

    sycl_utils::events events;
    events += queue_.ptr->submit(search_task);
    return events;
}

}  // namespace knn

}  // namespace algorithms

}  // namespace sycl_points
