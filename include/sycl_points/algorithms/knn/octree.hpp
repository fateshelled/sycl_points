#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <sycl_points/algorithms/common/filter_by_flags.hpp>
#include <sycl_points/algorithms/knn/knn.hpp>
#include <sycl_points/algorithms/knn/result.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace sycl_points {

namespace algorithms {

namespace knn {

/// @brief Octree data structure that will support parallel construction and neighbour search on SYCL devices.
class Octree : public KNNBase {
public:
    using Ptr = std::shared_ptr<Octree>;

    /// @brief Axis-aligned bounding box helper used during host-side operations.
    struct BoundingBox {
        Eigen::Vector3f min_bounds;
        Eigen::Vector3f max_bounds;

        [[nodiscard]] bool contains(const Eigen::Vector3f& point) const {
            return point.x() >= min_bounds.x() && point.x() <= max_bounds.x() && point.y() >= min_bounds.y() &&
                   point.y() <= max_bounds.y() && point.z() >= min_bounds.z() && point.z() <= max_bounds.z();
        }

        [[nodiscard]] bool contains(const PointType& point) const {
            return contains(Eigen::Vector3f(point.x(), point.y(), point.z()));
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

        [[nodiscard]] Eigen::Vector3f center() const { return 0.5f * (min_bounds + max_bounds); }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    /// @brief Compact representation of a node shared between host and device.
    struct Node {
        BoundingBox bounds;
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

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    static_assert(std::is_standard_layout_v<Node>, "Octree::Node must remain standard-layout");

    static_assert(sizeof(Node) == 64, "Octree::Node must remain 64 bytes");

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
    sycl_utils::events knn_search_async(const PointCloudShared& queries, const size_t k, KNNResult& result,
                                        const std::vector<sycl::event>& depends = std::vector<sycl::event>(),
                                        const TransformMatrix& transT = TransformMatrix::Identity()) const override;

    /// @brief Accessor for the resolution that was requested at build time.
    [[nodiscard]] float resolution() const { return this->resolution_; }

    /// @brief Accessor for the maximum number of points per node.
    [[nodiscard]] size_t max_points_per_node() const { return this->max_points_per_node_; }

    /// @brief Number of points stored in the Octree.
    [[nodiscard]] size_t size() const { return this->total_point_count_; }

    /// @brief Compact the tree by removing nodes flagged for deletion.
    void remove_nodes_by_flags(const shared_vector<uint8_t>& flags, const shared_vector<int32_t>& indices);

private:
    /// @brief Host-side storage for a point and its persistent identifier.
    struct PointRecord {
        PointType point;
        int32_t id;
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
    int32_t create_host_node(const Eigen::Vector3f& min_bounds, const Eigen::Vector3f& max_bounds,
                             std::vector<PointRecord>&& points, size_t depth);
    /// @brief Split the provided leaf node when it exceeds capacity.
    void subdivide_leaf(int32_t node_index, size_t depth);

    /// @brief Compute the bounding box of a specific child octant.
    BoundingBox child_bounds(const Node& node, size_t child_index) const;
    /// @brief Refresh the cached subtree size after structural changes.
    void recompute_subtree_size(int32_t node_index);
    /// @brief Synchronise host-side data into device-visible buffers.
    void sync_device_buffers();
    /// @brief Const-qualified overload that forwards to the non-const synchronisation path.
    void sync_device_buffers() const;
    /// @brief Reset all host/device bookkeeping to the empty-tree state.
    void reset_tree_state();

    /// @brief Compute the squared distance from a point to an axis-aligned bounding box.
    /// @param min_bounds Minimum coordinates of the bounding box.
    /// @param max_bounds Maximum coordinates of the bounding box.
    /// @param point The point to measure distance from.
    /// @return The squared distance from the point to the bounding box.
    static float distance_to_aabb(const Eigen::Vector3f& min_bounds, const Eigen::Vector3f& max_bounds,
                                  const Eigen::Vector3f& point);

    sycl_utils::DeviceQueue queue_;
    float resolution_;
    size_t max_points_per_node_;
    Eigen::Vector3f bbox_min_;
    Eigen::Vector3f bbox_max_;
    int32_t root_index_;
    size_t total_point_count_;
    int32_t next_point_id_;
    /// @brief Host-side mirror of the node array shared with device buffers.
    mutable std::vector<Node> host_nodes_;
    /// @brief Points stored per leaf node for host-side updates.
    mutable std::vector<std::vector<PointRecord>> host_leaf_points_;
    /// @brief Cached subtree sizes used to accelerate host queries and updates.
    mutable std::vector<size_t> host_subtree_sizes_;
    mutable shared_vector<Node> nodes_;
    mutable PointContainerShared device_points_;
    mutable shared_vector<int32_t> device_point_ids_;
    mutable bool device_dirty_;

    template <size_t MAX_K, size_t MAX_DEPTH>
    sycl_utils::events knn_search_async_impl(const PointCloudShared& queries, size_t k, KNNResult& result,
                                             const std::vector<sycl::event>& depends,
                                             const TransformMatrix& transT = TransformMatrix::Identity()) const;
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
      host_leaf_points_(),
      host_subtree_sizes_(),
      nodes_(*queue_.ptr),
      device_points_(*queue_.ptr),
      device_point_ids_(*queue_.ptr),
      device_dirty_(true) {}

inline void Octree::reset_tree_state() {
    this->bbox_min_.fill(std::numeric_limits<float>::infinity());
    this->bbox_max_.fill(std::numeric_limits<float>::lowest());
    this->root_index_ = -1;
    this->total_point_count_ = 0;
    this->next_point_id_ = 0;

    this->host_nodes_.clear();
    this->host_leaf_points_.clear();
    this->host_subtree_sizes_.clear();

    this->nodes_.clear();
    this->device_points_.clear();
    this->device_point_ids_.clear();

    this->device_dirty_ = true;
}

/// @brief Populate the octree from an entire point cloud.
/// @details The method resets any previously stored data and rebuilds the hierarchy from scratch.
/// @param points Input point cloud.
inline void Octree::build_from_cloud(const PointCloudShared& points) {
    if (!this->queue_.ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }

    // Reset cached host/device structures before rebuilding the tree.
    this->reset_tree_state();
    if (!points.points) {
        throw std::runtime_error("Point cloud is not initialised");
    }

    const size_t point_count = points.points->size();
    if (point_count == 0) {
        return;
    }

    this->bbox_min_ = Eigen::Vector3f(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                                      std::numeric_limits<float>::infinity());
    this->bbox_max_ = Eigen::Vector3f(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
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
    this->bbox_min_ -= Eigen::Vector3f(epsilon, epsilon, epsilon);
    this->bbox_max_ += Eigen::Vector3f(epsilon, epsilon, epsilon);

    this->root_index_ = this->create_host_node(this->bbox_min_, this->bbox_max_, std::move(records), 0);
    this->total_point_count_ = point_count;
    this->next_point_id_ = static_cast<int32_t>(point_count);
}

inline void Octree::remove_nodes_by_flags(const shared_vector<uint8_t>& flags, const shared_vector<int32_t>& indices) {
    if (!this->queue_.ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }

    if (flags.size() != indices.size()) {
        throw std::runtime_error("flags and indices must have the same size");
    }

    const size_t expected_size = static_cast<size_t>(this->next_point_id_);
    if (flags.size() != expected_size) {
        throw std::runtime_error("flags and indices must match the octree point identifier range");
    }

    if (expected_size == 0 || this->root_index_ < 0 || this->host_nodes_.empty()) {
        this->reset_tree_state();
        return;
    }

    bool modified = false;
    int32_t max_new_id = -1;

    for (size_t node_idx = 0; node_idx < this->host_nodes_.size(); ++node_idx) {
        Node& node = this->host_nodes_[node_idx];
        if (node.is_leaf == 0u) {
            continue;
        }

        auto& points = this->host_leaf_points_[node_idx];
        const size_t original_size = points.size();
        size_t write_pos = 0;

        for (size_t i = 0; i < original_size; ++i) {
            const PointRecord& record = points[i];
            const int32_t record_id = record.id;
            if (record_id < 0) {
                continue;
            }

            const size_t id_index = static_cast<size_t>(record_id);
            if (id_index >= expected_size) {
                throw std::runtime_error("flags array does not cover the stored point identifier");
            }

            const int32_t new_id = indices[id_index];
            const bool keep = flags[id_index] == filter::INCLUDE_FLAG && new_id >= 0;
            if (!keep) {
                modified = true;
                continue;
            }

            if (new_id >= static_cast<int32_t>(expected_size)) {
                throw std::runtime_error("remapped point identifier in indices exceeds the allocated range");
            }

            if (new_id != record_id) {
                modified = true;
            }

            points[write_pos++] = PointRecord{record.point, new_id};
            max_new_id = std::max(max_new_id, new_id);
        }

        if (write_pos != original_size) {
            points.resize(write_pos);
        }
    }

    if (!modified) {
        return;
    }

    this->device_dirty_ = true;

    for (size_t idx = this->host_nodes_.size(); idx-- > 0;) {
        this->recompute_subtree_size(static_cast<int32_t>(idx));
    }

    if (this->root_index_ >= 0) {
        this->total_point_count_ = this->host_subtree_sizes_[static_cast<size_t>(this->root_index_)];
    } else {
        this->total_point_count_ = 0;
    }

    if (this->total_point_count_ == 0) {
        this->reset_tree_state();
        return;
    }

    int64_t candidate_next_id = (max_new_id >= 0) ? (static_cast<int64_t>(max_new_id) + 1) : 0;
    if (candidate_next_id < 0) {
        candidate_next_id = 0;
    }

    const int64_t total_count_candidate = static_cast<int64_t>(this->total_point_count_);
    const int64_t next_id_64 = std::max(candidate_next_id, total_count_candidate);
    if (next_id_64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("compacted point identifiers exceed int32_t capacity");
    }

    this->next_point_id_ = static_cast<int32_t>(next_id_64);
}

/// @brief Create a host node and recursively subdivide it when necessary.
/// @param min_bounds Minimum coordinates of the node's bounding box.
/// @param max_bounds Maximum coordinates of the node's bounding box.
/// @param points Points contained within the node's bounding box.
/// @param depth Current depth of the node in the tree.
/// @return Index of the created node inside @c host_nodes_.
inline int32_t Octree::create_host_node(const Eigen::Vector3f& min_bounds, const Eigen::Vector3f& max_bounds,
                                        std::vector<PointRecord>&& points, size_t depth) {
    Node node{};
    node.bounds.min_bounds = min_bounds;
    node.bounds.max_bounds = max_bounds;
    node.is_leaf = 1u;
    node.data.leaf.start_index = 0u;
    node.data.leaf.point_count = static_cast<uint32_t>(points.size());

    const int32_t node_index = static_cast<int32_t>(this->host_nodes_.size());
    this->host_nodes_.push_back(node);
    this->host_leaf_points_.push_back(std::move(points));
    this->host_subtree_sizes_.push_back(this->host_leaf_points_.back().size());

    this->subdivide_leaf(node_index, depth);
    return node_index;
}

/// @brief Split a leaf node into up to eight child nodes when it exceeds capacity.
/// @param node_index Index of the leaf node that may be subdivided.
/// @param depth Depth of the node to guard against excessive recursion.
inline void Octree::subdivide_leaf(int32_t node_index, size_t depth) {
    const Node node_snapshot = this->host_nodes_[static_cast<size_t>(node_index)];
    if (node_snapshot.is_leaf == 0u) {
        return;
    }

    auto& points = this->host_leaf_points_[static_cast<size_t>(node_index)];
    const auto lengths = node_snapshot.bounds.max_bounds - node_snapshot.bounds.min_bounds;
    const float max_axis = std::max({lengths.x(), lengths.y(), lengths.z()});
    if (points.size() <= this->max_points_per_node_ || max_axis <= this->resolution_ || depth >= 32) {
        this->host_subtree_sizes_[static_cast<size_t>(node_index)] = points.size();
        this->host_nodes_[static_cast<size_t>(node_index)].data.leaf.point_count = static_cast<uint32_t>(points.size());
        return;
    }

    const Eigen::Vector3f center = 0.5f * (node_snapshot.bounds.min_bounds + node_snapshot.bounds.max_bounds);

    std::vector<PointRecord> local_points = std::move(points);
    std::array<std::vector<PointRecord>, 8> child_points;
    for (const auto& record : local_points) {
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

    Node& node_ref = this->host_nodes_[static_cast<size_t>(node_index)];
    node_ref.is_leaf = 0u;
    this->host_leaf_points_[static_cast<size_t>(node_index)].clear();
    this->host_subtree_sizes_[static_cast<size_t>(node_index)] = 0;

    for (size_t child = 0; child < 8; ++child) {
        node_ref.data.children[child] = -1;
    }

    for (size_t child = 0; child < child_points.size(); ++child) {
        if (child_points[child].empty()) {
            continue;
        }

        const auto child_bounds_value = this->child_bounds(node_snapshot, child);
        const int32_t child_index = this->create_host_node(child_bounds_value.min_bounds, child_bounds_value.max_bounds,
                                                           std::move(child_points[child]), depth + 1);
        Node& refreshed_node = this->host_nodes_[static_cast<size_t>(node_index)];
        refreshed_node.data.children[child] = child_index;
        this->host_subtree_sizes_[static_cast<size_t>(node_index)] +=
            this->host_subtree_sizes_[static_cast<size_t>(child_index)];
    }

    // Reacquire the node reference because create_host_node() may reallocate host_nodes_.
    Node& final_node_ref = this->host_nodes_[static_cast<size_t>(node_index)];
    if (this->host_subtree_sizes_[static_cast<size_t>(node_index)] == 0) {
        final_node_ref.is_leaf = 1u;
        this->host_leaf_points_[static_cast<size_t>(node_index)].clear();
        for (size_t child = 0; child < 8; ++child) {
            final_node_ref.data.children[child] = -1;
        }
        final_node_ref.data.leaf.point_count = 0;
    }
}

/// @brief Calculate the bounding box for a child octant of the provided node.
/// @param node Parent node whose child bounds are requested.
/// @param child_index Index of the child (0-7).
/// @return Bounding box describing the child octant.
inline Octree::BoundingBox Octree::child_bounds(const Node& node, size_t child_index) const {
    const Eigen::Vector3f min_bounds = node.bounds.min_bounds;
    const Eigen::Vector3f max_bounds = node.bounds.max_bounds;
    const Eigen::Vector3f center = node.bounds.center();

    BoundingBox result;
    result.min_bounds = min_bounds;
    result.max_bounds = max_bounds;

    result.min_bounds.x() = (child_index & 1) ? center.x() : min_bounds.x();
    result.max_bounds.x() = (child_index & 1) ? max_bounds.x() : center.x();
    result.min_bounds.y() = (child_index & 2) ? center.y() : min_bounds.y();
    result.max_bounds.y() = (child_index & 2) ? max_bounds.y() : center.y();
    result.min_bounds.z() = (child_index & 4) ? center.z() : min_bounds.z();
    result.max_bounds.z() = (child_index & 4) ? max_bounds.z() : center.z();

    return result;
}

/// @brief Recalculate the number of points contained inside the subtree rooted at @p node_index.
/// @param node_index Index of the subtree's root node.
inline void Octree::recompute_subtree_size(int32_t node_index) {
    Node& node = this->host_nodes_[static_cast<size_t>(node_index)];
    if (node.is_leaf) {
        const size_t count = this->host_leaf_points_[static_cast<size_t>(node_index)].size();
        this->host_subtree_sizes_[static_cast<size_t>(node_index)] = count;
        node.data.leaf.point_count = static_cast<uint32_t>(count);
        return;
    }

    size_t total = 0;
    for (size_t child = 0; child < 8; ++child) {
        const int32_t child_index = node.data.children[child];
        if (child_index >= 0) {
            total += this->host_subtree_sizes_[static_cast<size_t>(child_index)];
        }
    }

    this->host_subtree_sizes_[static_cast<size_t>(node_index)] = total;
    if (total == 0) {
        node.is_leaf = 1u;
        for (size_t child = 0; child < 8; ++child) {
            node.data.children[child] = -1;
        }
        this->host_leaf_points_[static_cast<size_t>(node_index)].clear();
        node.data.leaf.point_count = 0;
    }
}

SYCL_EXTERNAL float Octree::distance_to_aabb(const Eigen::Vector3f& min_bounds, const Eigen::Vector3f& max_bounds,
                                             const Eigen::Vector3f& point) {
    // Clamp the query position against the bounding box extents along each axis.
    const float dx = sycl::fmax(0.0f, sycl::fmax(min_bounds.x() - point.x(), point.x() - max_bounds.x()));
    const float dy = sycl::fmax(0.0f, sycl::fmax(min_bounds.y() - point.y(), point.y() - max_bounds.y()));
    const float dz = sycl::fmax(0.0f, sycl::fmax(min_bounds.z() - point.z(), point.z() - max_bounds.z()));
    // Return the squared Euclidean distance from the query point to the box surface.
    return dx * dx + dy * dy + dz * dz;
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
        this->device_dirty_ = false;
        return;
    }

    const size_t node_count = this->host_nodes_.size();
    const size_t point_count = this->total_point_count_;

    // Allocate fresh shared memory buffers for the current host-side tree snapshot.
    this->nodes_ = {node_count, Node{}, *this->queue_.ptr};
    this->device_points_ = {point_count, PointType(), *this->queue_.ptr};
    this->device_point_ids_ = {point_count, 0, *this->queue_.ptr};

    size_t offset = 0;

    for (size_t idx = 0; idx < node_count; ++idx) {
        Node device_node = this->host_nodes_[idx];
        if (device_node.is_leaf) {
            const auto& points = this->host_leaf_points_[idx];
            device_node.data.leaf.start_index = static_cast<uint32_t>(offset);
            device_node.data.leaf.point_count = static_cast<uint32_t>(points.size());
            for (size_t i = 0; i < points.size(); ++i) {
                const auto& record = points[i];
                this->device_points_[offset + i] = record.point;
                this->device_point_ids_[offset + i] = record.id;
            }
            offset += points.size();
        }
        this->nodes_[idx] = device_node;
    }

    this->device_dirty_ = false;
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
inline sycl_utils::events Octree::knn_search_async(const PointCloudShared& queries, const size_t k, KNNResult& result,
                                                   const std::vector<sycl::event>& depends,
                                                   const TransformMatrix& transT) const {
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
        return knn_search_async_impl<1, MAX_STACK_DEPTH>(queries, k, result, depends, transT);
    } else if (k <= 10) {
        return knn_search_async_impl<10, MAX_STACK_DEPTH>(queries, k, result, depends, transT);
    } else if (k <= 20) {
        return knn_search_async_impl<20, MAX_STACK_DEPTH>(queries, k, result, depends, transT);
    } else if (k <= 30) {
        return knn_search_async_impl<30, MAX_STACK_DEPTH>(queries, k, result, depends, transT);
    } else if (k <= 40) {
        return knn_search_async_impl<40, MAX_STACK_DEPTH>(queries, k, result, depends, transT);
    } else if (k <= 50) {
        return knn_search_async_impl<50, MAX_STACK_DEPTH>(queries, k, result, depends, transT);
    } else if (k <= 100) {
        return knn_search_async_impl<100, MAX_STACK_DEPTH>(queries, k, result, depends, transT);
    }

    throw std::runtime_error("Requested neighbour count exceeds the supported maximum");
}

template <size_t MAX_K, size_t MAX_DEPTH>
// Core implementation of the octree kNN search.
inline sycl_utils::events Octree::knn_search_async_impl(const PointCloudShared& queries, size_t k, KNNResult& result,
                                                        const std::vector<sycl::event>& depends,
                                                        const TransformMatrix& transT) const {
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

    auto search_task = [=](sycl::handler& handler) {
        const size_t work_group_size = this->queue_.get_work_group_size();
        const size_t global_size = this->queue_.get_global_size(query_size);

        if (!depends.empty()) {
            handler.depends_on(depends);
        }

        auto indices_ptr = result.indices->data();
        auto distances_ptr = result.distances->data();
        const auto query_points_ptr = queries.points->data();
        const auto nodes_ptr = this->nodes_.data();
        const auto leaf_points_ptr = this->device_points_.data();
        const auto leaf_ids_ptr = this->device_point_ids_.data();
        const auto trans_vec = eigen_utils::to_sycl_vec(transT);

        handler.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t query_idx = item.get_global_id(0);

            if (query_idx >= query_size) {
                return;
            }
            if (target_size == 0) {
                return;
            }

            Eigen::Vector4f query_point;
            transform::kernel::transform_point(query_points_ptr[query_idx], query_point, trans_vec);
            const Eigen::Vector3f query_point_vec(query_point.x(), query_point.y(), query_point.z());

            NodeEntry bestK[MAX_K];  // descending order. head data is largest dist_sq
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
                    const size_t right = left + 1;
                    const size_t largest =
                        right < heap_size && bestK[right].dist_sq > bestK[left].dist_sq ? right : left;

                    if (bestK[idx].dist_sq >= bestK[largest].dist_sq) {
                        break;
                    }
                    heap_swap(idx, largest);
                    idx = largest;
                }
            };

            auto current_worst_dist_sq = [&]() {
                return neighbour_count < k ? std::numeric_limits<float>::infinity() : bestK[0].dist_sq;
            };

            auto push_candidate = [&](float distance_sq, int32_t index) {
                if (neighbour_count < k) {
                    bestK[neighbour_count++] = {index, distance_sq};
                    sift_up(neighbour_count - 1);
                } else if (distance_sq < bestK[0].dist_sq) {
                    bestK[0] = {index, distance_sq};  // overwrite worst result
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

            push_node_to_stack(root_index, distance_to_aabb(nodes_ptr[root_index].bounds.min_bounds,
                                                            nodes_ptr[root_index].bounds.max_bounds, query_point_vec));

            while (stack_size > 0) {
                size_t best_pos = 0;
                float best_dist_sq = stack[0].dist_sq;
                for (size_t i = 1; i < stack_size; ++i) {
                    if (stack[i].dist_sq < best_dist_sq) {
                        best_dist_sq = stack[i].dist_sq;
                        best_pos = i;
                    }
                }

                const int32_t current_node_idx = stack[best_pos].nodeIdx;
                stack[best_pos] = stack[--stack_size];

                const auto worst_dist_sq = current_worst_dist_sq();
                if (best_dist_sq > worst_dist_sq) {
                    continue;
                }

                const Node current_node = nodes_ptr[current_node_idx];
                if (current_node.is_leaf) {
                    const uint32_t leaf_start = current_node.data.leaf.start_index;
                    const uint32_t leaf_count = current_node.data.leaf.point_count;
                    for (uint32_t i = 0; i < leaf_count; ++i) {
                        const auto target_point = leaf_points_ptr[leaf_start + i];
                        const int32_t point_id = leaf_ids_ptr[leaf_start + i];
                        const PointType diff = eigen_utils::subtract<4, 1>(query_point, target_point);
                        const float dist_sq = eigen_utils::dot<4>(diff, diff);
                        push_candidate(dist_sq, point_id);
                    }
                } else {
                    const auto worst_dist = sycl::sqrt(worst_dist_sq);
                    for (size_t child = 0; child < 8; ++child) {
                        const int32_t child_idx = current_node.data.children[child];
                        if (child_idx < 0) {
                            continue;
                        }
                        const auto min_bounds = nodes_ptr[child_idx].bounds.min_bounds;
                        const auto max_bounds = nodes_ptr[child_idx].bounds.max_bounds;

                        const float dx = sycl::fmax(0.0f, sycl::fmax(min_bounds.x() - query_point_vec.x(),
                                                                     query_point_vec.x() - max_bounds.x()));
                        const float dy = sycl::fmax(0.0f, sycl::fmax(min_bounds.y() - query_point_vec.y(),
                                                                     query_point_vec.y() - max_bounds.y()));
                        const float dz = sycl::fmax(0.0f, sycl::fmax(min_bounds.z() - query_point_vec.z(),
                                                                     query_point_vec.z() - max_bounds.z()));
                        if (dx > worst_dist || dy > worst_dist || dz > worst_dist) {
                            continue;
                        }
                        const float child_dist_sq = sycl::fma(dx, dx, sycl::fma(dy, dy, dz * dz));
                        if (child_dist_sq > worst_dist_sq) {
                            continue;
                        }
                        push_node_to_stack(child_idx, child_dist_sq);
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
