#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <sycl/sycl.hpp>
#include <sycl_points/algorithms/knn/result.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>
#include <vector>

namespace sycl_points {

namespace algorithms {

namespace knn {

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
    /// @tparam MaxStackSize Maximum number of nodes kept on the traversal stack.
    /// @return Result container that stores neighbour indices and squared distances.
    template <size_t MaxStackSize = 128>
    [[nodiscard]] KNNResult knn_search(const PointCloudShared& queries, size_t k) const;

    /// @brief Accessor for the resolution that was requested at build time.
    [[nodiscard]] float resolution() const { return resolution_; }

    /// @brief Accessor for the maximum number of points per node.
    [[nodiscard]] size_t max_points_per_node() const { return max_points_per_node_; }

    /// @brief Number of points stored in the Octree.
    [[nodiscard]] size_t size() const { return total_point_count_; }

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
    struct PointRecord {
        PointType point;
        int32_t id;
    };

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

    void build_from_cloud(const PointCloudShared& points);
    int32_t create_host_node(const sycl::float3& min_bounds, const sycl::float3& max_bounds, std::vector<PointRecord>&& points,
                             size_t depth);
    void subdivide_leaf(int32_t node_index, size_t depth);
    void insert_recursive(int32_t node_index, const PointType& point, int32_t id, size_t depth);
    bool remove_recursive(int32_t node_index, const PointType& point, float tolerance_sq);
    bool delete_box_recursive(int32_t node_index, const BoundingBox& region);
    void ensure_root_bounds(const PointType& point);
    BoundingBox child_bounds(const HostNode& node, size_t child_index) const;
    void recompute_subtree_size(int32_t node_index);
    void sync_device_buffers() const;
    static float squared_distance(const PointType& a, const PointType& b);
    SYCL_EXTERNAL static sycl::float3 axis_lengths(const sycl::float3& min_bounds, const sycl::float3& max_bounds);
    SYCL_EXTERNAL static float distance_to_aabb(const sycl::float3& min_bounds, const sycl::float3& max_bounds,
                                                const sycl::float3& point);

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
};

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

inline void Octree::build_from_cloud(const PointCloudShared& points) {
    if (!queue_.ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }

    host_nodes_.clear();
    total_point_count_ = 0;
    root_index_ = -1;
    device_dirty_ = true;
    snapshot_ids_.clear();

    if (!points.points) {
        throw std::runtime_error("Point cloud is not initialised");
    }

    const size_t point_count = points.points->size();
    if (point_count == 0) {
        bbox_min_ = sycl::float3(0.0f, 0.0f, 0.0f);
        bbox_max_ = sycl::float3(0.0f, 0.0f, 0.0f);
        nodes_.clear();
        device_points_.clear();
        device_point_ids_.clear();
        snapshot_ids_.clear();
        return;
    }

    bbox_min_ = sycl::float3(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity());
    bbox_max_ = sycl::float3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                              std::numeric_limits<float>::lowest());

    std::vector<PointRecord> records(point_count);
    for (size_t i = 0; i < point_count; ++i) {
        const auto point = (*points.points)[i];
        bbox_min_.x() = std::min(bbox_min_.x(), point.x());
        bbox_min_.y() = std::min(bbox_min_.y(), point.y());
        bbox_min_.z() = std::min(bbox_min_.z(), point.z());
        bbox_max_.x() = std::max(bbox_max_.x(), point.x());
        bbox_max_.y() = std::max(bbox_max_.y(), point.y());
        bbox_max_.z() = std::max(bbox_max_.z(), point.z());

        records[i] = PointRecord{point, static_cast<int32_t>(i)};
    }

    const float epsilon = std::max(1e-5f, resolution_ * 0.5f);
    bbox_min_ -= sycl::float3(epsilon, epsilon, epsilon);
    bbox_max_ += sycl::float3(epsilon, epsilon, epsilon);

    root_index_ = create_host_node(bbox_min_, bbox_max_, std::move(records), 0);
    total_point_count_ = point_count;
    next_point_id_ = static_cast<int32_t>(point_count);
}

inline int32_t Octree::create_host_node(const sycl::float3& min_bounds, const sycl::float3& max_bounds,
                                        std::vector<PointRecord>&& points, size_t depth) {
    HostNode node;
    node.min_bounds = min_bounds;
    node.max_bounds = max_bounds;
    node.points = std::move(points);
    node.subtree_size = node.points.size();

    const int32_t node_index = static_cast<int32_t>(host_nodes_.size());
    host_nodes_.push_back(std::move(node));

    subdivide_leaf(node_index, depth);
    return node_index;
}

inline void Octree::subdivide_leaf(int32_t node_index, size_t depth) {
    HostNode& node = host_nodes_[static_cast<size_t>(node_index)];
    if (!node.is_leaf) {
        return;
    }

    const auto lengths = axis_lengths(node.min_bounds, node.max_bounds);
    const float max_axis = std::max({lengths.x(), lengths.y(), lengths.z()});
    if (node.points.size() <= max_points_per_node_ || max_axis <= resolution_ || depth >= 32) {
        node.subtree_size = node.points.size();
        return;
    }

    const sycl::float3 min_bounds = node.min_bounds;
    const sycl::float3 max_bounds = node.max_bounds;
    const sycl::float3 center((min_bounds.x() + max_bounds.x()) * 0.5f, (min_bounds.y() + max_bounds.y()) * 0.5f,
                              (min_bounds.z() + max_bounds.z()) * 0.5f);

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

        const int32_t child_index = create_host_node(bounds.min_bounds, bounds.max_bounds, std::move(child_points[child]), depth + 1);
        HostNode& updated_node = host_nodes_[static_cast<size_t>(node_index)];
        updated_node.children[child] = child_index;
        updated_node.subtree_size += host_nodes_[static_cast<size_t>(child_index)].subtree_size;
    }

    HostNode& updated_node = host_nodes_[static_cast<size_t>(node_index)];
    if (updated_node.subtree_size == 0) {
        updated_node.is_leaf = true;
        updated_node.children.fill(-1);
    }
}

inline void Octree::insert(const PointType& point) {
    ensure_root_bounds(point);
    const int32_t id = next_point_id_++;
    insert_recursive(root_index_, point, id, 0);
    total_point_count_ += 1;
    device_dirty_ = true;
}

inline void Octree::insert_recursive(int32_t node_index, const PointType& point, int32_t id, size_t depth) {
    HostNode& node = host_nodes_[static_cast<size_t>(node_index)];
    if (node.is_leaf) {
        node.points.push_back(PointRecord{point, id});
        node.subtree_size = node.points.size();
        subdivide_leaf(node_index, depth);
        return;
    }

    const auto bounds = BoundingBox{node.min_bounds, node.max_bounds};
    const sycl::float3 center((bounds.min_bounds.x() + bounds.max_bounds.x()) * 0.5f,
                              (bounds.min_bounds.y() + bounds.max_bounds.y()) * 0.5f,
                              (bounds.min_bounds.z() + bounds.max_bounds.z()) * 0.5f);

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

    const size_t child_index = static_cast<size_t>(octant);
    if (host_nodes_[static_cast<size_t>(node_index)].children[child_index] < 0) {
        std::vector<PointRecord> new_points;
        new_points.push_back(PointRecord{point, id});
        const auto child_bounds_value = child_bounds(host_nodes_[static_cast<size_t>(node_index)], child_index);
        const int32_t new_child = create_host_node(child_bounds_value.min_bounds, child_bounds_value.max_bounds, std::move(new_points), depth + 1);
        host_nodes_[static_cast<size_t>(node_index)].children[child_index] = new_child;
    } else {
        const int32_t next_index = host_nodes_[static_cast<size_t>(node_index)].children[child_index];
        insert_recursive(next_index, point, id, depth + 1);
    }

    recompute_subtree_size(node_index);
}

inline bool Octree::remove(const PointType& point, float tolerance) {
    if (root_index_ < 0) {
        return false;
    }
    const float tolerance_sq = tolerance * tolerance;
    const bool removed = remove_recursive(root_index_, point, tolerance_sq);
    if (removed) {
        total_point_count_ -= 1;
        device_dirty_ = true;
    }
    return removed;
}

inline bool Octree::remove_recursive(int32_t node_index, const PointType& point, float tolerance_sq) {
    HostNode& node = host_nodes_[static_cast<size_t>(node_index)];
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
            if (!host_nodes_[static_cast<size_t>(child_index)].subtree_size) {
                continue;
            }
            const auto& child_node = host_nodes_[static_cast<size_t>(child_index)];
            const BoundingBox bounds{child_node.min_bounds, child_node.max_bounds};
            const float aabb_distance = distance_to_aabb(bounds.min_bounds, bounds.max_bounds,
                                                        sycl::float3(point.x(), point.y(), point.z()));
            if (aabb_distance > tolerance_sq) {
                continue;
            }
            if (remove_recursive(child_index, point, tolerance_sq)) {
                removed = true;
            }
            if (host_nodes_[static_cast<size_t>(child_index)].subtree_size == 0) {
                node.children[child] = -1;
            }
            if (removed) {
                break;
            }
        }
        recompute_subtree_size(node_index);
    }
    return removed;
}

inline size_t Octree::delete_box(const BoundingBox& region) {
    if (root_index_ < 0) {
        return 0;
    }
    const size_t before = total_point_count_;
    if (delete_box_recursive(root_index_, region)) {
        // Root cleared completely.
    }
    if (root_index_ >= 0) {
        host_nodes_[static_cast<size_t>(root_index_)].subtree_size = total_point_count_;
    }
    const size_t removed = before - total_point_count_;
    if (removed > 0) {
        device_dirty_ = true;
    }
    return removed;
}

inline bool Octree::delete_box_recursive(int32_t node_index, const BoundingBox& region) {
    HostNode& node = host_nodes_[static_cast<size_t>(node_index)];
    const BoundingBox node_bounds{node.min_bounds, node.max_bounds};
    if (!region.intersects(node_bounds)) {
        return false;
    }

    if (region.fully_contains(node_bounds)) {
        total_point_count_ -= node.subtree_size;
        node.points.clear();
        node.points.shrink_to_fit();
        node.children.fill(-1);
        node.is_leaf = true;
        node.subtree_size = 0;
        return true;
    }

    if (node.is_leaf) {
        auto& pts = node.points;
        auto it = std::remove_if(pts.begin(), pts.end(), [&](const PointRecord& record) {
            return region.contains(record.point);
        });
        if (it != pts.end()) {
            total_point_count_ -= static_cast<size_t>(std::distance(it, pts.end()));
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
        if (delete_box_recursive(child_idx, region)) {
            node.children[child] = -1;
        }
    }

    recompute_subtree_size(node_index);
    return node.subtree_size == 0;
}

inline std::vector<int32_t> Octree::radius_search(const PointType& query, float radius) const {
    sync_device_buffers();
    if (root_index_ < 0 || total_point_count_ == 0) {
        return {};
    }

    const float radius_sq = radius * radius;
    std::vector<int32_t> result;
    result.reserve(16);

    std::vector<int32_t> stack;
    stack.push_back(root_index_);

    while (!stack.empty()) {
        const int32_t node_index = stack.back();
        stack.pop_back();
        const HostNode& node = host_nodes_[static_cast<size_t>(node_index)];
        const BoundingBox bounds{node.min_bounds, node.max_bounds};
        const float dist_sq = distance_to_aabb(bounds.min_bounds, bounds.max_bounds,
                                               sycl::float3(query.x(), query.y(), query.z()));
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
                if (child >= 0 && host_nodes_[static_cast<size_t>(child)].subtree_size > 0) {
                    stack.push_back(child);
                }
            }
        }
    }

    return result;
}

inline void Octree::ensure_root_bounds(const PointType& point) {
    const sycl::float3 point_vec(point.x(), point.y(), point.z());
    if (root_index_ < 0) {
        bbox_min_ = point_vec - sycl::float3(resolution_, resolution_, resolution_);
        bbox_max_ = point_vec + sycl::float3(resolution_, resolution_, resolution_);
        std::vector<PointRecord> pts;
        root_index_ = create_host_node(bbox_min_, bbox_max_, std::move(pts), 0);
        return;
    }

    HostNode& root = host_nodes_[static_cast<size_t>(root_index_)];
    BoundingBox root_bounds{root.min_bounds, root.max_bounds};
    if (root_bounds.contains(point_vec)) {
        return;
    }

    sycl::float3 new_min = root_bounds.min_bounds;
    sycl::float3 new_max = root_bounds.max_bounds;
    new_min.x() = std::min(new_min.x(), point_vec.x());
    new_min.y() = std::min(new_min.y(), point_vec.y());
    new_min.z() = std::min(new_min.z(), point_vec.z());
    new_max.x() = std::max(new_max.x(), point_vec.x());
    new_max.y() = std::max(new_max.y(), point_vec.y());
    new_max.z() = std::max(new_max.z(), point_vec.z());

    HostNode new_root;
    new_root.min_bounds = new_min;
    new_root.max_bounds = new_max;
    new_root.is_leaf = false;
    new_root.subtree_size = root.subtree_size;
    new_root.start_index = 0;
    new_root.children.fill(-1);

    const sycl::float3 center((new_min.x() + new_max.x()) * 0.5f, (new_min.y() + new_max.y()) * 0.5f,
                              (new_min.z() + new_max.z()) * 0.5f);
    int octant = 0;
    const sycl::float3 old_center((root.min_bounds.x() + root.max_bounds.x()) * 0.5f,
                                  (root.min_bounds.y() + root.max_bounds.y()) * 0.5f,
                                  (root.min_bounds.z() + root.max_bounds.z()) * 0.5f);
    if (old_center.x() >= center.x()) {
        octant |= 1;
    }
    if (old_center.y() >= center.y()) {
        octant |= 2;
    }
    if (old_center.z() >= center.z()) {
        octant |= 4;
    }

    const int32_t new_root_index = static_cast<int32_t>(host_nodes_.size());
    host_nodes_.push_back(new_root);
    host_nodes_[static_cast<size_t>(new_root_index)].children[static_cast<size_t>(octant)] = root_index_;
    root_index_ = new_root_index;
    host_nodes_[static_cast<size_t>(root_index_)].subtree_size = total_point_count_;
    bbox_min_ = new_min;
    bbox_max_ = new_max;
}

inline Octree::BoundingBox Octree::child_bounds(const HostNode& node, size_t child_index) const {
    const sycl::float3 center((node.min_bounds.x() + node.max_bounds.x()) * 0.5f,
                              (node.min_bounds.y() + node.max_bounds.y()) * 0.5f,
                              (node.min_bounds.z() + node.max_bounds.z()) * 0.5f);

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

inline void Octree::recompute_subtree_size(int32_t node_index) {
    HostNode& node = host_nodes_[static_cast<size_t>(node_index)];
    if (node.is_leaf) {
        node.subtree_size = node.points.size();
        return;
    }

    size_t total = 0;
    for (int32_t child : node.children) {
        if (child >= 0) {
            total += host_nodes_[static_cast<size_t>(child)].subtree_size;
        }
    }

    node.subtree_size = total;
    if (total == 0) {
        node.is_leaf = true;
        node.children.fill(-1);
        node.points.clear();
    }
}

inline void Octree::sync_device_buffers() const {
    if (!device_dirty_) {
        return;
    }

    if (root_index_ < 0 || total_point_count_ == 0) {
        nodes_.clear();
        device_points_.clear();
        device_point_ids_.clear();
        snapshot_ids_.clear();
        device_dirty_ = false;
        return;
    }

    const size_t node_count = host_nodes_.size();
    const size_t point_count = total_point_count_;

    shared_allocator<Node> node_alloc(*queue_.ptr);
    nodes_ = shared_vector<Node>(node_count, Node{}, node_alloc);

    shared_allocator<PointType> point_alloc(*queue_.ptr);
    device_points_ = shared_vector<PointType>(point_count, PointType(), point_alloc);

    shared_allocator<int32_t> id_alloc(*queue_.ptr);
    device_point_ids_ = shared_vector<int32_t>(point_count, 0, id_alloc);

    snapshot_ids_.clear();
    snapshot_ids_.reserve(point_count);

    size_t offset = 0;

    for (size_t idx = 0; idx < node_count; ++idx) {
        HostNode& host_node = host_nodes_[idx];
        Node device_node{};
        device_node.min_bounds = host_node.min_bounds;
        device_node.max_bounds = host_node.max_bounds;
        device_node.is_leaf = host_node.is_leaf ? 1 : 0;
        std::copy(host_node.children.begin(), host_node.children.end(), std::begin(device_node.children));
        if (host_node.is_leaf) {
            device_node.start_index = static_cast<uint32_t>(offset);
            device_node.point_count = static_cast<uint32_t>(host_node.points.size());
            host_node.start_index = offset;
            for (size_t i = 0; i < host_node.points.size(); ++i) {
                const auto& record = host_node.points[i];
                device_points_[offset + i] = record.point;
                device_point_ids_[offset + i] = record.id;
                snapshot_ids_.push_back(record.id);
            }
            offset += host_node.points.size();
        } else {
            device_node.start_index = 0;
            device_node.point_count = 0;
            host_node.start_index = 0;
        }
        nodes_[idx] = device_node;
    }

    device_dirty_ = false;
}

inline std::vector<PointType> Octree::snapshot_points() const {
    sync_device_buffers();
    return std::vector<PointType>(device_points_.begin(), device_points_.end());
}

inline std::vector<int32_t> Octree::snapshot_ids() const {
    sync_device_buffers();
    return snapshot_ids_;
}

inline float Octree::squared_distance(const PointType& a, const PointType& b) {
    const PointType diff = eigen_utils::subtract<4, 1>(a, b);
    return eigen_utils::dot<4>(diff, diff);
}

SYCL_EXTERNAL inline sycl::float3 Octree::axis_lengths(const sycl::float3& min_bounds, const sycl::float3& max_bounds) {
    return max_bounds - min_bounds;
}

SYCL_EXTERNAL float Octree::distance_to_aabb(const sycl::float3& min_bounds, const sycl::float3& max_bounds,
                                             const sycl::float3& point) {
    const float dx = (point.x() < min_bounds.x())   ? (min_bounds.x() - point.x())
                     : (point.x() > max_bounds.x()) ? (point.x() - max_bounds.x())
                                                    : 0.0f;
    const float dy = (point.y() < min_bounds.y())   ? (min_bounds.y() - point.y())
                     : (point.y() > max_bounds.y()) ? (point.y() - max_bounds.y())
                                                    : 0.0f;
    const float dz = (point.z() < min_bounds.z())   ? (min_bounds.z() - point.z())
                     : (point.z() > max_bounds.z()) ? (point.z() - max_bounds.z())
                                                    : 0.0f;
    return dx * dx + dy * dy + dz * dz;
}

inline Octree::Ptr Octree::build(const sycl_utils::DeviceQueue& queue, const PointCloudShared& points, float resolution,
                                 size_t max_points_per_node) {
    auto tree = std::make_shared<Octree>(queue, resolution, max_points_per_node);
    if (!queue.ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }

    tree->build_from_cloud(points);
    return tree;
}

template <size_t MaxStackSize>
inline KNNResult Octree::knn_search(const PointCloudShared& queries, size_t k) const {
    static_assert(MaxStackSize > 0, "MaxStackSize must be greater than zero");
    if (!queue_.ptr) {
        throw std::runtime_error("Octree queue is not initialised");
    }
    if (!queries.points) {
        throw std::runtime_error("Query cloud is not initialised");
    }

    sync_device_buffers();

    const size_t target_size = total_point_count_;
    const size_t node_count = nodes_.size();
    const int32_t root_index = root_index_;

    KNNResult result;
    const size_t query_size = queries.points->size();
    result.allocate(queue_, query_size, k);

    if (target_size > 0 && (node_count == 0 || device_points_.empty())) {
        throw std::runtime_error("Octree structure has not been initialized");
    }

    auto event = queue_.ptr->submit([=](sycl::handler& handler) {
        const size_t work_group_size = this->queue_.get_work_group_size();
        const size_t global_size = this->queue_.get_global_size(query_size);

        auto indices_ptr = result.indices->data();
        auto distances_ptr = result.distances->data();
        const auto query_points_ptr = queries.points->data();
        const auto nodes_ptr = nodes_.data();
        const auto leaf_points_ptr = device_points_.data();
        const auto leaf_ids_ptr = device_point_ids_.data();

        handler.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t query_idx = item.get_global_id(0);

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

            constexpr size_t kMaxStackSize = MaxStackSize;
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

            if (node_count == 0 || root_index < 0) {
                return;
            }

            push_node(root_index,
                      distance_to_aabb(nodes_ptr[root_index].min_bounds, nodes_ptr[root_index].max_bounds,
                                       query_point_vec));

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
                        const auto target_point = leaf_points_ptr[node.start_index + i];
                        const int32_t point_id = leaf_ids_ptr[node.start_index + i];
                        const PointType diff = eigen_utils::subtract<4, 1>(query_point, target_point);
                        const float dist_sq = eigen_utils::dot<4>(diff, diff);
                        push_candidate(dist_sq, point_id);
                    }
                } else {
                    for (size_t child = 0; child < 8; ++child) {
                        const int32_t child_idx = node.children[child];
                        if (child_idx < 0) {
                            continue;
                        }
                        const float child_dist = distance_to_aabb(nodes_ptr[child_idx].min_bounds,
                                                                  nodes_ptr[child_idx].max_bounds, query_point_vec);
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

}  // namespace knn

}  // namespace algorithms

}  // namespace sycl_points
