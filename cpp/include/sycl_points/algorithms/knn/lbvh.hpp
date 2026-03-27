#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "sycl_points/algorithms/common/transform.hpp"
#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/algorithms/knn/result.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace knn {

/// @brief Linear BVH built from Morton-code-sorted points (Karras 2012).
///
/// Construction runs on the host:
///   1. Compute a 63-bit Morton code for each point (21 bits per axis).
///   2. Sort points by Morton code.
///   3. Build the binary BVH tree structure using the Karras (2012) parallel
///      algorithm (executed serially on the host here for simplicity).
///   4. Propagate AABBs bottom-up with a recursive post-order traversal.
///   5. Upload the flat node array to a shared USM buffer for GPU access.
///
/// kNN search runs entirely on the GPU via a best-first stack traversal that
/// is identical in spirit to the existing KDTree / Octree kernels.
class LBVH : public KNNBase {
public:
    using Ptr = std::shared_ptr<LBVH>;

    // ------------------------------------------------------------------ Node

    /// @brief A single BVH node — 64 bytes, standard-layout.
    ///
    /// Internal node  : left_idx  = left child index in nodes[]
    ///                  right_idx = right child index in nodes[]
    ///                  is_leaf   = 0
    ///
    /// Leaf node      : left_idx  = original point index in the input cloud
    ///                  right_idx = -1
    ///                  is_leaf   = 1
    ///                  aabb_min == aabb_max == point position
    struct BVHNode {
        Eigen::Vector3f aabb_min;  ///< 12 bytes
        int32_t left_idx;          ///<  4 bytes
        Eigen::Vector3f aabb_max;  ///< 12 bytes
        int32_t right_idx;         ///<  4 bytes
        uint32_t is_leaf;          ///<  4 bytes
        uint32_t padding[7];       ///< 28 bytes  → total 64 bytes

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    static_assert(std::is_standard_layout_v<BVHNode>, "BVHNode must be standard-layout");
    static_assert(sizeof(BVHNode) == 64, "BVHNode must be 64 bytes");

    // ----------------------------------------------------------- Construction

    /// @brief Construct an empty LBVH bound to the given queue.
    explicit LBVH(const sycl_utils::DeviceQueue& queue)
        : queue_(queue), root_index_(-1), total_point_count_(0), nodes_(*queue.ptr) {}

    /// @brief Build an LBVH from a point cloud and return a shared pointer.
    /// @param queue  SYCL device queue used for all GPU operations.
    /// @param cloud  Input point cloud (read on the host during construction).
    /// @return       Shared pointer to the constructed LBVH.
    static Ptr build(const sycl_utils::DeviceQueue& queue, const PointCloudShared& cloud);

    // ------------------------------------------------------- KNN search API

    /// @brief Async kNN search dispatching to the right compile-time MAX_K.
    sycl_utils::events knn_search_async(const PointCloudShared& queries, size_t k, KNNResult& result,
                                        const std::vector<sycl::event>& depends = {},
                                        const TransformMatrix& transT = TransformMatrix::Identity()) const override;

    /// @brief Number of points stored in the tree.
    [[nodiscard]] size_t size() const { return total_point_count_; }

private:
    // ------------------------------------------------- Traversal stack entry

    /// @brief Entry used in both the traversal stack and the k-best heap.
    /// @note  When stored in the k-best heap, `node_idx` holds a point index.
    struct NodeEntry {
        int32_t node_idx;
        float dist_sq;
    };

    // ------------------------------------------------ Host-side construction

    void build_from_cloud(const PointCloudShared& cloud);

    /// @brief Recursively compute AABBs from leaves to root (post-order DFS).
    static void compute_aabb_recursive(std::vector<BVHNode>& nodes, int idx);

    // ---------------------------------------------- Morton-code helpers

    /// @brief Interleave the lower 21 bits of @p v into every third bit.
    static inline uint64_t expand_bits_21(uint64_t v) noexcept {
        v &= 0x1fffffULL;
        v = (v | (v << 32)) & 0x1f00000000ffffULL;
        v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
        v = (v | (v << 8))  & 0x100f00f00f00f00fULL;
        v = (v | (v << 4))  & 0x10c30c30c30c30c3ULL;
        v = (v | (v << 2))  & 0x1249249249249249ULL;
        return v;
    }

    /// @brief Compute a 63-bit Morton code from normalised [0,1] coordinates.
    static inline uint64_t morton3d(float nx, float ny, float nz) noexcept {
        nx = std::clamp(nx, 0.0f, 1.0f);
        ny = std::clamp(ny, 0.0f, 1.0f);
        nz = std::clamp(nz, 0.0f, 1.0f);
        const uint64_t xi = static_cast<uint64_t>(nx * 2097151.0f);  // 2^21 - 1
        const uint64_t yi = static_cast<uint64_t>(ny * 2097151.0f);
        const uint64_t zi = static_cast<uint64_t>(nz * 2097151.0f);
        return expand_bits_21(xi) | (expand_bits_21(yi) << 1) | (expand_bits_21(zi) << 2);
    }

    // ------------------------------------------- Karras (2012) BVH builders

    /// @brief Number of common prefix bits between sorted codes[i] and codes[j].
    /// @return -1 when j is out of range; uses index as tiebreaker for duplicates.
    static inline int delta(const uint64_t* codes, int n, int i, int j) noexcept {
        if (j < 0 || j >= n) return -1;
        if (codes[i] == codes[j]) {
            // Tiebreak by index position so all deltas remain unique.
            const uint32_t idx_xor = static_cast<uint32_t>(i) ^ static_cast<uint32_t>(j);
            return 63 + static_cast<int>(__builtin_clz(idx_xor));
        }
        return static_cast<int>(__builtin_clzll(codes[i] ^ codes[j]));
    }

    /// @brief Determine the leaf range [first, last] covered by internal node i.
    static inline std::pair<int, int> determine_range(const uint64_t* codes, int n, int i) noexcept {
        const int d = (delta(codes, n, i, i + 1) - delta(codes, n, i, i - 1)) > 0 ? 1 : -1;
        const int delta_min = delta(codes, n, i, i - d);

        // Exponential search for the upper bound of the range length.
        int l_max = 2;
        while (delta(codes, n, i, i + l_max * d) > delta_min) l_max <<= 1;

        // Binary search to find the exact range end.
        int l = 0;
        for (int t = l_max >> 1; t >= 1; t >>= 1) {
            if (delta(codes, n, i, i + (l + t) * d) > delta_min) l += t;
        }

        const int j = i + l * d;
        return {std::min(i, j), std::max(i, j)};
    }

    /// @brief Find the Morton-prefix split position within [first, last].
    static inline int find_split(const uint64_t* codes, int n, int first, int last) noexcept {
        const int delta_node = delta(codes, n, first, last);
        int split = first;
        int step = last - first;

        do {
            step = (step + 1) / 2;  // ceil division
            const int new_split = split + step;
            if (new_split < last && delta(codes, n, first, new_split) > delta_node) {
                split = new_split;
            }
        } while (step > 1);

        return split;
    }

    // ---------------------------------------------------- kNN implementation

    template <size_t MAX_K, size_t MAX_DEPTH>
    sycl_utils::events knn_search_async_impl(const PointCloudShared& queries, size_t k, KNNResult& result,
                                             const std::vector<sycl::event>& depends,
                                             const TransformMatrix& transT) const;

    // ---------------------------------------------------------------- Fields

    sycl_utils::DeviceQueue queue_;
    int32_t root_index_;
    size_t total_point_count_;
    mutable shared_vector<BVHNode> nodes_;  ///< Flat BVH node array on shared USM memory.
};

// ============================================================================
//  Inline / template implementations
// ============================================================================

inline LBVH::Ptr LBVH::build(const sycl_utils::DeviceQueue& queue, const PointCloudShared& cloud) {
    if (!queue.ptr) throw std::runtime_error("[LBVH::build] queue is not initialised");
    auto tree = std::make_shared<LBVH>(queue);
    tree->build_from_cloud(cloud);
    return tree;
}

// ----------------------------------------------------------------------------
//  AABB propagation
// ----------------------------------------------------------------------------

inline void LBVH::compute_aabb_recursive(std::vector<BVHNode>& nodes, int idx) {
    BVHNode& node = nodes[static_cast<size_t>(idx)];
    if (node.is_leaf) return;

    compute_aabb_recursive(nodes, node.left_idx);
    compute_aabb_recursive(nodes, node.right_idx);

    const BVHNode& l = nodes[static_cast<size_t>(node.left_idx)];
    const BVHNode& r = nodes[static_cast<size_t>(node.right_idx)];
    node.aabb_min = l.aabb_min.cwiseMin(r.aabb_min);
    node.aabb_max = l.aabb_max.cwiseMax(r.aabb_max);
}

// ----------------------------------------------------------------------------
//  Host-side tree construction
// ----------------------------------------------------------------------------

inline void LBVH::build_from_cloud(const PointCloudShared& cloud) {
    if (!cloud.points || cloud.points->empty()) {
        root_index_ = -1;
        total_point_count_ = 0;
        nodes_.clear();
        return;
    }

    const size_t n = cloud.points->size();
    total_point_count_ = n;

    // ---- 1. Bounding box -----------------------------------------------
    Eigen::Vector3f bbox_min = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector3f bbox_max = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < n; ++i) {
        const auto& pt = (*cloud.points)[i];
        const Eigen::Vector3f xyz(pt.x(), pt.y(), pt.z());
        bbox_min = bbox_min.cwiseMin(xyz);
        bbox_max = bbox_max.cwiseMax(xyz);
    }

    // Expand slightly so boundary points are strictly inside [0,1].
    const float eps = std::max(1e-6f, (bbox_max - bbox_min).maxCoeff() * 1e-6f);
    bbox_min.array() -= eps;
    bbox_max.array() += eps;
    const Eigen::Vector3f extent = bbox_max - bbox_min;

    // ---- 2. Morton codes -----------------------------------------------
    std::vector<uint64_t> codes(n);
    std::vector<int32_t> sorted_indices(n);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

    for (size_t i = 0; i < n; ++i) {
        const auto& pt = (*cloud.points)[i];
        const float nx = std::clamp((pt.x() - bbox_min.x()) / extent.x(), 0.0f, 1.0f);
        const float ny = std::clamp((pt.y() - bbox_min.y()) / extent.y(), 0.0f, 1.0f);
        const float nz = std::clamp((pt.z() - bbox_min.z()) / extent.z(), 0.0f, 1.0f);
        codes[i] = morton3d(nx, ny, nz);
    }

    // ---- 3. Sort by Morton code ----------------------------------------
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](int32_t a, int32_t b) { return codes[a] < codes[b]; });

    std::vector<uint64_t> sorted_codes(n);
    for (size_t i = 0; i < n; ++i) sorted_codes[i] = codes[sorted_indices[i]];

    // ---- 4. Allocate node array ----------------------------------------
    // Layout: internal nodes [0, n-2], leaf nodes [n-1, 2n-2].
    // For n == 1 there are no internal nodes; the single leaf is the root.
    const size_t num_nodes = (n > 1) ? (2 * n - 1) : 1;
    std::vector<BVHNode> host_nodes(num_nodes);

    // ---- 5. Initialise leaf nodes --------------------------------------
    for (size_t i = 0; i < n; ++i) {
        const int32_t node_idx = static_cast<int32_t>(n) - 1 + static_cast<int32_t>(i);
        const int32_t pt_idx = sorted_indices[i];
        const auto& pt = (*cloud.points)[static_cast<size_t>(pt_idx)];

        BVHNode& node = host_nodes[static_cast<size_t>(node_idx)];
        node.aabb_min = Eigen::Vector3f(pt.x(), pt.y(), pt.z());
        node.aabb_max = node.aabb_min;
        node.left_idx = pt_idx;  // original point index returned by kNN
        node.right_idx = -1;
        node.is_leaf = 1u;
    }

    // Handle the degenerate single-point case.
    if (n == 1) {
        root_index_ = 0;
        nodes_ = {1, BVHNode{}, *queue_.ptr};
        nodes_[0] = host_nodes[0];
        return;
    }

    // ---- 6. Build internal nodes (Karras 2012) -------------------------
    const int ni = static_cast<int>(n);
    const uint64_t* sc = sorted_codes.data();
    std::vector<int32_t> parent(num_nodes, -1);

    for (int i = 0; i < ni - 1; ++i) {
        const auto [first, last] = determine_range(sc, ni, i);
        const int gamma = find_split(sc, ni, first, last);

        // Map split result to node indices:
        //   if γ == first  → left  is leaf node (n-1+γ)
        //   if γ+1 == last → right is leaf node (n-1+γ+1)
        const int left_child  = (gamma     == first) ? (ni - 1 + gamma)     : gamma;
        const int right_child = (gamma + 1 == last)  ? (ni - 1 + gamma + 1) : (gamma + 1);

        BVHNode& node = host_nodes[static_cast<size_t>(i)];
        node.left_idx  = left_child;
        node.right_idx = right_child;
        node.is_leaf   = 0u;

        parent[static_cast<size_t>(left_child)]  = i;
        parent[static_cast<size_t>(right_child)] = i;
    }

    // ---- 7. Locate root (the internal node with no parent) -------------
    root_index_ = -1;
    for (int i = 0; i < ni - 1; ++i) {
        if (parent[static_cast<size_t>(i)] == -1) {
            root_index_ = i;
            break;
        }
    }
    if (root_index_ < 0) {
        throw std::runtime_error("[LBVH::build_from_cloud] failed to find root node");
    }

    // ---- 8. Propagate AABBs from leaves to root ------------------------
    compute_aabb_recursive(host_nodes, root_index_);

    // ---- 9. Upload to device -------------------------------------------
    nodes_ = {num_nodes, BVHNode{}, *queue_.ptr};
    for (size_t i = 0; i < num_nodes; ++i) {
        nodes_[i] = host_nodes[i];
    }
}

// ----------------------------------------------------------------------------
//  kNN dispatch
// ----------------------------------------------------------------------------

inline sycl_utils::events LBVH::knn_search_async(const PointCloudShared& queries, size_t k, KNNResult& result,
                                                  const std::vector<sycl::event>& depends,
                                                  const TransformMatrix& transT) const {
    // BVH depth is O(log n), so MAX_DEPTH=64 is generous.
    constexpr size_t MAX_DEPTH = 64;

    if (k == 0) {
        const size_t query_size = queries.points ? queries.points->size() : 0;
        if (!result.indices || !result.distances) result.allocate(queue_, query_size, 0);
        else result.resize(query_size, 0);
        return sycl_utils::events{};
    }

    if (k == 1)    return knn_search_async_impl<1,   MAX_DEPTH>(queries, k, result, depends, transT);
    if (k <= 10)   return knn_search_async_impl<10,  MAX_DEPTH>(queries, k, result, depends, transT);
    if (k <= 20)   return knn_search_async_impl<20,  MAX_DEPTH>(queries, k, result, depends, transT);
    if (k <= 30)   return knn_search_async_impl<30,  MAX_DEPTH>(queries, k, result, depends, transT);
    if (k <= 40)   return knn_search_async_impl<40,  MAX_DEPTH>(queries, k, result, depends, transT);
    if (k <= 50)   return knn_search_async_impl<50,  MAX_DEPTH>(queries, k, result, depends, transT);
    if (k <= 100)  return knn_search_async_impl<100, MAX_DEPTH>(queries, k, result, depends, transT);

    throw std::runtime_error("[LBVH::knn_search_async] k exceeds the supported maximum (100)");
}

// ----------------------------------------------------------------------------
//  kNN kernel implementation
// ----------------------------------------------------------------------------

template <size_t MAX_K, size_t MAX_DEPTH>
inline sycl_utils::events LBVH::knn_search_async_impl(const PointCloudShared& queries, size_t k, KNNResult& result,
                                                       const std::vector<sycl::event>& depends,
                                                       const TransformMatrix& transT) const {
    static_assert(MAX_K > 0, "MAX_K must be positive");
    static_assert(MAX_DEPTH > 0, "MAX_DEPTH must be positive");

    if (!queue_.ptr) throw std::runtime_error("[LBVH::knn_search_async_impl] queue not initialised");
    if (!queries.points) throw std::runtime_error("[LBVH::knn_search_async_impl] query cloud not initialised");
    if (k > MAX_K) throw std::runtime_error("[LBVH::knn_search_async_impl] k exceeds compile-time MAX_K");

    const size_t query_size  = queries.points->size();
    const size_t target_size = total_point_count_;
    const size_t node_count  = nodes_.size();
    const int32_t root_idx   = root_index_;

    if (!result.indices || !result.distances) result.allocate(queue_, query_size, k);
    else result.resize(query_size, k);

    if (target_size > 0 && node_count == 0) {
        throw std::runtime_error("[LBVH::knn_search_async_impl] tree not built");
    }

    auto search_task = [&](sycl::handler& handler) {
        const size_t wg_size    = queue_.get_work_group_size();
        const size_t global_sz  = queue_.get_global_size(query_size);

        if (!depends.empty()) handler.depends_on(depends);

        auto* indices_ptr   = result.indices->data();
        auto* distances_ptr = result.distances->data();
        const auto* q_pts   = queries.points->data();
        const auto* nodes_ptr = nodes_.data();
        const auto trans_vec  = eigen_utils::to_sycl_vec(transT);

        handler.parallel_for(sycl::nd_range<1>(global_sz, wg_size), [=](sycl::nd_item<1> item) {
            const size_t qi = item.get_global_id(0);
            if (qi >= query_size || target_size == 0) return;

            // ---- Transform query point --------------------------------
            Eigen::Vector4f qp;
            transform::kernel::transform_point(q_pts[qi], qp, trans_vec);
            const float qx = qp.x(), qy = qp.y(), qz = qp.z();

            // ---- Max-heap for k best candidates ----------------------
            // Largest dist_sq sits at index 0 (heap root).
            NodeEntry best[MAX_K];
            for (size_t i = 0; i < MAX_K; ++i) best[i] = {-1, std::numeric_limits<float>::max()};
            size_t found = 0;

            auto heap_swap = [&](size_t a, size_t b) {
                NodeEntry tmp = best[a]; best[a] = best[b]; best[b] = tmp;
            };
            auto sift_up = [&](size_t idx) {
                while (idx > 0) {
                    const size_t par = (idx - 1) / 2;
                    if (best[par].dist_sq >= best[idx].dist_sq) break;
                    heap_swap(par, idx);
                    idx = par;
                }
            };
            auto sift_down = [&](size_t idx, size_t sz) {
                while (true) {
                    const size_t left = idx * 2 + 1;
                    if (left >= sz) break;
                    const size_t right   = left + 1;
                    const size_t largest = (right < sz && best[right].dist_sq > best[left].dist_sq) ? right : left;
                    if (best[idx].dist_sq >= best[largest].dist_sq) break;
                    heap_swap(idx, largest);
                    idx = largest;
                }
            };
            auto worst_dist_sq = [&]() -> float {
                return (found < k) ? std::numeric_limits<float>::infinity() : best[0].dist_sq;
            };
            // Push a point candidate into the k-best max-heap.
            auto push_candidate = [&](float d2, int32_t pt_idx) {
                if (found < k) {
                    best[found++] = {pt_idx, d2};
                    sift_up(found - 1);
                } else if (d2 < best[0].dist_sq) {
                    best[0] = {pt_idx, d2};
                    sift_down(0, found);
                }
            };

            // ---- Traversal stack (best-first) -------------------------
            NodeEntry stack[MAX_DEPTH];
            size_t stack_size = 0;

            // Squared distance from point (px,py,pz) to AABB [ax..bx, ay..by, az..bz].
            auto aabb_dist2 = [](float ax, float ay, float az, float bx, float by, float bz,
                                 float px, float py, float pz) -> float {
                const float dx = sycl::fmax(0.0f, sycl::fmax(ax - px, px - bx));
                const float dy = sycl::fmax(0.0f, sycl::fmax(ay - py, py - by));
                const float dz = sycl::fmax(0.0f, sycl::fmax(az - pz, pz - bz));
                return sycl::fma(dx, dx, sycl::fma(dy, dy, dz * dz));
            };

            // Push a node onto the traversal stack.
            // When the stack is full, evict the farthest entry.
            auto push_stack = [&](int32_t node_idx, float d2) {
                if (stack_size < MAX_DEPTH) {
                    stack[stack_size++] = {node_idx, d2};
                } else {
                    size_t worst_pos = 0;
                    float  worst_d   = stack[0].dist_sq;
                    for (size_t i = 1; i < stack_size; ++i) {
                        if (stack[i].dist_sq > worst_d) { worst_d = stack[i].dist_sq; worst_pos = i; }
                    }
                    if (d2 < worst_d) stack[worst_pos] = {node_idx, d2};
                }
            };

            if (node_count == 0 || root_idx < 0) return;

            // Seed the stack with the root node.
            {
                const auto& root = nodes_ptr[root_idx];
                push_stack(root_idx, aabb_dist2(root.aabb_min.x(), root.aabb_min.y(), root.aabb_min.z(),
                                                root.aabb_max.x(), root.aabb_max.y(), root.aabb_max.z(),
                                                qx, qy, qz));
            }

            // ---- Main traversal loop ----------------------------------
            while (stack_size > 0) {
                // Pop the entry with the smallest AABB distance (best-first).
                size_t best_pos = 0;
                float  best_d   = stack[0].dist_sq;
                for (size_t i = 1; i < stack_size; ++i) {
                    if (stack[i].dist_sq < best_d) { best_d = stack[i].dist_sq; best_pos = i; }
                }
                const int32_t cur_idx = stack[best_pos].node_idx;
                stack[best_pos] = stack[--stack_size];

                // Prune: the closest possible point in this subtree is farther
                // than our current k-th best.
                if (best_d > worst_dist_sq()) continue;

                const BVHNode& node = nodes_ptr[cur_idx];

                if (node.is_leaf) {
                    // Leaf: compute exact squared distance to the stored point.
                    // For leaf nodes aabb_min == aabb_max == point position.
                    const float dx = qx - node.aabb_min.x();
                    const float dy = qy - node.aabb_min.y();
                    const float dz = qz - node.aabb_min.z();
                    const float d2 = sycl::fma(dx, dx, sycl::fma(dy, dy, dz * dz));
                    push_candidate(d2, node.left_idx);  // left_idx = original point index
                } else {
                    // Internal: push children whose AABB is within reach.
                    const float wd = worst_dist_sq();
                    const int32_t children[2] = {node.left_idx, node.right_idx};
                    for (int ci = 0; ci < 2; ++ci) {
                        const int32_t child_idx = children[ci];
                        if (child_idx < 0) continue;
                        const BVHNode& child = nodes_ptr[child_idx];
                        const float cd = aabb_dist2(child.aabb_min.x(), child.aabb_min.y(), child.aabb_min.z(),
                                                    child.aabb_max.x(), child.aabb_max.y(), child.aabb_max.z(),
                                                    qx, qy, qz);
                        if (cd <= wd) push_stack(child_idx, cd);
                    }
                }
            }

            // ---- Sort results ascending (heap-sort) -------------------
            size_t hs = found;
            while (hs > 1) {
                heap_swap(0, hs - 1);
                sift_down(0, --hs);
            }

            // ---- Write output -----------------------------------------
            for (size_t i = 0; i < k; ++i) {
                indices_ptr[qi * k + i]   = best[i].node_idx;  // node_idx holds point index here
                distances_ptr[qi * k + i] = best[i].dist_sq;
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
