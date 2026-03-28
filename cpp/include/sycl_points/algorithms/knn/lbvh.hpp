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

    /// @brief A single BVH node — 32 bytes, standard-layout.
    ///
    /// Internal node  : left_idx  = left child index in nodes[]
    ///                  right_idx = right child index in nodes[]  (>= 0)
    ///
    /// Leaf node      : left_idx  = original point index in the input cloud
    ///                  right_idx = -1  (leaf sentinel)
    ///                  aabb_min[0..2] == aabb_max[0..2] == point position
    struct BVHNode {
        float   aabb_min[3];  ///< 12 bytes
        int32_t left_idx;     ///<  4 bytes
        float   aabb_max[3];  ///< 12 bytes
        int32_t right_idx;    ///<  4 bytes  (< 0 → leaf)
        // Total: 32 bytes, no padding
    };

    static_assert(std::is_standard_layout_v<BVHNode>, "BVHNode must be standard-layout");
    static_assert(sizeof(BVHNode) == 32, "BVHNode must be 32 bytes");

    // ----------------------------------------------------------- Construction

    /// @brief Construct an empty LBVH bound to the given queue.
    explicit LBVH(const sycl_utils::DeviceQueue& queue)
        : queue_(queue), root_index_(-1), total_point_count_(0) {}

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
    mutable shared_vector_ptr<BVHNode> nodes_;  ///< Flat BVH node array on shared USM memory.

    // Reusable build-time USM buffers — allocated once, grown as needed.
    shared_vector_ptr<uint64_t> codes_buf_;    ///< Morton code per original point index.
    shared_vector_ptr<int32_t>  si_buf_;       ///< Sorted indices (sorted by Morton code).
    shared_vector_ptr<uint64_t> sc_buf_;       ///< Morton codes in sorted order.
    shared_vector_ptr<int32_t>  parent_buf_;   ///< Parent index for each BVH node.
    shared_vector_ptr<int32_t>  aabb_flags_;   ///< Atomic visit counter for AABB propagation.
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
//  Host-side tree construction
// ----------------------------------------------------------------------------

inline void LBVH::build_from_cloud(const PointCloudShared& cloud) {
    if (!cloud.points || cloud.points->empty()) {
        root_index_ = -1;
        total_point_count_ = 0;
        if (nodes_) nodes_->clear();
        return;
    }

    const size_t n = cloud.points->size();
    total_point_count_ = n;

    const int ni = static_cast<int>(n);
    const size_t wg_size   = queue_.get_work_group_size();
    const size_t global_n  = queue_.get_global_size(n);

    // ---- 1. Bounding box (CPU sequential — trivial O(n) reduction) -----
    float bbox_min[3] = {std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::infinity()};
    float bbox_max[3] = {std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest()};

    const auto* pts_data = cloud.points->data();
    for (size_t i = 0; i < n; ++i) {
        const auto& pt = pts_data[i];
        bbox_min[0] = std::min(bbox_min[0], pt.x());
        bbox_min[1] = std::min(bbox_min[1], pt.y());
        bbox_min[2] = std::min(bbox_min[2], pt.z());
        bbox_max[0] = std::max(bbox_max[0], pt.x());
        bbox_max[1] = std::max(bbox_max[1], pt.y());
        bbox_max[2] = std::max(bbox_max[2], pt.z());
    }

    // Expand slightly so boundary points are strictly inside [0,1].
    const float ext_max = std::max({bbox_max[0] - bbox_min[0],
                                    bbox_max[1] - bbox_min[1],
                                    bbox_max[2] - bbox_min[2]});
    const float eps = std::max(1e-6f, ext_max * 1e-6f);
    float extent[3];
    for (int i = 0; i < 3; ++i) {
        bbox_min[i] -= eps;
        bbox_max[i] += eps;
        extent[i] = bbox_max[i] - bbox_min[i];
    }

    // ---- Reuse shared USM temporaries (grow if needed, no realloc otherwise) --
    // codes_buf_:   Morton code per original point index
    // si_buf_:      sorted original indices (sorted by Morton code on CPU)
    // sc_buf_:      Morton codes in sorted order (for Karras kernels)
    // parent_buf_:  parent index of each BVH node (-1 for root/unset)
    // aabb_flags_:  visit counter for parallel AABB propagation (init later)
    if (!codes_buf_)  codes_buf_  = std::make_shared<shared_vector<uint64_t>>(*queue_.ptr);
    if (!si_buf_)     si_buf_     = std::make_shared<shared_vector<int32_t>> (*queue_.ptr);
    if (!sc_buf_)     sc_buf_     = std::make_shared<shared_vector<uint64_t>>(*queue_.ptr);
    codes_buf_->resize(n);   // values written entirely by kernel — no init needed
    si_buf_->resize(n);      // values written entirely by kernel — no init needed
    sc_buf_->resize(n);      // values written entirely by kernel — no init needed
    const size_t num_nodes = (n > 1) ? (2 * n - 1) : 1;
    if (!parent_buf_) parent_buf_ = std::make_shared<shared_vector<int32_t>>(*queue_.ptr);
    parent_buf_->assign(num_nodes, int32_t{-1});  // must start at -1 for Karras kernel

    // ---- 2. Morton codes + iota (SYCL device kernel) -------------------
    // Both codes_buf_ and si_buf_ are computed in a single dispatch.
    {
        auto* c  = codes_buf_->data();
        auto* si = si_buf_->data();
        const float bx = bbox_min[0], by = bbox_min[1], bz = bbox_min[2];
        const float ex = extent[0],   ey = extent[1],   ez = extent[2];

        queue_.ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_n, wg_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= n) return;
                const auto& pt = pts_data[i];
                const float nx = sycl::clamp((pt.x() - bx) / ex, 0.0f, 1.0f);
                const float ny = sycl::clamp((pt.y() - by) / ey, 0.0f, 1.0f);
                const float nz = sycl::clamp((pt.z() - bz) / ez, 0.0f, 1.0f);
                c[i]  = morton3d(nx, ny, nz);
                si[i] = static_cast<int32_t>(i);
            });
        }).wait_and_throw();  // CPU needs codes_buf_ and si_buf_ for std::sort below
    }

    // ---- 3. Sort sorted_indices by Morton code (CPU std::sort) ---------
    // Shared USM is directly accessible from the CPU, so no copy needed.
    std::sort(si_buf_->begin(), si_buf_->end(),
              [&](int32_t a, int32_t b) { return (*codes_buf_)[a] < (*codes_buf_)[b]; });

    // ---- 3b. Gather sorted codes (SYCL device kernel) ------------------
    {
        const auto* si = si_buf_->data();
        const auto* c  = codes_buf_->data();
        auto*       sc = sc_buf_->data();

        queue_.ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_n, wg_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= n) return;
                sc[i] = c[si[i]];
            });
        }).wait_and_throw();
    }

    // ---- 4. Resize node array in shared USM ----------------------------
    // All node fields are written by subsequent kernels — no value-init needed.
    if (!nodes_) nodes_ = std::make_shared<shared_vector<BVHNode>>(*queue_.ptr);
    nodes_->resize(num_nodes);  // all fields written by subsequent kernels

    // Handle the degenerate single-point case.
    if (n == 1) {
        root_index_ = 0;
        const auto& pt = pts_data[0];
        (*nodes_)[0].aabb_min[0] = pt.x(); (*nodes_)[0].aabb_min[1] = pt.y(); (*nodes_)[0].aabb_min[2] = pt.z();
        (*nodes_)[0].aabb_max[0] = pt.x(); (*nodes_)[0].aabb_max[1] = pt.y(); (*nodes_)[0].aabb_max[2] = pt.z();
        (*nodes_)[0].left_idx  = 0;
        (*nodes_)[0].right_idx = -1;
        return;
    }

    // ---- 5. Initialise leaf nodes (SYCL device kernel) -----------------
    // Layout: internal nodes [0, n-2], leaf nodes [n-1, 2n-2].
    {
        auto* nodes = nodes_->data();
        const auto* si = si_buf_->data();

        queue_.ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_n, wg_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= n) return;
                const int32_t node_idx = ni - 1 + static_cast<int32_t>(i);
                const int32_t pt_idx   = si[i];
                const auto& pt = pts_data[pt_idx];

                BVHNode& nd = nodes[node_idx];
                nd.aabb_min[0] = pt.x(); nd.aabb_min[1] = pt.y(); nd.aabb_min[2] = pt.z();
                nd.aabb_max[0] = pt.x(); nd.aabb_max[1] = pt.y(); nd.aabb_max[2] = pt.z();
                nd.left_idx  = pt_idx;
                nd.right_idx = -1;  // leaf sentinel
            });
        }).wait_and_throw();
    }

    // ---- 6. Build internal nodes (SYCL device kernel, Karras 2012) -----
    // Each internal node i is independent: determine_range and find_split
    // only read sc_buf_; parent writes are race-free (each child has exactly
    // one parent).
    {
        auto* nodes = nodes_->data();
        auto* par   = parent_buf_->data();
        const auto* sc = sc_buf_->data();
        const size_t global_ni = queue_.get_global_size(static_cast<size_t>(ni - 1));

        queue_.ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_ni, wg_size), [=](sycl::nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= static_cast<size_t>(ni - 1)) return;
                const int i = static_cast<int>(idx);

                const auto [first, last] = determine_range(sc, ni, i);
                const int gamma = find_split(sc, ni, first, last);

                const int left_child  = (gamma     == first) ? (ni - 1 + gamma)     : gamma;
                const int right_child = (gamma + 1 == last)  ? (ni - 1 + gamma + 1) : (gamma + 1);

                BVHNode& nd = nodes[i];
                nd.left_idx  = left_child;
                nd.right_idx = right_child;

                par[left_child]  = i;  // each child has exactly one parent → no race
                par[right_child] = i;
            });
        }).wait_and_throw();
    }

    // ---- 7. Root node --------------------------------------------------
    // In the Karras 2012 binary radix tree, internal node 0 always covers
    // the full leaf range [0, n-1] and is therefore always the root.
    root_index_ = 0;

    // ---- 8. Propagate AABBs bottom-up (SYCL kernel, Karras atomic) -----
    // One thread per leaf. Each thread walks up the tree; the first child
    // to reach an internal node stops, the second computes the AABB and
    // continues. sycl::atomic_ref with acq_rel ensures the first child's
    // AABB write is visible before the second child reads it.
    {
        if (!aabb_flags_) aabb_flags_ = std::make_shared<shared_vector<int32_t>>(*queue_.ptr);
        aabb_flags_->assign(ni - 1, int32_t{0});  // must be 0 before kernel launch

        auto* nodes = nodes_->data();
        const auto* par  = parent_buf_->data();
        auto* flags = aabb_flags_->data();

        queue_.ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_n, wg_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= n) return;

                int32_t cur = par[ni - 1 + static_cast<int32_t>(i)];
                while (cur >= 0) {
                    sycl::atomic_ref<int32_t,
                                     sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        counter(flags[cur]);

                    if (counter.fetch_add(1) == 0) break;  // first to arrive: stop

                    // Second to arrive: merge children's AABBs.
                    const BVHNode& l = nodes[nodes[cur].left_idx];
                    const BVHNode& r = nodes[nodes[cur].right_idx];
                    for (int j = 0; j < 3; ++j) {
                        nodes[cur].aabb_min[j] = sycl::fmin(l.aabb_min[j], r.aabb_min[j]);
                        nodes[cur].aabb_max[j] = sycl::fmax(l.aabb_max[j], r.aabb_max[j]);
                    }
                    cur = par[cur];
                }
            });
        }).wait_and_throw();
    }
    // nodes_ is already in shared USM — no upload step needed.
}

// ----------------------------------------------------------------------------
//  kNN dispatch
// ----------------------------------------------------------------------------

inline sycl_utils::events LBVH::knn_search_async(const PointCloudShared& queries, size_t k, KNNResult& result,
                                                  const std::vector<sycl::event>& depends,
                                                  const TransformMatrix& transT) const {
    // BVH depth is O(log n): log2(n_max) ~= 17 for n=69k.
    // Each of the two stacks (near/far) gets MAX_DEPTH/2 entries.
    // Use 128 → each stack holds 64 entries, safe for practical point clouds.
    constexpr size_t MAX_DEPTH = 128;

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
    const size_t node_count  = nodes_ ? nodes_->size() : 0;
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
        const auto* nodes_ptr = nodes_->data();
        const auto trans_vec  = eigen_utils::to_sycl_vec(transT);

        handler.parallel_for(sycl::nd_range<1>(global_sz, wg_size), [=](sycl::nd_item<1> item) {
            const size_t qi = item.get_global_id(0);
            if (qi >= query_size || target_size == 0) return;

            // ---- Transform query point --------------------------------
            Eigen::Vector4f qp;
            transform::kernel::transform_point(q_pts[qi], qp, trans_vec);
            const float qx = qp.x(), qy = qp.y(), qz = qp.z();

            // ---- k-best sorted array (ascending by dist_sq) ----------
            NodeEntry best[MAX_K];
            for (size_t i = 0; i < MAX_K; ++i) best[i] = {-1, std::numeric_limits<float>::max()};
            size_t found = 0;

            auto worst_dist_sq = [&]() -> float {
                return (found < k) ? std::numeric_limits<float>::infinity() : best[k - 1].dist_sq;
            };
            // Insert candidate into sorted array (ascending dist_sq, like KDTree).
            auto push_candidate = [&](float d2, int32_t pt_idx) {
                if constexpr (MAX_K == 1) {
                    if (d2 < best[0].dist_sq) { best[0] = {pt_idx, d2}; found = 1; }
                    return;
                }
                if (found == k && d2 >= best[k - 1].dist_sq) return;
                size_t pos = (found < k) ? found : k - 1;
                while (pos > 0 && d2 < best[pos - 1].dist_sq) {
                    best[pos] = best[pos - 1];
                    --pos;
                }
                best[pos] = {pt_idx, d2};
                if (found < k) ++found;
            };

            // ---- AABB squared distance --------------------------------
            auto aabb_dist2 = [](const BVHNode* n, float px, float py, float pz) -> float {
                const float dx = sycl::fmax(0.0f, sycl::fmax(n->aabb_min[0] - px, px - n->aabb_max[0]));
                const float dy = sycl::fmax(0.0f, sycl::fmax(n->aabb_min[1] - py, py - n->aabb_max[1]));
                const float dz = sycl::fmax(0.0f, sycl::fmax(n->aabb_min[2] - pz, pz - n->aabb_max[2]));
                return sycl::fma(dx, dx, sycl::fma(dy, dy, dz * dz));
            };

            // ---- Depth-first traversal: near/far dual stack ----------
            // (mirrors KDTree approach: O(1) pop, no linear scan)
            constexpr size_t MAX_DEPTH_HALF = MAX_DEPTH / 2;
            NodeEntry nearStack[MAX_DEPTH_HALF];
            NodeEntry farStack[MAX_DEPTH_HALF];
            size_t nearPtr = 0, farPtr = 0;

            if (node_count == 0 || root_idx < 0) return;

            nearStack[nearPtr++] = {root_idx, 0.0f};

            while (nearPtr > 0 || farPtr > 0) {
                // Pop near first (depth-first priority), then far.
                const NodeEntry cur = (nearPtr > 0) ? nearStack[--nearPtr] : farStack[--farPtr];

                // Prune: AABB distance exceeds current k-th best.
                if (cur.dist_sq > worst_dist_sq()) continue;

                const BVHNode& node = nodes_ptr[cur.node_idx];

                if (node.right_idx < 0) {
                    // Leaf: aabb_min holds the point position.
                    const float dx = qx - node.aabb_min[0];
                    const float dy = qy - node.aabb_min[1];
                    const float dz = qz - node.aabb_min[2];
                    push_candidate(sycl::fma(dx, dx, sycl::fma(dy, dy, dz * dz)), node.left_idx);
                } else {
                    // Internal: compute child AABB distances and push near first.
                    const float wd      = worst_dist_sq();
                    const float d_left  = aabb_dist2(&nodes_ptr[node.left_idx],  qx, qy, qz);
                    const float d_right = aabb_dist2(&nodes_ptr[node.right_idx], qx, qy, qz);

                    const bool left_is_near   = (d_left <= d_right);
                    const int32_t near_child  = left_is_near ? node.left_idx  : node.right_idx;
                    const int32_t far_child   = left_is_near ? node.right_idx : node.left_idx;
                    const float   near_d      = left_is_near ? d_left  : d_right;
                    const float   far_d       = left_is_near ? d_right : d_left;

                    if (far_d  <= wd && farPtr  < MAX_DEPTH_HALF) farStack[farPtr++]   = {far_child,  far_d};
                    if (near_d <= wd && nearPtr < MAX_DEPTH_HALF) nearStack[nearPtr++] = {near_child, 0.0f};
                }
            }

            // ---- Write output (already sorted ascending) --------------
            for (size_t i = 0; i < k; ++i) {
                indices_ptr[qi * k + i]   = best[i].node_idx;
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
