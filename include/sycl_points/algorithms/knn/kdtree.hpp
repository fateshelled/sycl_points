#pragma once

#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sycl_points/algorithms/common/filter_by_flags.hpp>
#include <sycl_points/algorithms/knn/knn.hpp>
#include <sycl_points/algorithms/knn/result.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace knn {

namespace {

// Node structure for KD-Tree (ignoring w component)
struct FlatKDNode {
    PointType pt;
    int32_t idx;         // Index of the point in the original dataset
    int32_t left = -1;   // Index of left child node (-1 if none), or next leaf node for leaf nodes
    int32_t right = -1;  // Index of right child node (-1 if none), unused for leaf nodes
    uint8_t axis;        // Split axis (0=x, 1=y, 2=z) - unused for leaf nodes
    uint8_t is_leaf;     // 1 if leaf node, 0 if internal node
    uint8_t valid = 1;   // 0 is removed node, 1 is valid
    uint8_t pad;         // Padding for alignment (1 bytes)

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};  // Total: 32 bytes

// Data structure for non-recursive KD-tree construction
struct BuildTask {
    uint32_t nodeIdx;   // Node index in the tree
    uint32_t startIdx;  // global indices start index
    uint32_t endIdx;    // global indices end index
    BuildTask(uint32_t node, uint32_t start, uint32_t end) : nodeIdx(node), startIdx(start), endIdx(end) {}
};

// Stack to track nodes that need to be explored
struct NodeEntry {
    int32_t nodeIdx;  // node index
    float dist_sq;    // squared distance to splitting plane
};

// Helper function to find best split axis based on range
template <typename ALLOCATOR>
inline uint8_t find_axis_range(const std::vector<PointType, ALLOCATOR>& points, const std::vector<uint32_t>& indices,
                               uint32_t start, uint32_t end) {
    const int64_t size = end - start + 1;
    if (size <= 1) return 0;

    // Find approximate min/max for each axis
    sycl::float3 min_vals = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                             std::numeric_limits<float>::max()};
    sycl::float3 max_vals = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                             std::numeric_limits<float>::lowest()};

    const size_t step = static_cast<size_t>(std::max(size / 100, (int64_t)1));
    for (size_t i = start; i <= end; i += step) {
        const auto idx = indices[i];
#pragma unroll 3
        for (size_t axis = 0; axis < 3; ++axis) {
            min_vals[axis] = std::min(min_vals[axis], points[idx](axis));
            max_vals[axis] = std::max(max_vals[axis], points[idx](axis));
        }
    }

    // Find the axis with largest range
    const sycl::float3 ranges = max_vals - min_vals;

    // Return the axis with the largest range
    if (ranges[0] >= ranges[1] && ranges[0] >= ranges[2]) return 0;
    if (ranges[1] >= ranges[0] && ranges[1] >= ranges[2]) return 1;
    return 2;
}

// Helper function to find best split axis based on variance
template <typename ALLOCATOR>
inline uint8_t find_axis_variance(const std::vector<PointType, ALLOCATOR>& points, const std::vector<uint32_t>& indices,
                                  uint32_t start, uint32_t end) {
    const int64_t size = end - start + 1;
    if (size <= 1) return 0;

    // Find approximate min/max for each axis
    PointType sum = PointType::Zero();
    PointType sum_sq = PointType::Zero();

    const size_t step = static_cast<size_t>(std::max(size / 100, (int64_t)1));
    for (size_t i = start; i <= end; i += step) {
        const PointType& pt = points[indices[i]];
        sum += pt;
        sum_sq += pt.cwiseProduct(pt);
    }

    const PointType variance = (sum_sq - (sum / sum.w()).cwiseProduct(sum));

    // Return the axis with the largest variance
    if (variance[0] >= variance[1] && variance[0] >= variance[2]) return 0;
    if (variance[1] >= variance[0] && variance[1] >= variance[2]) return 1;
    return 2;
}

template <size_t MAX_K>
void insert_to_bestK(NodeEntry* bestK, float dist_sq, int32_t nodeIdx, size_t k, size_t found_num) {
    if constexpr (MAX_K == 1) {
        bestK[0].nodeIdx = dist_sq < bestK[0].dist_sq ? nodeIdx : bestK[0].nodeIdx;
        bestK[0].dist_sq = dist_sq < bestK[0].dist_sq ? dist_sq : bestK[0].dist_sq;
        return;
    }

    if (dist_sq >= bestK[k - 1].dist_sq) return;

    size_t insert_pos = std::min(found_num - 1, k - 1);
    while (insert_pos > 0 && dist_sq < bestK[insert_pos - 1].dist_sq) {
        bestK[insert_pos].nodeIdx = bestK[insert_pos - 1].nodeIdx;
        bestK[insert_pos].dist_sq = bestK[insert_pos - 1].dist_sq;
        --insert_pos;
    }
    bestK[insert_pos].nodeIdx = nodeIdx;
    bestK[insert_pos].dist_sq = dist_sq;
}

}  // namespace

/// @brief KDTree with SYCL implementation
class KDTree : public KNNBase {
private:
    using FlatKDNodeVector = shared_vector<FlatKDNode>;
    std::shared_ptr<FlatKDNodeVector> tree_;
    mutable std::shared_timed_mutex mutex_;

public:
    using Ptr = std::shared_ptr<KDTree>;
    sycl_utils::DeviceQueue queue;

    /// @brief Constructor
    /// @param q SYCL queue
    KDTree(const sycl_utils::DeviceQueue& q) : queue(q) {
        tree_ = std::make_shared<FlatKDNodeVector>(0, *this->queue.ptr);
    }

    /// @brief Destructor
    ~KDTree() {}

    /// @brief Build KDTree
    /// @param q SYCL queue
    /// @param points Point Container
    /// @param leaf_threshold The maximum number of points in a leaf node.
    /// @return KDTree shared_ptr
    static KDTree::Ptr build(const sycl_utils::DeviceQueue& q, const PointContainerShared& points,
                             size_t leaf_threshold = 16) {
        const size_t n = points.size();

        if (n == 0) {
            KDTree::Ptr flatTree = std::make_shared<KDTree>(q);
            flatTree->tree_->resize(0);
            return flatTree;
        }

        std::vector<FlatKDNode> tree;

        // Estimate tree size with some margin
        const size_t estimatedSize = n * 2;
        tree.resize(estimatedSize);

        std::vector<uint32_t> globalIndices(n);
        std::iota(globalIndices.begin(), globalIndices.end(), 0);

        std::vector<BuildTask> taskStack;
        taskStack.reserve(n);
        // Add the first task to the stack
        taskStack.emplace_back((uint32_t)0, (uint32_t)0, (uint32_t)(n - 1));

        uint32_t nextNodeIdx = 1;  // Node 0 is root, subsequent nodes start from 1

        // Process until task stack is empty
        while (!taskStack.empty()) {
            // Pop a task from the stack
            BuildTask task = std::move(taskStack.back());
            taskStack.pop_back();

            const auto nodeIdx = task.nodeIdx;
            const auto startIdx = task.startIdx;
            const auto endIdx = task.endIdx;
            const auto indices_size = endIdx - startIdx + 1;

            if (startIdx > endIdx || indices_size == 0) continue;

            auto& node = tree[nodeIdx];

            // Check if this should be a leaf node
            if (indices_size <= leaf_threshold) {
                // Create leaf nodes as a linked list
                int32_t currentLeafIdx = nodeIdx;

                for (size_t i = 0; i < indices_size; ++i) {
                    const auto pointIdx = globalIndices[startIdx + i];
                    auto& leafNode = tree[currentLeafIdx];

                    leafNode.pt = points[pointIdx];
                    leafNode.idx = pointIdx;
                    leafNode.is_leaf = 1;  // Mark as leaf node
                    leafNode.axis = 0;     // Unused for leaf nodes
                    leafNode.right = -1;   // Unused for leaf nodes
                    leafNode.valid = 1;

                    // Set left to next leaf node index, or -1 for the last one
                    if (i < indices_size - 1) {
                        leafNode.left = nextNodeIdx;
                        currentLeafIdx = nextNodeIdx++;
                    } else {
                        leafNode.left = -1;  // Last leaf node
                    }
                }
                continue;
            }

            // Create internal node
            node.is_leaf = 0;
            node.valid = 1;

            // Split axis
            const auto axis = find_axis_range(points, globalIndices, startIdx, endIdx);
            // const auto axis = find_axis_variance(points, subIndices, startIdx, endIdx);

            // Partial sort to find median
            const uint32_t medianIdx = startIdx + indices_size / 2;
#if __cplusplus >= 202002L
            std::nth_element(std::execution::unseq, globalIndices.begin() + startIdx, globalIndices.begin() + medianIdx,
                             globalIndices.begin() + endIdx + 1,
                             [&](uint32_t a, uint32_t b) { return points[a](axis) < points[b](axis); });
#else
            std::nth_element(globalIndices.begin() + startIdx, globalIndices.begin() + medianIdx,
                             globalIndices.begin() + endIdx + 1,
                             [&](uint32_t a, uint32_t b) { return points[a](axis) < points[b](axis); });
#endif

            // Get the median point
            const auto pointIdx = globalIndices[medianIdx];

            // Initialize internal node
            node.pt = points[pointIdx];
            node.idx = pointIdx;
            node.axis = axis;

            // Add left subtree to processing queue if not empty
            if (startIdx < medianIdx) {
                const auto leftNodeIdx = nextNodeIdx++;
                node.left = leftNodeIdx;
                taskStack.emplace_back(leftNodeIdx, startIdx, medianIdx - 1);
            }

            // Add right subtree to processing queue if not empty
            if (medianIdx < endIdx) {
                const auto rightNodeIdx = nextNodeIdx++;
                node.right = rightNodeIdx;
                taskStack.emplace_back(rightNodeIdx, medianIdx + 1, endIdx);
            }
        }

        // Trim the tree to actual used size
        tree.resize(nextNodeIdx);
        KDTree::Ptr flatTree = std::make_shared<KDTree>(q);
        flatTree->tree_ = std::make_shared<FlatKDNodeVector>(nextNodeIdx, *q.ptr);
        std::copy(tree.begin(), tree.end(), flatTree->tree_->begin());
        return flatTree;
    }

    /// @brief Build KDTree
    /// @param queue SYCL queue
    /// @param cloud Point Cloud
    /// @param leaf_threshold The maximum number of points in a leaf node.
    /// @return KDTree shared_ptr
    static KDTree::Ptr build(const sycl_utils::DeviceQueue& queue, const PointCloudShared& cloud,
                             size_t leaf_threshold = 16) {
        return KDTree::build(queue, *cloud.points, leaf_threshold);
    }

    void remove_nodes_by_flags(const shared_vector<uint8_t>& flags, const shared_vector<int32_t>& indices) {
        const size_t N = this->tree_->size();
        if (N != flags.size() || N != indices.size()) {
            throw std::runtime_error("flags size must be equal to tree size.");
        }
        // mem_advise to device
        {
            this->queue.set_accessed_by_device(this->tree_->data(), N);
            this->queue.set_accessed_by_device(flags.data(), N);
            this->queue.set_accessed_by_device(indices.data(), N);
        }

        const size_t work_group_size = this->queue.get_work_group_size();
        const size_t global_size = this->queue.get_global_size(N);

        auto remove_task = [&](sycl::handler& h) {
            // Get pointers
            const auto tree_ptr = this->tree_->data();
            const auto flags_ptr = flags.data();
            const auto indices_ptr = indices.data();

            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t idx = item.get_global_id(0);

                if (idx >= N) return;

                const auto point_idx = tree_ptr[idx].idx;
                if (point_idx < 0 || N < point_idx) return;

                // Direct assignment: REMOVE_FLAG(0) -> invalid(0), INCLUDE_FLAG(1) -> valid(1)
                tree_ptr[idx].valid = flags_ptr[point_idx];
                tree_ptr[idx].idx = indices_ptr[point_idx];
            });
        };

        static bool is_first_time = true;
        std::chrono::steady_clock::duration timeout = std::chrono::milliseconds(100);
        if (is_first_time) {
            is_first_time = false;
            timeout = std::chrono::steady_clock::duration(0);
        }

        this->queue.execute_with_mutex(this->mutex_, remove_task, {}, true, timeout).wait();

        // mem_advise clear
        {
            this->queue.clear_accessed_by_device(this->tree_->data(), N);
            this->queue.clear_accessed_by_device(flags.data(), N);
            this->queue.clear_accessed_by_device(indices.data(), N);
        }
    }

    /// @brief async kNN search
    /// @tparam MAX_K maximum of k
    /// @tparam MAX_DEPTH maximum of search depth
    /// @param queries query points
    /// @param query_size query num
    /// @param k number of search nearrest neightbor
    /// @param result Search result
    /// @param depends depends sycl events
    /// @return knn search event
    template <size_t MAX_K = 20, size_t MAX_DEPTH = 32>
    sycl_utils::events knn_search_async(const PointType* queries, const size_t query_size, const size_t k,
                                        KNNResult& result,
                                        const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const {
        if (query_size == 0) {
            if (result.indices == nullptr || result.distances == nullptr) {
                result.allocate(this->queue, 0, 0);
            } else {
                result.resize(0, 0);
            }
            return sycl_utils::events();
        }
        constexpr size_t MAX_DEPTH_HALF = MAX_DEPTH / 2;
        if (MAX_K < k) {
            throw std::runtime_error("template arg `MAX_K` must be larger than `k`.");
        }

        const size_t treeSize = this->tree_->size();

        // Initialize result structure
        const size_t total_size = query_size * k;
        if (result.indices == nullptr || result.distances == nullptr) {
            result.allocate(this->queue, query_size, k);
        } else {
            result.resize(query_size, k);
        }

        const size_t work_group_size = this->queue.get_work_group_size();
        const size_t global_size = this->queue.get_global_size(query_size);

        auto search_task = [&](sycl::handler& h) {
            // Get pointers
            const auto query_ptr = queries;
            const auto distance_ptr = result.distances->data();
            const auto index_ptr = result.indices->data();
            const auto tree_ptr = (*this->tree_).data();

            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t queryIdx = item.get_global_id(0);

                if (queryIdx >= query_size) return;  // Early return for extra threads

                // Query point
                const PointType query = query_ptr[queryIdx];

                // Arrays to store K nearest points
                NodeEntry bestK[MAX_K];
                // Initialize
                std::fill(bestK, bestK + MAX_K, NodeEntry{-1, std::numeric_limits<float>::max()});

                // Stack
                NodeEntry nearStack[MAX_DEPTH_HALF];
                NodeEntry farStack[MAX_DEPTH_HALF];
                size_t nearStackPtr = 0;
                size_t farStackPtr = 0;

                // Start from root node
                nearStack[nearStackPtr++] = {0, 0.0f};

                size_t found_num = 0;

                // Explore until stack is empty
                while (nearStackPtr > 0 || farStackPtr > 0) {
                    // Pop a node from stack
                    const NodeEntry current = nearStackPtr > 0 ? nearStack[--nearStackPtr] : farStack[--farStackPtr];
                    const auto nodeIdx = current.nodeIdx;

                    // Skip condition: nodes farther than current kth distance
                    if (current.dist_sq > bestK[k - 1].dist_sq) continue;

                    // Skip invalid nodes
                    if (nodeIdx == -1 || nodeIdx >= treeSize) continue;

                    const auto node = tree_ptr[nodeIdx];
                    const bool is_valid = (node.valid == filter::INCLUDE_FLAG);

                    // Calculate distance to current node
                    const PointType diff = eigen_utils::subtract<4, 1>(query, node.pt);
                    const float dist_sq =
                        is_valid ? eigen_utils::dot<4>(diff, diff) : std::numeric_limits<float>::max();
                    found_num = is_valid ? found_num + 1 : found_num;
                    insert_to_bestK<MAX_K>(bestK, dist_sq, node.idx, k, found_num);

                    // Calculate distance along split axis
                    const float axisDistance = diff[node.axis];

                    // Determine nearer and further subtrees
                    const bool is_leaf = (node.is_leaf != 0);
                    const auto nearerNode = (is_leaf || axisDistance <= 0) ? node.left : node.right;
                    const auto furtherNode = (is_leaf || axisDistance <= 0) ? node.right : node.left;

                    // Squared distance to splitting plane
                    const float splitDistSq = axisDistance * axisDistance;

                    // Check if further subtree needs to be explored
                    const bool searchFurther = (splitDistSq < bestK[k - 1].dist_sq);

                    // Push further subtree to stack (conditional)
                    if (searchFurther && furtherNode != -1 && farStackPtr < MAX_DEPTH_HALF) {
                        farStack[farStackPtr++] = {furtherNode, splitDistSq};
                    }

                    // Push nearer subtree to stack (always explore)
                    if (nearerNode != -1 && nearStackPtr < MAX_DEPTH_HALF) {
                        nearStack[nearStackPtr++] = {nearerNode, 0.0f};  // Prioritize near side with distance 0
                    }
                }

                // Write final results to global memory
                for (size_t i = 0; i < k; ++i) {
                    distance_ptr[queryIdx * k + i] = bestK[i].dist_sq;
                    index_ptr[queryIdx * k + i] = bestK[i].nodeIdx;
                }
            });
        };

        static bool is_first_time = true;
        std::chrono::steady_clock::duration timeout = std::chrono::milliseconds(100);
        if (is_first_time) {
            is_first_time = false;
            timeout = std::chrono::steady_clock::duration(0);
        }

        sycl_utils::events events;
        events += this->queue.execute_with_mutex(this->mutex_, search_task, depends, false, timeout);
        return events;
    }

    /// @brief async kNN search
    /// @param queries query points
    /// @param k number of search nearrest neightbor
    /// @param result Search result
    /// @param depends depends sycl events
    /// @return knn search event
    sycl_utils::events knn_search_async(const PointCloudShared& queries, const size_t k, KNNResult& result,
                                        const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const {
        constexpr size_t MAX_DEPTH = 32;
        if (k == 1) {
            return knn_search_async<1, MAX_DEPTH>(queries.points_ptr(), queries.size(), k, result, depends);
        } else if (k <= 10) {
            return knn_search_async<10, MAX_DEPTH>(queries.points_ptr(), queries.size(), k, result, depends);
        } else if (k <= 20) {
            return knn_search_async<20, MAX_DEPTH>(queries.points_ptr(), queries.size(), k, result, depends);
        } else if (k <= 30) {
            return knn_search_async<30, MAX_DEPTH>(queries.points_ptr(), queries.size(), k, result, depends);
        } else if (k <= 40) {
            return knn_search_async<40, MAX_DEPTH>(queries.points_ptr(), queries.size(), k, result, depends);
        } else if (k <= 50) {
            return knn_search_async<50, MAX_DEPTH>(queries.points_ptr(), queries.size(), k, result, depends);
        } else if (k <= 100) {
            return knn_search_async<100, MAX_DEPTH>(queries.points_ptr(), queries.size(), k, result, depends);
        } else {
            throw std::runtime_error("`k` is too large. not support.");
        }
    }

};

}  // namespace knn

}  // namespace algorithms

}  // namespace sycl_points
