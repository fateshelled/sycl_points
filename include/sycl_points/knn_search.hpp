#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "point_cloud.hpp"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace sycl_points {
// Structure to store K nearest neighbors and their distances
struct KNNResult {
    std::vector<std::vector<int>> indices;      // Indices of K nearest points for each query point
    std::vector<std::vector<float>> distances;  // Squared distances to K nearest points for each query point
};

struct KNNResultSYCL {
    std::shared_ptr<shared_vector<int>> indices = nullptr;
    std::shared_ptr<shared_vector<float>> distances = nullptr;
    size_t query_size;
    size_t k;
    KNNResultSYCL() : query_size(0), k(0) {}
    void allocate(sycl::queue& queue, size_t query_size = 0, size_t k = 0) {
        this->query_size = query_size;
        this->k = k;
        this->indices = std::make_shared<shared_vector<int>>(query_size * k, -1, shared_allocator<int>(queue, {}));
        this->distances = std::make_shared<shared_vector<float>>(query_size * k, std::numeric_limits<float>::max(),
                                                                 shared_allocator<float>(queue, {}));
    }
};

// Node structure for KD-Tree (ignoring w component)
struct FlatKDNode {
    float x, y, z;
    int idx;         // Index of the point in the original dataset
    int left;        // Index of left child node (-1 if none)
    int right;       // Index of right child node (-1 if none)
    uint8_t axis;    // Split axis (0=x, 1=y, 2=z)
    uint8_t pad[3];  // Padding for alignment (3 bytes)
};  // Total: 28 bytes, aligned to 4-byte boundary

namespace {

// Helper function to find best split axis based on range
template <typename T, typename ALLOCATOR>
inline int find_best_axis(const std::vector<T, ALLOCATOR>& points, const std::vector<int>& indices) {
    if (indices.size() <= 1) return 0;

    // Find min/max for each axis
    std::array<float, 3> min_vals = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::max()};
    std::array<float, 3> max_vals = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                                     std::numeric_limits<float>::lowest()};

    // Find the range of each axis
    for (const auto& idx : indices) {
        for (int axis = 0; axis < 3; ++axis) {
            min_vals[axis] = std::min(min_vals[axis], points[idx](axis));
            max_vals[axis] = std::max(max_vals[axis], points[idx](axis));
        }
    }

    // Find the axis with largest range
    std::array<float, 3> ranges = {max_vals[0] - min_vals[0], max_vals[1] - min_vals[1], max_vals[2] - min_vals[2]};

    // Return the axis with the largest range
    if (ranges[0] >= ranges[1] && ranges[0] >= ranges[2]) return 0;
    if (ranges[1] >= ranges[0] && ranges[1] >= ranges[2]) return 1;
    return 2;
}

}  // namespace

class KDTree {
public:
    std::shared_ptr<std::vector<FlatKDNode>> tree_;

    KDTree() { tree_ = std::make_shared<std::vector<FlatKDNode>>(); }

    ~KDTree() {}

    static KDTree build(const PointContainerCPU& points) {
        const size_t n = points.size();

        // Estimate tree size with some margin
        const int estimatedSize = n * 2;
        KDTree flatTree;

        flatTree.tree_->resize(estimatedSize);

        // Main index array
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);

        // Reusable temporary array for sorting
        std::vector<std::pair<float, int>> sortedValues(n);

        // Data structure for non-recursive KD-tree construction
        struct BuildTask {
            std::vector<int> indices;  // Indices corresponding to this node
            int nodeIdx;               // Node index in the tree
            int depth;                 // Depth in the tree
        };

        std::vector<BuildTask> taskStack;
        int nextNodeIdx = 1;  // Node 0 is root, subsequent nodes start from 1

        // Add the first task to the stack
        taskStack.push_back({indices, 0, 0});

        // Process until task stack is empty
        while (!taskStack.empty()) {
            // Pop a task from the stack
            BuildTask task = std::move(taskStack.back());
            taskStack.pop_back();

            std::vector<int>& subIndices = task.indices;
            const int nodeIdx = task.nodeIdx;
            const int depth = task.depth;

            if (subIndices.empty()) continue;

            // Split axis
            const int axis = find_best_axis(points, subIndices);

            // Create pairs of values and indices for sorting along the axis
            sortedValues.resize(subIndices.size());
            for (size_t i = 0; i < subIndices.size(); ++i) {
                const int idx = subIndices[i];
                sortedValues[i] = {points[idx](axis), idx};
            }

            // Partial sort to find median
            const size_t medianPos = subIndices.size() / 2;
            std::nth_element(sortedValues.begin(), sortedValues.begin() + medianPos, sortedValues.end());

            // Get the median point
            const int pointIdx = sortedValues[medianPos].second;

            // Initialize flat node
            auto& node = (*flatTree.tree_)[nodeIdx];
            node.x = points[pointIdx].x();
            node.y = points[pointIdx].y();
            node.z = points[pointIdx].z();
            node.idx = pointIdx;
            node.axis = axis;
            node.left = -1;
            node.right = -1;

            // Extract indices for left subtree
            std::vector<int> leftIndices(medianPos);
            for (size_t i = 0; i < medianPos; ++i) {
                leftIndices[i] = sortedValues[i].second;
            }

            // Extract indices for right subtree
            std::vector<int> rightIndices(subIndices.size() - medianPos - 1);
            size_t counter = 0;
            for (size_t i = medianPos + 1; i < sortedValues.size(); ++i) {
                rightIndices[counter++] = sortedValues[i].second;
            }

            // Add left subtree to processing queue if not empty
            if (!leftIndices.empty()) {
                const int leftNodeIdx = nextNodeIdx++;
                node.left = leftNodeIdx;
                taskStack.push_back({std::move(leftIndices), leftNodeIdx, depth + 1});
            }

            // Add right subtree to processing queue if not empty
            if (!rightIndices.empty()) {
                const int rightNodeIdx = nextNodeIdx++;
                node.right = rightNodeIdx;
                taskStack.push_back({std::move(rightIndices), rightNodeIdx, depth + 1});
            }
        }

        // Trim the tree to actual used size
        flatTree.tree_->resize(nextNodeIdx);
        return flatTree;
    }

    static KDTree build(const PointCloudCPU& points) { return build(points.points); }
};

class KDTreeSYCL {
public:
    using FlatKDNodeVector = shared_vector<FlatKDNode>;

    std::shared_ptr<FlatKDNodeVector> tree_;
    std::shared_ptr<sycl::queue> queue_ = nullptr;

    KDTreeSYCL(const std::shared_ptr<sycl::queue>& queue_ptr) : queue_(queue_ptr) {
        tree_ = std::make_shared<FlatKDNodeVector>(0, *this->queue_);
    }

    KDTreeSYCL(const std::shared_ptr<sycl::queue>& queue_ptr, const KDTree& kdtree) : queue_(queue_ptr) {
        tree_ = std::make_shared<FlatKDNodeVector>(kdtree.tree_->size(), *this->queue_);
        for (size_t i = 0; i < kdtree.tree_->size(); ++i) {
            (*tree_)[i] = (*kdtree.tree_)[i];
        }
    }

    ~KDTreeSYCL() {}

    static KDTreeSYCL build(const std::shared_ptr<sycl::queue>& queue_ptr, const PointContainerShared& points) {
        const size_t n = points.size();

        // Estimate tree size with some margin
        const int estimatedSize = n * 2;
        KDTreeSYCL flatTree(queue_ptr);

        flatTree.tree_->resize(estimatedSize);

        // Main index array
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);

        // Reusable temporary array for sorting
        std::vector<std::pair<float, int>> sortedValues(n);

        // Data structure for non-recursive KD-tree construction
        struct BuildTask {
            std::vector<int> indices;  // Indices corresponding to this node
            int nodeIdx;               // Node index in the tree
            int depth;                 // Depth in the tree
        };

        std::vector<BuildTask> taskStack;
        taskStack.reserve(n);
        int nextNodeIdx = 1;  // Node 0 is root, subsequent nodes start from 1

        // Add the first task to the stack
        taskStack.push_back({indices, 0, 0});

        // Process until task stack is empty
        while (!taskStack.empty()) {
            // Pop a task from the stack
            BuildTask task = std::move(taskStack.back());
            taskStack.pop_back();

            std::vector<int>& subIndices = task.indices;
            const int nodeIdx = task.nodeIdx;
            const int depth = task.depth;

            if (subIndices.empty()) continue;

            // Split axis
            const int axis = find_best_axis(points, subIndices);

            // Create pairs of values and indices for sorting along the axis
            sortedValues.resize(subIndices.size());
            for (size_t i = 0; i < subIndices.size(); ++i) {
                const int idx = subIndices[i];
                sortedValues[i] = {points[idx](axis), idx};
            }

            // Partial sort to find median
            const size_t medianPos = subIndices.size() / 2;
            std::nth_element(sortedValues.begin(), sortedValues.begin() + medianPos, sortedValues.end());

            // Get the median point
            const int pointIdx = sortedValues[medianPos].second;

            // Initialize flat node
            auto& node = (*flatTree.tree_)[nodeIdx];
            node.x = points[pointIdx].x();
            node.y = points[pointIdx].y();
            node.z = points[pointIdx].z();
            node.idx = pointIdx;
            node.axis = axis;
            node.left = -1;
            node.right = -1;

            // Extract indices for left subtree
            std::vector<int> leftIndices(medianPos);
            for (size_t i = 0; i < medianPos; ++i) {
                leftIndices[i] = sortedValues[i].second;
            }

            // Extract indices for right subtree
            std::vector<int> rightIndices(subIndices.size() - medianPos - 1);
            size_t counter = 0;
            for (size_t i = medianPos + 1; i < sortedValues.size(); ++i) {
                rightIndices[counter++] = sortedValues[i].second;
            }

            // Add left subtree to processing queue if not empty
            if (!leftIndices.empty()) {
                const int leftNodeIdx = nextNodeIdx++;
                node.left = leftNodeIdx;
                taskStack.push_back({std::move(leftIndices), leftNodeIdx, depth + 1});
            }

            // Add right subtree to processing queue if not empty
            if (!rightIndices.empty()) {
                const int rightNodeIdx = nextNodeIdx++;
                node.right = rightNodeIdx;
                taskStack.push_back({std::move(rightIndices), rightNodeIdx, depth + 1});
            }
        }

        // Trim the tree to actual used size
        flatTree.tree_->resize(nextNodeIdx);
        return flatTree;
    }

    static KDTreeSYCL build(const std::shared_ptr<sycl::queue>& queue_ptr, const PointCloudShared& cloud) {
        return KDTreeSYCL::build(queue_ptr, *cloud.points);
    }

    template <size_t MAX_K = 48, size_t MAX_DEPTH = 32>
    sycl_utils::events knn_search_async(const PointContainerShared& queries,  // Query points
                                        const size_t k,                       // Number of neighbors to find
                                        KNNResultSYCL& result,
                                        const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const {
        // constexpr size_t MAX_K = 48;      // Maximum number of neighbors to search
        // constexpr size_t MAX_DEPTH = 32;  // Maximum stack depth

        const size_t q = queries.size();  // Number of query points
        const size_t treeSize = this->tree_->size();

        // Initialize result structure
        if (result.indices == nullptr || result.distances == nullptr) {
            result.allocate(*this->queue_, q, k);
        } else {
            result.indices->resize(q * k, -1);
            result.distances->resize(q * k, std::numeric_limits<float>::max());
            result.k = k;
            result.query_size = q;
        }

        // Optimize work group size
        const size_t work_group_size = sycl_utils::get_work_group_size(*this->queue_);
        const size_t global_size = ((q + work_group_size - 1) / work_group_size) * work_group_size;

        auto event = this->queue_->submit([&](sycl::handler& h) {
            // Get pointers
            const auto query_ptr = queries.data();
            const auto distance_ptr = result.distances->data();
            const auto index_ptr = result.indices->data();
            const auto tree_ptr = (*this->tree_).data();

            h.depends_on(depends);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)),
                [=](sycl::nd_item<1> item) {
                    const size_t queryIdx = item.get_global_id(0);

                    if (queryIdx >= q) return;  // Early return for extra threads

                    // Query point
                    const auto query = query_ptr[queryIdx];

                    // Arrays to store K nearest points
                    float bestDists[MAX_K];
                    int bestIdxs[MAX_K];

                    // Initialize
                    for (int i = 0; i < k; ++i) {
                        bestDists[i] = std::numeric_limits<float>::max();
                        bestIdxs[i] = -1;
                    }

                    // Stack to track nodes that need to be explored
                    // {node index, squared distance to splitting plane}
                    struct NodeEntry {
                        int nodeIdx;
                        float dist_sq;
                    };

                    NodeEntry nearStack[MAX_DEPTH / 2];
                    int nearStackPtr = 0;

                    NodeEntry farStack[MAX_DEPTH / 2];
                    int farStackPtr = 0;

                    // Start from root node
                    nearStack[nearStackPtr++] = {0, 0.0f};

                    // Explore until stack is empty
                    while (nearStackPtr > 0 || farStackPtr > 0) {
                        // Pop a node from stack
                        NodeEntry current;
                        if (nearStackPtr > 0) {
                            current = nearStack[--nearStackPtr];
                        } else {
                            current = farStack[--farStackPtr];
                        }
                        const int nodeIdx = current.nodeIdx;

                        // Skip condition: nodes farther than current kth distance
                        if (current.dist_sq > bestDists[k - 1]) continue;

                        // Skip invalid nodes
                        if (nodeIdx == -1 || nodeIdx >= treeSize) continue;

                        const auto node = tree_ptr[nodeIdx];

                        // Calculate distance to current node
                        const sycl::float3 diff = {query.x() - node.x, query.y() - node.y, query.z() - node.z};
                        const float dist_sq = sycl::dot(diff, diff);

                        // Check if this point should be included in K nearest
                        if (dist_sq < bestDists[k - 1]) {
                            // Find insertion position in K-nearest list (insertion sort)
                            int insertPos = k - 1;
                            while (insertPos > 0 && dist_sq < bestDists[insertPos - 1]) {
                                bestDists[insertPos] = bestDists[insertPos - 1];
                                bestIdxs[insertPos] = bestIdxs[insertPos - 1];
                                insertPos--;
                            }

                            // Insert result
                            bestDists[insertPos] = dist_sq;
                            bestIdxs[insertPos] = node.idx;
                        }

                        // Calculate distance along split axis
                        const int axis = node.axis;
                        const float axisDistance = (axis == 0) ? (diff[0]) : (axis == 1) ? (diff[1]) : (diff[2]);

                        // Determine nearer and further subtrees
                        const int nearerNode = (axisDistance <= 0) ? node.left : node.right;
                        const int furtherNode = (axisDistance <= 0) ? node.right : node.left;

                        // Squared distance to splitting plane
                        const float splitDistSq = axisDistance * axisDistance;

                        // Check if further subtree needs to be explored
                        const bool searchFurther = (splitDistSq < bestDists[k - 1]);

                        // Optimization for efficient memory access in 64-byte units
                        // Push both near and far sides to stack, but with condition for far side

                        // Push further subtree to stack (conditional)
                        if (searchFurther && furtherNode != -1 && farStackPtr < MAX_DEPTH / 2) {
                            farStack[farStackPtr++] = {furtherNode, splitDistSq};
                        }

                        // Push nearer subtree to stack (always explore)
                        if (nearerNode != -1 && nearStackPtr < MAX_DEPTH / 2) {
                            nearStack[nearStackPtr++] = {nearerNode, 0.0f};  // Prioritize near side with distance 0
                        }
                    }

                    // Write final results to global memory
                    for (size_t i = 0; i < k; ++i) {
                        distance_ptr[queryIdx * k + i] = bestDists[i];
                        index_ptr[queryIdx * k + i] = bestIdxs[i];
                    }
                });
        });
        sycl_utils::events events;
        events.push_back(event);
        return events;
    }

    template <size_t MAX_K = 48, size_t MAX_DEPTH = 32>
    sycl_utils::events knn_search_async(const PointCloudShared& queries,  // Query points
                                        const size_t k,                   // Number of neighbors to find
                                        KNNResultSYCL& result,
                                        const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const {
        return knn_search_async<MAX_K, MAX_DEPTH>(*queries.points, k, result, depends);
    }

    template <size_t MAX_K = 48, size_t MAX_DEPTH = 32>
    KNNResultSYCL knn_search(const PointContainerShared& queries,  // Query points
                             const size_t k, const std::vector<sycl::event>& depends = std::vector<sycl::event>())
        const {  // Number of neighbors to find

        KNNResultSYCL result;
        knn_search_async<MAX_K, MAX_DEPTH>(queries, k, result, depends).wait();
        return result;
    }

    template <size_t MAX_K = 48, size_t MAX_DEPTH = 32>
    KNNResultSYCL knn_search(const PointCloudShared& queries,  // Query points
                             const size_t k, const std::vector<sycl::event>& depends = std::vector<sycl::event>())
        const {  // Number of neighbors to find

        KNNResultSYCL result;
        knn_search_async<MAX_K, MAX_DEPTH>(*queries.points, k, result, depends).wait();
        return result;
    }
};

inline KNNResult knn_search_bruteforce(const PointCloudCPU& queries, const PointCloudCPU& targets, const size_t k,
                                       const size_t num_threads = 8) {
    const size_t n = targets.points.size();  // Number of dataset points
    const size_t q = queries.points.size();  // Number of query points

    // Initialize result structure
    KNNResult result;
    result.indices.resize(q);
    result.distances.resize(q);

    for (size_t i = 0; i < q; ++i) {
        result.indices[i].resize(k, -1);
        result.distances[i].resize(k, std::numeric_limits<float>::max());
    }

// For each query point, find K nearest neighbors
#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < q; ++i) {
        const auto& query = queries.points[i];

        // Vector to store distances and indices of all points
        std::vector<std::pair<float, int>> distances(n);

        // Calculate distances to all dataset points
        for (size_t j = 0; j < n; ++j) {
            const auto dt = query - targets.points[j];
            const float dist = dt.dot(dt);
            distances[j] = {dist, j};
        }

        // Sort to find K smallest distances
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

        // Store the results
        for (int j = 0; j < k; ++j) {
            result.indices[i][j] = distances[j].second;
            result.distances[i][j] = distances[j].first;
        }
    }

    return result;
}

inline KNNResultSYCL knn_search_bruteforce_sycl(sycl::queue& queue, const PointCloudShared& queries,
                                                const PointCloudShared& targets, const size_t k) {
    constexpr size_t MAX_K = 48;

    const size_t n = targets.points->size();  // Number of dataset points
    const size_t q = queries.points->size();  // Number of query points

    // Initialize result structure
    KNNResultSYCL result;
    result.allocate(queue, q, k);

    // Optimize work group size
    const size_t work_group_size = sycl_utils::get_work_group_size(queue);
    const size_t global_size = ((q + work_group_size - 1) / work_group_size) * work_group_size;

    // memory ptr
    auto targets_ptr = (*targets.points).data();
    auto queries_ptr = (*queries.points).data();

    float* distance_ptr = result.distances->data();
    int* index_ptr = result.indices->data();

    // KNN search kernel BruteForce
    auto event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)),
                       [=](sycl::nd_item<1> item) {
                           const size_t queryIdx = item.get_global_id(0);
                           if (queryIdx >= q) return;
                           const auto query = queries_ptr[queryIdx];

                           // Arrays to store K nearest points
                           float kDistances[MAX_K];
                           int kIndices[MAX_K];

                           // Initialize
                           for (int i = 0; i < k; ++i) {
                               kDistances[i] = std::numeric_limits<float>::max();
                               kIndices[i] = -1;
                           }

                           // Calculate distances to all dataset points
                           for (size_t j = 0; j < n; ++j) {
                               // Calculate 3D distance
                               const auto target = targets_ptr[j];
                               const sycl::float4 diff = {query.x() - target.x(), query.y() - target.y(),
                                                          query.z() - target.z(), 0.0f};
                               const float dist = sycl::dot(diff, diff);

                               // Check if this point should be included in K nearest
                               if (dist < kDistances[k - 1]) {
                                   // Find insertion position
                                   int insertPos = k - 1;
                                   while (insertPos > 0 && dist < kDistances[insertPos - 1]) {
                                       kDistances[insertPos] = kDistances[insertPos - 1];
                                       kIndices[insertPos] = kIndices[insertPos - 1];
                                       insertPos--;
                                   }

                                   // Insert new point
                                   kDistances[insertPos] = dist;
                                   kIndices[insertPos] = j;
                               }
                           }

                           // Write results to global memory
                           for (int i = 0; i < k; i++) {
                               distance_ptr[queryIdx * k + i] = kDistances[i];
                               index_ptr[queryIdx * k + i] = kIndices[i];
                           }
                       });
    });
    event.wait();

    return result;
}
}  // namespace sycl_points
