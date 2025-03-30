#pragma once

#include "point_cloud.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

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
  std::shared_ptr<shared_vector<int>> indices;
  std::shared_ptr<shared_vector<float>> distances;
  size_t query_size;
  size_t k;
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

class KNNSearch {
public:
  std::shared_ptr<std::vector<FlatKDNode>> tree_;

  KNNSearch() { tree_ = std::make_shared<std::vector<FlatKDNode>>(); }

  ~KNNSearch() {}

  static KNNSearch buildKDTree(const PointContainerCPU& points) {
    const size_t n = points.size();

    // Estimate tree size with some margin
    const int estimatedSize = n * 2;
    KNNSearch flatTree;

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

      // Split axis based on current depth (x->y->z->x->...)
      const int axis = depth % 3;

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
      std::vector<int> rightIndices;
      rightIndices.reserve(subIndices.size() - medianPos - 1);
      for (size_t i = medianPos + 1; i < sortedValues.size(); ++i) {
        rightIndices.push_back(sortedValues[i].second);
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

  static KNNSearch buildKDTree(const PointCloudCPU& points) { return buildKDTree(points.points); }

  static KNNResult searchBruteForce(const PointCloudCPU& queries, const PointCloudCPU& targets, const size_t k, const size_t num_threads = 8) {
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
};

class KNNSearchSYCL {
public:
  std::shared_ptr<shared_vector<FlatKDNode>> tree_;
  std::shared_ptr<sycl::queue> queue_ = nullptr;

  KNNSearchSYCL(sycl::queue& queue) : queue_(std::make_shared<sycl::queue>(queue)) { tree_ = std::make_shared<shared_vector<FlatKDNode>>(0, *this->queue_); }

  KNNSearchSYCL(sycl::queue& queue, const KNNSearch& kdtree) : queue_(std::make_shared<sycl::queue>(queue)) {
    tree_ = std::make_shared<shared_vector<FlatKDNode>>(kdtree.tree_->size(), *this->queue_);
    for (size_t i = 0; i < kdtree.tree_->size(); ++i) {
      (*tree_)[i] = (*kdtree.tree_)[i];
    }
  }

  ~KNNSearchSYCL() {}

  static KNNSearchSYCL buildKDTree(sycl::queue& queue, const PointContainerShared& points) {
    const size_t n = points.size();

    // Estimate tree size with some margin
    const int estimatedSize = n * 2;
    KNNSearchSYCL flatTree(queue);

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

      // Split axis based on current depth (x->y->z->x->...)
      const int axis = depth % 3;

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
      std::vector<int> rightIndices;
      rightIndices.reserve(subIndices.size() - medianPos - 1);
      for (size_t i = medianPos + 1; i < sortedValues.size(); ++i) {
        rightIndices.push_back(sortedValues[i].second);
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

  static KNNSearchSYCL buildKDTree(sycl::queue& queue, const PointCloudShared& cloud) { return KNNSearchSYCL::buildKDTree(queue, *cloud.points); }

  static KNNResultSYCL searchBruteForce_sycl(sycl::queue& queue, const PointCloudShared& queries, const PointCloudShared& targets, const size_t k) {
    constexpr size_t MAX_K = 48;

    const size_t n = targets.points->size();  // Number of dataset points
    const size_t q = queries.points->size();  // Number of query points

    // Initialize result structure
    KNNResultSYCL result;
    result.indices = std::make_shared<shared_vector<int>>(q * k, -1, shared_allocator<int>(queue));
    result.distances = std::make_shared<shared_vector<float>>(q * k, std::numeric_limits<float>::max(), shared_allocator<float>(queue));
    result.k = k;
    result.query_size = q;

    // Optimize work group size
    const size_t work_group_size = std::min(sycl_utils::default_work_group_size, (size_t)queue.get_device().get_info<sycl::info::device::max_work_group_size>());
    const size_t global_size = ((q + work_group_size - 1) / work_group_size) * work_group_size;

    // memory ptr
    auto targets_ptr = (*targets.points).data();
    auto queries_ptr = (*queries.points).data();

    float* distance_ptr = result.distances->data();
    int* index_ptr = result.indices->data();

    // KNN search kernel BruteForce
    auto event = queue
      .submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)), [=](sycl::nd_item<1> item) {
          const size_t queryIdx = item.get_global_id(0);
          if (queryIdx >= q) return;
          const auto query = queries_ptr[queryIdx];

          // Arrays to store K nearest points
          float kDistances[MAX_K];
          int kIndices[MAX_K];

          // Initialize
          for (int i = 0; i < k; i++) {
            kDistances[i] = std::numeric_limits<float>::max();
            kIndices[i] = -1;
          }

          // Calculate distances to all dataset points
          for (size_t j = 0; j < n; j++) {
            // Calculate 3D distance
            const sycl::float4 diff = {query.x() - targets_ptr[j].x(), query.y() - targets_ptr[j].y(), query.z() - targets_ptr[j].z(), 0.0f};
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

  KNNResultSYCL searchKDTree_sycl(
    const PointContainerShared& queries,  // Query points
    const size_t k) const {               // Number of neighbors to find

    constexpr size_t MAX_K = 48;      // Maximum number of neighbors to search
    constexpr size_t MAX_DEPTH = 32;  // Maximum stack depth

    const size_t q = queries.size();  // Number of query points
    const size_t treeSize = this->tree_->size();

    // Initialize result structure
    KNNResultSYCL result;
    result.indices = std::make_shared<shared_vector<int>>(q * k, -1, shared_allocator<int>(*this->queue_));
    result.distances = std::make_shared<shared_vector<float>>(q * k, std::numeric_limits<float>::max(), shared_allocator<float>(*this->queue_));
    result.k = k;
    result.query_size = q;

    // Get pointers
    const auto query_ptr = queries.data();
    const auto distance_ptr = result.distances->data();
    const auto index_ptr = result.indices->data();
    const auto tree_ptr = (*this->tree_).data();

    // Optimize work group size
    const size_t work_group_size = std::min(sycl_utils::default_work_group_size, (size_t)this->queue_->get_device().get_info<sycl::info::device::max_work_group_size>());
    const size_t global_size = ((q + work_group_size - 1) / work_group_size) * work_group_size;

    auto event = this->queue_->submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)), [=](sycl::nd_item<1> item) {
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

        NodeEntry stack[MAX_DEPTH];
        int stackPtr = 0;

        // Start from root node
        stack[stackPtr++] = {0, 0.0f};

        // Explore until stack is empty
        while (stackPtr > 0) {
          // Pop a node from stack
          const NodeEntry current = stack[--stackPtr];
          const int nodeIdx = current.nodeIdx;

          // Skip condition: nodes farther than current kth distance
          if (current.dist_sq > bestDists[k - 1]) continue;

          // Skip invalid nodes
          if (nodeIdx == -1 || nodeIdx >= treeSize) continue;

          const auto node = tree_ptr[nodeIdx];

          // Calculate distance to current node
          const sycl::float4 diff = {query.x() - node.x, query.y() - node.y, query.z() - node.z, 0.0f};
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
          if (searchFurther && furtherNode != -1 && stackPtr < MAX_DEPTH) {
            stack[stackPtr++] = {furtherNode, splitDistSq};
          }

          // Push nearer subtree to stack (always explore)
          if (nearerNode != -1 && stackPtr < MAX_DEPTH) {
            stack[stackPtr++] = {nearerNode, 0.0f};  // Prioritize near side with distance 0
          }
        }

        // Write final results to global memory
        for (int i = 0; i < k; ++i) {
          distance_ptr[queryIdx * k + i] = bestDists[i];
          index_ptr[queryIdx * k + i] = bestIdxs[i];
        }
      });
    });
    event.wait();

    return result;
  }

  KNNResultSYCL searchKDTree_sycl(
    const PointCloudShared& queries,  // Query points
    const size_t k) const {           // Number of neighbors to find
    return searchKDTree_sycl(*queries.points, k);
  }
};

}  // namespace sycl_points
