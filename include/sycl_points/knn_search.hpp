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

// Node structure for KD-Tree (ignoring w component)
struct FlatKDNode {
  PointType point;  // Point coordinates (w is assumed to be 1.0)
  int idx;          // Index of the point in the original dataset
  int axis;         // Split axis (0=x, 1=y, 2=z)
  int left;         // Index of left child node (-1 if none)
  int right;        // Index of right child node (-1 if none)
};

class KNNSearch {
public:
  std::shared_ptr<std::vector<FlatKDNode>> tree_;

  KNNSearch() {
    tree_ = std::make_shared<std::vector<FlatKDNode>>();
  }

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
      node.point = points[pointIdx];
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
        const auto& target = targets.points[j];
        const float dx = query.x() - target.x();
        const float dy = query.y() - target.y();
        const float dz = query.z() - target.z();
        const float dist = dx * dx + dy * dy + dz * dz;
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

  KNNSearchSYCL(sycl::queue& queue) : queue_(std::make_shared<sycl::queue>(queue)) {
    tree_ = std::make_shared<shared_vector<FlatKDNode>>(0, *this->queue_);
  }

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
      node.point = points[pointIdx];
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

  static KNNSearchSYCL buildKDTree(sycl::queue& queue, const PointCloudShared& cloud) {
    return KNNSearchSYCL::buildKDTree(queue, *cloud.points);
  }

  static KNNResult searchBruteForce_sycl(sycl::queue& queue, const PointCloudShared& queries, const PointCloudShared& targets, const size_t k) {
    constexpr size_t MAX_K = 50;

    const size_t n = targets.points->size();  // Number of dataset points
    const size_t q = queries.points->size();  // Number of query points

    // Initialize result structure
    KNNResult result;
    result.indices.resize(q);
    result.distances.resize(q);

    for (size_t i = 0; i < q; ++i) {
      result.indices[i].resize(k, -1);
      result.distances[i].resize(k, std::numeric_limits<float>::max());
    }

    try {
      // Allocate device memory using USM
      auto targets_ptr = (*targets.points).data();
      auto queries_ptr = (*queries.points).data();

      auto distances = shared_vector<float>(q * k, std::numeric_limits<float>::max(), queue);
      auto neighbors = shared_vector<int>(q * k, -1, queue);
      float* distance_ptr = distances.data();
      int* neighbor_ptr = neighbors.data();

      // KNN search kernel
      queue
        .submit([&](sycl::handler& h) {
          h.parallel_for(sycl::range<1>(q), [=](sycl::id<1> idx) {
            const size_t queryIdx = idx[0];
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
              const float dx = query.x() - targets_ptr[j].x();
              const float dy = query.y() - targets_ptr[j].y();
              const float dz = query.z() - targets_ptr[j].z();
              const float dist = dx * dx + dy * dy + dz * dz;

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
              neighbor_ptr[queryIdx * k + i] = kIndices[i];
            }
          });
        })
        .wait_and_throw();

      // Copy to output structure
      for (size_t i = 0; i < q; i++) {
        for (int j = 0; j < k; j++) {
          result.indices[i][j] = neighbors[i * k + j];
          result.distances[i][j] = distances[i * k + j];
        }
      }
    } catch (const sycl::exception& e) {
      std::cerr << "SYCL exception caught: " << e.what() << std::endl;
    }

    return result;
  }

  KNNResult searchKDTree_sycl(
    const PointContainerShared& queries,  // Query points
    const size_t k) {                     // Number of neighbors to find

    constexpr size_t MAX_K = 50;

    const size_t q = queries.size();  // Number of query points
    const size_t treeSize = this->tree_->size();

    // Initialize result structure
    KNNResult result;
    result.indices.resize(q);
    result.distances.resize(q);

    for (size_t i = 0; i < q; ++i) {
      result.indices[i].resize(k, -1);
      result.distances[i].resize(k, std::numeric_limits<float>::max());
    }

    try {
      // Allocate device memory
      shared_vector<float> distances(q * k, std::numeric_limits<float>::max(), *this->queue_);
      shared_vector<int> neighbors(q * k, -1, *this->queue_);

      auto query_ptr = queries.data();
      auto distance_ptr = distances.data();
      auto neighbor_ptr = neighbors.data();
      auto tree_ptr = (*this->tree_).data();

      // SYCL KD-Tree KNN search kernel
      this->queue_
        ->submit([&](sycl::handler& h) {
          h.parallel_for(sycl::range<1>(q), [=](sycl::id<1> idx) {
            const size_t queryIdx = idx[0];
            // Query point
            const auto& query = query_ptr[queryIdx];

            // Arrays to store K nearest points
            float bestDists[MAX_K];
            int bestIdxs[MAX_K];

            // Initialize
            for (int i = 0; i < k; i++) {
              bestDists[i] = std::numeric_limits<float>::max();
              bestIdxs[i] = -1;
            }

            // Non-recursive KD-Tree traversal using a stack
            const int MAX_DEPTH = 32;  // Maximum stack depth
            int nodeStack[MAX_DEPTH];
            int stackPtr = 0;

            // Start from root node
            nodeStack[stackPtr++] = 0;

            while (stackPtr > 0) {
              const int nodeIdx = nodeStack[--stackPtr];

              // Skip invalid nodes
              if (nodeIdx == -1) continue;

              const auto& node = tree_ptr[nodeIdx];

              // Calculate distance to current node (3D space)
              const float dx = query.x() - node.point.x();
              const float dy = query.y() - node.point.y();
              const float dz = query.z() - node.point.z();
              const float dist = dx * dx + dy * dy + dz * dz;

              // Check if this node should be included in K nearest
              if (dist < bestDists[k - 1]) {
                // Find insertion position
                int insertPos = k - 1;
                while (insertPos > 0 && dist < bestDists[insertPos - 1]) {
                  bestDists[insertPos] = bestDists[insertPos - 1];
                  bestIdxs[insertPos] = bestIdxs[insertPos - 1];
                  insertPos--;
                }

                // Insert new point
                bestDists[insertPos] = dist;
                bestIdxs[insertPos] = node.idx;
              }

              // Distance along split axis
              const float axisDistance(node.axis == 0 ? dx : (node.axis == 1 ? dy : dz));

              // Determine nearer and further subtrees
              const int nearerNode = (axisDistance <= 0) ? node.left : node.right;
              const int furtherNode = (axisDistance <= 0) ? node.right : node.left;

              // If distance to splitting plane is less than current kth distance,
              // both subtrees must be searched
              if (axisDistance * axisDistance <= bestDists[k - 1]) {
                // Add further subtree to stack
                if (furtherNode != -1 && stackPtr < MAX_DEPTH) {
                  nodeStack[stackPtr++] = furtherNode;
                }
              }

              // Add nearer subtree to stack
              if (nearerNode != -1 && stackPtr < MAX_DEPTH) {
                nodeStack[stackPtr++] = nearerNode;
              }
            }

            // Write results to global memory
            for (int i = 0; i < k; i++) {
              distance_ptr[queryIdx * k + i] = bestDists[i];
              neighbor_ptr[queryIdx * k + i] = bestIdxs[i];
            }
          });
        })
        .wait_and_throw();

      // Copy to output structure
      for (size_t i = 0; i < q; i++) {
        for (int j = 0; j < k; j++) {
          result.indices[i][j] = neighbors[i * k + j];
          result.distances[i][j] = distances[i * k + j];
        }
      }
    } catch (const sycl::exception& e) {
      std::cerr << "SYCL exception caught: " << e.what() << std::endl;
    }

    return result;
  }

  KNNResult searchKDTree_sycl(
    const PointCloudShared& queries,  // Query points
    const size_t k) {                 // Number of neighbors to find
    return searchKDTree_sycl(*queries.points, k);
  }
};

}  // namespace sycl_points
