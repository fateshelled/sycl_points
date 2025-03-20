#pragma once

#include "point_cloud.hpp"
#include "knn_search.hpp"

namespace sycl_points {

// inline CovarianceContainerCPU compute_covariances(
//   KNNSearch& kdtree,             // KDTree
//   const PointContainerCPU& points,  // Point Cloud
//   const size_t k_correspondences,   // Number of neighbor points
//   const size_t num_threads = 1) {

//   // todo
//   const auto neightbors = kdtree.searchBruteForce(points, k_correspondences);

//   const size_t N = points.size();
//   CovarianceContainerCPU covs(N);

// #pragma omp parallel for num_threads(num_threads)
//   for (size_t i = 0; i < N; ++i) {
//     PointType sum_points = PointType::Zero();
//     Covariance sum_cross = Covariance::Zero();

//     const auto& indices = neightbors.indices[i];
//     for (size_t j = 0; j < k_correspondences; ++j) {
//       const auto& pt = points[indices[j]];
//       sum_points += pt;
//       sum_cross += pt * pt.transpose();
//     }
//     const PointType mean = sum_points / k_correspondences;
//     covs[i] = (sum_cross - mean * sum_points.transpose()) / k_correspondences;
//   }

//   return covs;
// }

// inline CovarianceContainerCPU compute_covariances(
//   const PointContainerCPU& points,  // Point Cloud
//   const size_t k_correspondences,   // Number of neighbor points
//   const size_t num_threads = 1) {
//   auto kdtree = KNNSearch::buildKDTree(points);
//   return compute_covariances(kdtree, points, k_correspondences, num_threads);
// }

// inline void compute_covariances(
//   KNNSearch& kdtree,            // KDTree
//   PointCloudCPU& points,     // Point Cloud
//   const size_t k_correspondences,  // Number of neighbor points
//   const size_t num_threads = 1) {
//   points.covs = compute_covariances(kdtree, points.points, k_correspondences, num_threads);
// }

// inline void compute_covariances(
//   PointCloudCPU& points,     // Point Cloud
//   const size_t k_correspondences,  // Number of neighbor points
//   const size_t num_threads = 1) {
//   points.covs = compute_covariances(queue, points.points, k_correspondences, num_threads);
// }


inline CovarianceContainerShared compute_covariances_sycl(
  const KNNSearchSYCL& kdtree,             // KDTree
  const PointContainerShared& points,  // Point Cloud
  const size_t k_correspondences,   // Number of neighbor points
  const size_t num_threads = 1) {

  const auto neightbors = kdtree.searchKDTree_sycl(points, k_correspondences);

  const size_t N = points.size();
  shared_allocator<Covariance> alloc(*kdtree.queue_);
  CovarianceContainerShared covs(N, alloc);

#pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < N; ++i) {
    PointType sum_points = PointType::Zero();
    Covariance sum_cross = Covariance::Zero();

    const auto& indices = neightbors.indices[i];
    for (size_t j = 0; j < k_correspondences; ++j) {
      const auto& pt = points[indices[j]];
      sum_points += pt;
      sum_cross += pt * pt.transpose();
    }
    const PointType mean = sum_points / k_correspondences;
    covs[i] = (sum_cross - mean * sum_points.transpose()) / k_correspondences;
  }

  return covs;
}

inline CovarianceContainerShared compute_covariances_sycl(
  sycl::queue& queue,
  const PointContainerShared& points,  // Point Cloud
  const size_t k_correspondences,   // Number of neighbor points
  const size_t num_threads = 1) {
  auto kdtree = KNNSearchSYCL::buildKDTree(queue, points);
  return compute_covariances_sycl(kdtree, points, k_correspondences, num_threads);
}

inline void compute_covariances_sycl(
  const KNNSearchSYCL& kdtree,            // KDTree
  const PointCloudShared& points,     // Point Cloud
  const size_t k_correspondences,  // Number of neighbor points
  const size_t num_threads = 1) {
  *points.covs = compute_covariances_sycl(kdtree, *points.points, k_correspondences, num_threads);
}

inline void compute_covariances_sycl(
  sycl::queue& queue,
  const PointCloudShared& points,     // Point Cloud
  const size_t k_correspondences,  // Number of neighbor points
  const size_t num_threads = 1) {
  *points.covs = compute_covariances_sycl(queue, *points.points, k_correspondences, num_threads);
}
}  // namespace sycl_points
