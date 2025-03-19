#pragma once

#include "point_cloud.hpp"
#include "knn_search.hpp"

namespace sycl_points {

template <typename T = float>
inline CovarianceContainerCPU computeCovariances(
  KNNSearch& kdtree,             // KDTree
  const PointContainerCPU& points,  // Point Cloud
  const size_t k_correspondences,   // Number of neighbor points
  const size_t num_threads = 1) {

  const auto neightbors = kdtree.searchKDTree_sycl(points, k_correspondences);

  const size_t N = points.size();
  CovarianceContainerCPU covs(N);

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

template <typename T = float>
inline CovarianceContainerCPU computeCovariances(
  sycl::queue& queue,               // SYCL execution queue
  const PointContainerCPU& points,  // Point Cloud
  const size_t k_correspondences,   // Number of neighbor points
  const size_t num_threads = 1) {
  auto kdtree = KNNSearch::buildKDTree(queue, points);
  return computeCovariances(kdtree, points, k_correspondences, num_threads);
}

template <typename T = float>
inline void computeCovariances(
  KNNSearch& kdtree,            // KDTree
  PointCloudCPU& points,     // Point Cloud
  const size_t k_correspondences,  // Number of neighbor points
  const size_t num_threads = 1) {
  points.covs = computeCovariances(kdtree, points.points, k_correspondences, num_threads);
}

template <typename T = float>
inline void computeCovariances(
  sycl::queue& queue,              // SYCL execution queue
  PointCloudCPU& points,     // Point Cloud
  const size_t k_correspondences,  // Number of neighbor points
  const size_t num_threads = 1) {
  points.covs = computeCovariances(queue, points.points, k_correspondences, num_threads);
}

}  // namespace sycl_points
